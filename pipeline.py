"""
pipeline.py — Shared data pipeline (used by both train.py and live.py)
======================================================================
Responsibilities:
  - Feature engineering (technical + institutional)
  - HTF alignment
  - Data loading from tick Parquet or MT5
  - Live data refresh (append fresh MT5 bars to historical cache)
  - Model/scaler loading from disk (inference mode)

NOT responsible for:
  - Training models  → train.py
  - Live trading loop → live.py
  - Optimisation      → train.py
"""

import os, sys, logging, warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*Converting sparse.*", category=UserWarning)

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

# ── Third-party module imports ─────────────────────────────────────────
from tick_pipeline import (
    load_ohlcv_parquet,
    parquet_exists,
    build_all_timeframes,
    DATA_DIR,
)
from institutional_features import add_institutional_features

import phase2_adaptive_engine as p2
from phase2_adaptive_engine import (
    connect_mt5, get_account_balance,
    fetch_bars, fetch_all_timeframes,
    get_feature_cols, apply_scaler,
    load_params,
    INSTRUMENTS, PARAM_SEEDS, HARD_LIMITS,
    MODEL_DIR, LOG_DIR, PARAMS_DIR, SEQ_LEN, MIN_BARS,
)
from db import init_db

load_dotenv()
init_db()

log = logging.getLogger("pipeline")

# ── Config ─────────────────────────────────────────────────────────────
_BASE_DIR = Path(os.getenv("BASE_DIR", r"F:\trading_ml"))

def _tick_path(sym: str) -> Path:
    return Path(os.getenv(f"TICK_FILE_{sym}",
                str(_BASE_DIR / "tick_data" / f"{sym}_ticks.csv")))

TICK_DATA_SYMBOLS = {
    sym: _tick_path(sym)
    for sym in os.getenv("TICK_SYMBOLS", "US30").split(",")
}

FORCE_RETRAIN    = os.getenv("FORCE_RETRAIN", "false").lower() == "true"
LIVE_REFRESH_BARS = 200
MAX_INST_BARS     = 750_000

SESSION_BARS = {
    1:  390, 3:  130, 5:  78,
    10: 39,  15: 26,  30: 13, 60: 35,
}


# ─────────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────

def engineer_full_features(df: pd.DataFrame,
                            tf_minutes: int = 5,
                            symbol: str = "",
                            is_tick_derived: bool = False) -> pd.DataFrame:
    """Standard technical + institutional features."""
    if df.empty or len(df) < MIN_BARS:
        return df
    d = p2.engineer_features(df)
    if d.empty:
        return d
    if not is_tick_derived:
        _estimate_microstructure(d)
    sb = SESSION_BARS.get(tf_minutes, 78)
    try:
        d = add_institutional_features(d, session_bars=sb, tf_minutes=tf_minutes, verbose=False)
    except Exception as e:
        log.warning(f"Institutional features partial failure on {symbol} {tf_minutes}m: {e}")
    d.dropna(inplace=True)
    return d


def _estimate_microstructure(df: pd.DataFrame):
    """Estimate tick-only columns from OHLCV (used for MT5-sourced data)."""
    c   = df["Close"]
    v   = df["Volume"]
    atr = df.get("atr14", (df["High"] - df["Low"]).rolling(14).mean())
    direction = np.sign(c.diff()).fillna(0)
    df["ask_volume"]    = v * ((direction + 1) / 2).clip(0, 1)
    df["bid_volume"]    = v * ((1 - direction) / 2).clip(0, 1)
    df["vol_imbalance"] = direction * 0.5
    df["spread_mean"]   = atr * 0.05
    df["spread_max"]    = atr * 0.15
    df["spread_std"]    = atr * 0.02
    vol_sma = v.rolling(20).mean()
    df["tick_count"]    = (v / (vol_sma + 1e-10) * 50).clip(1, 500)
    df["tick_velocity"] = df["tick_count"] / 300
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    dates   = df.index.date
    cum_tpv = (tp * v).groupby(dates).cumsum()
    cum_vol = v.groupby(dates).cumsum()
    df["vwap"]         = cum_tpv / (cum_vol + 1e-10)
    df["price_impact"] = (df["High"] - df["Low"]) / (v + 1e-10)
    df["true_return"]  = df.get("logr1", c.pct_change())


# ─────────────────────────────────────────────────────────────────────
# 2. HTF ALIGNMENT
# ─────────────────────────────────────────────────────────────────────

def add_htf_alignment_full(entry_df: pd.DataFrame,
                            htf_df: pd.DataFrame,
                            htf_weight: float,
                            htf_tf: int = 60) -> pd.DataFrame:
    """Composite HTF filter: EMA + VWAP + CVD + POC + regime."""
    if htf_df.empty or htf_weight == 0:
        return entry_df
    htf_feat = engineer_full_features(htf_df.copy(), tf_minutes=htf_tf)
    if htf_feat.empty:
        return entry_df

    ema_dir    = (htf_feat["Close"] > htf_feat.get("ema55", htf_feat["Close"])).astype(int) * 2 - 1
    vwap_dir   = ((htf_feat["Close"] > htf_feat["vwap_session"]).astype(int) * 2 - 1
                  if "vwap_session" in htf_feat.columns else pd.Series(0, index=htf_feat.index))
    delta_dir  = (np.sign(htf_feat["cvd_slope5"])
                  if "cvd_slope5" in htf_feat.columns else pd.Series(0, index=htf_feat.index))
    poc_dir    = (np.sign(htf_feat["vp_poc_dist"])
                  if "vp_poc_dist" in htf_feat.columns else pd.Series(0, index=htf_feat.index))
    regime_dir = ((htf_feat["regime_bull"] - htf_feat["regime_bear"])
                  if "regime_bull" in htf_feat.columns else pd.Series(0, index=htf_feat.index))

    htf_composite = (ema_dir*0.25 + vwap_dir*0.25 + delta_dir*0.20
                     + poc_dir*0.15 + regime_dir*0.15)
    htf_composite.name = "htf_bullish"
    htf_strength = (htf_composite.abs() * htf_weight).rename("htf_strength")

    htf_composite = htf_composite.resample("1min").last().ffill()
    htf_strength  = htf_strength.resample("1min").last().ffill()

    entry_df["htf_bullish"]  = htf_composite.reindex(entry_df.index, method="ffill").fillna(0)
    entry_df["htf_strength"] = htf_strength.reindex(entry_df.index, method="ffill").fillna(0)
    return entry_df

# Patch phase2 so get_signal() uses the upgraded HTF function
p2.add_htf_alignment = add_htf_alignment_full


# ─────────────────────────────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_symbol_data(symbol: str, save_featured: bool = False) -> dict:
    """
    Load best available data for a symbol across all TFs.
    save_featured=True  → also write {symbol}_{tf}m_featured.parquet
                          (needed for accurate backtesting in report.py)
                          Set True during training, False during live load.
    """
    log.info(f"Loading data: {symbol}")

    if symbol in TICK_DATA_SYMBOLS:
        tick_file  = TICK_DATA_SYMBOLS[symbol]
        tfs_needed = sorted(set(
            PARAM_SEEDS["entry_tf_options"] +
            [t for t in PARAM_SEEDS["htf_options"] if t > 0]
        ))

        if not all(parquet_exists(symbol, tf) for tf in tfs_needed):
            if tick_file.exists():
                log.info(f"  Building Parquet from tick CSV: {tick_file} (one-time, ~30-60 min)")
                build_all_timeframes(tick_file, symbol, tfs_needed)
            else:
                log.warning(f"  Tick CSV not found: {tick_file} — falling back to MT5")
                return _load_from_mt5(symbol)

        tf_data = {}
        for tf in tfs_needed:
            raw = load_ohlcv_parquet(symbol, tf)
            if raw.empty:
                continue

            enhanced = None
            if len(raw) > MAX_INST_BARS:
                log.info(f"  {symbol} {tf}m: {len(raw):,} bars — skipping institutional "
                         f"loops (>{MAX_INST_BARS:,} bars), using tick features directly")
            else:
                try:
                    enhanced = add_institutional_features(
                        raw, session_bars=SESSION_BARS.get(tf, 78), tf_minutes=tf, verbose=False)
                    if len(enhanced) < MIN_BARS:
                        log.warning(f"  {symbol} {tf}m: institutional features produced "
                                    f"only {len(enhanced)} rows — using raw Parquet")
                        enhanced = None
                except Exception as e:
                    log.warning(f"  {symbol} {tf}m: institutional features failed ({e}) "
                                f"— using raw Parquet")

            best   = enhanced if (enhanced is not None and len(enhanced) >= MIN_BARS) else raw
            source = "TICK+INSTITUTIONAL" if best is enhanced else "TICK"

            if len(best) >= MIN_BARS:
                tf_data[tf] = best
                log.info(f"  {symbol} {tf}m: {len(best):,} bars "
                         f"| {len(get_feature_cols(best))} features [{source}]")

                if save_featured:
                    _save_featured_parquet(symbol, tf, best)

        if tf_data:
            return tf_data
        log.warning(f"  No tick Parquet loaded for {symbol}, falling back to MT5")

    return _load_from_mt5(symbol)


def _save_featured_parquet(symbol: str, tf: int, df: pd.DataFrame):
    feat_path = DATA_DIR / f"{symbol}_{tf}m_featured.parquet"
    if not feat_path.exists() or FORCE_RETRAIN:
        try:
            df.to_parquet(feat_path, index=True)
            log.info(f"  Saved featured parquet → {feat_path.name} ({len(df.columns)} cols)")
        except Exception as e:
            log.warning(f"  Could not save featured parquet for {symbol} {tf}m: {e}")


def _load_from_mt5(symbol: str) -> dict:
    log.warning(f"  [{symbol}] Using MT5 data — ~3-5 months history only")
    raw_tfs = fetch_all_timeframes(symbol)
    tf_data = {}
    for tf, raw_df in raw_tfs.items():
        if raw_df.empty:
            continue
        enhanced = engineer_full_features(raw_df, tf_minutes=tf, symbol=symbol)
        if len(enhanced) >= MIN_BARS:
            tf_data[tf] = enhanced
            log.info(f"  {symbol} {tf}m: {enhanced.index[0].date()} → "
                     f"{enhanced.index[-1].date()} | "
                     f"{len(enhanced):,} bars | "
                     f"{len(get_feature_cols(enhanced))} features [MT5]")
        else:
            log.warning(f"  {symbol} {tf}m: only {len(enhanced):,} bars from MT5 — skipping")
    return tf_data


# ─────────────────────────────────────────────────────────────────────
# 4. LIVE DATA REFRESH
# ─────────────────────────────────────────────────────────────────────

def refresh_live_data(symbol: str, tf: int,
                      existing_df: pd.DataFrame,
                      is_tick_symbol: bool) -> pd.DataFrame:
    """
    Fetch latest N bars from MT5 and append to historical DataFrame.
    History stays tick-derived; only the live tail comes from MT5.
    This is the correct approach — deep tick history + fresh MT5 tail.
    """
    fresh = fetch_bars(symbol, tf, n_bars=LIVE_REFRESH_BARS)
    if fresh.empty:
        return existing_df
    fresh_feat = engineer_full_features(fresh, tf_minutes=tf, symbol=symbol)
    if fresh_feat.empty:
        return existing_df
    combined = pd.concat([existing_df, fresh_feat])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    if len(combined) > 5000:
        combined = combined.iloc[-5000:]
    return combined


# ─────────────────────────────────────────────────────────────────────
# 5. LOAD MODELS FROM DISK (inference mode — no training imports needed)
# ─────────────────────────────────────────────────────────────────────

def models_exist(symbol: str) -> bool:
    """True if all core model files are present for every entry TF."""
    for tf in PARAM_SEEDS["entry_tf_options"]:
        key = f"{symbol}_{tf}m"
        if not (MODEL_DIR / f"xgb_{key}.pkl").exists():  return False
        if not (MODEL_DIR / f"rf_{key}.pkl").exists():   return False
        if not (MODEL_DIR / f"scaler_{key}.pkl").exists(): return False
    if not (PARAMS_DIR / f"{symbol}_params.json").exists(): return False
    return True


def load_models_from_disk(symbol: str, tf_feat: dict) -> tuple[dict, dict]:
    """
    Load saved models and scalers for one symbol.
    Returns (models_cache, scalers_cache) dicts.
    """
    models_cache  = {}
    scalers_cache = {}

    for tf in PARAM_SEEDS["entry_tf_options"]:
        if tf not in tf_feat:
            continue
        key = f"{symbol}_{tf}m"

        sc_path = MODEL_DIR / f"scaler_{key}.pkl"
        if sc_path.exists():
            scalers_cache[key] = joblib.load(sc_path)

        calib_path = MODEL_DIR / f"calibrator_{key}.pkl"
        if calib_path.exists():
            scalers_cache[f"calibrator_{key}"] = joblib.load(calib_path)

        drift_path = MODEL_DIR / f"drift_ref_{key}.pkl"
        if drift_path.exists():
            scalers_cache[f"drift_ref_{key}"] = joblib.load(drift_path)

        xgb_path = MODEL_DIR / f"xgb_{key}.pkl"
        rf_path  = MODEL_DIR / f"rf_{key}.pkl"
        if xgb_path.exists():
            models_cache[f"xgb_{key}"] = joblib.load(xgb_path)
        if rf_path.exists():
            models_cache[f"rf_{key}"]  = joblib.load(rf_path)
        if xgb_path.exists() and rf_path.exists():
            log.info(f"  Loaded ensemble: {symbol} {tf}m"
                     + (" [calibrated]" if calib_path.exists() else ""))

    params = load_params(symbol)
    log.info(f"  Params : entry_tf={params.get('entry_tf')}m  "
             f"htf={params.get('htf_tf')}m  "
             f"confidence={params.get('confidence',0):.2f}")

    return models_cache, scalers_cache
