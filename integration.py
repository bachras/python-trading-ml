"""
integration.py — Unified Pipeline Connector
=============================================
Wires tick_pipeline.py + institutional_features.py + phase2_adaptive_engine.py
into a single coherent system.

This file REPLACES the following functions in phase2_adaptive_engine.py:
  - engineer_features()          → now calls full institutional stack
  - run_historical_training()    → now uses tick Parquet for US30,
                                   MT5 bars for all other symbols
  - add_htf_alignment()          → now uses institutional HTF features
  - run_live_loop() data refresh → blends tick history + live MT5 bars

How it works:
  - US30 (or any symbol with tick data) → load from Parquet,
    apply institutional features, feed to ML
  - All other symbols → fetch from MT5, apply institutional features
    where possible (spread/volume data may be limited)
  - HTF data for all symbols uses the same institutional feature stack
  - Live loop: on each candle close, fetches fresh MT5 bars and
    appends them to the tick-derived history so models stay current

Usage:
  Replace your phase2_adaptive_engine.py entry point with:

      from integration import run_full_system
      run_full_system()

  That's it. Everything else is handled internally.
"""

import os, sys, time, json, logging, warnings

# Suppress TensorFlow C++ startup noise BEFORE any TF import.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")

# Force UTF-8 on the Windows console so Unicode separators in log
# messages don't crash with cp1252 UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone, timedelta
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import joblib
from dotenv import load_dotenv
from tensorflow.keras.models import load_model as keras_load_model
from stable_baselines3 import PPO

# Suppress only known-harmless noisy warnings — leave everything else visible
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*Converting sparse.*", category=UserWarning)
load_dotenv()

# ── Import our three modules ───────────────────────────────────────
# tick_pipeline    : streams CSV, builds Parquet, loads tick-derived bars
# institutional_features : adds VWAP/VP/delta/liquidity/regime features
# phase2_adaptive_engine : ML models, RL, GA, Optuna, risk gate, MT5 orders

from tick_pipeline import (
    load_for_training       as load_tick_parquet,
    load_ohlcv_parquet,
    parquet_exists,
    build_all_timeframes,
    TARGET_TIMEFRAMES       as TICK_TF_LIST,
    DATA_DIR,
    SYMBOL                  as TICK_SYMBOL,
)

from institutional_features import (
    add_institutional_features,
    get_institutional_feature_report,
)

# Import everything from phase2 except the functions we are replacing
import phase2_adaptive_engine as p2
from phase2_adaptive_engine import (
    connect_mt5, get_account_balance,
    fetch_bars, fetch_all_timeframes,
    get_feature_cols, fit_scaler, apply_scaler,
    make_sequences, build_lstm, train_lstm,
    train_ensemble, train_rl_agent,
    TradingEnv, run_genetic_algo, run_optuna, run_per_tf_optimization,
    save_params, load_params,
    get_signal, RiskGate, place_order,
    log_trade, load_trade_log, should_retrain,
    incremental_update, seconds_to_next_candle_close,
    INSTRUMENTS, PARAM_SEEDS, HARD_LIMITS,
    MODEL_DIR, LOG_DIR, PARAMS_DIR, SEQ_LEN, MIN_BARS,
    OPTUNA_TRIALS, OPTUNA_LIVE_TRIALS,
)

from db import (
    init_db, upsert_strategy, save_equity_curve, save_monthly_pnl,
    save_optuna_trials, get_top_strategy, get_all_strategies,
    set_strategy_active, get_capital_at_risk, get_open_trades,
    log_live_trade, update_trade_be, close_live_trade,
)
from backtest_engine import backtest_all_strategies, load_featured_df

log = logging.getLogger("integration")

# ── New config from .env ──────────────────────────────────────────────
RISK_MODE        = os.getenv("RISK_MODE", "percent")          # percent or fixed
RISK_PCT         = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
FIXED_RISK_AMT   = float(os.getenv("FIXED_RISK_AMOUNT",  "100.0"))
RISK_CAP_PCT     = float(os.getenv("RISK_CAP_PCT",       "2.0"))
RISK_CAP_AMOUNT  = float(os.getenv("RISK_CAP_AMOUNT",    "200.0"))
ER_MULTIPLIER    = float(os.getenv("ER_MULTIPLIER",       "1.25"))
BACKTEST_START   = os.getenv("BACKTEST_START_DATE",       "2020-01-02")
PER_TF_TRIALS    = int(os.getenv("PER_TF_TRIALS",         "150"))
TOP_N_STRATEGIES = int(os.getenv("TOP_N_STRATEGIES",      "5"))
EXTRA_STRATEGIES = [s.strip() for s in os.getenv("EXTRA_STRATEGIES", "").split(",") if s.strip()]

# Initialise SQLite DB on import
init_db()

# ─────────────────────────────────────────────────────────────────────
# TRAINING CONTROL
# ─────────────────────────────────────────────────────────────────────

# Set FORCE_RETRAIN=true in .env to always retrain from scratch.
# Leave false (default) so restarting the machine just reloads saved models
# and goes straight to live trading — no waiting through training again.
# Set to true after changing features in tick_pipeline.py or
# institutional_features.py so the models re-learn the new feature set.
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

# Symbols that have tick Parquet files available
# Add more symbols here as you build their Parquet files
# ── Tick CSV paths — set in .env, no code changes needed to move drives ──
# In .env:
#   BASE_DIR=F:\trading_ml
#   TICK_FILE_US30=F:\trading_ml\tick_data\US30_ticks.csv
#   TICK_FILE_DE40=F:\trading_ml\tick_data\DE40_ticks.csv
_BASE_DIR = Path(os.getenv("BASE_DIR", r"C:\trading_ml"))
def _tick_path(sym: str) -> Path:
    env_key = f"TICK_FILE_{sym}"
    default = str(_BASE_DIR / "tick_data" / f"{sym}_ticks.csv")
    return Path(os.getenv(env_key, default))

TICK_DATA_SYMBOLS = {
    sym: _tick_path(sym)
    for sym in os.getenv("TICK_SYMBOLS", "US30").split(",")
}

# Session bars for volume profile lookback per timeframe
# = roughly 1 trading day of bars at each TF
SESSION_BARS = {
    1:  390,    # 1m  → 390 bars = 6.5h (US30 session)
    3:  130,    # 3m  → 130 bars
    5:  78,     # 5m  → 78 bars
    10: 39,     # 10m → 39 bars
    15: 26,     # 15m → 26 bars
    30: 13,     # 30m → 13 bars
    60: 35,     # 1H  → 35 bars ≈ 1 trading week
}

# Live bar buffer: how many fresh MT5 bars to append on each loop
LIVE_REFRESH_BARS = 200


# ─────────────────────────────────────────────────────────────────────
# 1. UNIFIED FEATURE ENGINEERING
#    Replaces phase2.engineer_features() for ALL symbols
# ─────────────────────────────────────────────────────────────────────

def engineer_full_features(df: pd.DataFrame,
                            tf_minutes: int = 5,
                            symbol: str = "",
                            is_tick_derived: bool = False) -> pd.DataFrame:
    """
    Full feature pipeline — standard technical + institutional features.

    For tick-derived bars (from Parquet): institutional features are
    fully populated including microstructure (spread, delta, tick count).

    For MT5-bar data (non-tick): microstructure columns are estimated
    from available OHLCV — delta from price direction * volume,
    spread from ATR fraction. Less precise but still useful.
    """
    if df.empty or len(df) < MIN_BARS:
        return df

    # ── Step 1: standard technical features (from phase2) ──────────
    d = p2.engineer_features(df)
    if d.empty:
        return d

    # ── Step 2: estimate microstructure if not tick-derived ─────────
    if not is_tick_derived:
        _estimate_microstructure(d)

    # ── Step 3: institutional features ──────────────────────────────
    sb = SESSION_BARS.get(tf_minutes, 78)
    try:
        d = add_institutional_features(d, session_bars=sb, verbose=False)
    except Exception as e:
        log.warning(f"Institutional features partial failure on {symbol} {tf_minutes}m: {e}")
        # Continue with standard features if institutional fails
        pass

    d.dropna(inplace=True)
    return d


def _estimate_microstructure(df: pd.DataFrame):
    """
    Estimate tick-exclusive columns from OHLCV when tick data unavailable.
    These are approximations — still useful signal, just less precise.
    """
    c = df["Close"]
    v = df["Volume"]
    atr = df.get("atr14", (df["High"] - df["Low"]).rolling(14).mean())

    # Estimate delta from price direction × volume
    direction = np.sign(c.diff()).fillna(0)
    df["ask_volume"]   = v * ((direction + 1) / 2).clip(0, 1)
    df["bid_volume"]   = v * ((1 - direction) / 2).clip(0, 1)
    df["vol_imbalance"] = direction * 0.5

    # Estimate spread from ATR fraction (typical for the instrument)
    df["spread_mean"]  = atr * 0.05
    df["spread_max"]   = atr * 0.15
    df["spread_std"]   = atr * 0.02

    # Estimate tick count from volume (rough proxy)
    vol_sma = v.rolling(20).mean()
    df["tick_count"]    = (v / (vol_sma + 1e-10) * 50).clip(1, 500)
    df["tick_velocity"] = df["tick_count"] / 300   # per second in 5m bar

    # VWAP approximation from typical price
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    dates    = df.index.date
    cum_tpv  = (tp * v).groupby(dates).cumsum()
    cum_vol  = v.groupby(dates).cumsum()
    df["vwap"] = cum_tpv / (cum_vol + 1e-10)

    df["price_impact"] = (df["High"] - df["Low"]) / (v + 1e-10)
    df["true_return"]  = df.get("logr1", c.pct_change())


# ─────────────────────────────────────────────────────────────────────
# 2. HTF ALIGNMENT — upgraded to use institutional HTF features
#    Replaces phase2.add_htf_alignment()
# ─────────────────────────────────────────────────────────────────────

def add_htf_alignment_full(entry_df: pd.DataFrame,
                            htf_df: pd.DataFrame,
                            htf_weight: float,
                            htf_tf: int = 60) -> pd.DataFrame:
    """
    Enhanced HTF alignment using full institutional feature stack.
    Now uses POC position, CVD direction, VWAP slope, and regime
    instead of just EMA direction.
    """
    if htf_df.empty or htf_weight == 0:
        return entry_df

    htf_feat = engineer_full_features(
        htf_df.copy(), tf_minutes=htf_tf, is_tick_derived=False
    )
    if htf_feat.empty:
        return entry_df

    # HTF direction signals (multiple confirmations)
    ema_dir = (htf_feat["Close"] > htf_feat.get("ema55", htf_feat["Close"])).astype(int) * 2 - 1

    # VWAP direction (if available)
    vwap_dir = pd.Series(0, index=htf_feat.index)
    if "vwap_session" in htf_feat.columns:
        vwap_dir = (htf_feat["Close"] > htf_feat["vwap_session"]).astype(int) * 2 - 1

    # Volume delta direction
    delta_dir = pd.Series(0, index=htf_feat.index)
    if "cvd_slope5" in htf_feat.columns:
        delta_dir = np.sign(htf_feat["cvd_slope5"])

    # POC position
    poc_dir = pd.Series(0, index=htf_feat.index)
    if "vp_poc_dist" in htf_feat.columns:
        poc_dir = np.sign(htf_feat["vp_poc_dist"])

    # Regime (1=bull trend, -1=bear trend, 0=other)
    regime_dir = pd.Series(0, index=htf_feat.index)
    if "regime_bull" in htf_feat.columns:
        regime_dir = htf_feat["regime_bull"] - htf_feat["regime_bear"]

    # Composite HTF signal: weighted vote across 5 signals
    htf_composite = (
        ema_dir   * 0.25 +
        vwap_dir  * 0.25 +
        delta_dir * 0.20 +
        poc_dir   * 0.15 +
        regime_dir * 0.15
    )
    htf_composite.name = "htf_bullish"

    # HTF strength (magnitude of composite score, scaled by htf_weight)
    htf_strength = (htf_composite.abs() * htf_weight).rename("htf_strength")

    # Resample HTF to entry TF index (forward fill — HTF bar persists)
    htf_composite = htf_composite.resample("1min").last().ffill()
    htf_strength  = htf_strength.resample("1min").last().ffill()

    entry_df["htf_bullish"]  = htf_composite.reindex(entry_df.index, method="ffill").fillna(0)
    entry_df["htf_strength"] = htf_strength.reindex(entry_df.index, method="ffill").fillna(0)

    return entry_df

# Monkey-patch phase2 so get_signal() uses our upgraded HTF function
p2.add_htf_alignment = add_htf_alignment_full


# ─────────────────────────────────────────────────────────────────────
# 3. DATA LOADING — tick Parquet for US30, MT5 for others
# ─────────────────────────────────────────────────────────────────────

def load_symbol_data(symbol: str) -> dict:
    """
    Load the best available data for a symbol across all TFs.
    Returns {tf_minutes: feature_DataFrame}.

    Priority order:
      1. Tick Parquet files (highest quality — US30 with your CSV)
      2. MT5 historical bars (all other symbols)

    For US30: if Parquet doesn't exist yet, builds it from CSV first.
    """
    log.info(f"Loading data: {symbol}")

    # ── Tick data path ──────────────────────────────────────────
    if symbol in TICK_DATA_SYMBOLS:
        tick_file = TICK_DATA_SYMBOLS[symbol]
        tfs_needed = sorted(set(
            PARAM_SEEDS["entry_tf_options"] +
            [t for t in PARAM_SEEDS["htf_options"] if t > 0]
        ))

        all_built = all(parquet_exists(symbol, tf) for tf in tfs_needed)

        if not all_built:
            if tick_file.exists():
                log.info(f"  Building Parquet from tick CSV: {tick_file}")
                log.info(f"  This takes 30-60 min for 10GB. Only runs once.")
                build_all_timeframes(tick_file, symbol, tfs_needed)
            else:
                log.warning(f"  Tick CSV not found: {tick_file}")
                log.warning(f"  Falling back to MT5 data for {symbol}")
                return _load_from_mt5(symbol)

        # Load from Parquet and apply institutional features.
        # Parquet files from tick_pipeline already have engineer_tick_features
        # applied (~80 features). Institutional features add ~45 more.
        #
        # For very high-frequency TFs (1m, 3m) the institutional Python loops
        # (volume profile, anchored VWAP) iterate over millions of bars and
        # take hours — we skip them for TFs above MAX_INST_BARS and use the
        # rich tick features already in the Parquet instead.
        #
        # CRITICAL: we ALWAYS prefer Parquet over MT5 fallback, even without
        # institutional features — 6 years of tick data >> 3.5 months from MT5.
        MAX_INST_BARS = 750_000   # skip slow Python loops above this bar count

        tf_data = {}
        for tf in tfs_needed:
            raw = load_ohlcv_parquet(symbol, tf)
            if raw.empty:
                continue

            sb = SESSION_BARS.get(tf, 78)
            enhanced = None

            if len(raw) > MAX_INST_BARS:
                # Too many bars for the O(n) Python loops in institutional
                # features — use raw Parquet data which already has 80+ features
                log.info(f"  {symbol} {tf}m: {len(raw):,} bars — skipping "
                         f"slow institutional loops (>{MAX_INST_BARS:,} bars), "
                         f"using tick features directly")
            else:
                try:
                    enhanced = add_institutional_features(
                        raw, session_bars=sb, verbose=False
                    )
                    if len(enhanced) < MIN_BARS:
                        log.warning(f"  {symbol} {tf}m: institutional features "
                                    f"produced only {len(enhanced)} rows "
                                    f"(dropna removed too many) — using raw Parquet")
                        enhanced = None
                except Exception as e:
                    log.warning(f"  {symbol} {tf}m: institutional features "
                                f"failed ({e}) — using raw Parquet")

            # Use enhanced if we got it, otherwise raw Parquet
            # NEVER fall through to MT5 if we have Parquet data
            best = enhanced if (enhanced is not None and len(enhanced) >= MIN_BARS) else raw
            if len(best) >= MIN_BARS:
                tf_data[tf] = best
                source = "TICK+INSTITUTIONAL" if best is enhanced else "TICK"
                log.info(f"  {symbol} {tf}m: {len(best):,} bars "
                         f"| {len(get_feature_cols(best))} features [{source}]")

                # Save featured parquet for accurate backtesting in report
                feat_path = DATA_DIR / f"{symbol}_{tf}m_featured.parquet"
                if not feat_path.exists() or FORCE_RETRAIN:
                    try:
                        best.to_parquet(feat_path, index=True)
                        log.info(f"  Saved featured parquet → {feat_path.name} ({len(best.columns)} cols)")
                    except Exception as e:
                        log.warning(f"  Could not save featured parquet for {symbol} {tf}m: {e}")

        if tf_data:
            return tf_data
        log.warning(f"  No tick Parquet loaded for {symbol}, falling back to MT5")

    # ── MT5 data path ────────────────────────────────────────────
    return _load_from_mt5(symbol)


def _load_from_mt5(symbol: str) -> dict:
    """Fetch max history from MT5 and apply full feature pipeline."""
    log.warning(f"  [{symbol}] Using MT5 data — expect ~3-5 months history only")
    raw_tfs = fetch_all_timeframes(symbol)
    tf_data = {}
    for tf, raw_df in raw_tfs.items():
        if raw_df.empty:
            continue
        enhanced = engineer_full_features(
            raw_df, tf_minutes=tf, symbol=symbol, is_tick_derived=False
        )
        if len(enhanced) >= MIN_BARS:
            tf_data[tf] = enhanced
            log.info(
                f"  {symbol} {tf}m: {enhanced.index[0].date()} → "
                f"{enhanced.index[-1].date()} | "
                f"{len(enhanced):,} bars | "
                f"{len(get_feature_cols(enhanced))} features [MT5]"
            )
        else:
            log.warning(f"  {symbol} {tf}m: only {len(enhanced):,} bars from MT5 "
                        f"(need {MIN_BARS}) — skipping this TF")
    return tf_data


# ─────────────────────────────────────────────────────────────────────
# 4. HISTORICAL TRAINING — replaces phase2.run_historical_training()
# ─────────────────────────────────────────────────────────────────────

def run_historical_training() -> tuple:
    """
    Full training pipeline using best available data per symbol.
    US30: tick Parquet + institutional features (~110 features)
    Others: MT5 bars + estimated microstructure (~85 features)
    """
    log.info("=" * 60)
    log.info("UNIFIED HISTORICAL TRAINING")
    log.info("=" * 60)

    models_cache  = {}
    scalers_cache = {}
    all_data      = {}

    for symbol in INSTRUMENTS:
        log.info(f"\n{'-'*50}")
        log.info(f"Symbol: {symbol}")

        tf_feat = load_symbol_data(symbol)

        if not tf_feat:
            log.warning(f"No data loaded for {symbol} — skipping")
            continue

        all_data[symbol] = tf_feat

        # ── Train models on each entry TF candidate ──────────────
        for tf in PARAM_SEEDS["entry_tf_options"]:
            if tf not in tf_feat:
                continue

            key = f"{symbol}_{tf}m"

            # Resume mode: skip TFs whose models are already on disk
            # (unless FORCE_RETRAIN=true which means retrain everything)
            if not FORCE_RETRAIN:
                lstm_p  = MODEL_DIR / f"lstm_{key}.keras"
                xgb_p   = MODEL_DIR / f"xgb_{key}.pkl"
                rf_p    = MODEL_DIR / f"rf_{key}.pkl"
                scaler_p = MODEL_DIR / f"scaler_{key}.pkl"
                if lstm_p.exists() and xgb_p.exists() and rf_p.exists():
                    log.info(f"  {symbol} {tf}m: models found on disk — loading (FORCE_RETRAIN=false)")
                    models_cache[f"lstm_{key}"] = keras_load_model(str(lstm_p))
                    models_cache[f"xgb_{key}"]  = joblib.load(xgb_p)
                    models_cache[f"rf_{key}"]   = joblib.load(rf_p)
                    if scaler_p.exists():
                        scalers_cache[key] = joblib.load(scaler_p)
                    continue

            df   = tf_feat[tf]
            n    = len(df)
            i_tr = int(n * 0.70)
            i_va = int(n * 0.85)

            train_df = df.iloc[:i_tr]
            val_df   = df.iloc[i_tr:i_va]
            test_df  = df.iloc[i_va:]
            feat_cols = get_feature_cols(df)

            log.info(
                f"  {symbol} {tf}m | "
                f"{df.index[0].date()} → {df.index[-1].date()} | "
                f"{n:,} bars total | "
                f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,} | "
                f"{len(feat_cols)} features"
            )

            # NaN audit — warn about any feature column with >10% missing in training set
            nan_pct = train_df[feat_cols].isna().mean()
            bad_cols = nan_pct[nan_pct > 0.10]
            if not bad_cols.empty:
                log.warning(
                    f"  {symbol} {tf}m: {len(bad_cols)} features have >10% NaN in train set — "
                    + ", ".join(f"{c}={v:.0%}" for c, v in bad_cols.items())
                )

            # Target class balance — warn if heavily skewed (>80/20)
            if "target" in train_df.columns:
                pos = train_df["target"].mean()
                if pos < 0.20 or pos > 0.80:
                    log.warning(
                        f"  {symbol} {tf}m: imbalanced target — "
                        f"{pos:.0%} positive in train set (ideal ~50%)"
                    )
                else:
                    log.info(f"  {symbol} {tf}m: target balance {pos:.0%} positive")

            # Fit scaler on training data only
            scaler   = fit_scaler(train_df, feat_cols)
            train_s  = apply_scaler(train_df, scaler, feat_cols)
            val_s    = apply_scaler(val_df,   scaler, feat_cols)

            scalers_cache[key] = scaler
            joblib.dump(scaler, MODEL_DIR / f"scaler_{key}.pkl")
            log.info(f"  Scaler saved → scaler_{key}.pkl")

            # LSTM
            lstm = train_lstm(train_s, val_s, feat_cols, symbol, tf)
            if lstm:
                models_cache[f"lstm_{key}"] = lstm

            # XGBoost + Random Forest
            xgb_m, rf_m = train_ensemble(
                train_s, val_s, feat_cols, symbol, tf
            )
            models_cache[f"xgb_{key}"] = xgb_m
            models_cache[f"rf_{key}"]  = rf_m

            # Feature importance report (shows what ML values)
            if xgb_m is not None:
                log.info(f"  Feature importance for {symbol} {tf}m:")
                report = get_institutional_feature_report(
                    xgb_m, feat_cols, top_n=10
                )

        # ── RL agent on default TF ────────────────────────────────
        default_tf = PARAM_SEEDS["entry_tf_default"]
        if default_tf in tf_feat:
            default_key = f"{symbol}_{default_tf}m"
            df_rl = apply_scaler(
                tf_feat[default_tf],
                scalers_cache.get(default_key),
                get_feature_cols(tf_feat[default_tf]),
            ) if scalers_cache.get(default_key) else tf_feat[default_tf]

            rl = train_rl_agent(
                df_rl,
                get_feature_cols(df_rl),
                symbol,
                sl_mult = PARAM_SEEDS["sl_atr_seed"],
                rr      = PARAM_SEEDS["rr_seed"],
            )
            models_cache[f"ppo_{symbol}"] = rl

        # ── Genetic algorithm + Optuna parameter search ───────────
        # Use per-TF models/scalers so any genome entry_tf is valid
        has_any_scaler = any(
            scalers_cache.get(f"{symbol}_{tf}m")
            for tf in PARAM_SEEDS["entry_tf_options"]
            if tf in tf_feat
        )

        if has_any_scaler:
            log.info(f"  Running GA parameter search: {symbol}")
            ga_params  = run_genetic_algo(
                tf_feat, models_cache, scalers_cache, symbol=symbol
            )

            log.info(f"  Running Optuna parameter search: {symbol}")
            opt_params = run_optuna(
                tf_feat, models_cache, scalers_cache,
                n_trials=OPTUNA_TRIALS, symbol=symbol
            )

            best = (opt_params
                    if opt_params.get("best_score", -999) >
                       ga_params.get("best_score", -999)
                    else ga_params)
            best["source"]     = "historical_optimisation"
            best["data_source"] = ("tick+institutional"
                                   if symbol in TICK_DATA_SYMBOLS
                                   else "mt5+estimated")
            save_params(symbol, best)

            log.info(f"\n  {symbol} OPTIMAL PARAMS:")
            log.info(f"    Entry TF   : {best['entry_tf']}m")
            log.info(f"    HTF        : {best['htf_tf']}m "
                     f"({'NONE' if best['htf_tf']==0 else 'active'})")
            log.info(f"    SL ATR×    : {best['sl_atr']:.3f}")
            log.info(f"    R:R        : 1:{best['rr']:.2f}")
            log.info(f"    TP mult    : {best['tp_mult']:.2f}x")
            log.info(f"    Confidence : {best['confidence']:.2f}")
            log.info(f"    HTF weight : {best['htf_weight']:.2f}")
            be = best.get('be_r', 0)
            log.info(f"    Break-even : {'OFF' if be == 0 else f'+{be}R trigger → entry+1pt'}")
            log.info(f"    Data source: {best['data_source']}")

        # ── Per-TF optimization: top-5 param sets per timeframe ──────
        log.info(f"\n  Per-TF optimization: {symbol} (top-{TOP_N_STRATEGIES} per TF)")
        log.info(f"  This runs once — results saved to SQLite for reporting")

        for tf in PARAM_SEEDS["entry_tf_options"]:
            if tf not in tf_feat:
                continue
            key = f"{symbol}_{tf}m"
            if not scalers_cache.get(key):
                log.info(f"  {symbol} {tf}m: no scaler — skipping per-TF opt")
                continue

            top_trials, all_trials = run_per_tf_optimization(
                df_dict       = tf_feat,
                models_by_tf  = models_cache,
                scalers_by_tf = scalers_cache,
                symbol        = symbol,
                locked_tf     = tf,
                n_trials      = PER_TF_TRIALS,
                top_n         = TOP_N_STRATEGIES,
            )

            # Store all raw trials to SQLite
            save_optuna_trials(symbol, tf, all_trials)

            # Run full backtest on top-N and store to SQLite
            if top_trials:
                top_params = [t["params"] for t in top_trials]
                bt_results = backtest_all_strategies(
                    symbol       = symbol,
                    tf           = tf,
                    top_params   = top_params,
                    risk_mode    = RISK_MODE,
                    risk_pct     = RISK_PCT,
                    fixed_amt    = FIXED_RISK_AMT,
                    start_balance= 10_000.0,
                )

                for rank, (trial, bt) in enumerate(zip(top_trials, bt_results), 1):
                    stats = bt.get("stats")
                    if stats is None:
                        continue
                    strategy_id = f"{symbol}_{tf}m_rank{rank}"
                    row = {
                        "strategy_id":      strategy_id,
                        "symbol":           symbol,
                        "tf":               tf,
                        "rank":             rank,
                        "entry_tf":         trial["params"]["entry_tf"],
                        "htf_tf":           trial["params"]["htf_tf"],
                        "sl_atr":           trial["params"]["sl_atr"],
                        "rr":               trial["params"]["rr"],
                        "tp_mult":          trial["params"]["tp_mult"],
                        "confidence":       trial["params"]["confidence"],
                        "htf_weight":       trial["params"]["htf_weight"],
                        "be_r":             trial["params"]["be_r"],
                        "sharpe":           trial["sharpe"],
                        "efficiency_ratio": stats["efficiency_ratio"],
                        "win_rate":         stats["win_rate"],
                        "profit_factor":    stats["profit_factor"],
                        "max_dd_pct":       stats["max_dd_pct"],
                        "max_dd_money":     stats["max_dd_money"],
                        "total_profit":     stats["total_profit"],
                        "n_trades":         stats["n_trades"],
                        "expectancy":       stats["expectancy"],
                        "is_active":        0,
                    }
                    upsert_strategy(row)

                    if not bt["equity_df"].empty:
                        save_equity_curve(strategy_id, bt["equity_df"])
                    if bt["monthly_pnl"]:
                        save_monthly_pnl(strategy_id, bt["monthly_pnl"])

                    log.info(f"  {strategy_id}: Sharpe={trial['sharpe']:.2f} "
                             f"ER={stats['efficiency_ratio']:.2f} "
                             f"WR={stats['win_rate']:.1f}% "
                             f"PF={stats['profit_factor']:.2f} "
                             f"MaxDD={stats['max_dd_pct']:.1f}%")

                # Auto-activate rank 1 (best by ER after backtest re-ranking)
                _activate_best_strategies(symbol)

    log.info("\nHistorical training complete.")
    return models_cache, scalers_cache, all_data


def _activate_best_strategies(symbol: str):
    """Mark top strategy by efficiency ratio as active, deactivate others."""
    from db import get_all_strategies
    strategies = get_all_strategies(symbol)
    if not strategies:
        return
    # Sort by efficiency_ratio descending, pick #1
    valid = [s for s in strategies if s.get("efficiency_ratio") is not None]
    if not valid:
        return
    valid.sort(key=lambda s: s["efficiency_ratio"], reverse=True)
    best_id = valid[0]["strategy_id"]
    for s in strategies:
        set_strategy_active(s["strategy_id"], s["strategy_id"] == best_id)
    log.info(f"  Auto-activated strategy: {best_id} "
             f"(ER={valid[0]['efficiency_ratio']:.2f})")


# ─────────────────────────────────────────────────────────────────────
# 5. LIVE DATA REFRESH — blends tick history + fresh MT5 bars
# ─────────────────────────────────────────────────────────────────────

def refresh_live_data(symbol: str, tf: int,
                      existing_df: pd.DataFrame,
                      is_tick_symbol: bool) -> pd.DataFrame:
    """
    On each candle close, fetch the latest N bars from MT5 and
    append them to the historical DataFrame so the model sees
    both deep historical context AND the most recent price action.

    For tick symbols: history is tick-derived (high quality),
    live tail is MT5 bars (slightly lower quality but current).
    This is the correct approach — don't discard tick history.
    """
    fresh = fetch_bars(symbol, tf, n_bars=LIVE_REFRESH_BARS)
    if fresh.empty:
        return existing_df

    # Apply feature engineering to the fresh bars
    fresh_feat = engineer_full_features(
        fresh, tf_minutes=tf, symbol=symbol,
        is_tick_derived=False   # live bars are from MT5, not tick
    )
    if fresh_feat.empty:
        return existing_df

    # Merge: keep historical data, append any new bars not already present
    combined = pd.concat([existing_df, fresh_feat])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    # Keep last 5000 bars to bound memory in live loop
    if len(combined) > 5000:
        combined = combined.iloc[-5000:]

    return combined


# ─────────────────────────────────────────────────────────────────────
# 5b. SMART STARTUP — check for saved models, load without retraining
# ─────────────────────────────────────────────────────────────────────

def _models_exist(symbol: str) -> bool:
    """
    Return True if the core models (XGB + RF + scaler for every entry TF,
    and optimised params) exist on disk.
    PPO is optional — its crash should not force a full retrain.
    """
    for tf in PARAM_SEEDS["entry_tf_options"]:
        key = f"{symbol}_{tf}m"
        if not (MODEL_DIR / f"xgb_{key}.pkl").exists():
            return False
        if not (MODEL_DIR / f"rf_{key}.pkl").exists():
            return False
        if not (MODEL_DIR / f"scaler_{key}.pkl").exists():
            return False
    if not (PARAMS_DIR / f"{symbol}_params.json").exists():
        return False
    return True  # PPO is optional — missing PPO does not trigger retrain


def load_trained_system() -> tuple:
    """
    Load all models, scalers, and data from disk — no retraining.
    Called on restart when FORCE_RETRAIN=false and all model files exist.

    Data loading (Parquet + institutional features) still runs (~2-5 min)
    because the live loop needs up-to-date DataFrames in memory.
    Model training (~20-40 min) and GA/Optuna (~10-20 min) are skipped.
    """
    log.info("=" * 60)
    log.info("LOADING SAVED MODELS FROM DISK")
    log.info("Skipping training — set FORCE_RETRAIN=true to retrain")
    log.info("=" * 60)

    models_cache  = {}
    scalers_cache = {}
    all_data      = {}

    for symbol in INSTRUMENTS:
        log.info(f"\n{'-'*50}")
        log.info(f"Loading: {symbol}")

        # ── Load feature data (always required for live loop) ────────
        tf_feat = load_symbol_data(symbol)
        if not tf_feat:
            log.warning(f"  No data for {symbol} — skipping")
            continue
        all_data[symbol] = tf_feat

        # ── Load scalers and models per entry TF ─────────────────────
        for tf in PARAM_SEEDS["entry_tf_options"]:
            if tf not in tf_feat:
                continue
            key = f"{symbol}_{tf}m"

            scaler_path = MODEL_DIR / f"scaler_{key}.pkl"
            if scaler_path.exists():
                scalers_cache[key] = joblib.load(scaler_path)

            lstm_path = MODEL_DIR / f"lstm_{key}.keras"
            if lstm_path.exists():
                models_cache[f"lstm_{key}"] = keras_load_model(str(lstm_path))
                log.info(f"  Loaded LSTM   : {lstm_path.name}")

            xgb_path = MODEL_DIR / f"xgb_{key}.pkl"
            rf_path  = MODEL_DIR / f"rf_{key}.pkl"
            if xgb_path.exists():
                models_cache[f"xgb_{key}"] = joblib.load(xgb_path)
            if rf_path.exists():
                models_cache[f"rf_{key}"]  = joblib.load(rf_path)
            if xgb_path.exists() and rf_path.exists():
                log.info(f"  Loaded ensemble: {symbol} {tf}m")

        # ── Load PPO (inference only — no env needed for predict) ─────
        ppo_path = MODEL_DIR / f"ppo_{symbol}.zip"
        if ppo_path.exists():
            models_cache[f"ppo_{symbol}"] = PPO.load(str(ppo_path))
            log.info(f"  Loaded PPO    : {ppo_path.name}")

        params = load_params(symbol)
        log.info(f"  Params loaded : entry_tf={params.get('entry_tf')}m  "
                 f"htf={params.get('htf_tf')}m  "
                 f"confidence={params.get('confidence', 0):.2f}")

    log.info(f"\nLoaded {len(models_cache)} model objects — ready for live loop.")
    return models_cache, scalers_cache, all_data


# ─────────────────────────────────────────────────────────────────────
# 6. LIVE TRADING LOOP — upgraded version
# ─────────────────────────────────────────────────────────────────────

def _get_active_strategies() -> list[dict]:
    """
    Build the list of active strategies for live trading.
    Auto: top-1 by efficiency ratio per symbol.
    Manual: EXTRA_STRATEGIES from .env (format: US30_3m_rank2).
    """
    from db import get_all_strategies, get_strategy
    active = []

    # Auto: best strategy per symbol
    for symbol in INSTRUMENTS:
        strategies = get_all_strategies(symbol)
        if not strategies:
            # Fall back to saved params.json if no DB strategies yet
            params = load_params(symbol)
            params["strategy_id"] = f"{symbol}_default"
            params["symbol"]      = symbol
            active.append(params)
            continue
        valid = [s for s in strategies if s.get("efficiency_ratio") is not None]
        if valid:
            valid.sort(key=lambda s: s["efficiency_ratio"], reverse=True)
            active.append(valid[0])
        elif strategies:
            active.append(strategies[0])

    # Manual extras from .env
    for sid in EXTRA_STRATEGIES:
        s = get_strategy(sid)
        if s:
            active.append(s)
        else:
            log.warning(f"  EXTRA_STRATEGIES: strategy '{sid}' not in DB — ignored")

    # Deduplicate by strategy_id
    seen = set()
    deduped = []
    for s in active:
        sid = s.get("strategy_id", s.get("symbol", "?"))
        if sid not in seen:
            seen.add(sid)
            deduped.append(s)

    return deduped


def _check_risk_cap(new_risk: float, balance: float) -> bool:
    """
    Returns True if opening a new trade with new_risk is within cap.
    Cap is sum of (risk_amount for open trades with be_done=0) + new_risk.
    Trades where be_done=1 have SL at BE — no capital at risk.
    """
    current_at_risk = get_capital_at_risk()

    if RISK_MODE == "fixed":
        cap = RISK_CAP_AMOUNT
    else:
        cap = balance * RISK_CAP_PCT / 100.0

    total = current_at_risk + new_risk
    if total > cap:
        log.info(f"  Risk cap: at_risk={current_at_risk:.2f} + new={new_risk:.2f} "
                 f"> cap={cap:.2f} — skipping trade")
        return False
    return True


def _monitor_open_positions():
    """
    Check MT5 open positions for BE status.
    If SL has been moved to >= entry (long) or <= entry (short), mark be_done=1.
    """
    open_trades = get_open_trades()
    if not open_trades:
        return

    positions = mt5.positions_get()
    if positions is None:
        return
    pos_by_ticket = {p.ticket: p for p in positions}

    for trade in open_trades:
        ticket = trade.get("ticket")
        if not ticket or ticket not in pos_by_ticket:
            # Position closed — update DB
            if ticket and ticket not in pos_by_ticket:
                close_live_trade(ticket, pnl=0.0)   # PnL unknown without history
            continue

        pos   = pos_by_ticket[ticket]
        entry = trade.get("entry_price", 0)
        direction = trade.get("direction", 1)

        # BE triggered if SL moved past entry
        be_done = trade.get("be_done", 0)
        if not be_done:
            if direction == 1 and pos.sl >= entry:
                update_trade_be(ticket, True)
                log.info(f"  BE detected: ticket={ticket} SL={pos.sl:.2f} >= entry={entry:.2f}")
            elif direction == -1 and pos.sl <= entry and pos.sl > 0:
                update_trade_be(ticket, True)
                log.info(f"  BE detected: ticket={ticket} SL={pos.sl:.2f} <= entry={entry:.2f}")


def run_live_loop(models_cache: dict, scalers_cache: dict,
                  all_data: dict):
    """
    Multi-strategy live loop with BE-aware risk cap.
    Wakes on closed candle of the fastest active entry TF.
    """
    log.info("=" * 60)
    log.info("LIVE TRADING LOOP — Multi-Strategy")
    log.info(f"Hard limits : {HARD_LIMITS}")
    log.info(f"Risk mode   : {RISK_MODE}  cap={'{}%'.format(RISK_CAP_PCT) if RISK_MODE=='percent' else '${}'.format(RISK_CAP_AMOUNT)}")
    log.info("=" * 60)

    risk_gate  = RiskGate()
    last_reset = datetime.now().date()
    live_cache = {}

    # Pre-populate live cache from trained data
    for symbol, tf_dict in all_data.items():
        for tf, df in tf_dict.items():
            live_cache[f"{symbol}_{tf}m"] = df
    log.info(f"Pre-loaded {len(live_cache)} data frames into live cache")

    # Load active strategies
    active_strategies = _get_active_strategies()
    log.info(f"Active strategies ({len(active_strategies)}):")
    for s in active_strategies:
        log.info(f"  {s.get('strategy_id', s.get('symbol','?'))}: "
                 f"TF={s.get('entry_tf')}m HTF={s.get('htf_tf')}m "
                 f"conf={s.get('confidence', 0):.2f} "
                 f"ER={s.get('efficiency_ratio', 'n/a')}")

    while True:
        try:
            now = datetime.now()

            # Midnight reset
            if now.date() > last_reset:
                risk_gate.reset_daily()
                last_reset = now.date()
                log.info("Daily counters reset")

            # Monitor BE status of open positions
            _monitor_open_positions()

            # Sleep until next candle close (fastest entry TF across all strategies)
            min_sleep = float("inf")
            for s in active_strategies:
                tf    = s.get("entry_tf", PARAM_SEEDS["entry_tf_default"])
                sleep = seconds_to_next_candle_close(tf)
                min_sleep = min(min_sleep, sleep)

            log.info(f"Sleeping {min_sleep:.0f}s to next candle close...")
            time.sleep(max(1.0, min_sleep))

            # Re-load strategies in case EXTRA_STRATEGIES changed
            active_strategies = _get_active_strategies()

            # Process each active strategy
            for strategy in active_strategies:
                sid      = strategy.get("strategy_id", strategy.get("symbol", "?"))
                symbol   = strategy.get("symbol", list(INSTRUMENTS.keys())[0])
                entry_tf = strategy.get("entry_tf", PARAM_SEEDS["entry_tf_default"])
                htf_tf   = strategy.get("htf_tf", 0)
                is_tick  = symbol in TICK_DATA_SYMBOLS

                # Build params dict compatible with get_signal()
                params = {
                    "entry_tf":   entry_tf,
                    "htf_tf":     htf_tf,
                    "sl_atr":     strategy.get("sl_atr", PARAM_SEEDS["sl_atr_seed"]),
                    "rr":         strategy.get("rr", PARAM_SEEDS["rr_seed"]),
                    "tp_mult":    strategy.get("tp_mult", PARAM_SEEDS["tp_mult_seed"]),
                    "confidence": strategy.get("confidence", PARAM_SEEDS["confidence_seed"]),
                    "htf_weight": strategy.get("htf_weight", PARAM_SEEDS["htf_weight_seed"]),
                    "be_r":       strategy.get("be_r", 0),
                }

                # Refresh entry TF data
                entry_key = f"{symbol}_{entry_tf}m"
                existing  = live_cache.get(entry_key, pd.DataFrame())
                updated   = refresh_live_data(symbol, entry_tf, existing, is_tick)
                live_cache[entry_key] = updated

                # Refresh HTF
                if htf_tf > 0:
                    htf_key   = f"{symbol}_{htf_tf}m"
                    htf_exist = live_cache.get(htf_key, pd.DataFrame())
                    live_cache[htf_key] = refresh_live_data(symbol, htf_tf, htf_exist, is_tick)

                signal = get_signal(
                    symbol          = symbol,
                    params          = params,
                    models_cache    = models_cache,
                    scalers_cache   = scalers_cache,
                    live_data_cache = live_cache,
                )

                if signal["direction"] == 0:
                    log.debug(f"{sid}: no signal — {signal.get('reason','')}")
                    continue

                # Hard risk gate (daily loss / drawdown)
                allowed, reason = risk_gate.can_trade()
                if not allowed:
                    log.warning(f"[{sid}] Trade blocked by risk gate: {reason}")
                    continue

                # Compute this trade's risk
                balance = get_account_balance()
                if RISK_MODE == "fixed":
                    trade_risk = FIXED_RISK_AMT
                else:
                    trade_risk = max(balance, 1.0) * RISK_PCT / 100.0

                # BE-aware risk cap check
                if not _check_risk_cap(trade_risk, balance):
                    continue

                # Position size
                lot = risk_gate.position_size(
                    sl_pips=signal["sl_pips"], symbol=symbol
                )

                log.info(
                    f"SIGNAL [{sid}] ► {symbol} "
                    f"{'LONG' if signal['direction']==1 else 'SHORT'} | "
                    f"conf:{signal['confidence']:.3f} | "
                    f"TF:{signal['entry_tf']}m HTF:{signal['htf_used']}m | "
                    f"lot:{lot} | SL:{signal['sl_price']} TP:{signal['tp_price']}"
                )

                result = place_order(
                    symbol    = symbol,
                    direction = signal["direction"],
                    lot       = lot,
                    sl        = signal["sl_price"],
                    tp        = signal["tp_price"],
                    comment   = f"{sid[:16]}",
                )

                if result["success"]:
                    # Log to both old JSON log and new SQLite
                    log_trade({
                        "symbol":      symbol,
                        "direction":   signal["direction"],
                        "entry":       signal["entry_price"],
                        "sl":          signal["sl_price"],
                        "tp":          signal["tp_price"],
                        "lot":         lot,
                        "confidence":  signal["confidence"],
                        "entry_tf":    signal["entry_tf"],
                        "htf_tf":      signal["htf_used"],
                        "params":      params,
                        "ticket":      result["ticket"],
                        "data_source": "tick" if is_tick else "mt5",
                        "timestamp":   now.isoformat(),
                        "used_for_retrain": False,
                    })
                    log_live_trade({
                        "strategy_id": sid,
                        "symbol":      symbol,
                        "direction":   signal["direction"],
                        "entry_price": signal["entry_price"],
                        "sl_price":    signal["sl_price"],
                        "tp_price":    signal["tp_price"],
                        "lot":         lot,
                        "confidence":  signal["confidence"],
                        "entry_tf":    signal["entry_tf"],
                        "htf_tf":      signal["htf_used"],
                        "ticket":      result["ticket"],
                        "risk_amount": trade_risk,
                        "data_source": "tick" if is_tick else "mt5",
                        "be_done":     0,
                    })

                # Incremental retrain if enough new trades
                if should_retrain(symbol):
                    log.info(f"Triggering incremental update: {symbol}")
                    incremental_update(
                        symbol        = symbol,
                        df_dict       = all_data.get(symbol, {}),
                        models_cache  = models_cache,
                        scalers_cache = scalers_cache,
                    )

        except KeyboardInterrupt:
            log.info("Live loop stopped.")
            break
        except Exception as e:
            log.error(f"Live loop error: {e}", exc_info=True)
            time.sleep(30)


# ─────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

def run_full_system():
    """
    Single entry point for the entire system.
    Call this instead of running phase2_adaptive_engine.py directly.
    """
    log.info("=" * 60)
    log.info("ADAPTIVE ML TRADING SYSTEM")
    log.info("Tick data + Institutional features + Multi-TF ML")
    log.info("=" * 60)

    if not connect_mt5():
        log.error("Cannot connect to MT5. Ensure terminal is open.")
        return

    try:
        # ── Decide: train fresh or load from disk ────────────────────
        missing = [s for s in INSTRUMENTS if not _models_exist(s)]

        if FORCE_RETRAIN:
            log.info("FORCE_RETRAIN=true — retraining all models from scratch.")
            models_cache, scalers_cache, all_data = run_historical_training()

        elif missing:
            log.info(f"Some models missing for: {missing} — training missing TFs only.")
            log.info("Already-trained TFs will be loaded from disk (per-TF resume).")
            log.info("(Set FORCE_RETRAIN=true in .env to retrain everything from scratch)")
            models_cache, scalers_cache, all_data = run_historical_training()

        else:
            log.info("All model files found on disk — loading without retraining.")
            log.info("Tip: set FORCE_RETRAIN=true after changing features.")
            models_cache, scalers_cache, all_data = load_trained_system()

        if not models_cache:
            log.error("No models available. Check data sources and model files.")
            return

        run_live_loop(models_cache, scalers_cache, all_data)

    finally:
        mt5.shutdown()
        log.info("MT5 disconnected. System stopped.")


if __name__ == "__main__":
    run_full_system()
