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

import os, time, json, logging, warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
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
    TradingEnv, run_genetic_algo, run_optuna,
    save_params, load_params,
    get_signal, RiskGate, place_order,
    log_trade, load_trade_log, should_retrain,
    incremental_update, seconds_to_next_candle_close,
    INSTRUMENTS, PARAM_SEEDS, HARD_LIMITS,
    MODEL_DIR, LOG_DIR, PARAMS_DIR, SEQ_LEN, MIN_BARS,
    OPTUNA_TRIALS, OPTUNA_LIVE_TRIALS,
)

log = logging.getLogger("integration")

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
    60: 7,      # 1H  → 7 bars
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

        # Load from Parquet and apply institutional features
        tf_data = {}
        for tf in tfs_needed:
            raw = load_ohlcv_parquet(symbol, tf)
            if raw.empty:
                continue
            # Parquet files from tick_pipeline already have
            # engineer_tick_features applied — just add institutional
            sb = SESSION_BARS.get(tf, 78)
            try:
                enhanced = add_institutional_features(
                    raw, session_bars=sb, verbose=False
                )
                if len(enhanced) >= MIN_BARS:
                    tf_data[tf] = enhanced
                    log.info(f"  {symbol} {tf}m: {len(enhanced):,} bars "
                             f"| {len(get_feature_cols(enhanced))} features "
                             f"[TICK+INSTITUTIONAL]")
            except Exception as e:
                log.warning(f"  Institutional features failed for {symbol} {tf}m: {e}")
                if len(raw) >= MIN_BARS:
                    tf_data[tf] = raw

        if tf_data:
            return tf_data
        log.warning(f"  No tick Parquet loaded for {symbol}, falling back to MT5")

    # ── MT5 data path ────────────────────────────────────────────
    return _load_from_mt5(symbol)


def _load_from_mt5(symbol: str) -> dict:
    """Fetch max history from MT5 and apply full feature pipeline."""
    raw_tfs = fetch_all_timeframes(symbol)
    tf_data = {}
    for tf, raw_df in raw_tfs.items():
        enhanced = engineer_full_features(
            raw_df, tf_minutes=tf, symbol=symbol, is_tick_derived=False
        )
        if len(enhanced) >= MIN_BARS:
            tf_data[tf] = enhanced
            log.info(f"  {symbol} {tf}m: {len(enhanced):,} bars "
                     f"| {len(get_feature_cols(enhanced))} features [MT5]")
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
        log.info(f"\n{'─'*50}")
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

            df   = tf_feat[tf]
            n    = len(df)
            i_tr = int(n * 0.70)
            i_va = int(n * 0.85)

            train_df = df.iloc[:i_tr]
            val_df   = df.iloc[i_tr:i_va]
            feat_cols = get_feature_cols(df)

            # Fit scaler on training data only
            scaler   = fit_scaler(train_df, feat_cols)
            train_s  = apply_scaler(train_df, scaler, feat_cols)
            val_s    = apply_scaler(val_df,   scaler, feat_cols)

            key = f"{symbol}_{tf}m"
            scalers_cache[key] = scaler

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
        default_tf  = PARAM_SEEDS["entry_tf_default"]
        default_key = f"{symbol}_{default_tf}m"
        scaler_def  = scalers_cache.get(default_key)
        feat_def    = tf_feat.get(default_tf)

        if scaler_def and feat_def is not None:
            feat_cols = get_feature_cols(feat_def)
            ga_models = {
                "xgb": models_cache.get(f"xgb_{default_key}"),
                "rf":  models_cache.get(f"rf_{default_key}"),
            }

            log.info(f"  Running GA parameter search: {symbol}")
            ga_params  = run_genetic_algo(
                tf_feat, ga_models, scaler_def, feat_cols
            )

            log.info(f"  Running Optuna parameter search: {symbol}")
            opt_params = run_optuna(
                tf_feat, ga_models, scaler_def, feat_cols,
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
            log.info(f"    Data source: {best['data_source']}")

    log.info("\nHistorical training complete.")
    return models_cache, scalers_cache, all_data


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
# 6. LIVE TRADING LOOP — upgraded version
# ─────────────────────────────────────────────────────────────────────

def run_live_loop(models_cache: dict, scalers_cache: dict,
                  all_data: dict):
    """
    Live loop with unified data refresh.
    Wakes on closed candle of each symbol's ML-chosen entry TF.
    Blends tick history + live MT5 bars for each symbol.
    """
    log.info("=" * 60)
    log.info("LIVE TRADING LOOP")
    log.info(f"Hard limits: {HARD_LIMITS}")
    log.info("=" * 60)

    risk_gate  = RiskGate()
    last_reset = datetime.now().date()
    live_cache = {}

    # Pre-populate live cache from trained data (deep history)
    for symbol, tf_dict in all_data.items():
        for tf, df in tf_dict.items():
            live_cache[f"{symbol}_{tf}m"] = df
    log.info(f"Pre-loaded {len(live_cache)} data frames into live cache")

    while True:
        try:
            now = datetime.now()

            # Midnight reset
            if now.date() > last_reset:
                risk_gate.reset_daily()
                last_reset = now.date()
                log.info("Daily counters reset")

            # Sleep until next candle close (fastest TF in use)
            min_sleep = float("inf")
            for symbol in INSTRUMENTS:
                params = load_params(symbol)
                sleep  = seconds_to_next_candle_close(params["entry_tf"])
                min_sleep = min(min_sleep, sleep)

            log.info(f"Sleeping {min_sleep:.0f}s to next candle close...")
            time.sleep(max(1.0, min_sleep))

            # Process each symbol
            for symbol in INSTRUMENTS:
                params   = load_params(symbol)
                entry_tf = params["entry_tf"]
                htf_tf   = params["htf_tf"]
                is_tick  = symbol in TICK_DATA_SYMBOLS

                # Refresh entry TF data
                entry_key = f"{symbol}_{entry_tf}m"
                existing  = live_cache.get(entry_key, pd.DataFrame())
                updated   = refresh_live_data(
                    symbol, entry_tf, existing, is_tick
                )
                live_cache[entry_key] = updated

                # Refresh HTF data if active
                if htf_tf > 0:
                    htf_key   = f"{symbol}_{htf_tf}m"
                    htf_exist = live_cache.get(htf_key, pd.DataFrame())
                    htf_upd   = refresh_live_data(
                        symbol, htf_tf, htf_exist, is_tick
                    )
                    live_cache[htf_key] = htf_upd

                # Generate signal using phase2's get_signal()
                # (now uses our patched add_htf_alignment_full)
                signal = get_signal(
                    symbol          = symbol,
                    params          = params,
                    models_cache    = models_cache,
                    scalers_cache   = scalers_cache,
                    live_data_cache = live_cache,
                )

                if signal["direction"] == 0:
                    log.debug(f"{symbol}: no signal — {signal.get('reason','')}")
                    continue

                # Risk gate
                allowed, reason = risk_gate.can_trade()
                if not allowed:
                    log.warning(f"Trade blocked: {reason}")
                    continue

                # Position size
                lot = risk_gate.position_size(
                    sl_pips=signal["sl_pips"], symbol=symbol
                )

                log.info(
                    f"SIGNAL ► {symbol} "
                    f"{'LONG' if signal['direction']==1 else 'SHORT'} | "
                    f"conf:{signal['confidence']:.3f} | "
                    f"TF:{signal['entry_tf']}m HTF:{signal['htf_used']}m | "
                    f"lot:{lot} | "
                    f"SL:{signal['sl_price']} TP:{signal['tp_price']} | "
                    f"data:{'tick' if is_tick else 'mt5'}"
                )

                result = place_order(
                    symbol    = symbol,
                    direction = signal["direction"],
                    lot       = lot,
                    sl        = signal["sl_price"],
                    tp        = signal["tp_price"],
                    comment   = f"ml_{signal['entry_tf']}m_{'tick' if is_tick else 'mt5'}",
                )

                if result["success"]:
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
        models_cache, scalers_cache, all_data = run_historical_training()

        if not models_cache:
            log.error("No models trained. Check data sources.")
            return

        run_live_loop(models_cache, scalers_cache, all_data)

    finally:
        mt5.shutdown()
        log.info("MT5 disconnected. System stopped.")


if __name__ == "__main__":
    run_full_system()
