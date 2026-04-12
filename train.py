"""
train.py — Training entry point
================================
Run this to train or retrain the ML system.
Never needs to run during live trading.

Usage:
  python train.py              # trains missing TFs only, skips existing models
  python train.py --force      # full retrain from scratch (all TFs)
  python train.py --report     # train then immediately generate HTML report

What it does:
  1. Loads tick Parquet data + applies institutional features (~2-15 min)
  2. Trains XGBoost + Random Forest per entry TF (skips TFs already on disk)
  3. Runs Genetic Algorithm + Optuna global optimisation
  4. Runs per-TF optimisation: top-5 param sets per TF
  5. Runs full backtest on all strategies, saves results to SQLite
  6. Saves all models to models/  and params to params/
  7. Optionally generates HTML report

After this finishes, start live.py in a separate terminal.
"""

import os, sys, json, logging, argparse, warnings

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")

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
from sklearn.isotonic import IsotonicRegression
from dotenv import load_dotenv
load_dotenv()

# ── Parse CLI args before importing heavy modules ──────────────────────
parser = argparse.ArgumentParser(description="ML Trading System — Training")
parser.add_argument("--force",  action="store_true",
                    help="Force full retrain (ignore saved models)")
parser.add_argument("--report", action="store_true",
                    help="Generate HTML report after training")
parser.add_argument("--symbol", type=str, default=None,
                    help="Train specific symbol only (e.g. US30)")
args = parser.parse_args()

# CLI --force overrides .env FORCE_RETRAIN
FORCE_RETRAIN = args.force or (os.getenv("FORCE_RETRAIN", "false").lower() == "true")

# ── Imports (after env is loaded) ──────────────────────────────────────
from pipeline import (
    load_symbol_data, models_exist, load_models_from_disk,
    TICK_DATA_SYMBOLS, SESSION_BARS,
)
from phase2_adaptive_engine import (
    get_feature_cols, fit_scaler, apply_scaler,
    train_ensemble,
    run_genetic_algo, run_optuna, run_per_tf_optimization,
    save_params, load_params,
    _select_label_col,
    INSTRUMENTS, PARAM_SEEDS, HARD_LIMITS,
    MODEL_DIR, LOG_DIR, PARAMS_DIR, SEQ_LEN, MIN_BARS,
    OPTUNA_TRIALS,
)
from db import (
    init_db, upsert_strategy, save_equity_curve, save_monthly_pnl,
    save_optuna_trials, save_backtest_trades,
    set_strategy_active, get_all_strategies,
)
from backtest_engine import backtest_all_strategies, run_monte_carlo, run_sensitivity, haircut_sharpe

# ── Config ──────────────────────────────────────────────────────────────
RISK_MODE        = os.getenv("RISK_MODE",         "percent")
RISK_PCT         = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
FIXED_RISK_AMT   = float(os.getenv("FIXED_RISK_AMOUNT",  "100.0"))
PER_TF_TRIALS    = int(os.getenv("PER_TF_TRIALS",        "500"))
TOP_N_STRATEGIES = int(os.getenv("TOP_N_STRATEGIES",     "5"))
ER_MULTIPLIER    = float(os.getenv("ER_MULTIPLIER",      "1.25"))

init_db()

# ── Logging ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"training_{datetime.now():%Y%m%d_%H%M}.log",
            encoding="utf-8"),
        logging.StreamHandler(
            open(1, mode="w", encoding="utf-8", errors="replace", closefd=False)),
    ],
)
log = logging.getLogger("train")

PURGE_BARS = 10   # bars to drop at each train/OOS boundary to prevent leakage


def _spa_bootstrap_test(pnls: list, n_boot: int = 2000) -> float:
    """
    Bootstrap test for H0: E[trade_pnl] <= 0 (no positive edge).

    Procedure:
      1. Center the trade P&L series under H0 (subtract observed mean).
      2. Resample N bootstrap draws from the centered series.
      3. p-value = fraction of bootstrap means >= observed mean.

    Returns p-value in [0, 1].  p < 0.05 → reject H0, significant edge.
    Returns 1.0 when there is insufficient data (<20 trades) or no edge.
    """
    arr = np.array(pnls, dtype=float)
    if len(arr) < 20:
        return 1.0
    actual_mean = float(arr.mean())
    if actual_mean <= 0:
        return 1.0
    # Center data so bootstrap distribution represents H0: mean = 0
    arr_c = arr - actual_mean
    rng = np.random.default_rng(42)
    boot_means = rng.choice(arr_c, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float((boot_means >= actual_mean).mean())


def _walk_forward_folds(df: pd.DataFrame):
    """
    Build 3 anchored expanding walk-forward folds.
    Returns list of (train_df, val_df, oos_label) tuples.

    Fold logic (based on calendar year of the data):
      Fold 1: train up to end of year[-3], OOS = year[-2]
      Fold 2: train up to end of year[-2], OOS = year[-1]
      Fold 3: train up to end of year[-1], OOS = current year (if data exists)

    If data spans < 3 years, falls back to single 70/30 split.
    PURGE_BARS rows are dropped at each train/OOS boundary.
    """
    if df.empty:
        return []

    years = sorted(df.index.year.unique())
    if len(years) < 3:
        # Fallback: single 70/30 split
        n     = len(df)
        i_cut = int(n * 0.70)
        train = df.iloc[:i_cut - PURGE_BARS]
        val   = df.iloc[i_cut:]
        if len(train) < 200 or len(val) < 50:
            return []
        return [(train, val, "single_fold")]

    folds = []
    for fold_idx in range(1, min(4, len(years))):
        # OOS year is years[-fold_idx]
        # Train is everything before OOS year minus purge
        oos_year  = years[-fold_idx]
        oos_mask  = df.index.year == oos_year
        pre_mask  = df.index.year < oos_year

        train_df  = df[pre_mask]
        oos_df    = df[oos_mask]

        if len(train_df) < 200 or len(oos_df) < 50:
            continue

        # Purge boundary: drop PURGE_BARS from end of train and start of OOS
        train_df  = train_df.iloc[:-PURGE_BARS] if len(train_df) > PURGE_BARS else train_df
        oos_df    = oos_df.iloc[PURGE_BARS:]    if len(oos_df)   > PURGE_BARS else oos_df

        if len(train_df) < 200 or len(oos_df) < 50:
            continue

        folds.append((train_df, oos_df, str(oos_year)))

    # Folds are in reverse order (newest OOS first) — reverse so earliest is fold 1
    folds.reverse()

    if not folds:
        # Fallback
        n     = len(df)
        i_cut = int(n * 0.70)
        train = df.iloc[:i_cut - PURGE_BARS]
        val   = df.iloc[i_cut:]
        return [(train, val, "single_fold")] if len(train) > 200 and len(val) > 50 else []

    return folds


# ─────────────────────────────────────────────────────────────────────
# Core training pipeline
# ─────────────────────────────────────────────────────────────────────

def run_training(symbols=None) -> tuple[dict, dict, dict]:
    """
    Full training pipeline.
    Returns (models_cache, scalers_cache, all_data) ready for live.py
    to consume, but train.py saves everything to disk — live.py loads
    independently.
    """
    log.info("=" * 60)
    log.info("UNIFIED HISTORICAL TRAINING")
    if FORCE_RETRAIN:
        log.info("Mode: FULL RETRAIN (--force or FORCE_RETRAIN=true)")
    else:
        log.info("Mode: RESUME (skip TFs already on disk)")
    log.info("=" * 60)

    target_symbols = symbols or list(INSTRUMENTS.keys())
    models_cache   = {}
    scalers_cache  = {}
    all_data       = {}

    for symbol in target_symbols:
        if symbol not in INSTRUMENTS:
            log.warning(f"Symbol {symbol} not in ACTIVE_SYMBOLS — skipping")
            continue

        log.info(f"\n{'─'*50}")
        log.info(f"Symbol: {symbol}")

        # ── Load data ──────────────────────────────────────────────────
        # save_featured=True writes full-featured parquet for report.py
        tf_feat = load_symbol_data(symbol, save_featured=True)
        if not tf_feat:
            log.warning(f"No data loaded for {symbol} — skipping")
            continue
        all_data[symbol] = tf_feat

        # ── Train / resume per entry TF ────────────────────────────────
        for tf in PARAM_SEEDS["entry_tf_options"]:
            if tf not in tf_feat:
                continue

            key = f"{symbol}_{tf}m"

            if not FORCE_RETRAIN:
                xgb_p    = MODEL_DIR / f"xgb_{key}.pkl"
                rf_p     = MODEL_DIR / f"rf_{key}.pkl"
                scaler_p = MODEL_DIR / f"scaler_{key}.pkl"
                if xgb_p.exists() and rf_p.exists():
                    log.info(f"  {symbol} {tf}m: found on disk — loading (resume mode)")
                    models_cache[f"xgb_{key}"]  = joblib.load(xgb_p)
                    models_cache[f"rf_{key}"]   = joblib.load(rf_p)
                    if scaler_p.exists():
                        scalers_cache[key] = joblib.load(scaler_p)
                    continue

            df        = tf_feat[tf]
            n         = len(df)
            feat_cols = get_feature_cols(df)

            log.info(f"  {symbol} {tf}m | {df.index[0].date()} → {df.index[-1].date()} | "
                     f"{n:,} bars | {len(feat_cols)} features")

            # ── Walk-forward 3-fold validation ────────────────────────
            folds = _walk_forward_folds(df)
            if not folds:
                log.warning(f"  {symbol} {tf}m: insufficient data for walk-forward — skipping")
                continue

            fold_xgb_scores = []
            fold_rf_scores  = []
            oos_probs_all   = []   # stacked OOS ensemble probs for calibration
            oos_labels_all  = []
            for fold_num, (train_df, val_df, oos_label) in enumerate(folds, 1):
                log.info(f"    Fold {fold_num}/{len(folds)} | OOS={oos_label} | "
                         f"train={len(train_df):,} val={len(val_df):,}")

                # NaN audit on first fold
                if fold_num == 1:
                    nan_pct  = train_df[feat_cols].isna().mean()
                    bad_cols = nan_pct[nan_pct > 0.10]
                    if not bad_cols.empty:
                        log.warning(f"  {symbol} {tf}m: {len(bad_cols)} features >10% NaN")

                # Target balance check
                if "target" in train_df.columns:
                    pos = train_df["target"].mean()
                    if pos < 0.20 or pos > 0.80:
                        log.warning(f"  {symbol} {tf}m fold {fold_num}: "
                                    f"imbalanced target {pos:.0%}")

                scaler  = fit_scaler(train_df, feat_cols)
                train_s = apply_scaler(train_df, scaler, feat_cols)
                val_s   = apply_scaler(val_df,   scaler, feat_cols)

                xgb_fold, rf_fold = train_ensemble(train_s, val_s, feat_cols, symbol, tf)
                X_val = val_s[feat_cols].values
                y_val = val_df["target"].values
                fold_xgb_scores.append((xgb_fold.predict(X_val) == y_val).mean())
                fold_rf_scores.append( (rf_fold.predict(X_val)  == y_val).mean())

                # Collect OOS ensemble probs for isotonic calibration
                p_xgb_oos = xgb_fold.predict_proba(X_val)[:, 1]
                p_rf_oos  = rf_fold.predict_proba(X_val)[:, 1]
                oos_probs_all.extend(((p_xgb_oos + p_rf_oos) / 2.0).tolist())
                oos_labels_all.extend(y_val.tolist())

            # Log cross-fold OOS accuracy summary
            log.info(f"  {symbol} {tf}m walk-forward OOS accuracy | "
                     f"XGB: {np.mean(fold_xgb_scores):.3f} ± {np.std(fold_xgb_scores):.3f} | "
                     f"RF: {np.mean(fold_rf_scores):.3f} ± {np.std(fold_rf_scores):.3f}")

            # Fit isotonic calibrator on all stacked OOS predictions
            # Makes confidence threshold meaningful: calibrated prob ≈ actual win rate
            if len(oos_probs_all) >= 50:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(oos_probs_all, oos_labels_all)
                joblib.dump(calibrator, MODEL_DIR / f"calibrator_{key}.pkl")
                log.info(f"  Calibrator saved → calibrator_{key}.pkl "
                         f"({len(oos_probs_all):,} OOS samples)")

            # ── Final production model: train on ALL data ─────────────
            # Use the last fold's scaler (fitted on most data) for production
            log.info(f"  {symbol} {tf}m: fitting final production model on all data")
            final_scaler   = fit_scaler(df, feat_cols)
            # Val for early stopping: last 15% of full dataset
            i_va           = int(n * 0.85)
            full_train_s   = apply_scaler(df.iloc[:i_va], final_scaler, feat_cols)
            full_val_s     = apply_scaler(df.iloc[i_va:], final_scaler, feat_cols)
            xgb_m, rf_m    = train_ensemble(full_train_s, full_val_s, feat_cols, symbol, tf)

            scalers_cache[key] = final_scaler
            models_cache[f"xgb_{key}"] = xgb_m
            models_cache[f"rf_{key}"]  = rf_m
            joblib.dump(final_scaler, MODEL_DIR / f"scaler_{key}.pkl")
            log.info(f"  Scaler saved → scaler_{key}.pkl")

            # ── Drift reference: feature histograms for top-10 features ──
            # Saved as drift_ref_{key}.pkl — loaded by live.py for PSI monitoring.
            try:
                importances = xgb_m.feature_importances_
                top10_idx   = np.argsort(importances)[::-1][:10]
                top10_cols  = [feat_cols[i] for i in top10_idx]
                ref_data    = apply_scaler(df, final_scaler, feat_cols)
                drift_ref   = {}
                for col in top10_cols:
                    vals = ref_data[col].dropna().values
                    counts, bin_edges = np.histogram(vals, bins=20)
                    pct = counts / (counts.sum() + 1e-10)
                    drift_ref[col] = {"bin_edges": bin_edges, "pct": pct}
                joblib.dump(drift_ref, MODEL_DIR / f"drift_ref_{key}.pkl")
                log.info(f"  Drift reference saved → drift_ref_{key}.pkl "
                         f"(top {len(top10_cols)} features)")
            except Exception as e:
                log.warning(f"  Could not save drift reference for {key}: {e}")


        # ── Global GA + Optuna (picks best TF across all) ─────────────
        has_any_scaler = any(
            scalers_cache.get(f"{symbol}_{tf}m")
            for tf in PARAM_SEEDS["entry_tf_options"] if tf in tf_feat
        )
        if has_any_scaler:
            log.info(f"  Running GA parameter search: {symbol}")
            ga_params = run_genetic_algo(tf_feat, models_cache, scalers_cache, symbol=symbol)

            log.info(f"  Running Optuna parameter search: {symbol}")
            opt_params = run_optuna(tf_feat, models_cache, scalers_cache,
                                    n_trials=OPTUNA_TRIALS, symbol=symbol)

            best = (opt_params if opt_params.get("best_score", -999) >
                    ga_params.get("best_score", -999) else ga_params)
            best["source"]      = "historical_optimisation"
            best["data_source"] = "tick+institutional" if symbol in TICK_DATA_SYMBOLS else "mt5+estimated"
            save_params(symbol, best)

            log.info(f"\n  {symbol} OPTIMAL PARAMS:")
            log.info(f"    Entry TF   : {best['entry_tf']}m")
            log.info(f"    HTF        : {best['htf_tf']}m")
            log.info(f"    SL ATR×    : {best['sl_atr']:.3f}")
            log.info(f"    TP mult    : {best['tp_mult']:.2f}x")
            log.info(f"    Confidence : {best['confidence']:.2f}")
            be = best.get("be_r", 0)
            log.info(f"    Break-even : {'OFF' if be==0 else f'+{be}R → entry+1pt'}")

        # ── Per-TF optimisation (top-5 per TF) ────────────────────────
        log.info(f"\n  Per-TF optimisation: {symbol} (top-{TOP_N_STRATEGIES} per TF)")
        for tf in PARAM_SEEDS["entry_tf_options"]:
            if tf not in tf_feat or not scalers_cache.get(f"{symbol}_{tf}m"):
                continue

            result = run_per_tf_optimization(
                df_dict=tf_feat, models_by_tf=models_cache,
                scalers_by_tf=scalers_cache, symbol=symbol,
                locked_tf=tf, n_trials=PER_TF_TRIALS, top_n=TOP_N_STRATEGIES,
            )
            top_trials, all_trials = result

            save_optuna_trials(symbol, tf, all_trials)

            if not top_trials:
                continue

            bt_results = backtest_all_strategies(
                symbol=symbol, tf=tf,
                top_params=[t["params"] for t in top_trials],
                risk_mode=RISK_MODE, risk_pct=RISK_PCT,
                fixed_amt=FIXED_RISK_AMT, start_balance=10_000.0,
            )

            # Load models once for sensitivity (same models used in backtest)
            from backtest_engine import load_featured_df
            import joblib as _joblib
            _df_sens  = load_featured_df(symbol, tf)
            _key_sens = f"{symbol}_{tf}m"
            try:
                _xgb_sens    = _joblib.load(MODEL_DIR / f"xgb_{_key_sens}.pkl")
                _rf_sens     = _joblib.load(MODEL_DIR / f"rf_{_key_sens}.pkl")
                _scaler_sens = _joblib.load(MODEL_DIR / f"scaler_{_key_sens}.pkl")
                _sens_models_ok = not _df_sens.empty
            except Exception:
                _sens_models_ok = False

            tf_summary_rows = []   # collect per-strategy rows for the post-TF table

            for rank, (trial, bt) in enumerate(zip(top_trials, bt_results), 1):
                stats = bt.get("stats")
                if not stats:
                    log.warning(f"  {symbol} {tf}m rank{rank}: backtest returned no stats — skipping")
                    continue
                strategy_id = f"{symbol}_{tf}m_rank{rank}"
                n_trades    = stats["n_trades"]

                # ── Sanity check raw backtest before running robustness ────────
                if n_trades == 0:
                    log.warning(f"  {strategy_id}: 0 trades — confidence threshold too high or no signal")
                if stats["win_rate"] >= 99.0:
                    log.warning(f"  {strategy_id}: WR={stats['win_rate']:.1f}% — likely overfitting (train data leak)")
                if stats["efficiency_ratio"] >= 500:
                    log.warning(f"  {strategy_id}: ER={stats['efficiency_ratio']:.1f} — unrealistic, check data split")
                if stats["max_dd_pct"] < 0.05:
                    log.warning(f"  {strategy_id}: MaxDD={stats['max_dd_pct']:.3f}% near zero — Calmar will be extreme")

                # ── Haircut Sharpe (deflated for n_trials multiple testing) ──
                hc_sharpe = None
                if bt.get("trades"):
                    hc_sharpe = haircut_sharpe(bt["trades"], n_trials=PER_TF_TRIALS)
                    if hc_sharpe is not None and hc_sharpe < 0:
                        log.warning(f"  {strategy_id}: Haircut Sharpe={hc_sharpe:.2f} < 0 "
                                    f"— likely selection bias, raw Sharpe={stats['sharpe']:.2f}")

                # ── Monte Carlo robustness ─────────────────────────────────────
                mc = {}
                if bt.get("trades"):
                    mc = run_monte_carlo(bt["trades"], n_sims=1000)
                    mc_tag = "PASS" if mc.get("mc_pass") else "FAIL"
                    spread = mc.get("mc_sharpe_p95", 0) - mc.get("mc_sharpe_p5", 0)
                    if spread > 5.0:
                        log.warning(f"  {strategy_id}: MC Sharpe spread={spread:.1f} "
                                    f"(p5={mc['mc_sharpe_p5']:.2f}→p95={mc['mc_sharpe_p95']:.2f}) "
                                    f"— high luck sensitivity")
                    if mc.get("mc_dd_p95", 0) > 50.0:
                        log.warning(f"  {strategy_id}: MC worst-case DD={mc['mc_dd_p95']:.1f}% "
                                    f"— catastrophic tail risk in some orderings")

                # ── SPA bootstrap edge test ────────────────────────────────────
                spa_p = None
                if bt.get("trades"):
                    trade_pnls = [t.get("pnl", 0) for t in bt["trades"]]
                    spa_p = _spa_bootstrap_test(trade_pnls)
                    spa_tag = f"p={spa_p:.3f}"
                    if spa_p < 0.05:
                        log.info(f"  {strategy_id}: SPA edge CONFIRMED ({spa_tag}) — rejects H0 at 5%")
                    elif spa_p < 0.10:
                        log.warning(f"  {strategy_id}: SPA edge MARGINAL ({spa_tag}) — borderline signal")
                    else:
                        log.warning(f"  {strategy_id}: SPA edge WEAK ({spa_tag}) — cannot reject H0, "
                                    f"strategy may have no real edge")

                # ── Parameter sensitivity ──────────────────────────────────────
                sens = {}
                if _sens_models_ok:
                    sens = run_sensitivity(
                        _df_sens, trial["params"], _scaler_sens,
                        _xgb_sens, _rf_sens,
                        risk_mode=RISK_MODE, risk_pct=RISK_PCT,
                        fixed_amt=FIXED_RISK_AMT, start_balance=10_000.0,
                    )
                    if sens.get("sensitivity_score", 100) < 20:
                        log.warning(f"  {strategy_id}: Sensitivity score={sens['sensitivity_score']:.0f}/100 "
                                    f"— cliff-edge params, collapses on small nudge")
                    # Log each nudge result so you can see exactly which param is fragile
                    for v in sens.get("variants", []):
                        sign  = "+" if v["delta"] > 0 else ""
                        delta_str = f"{sign}{v['delta']}"
                        log.info(f"    sensitivity {v['param']}{delta_str}: "
                                 f"Sharpe={v['sharpe']:.2f} ER={v['er']:.2f}")

                upsert_strategy({
                    "strategy_id":      strategy_id,
                    "symbol":           symbol,
                    "tf":               tf,
                    "rank":             rank,
                    "entry_tf":         trial["params"]["entry_tf"],
                    "htf_tf":           trial["params"]["htf_tf"],
                    "sl_atr":           trial["params"]["sl_atr"],
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
                    "sortino":          stats.get("sortino"),
                    "calmar":           stats.get("calmar"),
                    "haircut_sharpe":   hc_sharpe,
                    "mc_sharpe_p5":     mc.get("mc_sharpe_p5"),
                    "mc_sharpe_p50":    mc.get("mc_sharpe_p50"),
                    "mc_sharpe_p95":    mc.get("mc_sharpe_p95"),
                    "mc_dd_p95":        mc.get("mc_dd_p95"),
                    "mc_profit_p5":     mc.get("mc_profit_p5"),
                    "mc_pass":          int(mc.get("mc_pass", False)),
                    "sensitivity_score": sens.get("sensitivity_score"),
                    "robust":           int(sens.get("robust", False)),
                    "spa_p_value":      spa_p,
                    "is_active":        0,
                })
                if not bt["equity_df"].empty:
                    save_equity_curve(strategy_id, bt["equity_df"])
                if bt["monthly_pnl"]:
                    save_monthly_pnl(strategy_id, bt["monthly_pnl"])
                if bt.get("trades"):
                    save_backtest_trades(strategy_id, bt["trades"])

                tf_summary_rows.append({
                    "id":        strategy_id,
                    "trades":    n_trades,
                    "wr":        stats["win_rate"],
                    "sharpe":    stats["sharpe"],
                    "sortino":   stats.get("sortino", 0),
                    "calmar":    stats.get("calmar", 0),
                    "hc":        hc_sharpe or 0,
                    "er":        stats["efficiency_ratio"],
                    "dd":        stats["max_dd_pct"],
                    "mc":        "PASS" if mc.get("mc_pass") else "FAIL",
                    "mc_p5":     mc.get("mc_sharpe_p5", 0),
                    "mc_dd95":   mc.get("mc_dd_p95", 0),
                    "sens":      sens.get("sensitivity_score", 0),
                    "robust":    "Y" if sens.get("robust") else "N",
                    "spa":       f"{spa_p:.3f}" if spa_p is not None else "n/a",
                })

            # ── Retrain rank-1 final model on its best-matching label ─────────
            # The walk-forward folds and optimizer used the default be=0 label.
            # Now that we know the rank-1 params (sl_atr, tp_mult, be_r), retrain
            # the production model on the exact label column those params match.
            # This is the model that goes live — it predicts the outcome it will
            # actually trade (including BE semantics), fixing model miscalibration.
            if top_trials and tf_feat.get(tf) is not None:
                rank1_params   = top_trials[0]["params"]
                best_label_col = _select_label_col(
                    rank1_params["sl_atr"],
                    rank1_params["tp_mult"],
                    rank1_params.get("be_r", 0),
                )
                df_full       = tf_feat[tf]
                # feat_cols may not be set if we resumed from disk (skipped walk-forward)
                rt_feat_cols  = get_feature_cols(df_full)
                scaler_key    = f"{symbol}_{tf}m"
                retrain_scaler = scalers_cache.get(scaler_key)

                if best_label_col in df_full.columns and best_label_col != "target_sl1.5_tp2_be0":
                    log.info(f"  {symbol} {tf}m: retraining final model on label "
                             f"'{best_label_col}' (rank-1 params: "
                             f"sl={rank1_params['sl_atr']} tp={rank1_params['tp_mult']} "
                             f"be={rank1_params.get('be_r',0)})")
                    # Copy and swap target so train_ensemble trains on the correct label
                    df_rt = df_full.copy()
                    df_rt["target"] = df_rt[best_label_col]
                    df_rt.dropna(subset=["target"], inplace=True)

                    if retrain_scaler is not None and len(df_rt) > MIN_BARS:
                        i_va        = int(len(df_rt) * 0.85)
                        rt_train_s  = apply_scaler(df_rt.iloc[:i_va], retrain_scaler, rt_feat_cols)
                        rt_val_s    = apply_scaler(df_rt.iloc[i_va:], retrain_scaler, rt_feat_cols)
                        xgb_rt, rf_rt = train_ensemble(rt_train_s, rt_val_s, rt_feat_cols, symbol, tf)
                        # Overwrite the generic models in cache and on disk
                        models_cache[f"xgb_{scaler_key}"] = xgb_rt
                        models_cache[f"rf_{scaler_key}"]  = rf_rt
                        log.info(f"  {symbol} {tf}m: label-matched model saved "
                                 f"(overwrites generic be=0 model)")
                    else:
                        log.warning(f"  {symbol} {tf}m: skipping label retrain "
                                    f"— scaler missing or insufficient bars "
                                    f"({len(df_rt)} bars)")
                else:
                    log.info(f"  {symbol} {tf}m: label retrain not needed "
                             f"— rank-1 maps to default label (be=0, mid-grid)")

            # ── Post-TF summary table ──────────────────────────────────────────
            if tf_summary_rows:
                log.info(f"\n  ┌─ {symbol} {tf}m — Strategy Summary ({'─'*42}┐")
                log.info(f"  │ {'Strategy':<22} {'N':>5} {'WR%':>5} {'Sharpe':>7} "
                         f"{'Sortino':>7} {'Calmar':>6} {'HC_SR':>6} "
                         f"{'ER':>8} {'DD%':>5} {'MC':>4} {'MCp5':>6} {'MCdd95':>6} "
                         f"{'Sens':>5} {'Rob':>3} {'SPA-p':>6} │")
                log.info(f"  │ {'─'*22} {'─'*5} {'─'*5} {'─'*7} "
                         f"{'─'*7} {'─'*6} {'─'*6} "
                         f"{'─'*8} {'─'*5} {'─'*4} {'─'*6} {'─'*6} "
                         f"{'─'*5} {'─'*3} {'─'*6} │")
                for r in tf_summary_rows:
                    log.info(
                        f"  │ {r['id']:<22} {r['trades']:>5} {r['wr']:>5.1f} "
                        f"{r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['calmar']:>6.2f} "
                        f"{r['hc']:>6.2f} {r['er']:>8.2f} {r['dd']:>5.1f} "
                        f"{r['mc']:>4} {r['mc_p5']:>6.2f} {r['mc_dd95']:>6.1f} "
                        f"{r['sens']:>5.0f} {r['robust']:>3} {r['spa']:>6} │"
                    )
                log.info(f"  └{'─'*104}┘")

        _activate_best_strategy(symbol)

    # ── Final cross-symbol summary ─────────────────────────────────────
    _log_final_summary(target_symbols)

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("Start live trading: python live.py")
    log.info("Generate report  : python report.py")
    log.info("=" * 60)
    return models_cache, scalers_cache, all_data


def _log_final_summary(symbols: list):
    """
    Print a final consolidated table of ALL strategies across all symbols/TFs.
    Flags anything suspicious so you can spot bugs or over-penalisation at a glance.
    """
    all_strats = []
    for sym in symbols:
        all_strats.extend(get_all_strategies(sym))

    if not all_strats:
        return

    log.info("\n" + "=" * 60)
    log.info("FINAL STRATEGY AUDIT — check for red flags below")
    log.info("=" * 60)

    # Column widths
    log.info(
        f"  {'ID':<22} {'TF':>3} {'N':>5} {'WR%':>5} {'Sharpe':>7} "
        f"{'HC_SR':>6} {'ER':>8} {'DD%':>5} {'Calmar':>6} "
        f"{'MC':>4} {'MCp5':>6} {'Sens':>5} {'Rob':>3} {'Active':>6}"
    )
    log.info("  " + "─" * 102)

    flags = []
    for s in sorted(all_strats, key=lambda x: (x["symbol"], x["tf"], x["rank"])):
        sid    = s["strategy_id"]
        n      = s.get("n_trades") or 0
        wr     = s.get("win_rate") or 0
        sh     = s.get("sharpe") or 0
        hc     = s.get("haircut_sharpe") or 0
        er     = s.get("efficiency_ratio") or 0
        dd     = s.get("max_dd_pct") or 0
        cal    = s.get("calmar") or 0
        mc     = "PASS" if s.get("mc_pass") else "FAIL"
        mc_p5  = s.get("mc_sharpe_p5") or 0
        sens   = s.get("sensitivity_score") or 0
        rob    = "Y" if s.get("robust") else "N"
        active = "LIVE" if s.get("is_active") else "    "

        log.info(
            f"  {sid:<22} {s.get('tf',0):>3} {n:>5} {wr:>5.1f} "
            f"{sh:>7.2f} {hc:>6.2f} {er:>8.2f} {dd:>5.1f} {cal:>6.2f} "
            f"{mc:>4} {mc_p5:>6.2f} {sens:>5.0f} {rob:>3} {active:>6}"
        )

        # Flag suspicious patterns
        if n == 0:
            flags.append(f"  [NO TRADES]   {sid} — zero trades, params too restrictive")
        if wr >= 99.0:
            flags.append(f"  [OVERFIT WR]  {sid} — WR={wr:.1f}%, almost certainly train-data leak")
        if er >= 500:
            flags.append(f"  [OVERFIT ER]  {sid} — ER={er:.1f}, unrealistic (check 15% OOS split)")
        if hc < 0:
            flags.append(f"  [NEG HC_SR]   {sid} — Haircut Sharpe={hc:.2f}, selection bias likely")
        if mc == "FAIL" and n >= 50:
            flags.append(f"  [MC FAIL]     {sid} — not profitable in 5th percentile of MC simulations")
        if sens < 20 and n >= 50:
            flags.append(f"  [FRAGILE]     {sid} — Sensitivity={sens:.0f}/100, cliff-edge params")
        if dd < 0.05 and n >= 20:
            flags.append(f"  [ZERO DD]     {sid} — MaxDD={dd:.3f}%, Calmar unreliable (check data)")
        if sh > 50:
            flags.append(f"  [HUGE SHARPE] {sid} — Sharpe={sh:.1f}, check if OOS split applied")

    log.info("  " + "─" * 102)

    if flags:
        log.warning("\n  RED FLAGS — investigate before going live:")
        for f in flags:
            log.warning(f)
    else:
        log.info("\n  No red flags detected — results look realistic.")


def _activate_best_strategy(symbol: str):
    # Minimum trades per year per TF — scale by window length for statistical significance
    # At confidence=0.85 the system is very selective — signal rate ~0.05–0.1% of bars.
    # With one-trade-at-a-time blocking, realistic totals over 6 years:
    #   1m (2.1M bars) → ~800–2,000 trades   | floor at 600 (100/yr)
    #   3m (712k bars) → ~300–800 trades      | floor at 300 (50/yr)
    #   5m (427k bars) → ~200–500 trades      | floor at 180 (30/yr)
    #   10m (214k bars)→ ~100–300 trades      | floor at 120 (20/yr)
    #   15m (142k bars)→ ~80–200 trades       | floor at 90  (15/yr)
    #   30m (72k bars) → ~50–120 trades       | floor at 60  (10/yr)
    # Statistical note: ≥300 trades gives ±5% WR estimate at 95% confidence.
    # Lower floors for shorter-session instruments are a conscious trade-off.
    MIN_TRADES_PER_YEAR = {1: 100, 3: 50, 5: 30, 10: 20, 15: 15, 30: 10}
    BACKTEST_YEARS = 6.0  # ~6 years of tick data

    def min_trades_for(tf: int) -> int:
        return max(20, int(MIN_TRADES_PER_YEAR.get(tf, 20) * BACKTEST_YEARS))

    strategies = get_all_strategies(symbol)

    def _qualifies(s, strict=True):
        if s.get("efficiency_ratio") is None:
            return False
        if s.get("n_trades", 0) < min_trades_for(s.get("tf", 15)):
            return False
        if s.get("win_rate", 0) >= 99.0:      # 99%+ WR = overfitting
            return False
        if strict and s.get("efficiency_ratio", 0) >= 500:  # unrealistic
            return False
        return True

    # Robustness helpers
    def _dsr_pass(s) -> bool:
        """DSR (Haircut Sharpe) > 0 — corrects for multiple-testing bias."""
        hc = s.get("haircut_sharpe")
        return hc is not None and hc > 0.0

    def _sens_pass(s) -> bool:
        """Sensitivity score ≥ 50 — not a cliff-edge parameter set."""
        sc = s.get("sensitivity_score")
        return sc is not None and sc >= 50.0

    # Tier 1: DSR>0 + MC pass + sensitivity≥50 + realistic ER (full gates)
    tier1 = [s for s in strategies if _qualifies(s, strict=True)
             and _dsr_pass(s) and s.get("mc_pass", 0) == 1
             and _sens_pass(s) and s.get("robust", 0) == 1]
    # Tier 2: DSR>0 + MC pass (drop sensitivity)
    tier2 = [s for s in strategies if _qualifies(s, strict=True)
             and _dsr_pass(s) and s.get("mc_pass", 0) == 1]
    # Tier 3: DSR>0 only (drop MC and sensitivity requirement)
    tier3 = [s for s in strategies if _qualifies(s, strict=True)
             and _dsr_pass(s)]
    # Tier 4: ER-filter only (drop all robustness gates)
    tier5_er = [s for s in strategies if _qualifies(s, strict=True)]
    # Tier 5: last resort — drop ER cap too
    tier5 = [s for s in strategies if _qualifies(s, strict=False)]

    # Pick best available tier
    valid = tier1 or tier2 or tier3 or tier5_er or tier5

    if not valid:
        log.warning(f"  No qualifying strategies for {symbol} — nothing activated")
        return

    tier_used = ("DSR+MC+Sens" if valid is tier1 else
                 "DSR+MC"      if valid is tier2 else
                 "DSR-only"    if valid is tier3 else
                 "ER-filter"   if valid is tier5_er else "fallback")

    valid.sort(key=lambda s: s["efficiency_ratio"], reverse=True)
    best_id = valid[0]["strategy_id"]
    for s in strategies:
        set_strategy_active(s["strategy_id"], s["strategy_id"] == best_id)
    b = valid[0]
    log.info(f"  Auto-activated: {best_id} [{tier_used}] "
             f"ER={b['efficiency_ratio']:.2f} "
             f"WR={b.get('win_rate',0):.1f}% "
             f"DSR={b.get('haircut_sharpe') or 0:.2f} "
             f"MC={'PASS' if b.get('mc_pass') else 'FAIL'} "
             f"Sens={b.get('sensitivity_score') or 0:.0f} "
             f"Robust={'YES' if b.get('robust') else 'NO'} "
             f"trades={b.get('n_trades',0)}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("ML TRADING SYSTEM — TRAINING")
    log.info("=" * 60)

    try:
        symbol_filter = [args.symbol] if args.symbol else None
        run_training(symbols=symbol_filter)
    except KeyboardInterrupt:
        log.info("Training interrupted by user.")

    if args.report:
        log.info("Generating HTML report...")
        import report
        report.build_report()


if __name__ == "__main__":
    main()
