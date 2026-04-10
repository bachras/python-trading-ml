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
  2. Trains LSTM + XGB + RF per entry TF (skips TFs already on disk)
  3. Trains PPO RL agent
  4. Runs Genetic Algorithm + Optuna global optimisation
  5. Runs per-TF optimisation: top-5 param sets per TF (new feature)
  6. Runs full backtest on all strategies, saves results to SQLite
  7. Saves all models to models/  and params to params/
  8. Optionally generates HTML report

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
from dotenv import load_dotenv
from tensorflow.keras.models import load_model as keras_load_model

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
    train_lstm, train_ensemble, train_rl_agent,
    run_genetic_algo, run_optuna, run_per_tf_optimization,
    save_params, load_params,
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
                lstm_p   = MODEL_DIR / f"lstm_{key}.keras"
                xgb_p    = MODEL_DIR / f"xgb_{key}.pkl"
                rf_p     = MODEL_DIR / f"rf_{key}.pkl"
                scaler_p = MODEL_DIR / f"scaler_{key}.pkl"
                if lstm_p.exists() and xgb_p.exists() and rf_p.exists():
                    log.info(f"  {symbol} {tf}m: found on disk — loading (resume mode)")
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

            train_df  = df.iloc[:i_tr]
            val_df    = df.iloc[i_tr:i_va]
            feat_cols = get_feature_cols(df)

            log.info(f"  {symbol} {tf}m | {df.index[0].date()} → {df.index[-1].date()} | "
                     f"{n:,} bars | train={len(train_df):,} val={len(val_df):,} | "
                     f"{len(feat_cols)} features")

            # NaN audit
            nan_pct  = train_df[feat_cols].isna().mean()
            bad_cols = nan_pct[nan_pct > 0.10]
            if not bad_cols.empty:
                log.warning(f"  {symbol} {tf}m: {len(bad_cols)} features >10% NaN: "
                            + ", ".join(f"{c}={v:.0%}" for c, v in bad_cols.items()))

            # Target balance
            if "target" in train_df.columns:
                pos = train_df["target"].mean()
                if pos < 0.20 or pos > 0.80:
                    log.warning(f"  {symbol} {tf}m: imbalanced target {pos:.0%}")
                else:
                    log.info(f"  {symbol} {tf}m: target balance {pos:.0%} positive")

            # Fit scaler
            scaler  = fit_scaler(train_df, feat_cols)
            train_s = apply_scaler(train_df, scaler, feat_cols)
            val_s   = apply_scaler(val_df,   scaler, feat_cols)
            scalers_cache[key] = scaler
            joblib.dump(scaler, MODEL_DIR / f"scaler_{key}.pkl")
            log.info(f"  Scaler saved → scaler_{key}.pkl")

            # LSTM
            lstm = train_lstm(train_s, val_s, feat_cols, symbol, tf)
            if lstm:
                models_cache[f"lstm_{key}"] = lstm

            # XGBoost + Random Forest
            xgb_m, rf_m = train_ensemble(train_s, val_s, feat_cols, symbol, tf)
            models_cache[f"xgb_{key}"] = xgb_m
            models_cache[f"rf_{key}"]  = rf_m


        # ── PPO RL agent ───────────────────────────────────────────────
        default_tf  = PARAM_SEEDS["entry_tf_default"]
        default_key = f"{symbol}_{default_tf}m"
        if default_tf in tf_feat and scalers_cache.get(default_key):
            df_rl = apply_scaler(
                tf_feat[default_tf],
                scalers_cache[default_key],
                get_feature_cols(tf_feat[default_tf]),
            )
            rl = train_rl_agent(
                df_rl, get_feature_cols(df_rl), symbol,
                sl_mult=PARAM_SEEDS["sl_atr_seed"],
                rr=PARAM_SEEDS["rr_seed"],
            )
            models_cache[f"ppo_{symbol}"] = rl

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
            log.info(f"    R:R        : 1:{best['rr']:.2f}")
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
                })

            # ── Post-TF summary table ──────────────────────────────────────────
            if tf_summary_rows:
                log.info(f"\n  ┌─ {symbol} {tf}m — Strategy Summary ({'─'*38}┐")
                log.info(f"  │ {'Strategy':<22} {'N':>5} {'WR%':>5} {'Sharpe':>7} "
                         f"{'Sortino':>7} {'Calmar':>6} {'HC_SR':>6} "
                         f"{'ER':>8} {'DD%':>5} {'MC':>4} {'MCp5':>6} {'MCdd95':>6} "
                         f"{'Sens':>5} {'Rob':>3} │")
                log.info(f"  │ {'─'*22} {'─'*5} {'─'*5} {'─'*7} "
                         f"{'─'*7} {'─'*6} {'─'*6} "
                         f"{'─'*8} {'─'*5} {'─'*4} {'─'*6} {'─'*6} "
                         f"{'─'*5} {'─'*3} │")
                for r in tf_summary_rows:
                    log.info(
                        f"  │ {r['id']:<22} {r['trades']:>5} {r['wr']:>5.1f} "
                        f"{r['sharpe']:>7.2f} {r['sortino']:>7.2f} {r['calmar']:>6.2f} "
                        f"{r['hc']:>6.2f} {r['er']:>8.2f} {r['dd']:>5.1f} "
                        f"{r['mc']:>4} {r['mc_p5']:>6.2f} {r['mc_dd95']:>6.1f} "
                        f"{r['sens']:>5.0f} {r['robust']:>3} │"
                    )
                log.info(f"  └{'─'*100}┘")

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

    # Tier 1: passed both MC and sensitivity, realistic ER
    tier1 = [s for s in strategies if _qualifies(s, strict=True)
             and s.get("mc_pass", 0) == 1 and s.get("robust", 0) == 1]
    # Tier 2: passed MC only
    tier2 = [s for s in strategies if _qualifies(s, strict=True)
             and s.get("mc_pass", 0) == 1]
    # Tier 3: realistic ER, trade count OK (drop MC/sensitivity requirement)
    tier3 = [s for s in strategies if _qualifies(s, strict=True)]
    # Tier 4: last resort — drop ER cap
    tier4 = [s for s in strategies if _qualifies(s, strict=False)]

    valid = tier1 or tier2 or tier3 or tier4
    if not valid:
        log.warning(f"  No qualifying strategies for {symbol} — nothing activated")
        return

    tier_used = ("MC+Robust" if valid is tier1 else
                 "MC-pass"   if valid is tier2 else
                 "ER-filter" if valid is tier3 else "fallback")

    valid.sort(key=lambda s: s["efficiency_ratio"], reverse=True)
    best_id = valid[0]["strategy_id"]
    for s in strategies:
        set_strategy_active(s["strategy_id"], s["strategy_id"] == best_id)
    log.info(f"  Auto-activated: {best_id} [{tier_used}] "
             f"ER={valid[0]['efficiency_ratio']:.2f} "
             f"WR={valid[0].get('win_rate',0):.1f}% "
             f"MC={'PASS' if valid[0].get('mc_pass') else 'FAIL'} "
             f"Robust={'YES' if valid[0].get('robust') else 'NO'} "
             f"trades={valid[0].get('n_trades',0)}")


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
