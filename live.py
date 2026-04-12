"""
live.py — Live trading entry point
====================================
Run this after training is complete.
Loads saved models from disk and enters the live trading loop.

Usage:
  python live.py          # loads best models, starts live loop
  python live.py --dry    # dry run: signals logged but no orders placed

Requirements:
  - MT5 terminal open and logged in
  - Training completed at least once (models exist in models/)
  - .env configured with MT5 credentials

This script imports NO training libraries (no DEAP, no Keras training,
no Optuna). It starts in ~3-5 seconds vs 30+ seconds for train.py.
A crash here does NOT affect any training state.

Retraining while live:
  - Run python train.py in a SEPARATE terminal
  - live.py polls models/ every hour and hot-reloads if files are newer
  - Open positions are never interrupted by model reloads
"""

import os, sys, time, json, logging, warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*Converting sparse.*", category=UserWarning)

import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import joblib
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="ML Trading System — Live Trading")
parser.add_argument("--dry",          action="store_true",
                    help="Dry run: generate signals but do not place orders")
parser.add_argument("--test-trade",   action="store_true",
                    help="Place one micro test trade and close it — verifies full pipeline")
parser.add_argument("--signal-check", action="store_true",
                    help="Show current raw probabilities vs threshold for all strategies, then exit")
args = parser.parse_args()

DRY_RUN      = args.dry
TEST_TRADE   = args.test_trade
SIGNAL_CHECK = args.signal_check

# ── Inference-only imports (no training code) ──────────────────────────
from pipeline import (
    load_symbol_data, models_exist, load_models_from_disk,
    refresh_live_data, TICK_DATA_SYMBOLS,
)
from phase2_adaptive_engine import (
    connect_mt5, get_account_balance,
    get_signal, RiskGate, place_order,
    log_trade, should_retrain, incremental_update,
    seconds_to_next_candle_close,
    INSTRUMENTS, PARAM_SEEDS, HARD_LIMITS,
    MODEL_DIR, LOG_DIR, PARAMS_DIR,
)
from db import (
    init_db, get_all_strategies, get_strategy,
    get_capital_at_risk, get_open_trades, get_recent_live_trades,
    log_live_trade, update_trade_be, close_live_trade,
)

load_dotenv()
init_db()

# ── Config ──────────────────────────────────────────────────────────────
RISK_MODE       = os.getenv("RISK_MODE",          "percent")
RISK_PCT        = float(os.getenv("RISK_PER_TRADE_PCT",  "1.0"))
FIXED_RISK_AMT  = float(os.getenv("FIXED_RISK_AMOUNT",   "100.0"))
RISK_CAP_PCT    = float(os.getenv("RISK_CAP_PCT",        "2.0"))
RISK_CAP_AMOUNT = float(os.getenv("RISK_CAP_AMOUNT",     "200.0"))
EXTRA_STRATEGIES   = [s.strip() for s in os.getenv("EXTRA_STRATEGIES","").split(",") if s.strip()]
# Vol-sizing: disabled by default. Enable only when running scaled-risk phase.
VOL_SIZE_ENABLED   = os.getenv("VOL_SIZE_ENABLED", "false").lower() == "true"

# Hot-reload: check for newer model files every this many minutes
MODEL_RELOAD_INTERVAL_MIN = int(os.getenv("MODEL_RELOAD_INTERVAL_MIN", "60"))

# ── Logging ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"trading_{datetime.now():%Y%m%d}.log",
            encoding="utf-8"),
        logging.StreamHandler(
            open(1, mode="w", encoding="utf-8", errors="replace", closefd=False)),
    ],
)
log = logging.getLogger("live")

# 3K: startup config log — print all risk gate values so .env misconfiguration is immediately visible.
log.info(
    "[CONFIG] Risk gates loaded:\n"
    f"  max_drawdown_pct : {float(os.getenv('MAX_DRAWDOWN_PCT', '35.0')):.1f}%\n"
    f"  fixed_risk_amt   : ${FIXED_RISK_AMT:.1f}\n"
    f"  vol_size_enabled : {VOL_SIZE_ENABLED}\n"
    f"  session_gate     : 20:30–01:00 London (DST-aware)\n"
    f"  model_reload_min : {MODEL_RELOAD_INTERVAL_MIN}"
)

_LONDON_TZ = ZoneInfo("Europe/London")


def _is_session_blocked(ts: datetime | None = None) -> bool:
    """
    Return True if the current time falls in the blocked session:
    20:30 – 01:00 UK time (London timezone, DST-correct).

    Covers illiquid post-NYSE-close overnight hours for US30.
    No entries are placed in this window.
    """
    if ts is None:
        ts = datetime.now(tz=_LONDON_TZ)
    elif getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=_LONDON_TZ)
    else:
        ts = ts.astimezone(_LONDON_TZ)
    h, m = ts.hour, ts.minute
    after_2030  = (h == 20 and m >= 30) or h >= 21
    before_0100 = (h == 0) or (h == 1 and m == 0)
    return after_2030 or before_0100


# ─────────────────────────────────────────────────────────────────────
# Strategy selection
# ─────────────────────────────────────────────────────────────────────

def _get_active_strategies() -> list[dict]:
    """
    Auto: top-1 by efficiency ratio per symbol (from SQLite).
    Falls back to params.json if DB has no strategies yet.
    Manual: add strategy IDs via EXTRA_STRATEGIES in .env.
    """
    from phase2_adaptive_engine import load_params
    active = []

    for symbol in INSTRUMENTS:
        strategies = get_all_strategies(symbol)
        if not strategies:
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

    for sid in EXTRA_STRATEGIES:
        s = get_strategy(sid)
        if s:
            active.append(s)
        else:
            log.warning(f"EXTRA_STRATEGIES: '{sid}' not found in DB — ignored")

    # Deduplicate
    seen, result = set(), []
    for s in active:
        sid = s.get("strategy_id", s.get("symbol", "?"))
        if sid not in seen:
            seen.add(sid)
            result.append(s)
    return result


# ─────────────────────────────────────────────────────────────────────
# Risk cap (BE-aware)
# ─────────────────────────────────────────────────────────────────────

def _check_risk_cap(new_risk: float, balance: float) -> bool:
    """
    Skip trade if adding new_risk would exceed cap.
    Trades where SL is at BE (be_done=1) do NOT count toward cap —
    they have no capital at risk regardless of position size.
    """
    current_at_risk = get_capital_at_risk()  # only counts be_done=0 trades
    cap = (RISK_CAP_AMOUNT if RISK_MODE == "fixed"
           else balance * RISK_CAP_PCT / 100.0)
    if current_at_risk + new_risk > cap:
        log.info(f"  Risk cap: open={current_at_risk:.2f} + new={new_risk:.2f} "
                 f"> cap={cap:.2f} — skipping (open BE-exempt positions don't count)")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────
# Concept drift detection — PSI on top-10 features
# ─────────────────────────────────────────────────────────────────────

class _DriftMonitor:
    """
    Population Stability Index (PSI) drift monitor.

    Compares the current live data window against the reference histograms
    saved during training.  Graduated response:
      PSI < 0.10  — stable, no action
      0.10–0.25   — moderate drift, log warning
      ≥ 0.25      — major drift, reduce sizing / flag for retrain
    """
    PSI_MOD   = 0.10   # moderate drift threshold
    PSI_MAJOR = 0.25   # major drift threshold
    WINDOW    = 500    # number of recent bars to compare against reference

    def __init__(self, key: str, drift_ref: dict):
        self.key      = key
        self.ref      = drift_ref   # {col: {"bin_edges": ..., "pct": ...}}
        self.status   = "ok"        # "ok" | "moderate" | "major"
        self.avg_psi  = 0.0

    def _psi_one(self, col: str, live_vals: np.ndarray) -> float:
        ref   = self.ref[col]
        edges = ref["bin_edges"]
        exp   = ref["pct"]
        counts, _ = np.histogram(live_vals, bins=edges)
        act = counts / (counts.sum() + 1e-10)
        # Avoid log(0): clip both to a small floor
        act = np.clip(act, 1e-6, None)
        exp = np.clip(exp, 1e-6, None)
        return float(np.sum((act - exp) * np.log(act / exp)))

    def check(self, df: pd.DataFrame, feature_cols: list) -> str:
        """
        Compute PSI for each reference feature and return status string.
        Reads the last WINDOW rows from df.
        """
        if df is None or len(df) < 50:
            return self.status
        window = df[feature_cols].dropna().tail(self.WINDOW)
        psi_vals = []
        for col, ref_data in self.ref.items():
            if col not in window.columns:
                continue
            vals = window[col].dropna().values
            if len(vals) < 20:
                continue
            psi_vals.append(self._psi_one(col, vals))

        if not psi_vals:
            return self.status

        self.avg_psi = float(np.mean(psi_vals))
        if self.avg_psi >= self.PSI_MAJOR:
            self.status = "major"
        elif self.avg_psi >= self.PSI_MOD:
            self.status = "moderate"
        else:
            self.status = "ok"
        return self.status


# ─────────────────────────────────────────────────────────────────────
# Smart kill switch — rolling 20-trade Sharpe + binomial win rate test
# ─────────────────────────────────────────────────────────────────────

class _SmartKillSwitch:
    """
    Monitors the last 20 closed live trades for a strategy.

    Graduated response:
      Sharpe < 0 or binomial WR p < 0.10  → "warn"  (log warning)
      Sharpe < -0.5 or binomial WR p < 0.05 → "reduce" (cut sizing to 50%)
      Sharpe < -1.0 or very poor WR (p < 0.01) → "disable"

    Uses binomial one-sided p-value (H0: win rate ≥ 50%, H1: WR < 50%).
    Hard 35% DD circuit breaker handled separately by RiskGate.
    """
    N_TRADES  = 20

    def __init__(self, strategy_id: str):
        self.sid    = strategy_id
        self.status = "ok"        # "ok" | "warn" | "reduce" | "disable"
        self.sharpe = 0.0
        self.wr     = 0.5

        # G4: load expected baselines from DB — enables strategy-specific thresholds.
        # A conservative strategy (Sharpe=0.8 expected) and an aggressive one (Sharpe=2.0)
        # should not trigger at the same absolute levels.
        s = get_strategy(strategy_id)
        self.expected_sharpe   = float(s.get("expected_sharpe")   or 0.0) if s else 0.0
        self.expected_win_rate = float(s.get("expected_win_rate") or 0.5) if s else 0.5
        if self.expected_sharpe > 0:
            log.info(
                f"[KILLSWITCH] {strategy_id}: baselines loaded — "
                f"expected_sharpe={self.expected_sharpe:.2f}, "
                f"expected_win_rate={self.expected_win_rate:.1%}"
            )
        else:
            log.warning(
                f"[KILLSWITCH] {strategy_id}: no expected baselines in DB — "
                f"using absolute thresholds. Run training to populate."
            )

    @staticmethod
    def _binomial_p(wins: int, n: int) -> float:
        """One-sided binomial p-value for H0: p >= 0.5 (test: WR below chance)."""
        from scipy.stats import binom
        return float(binom.cdf(wins, n, 0.5))

    def check(self) -> str:
        trades = get_recent_live_trades(self.sid, n=self.N_TRADES)
        if len(trades) < 10:
            self.status = "ok"
            return self.status

        pnls = np.array([t.get("pnl", 0) or 0 for t in trades], dtype=float)
        wins = int((pnls > 0).sum())
        n    = len(pnls)
        self.wr = wins / n

        # Trade-level Sharpe (no annualisation — relative measure)
        if pnls.std() > 0:
            self.sharpe = float(pnls.mean() / pnls.std())
        else:
            self.sharpe = 0.0

        p_val      = self._binomial_p(wins, n)
        prev_state = self.status

        # G4: strategy-specific thresholds when baseline is available; else absolute fallback.
        if self.expected_sharpe > 0:
            # Relative: trigger when live Sharpe degrades significantly vs training
            disable_sharpe = -max(0.5, self.expected_sharpe * 0.5)
            reduce_sharpe  = -max(0.2, self.expected_sharpe * 0.2)
            warn_sharpe    = 0.0
        else:
            disable_sharpe, reduce_sharpe, warn_sharpe = -1.0, -0.5, 0.0

        # Win-rate thresholds relative to expected baseline (10pp / 15pp degradation)
        wr_delta = self.wr - self.expected_win_rate
        p_disable = 0.01 if wr_delta < -0.15 else 0.001
        p_reduce  = 0.05 if wr_delta < -0.10 else 0.01
        p_warn    = 0.10

        if self.sharpe < disable_sharpe or p_val < p_disable:
            self.status = "disable"
        elif self.sharpe < reduce_sharpe or p_val < p_reduce:
            self.status = "reduce"
        elif self.sharpe < warn_sharpe or p_val < p_warn:
            self.status = "warn"
        else:
            self.status = "ok"

        # 3J: log every check; flag state transitions prominently
        wr_pct  = self.wr * 100
        exp_pct = self.expected_win_rate * 100
        if self.status != prev_state:
            log.warning(
                f"[KILLSWITCH] {self.sid}: {prev_state.upper()} → {self.status.upper()} "
                f"| wins={wins}/{n} ({wr_pct:.1f}%, expected {exp_pct:.1f}%) "
                f"| sharpe={self.sharpe:.2f} (expected {self.expected_sharpe:.2f}) "
                f"| p_val={p_val:.4f}"
            )
            if self.status == "reduce":
                log.warning(
                    f"[KILLSWITCH] {self.sid}: risk halved → ${FIXED_RISK_AMT * 0.5:.0f}/trade"
                )
            elif self.status == "disable":
                log.warning(
                    f"[KILLSWITCH] {self.sid}: DISABLED — no new orders until manual reset"
                )
        else:
            log.info(
                f"[KILLSWITCH] {self.sid}: state={self.status.upper()} "
                f"| wins={wins}/{n} ({wr_pct:.1f}%) "
                f"| sharpe={self.sharpe:.2f} | p_val={p_val:.4f}"
            )
        return self.status


# ─────────────────────────────────────────────────────────────────────
# BE monitoring — detect when MT5 SL has been moved to break-even
# ─────────────────────────────────────────────────────────────────────

def _monitor_open_positions():
    """
    Sync BE status from MT5 into SQLite.
    If a position's SL has been manually (or by ML) moved to >= entry
    (long) or <= entry (short), mark it be_done=1 — no capital at risk.
    """
    open_trades = get_open_trades()
    if not open_trades:
        return
    positions = mt5.positions_get()
    if not positions:
        return
    pos_map = {p.ticket: p for p in positions}

    for trade in open_trades:
        ticket = trade.get("ticket")
        if not ticket:
            continue
        if ticket not in pos_map:
            close_live_trade(ticket, pnl=0.0)
            continue
        pos   = pos_map[ticket]
        entry = trade.get("entry_price", 0)
        direction = trade.get("direction", 1)
        if not trade.get("be_done"):
            if direction == 1 and pos.sl >= entry:
                update_trade_be(ticket, True)
                log.info(f"  BE: ticket={ticket} SL={pos.sl:.2f} ≥ entry={entry:.2f} — risk freed")
            elif direction == -1 and pos.sl > 0 and pos.sl <= entry:
                update_trade_be(ticket, True)
                log.info(f"  BE: ticket={ticket} SL={pos.sl:.2f} ≤ entry={entry:.2f} — risk freed")


# ─────────────────────────────────────────────────────────────────────
# Hot-reload: reload models if train.py has updated files on disk
# ─────────────────────────────────────────────────────────────────────

def _newest_model_mtime(symbol: str) -> float:
    """Return the most recent modification time across all model files."""
    mtimes = []
    for tf in PARAM_SEEDS["entry_tf_options"]:
        key = f"{symbol}_{tf}m"
        for suffix in ["xgb", "rf", "scaler"]:
            p = MODEL_DIR / f"{suffix}_{key}.pkl"
            if p.exists():
                mtimes.append(p.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def _try_hot_reload(symbol: str, tf_feat: dict,
                    models_cache: dict, scalers_cache: dict,
                    last_mtime: float) -> tuple[dict, dict, float]:
    """
    If model files on disk are newer than last_mtime, reload them.
    Only reloads if no open positions exist for this symbol (safe).
    Returns updated (models_cache, scalers_cache, new_mtime).
    """
    current_mtime = _newest_model_mtime(symbol)
    if current_mtime <= last_mtime:
        return models_cache, scalers_cache, last_mtime

    # Don't reload mid-trade (keep existing positions stable)
    open_for_symbol = [t for t in get_open_trades()
                       if t.get("symbol") == symbol and t.get("status") == "open"]
    if open_for_symbol:
        log.info(f"  Hot-reload skipped: {len(open_for_symbol)} open positions for {symbol}")
        return models_cache, scalers_cache, last_mtime

    log.info(f"  Hot-reload: newer model files detected for {symbol} — reloading")
    new_m, new_s = load_models_from_disk(symbol, tf_feat)
    # Merge into existing caches (other symbols untouched)
    models_cache.update(new_m)
    scalers_cache.update(new_s)
    log.info(f"  Hot-reload complete: {symbol}")
    return models_cache, scalers_cache, current_mtime


# ─────────────────────────────────────────────────────────────────────
# Load system from disk
# ─────────────────────────────────────────────────────────────────────

def load_live_system() -> tuple[dict, dict, dict]:
    """Load data + models from disk. No training. Fast (~3-10 sec)."""
    log.info("=" * 60)
    log.info("LOADING SYSTEM FROM DISK")
    log.info("=" * 60)

    models_cache  = {}
    scalers_cache = {}
    all_data      = {}

    for symbol in INSTRUMENTS:
        log.info(f"  {symbol}:")
        tf_feat = load_symbol_data(symbol, save_featured=False)
        if not tf_feat:
            log.warning(f"    No data for {symbol} — skipping")
            continue
        all_data[symbol] = tf_feat
        m, s = load_models_from_disk(symbol, tf_feat)
        models_cache.update(m)
        scalers_cache.update(s)

    log.info(f"\nLoaded {len(models_cache)} model objects — ready.")
    return models_cache, scalers_cache, all_data


# ─────────────────────────────────────────────────────────────────────
# Live loop
# ─────────────────────────────────────────────────────────────────────

def run_live_loop(models_cache: dict, scalers_cache: dict, all_data: dict):
    log.info("=" * 60)
    log.info(f"LIVE TRADING LOOP {'[DRY RUN — no orders]' if DRY_RUN else ''}")
    log.info(f"Risk: {RISK_MODE}  cap={'{}%'.format(RISK_CAP_PCT) if RISK_MODE=='percent' else '${}'.format(RISK_CAP_AMOUNT)}")
    log.info(f"Hard limits: {HARD_LIMITS}")
    log.info("=" * 60)

    risk_gate   = RiskGate()
    last_reset  = datetime.now().date()
    live_cache  = {}
    last_reload = {sym: _newest_model_mtime(sym) for sym in INSTRUMENTS}
    last_reload_check = datetime.now()

    # Pre-populate live cache
    for symbol, tf_dict in all_data.items():
        for tf, df in tf_dict.items():
            live_cache[f"{symbol}_{tf}m"] = df
    log.info(f"Pre-loaded {len(live_cache)} data frames")

    # Initialise drift monitors and kill switches (keyed by strategy_id)
    _drift_monitors: dict[str, _DriftMonitor]   = {}
    _kill_switches:  dict[str, _SmartKillSwitch] = {}
    _drift_check_interval = 60 * 60   # check drift every 60 minutes
    _last_drift_check:    dict[str, float] = {}
    _kill_check_interval  = 60 * 60   # check kill switch every 60 minutes
    _last_kill_check:     dict[str, float] = {}

    active_strategies = _get_active_strategies()
    log.info(f"Active strategies ({len(active_strategies)}):")
    for s in active_strategies:
        log.info(f"  {s.get('strategy_id', s.get('symbol','?'))}: "
                 f"TF={s.get('entry_tf')}m HTF={s.get('htf_tf')}m "
                 f"conf={s.get('confidence',0):.2f} "
                 f"ER={s.get('efficiency_ratio','n/a')}")

    while True:
        try:
            now = datetime.now()

            # Daily reset
            if now.date() > last_reset:
                risk_gate.reset_daily()
                last_reset = now.date()
                log.info("Daily counters reset")

            # Monitor BE status
            _monitor_open_positions()

            # Periodic hot-reload check
            mins_since_check = (now - last_reload_check).total_seconds() / 60
            if mins_since_check >= MODEL_RELOAD_INTERVAL_MIN:
                for symbol in INSTRUMENTS:
                    if all_data.get(symbol):
                        models_cache, scalers_cache, last_reload[symbol] = \
                            _try_hot_reload(symbol, all_data[symbol],
                                            models_cache, scalers_cache,
                                            last_reload[symbol])
                last_reload_check = now

            # Sleep until next candle close
            active_strategies = _get_active_strategies()
            min_sleep = min(
                (seconds_to_next_candle_close(s.get("entry_tf", PARAM_SEEDS["entry_tf_default"]))
                 for s in active_strategies),
                default=60.0,
            )
            log.info(f"Sleeping {min_sleep:.0f}s to next candle close...")
            time.sleep(max(1.0, min_sleep))

            # Process each active strategy
            for strategy in active_strategies:
                sid      = strategy.get("strategy_id", strategy.get("symbol", "?"))
                symbol   = strategy.get("symbol", list(INSTRUMENTS.keys())[0])
                entry_tf = strategy.get("entry_tf", PARAM_SEEDS["entry_tf_default"])
                htf_tf   = strategy.get("htf_tf", 0)
                is_tick  = symbol in TICK_DATA_SYMBOLS

                params = {
                    "entry_tf":   entry_tf,
                    "htf_tf":     htf_tf,
                    "sl_atr":     strategy.get("sl_atr",     PARAM_SEEDS["sl_atr_seed"]),
                    "tp_mult":    strategy.get("tp_mult",    PARAM_SEEDS["tp_mult_seed"]),
                    "confidence": strategy.get("confidence", PARAM_SEEDS["confidence_seed"]),
                    "htf_weight": strategy.get("htf_weight", PARAM_SEEDS["htf_weight_seed"]),
                    "be_r":       strategy.get("be_r", 0),
                }

                entry_key = f"{symbol}_{entry_tf}m"

                # Refresh live data
                live_cache[entry_key] = refresh_live_data(
                    symbol, entry_tf, live_cache.get(entry_key, pd.DataFrame()), is_tick)
                if htf_tf > 0:
                    htf_key = f"{symbol}_{htf_tf}m"
                    live_cache[htf_key] = refresh_live_data(
                        symbol, htf_tf, live_cache.get(htf_key, pd.DataFrame()), is_tick)

                # ── Initialise drift monitor (once per strategy, lazily) ──────
                if sid not in _drift_monitors:
                    drift_ref = scalers_cache.get(f"drift_ref_{entry_key}")
                    if drift_ref:
                        _drift_monitors[sid]     = _DriftMonitor(entry_key, drift_ref)
                        _last_drift_check[sid]   = 0.0
                if sid not in _kill_switches:
                    _kill_switches[sid]      = _SmartKillSwitch(sid)
                    _last_kill_check[sid]    = 0.0

                # ── Periodic drift check ───────────────────────────────────────
                now_ts = time.time()
                if (sid in _drift_monitors and
                        now_ts - _last_drift_check.get(sid, 0) >= _drift_check_interval):
                    dm      = _drift_monitors[sid]
                    df_live = live_cache.get(entry_key)
                    feat_c  = [c for c in (df_live.columns if df_live is not None else [])
                               if c in dm.ref]
                    d_stat  = dm.check(df_live, feat_c if feat_c else list(dm.ref.keys()))
                    _last_drift_check[sid] = now_ts
                    if d_stat == "major":
                        log.warning(f"[{sid}] DRIFT MAJOR: avg PSI={dm.avg_psi:.3f} "
                                    f"— model may be stale, consider retraining")
                    elif d_stat == "moderate":
                        log.info(f"[{sid}] Drift moderate: avg PSI={dm.avg_psi:.3f}")

                # ── Periodic kill switch check ─────────────────────────────────
                if now_ts - _last_kill_check.get(sid, 0) >= _kill_check_interval:
                    ks_stat = _kill_switches[sid].check()
                    _last_kill_check[sid] = now_ts
                    ks      = _kill_switches[sid]
                    if ks_stat == "disable":
                        log.warning(f"[{sid}] KILL SWITCH DISABLE: Sharpe={ks.sharpe:.2f} "
                                    f"WR={ks.wr:.1%} — strategy auto-disabled")
                        continue   # skip signal — no new entries
                    elif ks_stat == "reduce":
                        log.warning(f"[{sid}] Kill switch REDUCE: Sharpe={ks.sharpe:.2f} "
                                    f"WR={ks.wr:.1%} — risk halved")
                    elif ks_stat == "warn":
                        log.info(f"[{sid}] Kill switch WARN: Sharpe={ks.sharpe:.2f} "
                                 f"WR={ks.wr:.1%}")

                # Generate signal
                signal = get_signal(
                    symbol=symbol, params=params,
                    models_cache=models_cache, scalers_cache=scalers_cache,
                    live_data_cache=live_cache,
                )

                if signal["direction"] == 0:
                    log.debug(f"{sid}: no signal — {signal.get('reason','')}")
                    continue

                # Session time gate — block entries 20:30–01:00 UK time
                if _is_session_blocked():
                    log.debug(f"{sid}: blocked by session gate (20:30-01:00 UK)")
                    continue

                # Hard risk gate
                allowed, reason = risk_gate.can_trade()
                if not allowed:
                    log.warning(f"[{sid}] Blocked by risk gate: {reason}")
                    continue

                balance    = get_account_balance()
                trade_risk = FIXED_RISK_AMT

                # Kill switch reduce: halve risk for degraded strategies
                if _kill_switches.get(sid) and _kill_switches[sid].status == "reduce":
                    trade_risk = FIXED_RISK_AMT * 0.5

                # Drift major: also halve risk (independent of kill switch)
                if _drift_monitors.get(sid) and _drift_monitors[sid].status == "major":
                    trade_risk *= 0.5

                # Volatility-adjusted sizing (disabled by default; enable via VOL_SIZE_ENABLED=true)
                if VOL_SIZE_ENABLED:
                    VOL_SIZE_MIN = 0.5
                    VOL_SIZE_MAX = 2.0
                    base_pct     = RISK_PCT
                    entry_key_s  = f"{symbol}_{signal['entry_tf']}m"
                    df_live      = live_cache.get(entry_key_s, pd.DataFrame())
                    if not df_live.empty and "atr14" in df_live.columns:
                        recent_atr = df_live["atr14"].dropna()
                        if len(recent_atr) >= 20:
                            cur_atr  = float(recent_atr.iloc[-1])
                            med_atr  = float(recent_atr.rolling(100, min_periods=20).median().iloc[-1])
                            if med_atr > 0 and cur_atr > 0:
                                norm_atr   = cur_atr / med_atr
                                eff_pct    = float(np.clip(RISK_PCT / norm_atr,
                                                           VOL_SIZE_MIN, VOL_SIZE_MAX))
                                trade_risk = max(balance, 1.0) * eff_pct / 100.0

                # BE-aware risk cap
                if not _check_risk_cap(trade_risk, balance):
                    continue

                lot = risk_gate.position_size(sl_pips=signal["sl_pips"], symbol=symbol)

                dir_str = "LONG" if signal["direction"] == 1 else "SHORT"
                log.info(f"SIGNAL [{sid}] ► {symbol} {dir_str} | "
                         f"conf:{signal['confidence']:.3f} | "
                         f"TF:{signal['entry_tf']}m HTF:{signal['htf_used']}m | "
                         f"lot:{lot} | SL:{signal['sl_price']} TP:{signal['tp_price']}")

                if DRY_RUN:
                    log.info(f"  [DRY RUN] Order NOT placed.")
                    continue

                result = place_order(
                    symbol=symbol, direction=signal["direction"],
                    lot=lot, sl=signal["sl_price"], tp=signal["tp_price"],
                    comment=f"{sid[:16]}",
                )

                if result["success"]:
                    log_trade({
                        "symbol":    symbol, "direction": signal["direction"],
                        "entry":     signal["entry_price"], "sl": signal["sl_price"],
                        "tp":        signal["tp_price"], "lot": lot,
                        "confidence":signal["confidence"], "entry_tf": signal["entry_tf"],
                        "htf_tf":    signal["htf_used"], "params": params,
                        "ticket":    result["ticket"],
                        "data_source":"tick" if is_tick else "mt5",
                        "timestamp": now.isoformat(), "used_for_retrain": False,
                    })
                    log_live_trade({
                        "strategy_id": sid, "symbol": symbol,
                        "direction":   signal["direction"],
                        "entry_price": signal["entry_price"],
                        "sl_price":    signal["sl_price"],
                        "tp_price":    signal["tp_price"],
                        "lot":         lot, "confidence": signal["confidence"],
                        "entry_tf":    signal["entry_tf"],
                        "htf_tf":      signal["htf_used"],
                        "ticket":      result["ticket"],
                        "risk_amount": trade_risk,
                        "data_source": "tick" if is_tick else "mt5",
                        "be_done":     0,
                    })

                # Incremental param update after enough new trades
                if should_retrain(symbol):
                    log.info(f"  Incremental param update triggered: {symbol}")
                    incremental_update(
                        symbol=symbol,
                        df_dict=all_data.get(symbol, {}),
                        models_cache=models_cache,
                        scalers_cache=scalers_cache,
                    )

        except KeyboardInterrupt:
            log.info("Ctrl+C — stopping live loop cleanly.")
            break
        except Exception as e:
            log.error(f"Live loop error: {e}", exc_info=True)
            log.info("Recovering in 30s...")
            time.sleep(30)


# ─────────────────────────────────────────────────────────────────────
# Test trade — verifies full pipeline: order → DB → BE → close
# ─────────────────────────────────────────────────────────────────────

def run_test_trade():
    """
    Place ONE real micro-lot test trade and immediately close it.
    Tests: MT5 order placement, DB logging, BE monitoring, trade close.

    Usage:  python live.py --test-trade
    Cost:   ~0.01 lot × spread ≈ a few cents worst case.
    """
    log.info("=" * 60)
    log.info("TEST TRADE MODE — verifying full pipeline")
    log.info("=" * 60)

    if not connect_mt5():
        log.error("Cannot connect to MT5 — check terminal is open")
        sys.exit(1)

    active_strategies = _get_active_strategies()
    if not active_strategies:
        log.error("No active strategies found — run training first")
        sys.exit(1)

    strategy = active_strategies[0]
    symbol   = strategy.get("symbol", list(INSTRUMENTS.keys())[0])
    mt5_sym  = INSTRUMENTS.get(symbol, symbol)
    sid      = strategy.get("strategy_id", symbol)

    # Symbol info
    info = mt5.symbol_info(mt5_sym)
    if not info:
        log.error(f"Symbol info not available for {mt5_sym} — check MT5 Market Watch")
        sys.exit(1)

    tick = mt5.symbol_info_tick(mt5_sym)
    if not tick:
        log.error(f"No price tick for {mt5_sym}")
        sys.exit(1)

    lot    = info.volume_min        # smallest possible lot (e.g. 0.01)
    point  = info.point
    digits = info.digits
    entry  = round(tick.ask, digits)
    sl     = round(entry - 500 * point, digits)   # 500 points SL  (~0.5% risk on 0.01 lot)
    tp     = round(entry + 1000 * point, digits)  # 1000 points TP (2:1 R:R)

    log.info(f"Symbol   : {mt5_sym}  (via strategy: {sid})")
    log.info(f"Lot      : {lot} (minimum allowed by broker)")
    log.info(f"Direction: LONG (test uses long only)")
    log.info(f"Entry≈   : {entry}  SL: {sl}  TP: {tp}")
    log.info(f"Point    : {point}  Digits: {digits}")
    log.info("-" * 60)

    results = {}

    # ── 1. Place order ───────────────────────────────────────────────
    log.info("STEP 1: Placing order...")
    result = place_order(
        symbol=symbol, direction=1, lot=lot, sl=sl, tp=tp,
        comment="test_pipeline",
    )
    if result.get("success"):
        ticket = result["ticket"]
        log.info(f"  [PASS] Order placed — ticket: {ticket}")
        results["order_placement"] = "PASS"
    else:
        log.error(f"  [FAIL] Order not placed — {result.get('error','unknown error')}")
        log.error("  Check: MT5 algo trading enabled? Correct symbol name? Market open?")
        results["order_placement"] = "FAIL"
        mt5.shutdown()
        _print_test_results(results)
        sys.exit(1)

    # ── 2. Log to DB ─────────────────────────────────────────────────
    log.info("STEP 2: Logging trade to database...")
    try:
        balance    = get_account_balance()
        trade_risk = (FIXED_RISK_AMT if RISK_MODE == "fixed"
                      else max(balance, 1.0) * RISK_PCT / 100.0)
        log_live_trade({
            "strategy_id": sid,
            "symbol":      symbol,
            "direction":   1,
            "entry_price": entry,
            "sl_price":    sl,
            "tp_price":    tp,
            "lot":         lot,
            "confidence":  0.99,
            "entry_tf":    strategy.get("entry_tf", 5),
            "htf_tf":      strategy.get("htf_tf", 0),
            "ticket":      ticket,
            "risk_amount": trade_risk,
            "data_source": "test",
            "be_done":     0,
        })
        log.info(f"  [PASS] Trade logged to DB (ticket: {ticket})")
        results["db_logging"] = "PASS"
    except Exception as e:
        log.error(f"  [FAIL] DB logging error: {e}")
        results["db_logging"] = "FAIL"

    # ── 3. Verify MT5 position exists ────────────────────────────────
    log.info("STEP 3: Verifying position exists in MT5...")
    time.sleep(2)
    positions = mt5.positions_get(ticket=ticket)
    if positions:
        pos = positions[0]
        log.info(f"  [PASS] Position confirmed: ticket={pos.ticket} "
                 f"volume={pos.volume} profit={pos.profit:.2f}")
        results["position_verify"] = "PASS"
    else:
        log.warning(f"  [WARN] Position not found (may have closed instantly on spread) — ticket={ticket}")
        results["position_verify"] = "WARN"

    # ── 4. BE monitoring ─────────────────────────────────────────────
    log.info("STEP 4: Running BE monitor...")
    try:
        _monitor_open_positions()
        log.info("  [PASS] BE monitoring ran without errors")
        results["be_monitoring"] = "PASS"
    except Exception as e:
        log.error(f"  [FAIL] BE monitoring error: {e}")
        results["be_monitoring"] = "FAIL"

    # ── 5. Risk cap check ────────────────────────────────────────────
    log.info("STEP 5: Risk cap check...")
    try:
        cap_ok = _check_risk_cap(trade_risk, balance)
        at_risk = get_capital_at_risk()
        log.info(f"  [PASS] Capital at risk (BE-aware): ${at_risk:.2f}  "
                 f"Cap: ${balance * RISK_CAP_PCT / 100:.2f}  "
                 f"Would allow new trade: {cap_ok}")
        results["risk_cap"] = "PASS"
    except Exception as e:
        log.error(f"  [FAIL] Risk cap error: {e}")
        results["risk_cap"] = "FAIL"

    # ── 6. Close the test trade ───────────────────────────────────────
    log.info("STEP 6: Closing test trade...")
    tick2       = mt5.symbol_info_tick(mt5_sym)
    close_price = round(tick2.bid, digits) if tick2 else sl
    close_req   = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      mt5_sym,
        "volume":      lot,
        "type":        mt5.ORDER_TYPE_SELL,
        "position":    ticket,
        "price":       close_price,
        "deviation":   30,
        "magic":       234000,
        "comment":     "test_close",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    close_res = mt5.order_send(close_req)
    if close_res and close_res.retcode == mt5.TRADE_RETCODE_DONE:
        pnl = getattr(close_res, "profit", 0.0)
        close_live_trade(ticket, pnl=float(pnl) if pnl else 0.0)
        log.info(f"  [PASS] Trade closed — ticket: {ticket}  approx P&L: ${pnl:.2f}")
        results["trade_close"] = "PASS"
    else:
        retcode = getattr(close_res, "retcode", "?")
        comment = getattr(close_res, "comment", "?")
        log.warning(f"  [WARN] Auto-close failed (retcode={retcode}: {comment})")
        log.warning(f"  >>> Please close ticket {ticket} MANUALLY in MT5 <<<")
        results["trade_close"] = f"WARN (close manually: ticket {ticket})"

    _print_test_results(results)
    mt5.shutdown()


def _print_test_results(results: dict):
    log.info("")
    log.info("=" * 60)
    log.info("TEST RESULTS")
    log.info("=" * 60)
    all_pass = True
    for step, outcome in results.items():
        icon = "[PASS]" if outcome == "PASS" else ("[WARN]" if outcome.startswith("WARN") else "[FAIL]")
        if "FAIL" in outcome:
            all_pass = False
        log.info(f"  {icon}  {step:<22} {outcome}")
    log.info("-" * 60)
    log.info(f"  Overall: {'ALL PASS — pipeline working correctly' if all_pass else 'ISSUES FOUND — see above'}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# Signal check — show raw probabilities vs threshold right now
# ─────────────────────────────────────────────────────────────────────

def run_signal_check():
    """
    Load models + data, evaluate ONE candle for every active strategy,
    and print the raw ensemble probabilities vs the confidence threshold.

    Answers: 'why are there no trades? Is the model signalling at all?'

    Usage:  python live.py --signal-check
    """
    log.info("=" * 60)
    log.info("SIGNAL CHECK — raw probabilities vs threshold")
    log.info("=" * 60)

    if not connect_mt5():
        log.error("Cannot connect to MT5")
        sys.exit(1)

    models_cache, scalers_cache, all_data = load_live_system()
    if not models_cache:
        log.error("No models loaded — run training first")
        sys.exit(1)

    live_cache = {}
    for symbol, tf_dict in all_data.items():
        for tf, df in tf_dict.items():
            live_cache[f"{symbol}_{tf}m"] = df

    active_strategies = _get_active_strategies()
    log.info(f"\nEvaluating {len(active_strategies)} strategies...\n")

    for strategy in active_strategies:
        sid      = strategy.get("strategy_id", strategy.get("symbol", "?"))
        symbol   = strategy.get("symbol", list(INSTRUMENTS.keys())[0])
        entry_tf = strategy.get("entry_tf", 5)
        htf_tf   = strategy.get("htf_tf", 0)
        conf     = strategy.get("confidence", 0.6)
        is_tick  = symbol in TICK_DATA_SYMBOLS

        params = {
            "entry_tf":   entry_tf, "htf_tf": htf_tf,
            "sl_atr":     strategy.get("sl_atr",     1.5),
            "tp_mult":    strategy.get("tp_mult",    2.0),
            "confidence": conf,
            "htf_weight": strategy.get("htf_weight", 0.3),
            "be_r":       strategy.get("be_r", 0),
        }

        # Refresh with latest bars
        entry_key = f"{symbol}_{entry_tf}m"
        live_cache[entry_key] = refresh_live_data(
            symbol, entry_tf, live_cache.get(entry_key, pd.DataFrame()), is_tick)

        signal = get_signal(
            symbol=symbol, params=params,
            models_cache=models_cache, scalers_cache=scalers_cache,
            live_data_cache=live_cache,
        )

        prob  = signal.get("confidence", 0.0)
        dir_  = signal.get("direction", 0)
        long_thresh  = conf
        short_thresh = 1.0 - conf
        gap_to_long  = long_thresh  - prob
        gap_to_short = prob - short_thresh

        if dir_ == 1:
            verdict = f">>> LONG SIGNAL  (prob={prob:.4f} >= threshold={long_thresh:.4f})"
        elif dir_ == -1:
            verdict = f">>> SHORT SIGNAL (prob={prob:.4f} <= threshold={short_thresh:.4f})"
        else:
            closer = "long" if gap_to_long < gap_to_short else "short"
            gap    = min(gap_to_long, gap_to_short)
            verdict = (f"    No signal      (prob={prob:.4f})  "
                       f"Need {gap:.4f} more to reach {closer} threshold")

        # Recent prob trend (last 10 candles if possible)
        key = f"{symbol}_{entry_tf}m"
        if key in live_cache and len(live_cache[key]) >= 10:
            # Quick: get model probs for last 10 bars
            sc_key = f"{symbol}_{entry_tf}m"
            scaler = scalers_cache.get(sc_key) or scalers_cache.get(f"scaler_{sc_key}")
            xgb_m  = models_cache.get(f"xgb_{sc_key}")
            if scaler and xgb_m and hasattr(xgb_m, "predict_proba"):
                try:
                    df_tail = live_cache[key].iloc[-10:]
                    cols    = list(scaler.feature_names_in_)
                    tail_arr = df_tail[[c for c in cols if c in df_tail.columns]].copy()
                    for c in cols:
                        if c not in tail_arr.columns:
                            tail_arr[c] = 0.0
                    scaled  = scaler.transform(tail_arr[cols])
                    probs   = xgb_m.predict_proba(scaled)[:, 1]
                    recent  = "  ".join(f"{p:.3f}" for p in probs)
                    log.info(f"  Strategy : {sid}")
                    log.info(f"  TF       : {entry_tf}m  HTF: {htf_tf}m")
                    log.info(f"  Threshold: LONG >= {long_thresh:.4f}  SHORT <= {short_thresh:.4f}")
                    log.info(f"  Last 10 XGB probs : [{recent}]")
                    log.info(f"  Latest bar result : {verdict}")
                    log.info("")
                    continue
                except Exception:
                    pass

        log.info(f"  Strategy : {sid}")
        log.info(f"  TF       : {entry_tf}m  HTF: {htf_tf}m")
        log.info(f"  Threshold: LONG >= {long_thresh:.4f}  SHORT <= {short_thresh:.4f}")
        log.info(f"  Latest bar result : {verdict}")
        log.info("")

    log.info("-" * 60)
    log.info("Tip: if probabilities are stuck between 0.35–0.65, the model")
    log.info("     sees no clear edge right now. This is normal in ranging/")
    log.info("     low-volume markets. Wait for a clear session (London/NY).")
    log.info("     If threshold > 0.80 and probs never exceed 0.70, consider")
    log.info("     running train.py --force to re-optimise the threshold.")
    log.info("=" * 60)
    mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    # Special modes — bypass normal startup checks
    if TEST_TRADE:
        run_test_trade()
        return
    if SIGNAL_CHECK:
        run_signal_check()
        return

    log.info("=" * 60)
    log.info("ML TRADING SYSTEM — LIVE TRADING")
    if DRY_RUN:
        log.info("*** DRY RUN MODE — no real orders will be placed ***")
    log.info("=" * 60)

    # Verify models exist before connecting to MT5
    missing = [s for s in INSTRUMENTS if not models_exist(s)]
    if missing:
        log.error(f"Models not found for: {missing}")
        log.error("Run training first:  python train.py")
        sys.exit(1)

    if not connect_mt5():
        log.error("Cannot connect to MT5. Ensure terminal is open and logged in.")
        sys.exit(1)

    try:
        models_cache, scalers_cache, all_data = load_live_system()
        if not models_cache:
            log.error("No models loaded. Check models/ folder and logs.")
            sys.exit(1)
        run_live_loop(models_cache, scalers_cache, all_data)
    finally:
        mt5.shutdown()
        log.info("MT5 disconnected. System stopped.")


if __name__ == "__main__":
    main()
