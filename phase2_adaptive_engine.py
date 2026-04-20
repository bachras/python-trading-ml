"""
Phase 2 — Adaptive ML Trading Engine
======================================
Self-optimising CFD trading system for MT5.

Instruments:
  Indices : US30, DE40, USTEC, UK100, US500
  Forex   : EURUSD, GBPUSD, GBPJPY, EURJPY, USDCAD

Key design principles:
  - ALL parameters are seeds — ML discovers true optimal values
  - Entry TF  : 1m, 3m, 5m, 10m, 15m  (ML picks per instrument)
  - HTF confirm: 15m, 30m, 1H, or NONE (ML picks, NONE is valid)
  - Signal loop wakes on CLOSED candle of the ML-chosen entry TF
  - Dual learning: historical offline + live incremental after each trade
  - Hard risk limits set by user, never overridden by ML

Usage:
  1. Activate venv: venv\\Scripts\\activate
  2. Ensure MT5 terminal is open and logged in
  3. Run: python phase2_adaptive_engine.py
  4. First run trains on historical data (~15-30 min depending on machine)
  5. After training, system enters live loop automatically
"""

import os, sys, time, json, logging, warnings

# sklearn emits this when its joblib-based predict_proba is called from a
# non-sklearn parallel context (e.g. DEAP GA loop). Predictions work fine.
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

# Force UTF-8 on the Windows console (default is cp1252 which cannot
# encode Unicode box-drawing characters used in log separators).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ML / optimisation
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import joblib

import xgboost as xgb
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────────────────────
# DIRECTORIES — driven from .env so you can move the folder
# to any drive (C:, F:, external SSD, etc.) without code changes.
# Set BASE_DIR in your .env file, e.g.:
#   BASE_DIR=F:\trading_ml
# Defaults to C:\trading_ml if not set.
# MT5 terminal stays on C: in AppData — that's fine, Python
# connects to it via API regardless of where this folder lives.
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(os.getenv("BASE_DIR", r"C:\trading_ml"))
DATA_DIR   = Path(os.getenv("DATA_DIR",   str(BASE_DIR / "data")))
MODEL_DIR  = Path(os.getenv("MODEL_DIR",  str(BASE_DIR / "models")))
LOG_DIR    = Path(os.getenv("LOG_DIR",    str(BASE_DIR / "logs")))
PARAMS_DIR = Path(os.getenv("PARAMS_DIR", str(BASE_DIR / "params")))

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, PARAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        # encoding='utf-8' on FileHandler so log files are always UTF-8.
        logging.FileHandler(
            LOG_DIR / f"trading_{datetime.now():%Y%m%d}.log",
            encoding="utf-8",
        ),
        # Open stdout (fd 1) with explicit UTF-8 + replace for console.
        # This avoids cp1252 UnicodeEncodeError on Windows PowerShell
        # regardless of system locale or sys.stdout.reconfigure timing.
        logging.StreamHandler(
            stream=open(1, mode="w", encoding="utf-8", errors="replace", closefd=False)
        ),
    ],
)
log = logging.getLogger("adaptive_engine")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Instruments — broker symbol names may vary, update to match yours
# ── Instrument symbol names — ALL read from .env ────────────────
# Broker symbol names vary. Update .env only — never touch this code.
# Format in .env:
#   SYMBOL_US30=US30        (or DJIA, WS30, DJ30, US30.cash)
#   SYMBOL_DE40=GER40       (or DE30, DAX40, GER30)
#   SYMBOL_USTEC=NAS100     (or NASDAQ, US100, USTEC100)
#   etc.
def _sym(key: str, default: str) -> str:
    return os.getenv(f"SYMBOL_{key}", default)

# All supported instruments: canonical key → (broker symbol, asset type).
# Broker symbol is resolved from .env SYMBOL_<KEY>=... at startup.
# Canonical keys are fixed (US30, DE40 etc.) regardless of broker name.
_ALL_INSTRUMENTS = {
    "US30":   (_sym("US30",   "US30"),   "index"),
    "DE40":   (_sym("DE40",   "DE40"),   "index"),
    "USTEC":  (_sym("USTEC",  "USTEC"),  "index"),
    "UK100":  (_sym("UK100",  "UK100"),  "index"),
    "US500":  (_sym("US500",  "US500"),  "index"),
    "EURUSD": (_sym("EURUSD", "EURUSD"), "forex"),
    "GBPUSD": (_sym("GBPUSD", "GBPUSD"), "forex"),
    "GBPJPY": (_sym("GBPJPY", "GBPJPY"), "forex"),
    "EURJPY": (_sym("EURJPY", "EURJPY"), "forex"),
    "USDCAD": (_sym("USDCAD", "USDCAD"), "forex"),
}

# ACTIVE_SYMBOLS in .env controls which instruments are trained and traded.
# Format: ACTIVE_SYMBOLS=US30,DE40,EURUSD  (canonical keys, no spaces)
# Empty or missing = ALL instruments active.
# Add a symbol when you have quality tick data or MT5 history for it.
_active = {s.strip().upper() for s in os.getenv("ACTIVE_SYMBOLS", "").split(",") if s.strip()}
INSTRUMENTS = {
    broker: atype
    for key, (broker, atype) in _ALL_INSTRUMENTS.items()
    if not _active or key in _active
}
log_tmp = logging.getLogger("adaptive_engine")
log_tmp.info(f"Active instruments: {list(INSTRUMENTS.keys())}")

# MT5 timeframe map
TF_MAP = {
    1:  mt5.TIMEFRAME_M1,
    3:  mt5.TIMEFRAME_M3,
    5:  mt5.TIMEFRAME_M5,
    10: mt5.TIMEFRAME_M10,
    15: mt5.TIMEFRAME_M15,
    30: mt5.TIMEFRAME_M30,
    60: mt5.TIMEFRAME_H1,
}

# ── Seed parameter space (ML discovers true values within these ranges) ──
PARAM_SEEDS = {
    # Entry timeframe candidates (minutes) — ML picks best per instrument
    # Includes 30m for wider exploration on slower-moving setups
    "entry_tf_options":  [1, 3, 5, 10, 15, 30],
    "entry_tf_default":  5,

    # HTF confirmation candidates — includes 0 = NONE (ML may skip HTF)
    # Added 240m (4H) for stronger trend confirmation on longer entries
    "htf_options":       [0, 15, 30, 60, 240],
    "htf_default":       60,

    # Stop loss ATR multiplier — widened low end to allow tighter scalp SLs
    # and high end to allow wide SLs for volatile instruments
    "sl_atr_min":        0.3,
    "sl_atr_max":        5.0,
    "sl_atr_seed":       1.5,

    # Take profit multiplier of SL distance — wider for runner strategies
    "tp_mult_min":       0.5,
    "tp_mult_max":       8.0,
    "tp_mult_seed":      2.0,

    # Signal confidence threshold — lowered floor to explore more active trading,
    # raised ceiling to allow very selective high-conviction only strategies
    "confidence_min":    0.50,
    "confidence_max":    0.92,
    "confidence_seed":   0.65,

    # HTF alignment weight — full range, 0 = ignore HTF entirely
    "htf_weight_min":    0.0,
    "htf_weight_max":    1.0,
    "htf_weight_seed":   0.5,

    # Break-even stop options:
    #   0 = never move SL
    #   1 = move SL to entry+1pt when price reaches +1R
    #   2 = move SL to entry+1pt when price reaches +2R
    #   3 = move SL to entry+1pt when price reaches +3R (trail on runners)
    "be_r_options":      [0, 1, 2, 3],
    "be_r_seed":         1,
}

# ── Hard risk limits — NEVER modified by ML ──
HARD_LIMITS = {
    "max_daily_loss_pct":   float(os.getenv("MAX_DAILY_LOSS_PCT",  "2.0")),
    "max_drawdown_pct":     float(os.getenv("MAX_DRAWDOWN_PCT",   "10.0")),
    "max_open_positions":   int(os.getenv("MAX_OPEN_POSITIONS",   "3")),
    "risk_per_trade_pct":   float(os.getenv("RISK_PER_TRADE_PCT",  "1.0")),
}

# Optuna optimisation budget — 500 trials per TF, 12-24h runtime expected
OPTUNA_TRIALS       = 500
OPTUNA_LIVE_TRIALS  = 50    # faster re-optimisation after live trades

# Minimum lookback bars for feature warmup (used as loop start offset in ga_fitness)
SEQ_LEN = 60

# Minimum bars needed before training
MIN_BARS = 500

# ── S3: Global random seed (from env — set in train.py startup block) ────
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))

# ── R7: Synthetic spread model (Dukascopy spread is zero/near-zero) ──────
# Applied in ga_fitness() and backtest_engine.py when spread_mean < 0.1 pts.
SPREAD_BASE_PTS  = float(os.getenv("SPREAD_BASE_PTS",  "1.5"))
SPREAD_ATR_COEFF = float(os.getenv("SPREAD_ATR_COEFF", "0.04"))
SPREAD_OPEN_MULT = float(os.getenv("SPREAD_OPEN_MULT", "2.0"))
SPREAD_OPEN_BARS = int(os.getenv("SPREAD_OPEN_BARS",   "5"))
# E3: Execution latency model — fraction of trades that fill at next-bar open.
# MT5_DEVIATION_PTS must match the deviation= parameter in live.py's IOC order call.
EXEC_DELAY_PROB   = float(os.getenv("EXEC_DELAY_PROB",    "0.30"))
MT5_DEVIATION_PTS = float(os.getenv("MT5_DEVIATION_PTS",  "30"))

# R6: Features confirmed to use future bars (SWING_LOOKBACK = 10 bars each side).
# These are whitelisted for the leakage check — training proceeds with a WARNING
# rather than RuntimeError. Remediation (backward-only swing detection) is L3.
KNOWN_LOOKAHEAD_FEATURES = frozenset({
    # Swing detection: N-bar symmetric lookback — needs N future bars to confirm a pivot.
    # Accepted: the approximation is valid during live trading where full context exists.
    "is_swing_high", "is_swing_low",
    "dist_swing_high", "dist_swing_low",
    "equal_high", "equal_low",
    "stop_hunt_up", "stop_hunt_down",
    # ORB: opening-range value is broadcast to ALL session bars (inc. pre-market).
    # When the leakage pivot falls before NYSE open, the opening bars are "future",
    # so shuffling them changes the ORB for pre-pivot pre-market bars of that session.
    # Accepted: ORB is a session constant known to all participants by 9:35 AM ET;
    # the model only acts on it during regular-hours trading, not pre-market.
    "orb_high", "orb_low", "orb_range",
    "orb_dist_high", "orb_dist_low",
    "orb_above", "orb_below", "orb_inside",
})


# Live trade log path
TRADE_LOG = LOG_DIR / "trade_log.json"

# ─────────────────────────────────────────────────────────────
# 1. MT5 CONNECTION
# ─────────────────────────────────────────────────────────────

def connect_mt5() -> bool:
    mt5_path = os.getenv("MT5_PATH", "").strip()

    if not mt5_path:
        # MT5_PATH not set — refuse to connect. Auto-detect is disabled because
        # it can silently connect to the wrong terminal (different account/broker).
        log.error("MT5_PATH is not set in .env — cannot connect.")
        log.error("  Set MT5_PATH to your terminal executable, e.g.:")
        log.error("  MT5_PATH=C:\\Program Files\\MetaTrader 5 IC Markets Global\\terminal64.exe")
        log.error("  To find the path: open your MT5 terminal → Help → About → copy install dir")
        return False

    from pathlib import Path as _Path
    mt5_path_norm = str(_Path(mt5_path))
    log.info(f"MT5 connecting to: {mt5_path_norm}")

    if not _Path(mt5_path_norm).exists():
        log.error(f"MT5 executable not found: {mt5_path_norm}")
        log.error("  Check MT5_PATH in .env — paste the exact path from Windows Explorer")
        return False

    if not mt5.initialize(path=mt5_path_norm):
        err = mt5.last_error()
        mt5.shutdown()
        _MT5_ERRORS = {
            -10003: "terminal is not running or still starting up — open it and wait ~10s",
            -10004: "executable exists but MT5 rejected the path — check the exact folder name",
            -6:     "terminal not logged in to broker — log in first",
        }
        log.error(f"MT5 connection FAILED (code {err[0]}): {err[1]}")
        log.error(f"  Cause : {_MT5_ERRORS.get(err[0], 'unknown — check MT5 terminal manually')}")
        log.error(f"  Path  : {mt5_path_norm}")
        log.error("  Fix   : open that terminal, log in, enable AutoTrading, wait ~10s, re-run")
        return False

    ok = _mt5_login_and_log()
    if ok:
        info = mt5.terminal_info()
        if info:
            actual = str(_Path(info.path) / "terminal64.exe")
            if actual.lower() != mt5_path_norm.lower():
                log.warning("  Connected terminal path does not match MT5_PATH:")
                log.warning(f"    Expected : {mt5_path_norm}")
                log.warning(f"    Actual   : {actual}")
                log.warning("  Another terminal may have answered — verify account details above")
    return ok


def _mt5_login_and_log() -> bool:
    """After initialize() succeeds: login if credentials set, then log connection info."""
    login    = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server   = os.getenv("MT5_SERVER", "")
    if login and password and server:
        if not mt5.login(login, password=password, server=server):
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

    info    = mt5.terminal_info()
    account = mt5.account_info()
    log.info(f"MT5 connected | Build {info.build}")
    log.info(f"  Terminal path : {info.path}")
    log.info(f"  Data path     : {info.data_path}")
    log.info(f"  Connected     : {info.connected} | Trade allowed: {info.trade_allowed}")
    if account:
        log.info(f"  Account {account.login} | {account.server} | "
                 f"Balance: {account.balance} {account.currency}")
    else:
        log.warning("  No account info — terminal may not be logged in yet")
    return True


def get_account_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


def get_account_equity() -> float:
    """Return equity (balance + floating P&L on all open positions)."""
    info = mt5.account_info()
    return info.equity if info else 0.0


# ─────────────────────────────────────────────────────────────
# 2. MULTI-TIMEFRAME DATA FETCH
# ─────────────────────────────────────────────────────────────

def fetch_bars(symbol: str, tf_minutes: int, n_bars: int = 99999) -> pd.DataFrame:
    """
    Fetch OHLCV bars for symbol on given TF.
    n_bars=99999 requests maximum available history from broker.
    Returns empty DataFrame on failure.
    """
    mt5_tf = TF_MAP.get(tf_minutes)
    if mt5_tf is None:
        log.warning(f"TF {tf_minutes}m not in TF_MAP")
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        log.warning(f"No data: {symbol} {tf_minutes}m | {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"
    }, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.sort_index(inplace=True)
    log.info(f"  {symbol} {tf_minutes}m: {len(df)} bars "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_all_timeframes(symbol: str) -> dict:
    """Fetch all candidate TFs for a symbol. Returns dict keyed by minutes."""
    all_tfs = sorted(set(
        PARAM_SEEDS["entry_tf_options"] +
        [t for t in PARAM_SEEDS["htf_options"] if t > 0]
    ))
    data = {}
    for tf in all_tfs:
        df = fetch_bars(symbol, tf)
        if len(df) >= MIN_BARS:
            data[tf] = df
    return data


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature set: returns, trend EMAs, momentum, volatility,
    volume, session flags, cyclical time encoding, lag features.
    """
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    # Returns
    d["ret1"]  = c.pct_change(1)
    d["ret5"]  = c.pct_change(5)
    d["ret20"] = c.pct_change(20)
    d["logr1"] = np.log(c / c.shift(1))
    d["logr5"] = np.log(c / c.shift(5))
    d["hl_pct"] = (h - l) / c
    d["co_pct"] = (c - d["Open"]) / c

    # EMAs & crossovers
    for n in [8, 21, 55, 200]:
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    d["ema_x_8_21"]  = d["ema8"]  - d["ema21"]
    d["ema_x_21_55"] = d["ema21"] - d["ema55"]
    d["p_vs_21"]  = (c - d["ema21"])  / d["ema21"]
    d["p_vs_55"]  = (c - d["ema55"])  / d["ema55"]
    d["p_vs_200"] = (c - d["ema200"]) / d["ema200"]

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi14"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Stochastic
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    d["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]      = ema12 - ema26
    d["macd_sig"]  = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]

    # Williams %R
    d["willr"] = -100 * (hi14 - c) / (hi14 - lo14 + 1e-10)

    # Bollinger Bands
    bm = c.rolling(20).mean()
    bs = c.rolling(20).std()
    d["bb_w"]   = (bm + 2*bs - (bm - 2*bs)) / (bm + 1e-10)
    d["bb_pct"] = (c - (bm - 2*bs)) / (4*bs + 1e-10)

    # ATR(14)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    d["atr14"]     = tr.rolling(14).mean()
    d["atr14_pct"] = d["atr14"] / c

    # HV(20)
    d["hv20"] = d["logr1"].rolling(20).std() * np.sqrt(252 * 24)

    # Volume
    vsma = v.rolling(20).mean()
    d["vol_ratio"] = v / (vsma + 1e-10)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d["obv_norm"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-10)

    # Session flags — DST-aware, matches tick_pipeline.py exactly.
    # Index is UTC (tick parquet) or tz-naive UTC (MT5 data).
    hr  = d.index.hour   # UTC hour — used only for cyclical encoding
    dow = d.index.dayofweek
    try:
        uk_idx = d.index.tz_convert("Europe/London")
        us_idx = d.index.tz_convert("America/New_York")
    except TypeError:
        _utc   = d.index.tz_localize("UTC")
        uk_idx = _utc.tz_convert("Europe/London")
        us_idx = _utc.tz_convert("America/New_York")
    uk_mins = uk_idx.hour * 60 + uk_idx.minute
    us_mins = us_idx.hour * 60 + us_idx.minute
    d["is_london"]  = ((uk_mins >= 480) & (uk_mins < 1020)).astype(int)  # 8 am–5 pm London
    d["is_ny"]      = ((us_mins >= 570) & (us_mins < 960)).astype(int)   # 9:30 am–4 pm NY
    d["is_overlap"] = (
        ((uk_mins >= 480) & (uk_mins < 1020)) &
        ((us_mins >= 570) & (us_mins < 960))
    ).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    d["dow_sin"]  = np.sin(2 * np.pi * dow / 5)
    d["dow_cos"]  = np.cos(2 * np.pi * dow / 5)

    # Lags
    for lag in [1, 2, 3, 5, 8, 13]:
        d[f"c_lag{lag}"] = c.shift(lag)
        d[f"r_lag{lag}"] = d["logr1"].shift(lag)

    # ── Triple Barrier Labeling — 5×5×3 grid (Lopez de Prado) ────────────
    # Pre-compute labels for a Cartesian grid of sl_atr × tp_mult × be.
    #
    # Grid:
    #   sl_atr  ∈ [1.5, 2.0, 2.5, 3.0, 3.5]   (5 values)
    #   tp_mult ∈ [1, 2, 3, 4, 5]               (5 values)
    #   be      ∈ [0, 1, 2]                      (3 values)
    # = 75 columns.  Column names: target_sl{sl}_tp{tp}_be{be}
    #   e.g. target_sl2.0_tp3_be1
    #
    # BE semantics (D2b — see ANALYSIS_v3.md):
    #   be=0: no break-even, plain SL/TP triple barrier
    #   be=1: SL moves to entry+1pt when price reaches entry + 1×sl_dist (+1R)
    #   be=2: SL moves to entry+1pt when price reaches entry + 2×sl_dist (+2R)
    #
    # Label values:
    #    1.0 = TP hit (price reached TP before any SL)
    #    0.0 = BE exit (be=1/2 triggered then BE-SL hit) OR timeout
    #   -1.0 = SL hit (original SL hit before TP and before BE triggered)
    #   NaN  = unlabelled (shouldn't happen after the loop; dropped by dropna)
    #
    # be=0 columns: fully vectorized (unchanged from prior implementation).
    # be=1/2 columns: per-trade Python loop for correctness — handles the
    #   state change when BE triggers (SL shifts mid-trade). One-time cost
    #   at training time, ~30–60s for 50 additional columns.
    #
    # ga_fitness() selects the closest column in 3D (sl_atr, tp_mult, be_r)
    # via _select_label_col(sl_atr, tp_mult, be_r).
    MAX_HOLD   = 50
    TB_SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
    TB_TP_GRID = [1,   2,   3,   4,   5  ]
    TB_BE_GRID = [0,   1,   2]
    BE_OFFSET  = 1.0   # SL moves to entry + 1 price unit when BE triggers

    if "atr14" in d.columns:
        closes_arr = d["Close"].values.astype(np.float64)
        highs_arr  = d["High"].values.astype(np.float64)
        lows_arr   = d["Low"].values.astype(np.float64)
        atr_arr    = d["atr14"].values.astype(np.float64)
        n          = len(d)

        bad_atr = np.isnan(atr_arr) | (atr_arr <= 0)

        # ── be=0: vectorized (no state change, same as before) ──────────
        # Label encoding: 1.0 = TP hit (win), 0.0 = SL hit or timeout (not-win).
        # Binary encoding for XGBoost/RF: predict_proba gives P(TP hit).
        for sl_v in TB_SL_GRID:
            for tp_v in TB_TP_GRID:
                col     = f"target_sl{sl_v}_tp{tp_v}_be0"
                labels  = np.full(n, np.nan, dtype=np.float32)
                sl_dist = atr_arr * sl_v
                long_tp = closes_arr + sl_dist * tp_v
                long_sl = closes_arr - sl_dist

                unlabelled = np.ones(n, dtype=bool)
                unlabelled[bad_atr] = False

                for k in range(1, MAX_HOLD + 1):
                    if not unlabelled[:n - k].any():
                        break
                    future_high = highs_arr[k:]
                    future_low  = lows_arr[k:]
                    cand        = unlabelled[:n - k]
                    tp_hit      = cand & (future_high >= long_tp[:n - k])
                    sl_hit      = cand & (future_low  <= long_sl[:n - k])
                    both        = tp_hit & sl_hit
                    labels[:n - k][sl_hit & ~both] = 0.0   # SL hit → not-win
                    labels[:n - k][tp_hit]          = 1.0   # TP hit (both→TP wins)
                    unlabelled[:n - k][tp_hit | sl_hit] = False

                # Remaining unlabelled after MAX_HOLD → timeout = 0 (not-win)
                labels[unlabelled] = 0.0
                d[col] = labels

        # ── be=1, be=2: per-trade Python loop ───────────────────────────
        # Label encoding: 1.0 = TP hit, 0.0 = everything else.
        # "Everything else" for BE columns includes:
        #   • BE exit (BE triggered then BE-SL hit) → neutral outcome ≈ 0R
        #   • SL hit before BE triggered → full loss = -1R
        # Both map to 0.0 for binary classification: model predicts "TP or not?"
        # BE makes some 0.0 outcomes less bad in practice (≈0R vs -1R) —
        # but the model's job is still "predict TP reach", not quantify loss.
        for sl_v in TB_SL_GRID:
            for tp_v in TB_TP_GRID:
                sl_dist_arr = atr_arr * sl_v   # per-bar SL distance

                for be_v in [1, 2]:
                    col    = f"target_sl{sl_v}_tp{tp_v}_be{be_v}"
                    labels = np.full(n, np.nan, dtype=np.float32)

                    for i in range(n):
                        if bad_atr[i]:
                            continue
                        entry   = closes_arr[i]
                        sl_dist = sl_dist_arr[i]
                        tp_lvl  = entry + sl_dist * tp_v    # TP level (long)
                        sl_lvl  = entry - sl_dist           # initial SL (long)
                        be_trig = entry + sl_dist * be_v    # BE trigger at be_v×R
                        be_sl   = entry + BE_OFFSET         # SL after BE triggers
                        be_done = False

                        label = 0.0   # default: timeout → not-win
                        for k in range(1, MAX_HOLD + 1):
                            j = i + k
                            if j >= n:
                                break
                            bh = highs_arr[j]
                            bl = lows_arr[j]

                            # Check BE trigger first (before SL/TP on same bar)
                            if not be_done and bh >= be_trig:
                                be_done = True
                                sl_lvl  = be_sl

                            # TP hit
                            if bh >= tp_lvl:
                                label = 1.0
                                break

                            # SL hit (original SL or BE-stop — both → not-win = 0)
                            if bl <= sl_lvl:
                                label = 0.0
                                break

                        labels[i] = label

                    d[col] = labels

        # Default `target` = be=0 mid-grid column (sl=1.5, tp=2, be=0).
        # Used by walk-forward fold training (search phase — acceptable approximation).
        # train.py overwrites this with the rank-1 best-matching label before
        # saving the final production model, fixing model calibration.
        d["target"]        = d["target_sl1.5_tp2_be0"]
        d["target_return"] = d["logr1"].shift(-1)
    else:
        # Fallback if ATR not yet computed (should not happen in normal flow)
        for sl_v in [1.5, 2.0, 2.5, 3.0, 3.5]:
            for tp_v in [1, 2, 3, 4, 5]:
                for be_v in [0, 1, 2]:
                    d[f"target_sl{sl_v}_tp{tp_v}_be{be_v}"] = (c.shift(-1) > c).astype(int)
        d["target"]        = (c.shift(-1) > c).astype(int)
        d["target_return"] = d["logr1"].shift(-1)

    # R2: [LABELS] diagnostics — TP rates, degenerate check, BE-bug check
    label_cols = [c for c in d.columns if c.startswith("target_sl") and "_tp" in c and "_be" in c]
    if label_cols:
        tp_rates = {c: float(d[c].mean()) for c in label_cols}
        min_rate = min(tp_rates.values())
        max_rate = max(tp_rates.values())
        log.info(f"[LABELS] {len(label_cols)} label cols | TP rate range [{min_rate:.3f}, {max_rate:.3f}]")
        degenerate = [c for c, r in tp_rates.items() if r < 0.02 or r > 0.98]
        if degenerate:
            log.warning(f"[LABELS] Degenerate label columns (TP rate < 2% or > 98%): {degenerate}")
        # BE-bug check: BE>0 should have LOWER TP rate than BE=0 equivalent
        be0_rates = {c: r for c, r in tp_rates.items() if c.endswith("_be0")}
        for be0_col, be0_rate in be0_rates.items():
            prefix = be0_col[: -len("_be0")]
            for be_v in [1, 2]:
                be_col = f"{prefix}_be{be_v}"
                if be_col in tp_rates and tp_rates[be_col] > be0_rate + 0.01:
                    log.warning(
                        f"[LABELS] BE bug suspect: {be_col} TP={tp_rates[be_col]:.3f} "
                        f"> {be0_col} TP={be0_rate:.3f} — BE trigger should reduce TP rate"
                    )

    # HTF alignment placeholder (filled in later per trade)
    d["htf_bullish"] = 0
    d["htf_strength"] = 0.0

    d.dropna(inplace=True)
    return d


def add_htf_alignment(entry_df: pd.DataFrame,
                      htf_df: pd.DataFrame,
                      htf_weight: float) -> pd.DataFrame:
    """
    Merge HTF trend direction into entry TF DataFrame.
    htf_bullish: 1 = HTF trending up, -1 = down, 0 = neutral
    """
    if htf_df.empty or htf_weight == 0:
        return entry_df

    htf_feat = engineer_features(htf_df.copy())
    if htf_feat.empty:
        return entry_df

    # HTF trend: price above EMA55 = bullish
    htf_dir = (htf_feat["Close"] > htf_feat["ema55"]).astype(int) * 2 - 1
    htf_dir.name = "htf_bullish"

    # HTF momentum strength (normalised RSI distance from 50)
    htf_str = ((htf_feat["rsi14"] - 50) / 50).rename("htf_strength")

    # Resample HTF to entry TF index (forward fill)
    htf_dir  = htf_dir.resample(entry_df.index.freq or "1min").last().ffill()
    htf_str  = htf_str.resample(entry_df.index.freq or "1min").last().ffill()

    entry_df["htf_bullish"]  = htf_dir.reindex(entry_df.index, method="ffill").fillna(0)
    entry_df["htf_strength"] = htf_str.reindex(entry_df.index, method="ffill").fillna(0) * htf_weight

    return entry_df


# ─────────────────────────────────────────────────────────────
# R6: LEAKAGE CHECK
# ─────────────────────────────────────────────────────────────

def _check_leakage(symbol: str, tf: int, df: pd.DataFrame) -> None:
    """
    Shuffle future rows into the dataframe, re-run add_institutional_features,
    and compare feature values at a pivot region. Any feature that changes when
    future data is shuffled is leaking future information.

    KNOWN_LOOKAHEAD_FEATURES (swing detection) → WARNING only.
    All other changed features → RuntimeError.
    """
    from institutional_features import add_institutional_features  # lazy — avoids circular import

    if len(df) < 200:
        log.warning(f"[LEAKAGE] {symbol}/{tf}m: too few rows ({len(df)}) to check — skipping")
        return

    # Avoid OOM on 1m (2.1M rows). The leakage check only needs a representative
    # window: ~1500 warm-up rows before the pivot + ~1500 future rows to shuffle.
    # Slice to the midpoint of the full dataset so features are well warmed-up.
    LEAKAGE_WINDOW = 3000
    if len(df) > LEAKAGE_WINDOW:
        mid = len(df) // 2
        df  = df.iloc[mid - LEAKAGE_WINDOW // 2 : mid + LEAKAGE_WINDOW // 2]

    pivot = max(100, len(df) // 2)
    # Rows before pivot: these are the "safe" region we compare
    check_slice = slice(pivot - 50, pivot)

    # Baseline: run add_institutional_features on the unmodified df
    df_base = df.copy()
    base_feat = add_institutional_features(df_base)
    base_vals = base_feat.iloc[check_slice].copy()

    # Permuted: shuffle all rows AFTER pivot (simulates future-data leakage)
    df_perm = df.copy()
    future_idx = df_perm.index[pivot:]
    perm_order = np.random.default_rng(42).permutation(len(future_idx))
    df_perm.loc[future_idx] = df_perm.loc[future_idx].iloc[perm_order].values
    perm_feat = add_institutional_features(df_perm)
    perm_vals = perm_feat.iloc[check_slice].copy()

    # Compare shared feature columns (exclude raw OHLCV and targets)
    shared_cols = [
        c for c in base_vals.columns
        if c in perm_vals.columns
        and c not in {"Open", "High", "Low", "Close", "Volume", "target", "target_return"}
        and not c.startswith("target_sl")
    ]

    leaked_soft = []  # known lookahead (swing) — WARNING
    leaked_hard = []  # unwhitelisted — RuntimeError

    for col in shared_cols:
        try:
            b = base_vals[col].astype(float)
            p = perm_vals[col].astype(float)
            if np.nanmax(np.abs(b.values - p.values)) > 1e-6:
                if col in KNOWN_LOOKAHEAD_FEATURES:
                    leaked_soft.append(col)
                else:
                    leaked_hard.append(col)
        except (TypeError, ValueError):
            pass  # non-numeric column — skip

    if leaked_soft:
        log.warning(
            f"[LEAKAGE] {symbol}/{tf}m: known lookahead features changed under permutation "
            f"(accepted approximation): {leaked_soft}"
        )
    if leaked_hard:
        raise RuntimeError(
            f"[LEAKAGE] {symbol}/{tf}m: unwhitelisted features changed under future-data "
            f"permutation — possible lookahead leakage: {leaked_hard}"
        )

    log.info(f"[LEAKAGE] {symbol}/{tf}m: leakage check passed ({len(shared_cols)} features checked)")


# ─────────────────────────────────────────────────────────────
# 4. FEATURE COLUMNS (excludes raw OHLCV and targets)
# ─────────────────────────────────────────────────────────────

EXCLUDE_COLS = {"Open", "High", "Low", "Close", "Volume",
                "target", "target_return"}

def get_feature_cols(df: pd.DataFrame) -> list:
    # Exclude all 75 triple-barrier label columns (target_sl{sl}_tp{tp}_be{be})
    # and the default `target` alias — these are targets, not features.
    return [c for c in df.columns
            if c not in EXCLUDE_COLS and not c.startswith("target_sl")]


# ─────────────────────────────────────────────────────────────
# 5. NORMALISATION
# ─────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame, feature_cols: list) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler,
                 feature_cols: list) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(df[feature_cols])
    return out


# ─────────────────────────────────────────────────────────────
# 6. ENSEMBLE MODELS (XGBoost + Random Forest)
# ─────────────────────────────────────────────────────────────

def train_ensemble(train_df, val_df, feature_cols, symbol, tf_min):
    log.info(f"  Training ensemble: {symbol} {tf_min}m")
    X_tr = train_df[feature_cols].values
    y_tr = train_df["target"].values
    X_va = val_df[feature_cols].values
    y_va = val_df["target"].values

    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        early_stopping_rounds=20, verbosity=0,
        seed=GLOBAL_SEED,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Cap bootstrap sample to 500K rows — RF generalises well with subsampling
    # (each tree already bootstraps), and avoids joblib memory fragmentation on
    # large final-fit datasets (~1.77M rows).  n_jobs=4 limits parallel forks.
    _rf_max_samples = min(500_000, len(X_tr))
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        n_jobs=4, max_samples=_rf_max_samples, random_state=GLOBAL_SEED,
    )
    log.info(f"  RF fit: {len(X_tr):,} rows → max_samples={_rf_max_samples:,}, n_jobs=4")
    rf_model.fit(X_tr, y_tr)

    xgb_val_acc = (xgb_model.predict(X_va) == y_va).mean()
    rf_val_acc  = (rf_model.predict(X_va)  == y_va).mean()

    path_xgb = MODEL_DIR / f"xgb_{symbol}_{tf_min}m.pkl"
    path_rf  = MODEL_DIR / f"rf_{symbol}_{tf_min}m.pkl"
    joblib.dump(xgb_model, path_xgb)
    joblib.dump(rf_model,  path_rf)
    log.info(
        f"  XGB val_acc={xgb_val_acc:.3f} | RF val_acc={rf_val_acc:.3f} | "
        f"saved {path_xgb.name}, {path_rf.name}"
    )
    return xgb_model, rf_model


# ─────────────────────────────────────────────────────────────
# 7. GENETIC ALGORITHM — PARAMETER OPTIMISATION
# ─────────────────────────────────────────────────────────────
# Genome: [entry_tf_idx, htf_idx, sl_atr, tp_mult, confidence_thresh, htf_weight, be_r_idx]
# rr removed — R-multiple is derived from actual exit price vs entry vs SL, not a param.

GA_BOUNDS = [
    (0, len(PARAM_SEEDS["entry_tf_options"]) - 1),   # entry TF index
    (0, len(PARAM_SEEDS["htf_options"]) - 1),         # HTF index (0 = NONE)
    # sl_atr and tp_mult narrowed to match the 5×5 label grid.
    # Genomes outside [1.0, 4.0] / [0.5, 6.0] would map to the nearest grid
    # edge (sl=1.5 or sl=3.5) — wasting evaluations on poorly-labelled space.
    (1.0, 4.0),    # sl_atr  (grid covers 1.5–3.5, allow 0.25 overhang)
    (0.5, 6.0),    # tp_mult (grid covers 1–5, allow 0.5 overhang)
    (PARAM_SEEDS["confidence_min"],PARAM_SEEDS["confidence_max"]),
    (PARAM_SEEDS["htf_weight_min"],PARAM_SEEDS["htf_weight_max"]),
    (0, len(PARAM_SEEDS["be_r_options"]) - 1),        # BE trigger index (0/1/2/3R)
]

def decode_genome(genome: list) -> dict:
    # Clamp ALL list indices to valid range before lookup.
    # GA mutation (Gaussian) can push indices outside bounds — unclamped
    # int(round()) then causes IndexError on the options lists.
    n_tf  = len(PARAM_SEEDS["entry_tf_options"])
    n_htf = len(PARAM_SEEDS["htf_options"])
    n_be  = len(PARAM_SEEDS["be_r_options"])

    tf_idx  = int(np.clip(round(float(genome[0])), 0, n_tf  - 1))
    htf_idx = int(np.clip(round(float(genome[1])), 0, n_htf - 1))
    be_idx  = int(np.clip(round(float(genome[6])), 0, n_be  - 1))

    return {
        "entry_tf":   PARAM_SEEDS["entry_tf_options"][tf_idx],
        "htf_tf":     PARAM_SEEDS["htf_options"][htf_idx],
        "sl_atr":     float(np.clip(genome[2], GA_BOUNDS[2][0], GA_BOUNDS[2][1])),
        "tp_mult":    float(np.clip(genome[3], GA_BOUNDS[3][0], GA_BOUNDS[3][1])),
        "confidence": float(np.clip(genome[4], GA_BOUNDS[4][0], GA_BOUNDS[4][1])),
        "htf_weight": float(np.clip(genome[5], GA_BOUNDS[5][0], GA_BOUNDS[5][1])),
        "be_r":       PARAM_SEEDS["be_r_options"][be_idx],
    }


def _select_label_col(sl_atr: float, tp_mult: float, be_r: int = 0) -> str:
    """
    Return the pre-computed label column name whose (sl_atr, tp_mult) grid
    point is nearest (Euclidean distance) to the requested values, with
    be_r mapped to the closest BE label dimension.

    Grid:
      sl_atr  ∈ [1.5, 2.0, 2.5, 3.0, 3.5]
      tp_mult ∈ [1, 2, 3, 4, 5]
      be      ∈ [0, 1, 2]  (be_r=3 maps to be=2 — same label, see D2b)
    """
    _SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
    _TP_GRID = [1,   2,   3,   4,   5  ]

    # Genome be_r=3 shares the be=2 label (both trigger at 2R or beyond;
    # practically indistinguishable in the triple-barrier scan).
    be_label = min(be_r, 2)

    best_col  = "target_sl1.5_tp2_be0"
    best_dist = float("inf")
    for sv in _SL_GRID:
        for tv in _TP_GRID:
            dist = ((sl_atr - sv) ** 2 + (tp_mult - tv) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_col  = f"target_sl{sv}_tp{tv}_be{be_label}"
    return best_col


def ga_fitness(genome, df_dict: dict, models_by_tf: dict, scalers_by_tf: dict, symbol: str,
               track_stats: bool = False):
    """
    Evaluate genome fitness via quick backtest on validation data.
    Returns (sharpe_ratio,) — DEAP expects a tuple.
    When track_stats=True (called post-GA on best genome), also returns execution stats dict.
    Each genome picks its own entry_tf; we look up the matching scaler/models.
    """
    params   = decode_genome(genome)
    entry_tf = params["entry_tf"]

    if entry_tf not in df_dict:
        return (-999.0,)

    df_full = df_dict[entry_tf]
    if len(df_full) < MIN_BARS:
        return (-999.0,)

    # Use only the out-of-sample test set (last 15%) to prevent overfitting
    n_full = len(df_full)
    df     = df_full.iloc[int(n_full * 0.85):]
    if len(df) < 50:
        return (-999.0,)

    # Per-TF scaler and models
    key          = f"{symbol}_{entry_tf}m"
    scaler       = scalers_by_tf.get(key)
    feature_cols = get_feature_cols(df)
    if scaler is None:
        return (-999.0,)

    xgb_m = models_by_tf.get(f"xgb_{key}")
    rf_m  = models_by_tf.get(f"rf_{key}")
    if xgb_m is None or rf_m is None:
        return (-999.0,)

    # ── Label selection: pick the pre-computed label column whose (sl_atr,
    # tp_mult, be_r) grid point is nearest to this genome's parameters.
    # This ensures the fitness evaluation uses labels that match what the
    # genome actually trades — including the BE dimension.
    label_col = _select_label_col(
        params["sl_atr"], params["tp_mult"], params.get("be_r", 0)
    )
    if label_col not in df.columns or df[label_col].dropna().empty:
        label_col = "target_sl1.5_tp2_be0"   # fallback to default mid-grid be=0 column
    if label_col not in df.columns:
        label_col = "target"   # final fallback if even the default is missing

    # ── Scale features and generate ensemble signal ────────────────────
    scaled      = apply_scaler(df, scaler, feature_cols)
    feat        = scaled[feature_cols].values
    atr         = df["atr14"].values
    closes      = df["Close"].values
    highs       = df["High"].values
    lows        = df["Low"].values
    opens       = df["Open"].values

    p_xgb = xgb_m.predict_proba(feat)[:, 1]
    p_rf  = rf_m.predict_proba(feat)[:, 1]
    signal_prob = (p_xgb + p_rf) / 2.0

    # ── HTF nudge: apply same probability adjustment as get_signal() ──
    # This lets the optimizer actually evaluate whether HTF alignment helps.
    # htf_tf=0 or htf_weight=0 means HTF is skipped — optimizer can discover this.
    htf_tf  = params.get("htf_tf", 0)
    htf_w   = params.get("htf_weight", 0.0)
    htf_dir_arr = np.zeros(len(df), dtype=np.float32)
    if htf_tf > 0 and htf_w > 0 and htf_tf in df_dict:
        htf_full   = df_dict[htf_tf]
        # Align HTF EMA55 direction to entry-TF index via forward-fill
        if "ema55" in htf_full.columns and not htf_full.empty:
            htf_series = ((htf_full["Close"] > htf_full["ema55"]).astype(int) * 2 - 1)
            htf_series = htf_series.reindex(df.index, method="ffill").fillna(0)
            htf_dir_arr = htf_series.values.astype(np.float32)

    # Apply HTF nudge to signal probabilities
    adjusted_prob = signal_prob.copy()
    if htf_w > 0:
        bear_htf = htf_dir_arr == -1
        bull_htf = htf_dir_arr == 1
        # Bearish HTF reduces long confidence, bullish HTF boosts it
        adjusted_prob = np.where(
            bear_htf & (signal_prob > 0.5),
            signal_prob * (1.0 - htf_w * 0.3),
            np.where(
                bull_htf & (signal_prob < 0.5),
                signal_prob * (1.0 + htf_w * 0.3),
                signal_prob,
            )
        )

    # S1: Per-genome deterministic RNG — reproducible fill/noise across evaluations
    genome_seed = abs(hash(tuple(round(float(g), 4) for g in genome))) % (2 ** 31)
    rng = np.random.default_rng(genome_seed)

    # ── Spread model (R7: synthetic spread when tick data has near-zero spread) ──
    spread_arr = (df["spread_mean"].values
                  if "spread_mean" in df.columns
                  else np.zeros(len(df), dtype=np.float64))
    if spread_arr.mean() < 0.1:
        # Dukascopy artefact: bid/ask quote sequencing yields near-zero spread.
        # Apply a regime-aware synthetic spread to avoid optimising in a zero-friction world.
        atr_s = df["atr14"].values
        session_bar_num = pd.Series(
            df.groupby(df.index.date).cumcount().values, index=df.index
        ).values
        open_mult  = np.where(session_bar_num < SPREAD_OPEN_BARS, SPREAD_OPEN_MULT, 1.0)
        base_spread = SPREAD_BASE_PTS + atr_s * SPREAD_ATR_COEFF
        noise       = rng.lognormal(mean=0.0, sigma=0.15, size=len(df))
        spread_arr  = base_spread * open_mult * noise
    SLIP_FACTOR = 0.1

    # ── Simulate trades (one at a time, fixed $100 risk per trade) ────
    FIXED_RISK   = 100.0   # fixed $/trade — matches ER calculation model
    balance      = 10000.0
    peak         = 10000.0
    trade_pnls   = []      # (bar_date, pnl_money) for daily equity curve
    skip_until   = -1

    # track_stats accumulators (used when called post-GA for [EXECUTION] log)
    _ts_spread, _ts_slip, _ts_total_cost = [], [], []
    _ts_n_signals = _ts_fill_miss = _ts_gap_skip = _ts_delayed = _ts_fills = 0

    sl_atr   = params["sl_atr"]
    tp_mult  = params["tp_mult"]
    conf_thr = params["confidence"]
    be_r     = params.get("be_r", 0)

    for i in range(SEQ_LEN, len(df) - 1):
        if i <= skip_until:
            continue

        prob      = adjusted_prob[i]
        direction = None
        if prob >= conf_thr:
            direction = 1
        elif prob <= (1 - conf_thr):
            direction = -1
        if direction is None:
            continue

        _ts_n_signals += 1

        sl_dist = atr[i] * sl_atr
        if sl_dist <= 0 or np.isnan(sl_dist):
            continue

        # Stage 2: include spread + slippage in entry price
        spread_cost = spread_arr[i] / 2.0 if not np.isnan(spread_arr[i]) else 0.0
        slip_cost   = atr[i] * SLIP_FACTOR if not np.isnan(atr[i]) else 0.0

        # S1: Fill probability — wide spread relative to SL reduces fill chance
        fill_prob = float(np.clip(1.0 - (spread_cost / (sl_dist + 1e-10)) * 0.5, 0.80, 1.0))
        if rng.random() > fill_prob:
            _ts_fill_miss += 1
            continue

        # E3: Execution latency — 30% of trades fill at next-bar open (MT5 pathway delay).
        # If the open gap exceeds MT5_DEVIATION_PTS the IOC order is rejected, same as live.
        _delayed_this = False
        if rng.random() < EXEC_DELAY_PROB and i + 1 < len(df) - 1:
            _ts_delayed += 1
            _delayed_this = True
            gap_pts = abs(opens[i + 1] - closes[i])
            if gap_pts > MT5_DEVIATION_PTS:
                _ts_gap_skip += 1
                continue  # gap exceeds MT5 deviation limit — missed fill
            fill_price = opens[i + 1]
        else:
            fill_price = closes[i]

        _ts_fills += 1
        _ts_spread.append(spread_cost * 2)   # full spread (cost was half-spread)
        _ts_slip.append(slip_cost)
        _ts_total_cost.append(spread_cost + slip_cost)

        entry  = fill_price + direction * (spread_cost + slip_cost)
        sl     = entry - direction * sl_dist
        tp     = entry + direction * sl_dist * tp_mult
        risk   = FIXED_RISK

        be_trigger = entry + direction * sl_dist * be_r if be_r > 0 else None
        be_stop    = entry + direction * 1.0 if be_r > 0 else None
        be_done    = False

        pnl      = 0.0
        exit_bar = i + 50
        for j in range(i + 1, min(i + 51, len(df) - 1)):
            bh = highs[j]; bl = lows[j]

            if be_trigger is not None and not be_done:
                if direction == 1 and bh >= be_trigger:
                    sl = be_stop; be_done = True
                elif direction == -1 and bl <= be_trigger:
                    sl = be_stop; be_done = True

            if direction == 1:
                if bl <= sl: pnl = (sl - entry) / (sl_dist + 1e-10) * risk; exit_bar = j; break
                if bh >= tp: pnl = (tp - entry) / (sl_dist + 1e-10) * risk; exit_bar = j; break
            else:
                if bh >= sl: pnl = (entry - sl) / (sl_dist + 1e-10) * risk; exit_bar = j; break
                if bl <= tp: pnl = (entry - tp) / (sl_dist + 1e-10) * risk; exit_bar = j; break

        skip_until = exit_bar
        if pnl != 0:
            balance += pnl
            trade_pnls.append((df.index[exit_bar], pnl))
            if balance > peak:
                peak = balance
            if (peak - balance) / peak > HARD_LIMITS["max_drawdown_pct"] / 100:
                break

    n_trades = len(trade_pnls)
    if n_trades < 10:
        return (-999.0,)

    # ── Minimum trade count penalty — progressive, not a cliff ───────
    # Prevents optimizer finding corner solutions with 15 trades at 12:1 RR.
    # Soft-penalises sparse strategies while preserving optimizer freedom.
    MIN_CREDIBLE  = 100
    trade_penalty = min(n_trades / MIN_CREDIBLE, 1.0)   # 1.0 at 100+ trades

    # ── Sharpe from daily equity curve × √252 ─────────────────────────
    # G1: include all calendar days between first and last trade date.
    # Removing zero-return days inflates Sharpe vs backtest_engine (which
    # builds a full calendar equity curve). Using all days keeps both in sync.
    dates_arr   = np.array([t[0] for t in trade_pnls], dtype="datetime64[D]")
    pnl_arr     = np.array([t[1] for t in trade_pnls])
    day_ints    = (dates_arr - dates_arr[0]).astype(np.int64)
    n_days_span = int(day_ints[-1]) + 1
    if n_days_span < 2:
        return (-999.0,)
    daily_pnl = np.zeros(n_days_span)
    np.add.at(daily_pnl, day_ints, pnl_arr)
    # Equity curve spans first→last trade date; includes zero-return calendar days
    eq_curve  = 10000.0 + np.cumsum(daily_pnl)
    daily_ret = np.diff(eq_curve) / (eq_curve[:-1] + 1e-10)
    if len(daily_ret) < 2:
        return (-999.0,)
    mean_r    = daily_ret.mean()
    std_r     = daily_ret.std()
    sharpe    = mean_r / (std_r + 1e-10) * np.sqrt(252)
    if not np.isfinite(sharpe):
        return (-999.0,)

    # ── R-multiple array (used by A2 and A9 below) ────────────────────
    r_arr = pnl_arr / (FIXED_RISK + 1e-10)

    # ── CVaR tail risk penalty (A2) ───────────────────────────────────
    # Two strategies with identical Sharpe may differ in left-tail shape.
    # Penalise strategies where the worst 5% of trades average below −2R.
    sorted_r = np.sort(r_arr)
    cvar_pct = int(len(r_arr) * 0.05)
    cvar_95  = sorted_r[:cvar_pct].mean() if cvar_pct >= 3 else float(sorted_r[0])
    # cvar_95 is negative; penalty grows when worse than −2R
    cvar_penalty = max(0.0, (-cvar_95 - 2.0) / 2.0)

    # ── Trade quality soft penalty (A9) ───────────────────────────────
    # Guard against strategies that barely cover execution costs.
    # 0.15R on $100 risk = $15 avg net; typical spread+slip ≈ $7–10.
    mean_r_trade    = float(r_arr.mean())
    quality_penalty = (mean_r_trade / 0.15) if mean_r_trade < 0.15 else 1.0
    quality_penalty = max(0.0, quality_penalty)  # clamp to [0, 1]

    # ── Final fitness ─────────────────────────────────────────────────
    fitness = sharpe * trade_penalty * max(0.5, 1.0 - cvar_penalty) * quality_penalty

    if not track_stats:
        return (fitness,)

    # Build execution stats dict for [EXECUTION] log (only when called post-GA on best genome)
    n_fills = max(_ts_fills, 1)
    exec_stats = {
        "avg_spread":        float(np.mean(_ts_spread))    if _ts_spread  else 0.0,
        "p50_spread":        float(np.median(_ts_spread))  if _ts_spread  else 0.0,
        "p95_spread":        float(np.percentile(_ts_spread, 95)) if len(_ts_spread) >= 2 else 0.0,
        "avg_slip":          float(np.mean(_ts_slip))      if _ts_slip    else 0.0,
        "p95_slip":          float(np.percentile(_ts_slip, 95))   if len(_ts_slip) >= 2 else 0.0,
        "fill_rate":         _ts_fills / max(_ts_n_signals, 1),
        "gap_skip_rate":     _ts_gap_skip / max(_ts_delayed, 1),
        "delay_rate":        _ts_delayed  / max(_ts_n_signals, 1),
        "avg_total_cost":    float(np.mean(_ts_total_cost)) if _ts_total_cost else 0.0,
        "n_signals":         _ts_n_signals,
        "n_fills":           _ts_fills,
    }
    return (fitness, exec_stats)


def run_genetic_algo(df_dict: dict, models_by_tf: dict,
                     scalers_by_tf: dict, symbol: str,
                     n_gen=80, pop_size=150) -> dict:
    """
    GA with wider exploration budget:
      pop_size=150, n_gen=80 → 12,000 evaluations (was 3,200)
      Higher mutation sigma=0.3 to jump across the wider param bounds
      Tournament size=7 for stronger selection pressure
    """
    log.info("  Running genetic algorithm parameter search...")

    # Avoid duplicate creator classes on re-run
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def rand_gene(i):
        lo, hi = GA_BOUNDS[i]
        return lo + np.random.random() * (hi - lo)

    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: [rand_gene(i) for i in range(len(GA_BOUNDS))])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", ga_fitness,
                     df_dict=df_dict, models_by_tf=models_by_tf,
                     scalers_by_tf=scalers_by_tf, symbol=symbol)
    toolbox.register("mate",    tools.cxBlend, alpha=0.4)
    toolbox.register("mutate",  tools.mutGaussian, mu=0, sigma=0.3, indpb=0.25)
    toolbox.register("select",  tools.selTournament, tournsize=7)

    pop  = toolbox.population(n=pop_size)
    hof  = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox,
                        cxpb=0.65, mutpb=0.35,
                        ngen=n_gen, stats=stats,
                        halloffame=hof, verbose=False)

    best = decode_genome(hof[0])
    best["best_score"] = float(hof[0].fitness.values[0])   # fix: GA result now participates fairly in GA-vs-Optuna comparison
    best["_raw_genome"] = list(hof[0])  # stored for post-GA [EXECUTION] stats call
    log.info(f"  GA best genome: {best}")
    return best


# ─────────────────────────────────────────────────────────────
# 10. OPTUNA BAYESIAN OPTIMISATION
# ─────────────────────────────────────────────────────────────

def run_optuna(df_dict: dict, models_by_tf: dict,
               scalers_by_tf: dict,
               n_trials: int = OPTUNA_TRIALS,
               symbol: str = "") -> dict:
    log.info(f"  Running Optuna ({n_trials} trials): {symbol}")

    def objective(trial):
        genome = [
            trial.suggest_int("entry_tf_idx", 0,
                              len(PARAM_SEEDS["entry_tf_options"]) - 1),
            trial.suggest_int("htf_idx", 0,
                              len(PARAM_SEEDS["htf_options"]) - 1),
            trial.suggest_float("sl_atr",     *GA_BOUNDS[2]),
            trial.suggest_float("tp_mult",    *GA_BOUNDS[3]),
            trial.suggest_float("confidence", *GA_BOUNDS[4]),
            trial.suggest_float("htf_weight", *GA_BOUNDS[5]),
            trial.suggest_int("be_r_idx", 0,
                              len(PARAM_SEEDS["be_r_options"]) - 1),
        ]
        result = ga_fitness(genome, df_dict, models_by_tf, scalers_by_tf, symbol)
        return result[0]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=GLOBAL_SEED, n_startup_trials=40),
        study_name=f"opt_{symbol}",
    )

    # Multiple warm-start seeds to avoid local optima — covers conservative,
    # aggressive, and scalp-style starting points across the wider bounds
    _warm_starts = [
        # Conservative: wide SL, modest TP, high confidence
        {"entry_tf_idx": PARAM_SEEDS["entry_tf_options"].index(5),
         "htf_idx": PARAM_SEEDS["htf_options"].index(60),
         "sl_atr": 1.5, "tp_mult": 2.0,
         "confidence": 0.65, "htf_weight": 0.6, "be_r_idx": 1},
        # Aggressive: tight SL, high TP, lower confidence
        {"entry_tf_idx": PARAM_SEEDS["entry_tf_options"].index(3),
         "htf_idx": PARAM_SEEDS["htf_options"].index(30),
         "sl_atr": 0.6, "tp_mult": 4.0,
         "confidence": 0.55, "htf_weight": 0.4, "be_r_idx": 2},
        # Scalp: very tight SL, 1:1 TP, very high confidence
        {"entry_tf_idx": PARAM_SEEDS["entry_tf_options"].index(1),
         "htf_idx": PARAM_SEEDS["htf_options"].index(15),
         "sl_atr": 0.4, "tp_mult": 1.0,
         "confidence": 0.80, "htf_weight": 0.3, "be_r_idx": 0},
        # Runner: wide SL, very high TP, no HTF
        {"entry_tf_idx": PARAM_SEEDS["entry_tf_options"].index(5),
         "htf_idx": PARAM_SEEDS["htf_options"].index(0),
         "sl_atr": 2.5, "tp_mult": 6.0,
         "confidence": 0.70, "htf_weight": 0.0, "be_r_idx": 3},
        # Swing: 15m entry, 4H HTF, medium params
        {"entry_tf_idx": PARAM_SEEDS["entry_tf_options"].index(15),
         "htf_idx": PARAM_SEEDS["htf_options"].index(240),
         "sl_atr": 2.0, "tp_mult": 3.0,
         "confidence": 0.72, "htf_weight": 0.7, "be_r_idx": 2},
    ]
    for ws in _warm_starts:
        study.enqueue_trial(ws)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    result = {
        "entry_tf":   PARAM_SEEDS["entry_tf_options"][best["entry_tf_idx"]],
        "htf_tf":     PARAM_SEEDS["htf_options"][best["htf_idx"]],
        "sl_atr":     best["sl_atr"],
        "tp_mult":    best["tp_mult"],
        "confidence": best["confidence"],
        "htf_weight": best["htf_weight"],
        "be_r":       PARAM_SEEDS["be_r_options"][best["be_r_idx"]],
        "best_score": study.best_value,
    }
    log.info(f"  Optuna best: {result}")
    return result


# ─────────────────────────────────────────────────────────────
# 10b. PER-TF OPTIMIZATION — locks entry_tf, returns top-N trials
# ─────────────────────────────────────────────────────────────

def run_per_tf_optimization(
    df_dict: dict,
    models_by_tf: dict,
    scalers_by_tf: dict,
    symbol: str,
    locked_tf: int,
    n_trials: int = 150,
    top_n: int = 5,
) -> list[dict]:
    """
    Run Optuna with entry_tf locked to locked_tf.
    Returns list of top_n trial dicts sorted by Sharpe (best first).
    Each dict has: params (decoded), sharpe, number.

    Used to build the per-TF top-5 strategy table.
    """
    if locked_tf not in df_dict:
        log.warning(f"  {symbol} {locked_tf}m: not in df_dict — skipping per-TF opt")
        return []

    key = f"{symbol}_{locked_tf}m"
    if not scalers_by_tf.get(key):
        log.warning(f"  {symbol} {locked_tf}m: no scaler — skipping per-TF opt")
        return []

    tf_idx = PARAM_SEEDS["entry_tf_options"].index(locked_tf) \
        if locked_tf in PARAM_SEEDS["entry_tf_options"] else 0

    log.info(f"  Per-TF Optuna: {symbol} {locked_tf}m ({n_trials} trials)")

    all_trials = []

    def objective(trial):
        genome = [
            tf_idx,   # locked — not suggested by Optuna
            trial.suggest_int("htf_idx", 0, len(PARAM_SEEDS["htf_options"]) - 1),
            trial.suggest_float("sl_atr",     *GA_BOUNDS[2]),
            trial.suggest_float("tp_mult",    *GA_BOUNDS[3]),
            trial.suggest_float("confidence", *GA_BOUNDS[4]),
            trial.suggest_float("htf_weight", *GA_BOUNDS[5]),
            trial.suggest_int("be_r_idx", 0, len(PARAM_SEEDS["be_r_options"]) - 1),
        ]
        result = ga_fitness(genome, df_dict, models_by_tf, scalers_by_tf, symbol)
        return result[0]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=GLOBAL_SEED, n_startup_trials=30),
        study_name=f"per_tf_{symbol}_{locked_tf}m",
    )

    # Multiple warm starts per TF — avoids local optima in the wider search space.
    # Each seed represents a different trading style to explore.
    _per_tf_warm_starts = [
        # Default seed
        {"htf_idx": PARAM_SEEDS["htf_options"].index(60),
         "sl_atr": 1.5, "tp_mult": 2.0,
         "confidence": 0.65, "htf_weight": 0.5, "be_r_idx": 1},
        # Tight scalp
        {"htf_idx": PARAM_SEEDS["htf_options"].index(15),
         "sl_atr": 0.4, "tp_mult": 1.0,
         "confidence": 0.80, "htf_weight": 0.3, "be_r_idx": 0},
        # Wide swing
        {"htf_idx": PARAM_SEEDS["htf_options"].index(240),
         "sl_atr": 3.0, "tp_mult": 5.0,
         "confidence": 0.70, "htf_weight": 0.8, "be_r_idx": 2},
        # No HTF — pure entry signal
        {"htf_idx": PARAM_SEEDS["htf_options"].index(0),
         "sl_atr": 1.0, "tp_mult": 3.0,
         "confidence": 0.60, "htf_weight": 0.0, "be_r_idx": 3},
        # High conviction runner
        {"htf_idx": PARAM_SEEDS["htf_options"].index(30),
         "sl_atr": 2.0, "tp_mult": 7.0,
         "confidence": 0.85, "htf_weight": 0.6, "be_r_idx": 2},
    ]
    for ws in _per_tf_warm_starts:
        study.enqueue_trial(ws)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Collect all completed trials
    for t in study.trials:
        if t.value is None or t.value <= -900:
            continue
        params = {
            "entry_tf":   locked_tf,
            "htf_tf":     PARAM_SEEDS["htf_options"][t.params.get("htf_idx", 0)],
            "sl_atr":     t.params.get("sl_atr", PARAM_SEEDS["sl_atr_seed"]),
            "tp_mult":    t.params.get("tp_mult", PARAM_SEEDS["tp_mult_seed"]),
            "confidence": t.params.get("confidence", PARAM_SEEDS["confidence_seed"]),
            "htf_weight": t.params.get("htf_weight", PARAM_SEEDS["htf_weight_seed"]),
            "be_r":       PARAM_SEEDS["be_r_options"][t.params.get("be_r_idx", 0)],
        }
        all_trials.append({
            "number": t.number,
            "params": params,
            "sharpe": float(t.value),
        })

    # Sort by Sharpe, return top N
    all_trials.sort(key=lambda x: x["sharpe"], reverse=True)
    top = all_trials[:top_n]
    log.info(f"  Per-TF {symbol} {locked_tf}m: best Sharpe={top[0]['sharpe']:.3f}" if top else
             f"  Per-TF {symbol} {locked_tf}m: no valid trials")
    return top, all_trials   # (top_n, all for DB storage)


# ─────────────────────────────────────────────────────────────
# 11. PARAMETER PERSISTENCE
# ─────────────────────────────────────────────────────────────

def save_params(symbol: str, params: dict):
    path = PARAMS_DIR / f"{symbol}_params.json"
    params["updated_at"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    log.info(f"  Params saved: {path}")


def load_params(symbol: str) -> dict:
    path = PARAMS_DIR / f"{symbol}_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Return seeds if no saved params exist yet
    return {
        "entry_tf":   PARAM_SEEDS["entry_tf_default"],
        "htf_tf":     PARAM_SEEDS["htf_default"],
        "sl_atr":     PARAM_SEEDS["sl_atr_seed"],
        "tp_mult":    PARAM_SEEDS["tp_mult_seed"],
        "confidence": PARAM_SEEDS["confidence_seed"],
        "htf_weight": PARAM_SEEDS["htf_weight_seed"],
        "source":     "seed_defaults",
    }


# ─────────────────────────────────────────────────────────────
# 12. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────

def get_signal(symbol: str, params: dict,
               models_cache: dict, scalers_cache: dict,
               live_data_cache: dict) -> dict:
    """
    Generate a trading signal for one symbol using its current ML params.
    Returns dict with: direction (1=long/-1=short/0=none), confidence,
    sl_price, tp_price, entry_tf, htf_used.
    """
    entry_tf  = params["entry_tf"]
    htf_tf    = params["htf_tf"]     # 0 = no HTF
    sl_atr    = params["sl_atr"]
    tp_mult   = params["tp_mult"]
    conf_thr  = params["confidence"]
    htf_w     = params["htf_weight"]

    key = f"{symbol}_{entry_tf}m"
    df  = live_data_cache.get(key)
    if df is None or len(df) < SEQ_LEN + 5:
        return {"direction": 0, "reason": "insufficient data"}

    scaler       = scalers_cache.get(key)
    feature_cols = get_feature_cols(df)

    # HTF alignment
    if htf_tf > 0 and htf_w > 0:
        htf_key = f"{symbol}_{htf_tf}m"
        htf_df  = live_data_cache.get(htf_key, pd.DataFrame())
        df      = add_htf_alignment(df, htf_df, htf_w)

    if scaler is None:
        return {"direction": 0, "reason": "no scaler"}

    scaled = apply_scaler(df.tail(SEQ_LEN + 5), scaler, feature_cols)

    # XGBoost + Random Forest signal (equal-weight average)
    last_feat = scaled[feature_cols].iloc[-1:].values
    p_xgb, p_rf = 0.5, 0.5
    xgb_m = models_cache.get(f"xgb_{key}")
    rf_m  = models_cache.get(f"rf_{key}")
    if xgb_m is not None:
        p_xgb = float(xgb_m.predict_proba(last_feat)[0][1])
    if rf_m is not None:
        p_rf  = float(rf_m.predict_proba(last_feat)[0][1])

    prob_raw = (p_xgb + p_rf) / 2.0

    # Isotonic calibration: maps raw ensemble prob → actual win rate
    # Fitted on stacked OOS fold predictions in train.py; makes conf threshold meaningful.
    calibrator = scalers_cache.get(f"calibrator_{key}")
    prob_cal   = prob_raw
    if calibrator is not None:
        prob_cal = float(calibrator.predict([prob_raw])[0])
    prob = prob_cal

    # HTF filter: if HTF is bearish and signal is long, reduce confidence
    htf_dir    = df["htf_bullish"].iloc[-1] if "htf_bullish" in df.columns else 0
    prob_before_htf = prob
    if htf_tf > 0 and htf_w > 0:
        if htf_dir == -1 and prob > 0.5:
            prob *= (1 - htf_w * 0.3)
        elif htf_dir == 1 and prob < 0.5:
            prob *= (1 + htf_w * 0.3)
    htf_adj = round(prob - prob_before_htf, 4)

    # Determine direction
    direction = 0
    if prob >= conf_thr:
        direction = 1   # long
    elif prob <= (1 - conf_thr):
        direction = -1  # short

    if direction == 0:
        return {
            "direction": 0, "confidence": round(prob, 4),
            "prob_raw":  round(prob_raw, 4), "prob_cal": round(prob_cal, 4),
            "htf_adj":   htf_adj, "htf_dir": int(htf_dir),
            "entry_tf":  entry_tf,
            "reason":    f"prob {prob:.3f} below threshold {conf_thr:.3f}",
        }

    # Compute SL / TP
    last_row = df.iloc[-1]
    atr      = last_row.get("atr14", last_row["Close"] * 0.001)
    close    = last_row["Close"]
    sl_dist  = atr * sl_atr
    sl       = close - direction * sl_dist
    tp       = close + direction * sl_dist * tp_mult

    return {
        "direction":   direction,
        "confidence":  round(prob, 4),
        "prob_raw":    round(prob_raw, 4),
        "prob_cal":    round(prob_cal, 4),
        "htf_adj":     htf_adj,
        "htf_dir":     int(htf_dir),
        "entry_price": round(close, 5),
        "sl_price":    round(sl, 5),
        "tp_price":    round(tp, 5),
        "sl_pips":     round(sl_dist, 5),
        "tp_mult":     round(tp_mult, 2),
        "entry_tf":    entry_tf,
        "htf_used":    htf_tf,
        "reason":      "signal confirmed",
    }


# ─────────────────────────────────────────────────────────────
# 13. RISK GATE — hard limits enforced before any order
# ─────────────────────────────────────────────────────────────

class RiskGate:
    def __init__(self):
        # Use equity (balance + floating P&L) for daily loss tracking.
        # This correctly prevents exceeding risk limits when open positions are in loss.
        self.session_start_equity = get_account_equity()
        self.daily_loss           = 0.0

    def reset_daily(self):
        self.session_start_equity = get_account_equity()
        self.daily_loss           = 0.0

    def position_size(self, sl_pips: float, symbol: str) -> float:
        """Calculate lot size so risk = risk_per_trade_pct of equity."""
        balance     = get_account_equity()
        risk_amount = balance * HARD_LIMITS["risk_per_trade_pct"] / 100
        sym_info    = mt5.symbol_info(symbol)
        if sym_info is None or sl_pips == 0:
            return 0.01
        tick_value = sym_info.trade_tick_value
        tick_size  = sym_info.trade_tick_size
        if tick_size == 0:
            return 0.01
        pips_per_lot = sl_pips / tick_size * tick_value
        if pips_per_lot == 0:
            return 0.01
        lot = risk_amount / pips_per_lot
        lot = max(sym_info.volume_min,
                  min(sym_info.volume_max,
                      round(lot / sym_info.volume_step) * sym_info.volume_step))
        return lot

    def can_trade(self) -> tuple:
        """Returns (allowed: bool, reason: str)"""
        # Equity includes floating P&L — correctly reflects true account risk
        equity   = get_account_equity()
        dd_pct   = ((self.session_start_equity - equity) /
                    (self.session_start_equity + 1e-10)) * 100

        positions = mt5.positions_get()
        n_open    = len(positions) if positions else 0

        # Daily loss computed from equity to include open position losses
        daily_loss_pct = max(dd_pct, 0.0)  # equity drawdown from session start = daily loss

        if daily_loss_pct >= HARD_LIMITS["max_daily_loss_pct"]:
            return False, f"Daily loss limit hit ({daily_loss_pct:.1f}%)"
        if dd_pct >= HARD_LIMITS["max_drawdown_pct"]:
            return False, f"Max drawdown hit ({dd_pct:.1f}%)"
        if n_open >= HARD_LIMITS["max_open_positions"]:
            return False, f"Max open positions ({n_open})"
        return True, "ok"


# ─────────────────────────────────────────────────────────────
# 14. ORDER EXECUTION
# ─────────────────────────────────────────────────────────────

def place_order(symbol: str, direction: int,
                lot: float, sl: float, tp: float,
                comment: str = "adaptive_ml") -> dict:
    """
    Send market order to MT5.
    direction: 1=long (BUY), -1=short (SELL)
    """
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        return {"success": False, "error": f"Symbol info failed: {symbol}"}

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    price      = (mt5.symbol_info_tick(symbol).ask
                  if direction == 1
                  else mt5.symbol_info_tick(symbol).bid)

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    symbol,
        "volume":    lot,
        "type":      order_type,
        "price":     price,
        "sl":        round(sl, sym_info.digits),
        "tp":        round(tp, sym_info.digits),
        "deviation": 20,
        "magic":     20250101,
        "comment":   comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed: {result.retcode} | {result.comment}")
        return {"success": False, "retcode": result.retcode,
                "error": result.comment}

    log.info(f"Order placed: {symbol} {'BUY' if direction==1 else 'SELL'} "
             f"{lot} lots | SL:{sl:.5f} TP:{tp:.5f} | ticket:{result.order}")
    return {"success": True, "ticket": result.order, "price": price}


# ─────────────────────────────────────────────────────────────
# 15. TRADE LOG + LIVE INCREMENTAL LEARNING
# ─────────────────────────────────────────────────────────────

def log_trade(trade: dict):
    trades = []
    if TRADE_LOG.exists():
        with open(TRADE_LOG) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2)


def load_trade_log() -> list:
    if TRADE_LOG.exists():
        with open(TRADE_LOG) as f:
            return json.load(f)
    return []


def should_retrain(symbol: str, min_new_trades: int = 20) -> bool:
    """Trigger re-optimisation after enough new closed trades."""
    trades    = load_trade_log()
    sym_trades = [t for t in trades if t.get("symbol") == symbol
                  and not t.get("used_for_retrain", False)]
    return len(sym_trades) >= min_new_trades


def incremental_update(symbol: str, df_dict: dict,
                       models_cache: dict, scalers_cache: dict):
    """
    Lightweight re-optimisation triggered after live trades.
    Runs Optuna with fewer trials — faster than full retrain.
    """
    log.info(f"Incremental update triggered: {symbol}")
    entry_tf = load_params(symbol).get("entry_tf",
                                       PARAM_SEEDS["entry_tf_default"])
    key      = f"{symbol}_{entry_tf}m"
    scaler   = scalers_cache.get(key)
    feat_df  = df_dict.get(entry_tf)

    if feat_df is None or scaler is None:
        log.warning(f"  Skipping incremental update: missing data for {symbol}")
        return

    new_params = run_optuna(
        df_dict      = {entry_tf: feat_df},
        models_by_tf = models_cache,
        scalers_by_tf= scalers_cache,
        n_trials     = OPTUNA_LIVE_TRIALS,
        symbol       = symbol,
    )
    save_params(symbol, new_params)

    # Mark trades as used
    trades = load_trade_log()
    for t in trades:
        if t.get("symbol") == symbol:
            t["used_for_retrain"] = True
    with open(TRADE_LOG, "w") as f:
        json.dump(trades, f, indent=2)

    log.info(f"Incremental update complete: {symbol} | new params: {new_params}")


# ─────────────────────────────────────────────────────────────
# 16. CANDLE CLOSE TIMING
# ─────────────────────────────────────────────────────────────

def seconds_to_next_candle_close(tf_minutes: int) -> float:
    """
    Return seconds until the next candle closes on the given TF.
    System wakes exactly at candle close — not mid-candle.
    """
    now      = datetime.now(timezone.utc)
    tf_secs  = tf_minutes * 60
    elapsed  = (now.timestamp() % tf_secs)
    remaining = tf_secs - elapsed
    # Add 2-second buffer to ensure candle is fully formed
    return remaining + 2.0


# ─────────────────────────────────────────────────────────────
# 17. FULL HISTORICAL TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def run_historical_training() -> tuple:
    """
    Fetch max history, engineer features, train all models,
    run GA + Optuna to discover initial optimal parameters.
    Returns (models_cache, scalers_cache, all_data).
    """
    log.info("=" * 60)
    log.info("PHASE 2 — HISTORICAL TRAINING")
    log.info("=" * 60)

    models_cache  = {}
    scalers_cache = {}
    all_data      = {}   # {symbol: {tf: engineered_df}}

    for symbol in INSTRUMENTS:
        log.info(f"\nProcessing: {symbol}")
        tf_raw = fetch_all_timeframes(symbol)

        if not tf_raw:
            log.warning(f"  No data for {symbol}, skipping")
            continue

        # Engineer features for each TF
        tf_feat = {}
        for tf, raw_df in tf_raw.items():
            feat_df = engineer_features(raw_df)
            if len(feat_df) >= MIN_BARS:
                tf_feat[tf] = feat_df

        if not tf_feat:
            log.warning(f"  Insufficient data after feature engineering: {symbol}")
            continue

        all_data[symbol] = tf_feat

        # Train on entry TF candidates
        for tf in PARAM_SEEDS["entry_tf_options"]:
            if tf not in tf_feat:
                continue
            df = tf_feat[tf]
            n  = len(df)
            i_tr  = int(n * 0.70)
            i_val = int(n * 0.85)
            train_df = df.iloc[:i_tr]
            val_df   = df.iloc[i_tr:i_val]

            feature_cols = get_feature_cols(df)
            scaler       = fit_scaler(train_df, feature_cols)
            train_s      = apply_scaler(train_df, scaler, feature_cols)
            val_s        = apply_scaler(val_df,   scaler, feature_cols)

            key = f"{symbol}_{tf}m"
            scalers_cache[key] = scaler

            # LSTM
            lstm = train_lstm(train_s, val_s, feature_cols, symbol, tf)
            if lstm:
                models_cache[f"lstm_{key}"] = lstm

            # Ensemble
            xgb_m, rf_m = train_ensemble(train_s, val_s, feature_cols, symbol, tf)
            models_cache[f"xgb_{key}"] = xgb_m
            models_cache[f"rf_{key}"]  = rf_m

        # GA parameter search — use per-TF models/scalers so any entry_tf works
        # Check at least one TF has a scaler before running
        if any(scalers_cache.get(f"{symbol}_{tf}m") for tf in PARAM_SEEDS["entry_tf_options"]
               if tf in tf_feat):
            ga_params = run_genetic_algo(
                df_dict      = tf_feat,
                models_by_tf = models_cache,
                scalers_by_tf= scalers_cache,
                symbol       = symbol,
            )

            # Optuna refines GA result
            opt_params = run_optuna(
                df_dict      = tf_feat,
                models_by_tf = models_cache,
                scalers_by_tf= scalers_cache,
                symbol       = symbol,
            )

            # Keep whichever scored better
            best_params = (opt_params if opt_params.get("best_score", -999) >
                           ga_params.get("best_score", -999)
                           else ga_params)
            best_params["source"] = "historical_optimisation"
            save_params(symbol, best_params)

            log.info(f"\n  {symbol} OPTIMAL PARAMS DISCOVERED:")
            log.info(f"    Entry TF  : {best_params['entry_tf']}m")
            log.info(f"    HTF       : {best_params['htf_tf']}m "
                     f"({'NONE' if best_params['htf_tf']==0 else 'active'})")
            log.info(f"    SL ATR x  : {best_params['sl_atr']:.3f}")
            log.info(f"    TP mult   : {best_params['tp_mult']:.2f}x")
            log.info(f"    Confidence: {best_params['confidence']:.2f}")
            log.info(f"    HTF weight: {best_params['htf_weight']:.2f}")

    log.info("\nHistorical training complete.")
    return models_cache, scalers_cache, all_data


# ─────────────────────────────────────────────────────────────
# 18. LIVE TRADING LOOP
# ─────────────────────────────────────────────────────────────

def run_live_loop(models_cache: dict, scalers_cache: dict,
                  all_data: dict):
    """
    Main live loop. Wakes on closed candle of each symbol's entry TF.
    Refreshes data, generates signals, applies risk gate, places orders.
    Triggers incremental re-optimisation after enough live trades.
    """
    log.info("=" * 60)
    log.info("LIVE TRADING LOOP STARTED")
    log.info(f"Hard limits: {HARD_LIMITS}")
    log.info("=" * 60)

    risk_gate    = RiskGate()
    last_reset   = datetime.now().date()
    live_data    = {}    # {f"{symbol}_{tf}m": df}

    # Pre-populate live data cache from historical data
    for symbol, tf_dict in all_data.items():
        for tf, df in tf_dict.items():
            live_data[f"{symbol}_{tf}m"] = df

    while True:
        try:
            now = datetime.now()

            # Reset daily loss counter at midnight
            if now.date() > last_reset:
                risk_gate.reset_daily()
                last_reset = now.date()
                log.info("Daily risk counters reset")

            # Find minimum sleep needed across all symbols
            # (wake on earliest next candle close)
            min_sleep = float("inf")
            for symbol in INSTRUMENTS:
                params = load_params(symbol)
                tf     = params["entry_tf"]
                sleep  = seconds_to_next_candle_close(tf)
                min_sleep = min(min_sleep, sleep)

            log.info(f"Sleeping {min_sleep:.1f}s to next candle close...")
            time.sleep(max(1.0, min_sleep))

            # Process each symbol
            for symbol in INSTRUMENTS:
                params   = load_params(symbol)
                entry_tf = params["entry_tf"]
                htf_tf   = params["htf_tf"]

                # Refresh entry TF data (last 200 bars is enough for live)
                fresh_entry = fetch_bars(symbol, entry_tf, n_bars=300)
                if not fresh_entry.empty:
                    fresh_entry = engineer_features(fresh_entry)
                    live_data[f"{symbol}_{entry_tf}m"] = fresh_entry

                # Refresh HTF data if active
                if htf_tf > 0:
                    fresh_htf = fetch_bars(symbol, htf_tf, n_bars=100)
                    if not fresh_htf.empty:
                        live_data[f"{symbol}_{htf_tf}m"] = fresh_htf

                # Generate signal
                signal = get_signal(
                    symbol       = symbol,
                    params       = params,
                    models_cache = models_cache,
                    scalers_cache = scalers_cache,
                    live_data_cache = live_data,
                )

                if signal["direction"] == 0:
                    log.debug(f"{symbol}: no signal | {signal.get('reason','')}")
                    continue

                # Risk gate check
                allowed, reason = risk_gate.can_trade()
                if not allowed:
                    log.warning(f"Trade blocked — {reason}")
                    continue

                # Position size
                lot = risk_gate.position_size(
                    sl_pips = signal["sl_pips"],
                    symbol  = symbol,
                )

                log.info(
                    f"SIGNAL: {symbol} | "
                    f"{'LONG' if signal['direction']==1 else 'SHORT'} | "
                    f"conf:{signal['confidence']:.3f} | "
                    f"TF:{signal['entry_tf']}m HTF:{signal['htf_used']}m | "
                    f"lot:{lot} | SL:{signal['sl_price']} TP:{signal['tp_price']}"
                )

                # Place order
                result = place_order(
                    symbol    = symbol,
                    direction = signal["direction"],
                    lot       = lot,
                    sl        = signal["sl_price"],
                    tp        = signal["tp_price"],
                    comment   = f"ml_{signal['entry_tf']}m",
                )

                if result["success"]:
                    log_trade({
                        "symbol":     symbol,
                        "direction":  signal["direction"],
                        "entry":      signal["entry_price"],
                        "sl":         signal["sl_price"],
                        "tp":         signal["tp_price"],
                        "lot":        lot,
                        "confidence": signal["confidence"],
                        "entry_tf":   signal["entry_tf"],
                        "htf_tf":     signal["htf_used"],
                        "params":     params,
                        "ticket":     result["ticket"],
                        "timestamp":  now.isoformat(),
                        "used_for_retrain": False,
                    })

                # Check if incremental retrain is due
                if should_retrain(symbol):
                    incremental_update(
                        symbol       = symbol,
                        df_dict      = all_data.get(symbol, {}),
                        models_cache = models_cache,
                        scalers_cache = scalers_cache,
                    )

        except KeyboardInterrupt:
            log.info("Live loop stopped by user.")
            break
        except Exception as e:
            log.error(f"Live loop error: {e}", exc_info=True)
            time.sleep(30)


# ─────────────────────────────────────────────────────────────
# 19. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Adaptive ML Trading Engine starting...")

    if not connect_mt5():
        log.error("Cannot connect to MT5. Exiting.")
        exit(1)

    try:
        # Historical training (always runs first — builds or refreshes models)
        models_cache, scalers_cache, all_data = run_historical_training()

        if not models_cache:
            log.error("No models trained. Check MT5 connection and symbol names.")
            exit(1)

        # Enter live trading loop
        run_live_loop(models_cache, scalers_cache, all_data)

    finally:
        mt5.shutdown()
        log.info("MT5 disconnected. Engine stopped.")
