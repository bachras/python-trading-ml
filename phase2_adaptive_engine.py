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

import os, time, json, logging, warnings
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

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel("ERROR")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

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
        logging.FileHandler(LOG_DIR / f"trading_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
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

INSTRUMENTS = {
    # Indices
    _sym("US30",  "US30"):  "index",
    _sym("DE40",  "DE40"):  "index",
    _sym("USTEC", "USTEC"): "index",
    _sym("UK100", "UK100"): "index",
    _sym("US500", "US500"): "index",
    # Forex
    _sym("EURUSD", "EURUSD"): "forex",
    _sym("GBPUSD", "GBPUSD"): "forex",
    _sym("GBPJPY", "GBPJPY"): "forex",
    _sym("EURJPY", "EURJPY"): "forex",
    _sym("USDCAD", "USDCAD"): "forex",
}

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
    "entry_tf_options":  [1, 3, 5, 10, 15],
    "entry_tf_default":  5,

    # HTF confirmation candidates — includes 0 = NONE (ML may skip HTF)
    "htf_options":       [0, 15, 30, 60],   # 0 = no HTF confirmation
    "htf_default":       60,

    # Stop loss ATR multiplier range
    "sl_atr_min":        0.5,
    "sl_atr_max":        3.0,
    "sl_atr_seed":       1.5,

    # Risk:Reward ratio range
    "rr_min":            1.0,
    "rr_max":            4.0,
    "rr_seed":           2.0,

    # Take profit multiplier of SL distance
    "tp_mult_min":       1.0,
    "tp_mult_max":       5.0,
    "tp_mult_seed":      2.0,

    # Signal confidence threshold
    "confidence_min":    0.50,
    "confidence_max":    0.85,
    "confidence_seed":   0.60,

    # HTF alignment weight (0 = ignore HTF, 1 = require strong alignment)
    "htf_weight_min":    0.0,
    "htf_weight_max":    1.0,
    "htf_weight_seed":   0.5,
}

# ── Hard risk limits — NEVER modified by ML ──
HARD_LIMITS = {
    "max_daily_loss_pct":   float(os.getenv("MAX_DAILY_LOSS_PCT",  "2.0")),
    "max_drawdown_pct":     float(os.getenv("MAX_DRAWDOWN_PCT",   "10.0")),
    "max_open_positions":   int(os.getenv("MAX_OPEN_POSITIONS",   "3")),
    "risk_per_trade_pct":   float(os.getenv("RISK_PER_TRADE_PCT",  "1.0")),
}

# Optuna optimisation budget
OPTUNA_TRIALS       = 150   # increase for more thorough search
OPTUNA_LIVE_TRIALS  = 30    # faster re-optimisation after live trades

# LSTM sequence length (bars to look back)
SEQ_LEN = 60

# Minimum bars needed before training
MIN_BARS = 500

# Live trade log path
TRADE_LOG = LOG_DIR / "trade_log.json"

# ─────────────────────────────────────────────────────────────
# 1. MT5 CONNECTION
# ─────────────────────────────────────────────────────────────

def connect_mt5() -> bool:
    if not mt5.initialize():
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False
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
    if account:
        log.info(f"Account {account.login} | {account.server} | "
                 f"Balance: {account.balance} {account.currency}")
    return True


def get_account_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


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

    # Session flags
    hr = d.index.hour
    d["is_london"]  = ((hr >= 8)  & (hr < 16)).astype(int)
    d["is_ny"]      = ((hr >= 13) & (hr < 21)).astype(int)
    d["is_overlap"] = ((hr >= 13) & (hr < 16)).astype(int)
    d["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    dow = d.index.dayofweek
    d["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # Lags
    for lag in [1, 2, 3, 5, 8, 13]:
        d[f"c_lag{lag}"] = c.shift(lag)
        d[f"r_lag{lag}"] = d["logr1"].shift(lag)

    # Targets
    d["target"]        = (c.shift(-1) > c).astype(int)
    d["target_return"] = d["logr1"].shift(-1)

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
# 4. FEATURE COLUMNS (excludes raw OHLCV and targets)
# ─────────────────────────────────────────────────────────────

EXCLUDE_COLS = {"Open", "High", "Low", "Close", "Volume",
                "target", "target_return"}

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


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
# 6. LSTM MODEL
# ─────────────────────────────────────────────────────────────

def build_lstm(n_features: int, seq_len: int = SEQ_LEN) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def make_sequences(df: pd.DataFrame, feature_cols: list,
                   seq_len: int = SEQ_LEN):
    X, y = [], []
    arr    = df[feature_cols].values
    target = df["target"].values
    for i in range(seq_len, len(df)):
        X.append(arr[i - seq_len:i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(train_df, val_df, feature_cols, symbol, tf_min):
    log.info(f"  Training LSTM: {symbol} {tf_min}m")
    X_tr, y_tr = make_sequences(train_df, feature_cols)
    X_va, y_va = make_sequences(val_df,   feature_cols)
    if len(X_tr) < 100:
        log.warning(f"  Insufficient sequences for {symbol} {tf_min}m LSTM")
        return None
    model = build_lstm(len(feature_cols))
    cbs = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
    ]
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
              epochs=100, batch_size=64, callbacks=cbs, verbose=0)
    path = MODEL_DIR / f"lstm_{symbol}_{tf_min}m.keras"
    model.save(path)
    log.info(f"  LSTM saved: {path}")
    return model


# ─────────────────────────────────────────────────────────────
# 7. ENSEMBLE MODELS (XGBoost + Random Forest)
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
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10,
        n_jobs=-1, random_state=42
    )
    rf_model.fit(X_tr, y_tr)

    path_xgb = MODEL_DIR / f"xgb_{symbol}_{tf_min}m.pkl"
    path_rf  = MODEL_DIR / f"rf_{symbol}_{tf_min}m.pkl"
    joblib.dump(xgb_model, path_xgb)
    joblib.dump(rf_model,  path_rf)
    log.info(f"  Ensemble saved: {path_xgb.name}, {path_rf.name}")
    return xgb_model, rf_model


# ─────────────────────────────────────────────────────────────
# 8. REINFORCEMENT LEARNING ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    RL environment for the PPO agent.
    State : last SEQ_LEN rows of features (flattened)
    Action: 0=hold, 1=long, 2=short
    Reward: risk-adjusted return, penalised for drawdown
    The agent learns entry TF weight, HTF weight, and when to act.
    """
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 sl_atr_mult: float = 1.5, rr: float = 2.0):
        super().__init__()
        self.df           = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.sl_mult      = sl_atr_mult
        self.rr           = rr
        self.n_features   = len(feature_cols)

        self.observation_space = spaces.Box(
            low=-10, high=10,
            shape=(SEQ_LEN * self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=long, 2=short

        self.reset()

    def _obs(self):
        start = max(0, self.idx - SEQ_LEN)
        block = self.df[self.feature_cols].iloc[start:self.idx].values
        if len(block) < SEQ_LEN:
            block = np.pad(block, ((SEQ_LEN - len(block), 0), (0, 0)))
        return block.flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx        = SEQ_LEN
        self.balance    = 10000.0
        self.peak       = 10000.0
        self.daily_pnl  = 0.0
        self.open_trade = None
        return self._obs(), {}

    def step(self, action):
        row  = self.df.iloc[self.idx]
        atr  = row.get("atr14", row["Close"] * 0.001)
        done = self.idx >= len(self.df) - 2
        reward = 0.0

        # Close open trade if target/stop hit
        if self.open_trade:
            t    = self.open_trade
            pnl  = 0.0
            hit  = False
            if t["dir"] == 1:   # long
                if row["Low"]  <= t["sl"]: pnl = -t["risk"]; hit = True
                if row["High"] >= t["tp"]: pnl =  t["risk"] * self.rr; hit = True
            else:                # short
                if row["High"] >= t["sl"]: pnl = -t["risk"]; hit = True
                if row["Low"]  <= t["tp"]: pnl =  t["risk"] * self.rr; hit = True
            if hit:
                self.balance   += pnl
                self.daily_pnl += pnl
                reward          = pnl / (t["risk"] + 1e-10)
                self.open_trade = None
                if self.balance > self.peak:
                    self.peak = self.balance

        # Drawdown penalty
        dd = (self.peak - self.balance) / (self.peak + 1e-10)
        reward -= dd * 2.0

        # Open new trade
        if action != 0 and self.open_trade is None:
            risk = self.balance * HARD_LIMITS["risk_per_trade_pct"] / 100
            sl_dist = atr * self.sl_mult
            direction = 1 if action == 1 else -1
            sl = row["Close"] - direction * sl_dist
            tp = row["Close"] + direction * sl_dist * self.rr
            self.open_trade = {
                "dir": direction, "sl": sl, "tp": tp, "risk": risk
            }

        # Hard limit: max drawdown terminates episode
        if dd > HARD_LIMITS["max_drawdown_pct"] / 100:
            done = True
            reward -= 10.0

        self.idx += 1
        return self._obs(), reward, done, False, {}


def train_rl_agent(df: pd.DataFrame, feature_cols: list,
                   symbol: str, sl_mult: float, rr: float):
    log.info(f"  Training RL agent: {symbol}")

    def make_env():
        return TradingEnv(df, feature_cols, sl_mult, rr)

    env  = DummyVecEnv([make_env])
    path = str(MODEL_DIR / f"ppo_{symbol}")

    if Path(path + ".zip").exists():
        model = PPO.load(path, env=env)
        model.set_env(env)
        model.learn(total_timesteps=50_000, reset_num_timesteps=False)
    else:
        model = PPO("MlpPolicy", env, verbose=0,
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, clip_range=0.2)
        model.learn(total_timesteps=200_000)

    model.save(path)
    log.info(f"  RL agent saved: {path}.zip")
    return model


# ─────────────────────────────────────────────────────────────
# 9. GENETIC ALGORITHM — PARAMETER OPTIMISATION
# ─────────────────────────────────────────────────────────────
# Genome: [entry_tf_idx, htf_idx, sl_atr, rr, tp_mult, confidence_thresh, htf_weight]

GA_BOUNDS = [
    (0, len(PARAM_SEEDS["entry_tf_options"]) - 1),   # entry TF index
    (0, len(PARAM_SEEDS["htf_options"]) - 1),         # HTF index (0 = NONE)
    (PARAM_SEEDS["sl_atr_min"],    PARAM_SEEDS["sl_atr_max"]),
    (PARAM_SEEDS["rr_min"],        PARAM_SEEDS["rr_max"]),
    (PARAM_SEEDS["tp_mult_min"],   PARAM_SEEDS["tp_mult_max"]),
    (PARAM_SEEDS["confidence_min"],PARAM_SEEDS["confidence_max"]),
    (PARAM_SEEDS["htf_weight_min"],PARAM_SEEDS["htf_weight_max"]),
]

def decode_genome(genome: list) -> dict:
    return {
        "entry_tf":   PARAM_SEEDS["entry_tf_options"][int(round(genome[0]))],
        "htf_tf":     PARAM_SEEDS["htf_options"][int(round(genome[1]))],
        "sl_atr":     float(np.clip(genome[2], GA_BOUNDS[2][0], GA_BOUNDS[2][1])),
        "rr":         float(np.clip(genome[3], GA_BOUNDS[3][0], GA_BOUNDS[3][1])),
        "tp_mult":    float(np.clip(genome[4], GA_BOUNDS[4][0], GA_BOUNDS[4][1])),
        "confidence": float(np.clip(genome[5], GA_BOUNDS[5][0], GA_BOUNDS[5][1])),
        "htf_weight": float(np.clip(genome[6], GA_BOUNDS[6][0], GA_BOUNDS[6][1])),
    }


def ga_fitness(genome, df_dict: dict, models: dict, scaler, feature_cols):
    """
    Evaluate genome fitness via quick backtest on validation data.
    Returns (sharpe_ratio,) — DEAP expects a tuple.
    """
    params = decode_genome(genome)
    entry_tf = params["entry_tf"]

    if entry_tf not in df_dict:
        return (-999.0,)

    df = df_dict[entry_tf]
    if len(df) < MIN_BARS:
        return (-999.0,)

    # Quick vectorised backtest
    scaled = apply_scaler(df, scaler, feature_cols)
    feat   = scaled[feature_cols].values
    target = df["target"].values
    atr    = df["atr14"].values
    closes = df["Close"].values

    # Get ensemble signal
    xgb_m = models.get("xgb")
    rf_m  = models.get("rf")
    if xgb_m is None or rf_m is None:
        return (-999.0,)

    p_xgb = xgb_m.predict_proba(feat)[:, 1]
    p_rf  = rf_m.predict_proba(feat)[:, 1]
    signal_prob = (p_xgb + p_rf) / 2.0

    # Simulate trades
    balance  = 10000.0
    peak     = 10000.0
    returns  = []
    in_trade = False

    for i in range(SEQ_LEN, len(df) - 1):
        if in_trade:
            continue

        prob = signal_prob[i]
        direction = None
        if prob >= params["confidence"]:
            direction = 1    # long
        elif prob <= (1 - params["confidence"]):
            direction = -1   # short

        if direction is None:
            continue

        sl_dist = atr[i] * params["sl_atr"]
        if sl_dist == 0:
            continue

        entry = closes[i]
        sl    = entry - direction * sl_dist
        tp    = entry + direction * sl_dist * params["rr"]
        risk  = balance * HARD_LIMITS["risk_per_trade_pct"] / 100

        # Check next bar for hit
        next_high = df["High"].iloc[i + 1]
        next_low  = df["Low"].iloc[i + 1]

        pnl = 0.0
        if direction == 1:
            if next_low  <= sl: pnl = -risk
            elif next_high >= tp: pnl = risk * params["rr"]
        else:
            if next_high >= sl: pnl = -risk
            elif next_low  <= tp: pnl = risk * params["rr"]

        if pnl != 0:
            balance += pnl
            returns.append(pnl / risk)
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            if dd > HARD_LIMITS["max_drawdown_pct"] / 100:
                break

    if len(returns) < 10:
        return (-999.0,)

    r_arr  = np.array(returns)
    sharpe = r_arr.mean() / (r_arr.std() + 1e-10) * np.sqrt(252)
    return (sharpe,)


def run_genetic_algo(df_dict: dict, models: dict,
                     scaler, feature_cols: list,
                     n_gen=40, pop_size=80) -> dict:
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
                     df_dict=df_dict, models=models,
                     scaler=scaler, feature_cols=feature_cols)
    toolbox.register("mate",    tools.cxBlend, alpha=0.3)
    toolbox.register("mutate",  tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select",  tools.selTournament, tournsize=5)

    pop  = toolbox.population(n=pop_size)
    hof  = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox,
                        cxpb=0.6, mutpb=0.3,
                        ngen=n_gen, stats=stats,
                        halloffame=hof, verbose=False)

    best = decode_genome(hof[0])
    log.info(f"  GA best genome: {best}")
    return best


# ─────────────────────────────────────────────────────────────
# 10. OPTUNA BAYESIAN OPTIMISATION
# ─────────────────────────────────────────────────────────────

def run_optuna(df_dict: dict, models: dict,
               scaler, feature_cols: list,
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
            trial.suggest_float("rr",         *GA_BOUNDS[3]),
            trial.suggest_float("tp_mult",    *GA_BOUNDS[4]),
            trial.suggest_float("confidence", *GA_BOUNDS[5]),
            trial.suggest_float("htf_weight", *GA_BOUNDS[6]),
        ]
        result = ga_fitness(genome, df_dict, models, scaler, feature_cols)
        return result[0]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name=f"opt_{symbol}",
    )

    # Warm-start with seed values
    seed_genome = [
        PARAM_SEEDS["entry_tf_options"].index(PARAM_SEEDS["entry_tf_default"]),
        PARAM_SEEDS["htf_options"].index(PARAM_SEEDS["htf_default"]),
        PARAM_SEEDS["sl_atr_seed"],
        PARAM_SEEDS["rr_seed"],
        PARAM_SEEDS["tp_mult_seed"],
        PARAM_SEEDS["confidence_seed"],
        PARAM_SEEDS["htf_weight_seed"],
    ]
    study.enqueue_trial({
        "entry_tf_idx": seed_genome[0],
        "htf_idx":      seed_genome[1],
        "sl_atr":       seed_genome[2],
        "rr":           seed_genome[3],
        "tp_mult":      seed_genome[4],
        "confidence":   seed_genome[5],
        "htf_weight":   seed_genome[6],
    })

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    result = {
        "entry_tf":   PARAM_SEEDS["entry_tf_options"][best["entry_tf_idx"]],
        "htf_tf":     PARAM_SEEDS["htf_options"][best["htf_idx"]],
        "sl_atr":     best["sl_atr"],
        "rr":         best["rr"],
        "tp_mult":    best["tp_mult"],
        "confidence": best["confidence"],
        "htf_weight": best["htf_weight"],
        "best_score": study.best_value,
    }
    log.info(f"  Optuna best: {result}")
    return result


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
        "rr":         PARAM_SEEDS["rr_seed"],
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
    rr        = params["rr"]
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

    # LSTM signal
    lstm_model = models_cache.get(f"lstm_{key}")
    p_lstm = 0.5
    if lstm_model is not None:
        X, _ = make_sequences(scaled.tail(SEQ_LEN + 2), feature_cols)
        if len(X) > 0:
            p_lstm = float(lstm_model.predict(X[-1:], verbose=0)[0][0])

    # Ensemble signal
    last_feat = scaled[feature_cols].iloc[-1:].values
    p_xgb, p_rf = 0.5, 0.5
    xgb_m = models_cache.get(f"xgb_{key}")
    rf_m  = models_cache.get(f"rf_{key}")
    if xgb_m is not None:
        p_xgb = float(xgb_m.predict_proba(last_feat)[0][1])
    if rf_m is not None:
        p_rf  = float(rf_m.predict_proba(last_feat)[0][1])

    # RL signal
    p_rl = 0.5
    rl_m = models_cache.get(f"ppo_{symbol}")
    if rl_m is not None:
        obs   = scaled[feature_cols].tail(SEQ_LEN).values.flatten()
        obs   = np.clip(obs, -10, 10).astype(np.float32)
        action, _ = rl_m.predict(obs, deterministic=True)
        p_rl = 0.8 if action == 1 else (0.2 if action == 2 else 0.5)

    # Weighted ensemble
    prob = (p_lstm * 0.30 + p_xgb * 0.25 + p_rf * 0.25 + p_rl * 0.20)

    # HTF filter: if HTF is bearish and signal is long, reduce confidence
    htf_dir = df["htf_bullish"].iloc[-1] if "htf_bullish" in df.columns else 0
    if htf_tf > 0 and htf_w > 0:
        if htf_dir == -1 and prob > 0.5:
            prob *= (1 - htf_w * 0.3)
        elif htf_dir == 1 and prob < 0.5:
            prob *= (1 + htf_w * 0.3)

    # Determine direction
    direction = 0
    if prob >= conf_thr:
        direction = 1   # long
    elif prob <= (1 - conf_thr):
        direction = -1  # short

    if direction == 0:
        return {"direction": 0, "confidence": prob,
                "reason": f"prob {prob:.3f} below threshold {conf_thr:.3f}"}

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
        "entry_price": round(close, 5),
        "sl_price":    round(sl, 5),
        "tp_price":    round(tp, 5),
        "sl_pips":     round(sl_dist, 5),
        "rr":          round(rr, 2),
        "entry_tf":    entry_tf,
        "htf_used":    htf_tf,
        "htf_dir":     int(htf_dir),
        "reason":      "signal confirmed",
    }


# ─────────────────────────────────────────────────────────────
# 13. RISK GATE — hard limits enforced before any order
# ─────────────────────────────────────────────────────────────

class RiskGate:
    def __init__(self):
        self.session_start_balance = get_account_balance()
        self.daily_loss            = 0.0

    def reset_daily(self):
        self.session_start_balance = get_account_balance()
        self.daily_loss            = 0.0

    def position_size(self, sl_pips: float, symbol: str) -> float:
        """Calculate lot size so risk = risk_per_trade_pct of balance."""
        balance     = get_account_balance()
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
        balance  = get_account_balance()
        dd_pct   = ((self.session_start_balance - balance) /
                    (self.session_start_balance + 1e-10)) * 100

        positions = mt5.positions_get()
        n_open    = len(positions) if positions else 0

        daily_loss_pct = (self.daily_loss /
                          (self.session_start_balance + 1e-10)) * 100

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

    feature_cols = get_feature_cols(feat_df)
    models = {
        "xgb": models_cache.get(f"xgb_{key}"),
        "rf":  models_cache.get(f"rf_{key}"),
    }

    new_params = run_optuna(
        df_dict    = {entry_tf: feat_df},
        models     = models,
        scaler     = scaler,
        feature_cols = feature_cols,
        n_trials   = OPTUNA_LIVE_TRIALS,
        symbol     = symbol,
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

        # RL agent (train on default entry TF)
        default_tf = PARAM_SEEDS["entry_tf_default"]
        if default_tf in tf_feat:
            df_rl = apply_scaler(
                tf_feat[default_tf],
                scalers_cache.get(f"{symbol}_{default_tf}m", StandardScaler()),
                get_feature_cols(tf_feat[default_tf])
            )
            rl_agent = train_rl_agent(
                df_rl,
                get_feature_cols(df_rl),
                symbol,
                sl_mult = PARAM_SEEDS["sl_atr_seed"],
                rr      = PARAM_SEEDS["rr_seed"],
            )
            models_cache[f"ppo_{symbol}"] = rl_agent

        # GA parameter search
        default_key  = f"{symbol}_{default_tf}m"
        scaler_def   = scalers_cache.get(default_key)
        feat_def     = tf_feat.get(default_tf)
        if scaler_def and feat_def is not None:
            feat_cols = get_feature_cols(feat_def)
            models_ga = {
                "xgb": models_cache.get(f"xgb_{default_key}"),
                "rf":  models_cache.get(f"rf_{default_key}"),
            }
            ga_params = run_genetic_algo(
                df_dict      = tf_feat,
                models       = models_ga,
                scaler       = scaler_def,
                feature_cols = feat_cols,
            )

            # Optuna refines GA result
            opt_params = run_optuna(
                df_dict      = tf_feat,
                models       = models_ga,
                scaler       = scaler_def,
                feature_cols = feat_cols,
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
            log.info(f"    R:R       : 1:{best_params['rr']:.2f}")
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
