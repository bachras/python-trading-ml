"""
Phase 1 — MT5 Data Connection + Feature Engineering
=====================================================
ML Trading System | Forex & Index CFDs
Instruments: EUR/USD, GBP/USD, DAX (DE40), S&P500 (US500)

Requirements:
    pip install MetaTrader5 pandas numpy scikit-learn ta matplotlib

Usage:
    1. Open MetaTrader 5 terminal and log in to your broker account
    2. In MT5: Tools > Options > Expert Advisors > Allow algorithmic trading
    3. Run: python phase1_mt5_data.py
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit these to match your broker
# ─────────────────────────────────────────────
MT5_LOGIN    = 0           # your MT5 account number (int)
MT5_PASSWORD = ""          # your MT5 password
MT5_SERVER   = ""          # your broker server name (e.g. "ICMarkets-Demo")

INSTRUMENTS = {
    "EURUSD": "Forex",
    "GBPUSD": "Forex",
    "DE40":   "Index",   # DAX — some brokers use GER40, DE30, check yours
    "US500":  "Index",   # S&P500 — some brokers use SP500, US500.cash
}

TIMEFRAME   = mt5.TIMEFRAME_H1   # 1-hour bars (good balance for ML)
BARS        = 5000               # how many bars of history to fetch
DATA_DIR    = "data"             # folder to save CSVs


# ─────────────────────────────────────────────
# 1. MT5 CONNECTION
# ─────────────────────────────────────────────

def connect_mt5() -> bool:
    """Initialise and authenticate MT5 terminal."""
    if not mt5.initialize():
        print(f"[ERROR] MT5 initialise failed: {mt5.last_error()}")
        return False

    # If credentials supplied, log in
    if MT5_LOGIN and MT5_PASSWORD and MT5_SERVER:
        authorised = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorised:
            print(f"[ERROR] MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

    info = mt5.terminal_info()
    account = mt5.account_info()
    print(f"[OK] Connected to MT5 | Build {info.build}")
    if account:
        print(f"[OK] Account: {account.login} | {account.server} | Balance: {account.balance} {account.currency}")
    return True


def disconnect_mt5():
    mt5.shutdown()
    print("[OK] MT5 disconnected")


# ─────────────────────────────────────────────
# 2. RAW OHLCV DATA FETCH
# ─────────────────────────────────────────────

def fetch_ohlcv(symbol: str, timeframe=TIMEFRAME, bars=BARS) -> pd.DataFrame:
    """
    Fetch historical OHLCV bars from MT5 for a given symbol.
    Returns a clean DataFrame indexed by datetime (UTC).
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None or len(rates) == 0:
        print(f"[WARN] No data for {symbol}: {mt5.last_error()}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open":        "Open",
        "high":        "High",
        "low":         "Low",
        "close":       "Close",
        "tick_volume": "Volume",
    }, inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.sort_index(inplace=True)

    print(f"[OK] {symbol}: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def add_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Build the full feature set used by the ML models.

    Groups:
      A. Returns & price transforms
      B. Trend indicators
      C. Momentum indicators
      D. Volatility indicators
      E. Volume features
      F. Time / calendar features
      G. Lag features
    """
    d = df.copy()
    close = d["Close"]
    high  = d["High"]
    low   = d["Low"]
    vol   = d["Volume"]

    # ── A. Returns & price transforms ──────────────
    d["return_1"]  = close.pct_change(1)
    d["return_5"]  = close.pct_change(5)
    d["return_20"] = close.pct_change(20)

    d["log_return_1"]  = np.log(close / close.shift(1))
    d["log_return_5"]  = np.log(close / close.shift(5))

    d["hl_spread"]  = (high - low) / close          # bar range as fraction of price
    d["co_spread"]  = (close - d["Open"]) / close   # open→close move

    # ── B. Trend indicators ─────────────────────────
    for n in [8, 21, 55, 200]:
        d[f"ema_{n}"] = close.ewm(span=n, adjust=False).mean()

    # EMA crossover signals (positive = bullish)
    d["ema_cross_8_21"]  = d["ema_8"]  - d["ema_21"]
    d["ema_cross_21_55"] = d["ema_21"] - d["ema_55"]

    # Price position relative to EMAs (normalised)
    d["price_vs_ema21"]  = (close - d["ema_21"])  / d["ema_21"]
    d["price_vs_ema55"]  = (close - d["ema_55"])  / d["ema_55"]
    d["price_vs_ema200"] = (close - d["ema_200"]) / d["ema_200"]

    # ── C. Momentum indicators ──────────────────────
    # RSI (14)
    delta   = close.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / (loss + 1e-10)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # Stochastic %K / %D (14,3)
    low14       = low.rolling(14).min()
    high14      = high.rolling(14).max()
    d["stoch_k"] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # MACD (12, 26, 9)
    ema12         = close.ewm(span=12, adjust=False).mean()
    ema26         = close.ewm(span=26, adjust=False).mean()
    d["macd"]          = ema12 - ema26
    d["macd_signal"]   = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]     = d["macd"] - d["macd_signal"]

    # Rate of change
    d["roc_5"]  = close.pct_change(5)  * 100
    d["roc_14"] = close.pct_change(14) * 100

    # Williams %R (14)
    d["williams_r"] = -100 * (high14 - close) / (high14 - low14 + 1e-10)

    # ── D. Volatility indicators ────────────────────
    # Bollinger Bands (20, 2σ)
    bb_mid           = close.rolling(20).mean()
    bb_std           = close.rolling(20).std()
    d["bb_upper"]    = bb_mid + 2 * bb_std
    d["bb_lower"]    = bb_mid - 2 * bb_std
    d["bb_mid"]      = bb_mid
    d["bb_width"]    = (d["bb_upper"] - d["bb_lower"]) / (bb_mid + 1e-10)
    d["bb_pct"]      = (close - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-10)

    # ATR (14) — Average True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr_14"]      = tr.rolling(14).mean()
    d["atr_pct"]     = d["atr_14"] / close   # normalised

    # Historical volatility (20-bar)
    d["hv_20"] = d["log_return_1"].rolling(20).std() * np.sqrt(252 * 24)  # annualised

    # ── E. Volume features ──────────────────────────
    d["volume_sma20"]  = vol.rolling(20).mean()
    d["volume_ratio"]  = vol / (d["volume_sma20"] + 1e-10)  # relative volume
    d["volume_trend"]  = vol.rolling(5).mean() / (vol.rolling(20).mean() + 1e-10)

    # On-balance volume
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    d["obv_norm"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-10)

    # ── F. Time / calendar features ────────────────
    d["hour"]       = d.index.hour
    d["day_of_week"] = d.index.dayofweek   # 0=Mon, 4=Fri
    d["is_london"]  = ((d["hour"] >= 8)  & (d["hour"] < 16)).astype(int)
    d["is_ny"]      = ((d["hour"] >= 13) & (d["hour"] < 21)).astype(int)
    d["is_overlap"] = ((d["hour"] >= 13) & (d["hour"] < 16)).astype(int)  # London/NY overlap

    # Encode hour as cyclical features (avoids midnight discontinuity)
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["dow_sin"]  = np.sin(2 * np.pi * d["day_of_week"] / 5)
    d["dow_cos"]  = np.cos(2 * np.pi * d["day_of_week"] / 5)

    # ── G. Lag features (for LSTM sequence context) ──
    for lag in [1, 2, 3, 5, 8, 13]:
        d[f"close_lag_{lag}"]  = close.shift(lag)
        d[f"return_lag_{lag}"] = d["log_return_1"].shift(lag)

    # ── H. Target variable ──────────────────────────
    # Predict direction of next bar's close (1 = up, 0 = down)
    d["target"] = (close.shift(-1) > close).astype(int)
    # Also store continuous next-bar return for regression models
    d["target_return"] = d["log_return_1"].shift(-1)

    # Drop rows with NaN (from rolling windows)
    d.dropna(inplace=True)

    feature_count = len([c for c in d.columns if c not in
                         ["Open","High","Low","Close","Volume","target","target_return"]])
    print(f"[OK] {symbol}: {feature_count} features engineered | {len(d)} clean rows")
    return d


# ─────────────────────────────────────────────
# 4. NORMALISATION
# ─────────────────────────────────────────────

# These features should NOT be normalised (already bounded or categorical)
SKIP_SCALE = {
    "target", "target_return",
    "rsi_14", "stoch_k", "stoch_d", "williams_r",
    "bb_pct", "hour", "day_of_week",
    "is_london", "is_ny", "is_overlap",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
}

def normalise(df: pd.DataFrame):
    """
    Fit a StandardScaler on training data and return scaled DataFrame.
    Returns (scaled_df, scaler) so the scaler can be applied to live data.
    """
    feature_cols = [c for c in df.columns
                    if c not in SKIP_SCALE
                    and c not in ["Open","High","Low","Close","Volume"]]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"[OK] Normalised {len(feature_cols)} feature columns")
    return df_scaled, scaler, feature_cols


# ─────────────────────────────────────────────
# 5. TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────

def time_split(df: pd.DataFrame, train=0.70, val=0.15):
    """
    Chronological split — never shuffle financial time series!
    train: first 70% | val: next 15% | test: final 15%
    """
    n     = len(df)
    i_tr  = int(n * train)
    i_val = int(n * (train + val))

    train_df = df.iloc[:i_tr]
    val_df   = df.iloc[i_tr:i_val]
    test_df  = df.iloc[i_val:]

    print(f"[OK] Split → train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# 6. SEQUENCE BUILDER (for LSTM)
# ─────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feature_cols: list,
                    sequence_len: int = 60):
    """
    Convert a flat DataFrame into (X, y) 3D arrays for LSTM input.
    X shape: (samples, sequence_len, n_features)
    y shape: (samples,) — binary classification target
    """
    X_list, y_list = [], []

    feature_arr = df[feature_cols].values
    target_arr  = df["target"].values

    for i in range(sequence_len, len(df)):
        X_list.append(feature_arr[i - sequence_len : i])
        y_list.append(target_arr[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"[OK] Sequences: X{X.shape} | y{y.shape}")
    return X, y


# ─────────────────────────────────────────────
# 7. INSTRUMENT SELECTOR (model output helper)
# ─────────────────────────────────────────────

def select_best_instrument(signal_scores: dict) -> str:
    """
    Given a dict of {symbol: confidence_score},
    return the symbol with the highest model confidence.
    Example: {"EURUSD": 0.72, "GBPUSD": 0.61, "DE40": 0.58, "US500": 0.65}
    """
    best = max(signal_scores, key=signal_scores.get)
    print(f"[OK] Best instrument: {best} (score: {signal_scores[best]:.3f})")
    return best


# ─────────────────────────────────────────────
# 8. SAVE / LOAD HELPERS
# ─────────────────────────────────────────────

def save_data(df: pd.DataFrame, symbol: str, folder: str = DATA_DIR):
    """Save processed DataFrame to CSV."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{symbol}_features.csv")
    df.to_csv(path)
    print(f"[OK] Saved: {path}")


def load_data(symbol: str, folder: str = DATA_DIR) -> pd.DataFrame:
    """Load previously saved feature CSV."""
    path = os.path.join(folder, f"{symbol}_features.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[OK] Loaded: {path} | {len(df)} rows")
    return df


# ─────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    """
    Full Phase 1 pipeline:
      1. Connect to MT5
      2. Fetch OHLCV for all instruments
      3. Engineer features
      4. Normalise
      5. Split into train/val/test
      6. Build LSTM sequences
      7. Save everything to disk
    Returns a dict with processed data for each instrument.
    """
    if not connect_mt5():
        print("[ERROR] Cannot connect to MT5. Check terminal is open and logged in.")
        return {}

    results = {}

    for symbol in INSTRUMENTS:
        print(f"\n{'─'*50}")
        print(f"Processing: {symbol} ({INSTRUMENTS[symbol]})")
        print(f"{'─'*50}")

        # Fetch raw data
        raw_df = fetch_ohlcv(symbol)
        if raw_df.empty:
            print(f"[SKIP] {symbol} — no data returned")
            continue

        # Engineer features
        feat_df = add_features(raw_df, symbol)

        # Normalise
        scaled_df, scaler, feature_cols = normalise(feat_df)

        # Train/val/test split
        train_df, val_df, test_df = time_split(scaled_df)

        # Build LSTM sequences (from training set)
        X_train, y_train = build_sequences(train_df, feature_cols, sequence_len=60)
        X_val,   y_val   = build_sequences(val_df,   feature_cols, sequence_len=60)
        X_test,  y_test  = build_sequences(test_df,  feature_cols, sequence_len=60)

        # Save feature CSV
        save_data(feat_df, symbol)

        results[symbol] = {
            "raw":          raw_df,
            "features":     feat_df,
            "scaled":       scaled_df,
            "scaler":       scaler,
            "feature_cols": feature_cols,
            "train":        train_df,
            "val":          val_df,
            "test":         test_df,
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
        }

        print(f"[✓] {symbol} ready for ML models")

    disconnect_mt5()

    print(f"\n{'='*50}")
    print(f"Phase 1 complete | {len(results)} instruments ready")
    print(f"{'='*50}")
    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = run_pipeline()

    # Quick sanity check — print feature list for first instrument
    if data:
        sym = list(data.keys())[0]
        feat_cols = data[sym]["feature_cols"]
        print(f"\nFeature columns for {sym} ({len(feat_cols)} total):")
        for i, col in enumerate(feat_cols, 1):
            print(f"  {i:3d}. {col}")

        # Show class balance of targets
        for sym, d in data.items():
            target = d["features"]["target"]
            up_pct = target.mean() * 100
            print(f"\n{sym} target balance: {up_pct:.1f}% up | {100-up_pct:.1f}% down")
