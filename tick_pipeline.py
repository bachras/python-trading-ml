"""
Tick Data Pipeline — Dukascopy JForex CSV → ML-Ready Features
==============================================================
Handles 10GB+ files via chunked streaming — never loads full file into RAM.

Dukascopy JForex CSV format (confirmed):
    time, ask, bid, ask_volume, bid_volume
    e.g. 2019-01-02 00:00:02.533, 23251.4, 23249.7, 1.0, 1.0

What this module does:
  1. Streams tick CSV in chunks (memory safe for 10GB+)
  2. Cleans and validates every tick (removes outliers, fill gaps)
  3. Builds true OHLCV bars at ANY timeframe from raw ticks
     — uses BID price for candles (industry standard for indices/forex)
     — uses real bid/ask spread per bar (not broker-estimated)
  4. Calculates microstructure features only possible from tick data:
       - True spread per bar
       - Volume imbalance (buy vs sell pressure)
       - Tick velocity (ticks per second = liquidity proxy)
       - Price impact (how much price moved per unit of volume)
       - VWAP per bar
  5. Exports Parquet files (10x smaller than CSV, 50x faster to load)
  6. Integrates directly with phase2_adaptive_engine.py

Usage:
    python tick_pipeline.py

    First run builds all timeframe Parquet files (~30-60 min for 10GB).
    Subsequent runs load from Parquet in seconds.
"""

import os
import gc
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these paths to match your setup
# ─────────────────────────────────────────────────────────────

# Path to your Dukascopy tick CSV file
# If you have multiple files (one per month), set TICK_FILES to a list
# ── Paths from .env (change BASE_DIR in .env to move to any drive) ──
_BASE_DIR  = Path(os.getenv("BASE_DIR", r"C:\trading_ml"))
DATA_DIR   = Path(os.getenv("DATA_DIR",  str(_BASE_DIR / "data")))
SYMBOL     = os.getenv("TICK_SYMBOL", "US30")
TICK_FILE  = Path(os.getenv(
    f"TICK_FILE_{SYMBOL}",
    str(_BASE_DIR / "tick_data" / f"{SYMBOL}_ticks.csv")
))
TICK_FILES = None   # set to list of Paths if data is split across files
                    # e.g. [Path("US30_2019.csv"), Path("US30_2020.csv")]

CHUNK_SIZE = int(os.getenv("TICK_CHUNK_SIZE", "500000"))
# Tip: set TICK_CHUNK_SIZE=200000 in .env if you only have 8GB RAM

# Timeframes to build (minutes) — builds all entry + HTF candidates
TARGET_TIMEFRAMES = [1, 3, 5, 10, 15, 30, 60]

# Realistic spread cap — ticks with spread > this are outliers (in price units)
# US30 spread is typically 1-5 points. Cap at 200 to catch data errors only.
# Set wider than you think — a 50pt spread during a flash crash is real data.
MAX_SPREAD = 200.0

# Price sanity bounds for US30 across full history
# COVID low ~18,213 (Mar 2020), ATH ~50,512 (Feb 2026)
# Set wide margins so real extreme moves are never rejected
# Only truly corrupt ticks (e.g. 0 or 999999) get filtered
PRICE_MIN = 15_000.0   # below any realistic US30 level in this dataset
PRICE_MAX = 60_000.0   # well above current ATH — safe headroom

# ─────────────────────────────────────────────────────────────
# DUKASCOPY CSV COLUMN SPEC
# ─────────────────────────────────────────────────────────────
# Confirmed format from JForex Historical Data Manager:
#   time            — "YYYY.MM.DD HH:MM:SS.mmm" or "YYYY-MM-DD HH:MM:SS.mmm"
#   ask             — ask price
#   bid             — bid price
#   ask_volume      — ask-side volume
#   bid_volume      — bid-side volume

DUKA_COLS    = ["time", "ask", "bid", "ask_volume", "bid_volume"]
DUKA_DTYPES  = {
    "ask":        np.float32,
    "bid":        np.float32,
    "ask_volume": np.float32,
    "bid_volume": np.float32,
}

# ─────────────────────────────────────────────────────────────
# 1. TICK FILE DETECTION & VALIDATION
# ─────────────────────────────────────────────────────────────

def detect_csv_format(filepath: Path) -> dict:
    """
    Peek at first few lines to auto-detect separator, datetime format,
    whether there's a header row, and column order.
    Returns a format dict used by the streaming reader.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = [f.readline().strip() for _ in range(5)]

    # Detect separator
    sep = "," if lines[0].count(",") > lines[0].count(";") else ";"

    # Detect header
    has_header = not lines[0][0].isdigit()

    # Detect datetime format by inspecting first data line
    data_line = lines[1] if has_header else lines[0]
    first_field = data_line.split(sep)[0].strip().strip('"')

    if "." in first_field and first_field.count("-") == 2:
        dt_format = "%Y-%m-%d %H:%M:%S.%f"
    elif "." in first_field and first_field.count(".") == 3:
        dt_format = "%Y.%m.%d %H:%M:%S.%f"
    elif first_field.count(".") == 2:
        dt_format = "%Y.%m.%d %H:%M:%S"
    else:
        dt_format = "%Y-%m-%d %H:%M:%S"

    print(f"[FORMAT] sep='{sep}' | header={has_header} | dt_fmt='{dt_format}'")
    print(f"[FORMAT] Sample line: {data_line[:80]}")

    return {
        "sep":        sep,
        "has_header": has_header,
        "dt_format":  dt_format,
    }


def estimate_row_count(filepath: Path) -> int:
    """Fast row count estimate from file size."""
    size_bytes = filepath.stat().st_size
    # Dukascopy tick rows are ~45-55 bytes average
    return int(size_bytes / 50)


# ─────────────────────────────────────────────────────────────
# 2. CHUNKED TICK READER (memory-safe for 10GB+)
# ─────────────────────────────────────────────────────────────

def stream_tick_chunks(filepath: Path, fmt: dict):
    """
    Generator that yields cleaned tick DataFrames in chunks.
    Never loads the full file into memory.
    Each chunk has columns: [bid, ask, mid, spread, ask_vol, bid_vol, total_vol]
    Index: UTC datetime with millisecond precision.
    """
    header = 0 if fmt["has_header"] else None
    names  = DUKA_COLS if not fmt["has_header"] else None

    reader = pd.read_csv(
        filepath,
        sep        = fmt["sep"],
        header     = header,
        names      = names,
        dtype      = DUKA_DTYPES,
        chunksize  = CHUNK_SIZE,
        encoding   = "utf-8",
        on_bad_lines = "skip",   # skip malformed rows silently
    )

    total_rows    = 0
    rejected_rows = 0

    for i, chunk in enumerate(reader):
        # Rename columns if needed
        chunk.columns = [c.strip().lower().replace(" ", "_")
                         for c in chunk.columns]
        if "time" not in chunk.columns:
            chunk.columns = DUKA_COLS

        # Parse timestamp — as naive first, then localise EET → UTC.
        # Dukascopy exports in EET (UTC+2 winter / UTC+3 summer).
        # Passing utc=True directly would label EET times as UTC, shifting
        # every timestamp 2-3 hours forward — corrupting all session features.
        try:
            chunk["time"] = pd.to_datetime(
                chunk["time"].astype(str).str.strip(),
                format=fmt["dt_format"],
                errors="coerce",
            )
        except Exception:
            chunk["time"] = pd.to_datetime(
                chunk["time"].astype(str).str.strip(),
                format="mixed",
                errors="coerce",
            )

        # EET → UTC: handles US and UK DST transitions automatically.
        # ambiguous="NaT"  — drops the duplicated hour at autumn clock-back
        #                    (affects < 100 ticks/year, acceptable loss).
        # nonexistent="shift_forward" — maps the missing hour at spring
        #                    clock-forward to the next valid UTC instant.
        chunk["time"] = (
            chunk["time"]
            .dt.tz_localize(
                "Europe/Helsinki",
                ambiguous="NaT",
                nonexistent="shift_forward",
            )
            .dt.tz_convert("UTC")
        )

        # Drop rows with unparseable timestamps
        chunk.dropna(subset=["time"], inplace=True)
        chunk.set_index("time", inplace=True)
        chunk.sort_index(inplace=True)

        # ── Data quality filters ──────────────────────────
        n_before = len(chunk)

        # Price sanity bounds
        chunk = chunk[
            (chunk["bid"] >= PRICE_MIN) & (chunk["bid"] <= PRICE_MAX) &
            (chunk["ask"] >= PRICE_MIN) & (chunk["ask"] <= PRICE_MAX)
        ]

        # Ask must be >= bid (spread >= 0)
        chunk = chunk[chunk["ask"] >= chunk["bid"]]

        # Max spread filter (removes data errors)
        chunk["spread"] = chunk["ask"] - chunk["bid"]
        chunk = chunk[chunk["spread"] <= MAX_SPREAD]

        # Volume sanity — require total volume > 0 only.
        # The old filter (ask_volume > 0 AND bid_volume > 0) rejected valid
        # ticks where Dukascopy exports 0 on one side during one-sided flow
        # or thin off-hours markets. Total > 0 is the correct threshold.
        chunk["total_vol"] = chunk["ask_volume"] + chunk["bid_volume"]
        chunk = chunk[chunk["total_vol"] > 0]

        n_after    = len(chunk)
        rejected   = n_before - n_after
        rejected_rows += rejected
        total_rows    += n_after

        if len(chunk) == 0:
            continue

        # ── Derived tick columns ──────────────────────────
        chunk["mid"] = (chunk["bid"] + chunk["ask"]) / 2.0

        # Volume imbalance: +1 = all buy pressure, -1 = all sell pressure
        chunk["vol_imbalance"] = (
            (chunk["ask_volume"] - chunk["bid_volume"]) /
            (chunk["total_vol"] + 1e-10)
        )

        yield chunk.astype(np.float32, errors="ignore")

        if (i + 1) % 10 == 0:
            est_total = estimate_row_count(filepath)
            pct = min(100, total_rows / max(1, est_total) * 100)
            print(f"  Chunk {i+1} | {total_rows:,} ticks processed "
                  f"({pct:.1f}%) | rejected: {rejected_rows:,}")
            gc.collect()

    print(f"\n[OK] Streaming complete: {total_rows:,} clean ticks "
          f"| {rejected_rows:,} rejected ({rejected_rows/(total_rows+rejected_rows)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# 3. TRUE OHLCV BAR BUILDER FROM TICKS
# ─────────────────────────────────────────────────────────────

def ticks_to_ohlcv(tick_df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    """
    Resample tick data to OHLCV bars at any timeframe.
    Uses BID price for OHLC (industry standard for backtesting indices/forex).
    Adds tick-derived microstructure features unavailable from bar data.
    """
    freq = f"{tf_minutes}min"
    bid  = tick_df["bid"]
    mid  = tick_df["mid"]
    vol  = tick_df["total_vol"]
    spr  = tick_df["spread"]
    vimb = tick_df["vol_imbalance"]

    # True OHLCV from BID price
    ohlcv = pd.DataFrame({
        "Open":   bid.resample(freq).first(),
        "High":   bid.resample(freq).max(),
        "Low":    bid.resample(freq).min(),
        "Close":  bid.resample(freq).last(),
        "Volume": vol.resample(freq).sum(),
    })

    # ── Microstructure features (ONLY available from tick data) ──

    # True spread statistics per bar
    ohlcv["spread_mean"] = spr.resample(freq).mean()
    ohlcv["spread_max"]  = spr.resample(freq).max()
    ohlcv["spread_std"]  = spr.resample(freq).std()

    # Tick count per bar (liquidity proxy — more ticks = more liquid)
    ohlcv["tick_count"]  = bid.resample(freq).count()

    # Tick velocity: ticks per second in the bar
    ohlcv["tick_velocity"] = ohlcv["tick_count"] / (tf_minutes * 60)

    # VWAP (volume-weighted average price)
    vwap_num = (mid * vol).resample(freq).sum()
    vwap_den = vol.resample(freq).sum()
    ohlcv["vwap"] = vwap_num / (vwap_den + 1e-10)
    ohlcv["vwap_dist"] = (ohlcv["Close"] - ohlcv["vwap"]) / (ohlcv["vwap"] + 1e-10)

    # Volume imbalance per bar — net buy/sell pressure
    ohlcv["vol_imbalance"] = vimb.resample(freq).mean()

    # Price impact: price change per unit volume (large = thin market)
    price_change = (ohlcv["Close"] - ohlcv["Open"]).abs()
    ohlcv["price_impact"] = price_change / (ohlcv["Volume"] + 1e-10)

    # Bar return with true spread cost already embedded
    ohlcv["true_return"] = (
        (ohlcv["Close"] - ohlcv["Open"] - ohlcv["spread_mean"]) /
        (ohlcv["Open"] + 1e-10)
    )

    # Drop empty bars (outside market hours — US30 closes on weekends)
    ohlcv.dropna(subset=["Open", "Close"], inplace=True)
    ohlcv = ohlcv[ohlcv["Volume"] > 0]

    return ohlcv


# ─────────────────────────────────────────────────────────────
# 4. FULL FEATURE ENGINEERING ON TICK-DERIVED BARS
# ─────────────────────────────────────────────────────────────

def engineer_tick_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Adds full technical + microstructure feature set to tick-derived bars.
    Extends the features from phase2 with tick-exclusive signals.
    """
    d = ohlcv.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    # ── Standard technical features ──────────────────────
    # Returns
    d["ret1"]  = c.pct_change(1)
    d["ret5"]  = c.pct_change(5)
    d["ret20"] = c.pct_change(20)
    d["logr1"] = np.log(c / c.shift(1))
    d["hl_pct"] = (h - l) / c
    d["co_pct"] = (c - d["Open"]) / c

    # EMAs
    for n in [8, 21, 55, 200]:
        d[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    d["ema_x_8_21"]  = d["ema8"]  - d["ema21"]
    d["ema_x_21_55"] = d["ema21"] - d["ema55"]
    d["p_vs_21"]  = (c - d["ema21"])  / (d["ema21"]  + 1e-10)
    d["p_vs_55"]  = (c - d["ema55"])  / (d["ema55"]  + 1e-10)
    d["p_vs_200"] = (c - d["ema200"]) / (d["ema200"] + 1e-10)

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

    # Bollinger
    bm = c.rolling(20).mean()
    bs = c.rolling(20).std()
    d["bb_w"]   = (4 * bs) / (bm + 1e-10)
    d["bb_pct"] = (c - (bm - 2*bs)) / (4*bs + 1e-10)

    # ATR(14)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr14"]     = tr.rolling(14).mean()
    d["atr14_pct"] = d["atr14"] / (c + 1e-10)

    # Williams %R
    d["willr"] = -100 * (hi14 - c) / (hi14 - lo14 + 1e-10)

    # Volume
    vsma = v.rolling(20).mean()
    d["vol_ratio"] = v / (vsma + 1e-10)
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d["obv_norm"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-10)

    # ── Tick-exclusive microstructure features ───────────
    # Spread momentum: is spread widening (risk-off) or tightening?
    if "spread_mean" in d.columns:
        d["spread_ma5"]  = d["spread_mean"].rolling(5).mean()
        d["spread_mom"]  = d["spread_mean"] - d["spread_ma5"]
        d["spread_pct"]  = d["spread_mean"] / (c + 1e-10)   # spread as % of price
        d["spread_z"]    = (
            (d["spread_mean"] - d["spread_mean"].rolling(50).mean()) /
            (d["spread_mean"].rolling(50).std() + 1e-10)
        )

    # Tick velocity momentum: acceleration in activity
    if "tick_velocity" in d.columns:
        d["tv_ma10"]  = d["tick_velocity"].rolling(10).mean()
        d["tv_ratio"] = d["tick_velocity"] / (d["tv_ma10"] + 1e-10)
        d["tv_z"]     = (
            (d["tick_velocity"] - d["tick_velocity"].rolling(50).mean()) /
            (d["tick_velocity"].rolling(50).std() + 1e-10)
        )

    # Volume imbalance signal: sustained buy/sell pressure
    if "vol_imbalance" in d.columns:
        d["vimb_ma5"]  = d["vol_imbalance"].rolling(5).mean()
        d["vimb_ma20"] = d["vol_imbalance"].rolling(20).mean()
        d["vimb_diff"] = d["vimb_ma5"] - d["vimb_ma20"]   # short vs long pressure

    # VWAP distance dynamics
    if "vwap_dist" in d.columns:
        d["vwap_dist_ma"] = d["vwap_dist"].rolling(10).mean()
        d["vwap_cross"]   = np.sign(d["vwap_dist"])   # above/below VWAP

    # Price impact z-score: unusually high impact = low liquidity warning
    if "price_impact" in d.columns:
        d["impact_z"] = (
            (d["price_impact"] - d["price_impact"].rolling(50).mean()) /
            (d["price_impact"].rolling(50).std() + 1e-10)
        )

    # True return (spread-adjusted) momentum
    if "true_return" in d.columns:
        d["true_ret5"]  = d["true_return"].rolling(5).sum()
        d["true_ret20"] = d["true_return"].rolling(20).sum()

    # ── DST-aware session features ────────────────────────────────
    # Index is UTC (converted from EET in tick_pipeline).
    # Convert to local time per timezone so US and UK DST transitions
    # are handled independently. Without this, NYSE open appears at the
    # wrong UTC hour for ~4 months/year, mislabelling every bar in those
    # windows (late Oct and late Mar when UK/US clocks don't move together).
    us_idx  = d.index.tz_convert("America/New_York")
    uk_idx  = d.index.tz_convert("Europe/London")
    us_mins = us_idx.hour * 60 + us_idx.minute
    uk_mins = uk_idx.hour * 60 + uk_idx.minute

    # ── Binary session flags (hand-crafted priors) ───────────────────
    # NYSE session: 9:30 AM – 4:00 PM US Eastern
    d["is_us_open"]    = ((us_mins >= 570) & (us_mins < 960)).astype(np.int8)
    # Final 30 min of NYSE session (high-volume close window)
    d["is_us_close"]   = ((us_mins >= 930) & (us_mins < 960)).astype(np.int8)
    # Pre-market futures: 4:00 AM – 9:30 AM US Eastern
    d["is_premarket"]  = ((us_mins >= 240) & (us_mins < 570)).astype(np.int8)
    # London session: 8:00 AM – 5:00 PM UK time
    d["is_london"]     = ((uk_mins >= 480) & (uk_mins < 1020)).astype(np.int8)
    # London/NY overlap — both exchanges open simultaneously
    d["is_us_overlap"] = (d["is_us_open"] & d["is_london"]).astype(np.int8)
    # Asian / early-Frankfurt session: 2:00 AM – 8:00 AM UK time.
    # US30 futures trade through this window and often produce clean trends
    # driven by Asian macro flows before European participants enter.
    d["is_asian"]      = ((uk_mins >= 120) & (uk_mins < 480)).astype(np.int8)

    # ── Raw local-hour features (let ML discover its own session edges) ──
    # Giving the model the continuous UK/US hour means it can find that,
    # say, hours 3–6 UK have a different character than 6–8 without being
    # constrained by our hand-drawn session boundaries. Optuna can also
    # tune session thresholds as hyperparameters using these as inputs.
    d["uk_hour"] = uk_idx.hour.astype(np.int8)
    d["us_hour"] = us_idx.hour.astype(np.int8)

    # Cyclical encoding keeps midnight continuity for gradient-based models
    hr  = d.index.hour
    dow = d.index.dayofweek
    d["hour_sin"]    = np.sin(2 * np.pi * hr / 24).astype(np.float32)
    d["hour_cos"]    = np.cos(2 * np.pi * hr / 24).astype(np.float32)
    d["uk_hour_sin"] = np.sin(2 * np.pi * uk_idx.hour / 24).astype(np.float32)
    d["uk_hour_cos"] = np.cos(2 * np.pi * uk_idx.hour / 24).astype(np.float32)
    d["dow_sin"]     = np.sin(2 * np.pi * dow / 5).astype(np.float32)
    d["dow_cos"]     = np.cos(2 * np.pi * dow / 5).astype(np.float32)

    # ── Lag features ──────────────────────────────────────
    for lag in [1, 2, 3, 5, 8, 13]:
        d[f"r_lag{lag}"]    = d["logr1"].shift(lag)
        d[f"vimb_lag{lag}"] = d.get("vol_imbalance", pd.Series(0, index=d.index)).shift(lag)

    # ── Targets ───────────────────────────────────────────
    # Direction: did price go up on next bar? (classification)
    d["target"] = (c.shift(-1) > c).astype(np.int8)

    # Spread-adjusted return on next bar (regression — for RL reward)
    if "true_return" in d.columns:
        d["target_return"] = d["true_return"].shift(-1)
    else:
        d["target_return"] = d["logr1"].shift(-1)

    # HTF placeholders (filled in by phase2 engine at runtime)
    d["htf_bullish"]  = np.int8(0)
    d["htf_strength"] = np.float32(0.0)

    d.dropna(inplace=True)
    return d


# ─────────────────────────────────────────────────────────────
# 5. PARQUET WRITER (streaming, memory-safe)
# ─────────────────────────────────────────────────────────────

def write_ohlcv_parquet(symbol: str, tf_minutes: int,
                        ohlcv: pd.DataFrame):
    """Save OHLCV + microstructure features to Parquet."""
    path = DATA_DIR / f"{symbol}_{tf_minutes}m_ticks.parquet"
    ohlcv.to_parquet(path, engine="pyarrow", compression="snappy")
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[OK] Saved: {path.name} | {len(ohlcv):,} bars | {size_mb:.1f} MB")


def load_ohlcv_parquet(symbol: str, tf_minutes: int) -> pd.DataFrame:
    """Load previously built OHLCV Parquet file."""
    path = DATA_DIR / f"{symbol}_{tf_minutes}m_ticks.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    print(f"[OK] Loaded: {path.name} | {len(df):,} bars")
    return df


def parquet_exists(symbol: str, tf_minutes: int) -> bool:
    path = DATA_DIR / f"{symbol}_{tf_minutes}m_ticks.parquet"
    return path.exists()


# ─────────────────────────────────────────────────────────────
# 6. MAIN BUILD PIPELINE (streams full file once, builds all TFs)
# ─────────────────────────────────────────────────────────────

def build_all_timeframes(tick_file: Path, symbol: str,
                         timeframes: list = TARGET_TIMEFRAMES,
                         force_rebuild: bool = False):
    """
    Stream tick CSV once, accumulate bars for all target timeframes,
    engineer features, save to Parquet.

    Streams the file only once regardless of how many TFs you want.
    Memory usage is bounded by CHUNK_SIZE ticks at a time.
    """
    # Check if all TFs already built
    if not force_rebuild:
        all_built = all(parquet_exists(symbol, tf) for tf in timeframes)
        if all_built:
            print(f"[OK] All Parquet files already exist for {symbol}.")
            print("     Use force_rebuild=True to regenerate from ticks.")
            return

    print(f"\n{'='*60}")
    print(f"Building tick-derived OHLCV for {symbol}")
    print(f"Source: {tick_file}")
    print(f"Size  : {tick_file.stat().st_size / 1024**3:.2f} GB")
    print(f"TFs   : {timeframes}")
    print(f"{'='*60}\n")

    fmt = detect_csv_format(tick_file)

    # Accumulators: one rolling DataFrame per TF
    # We stream ticks and build bars incrementally
    tick_buffers = {tf: [] for tf in timeframes}
    bar_builders = {tf: [] for tf in timeframes}
    chunk_count  = 0

    print("Streaming tick data...")
    for tick_chunk in stream_tick_chunks(tick_file, fmt):
        chunk_count += 1

        for tf in timeframes:
            tick_buffers[tf].append(tick_chunk)

            # Process accumulated ticks into bars every 20 chunks
            # This bounds memory: never more than 10M ticks in RAM
            if chunk_count % 20 == 0:
                combined = pd.concat(tick_buffers[tf])
                combined.sort_index(inplace=True)

                # Keep last partial bar's ticks for next batch
                # (find last complete bar boundary)
                freq    = f"{tf}min"
                last_complete = combined.index[-1].floor(freq)
                mask_complete = combined.index < last_complete

                if mask_complete.any():
                    complete_ticks  = combined[mask_complete]
                    remainder_ticks = combined[~mask_complete]

                    bars = ticks_to_ohlcv(complete_ticks, tf)
                    if len(bars) > 0:
                        bar_builders[tf].append(bars)

                    tick_buffers[tf] = [remainder_ticks]
                else:
                    tick_buffers[tf] = [combined]

    # Final flush: process remaining ticks
    print("\nFlushing remaining ticks...")
    for tf in timeframes:
        if tick_buffers[tf]:
            remaining = pd.concat(tick_buffers[tf])
            remaining.sort_index(inplace=True)
            if len(remaining) > 0:
                bars = ticks_to_ohlcv(remaining, tf)
                if len(bars) > 0:
                    bar_builders[tf].append(bars)

    # Combine all bar batches, engineer features, save
    print("\nEngineering features and saving Parquet files...")
    results = {}
    for tf in timeframes:
        if not bar_builders[tf]:
            print(f"[WARN] No bars built for {symbol} {tf}m — skipping")
            continue

        all_bars = pd.concat(bar_builders[tf])
        all_bars.sort_index(inplace=True)

        # Remove duplicate bar timestamps (shouldn't happen but safety check)
        all_bars = all_bars[~all_bars.index.duplicated(keep="last")]

        print(f"\n  {symbol} {tf}m: {len(all_bars):,} bars | "
              f"{all_bars.index[0].date()} → {all_bars.index[-1].date()}")

        # Engineer features
        feat_df = engineer_tick_features(all_bars)
        print(f"  Features: {len([c for c in feat_df.columns if c not in ['Open','High','Low','Close','Volume','target','target_return']])} columns")

        # Save
        write_ohlcv_parquet(symbol, tf, feat_df)
        results[tf] = feat_df

        # Free memory before next TF
        del all_bars, feat_df
        gc.collect()

    print(f"\n{'='*60}")
    print(f"Build complete for {symbol}")
    print(f"{'='*60}")
    return results


# ─────────────────────────────────────────────────────────────
# 7. INCREMENTAL APPEND (add new months without full rebuild)
# ─────────────────────────────────────────────────────────────

def append_tick_data(new_csv_path: Path, symbol: str,
                     timeframes: list = TARGET_TIMEFRAMES):
    """
    Append a new tick CSV to existing Parquet files without a full rebuild.

    Typical use: you downloaded last month's ticks from Dukascopy.
    This processes only the new ticks (after the last bar in each Parquet)
    and appends new bars. Runs in ~2-10 min instead of 30-60 min.

    Handles overlaps automatically — if your new CSV starts a few days
    before the existing data ends, duplicate bars are dropped.

    Usage:
        python tick_pipeline.py --append US30_april2026.csv
    """
    new_csv_path = Path(new_csv_path)
    if not new_csv_path.exists():
        print(f"[ERROR] File not found: {new_csv_path}")
        return

    # Find the earliest last-bar timestamp across all TFs.
    # We use the minimum so we don't miss bars on any TF.
    cutoff = None
    for tf in timeframes:
        existing = load_ohlcv_parquet(symbol, tf)
        if not existing.empty:
            last_ts = existing.index[-1]
            if cutoff is None or last_ts < cutoff:
                cutoff = last_ts

    if cutoff is None:
        print("[INFO] No existing Parquets found — running full build instead.")
        build_all_timeframes(new_csv_path, symbol, timeframes)
        return

    print(f"\n{'='*60}")
    print(f"INCREMENTAL APPEND — {symbol}")
    print(f"New file : {new_csv_path.name}")
    print(f"Cutoff   : {cutoff}  (last bar in existing Parquets)")
    print(f"{'='*60}\n")

    # Stream new CSV, collect only ticks strictly after cutoff
    fmt = detect_csv_format(new_csv_path)
    new_tick_chunks = []
    total_new = 0
    for tick_chunk in stream_tick_chunks(new_csv_path, fmt):
        filtered = tick_chunk[tick_chunk.index > cutoff]
        if len(filtered) > 0:
            new_tick_chunks.append(filtered)
            total_new += len(filtered)

    if not new_tick_chunks:
        print(f"[INFO] No new ticks found after {cutoff.date()} — nothing to append.")
        return

    new_ticks = pd.concat(new_tick_chunks).sort_index()
    new_ticks = new_ticks[~new_ticks.index.duplicated(keep="last")]
    print(f"[INFO] {total_new:,} new ticks to process "
          f"({new_ticks.index[0].date()} → {new_ticks.index[-1].date()})\n")

    # Per-TF: build new bars, re-engineer with warmup history, append
    WARMUP_BARS = 250   # bars of existing history needed for indicator warmup

    for tf in timeframes:
        existing = load_ohlcv_parquet(symbol, tf)
        if existing.empty:
            print(f"  [{tf}m] No existing data — skipping")
            continue

        # Build raw OHLCV bars from the new ticks
        new_bars = ticks_to_ohlcv(new_ticks, tf)
        if new_bars.empty:
            print(f"  [{tf}m] No new complete bars")
            continue

        # Identify the raw (pre-feature) columns produced by ticks_to_ohlcv
        raw_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "spread_mean", "spread_max", "spread_std",
            "tick_count", "tick_velocity", "vwap", "vwap_dist",
            "vol_imbalance", "price_impact", "true_return",
        ]

        # Take the warmup tail from the existing Parquet (raw columns only)
        warmup = existing.iloc[-WARMUP_BARS:]
        warmup_raw = warmup[[c for c in raw_cols if c in warmup.columns]]

        # Combine warmup tail + new bars, then re-engineer features
        new_raw = new_bars[[c for c in raw_cols if c in new_bars.columns]]
        combined_raw = pd.concat([warmup_raw, new_raw]).sort_index()
        combined_raw = combined_raw[~combined_raw.index.duplicated(keep="last")]

        combined_feat = engineer_tick_features(combined_raw)

        # Keep only the bars that are genuinely new (after existing end)
        new_end = existing.index[-1]
        new_featured = combined_feat[combined_feat.index > new_end]

        if new_featured.empty:
            print(f"  [{tf}m] No new bars survived feature engineering")
            continue

        # Align columns: keep only columns present in both
        shared_cols = existing.columns.intersection(new_featured.columns)
        updated = pd.concat([existing[shared_cols],
                             new_featured[shared_cols]]).sort_index()
        updated = updated[~updated.index.duplicated(keep="last")]

        write_ohlcv_parquet(symbol, tf, updated)
        print(f"  [{tf}m] +{len(new_featured):,} bars appended | "
              f"total: {len(updated):,} | "
              f"now through: {updated.index[-1].date()}")

        del combined_raw, combined_feat, new_featured, updated
        gc.collect()

    print(f"\n{'='*60}")
    print("Append complete.")
    print("Next step: python train.py  (retrains on expanded dataset)")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────
# 8. MULTI-FILE SUPPORT (if data is split by month/year)
# ─────────────────────────────────────────────────────────────

def build_from_multiple_files(file_list: list, symbol: str,
                               timeframes: list = TARGET_TIMEFRAMES):
    """
    If your 75 months of data is split into multiple CSV files,
    this merges them on-the-fly without loading all into memory.
    Pass TICK_FILES list instead of single TICK_FILE.
    """
    print(f"Multi-file mode: {len(file_list)} files")
    all_results = {tf: [] for tf in timeframes}

    for i, filepath in enumerate(sorted(file_list)):
        filepath = Path(filepath)
        print(f"\nFile {i+1}/{len(file_list)}: {filepath.name}")
        fmt = detect_csv_format(filepath)

        for tick_chunk in stream_tick_chunks(filepath, fmt):
            for tf in timeframes:
                bars = ticks_to_ohlcv(tick_chunk, tf)
                if len(bars) > 0:
                    all_results[tf].append(bars)

    # Combine and save
    for tf in timeframes:
        if not all_results[tf]:
            continue
        combined = pd.concat(all_results[tf])
        combined.sort_index(inplace=True)
        combined = combined[~combined.index.duplicated(keep="last")]
        feat_df  = engineer_tick_features(combined)
        write_ohlcv_parquet(symbol, tf, feat_df)
        del combined, feat_df
        gc.collect()


# ─────────────────────────────────────────────────────────────
# 8. INTEGRATION WITH PHASE 2 ENGINE
# ─────────────────────────────────────────────────────────────

def load_for_training(symbol: str,
                      timeframes: list = TARGET_TIMEFRAMES) -> dict:
    """
    Load all built Parquet files ready for phase2_adaptive_engine.py.
    Returns dict keyed by TF minutes, values are feature DataFrames.

    Usage in phase2:
        from tick_pipeline import load_for_training
        tick_data = load_for_training("US30")
        # tick_data[5] = 5m bar DataFrame with all features
        # tick_data[1] = 1m bar DataFrame
        # etc.
    """
    data = {}
    for tf in timeframes:
        df = load_ohlcv_parquet(symbol, tf)
        if not df.empty:
            data[tf] = df
            print(f"  {symbol} {tf}m: {len(df):,} bars | "
                  f"{df.index[0].date()} → {df.index[-1].date()}")
    return data


def get_data_quality_report(symbol: str,
                             timeframes: list = TARGET_TIMEFRAMES) -> pd.DataFrame:
    """
    Print a quality report showing bar count, date range, missing bars,
    and microstructure feature availability per timeframe.
    """
    rows = []
    for tf in timeframes:
        df = load_ohlcv_parquet(symbol, tf)
        if df.empty:
            rows.append({"TF": f"{tf}m", "status": "NOT BUILT"})
            continue

        # Expected bars (US30 trades ~23.5h/day, 5 days/week)
        total_days  = (df.index[-1] - df.index[0]).days
        trading_days = total_days * 5 / 7
        expected_bars_per_day = (23.5 * 60) / tf
        expected_total = int(trading_days * expected_bars_per_day)
        actual_total   = len(df)
        coverage_pct   = min(100, actual_total / max(1, expected_total) * 100)

        has_micro = "tick_count" in df.columns and "vol_imbalance" in df.columns

        rows.append({
            "TF":           f"{tf}m",
            "bars":         f"{actual_total:,}",
            "from":         str(df.index[0].date()),
            "to":           str(df.index[-1].date()),
            "coverage":     f"{coverage_pct:.1f}%",
            "microstructure": "YES" if has_micro else "no",
            "features":     len([c for c in df.columns
                                 if c not in ["Open","High","Low","Close",
                                              "Volume","target","target_return"]]),
            "status":       "OK",
        })

    report = pd.DataFrame(rows)
    print(f"\n{'='*60}")
    print(f"DATA QUALITY REPORT — {symbol}")
    print(f"{'='*60}")
    print(report.to_string(index=False))
    print()
    return report


# ─────────────────────────────────────────────────────────────
# 9. ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Incremental append mode ──────────────────────────────────
    # python tick_pipeline.py --append path\to\new_month.csv
    if "--append" in sys.argv:
        idx = sys.argv.index("--append")
        if idx + 1 >= len(sys.argv):
            print("[ERROR] --append requires a file path argument.")
            print("  Usage: python tick_pipeline.py --append US30_april2026.csv")
            sys.exit(1)
        new_file = Path(sys.argv[idx + 1])
        append_tick_data(new_file, SYMBOL, TARGET_TIMEFRAMES)
        get_data_quality_report(SYMBOL, TARGET_TIMEFRAMES)
        sys.exit(0)

    # ── Full build mode (default) ────────────────────────────────
    if TICK_FILES:
        print("Multi-file mode detected.")
        build_from_multiple_files(TICK_FILES, SYMBOL, TARGET_TIMEFRAMES)
    else:
        if not TICK_FILE.exists():
            print(f"[ERROR] Tick file not found: {TICK_FILE}")
            print("  Please update TICK_FILE path at top of this script.")
            sys.exit(1)
        build_all_timeframes(TICK_FILE, SYMBOL, TARGET_TIMEFRAMES)

    # Quality report
    get_data_quality_report(SYMBOL, TARGET_TIMEFRAMES)

    # Show integration example
    print("\nReady to use in phase2_adaptive_engine.py:")
    print("  from tick_pipeline import load_for_training")
    print(f'  data = load_for_training("{SYMBOL}")')
    print("  # data[5]  = 5m DataFrame with 75+ months of tick-derived bars")
    print("  # data[1]  = 1m DataFrame")
    print("  # data[15] = 15m DataFrame (HTF confirmation)")
