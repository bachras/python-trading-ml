"""
Institutional Feature Engine
==============================
Adds ~45 institutional-grade features on top of existing tick-derived bars.
All features are computed from tick data — impossible to replicate from OHLCV alone.

Feature groups:
  1. VWAP family          — session, anchored, bands, slope, deviation
  2. Volume profile       — POC, VAH, VAL, HVN/LVN, distance, z-scores
  3. Order flow proxies   — quote_delta/quote_cvd (OHLCV-derived, not true tape)
  4. Liquidity structure  — swing levels, equal H/L pools, LVN voids, absorption
  5. Market regime        — trend/range/expanding/contracting classifier

Usage:
    from institutional_features import add_institutional_features
    enhanced_df = add_institutional_features(ohlcv_with_tick_cols)

    The input DataFrame must have come from tick_pipeline.py so it contains:
      bid_volume, ask_volume, vol_imbalance, tick_count, spread_mean, vwap, etc.
    Standard OHLCV columns (Open, High, Low, Close, Volume) also required.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

VALUE_AREA_PCT   = 0.70    # 70% of volume defines the Value Area
HVN_MULTIPLIER   = 1.5     # node is HVN if volume > median * this
LVN_MULTIPLIER   = 0.5     # node is LVN if volume < median * this
PROFILE_BINS     = 50      # price buckets per volume profile period
SWING_LOOKBACK   = 10      # bars each side to confirm swing high/low
EQUAL_LEVEL_TOL  = 0.0003  # 0.03% tolerance for "equal" highs/lows
STACKED_THRESH   = 3.0     # buy/sell ratio to flag imbalance (3:1)
STACKED_MIN_BARS = 3       # consecutive imbalanced bars = stacked
CVD_LOOKBACK     = 100     # bars for CVD normalisation window
VWAP_STD_WINDOW  = 20      # bars for rolling VWAP band calculation
REGIME_LOOKBACK  = 50      # bars for regime classification


# ─────────────────────────────────────────────────────────────
# 1. VWAP FAMILY
# ─────────────────────────────────────────────────────────────

def add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session VWAP   — resets each trading day (institutional benchmark)
    Anchored VWAP  — from rolling significant swing points
    VWAP bands     — 1σ, 2σ, 3σ standard deviation bands
    VWAP slope     — rate of change (trend direction of smart money)
    VWAP deviation — how far price has stretched from fair value

    All institutions use VWAP as their execution benchmark.
    Price above VWAP = institutions are net long on the day.
    Price returning to VWAP = mean reversion opportunity.
    Price breaking VWAP bands = stretched / exhausted move.
    """
    d     = df.copy()
    c     = d["Close"]
    v     = d["Volume"]
    tp    = (d["High"] + d["Low"] + d["Close"]) / 3.0   # typical price

    # ── Session VWAP (daily reset) ──────────────────────
    # Group by date, compute cumulative VWAP within each session
    dates       = d.index.date
    cum_tpv     = (tp * v).groupby(dates).cumsum()
    cum_vol     = v.groupby(dates).cumsum()
    d["vwap_session"] = cum_tpv / (cum_vol + 1e-10)

    # ── VWAP standard deviation bands ───────────────────
    # Rolling variance of (price - session VWAP)
    dev         = tp - d["vwap_session"]
    rolling_var = (dev ** 2 * v).groupby(dates).cumsum() / (cum_vol + 1e-10)
    vwap_std    = np.sqrt(rolling_var.clip(lower=0))

    d["vwap_upper1"] = d["vwap_session"] + 1.0 * vwap_std
    d["vwap_lower1"] = d["vwap_session"] - 1.0 * vwap_std
    d["vwap_upper2"] = d["vwap_session"] + 2.0 * vwap_std
    d["vwap_lower2"] = d["vwap_session"] - 2.0 * vwap_std
    d["vwap_upper3"] = d["vwap_session"] + 3.0 * vwap_std
    d["vwap_lower3"] = d["vwap_session"] - 3.0 * vwap_std

    # ── VWAP deviation features (ML-ready normalised) ───
    # Distance from session VWAP as fraction of ATR
    atr   = d.get("atr14", (d["High"] - d["Low"]).rolling(14).mean())
    d["vwap_dist_atr"]  = (c - d["vwap_session"]) / (atr + 1e-10)
    d["vwap_dist_pct"]  = (c - d["vwap_session"]) / (d["vwap_session"] + 1e-10)

    # Which band is price in? (0=below lower2, 1=lower1-2, 2=near VWAP,
    #                          3=upper1-2, 4=above upper2)
    d["vwap_band_pos"] = pd.cut(
        d["vwap_dist_atr"],
        bins=[-np.inf, -2, -1, 1, 2, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # ── VWAP slope — rate of change over N bars ──────────
    d["vwap_slope5"]  = d["vwap_session"].diff(5)  / (d["vwap_session"].shift(5) + 1e-10)
    d["vwap_slope20"] = d["vwap_session"].diff(20) / (d["vwap_session"].shift(20) + 1e-10)

    # ── Anchored VWAP from rolling 20-bar high/low ───────
    # Simulates "anchored from recent significant pivot"
    # Uses the highest-volume bar in past 20 as anchor
    # fillna(0) before astype(int): rolling(20) returns NaN for first 19 bars,
    # and pandas 2.x raises if you cast NaN directly to int.
    roll_maxvol_idx = v.rolling(20).apply(lambda x: x.argmax(), raw=True).fillna(0).astype(int)

    anchored_vwap = pd.Series(np.nan, index=d.index)
    for i in range(20, len(d)):
        anchor_offset  = int(roll_maxvol_idx.iloc[i])
        anchor_i       = i - (19 - anchor_offset)
        if anchor_i < 0:
            continue
        slice_tp  = tp.iloc[anchor_i:i+1]
        slice_vol = v.iloc[anchor_i:i+1]
        anchored_vwap.iloc[i] = (slice_tp * slice_vol).sum() / (slice_vol.sum() + 1e-10)

    d["vwap_anchored"]     = anchored_vwap
    d["vwap_anch_dist_pct"] = (c - d["vwap_anchored"]) / (d["vwap_anchored"] + 1e-10)

    # ── VWAP reclaim / rejection signals ─────────────────
    # vwap_crossed: crossing EVENT this bar
    #   +1 = crossed above VWAP (bullish reclaim)
    #   -1 = crossed below VWAP (bearish rejection)
    #    0 = no cross
    # Named "vwap_crossed" (not "vwap_cross") to avoid collision with
    # tick_pipeline's vwap_cross = np.sign(vwap_dist) (position above/below).
    above_now  = (c >= d["vwap_session"]).astype(int)
    above_prev = above_now.shift(1)
    d["vwap_crossed"] = (above_now - above_prev).fillna(0).astype(int)

    return d


# ─────────────────────────────────────────────────────────────
# 2. VOLUME PROFILE — POC, VAH, VAL, HVN, LVN
# ─────────────────────────────────────────────────────────────

def compute_volume_profile(prices: np.ndarray, volumes: np.ndarray,
                           n_bins: int = PROFILE_BINS) -> dict:
    """
    Build a volume profile for a price/volume array.
    Returns POC, VAH, VAL, HVN list, LVN list, and the full histogram.

    Based on Auction Market Theory:
    - POC = price level with most volume = fair value / magnet
    - Value Area = 70% of volume = institutional acceptance zone
    - HVN = resistance/support (institutions defended this level)
    - LVN = fast-move zones (price passes through quickly)
    """
    if len(prices) < 10:
        return {}

    p_min, p_max = prices.min(), prices.max()
    if p_min == p_max:
        return {"poc": p_min, "vah": p_max, "val": p_min}

    bin_edges  = np.linspace(p_min, p_max, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each tick to a bin and sum volumes
    bin_idx    = np.clip(
        np.digitize(prices, bin_edges) - 1, 0, n_bins - 1
    )
    vol_profile = np.zeros(n_bins)
    for i, vi in zip(bin_idx, volumes):
        vol_profile[i] += vi

    total_vol = vol_profile.sum()
    if total_vol == 0:
        return {}

    # POC = bin with maximum volume
    poc_bin = vol_profile.argmax()
    poc     = float(bin_centres[poc_bin])

    # Value Area: expand outward from POC until 70% of volume accumulated
    target_vol = total_vol * VALUE_AREA_PCT
    accumulated = vol_profile[poc_bin]
    lo_idx, hi_idx = poc_bin, poc_bin

    while accumulated < target_vol:
        # Expand toward whichever neighbour has more volume
        lo_next = lo_idx - 1
        hi_next = hi_idx + 1
        lo_vol  = vol_profile[lo_next] if lo_next >= 0      else -1
        hi_vol  = vol_profile[hi_next] if hi_next < n_bins  else -1

        if lo_vol <= 0 and hi_vol <= 0:
            break
        if lo_vol >= hi_vol:
            lo_idx = lo_next
            accumulated += lo_vol
        else:
            hi_idx = hi_next
            accumulated += hi_vol

    vah = float(bin_centres[hi_idx])
    val = float(bin_centres[lo_idx])

    # HVN / LVN identification
    median_vol = float(np.median(vol_profile[vol_profile > 0]))
    hvn_prices = bin_centres[vol_profile > median_vol * HVN_MULTIPLIER].tolist()
    lvn_prices = bin_centres[
        (vol_profile > 0) & (vol_profile < median_vol * LVN_MULTIPLIER)
    ].tolist()

    return {
        "poc":        poc,
        "vah":        vah,
        "val":        val,
        "hvn_prices": hvn_prices,
        "lvn_prices": lvn_prices,
        "profile":    vol_profile,
        "bin_centres": bin_centres,
        "total_vol":  total_vol,
    }


def add_volume_profile_features(df: pd.DataFrame,
                                 session_bars: int = 390) -> pd.DataFrame:
    """
    Session-level volume profile: resets at each calendar day open.

    For each bar, only bars from the same calendar day up to (and including)
    the current bar contribute — matches VWAP's groupby-date pattern.
    No rolling window bleed across sessions.

    ML features derived from profile:
      - Distance to POC (normalised by ATR)
      - Price position relative to VAH/VAL
      - Distance to nearest HVN and LVN
      - Value area width (measures market agreement/disagreement)
      - POC migration (is institutional fair value shifting?)
    """
    d   = df.copy()
    c   = d["Close"].values
    h   = d["High"].values
    l   = d["Low"].values
    v   = d["Volume"].values
    atr = d.get("atr14", pd.Series(
        (d["High"] - d["Low"]).rolling(14).mean(), index=d.index
    )).values

    n = len(d)
    poc_arr  = np.full(n, np.nan)
    vah_arr  = np.full(n, np.nan)
    val_arr  = np.full(n, np.nan)
    va_width = np.full(n, np.nan)
    poc_dist = np.full(n, np.nan)
    vah_dist = np.full(n, np.nan)
    val_dist = np.full(n, np.nan)
    hvn_dist = np.full(n, np.nan)
    lvn_dist = np.full(n, np.nan)
    in_va    = np.full(n, np.nan)   # 1=inside value area, 0=outside

    # Group bar indices by calendar date — same pattern as VWAP
    dates        = d.index.date
    unique_dates = sorted(set(dates))

    for day in unique_dates:
        day_idxs = np.where(dates == day)[0]
        if len(day_idxs) < 10:
            continue

        # Accumulate within session: profile grows as the day progresses.
        # Profile at bar k uses only bars [0..k] of this session — zero lookahead.
        for k, idx in enumerate(day_idxs):
            seg = day_idxs[:k + 1]
            if len(seg) < 10:
                continue
            p_slice = (h[seg] + l[seg]) / 2.0
            v_slice = v[seg]

            profile = compute_volume_profile(p_slice, v_slice)
            if not profile:
                continue

            poc = profile["poc"]
            vah = profile["vah"]
            val = profile["val"]
            a   = atr[idx] if atr[idx] > 0 else c[idx] * 0.001

            poc_arr[idx] = poc
            vah_arr[idx] = vah
            val_arr[idx] = val
            va_width[idx] = (vah - val) / a
            poc_dist[idx] = (c[idx] - poc) / a
            vah_dist[idx] = (c[idx] - vah) / a
            val_dist[idx] = (c[idx] - val) / a
            in_va[idx]    = 1.0 if val <= c[idx] <= vah else 0.0

            hvns = profile.get("hvn_prices", [])
            if hvns:
                hvn_dist[idx] = min(abs(c[idx] - p) / a for p in hvns)

            lvns = profile.get("lvn_prices", [])
            if lvns:
                lvn_dist[idx] = min(abs(c[idx] - p) / a for p in lvns)

    d["vp_poc"]       = poc_arr
    d["vp_vah"]       = vah_arr
    d["vp_val"]       = val_arr
    d["vp_va_width"]  = va_width
    d["vp_poc_dist"]  = poc_dist      # negative = below POC
    d["vp_vah_dist"]  = vah_dist      # negative = below VAH
    d["vp_val_dist"]  = val_dist      # positive = above VAL
    d["vp_in_va"]     = in_va
    d["vp_hvn_dist"]  = hvn_dist      # distance to nearest HVN
    d["vp_lvn_dist"]  = lvn_dist      # distance to nearest LVN (fast-move zone)

    # POC migration: how fast is institutional fair value shifting?
    poc_series = pd.Series(poc_arr, index=d.index)
    d["vp_poc_migration"] = poc_series.diff(5) / (d["Close"].shift(5) + 1e-10)

    # Value area breakout signal
    # +1 = broke above VAH (potential breakout long)
    # -1 = broke below VAL (potential breakout short)
    # 0  = inside or no break
    vah_series = pd.Series(vah_arr, index=d.index)
    val_series = pd.Series(val_arr, index=d.index)
    above_vah  = (d["Close"] > vah_series).astype(int)
    below_val  = (d["Close"] < val_series).astype(int)
    d["vp_va_breakout"] = above_vah - below_val

    return d


# ─────────────────────────────────────────────────────────────
# 3. ORDER FLOW / QUOTE DELTA PROXIES
# ─────────────────────────────────────────────────────────────

def add_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV-proxy order flow features. Columns are prefixed `quote_` to signal
    they are derived from OHLCV volume, NOT true L2/tape order flow.

    quote_delta = ask_volume - bid_volume per bar.
    When tick data is unavailable: ask_vol = bid_vol = Volume * 0.5,
    so quote_delta collapses to zero and CVD carries no directional signal.
    The signal is only meaningful when real tick data is piped via tick_pipeline.

    Key signals (when tick data is real):
    - quote_delta_divergence: price up but delta falling = bullish exhaustion
    - stacked_imbalance: 3+ consecutive bars with 3:1 vol ratio = institutional push
    - quote_cvd trend: cumulative delta trending = sustained institutional direction
    - absorption: high volume but small price move = large limit orders absorbing
    """
    d = df.copy()

    # Require tick-derived columns from tick_pipeline.py
    ask_vol = d.get("ask_volume", d["Volume"] * 0.5)
    bid_vol = d.get("bid_volume", d["Volume"] * 0.5)

    # ── Quote delta per bar ───────────────────────────────
    # NOTE: "quote_delta" = ask_vol - bid_vol estimated from OHLCV volume.
    # This is NOT true tape delta from L2 order flow. When tick data is absent,
    # ask_vol = bid_vol = Volume * 0.5, making these proxies symmetrically zero.
    # The `quote_` prefix signals OHLCV-proxy, not exchange-feed order flow.
    d["quote_delta"]     = ask_vol - bid_vol
    d["quote_delta_pct"] = d["quote_delta"] / (d["Volume"] + 1e-10)   # normalised [-1, 1]

    # Quote delta magnitude z-score (unusually large delta = institutional)
    d["quote_delta_z"] = (
        (d["quote_delta"] - d["quote_delta"].rolling(CVD_LOOKBACK).mean()) /
        (d["quote_delta"].rolling(CVD_LOOKBACK).std() + 1e-10)
    )

    # ── Quote CVD (Cumulative Volume Delta) ───────────────
    # Resets each session (daily)
    dates        = d.index.date
    d["quote_cvd"] = d["quote_delta"].groupby(dates).cumsum()

    # Normalised quote CVD (z-score within rolling window)
    d["quote_cvd_z"] = (
        (d["quote_cvd"] - d["quote_cvd"].rolling(CVD_LOOKBACK).mean()) /
        (d["quote_cvd"].rolling(CVD_LOOKBACK).std() + 1e-10)
    )

    # Quote CVD slope: is buying/selling pressure accelerating?
    d["quote_cvd_slope5"]  = d["quote_cvd"].diff(5)
    d["quote_cvd_slope20"] = d["quote_cvd"].diff(20)

    # ── Quote delta divergence ────────────────────────────
    # Price makes higher high but delta makes lower high → exhaustion
    # Compute as: sign(price_change_5) != sign(delta_change_5)
    price_dir = np.sign(d["Close"].diff(5))
    delta_dir = np.sign(d["quote_delta"].diff(5))
    d["quote_delta_divergence"] = (price_dir != delta_dir).astype(float)

    # Quote delta momentum vs price momentum (key divergence signal)
    # If delta is near-zero everywhere (e.g. ask/bid estimated as 50/50),
    # rolling corr returns NaN for all rows — fill with 0 (neutral) instead.
    _corr = d["quote_delta"].rolling(20).corr(d["Close"].diff())
    d["quote_delta_price_corr"] = _corr.fillna(0.0)

    # ── Stacked imbalance detection ───────────────────────
    # A "stacked imbalance" is 3+ consecutive bars where
    # ask_vol / bid_vol > STACKED_THRESH (buy-side) OR
    # bid_vol / ask_vol > STACKED_THRESH (sell-side)

    buy_imb  = (ask_vol / (bid_vol + 1e-10) >= STACKED_THRESH).astype(int)
    sell_imb = (bid_vol / (ask_vol + 1e-10) >= STACKED_THRESH).astype(int)

    # Count consecutive imbalances using rolling sum
    d["buy_imbalance_count"]  = buy_imb.rolling(STACKED_MIN_BARS).sum()
    d["sell_imbalance_count"] = sell_imb.rolling(STACKED_MIN_BARS).sum()

    # Stacked imbalance flag: +1=buy stack, -1=sell stack, 0=none
    d["stacked_imbalance"] = np.where(
        d["buy_imbalance_count"]  >= STACKED_MIN_BARS,  1.0,
        np.where(
        d["sell_imbalance_count"] >= STACKED_MIN_BARS, -1.0,
        0.0)
    )

    # ── Absorption detection ──────────────────────────────
    # High volume + small price move = absorption by limit orders
    # (institutions absorbing retail market orders at a key level)
    price_move  = (d["High"] - d["Low"])
    atr         = d.get("atr14", price_move.rolling(14).mean())
    vol_z       = (d["Volume"] - d["Volume"].rolling(50).mean()) / \
                  (d["Volume"].rolling(50).std() + 1e-10)
    move_ratio  = price_move / (atr + 1e-10)

    # Absorption = high volume (z > 1.5) but small range (< 0.5 ATR)
    d["absorption"] = ((vol_z > 1.5) & (move_ratio < 0.5)).astype(float)

    # Absorption direction: which side was defending?
    d["absorption_bull"] = (d["absorption"] * (d["quote_delta"] > 0)).astype(float)
    d["absorption_bear"] = (d["absorption"] * (d["quote_delta"] < 0)).astype(float)

    # ── Failed auction signal ─────────────────────────────
    # Price makes new high/low but closes back inside range = failed auction
    # Institutions rejected the new price level
    prev_high  = d["High"].shift(1)
    prev_low   = d["Low"].shift(1)
    new_high   = d["High"] > prev_high
    new_low    = d["Low"]  < prev_low
    close_back_h = d["Close"] < prev_high    # closed back below prior high
    close_back_l = d["Close"] > prev_low     # closed back above prior low

    d["failed_auction_up"]   = (new_high & close_back_h).astype(float)
    d["failed_auction_down"] = (new_low  & close_back_l).astype(float)
    d["failed_auction"] = d["failed_auction_up"] - d["failed_auction_down"]

    return d


# ─────────────────────────────────────────────────────────────
# 4. LIQUIDITY STRUCTURE
# ─────────────────────────────────────────────────────────────

def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps where retail stop orders are likely clustered.
    Institutions know where stops sit and often run them.

    Key concepts:
    - Swing highs/lows: most retail traders place stops just beyond these
    - Equal highs/lows: double/triple tops/bottoms = liquidity pools
    - LVN voids: price gaps in the volume profile = fast-move zones
    - Stop hunt detection: price briefly spikes past swing, then reverses
    """
    d = df.copy()
    c = d["Close"]
    h = d["High"]
    l = d["Low"]
    atr = d.get("atr14", (h - l).rolling(14).mean())

    # ── Swing highs and lows ──────────────────────────────
    # A swing high: higher than N bars on both sides
    # A swing low:  lower than N bars on both sides
    N = SWING_LOOKBACK

    swing_high = pd.Series(False, index=d.index)
    swing_low  = pd.Series(False, index=d.index)

    h_arr = h.values
    l_arr = l.values

    for i in range(N, len(d) - 1):
        window_h_left  = h_arr[i-N:i]
        window_l_left  = l_arr[i-N:i]
        # G4: 1-bar right lookahead only (was N-bar right window — forward-looking bias).
        # Swing high confirmed when current bar is the highest of N left bars AND above
        # the immediate next bar only. This still confirms local peaks without future leakage.
        if h_arr[i] > max(window_h_left) and h_arr[i] > h_arr[i + 1]:
            swing_high.iloc[i] = True
        if l_arr[i] < min(window_l_left) and l_arr[i] < l_arr[i + 1]:
            swing_low.iloc[i] = True

    d["is_swing_high"] = swing_high.astype(float)
    d["is_swing_low"]  = swing_low.astype(float)

    # Distance to nearest recent swing high/low (normalised by ATR)
    # Look back up to 100 bars for the nearest confirmed swing
    swing_high_prices = h.where(swing_high)
    swing_low_prices  = l.where(swing_low)

    recent_swing_high = swing_high_prices.rolling(100, min_periods=1).max()
    recent_swing_low  = swing_low_prices.rolling(100, min_periods=1).min()

    d["dist_swing_high"] = (recent_swing_high - c) / (atr + 1e-10)
    d["dist_swing_low"]  = (c - recent_swing_low)  / (atr + 1e-10)

    # ── Equal highs / lows (liquidity pools) ─────────────
    # When price makes two or more highs at nearly the same level,
    # retail traders place stops just above — this is a liquidity pool.
    tol = c * EQUAL_LEVEL_TOL

    # Rolling equal high: current high within tolerance of recent swing high
    recent_sh = swing_high_prices.ffill()
    d["equal_high"] = (
        (recent_sh.notna()) &
        ((h - recent_sh).abs() < tol)
    ).astype(float)

    recent_sl = swing_low_prices.ffill()
    d["equal_low"] = (
        (recent_sl.notna()) &
        ((l - recent_sl).abs() < tol)
    ).astype(float)

    # ── Stop hunt detection ───────────────────────────────
    # Pattern: price spikes past swing H/L but closes back inside
    # Classic institutional stop hunt: take retail stops, then reverse

    # Upward stop hunt: high exceeded prior swing high, close below it
    stop_hunt_up = (
        (h > recent_swing_high.shift(1)) &
        (c < recent_swing_high.shift(1))
    ).astype(float)

    # Downward stop hunt: low exceeded prior swing low, close above it
    stop_hunt_down = (
        (l < recent_swing_low.shift(1)) &
        (c > recent_swing_low.shift(1))
    ).astype(float)

    d["stop_hunt"]      = stop_hunt_up - stop_hunt_down
    d["stop_hunt_up"]   = stop_hunt_up
    d["stop_hunt_down"] = stop_hunt_down

    # ── Liquidity void proximity ──────────────────────────
    # LVN from volume profile already computed in vp_lvn_dist
    # Here we add a directional flag: is price heading INTO a void?
    if "vp_lvn_dist" in d.columns:
        # Price moving toward void = potential fast move
        lvn_dist_change = d["vp_lvn_dist"].diff(3)
        d["approaching_lvn"] = (lvn_dist_change < 0).astype(float)

    # ── Liquidity grab magnitude ──────────────────────────
    # How far did price reach beyond swing before reversing?
    # Larger grab = stronger institutional presence
    wick_up   = h - d[["Open", "Close"]].max(axis=1)
    wick_down = d[["Open", "Close"]].min(axis=1) - l
    d["wick_up_atr"]   = wick_up   / (atr + 1e-10)
    d["wick_down_atr"] = wick_down / (atr + 1e-10)

    # Significant wick (> 0.5 ATR) with small body = pin bar / liquidity grab
    body   = (d["Close"] - d["Open"]).abs()
    d["pin_bar_bull"] = ((wick_down > atr * 0.5) & (body < atr * 0.3)).astype(float)
    d["pin_bar_bear"] = ((wick_up   > atr * 0.5) & (body < atr * 0.3)).astype(float)

    return d


# ─────────────────────────────────────────────────────────────
# 5. MARKET REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies the current market state into one of 6 regimes.
    This is fed to the ML models so they can weight features correctly
    by regime — e.g. momentum features matter more in trending regimes,
    mean-reversion features in ranging regimes.

    Regimes:
      0 = low volatility range     (small ATR, no direction)
      1 = high volatility range    (large ATR, no direction — choppy)
      2 = bullish trend            (upward EMA slope, strong quote_delta)
      3 = bearish trend            (downward EMA slope, strong quote_delta)
      4 = expanding volatility     (ATR increasing rapidly)
      5 = contracting volatility   (ATR decreasing — squeeze forming)
    """
    d   = df.copy()
    c   = d["Close"]
    atr = d.get("atr14", (d["High"] - d["Low"]).rolling(14).mean())
    n   = REGIME_LOOKBACK

    # Directional metrics
    ema21   = c.ewm(span=21, adjust=False).mean()
    ema55   = c.ewm(span=55, adjust=False).mean()
    slope21 = (ema21 - ema21.shift(n)) / (ema21.shift(n) + 1e-10)

    # Volatility metrics
    atr_sma       = atr.rolling(n).mean()
    atr_ratio     = atr / (atr_sma + 1e-10)   # > 1 = expanding, < 1 = contracting
    atr_slope     = atr.diff(10) / (atr.shift(10) + 1e-10)
    hv20          = d.get("hv20", c.pct_change().rolling(20).std() * np.sqrt(252))
    # Rolling percentile rank within a 252-bar window — no future leakage.
    # The old rank(pct=True) ranked against the full series including test
    # data, encoding future volatility context into training features.
    hv20_pct      = hv20.rolling(252).rank(pct=True)

    # Trend strength: ADX proxy (directional movement index)
    high_diff = d["High"].diff()
    low_diff  = d["Low"].diff().abs()
    dm_plus   = high_diff.where(high_diff > low_diff, 0).clip(lower=0)
    dm_minus  = low_diff.where(low_diff > high_diff, 0).clip(lower=0)
    atr14_raw = atr.clip(lower=1e-10)
    di_plus   = (dm_plus.rolling(14).mean()  / atr14_raw) * 100
    di_minus  = (dm_minus.rolling(14).mean() / atr14_raw) * 100
    dx        = (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10) * 100
    d["adx"]  = dx.rolling(14).mean()

    # Raw regime signals
    is_trending   = d["adx"] > 25
    is_bull_trend = is_trending & (ema21 > ema55) & (slope21 > 0)
    is_bear_trend = is_trending & (ema21 < ema55) & (slope21 < 0)
    is_expanding  = atr_slope > 0.02
    is_contracting = atr_slope < -0.02
    is_high_vol   = hv20_pct > 0.75
    is_low_vol    = hv20_pct < 0.25

    # Regime integer label (for ML embedding)
    regime = pd.Series(0, index=d.index, dtype=int)
    regime[is_contracting]              = 5
    regime[is_expanding]                = 4
    regime[is_bear_trend]               = 3
    regime[is_bull_trend]               = 2
    regime[~is_trending & is_high_vol]  = 1
    regime[~is_trending & is_low_vol]   = 0

    d["regime"]         = regime
    d["regime_trending"]    = is_trending.astype(float)
    d["regime_bull"]        = is_bull_trend.astype(float)
    d["regime_bear"]        = is_bear_trend.astype(float)
    d["regime_expanding"]   = is_expanding.astype(float)
    d["regime_contracting"] = is_contracting.astype(float)
    d["regime_high_vol"]    = is_high_vol.astype(float)
    d["adx_norm"]           = d["adx"] / 100.0

    # Regime cyclical encoding (so ML understands ordinal proximity)
    d["regime_sin"] = np.sin(2 * np.pi * regime / 6)
    d["regime_cos"] = np.cos(2 * np.pi * regime / 6)

    # Volatility context
    d["atr_ratio"]  = atr_ratio
    d["atr_slope"]  = atr_slope
    d["hv20_pct"]   = hv20_pct

    return d


# ─────────────────────────────────────────────────────────────
# 6. SESSION OPEN RANGE (ORB) — US30 specific
# ─────────────────────────────────────────────────────────────

def add_session_open_range(df: pd.DataFrame,
                           open_range_bars: int = 5) -> pd.DataFrame:
    """
    Opening Range Breakout features for US30 (NYSE open = 14:30 UTC).
    The first N bars after open define the range.
    Institutions watch this range closely — breakouts and fades are key setups.
    """
    d = df.copy()

    # NYSE open is 9:30 AM US Eastern. Convert UTC index → ET so DST is
    # handled automatically: open appears at 14:30 UTC in winter (EST) and
    # 13:30 UTC in summer (EDT). The old hardcoded 13:30 UTC was wrong for
    # ~4 months/year.
    _us_idx = (d.index if d.index.tz is not None else d.index.tz_localize("UTC")
               ).tz_convert("America/New_York")
    # For 1m/5m: look for exact 9:30+ bars. For coarser TFs (10m, 15m, 60m)
    # there may be no bar with minute==30, so fall back to any bar in the 9 AM
    # ET hour that covers the open period.
    is_open_bar = (_us_idx.hour == 9) & (_us_idx.minute >= 30)
    if not is_open_bar.any():
        is_open_bar = _us_idx.hour == 9  # e.g. 60m bars land at 9:00 ET

    # Mark the first N bars after open each day
    dates = d.index.date
    session_high = pd.Series(np.nan, index=d.index)
    session_low  = pd.Series(np.nan, index=d.index)

    for date in np.unique(dates):
        day_mask = np.array(dates) == date
        day_df   = d[day_mask]

        open_bars = day_df[is_open_bar[day_mask]]
        if len(open_bars) == 0:
            continue

        or_range = open_bars.iloc[:open_range_bars]
        or_high  = or_range["High"].max()
        or_low   = or_range["Low"].min()

        session_high[day_mask] = or_high
        session_low[day_mask]  = or_low

    d["orb_high"]      = session_high
    d["orb_low"]       = session_low
    d["orb_range"]     = session_high - session_low

    atr = d.get("atr14", (d["High"] - d["Low"]).rolling(14).mean())
    d["orb_dist_high"] = (d["Close"] - d["orb_high"]) / (atr + 1e-10)
    d["orb_dist_low"]  = (d["Close"] - d["orb_low"])  / (atr + 1e-10)
    d["orb_above"]     = (d["Close"] > d["orb_high"]).astype(float)
    d["orb_below"]     = (d["Close"] < d["orb_low"]).astype(float)
    d["orb_inside"]    = (
        (d["Close"] >= d["orb_low"]) & (d["Close"] <= d["orb_high"])
    ).astype(float)

    return d


# ─────────────────────────────────────────────────────────────
# 7. MASTER FUNCTION — adds all institutional features
# ─────────────────────────────────────────────────────────────

TICK_MICROSTRUCTURE_FEATURES = [
    "vwap_slope5", "vwap_slope20",
    "vp_poc_migration",
    "quote_delta", "quote_delta_z",
    "quote_cvd", "quote_cvd_z",
    "quote_cvd_slope5", "quote_cvd_slope20",
    "quote_delta_price_corr",
    "buy_imbalance_count", "sell_imbalance_count",
    "stacked_imbalance", "absorption_bull", "absorption_bear",
    # G8: pct-based VWAP/delta features also collapse to near-zero std at 3m+ (microstructure artefacts)
    "vwap_dist_pct", "vwap_anch_dist_pct", "quote_delta_pct",
]
_TICK_MICROSTRUCTURE_SET = frozenset(TICK_MICROSTRUCTURE_FEATURES)


def add_institutional_features(df: pd.DataFrame,
                                session_bars: int = 390,
                                open_range_bars: int = 5,
                                tf_minutes: int = 1,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Master function — applies all 5 institutional feature groups
    in the correct dependency order.

    Input : tick-derived OHLCV DataFrame from tick_pipeline.py
            (must have Volume, bid/ask volume columns ideally)
    Output: same DataFrame with ~45 additional institutional features

    Call this AFTER tick_pipeline.engineer_tick_features() so ATR14,
    HV20, etc. are already present.

    tf_minutes: bar timeframe in minutes.  Features in _TICK_ONLY_FEATURES
    are dropped for tf_minutes > 1 — they collapse to near-zero std at coarser
    bars and cause StandardScaler to produce arbitrarily large scaled values.
    """
    if verbose:
        before = len(df.columns)
        print(f"Adding institutional features to {len(df):,} bars...")

    d = df.copy()

    def _check_step(d_before, d_after, step_name):
        """Log any columns added by this step that are entirely NaN."""
        new_cols = [c for c in d_after.columns if c not in d_before.columns]
        bad = [c for c in new_cols if d_after[c].isna().all()]
        if bad:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"  institutional [{step_name}]: {len(bad)} all-NaN columns: {bad}"
            )
        return d_after

    before_cols = set(d.columns)

    # 1. VWAP family
    if verbose: print("  [1/5] VWAP family...")
    d = _check_step(d, add_vwap_features(d), "VWAP")

    # 2. Volume profile (POC, VAH, VAL, HVN, LVN)
    if verbose: print("  [2/5] Volume profile (POC / VAH / VAL / HVN / LVN)...")
    d = _check_step(d, add_volume_profile_features(d, session_bars=session_bars), "VolumeProfile")

    # 3. Order flow / delta
    if verbose: print("  [3/5] Order flow & delta...")
    d = _check_step(d, add_order_flow_features(d), "OrderFlow")

    # 4. Liquidity structure
    if verbose: print("  [4/5] Liquidity structure...")
    d = _check_step(d, add_liquidity_features(d), "Liquidity")

    # 5. Market regime
    if verbose: print("  [5/5] Market regime classifier...")
    d = _check_step(d, add_regime_features(d), "Regime")

    # 6. Session open range (US30)
    if verbose: print("  [6/6] Session open range (ORB)...")
    d = _check_step(d, add_session_open_range(d, open_range_bars=open_range_bars), "ORB")

    # Forward-fill NaNs introduced by rolling lookbacks (e.g. first N bars of
    # anchored VWAP, ORB before market open) then zero-fill any remaining NaNs
    # in indicator columns.  bfill() was removed — it propagates future values
    # into past rows (lookahead violation).
    # Only hard-drop rows where core OHLCV columns are missing.
    rows_before = len(d)
    d.ffill(inplace=True)
    d.fillna(0, inplace=True)
    core_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in d.columns]
    d.dropna(subset=core_cols, inplace=True)
    rows_after = len(d)
    if rows_before > 0 and rows_after < rows_before * 0.95:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"  institutional dropna removed {rows_before - rows_after:,} rows "
            f"({100*(rows_before-rows_after)/rows_before:.1f}%) — "
            f"{rows_after:,} remain"
        )

    if tf_minutes > 1:
        drop_cols = [c for c in _TICK_MICROSTRUCTURE_SET if c in d.columns]
        if drop_cols:
            d.drop(columns=drop_cols, inplace=True)

    # Step 4 safety check — catch any remaining near-zero-std institutional columns
    # (only checks columns added by this function, not raw OHLCV / return features)
    _inst_cols = [c for c in d.columns if c not in before_cols]
    if _inst_cols:
        _low_std = [c for c in _inst_cols if d[c].std() < 0.01]
        if _low_std:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"[FEATURE] Low-variance institutional columns detected "
                f"(tf={tf_minutes}m): {_low_std}"
            )

    if verbose:
        added = len(d.columns) - before
        print(f"\n  Features before : {before}")
        print(f"  Features added  : {added}")
        print(f"  Features total  : {len(d.columns)}")
        print(f"  Clean rows      : {len(d):,}")

        # Feature group summary
        groups = {
            "VWAP":            [c for c in d.columns if "vwap" in c],
            "Volume profile":  [c for c in d.columns if c.startswith("vp_")],
            "Order flow":      [c for c in d.columns if any(
                                x in c for x in ["delta","cvd","absorption",
                                                  "stacked","failed_auction",
                                                  "imbalance"])],
            "Liquidity":       [c for c in d.columns if any(
                                x in c for x in ["swing","equal_","stop_hunt",
                                                  "wick","pin_bar","lvn",
                                                  "approaching"])],
            "Regime":          [c for c in d.columns if "regime" in c
                                or c in ["adx","adx_norm","atr_ratio",
                                         "atr_slope","hv20_pct"]],
            "ORB":             [c for c in d.columns if "orb_" in c],
        }
        print()
        for group, cols in groups.items():
            print(f"  {group:<18} {len(cols):>3} features: "
                  f"{', '.join(cols[:4])}{'...' if len(cols) > 4 else ''}")

    return d


# ─────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE REPORT (post-training helper)
# ─────────────────────────────────────────────────────────────

def get_institutional_feature_report(model, feature_cols: list,
                                      top_n: int = 30) -> pd.DataFrame:
    """
    After training XGBoost/RF, extract feature importances and
    show which institutional features the ML values most.
    Use this to understand what the model discovered.

    Usage:
        report = get_institutional_feature_report(xgb_model, feature_cols)
        print(report.head(20))
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not expose feature_importances_")
        return pd.DataFrame()

    importances = model.feature_importances_
    report = pd.DataFrame({
        "feature":    feature_cols[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # Tag each feature with its group
    def tag_group(feat):
        if "vwap" in feat:         return "VWAP"
        if feat.startswith("vp_"): return "Volume Profile"
        if any(x in feat for x in ["delta","cvd","absorption","stacked","auction"]):
            return "Order Flow"
        if any(x in feat for x in ["swing","equal_","stop_hunt","wick","pin_bar"]):
            return "Liquidity"
        if "regime" in feat or feat in ["adx","adx_norm","atr_ratio"]:
            return "Regime"
        if "orb_" in feat:         return "ORB"
        return "Technical"

    report["group"] = report["feature"].apply(tag_group)
    report["rank"]  = range(1, len(report) + 1)

    print(f"\nTop {top_n} features by importance:")
    print(report[["rank","feature","group","importance"]].head(top_n).to_string(index=False))

    # Group summary
    group_sum = report.groupby("group")["importance"].sum().sort_values(ascending=False)
    print(f"\nFeature group contribution:")
    for grp, imp in group_sum.items():
        print(f"  {grp:<20} {imp*100:.1f}%")

    return report
