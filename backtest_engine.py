"""
backtest_engine.py — Full backtesting with equity curves and statistics.

Uses featured parquets (161 cols, saved during training) for accurate results.
Falls back to raw parquets if featured version not found.

Key metrics:
  - Equity curve (daily)
  - Monthly P&L breakdown
  - Efficiency ratio: (peak - start) / max_drawdown_money * ER_MULTIPLIER
  - Win rate, profit factor, Sharpe, max drawdown, expectancy
"""

import os
import json
import logging
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(os.getenv("BASE_DIR",   r"F:\trading_ml"))
DATA_DIR   = Path(os.getenv("DATA_DIR",   str(BASE_DIR / "data")))
MODEL_DIR  = Path(os.getenv("MODEL_DIR",  str(BASE_DIR / "models")))

BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2020-01-02")
ER_MULTIPLIER       = float(os.getenv("ER_MULTIPLIER", "1.25"))
SEQ_LEN             = 20   # must match phase2

log = logging.getLogger("backtest_engine")


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_featured_df(symbol: str, tf: int) -> pd.DataFrame:
    """
    Load the full-featured dataframe for backtesting.
    Prefers the featured parquet (161 cols) saved during training.
    Falls back to raw parquet if not found.
    """
    featured_path = DATA_DIR / f"{symbol}_{tf}m_featured.parquet"
    raw_path      = DATA_DIR / f"{symbol}_{tf}m_ticks.parquet"

    if featured_path.exists():
        df = pd.read_parquet(featured_path)
        log.debug(f"  Loaded featured parquet: {featured_path.name} ({len(df):,} rows, {len(df.columns)} cols)")
    elif raw_path.exists():
        df = pd.read_parquet(raw_path)
        log.debug(f"  Loaded raw parquet (featured not found): {raw_path.name}")
    else:
        return pd.DataFrame()

    # Filter to backtest start date
    start = pd.Timestamp(BACKTEST_START_DATE, tz="UTC")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index >= start]

    return df


def _detect_session(row) -> str:
    """Classify a bar's session from its session flag columns."""
    get = lambda k: float(row[k]) if k in row.index else 0.0
    if get("is_us_overlap"):  return "NY/London Overlap"
    if get("is_us_open"):     return "New York"
    if get("is_london"):      return "London"
    if get("is_premarket"):   return "Pre-Market"
    if get("is_asian"):       return "Asian"
    return "Other"


def _get_scaler_cols(scaler) -> list[str]:
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return []


def _apply_scaler(df: pd.DataFrame, scaler) -> tuple[np.ndarray, list[str]]:
    """Apply scaler, filling any missing cols with 0. Returns (array, col_names)."""
    cols = _get_scaler_cols(scaler)
    if not cols:
        return np.zeros((len(df), 1)), []
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    scaled = scaler.transform(out[cols])
    return scaled, cols


# ─────────────────────────────────────────────────────────────────────
# Core backtest
# ─────────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    params: dict,
    scaler,
    xgb_m,
    rf_m,
    risk_mode: str  = "percent",   # "percent" or "fixed"
    risk_pct:  float = 1.0,        # % of current balance per trade
    fixed_amt: float = 100.0,      # fixed $ per trade (if risk_mode="fixed")
    start_balance: float = 10_000.0,
    er_multiplier: float = None,
) -> dict:
    """
    Full backtest of one strategy on one dataframe.

    Returns dict with:
      equity_df   — pd.DataFrame with daily equity, balance, drawdown_pct
      monthly_pnl — list of dicts {year, month, pnl_r, pnl_money, n_trades, win_rate}
      stats       — full stats dict
    """
    if er_multiplier is None:
        er_multiplier = ER_MULTIPLIER

    if df is None or len(df) < SEQ_LEN + 10:
        return {"stats": None, "equity_df": pd.DataFrame(), "monthly_pnl": []}

    confidence = params.get("confidence", 0.6)
    sl_atr     = params.get("sl_atr", 1.5)
    rr         = params.get("rr", 2.0)
    tp_mult    = params.get("tp_mult", 2.0)
    be_r       = params.get("be_r", 0)

    # Scale features
    feat_arr, feat_cols = _apply_scaler(df, scaler)
    if len(feat_cols) == 0:
        return {"stats": None, "equity_df": pd.DataFrame(), "monthly_pnl": []}

    # Ensemble probabilities
    try:
        p_xgb = xgb_m.predict_proba(feat_arr)[:, 1]
        p_rf  = rf_m.predict_proba(feat_arr)[:, 1]
    except Exception as e:
        log.warning(f"  Backtest predict_proba failed: {e}")
        return {"stats": None, "equity_df": pd.DataFrame(), "monthly_pnl": []}

    prob = (p_xgb + p_rf) / 2.0

    # Extract OHLC arrays
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    atr_arr = df["atr14"].values if "atr14" in df.columns else (
        pd.Series(highs - lows).rolling(14).mean().values
    )
    dates  = df.index

    # ── Simulate trades ──────────────────────────────────────────────
    balance  = start_balance
    peak     = start_balance
    trades   = []          # full per-trade records
    equity_records = []    # (date, equity)
    in_trade = False
    open_trade = {}

    for i in range(SEQ_LEN, len(df) - 1):
        bar_date = dates[i]

        # Check open trade
        if in_trade:
            t = open_trade
            bh = highs[i]; bl = lows[i]

            # Break-even check
            if be_r > 0 and not t.get("be_done"):
                be_trigger = t["entry"] + t["dir"] * t["sl_dist"] * be_r
                if (t["dir"] == 1 and bh >= be_trigger) or \
                   (t["dir"] == -1 and bl <= be_trigger):
                    t["sl"]      = t["entry"] + t["dir"] * 1.0
                    t["be_done"] = True

            pnl_r = 0.0; hit = False
            if t["dir"] == 1:
                if bl <= t["sl"]:
                    pnl_r = (t["sl"] - t["entry"]) / (t["sl_dist"] + 1e-10); hit = True
                elif bh >= t["tp"]:
                    pnl_r = t["rr"]; hit = True
            else:
                if bh >= t["sl"]:
                    pnl_r = (t["entry"] - t["sl"]) / (t["sl_dist"] + 1e-10); hit = True
                elif bl <= t["tp"]:
                    pnl_r = t["rr"]; hit = True

            if hit:
                pnl_money = pnl_r * t["risk"]
                balance  += pnl_money
                entry_ts  = t["entry_time"]
                trades.append({
                    "entry_time":  str(entry_ts)[:19],
                    "close_time":  str(bar_date)[:19],
                    "session":     t["session"],
                    "direction":   t["dir"],
                    "pnl_r":       pnl_r,
                    "pnl_money":   pnl_money,
                    "win":         int(pnl_r > 0),
                    "hour_utc":    getattr(entry_ts, "hour", 0),
                    "day_of_week": getattr(entry_ts, "dayofweek",
                                          getattr(entry_ts, "weekday", lambda: 0)()),
                })
                if balance > peak:
                    peak = balance
                in_trade = False
                equity_records.append((bar_date, balance))
            continue

        # Check for signal
        p = prob[i]
        direction = None
        if p >= confidence:
            direction = 1
        elif p <= (1 - confidence):
            direction = -1

        if direction is None:
            continue

        sl_dist = atr_arr[i] * sl_atr
        if sl_dist <= 0 or np.isnan(sl_dist):
            continue

        entry = closes[i]
        sl    = entry - direction * sl_dist
        tp    = entry + direction * sl_dist * tp_mult

        if risk_mode == "fixed":
            risk = fixed_amt
        else:
            risk = max(balance, 1.0) * risk_pct / 100.0

        open_trade = {
            "dir":        direction,
            "entry":      entry,
            "sl":         sl,
            "tp":         tp,
            "sl_dist":    sl_dist,
            "rr":         rr,
            "risk":       risk,
            "be_done":    False,
            "entry_time": bar_date,
            "entry_idx":  i,
            "session":    _detect_session(df.iloc[i]),
        }
        in_trade = True
        equity_records.append((bar_date, balance))

    # ── Build daily equity curve ─────────────────────────────────────
    if not equity_records:
        return {"stats": None, "equity_df": pd.DataFrame(), "monthly_pnl": []}

    eq_series = pd.Series(
        [v for _, v in equity_records],
        index=pd.DatetimeIndex([d for d, _ in equity_records])
    )
    # Resample to daily (last value of each day)
    if eq_series.index.tz is None:
        eq_series.index = eq_series.index.tz_localize("UTC")
    daily_eq = eq_series.resample("1D").last().ffill()
    daily_eq.name = "equity"

    # Running peak and drawdown
    running_peak = daily_eq.cummax()
    drawdown_pct = (running_peak - daily_eq) / (running_peak + 1e-10) * 100

    equity_df = pd.DataFrame({
        "date":         daily_eq.index.strftime("%Y-%m-%d"),
        "equity":       daily_eq.values,
        "balance":      daily_eq.values,
        "drawdown_pct": drawdown_pct.values,
    })

    # ── Summary stats ────────────────────────────────────────────────
    if not trades:
        return {"stats": None, "equity_df": equity_df, "monthly_pnl": []}

    r     = np.array([t["pnl_r"]     for t in trades])
    money = np.array([t["pnl_money"] for t in trades])
    wins  = r[r > 0]
    loss  = r[r < 0]

    # ── Standard Sharpe (annualised, using trade R-values) ───────────
    sharpe    = r.mean() / (r.std() + 1e-10) * np.sqrt(252)

    # ── Sortino: only penalise downside deviation ────────────────────
    downside  = r[r < 0]
    down_std  = downside.std() if len(downside) > 1 else 1e-10
    sortino   = r.mean() / (down_std + 1e-10) * np.sqrt(252)

    # ── Equity curve stats (using live risk % / fixed $ as simulated) ─
    max_dd_pct   = float(drawdown_pct.max())
    peak_series  = daily_eq.cummax()
    dd_money     = (peak_series - daily_eq).max()
    total_profit = float(daily_eq.iloc[-1] - start_balance)
    peak_profit  = float(peak_series.iloc[-1] - start_balance)

    # ── Normalised ER: fixed $100 per trade (pure strategy signal) ───
    # Removes compounding luck — each trade risks exactly $100, linear equity.
    # This is what ER is compared on for strategy selection.
    r100         = r * 100.0                         # each R = $100 at risk
    eq100        = start_balance + np.cumsum(r100)
    peak100      = np.maximum.accumulate(eq100)
    dd100        = (peak100 - eq100).max()
    profit100    = eq100[-1] - start_balance
    er           = (profit100 / (dd100 + 1e-10)) * er_multiplier if dd100 > 0 else 0.0

    # ── Calmar: annualised return / max DD% ──────────────────────────
    n_days     = max((daily_eq.index[-1] - daily_eq.index[0]).days, 1)
    ann_return = (daily_eq.iloc[-1] / (start_balance + 1e-10)) ** (365.0 / n_days) - 1
    # Floor DD at 0.1% so Calmar doesn't blow up on nearly-zero DD periods
    calmar     = (ann_return * 100) / max(max_dd_pct, 0.1)
    calmar     = float(np.clip(calmar, -999.0, 999.0))

    # ── Consecutive losses ───────────────────────────────────────────
    cur_consec = max_consec = 0
    for v in r:
        if v < 0:
            cur_consec += 1; max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    stats = {
        "n_trades":          len(trades),
        "win_rate":          float(len(wins) / len(trades) * 100),
        "avg_win_r":         float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss_r":        float(loss.mean()) if len(loss) > 0 else 0.0,
        "profit_factor":     float(wins.sum() / (-loss.sum() + 1e-10)) if len(loss) > 0 else 99.0,
        "sharpe":            float(sharpe),
        "sortino":           float(sortino),
        "calmar":            float(calmar),
        "max_dd_pct":        max_dd_pct,
        "max_dd_money":      float(dd_money),
        "total_profit":      total_profit,
        "peak_profit":       peak_profit,
        "efficiency_ratio":  float(er),       # normalised: fixed $100/trade
        "expectancy":        float(r.mean()),
        "max_consec_loss":   max_consec,
        "final_balance":     float(daily_eq.iloc[-1]),
    }

    # ── Monthly P&L ─────────────────────────────────────────────────
    monthly_pnl = []
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        if not pd.api.types.is_datetime64_any_dtype(trade_df["entry_time"]):
            trade_df["entry_time"] = pd.to_datetime(trade_df["entry_time"], utc=True)
        trade_df["year"]  = trade_df["entry_time"].dt.year
        trade_df["month"] = trade_df["entry_time"].dt.month
        for (yr, mo), grp in trade_df.groupby(["year", "month"]):
            monthly_pnl.append({
                "year":      int(yr),
                "month":     int(mo),
                "pnl_r":     float(grp["pnl_r"].sum()),
                "pnl_money": float(grp["pnl_money"].sum()),
                "n_trades":  len(grp),
                "win_rate":  float((grp["win"].sum() / len(grp)) * 100),
            })

    return {
        "stats":       stats,
        "equity_df":   equity_df,
        "monthly_pnl": monthly_pnl,
        "trades":      trades,   # full per-trade list for session/timing analysis
    }


# ─────────────────────────────────────────────────────────────────────
# Deflated / Haircut Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────

def haircut_sharpe(trades: list[dict], n_trials: int = 150) -> float:
    """
    Simplified Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    When you run n_trials Optuna trials and pick the best, the winner
    benefits from selection bias — it looks better than it truly is.
    This adjusts downward based on:
      - Number of trials tested (more trials = more fishing = bigger haircut)
      - Skewness of returns (negative skew = fat left tail = bigger haircut)
      - Excess kurtosis (fat tails = more extreme but rarer outcomes)

    Formula: SR_haircut = SR * (1 - skew*SR/6 - (kurt-3)*SR²/24)
                          then apply multiple-testing penalty: / sqrt(log(n_trials))

    Returns haircut Sharpe. Values below 0 = likely lucky, not skilled.
    """
    if len(trades) < 10:
        return -999.0

    r       = np.array([t["pnl_r"] for t in trades])
    sr      = r.mean() / (r.std() + 1e-10) * np.sqrt(252)

    # Moments adjustment (corrects SR for non-normality of returns)
    skew    = float(pd.Series(r).skew())
    kurt    = float(pd.Series(r).kurt())   # excess kurtosis
    sr_adj  = sr * (1.0 - (skew * sr / 6.0) - ((kurt) * sr**2 / 24.0))

    # Multiple-testing penalty: sqrt(log(n_trials)) — rough but practical
    # At 150 trials this is ~sqrt(5.01) ≈ 2.24, so a SR of 2.0 becomes ~0.89
    penalty = max(1.0, np.sqrt(np.log(max(n_trials, 2))))
    return float(sr_adj / penalty)


# ─────────────────────────────────────────────────────────────────────
# Monte Carlo simulation
# ─────────────────────────────────────────────────────────────────────

def run_monte_carlo(trades: list[dict], n_sims: int = 1000, seed: int = 42) -> dict:
    """
    Bootstrap resample the trade sequence to estimate distribution of outcomes.

    Industry standard robustness check: if real performance relied on a lucky
    ordering of wins/losses, MC will expose it (wide p5/p95 spread, p5 DD >> actual DD).

    Returns dict with:
      mc_sharpe_p5, mc_sharpe_p50, mc_sharpe_p95
      mc_dd_p50, mc_dd_p95          — worst-case drawdown percentiles
      mc_profit_p5, mc_profit_p50   — final profit percentiles
      mc_pass                       — True if p5 Sharpe > 0 (profitable in 95% of orderings)
    """
    if len(trades) < 10:
        return {"mc_pass": False, "mc_sharpe_p5": -999.0, "mc_sharpe_p50": 0.0,
                "mc_sharpe_p95": 0.0, "mc_dd_p50": 100.0, "mc_dd_p95": 100.0,
                "mc_profit_p5": 0.0, "mc_profit_p50": 0.0}

    rng    = np.random.default_rng(seed)
    pnl_r  = np.array([t["pnl_r"] for t in trades])

    sharpes, max_dds, final_profits = [], [], []

    for _ in range(n_sims):
        sim = rng.choice(pnl_r, size=len(pnl_r), replace=True)
        # Sharpe
        s = sim.mean() / (sim.std() + 1e-10) * np.sqrt(252)
        sharpes.append(s)
        # Equity curve from cumulative sum of returns
        equity = np.cumprod(1 + sim * 0.01)   # treat each R as 1% account risk
        peak   = np.maximum.accumulate(equity)
        dd     = ((peak - equity) / (peak + 1e-10) * 100).max()
        max_dds.append(dd)
        final_profits.append(float(equity[-1] - 1) * 100)   # % gain

    sharpes      = np.array(sharpes)
    max_dds      = np.array(max_dds)
    final_profits= np.array(final_profits)

    result = {
        "mc_sharpe_p5":   float(np.percentile(sharpes, 5)),
        "mc_sharpe_p50":  float(np.percentile(sharpes, 50)),
        "mc_sharpe_p95":  float(np.percentile(sharpes, 95)),
        "mc_dd_p50":      float(np.percentile(max_dds, 50)),
        "mc_dd_p95":      float(np.percentile(max_dds, 95)),
        "mc_profit_p5":   float(np.percentile(final_profits, 5)),
        "mc_profit_p50":  float(np.percentile(final_profits, 50)),
        "mc_pass":        bool(np.percentile(sharpes, 5) > 0),
    }
    return result


# ─────────────────────────────────────────────────────────────────────
# Parameter sensitivity (robustness) check
# ─────────────────────────────────────────────────────────────────────

def run_sensitivity(
    df: pd.DataFrame, params: dict, scaler, xgb_m, rf_m,
    risk_mode: str = "percent", risk_pct: float = 1.0,
    fixed_amt: float = 100.0, start_balance: float = 10_000.0,
) -> dict:
    """
    Nudge confidence, sl_atr, and rr each by ±step and re-backtest.
    A robust strategy degrades gracefully — not a cliff edge.

    Returns:
      sensitivity_score  — 0-100, higher = more robust
      robust             — True if score >= 50
      variants           — list of {param, delta, sharpe, er} for each nudge
    """
    NUDGES = [
        ("confidence", [-0.05, +0.05]),
        ("sl_atr",     [-0.2,  +0.2 ]),
        ("rr",         [-0.3,  +0.3 ]),
    ]

    base_bt    = run_backtest(df, params, scaler, xgb_m, rf_m,
                              risk_mode=risk_mode, risk_pct=risk_pct,
                              fixed_amt=fixed_amt, start_balance=start_balance)
    base_stats = base_bt.get("stats")
    if not base_stats:
        return {"sensitivity_score": 0.0, "robust": False, "variants": []}

    base_sharpe = base_stats["sharpe"]
    base_er     = base_stats["efficiency_ratio"]
    variants    = []
    scores      = []

    for param, deltas in NUDGES:
        for delta in deltas:
            nudged = dict(params)
            nudged[param] = max(0.01, params.get(param, 0) + delta)
            bt = run_backtest(df, nudged, scaler, xgb_m, rf_m,
                              risk_mode=risk_mode, risk_pct=risk_pct,
                              fixed_amt=fixed_amt, start_balance=start_balance)
            st = bt.get("stats")
            if not st:
                scores.append(0.0)
                variants.append({"param": param, "delta": delta,
                                 "sharpe": -999.0, "er": 0.0})
                continue
            # Score: 1.0 = nudge has no effect, 0.0 = total collapse or huge swing
            # We penalise instability in EITHER direction — a cliff edge where nudging
            # makes things dramatically better also signals overfitting to exact params
            if abs(base_sharpe) < 0.01:
                score = 0.0   # base strategy is essentially flat — not robust
            else:
                change = abs(st["sharpe"] - base_sharpe) / (abs(base_sharpe) + 1e-10)
                score  = float(np.clip(1.0 - change, 0.0, 1.0))
            scores.append(score)
            variants.append({"param": param, "delta": delta,
                             "sharpe": st["sharpe"], "er": st["efficiency_ratio"]})

    sensitivity_score = float(np.mean(scores) * 100) if scores else 0.0
    return {
        "sensitivity_score": sensitivity_score,
        "robust":            sensitivity_score >= 50.0,
        "variants":          variants,
        "base_sharpe":       base_sharpe,
        "base_er":           base_er,
    }


# ─────────────────────────────────────────────────────────────────────
# Batch backtesting — backtest all top-5 strategies for a symbol/TF
# ─────────────────────────────────────────────────────────────────────

def backtest_all_strategies(
    symbol: str,
    tf: int,
    top_params: list[dict],
    risk_mode: str  = "percent",
    risk_pct: float = 1.0,
    fixed_amt: float = 100.0,
    start_balance: float = 10_000.0,
) -> list[dict]:
    """
    Run backtest for each param set in top_params.
    Returns list of result dicts (same order as top_params).
    Each dict has: params, stats, equity_df, monthly_pnl
    """
    df = load_featured_df(symbol, tf)
    if df.empty:
        log.warning(f"  No data found for {symbol} {tf}m")
        return []

    key      = f"{symbol}_{tf}m"
    xgb_path = MODEL_DIR / f"xgb_{key}.pkl"
    rf_path  = MODEL_DIR / f"rf_{key}.pkl"
    sc_path  = MODEL_DIR / f"scaler_{key}.pkl"

    if not (xgb_path.exists() and rf_path.exists() and sc_path.exists()):
        log.warning(f"  Models missing for {symbol} {tf}m")
        return []

    xgb_m  = joblib.load(xgb_path)
    rf_m   = joblib.load(rf_path)
    scaler = joblib.load(sc_path)

    results = []
    for i, params in enumerate(top_params, 1):
        log.info(f"  Backtesting {symbol} {tf}m rank{i} ...")
        result = run_backtest(
            df, params, scaler, xgb_m, rf_m,
            risk_mode=risk_mode, risk_pct=risk_pct, fixed_amt=fixed_amt,
            start_balance=start_balance,
        )
        result["params"] = params
        result["rank"]   = i
        results.append(result)

    return results
