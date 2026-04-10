"""
report.py — Interactive HTML trading report.

Run:   python report.py
Opens: trading_report.html  (double-click to view in browser)

Contents:
  1. Summary cards — best strategy, key metrics including DD detail
  2. ML timing parameters — what TF/HTF/session the model chose
  3. Equity curve + drawdown chart (Plotly, interactive)
  4. Drawdown analysis — top 5 worst days, DD period, recovery
  5. Session timing — win rate/trade count/P&L by trading session
  6. Hour-of-day heatmap — when the system trades and profits
  7. Per-TF top-5 strategy comparison table
  8. Monthly P&L heatmap
  9. Trade frequency statistics
  10. Feature importance (entry TF of best strategy)
  11. Optimisation criteria explanation
  12. Live trade log (if any)
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = Path(os.getenv("BASE_DIR",   r"F:\trading_ml"))
MODEL_DIR  = Path(os.getenv("MODEL_DIR",  str(BASE_DIR / "models")))
PARAMS_DIR = Path(os.getenv("PARAMS_DIR", str(BASE_DIR / "params")))
DATA_DIR   = Path(os.getenv("DATA_DIR",   str(BASE_DIR / "data")))

RISK_MODE       = os.getenv("RISK_MODE", "percent")
RISK_PCT        = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
FIXED_RISK_AMT  = float(os.getenv("FIXED_RISK_AMOUNT",  "100.0"))
ER_MULTIPLIER   = float(os.getenv("ER_MULTIPLIER",       "1.25"))
BACKTEST_START  = os.getenv("BACKTEST_START_DATE",       "2020-01-02")
START_BALANCE   = 10_000.0

OUTPUT_FILE = BASE_DIR / "trading_report.html"

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("WARNING: plotly not installed. Run: pip install plotly")

try:
    from db import (
        get_all_strategies, get_equity_curve, get_monthly_pnl,
        get_open_trades, get_backtest_trades, init_db,
    )
    init_db()
    HAS_DB = True
except Exception as e:
    HAS_DB = False
    print(f"WARNING: DB not available ({e}). Run training first.")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

DOW_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

SESSION_ORDER = [
    "NY/London Overlap",
    "New York",
    "London",
    "Pre-Market",
    "Asian",
    "Other",
]
SESSION_COLORS = {
    "NY/London Overlap": "#e74c3c",
    "New York":          "#e67e22",
    "London":            "#3498db",
    "Pre-Market":        "#9b59b6",
    "Asian":             "#1abc9c",
    "Other":             "#7f8c8d",
}


def _color(v, good_positive=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#888"
    if good_positive:
        return "#2ecc71" if v > 0 else ("#e74c3c" if v < 0 else "#888")
    else:
        return "#e74c3c" if v > 0 else ("#2ecc71" if v < 0 else "#888")


def _fmt(v, fmt=".2f", suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:{fmt}}{suffix}"


def _load_feature_importance(symbol: str, tf: int) -> list:
    key      = f"{symbol}_{tf}m"
    xgb_path = MODEL_DIR / f"xgb_{key}.pkl"
    feat_path = DATA_DIR / f"{symbol}_{tf}m_featured.parquet"
    raw_path  = DATA_DIR / f"{symbol}_{tf}m_ticks.parquet"

    if not xgb_path.exists():
        return []
    xgb_m = joblib.load(xgb_path)
    skip  = {"Open","High","Low","Close","Volume","target","bar_time"}

    if hasattr(xgb_m, "feature_names_in_"):
        feat_cols = list(xgb_m.feature_names_in_)
    elif feat_path.exists():
        df = pd.read_parquet(feat_path)
        feat_cols = [c for c in df.columns if c not in skip
                     and pd.api.types.is_numeric_dtype(df[c])]
    elif raw_path.exists():
        df = pd.read_parquet(raw_path)
        feat_cols = [c for c in df.columns if c not in skip
                     and pd.api.types.is_numeric_dtype(df[c])]
    else:
        return []

    if not hasattr(xgb_m, "feature_importances_"):
        return []

    imp   = xgb_m.feature_importances_
    n     = min(len(imp), len(feat_cols))
    pairs = sorted(zip(feat_cols[:n], imp[:n]),
                   key=lambda x: x[1], reverse=True)[:15]
    return pairs


def _compute_dd_periods(curve: list) -> dict:
    """
    From an equity curve list, compute:
      - max_dd_pct, max_dd_money
      - dd start date, trough date, recovery date (if recovered)
      - top 5 worst days by equity drop
    """
    if not curve:
        return {}

    dates  = pd.to_datetime([r["date"] for r in curve])
    equity = pd.Series([r["equity"] for r in curve], index=dates)

    peak        = equity.cummax()
    dd_money    = peak - equity
    dd_pct      = dd_money / (peak + 1e-10) * 100

    max_dd_pct   = float(dd_pct.max())
    max_dd_money = float(dd_money.max())

    # Find the max DD trough
    trough_idx  = dd_money.idxmax()
    trough_val  = equity[trough_idx]
    # Peak before trough
    peak_before  = peak[trough_idx]
    dd_start_idx = equity[:trough_idx][equity[:trough_idx] >= peak_before - 0.01].index
    dd_start     = dd_start_idx[-1] if len(dd_start_idx) > 0 else dates[0]
    # Recovery: first date after trough where equity >= peak_before
    after_trough = equity[trough_idx:]
    recovered    = after_trough[after_trough >= peak_before]
    recovery_dt  = recovered.index[0] if len(recovered) > 0 else None

    dd_duration_days = (trough_idx - dd_start).days
    if recovery_dt:
        recovery_days = (recovery_dt - trough_idx).days
    else:
        recovery_days = None

    # Top 5 worst single days by equity drop
    daily_diff  = equity.diff().dropna()
    worst_days  = daily_diff.nsmallest(5)
    worst_list  = [
        {"date": str(d.date()), "drop_money": float(v),
         "drop_pct": float(v / (equity.get(d, equity.iloc[0]) - v + 1e-10) * 100)}
        for d, v in worst_days.items()
    ]

    return {
        "max_dd_pct":       max_dd_pct,
        "max_dd_money":     max_dd_money,
        "dd_start":         str(dd_start.date()),
        "dd_trough":        str(trough_idx.date()),
        "trough_equity":    float(trough_val),
        "dd_duration_days": dd_duration_days,
        "recovery_dt":      str(recovery_dt.date()) if recovery_dt else None,
        "recovery_days":    recovery_days,
        "worst_days":       worst_list,
    }


def _session_stats(trades: list) -> dict:
    """Aggregate trades by session and by hour/DOW."""
    by_session = {}
    by_hour    = {h: {"n": 0, "wins": 0, "pnl": 0.0} for h in range(24)}
    by_dow     = {d: {"n": 0, "wins": 0, "pnl": 0.0} for d in range(7)}

    for t in trades:
        sess = t.get("session") or "Other"
        win  = int(t.get("win", 0))
        pnl  = float(t.get("pnl_money", 0) or 0)
        hr   = int(t.get("hour_utc", 0) or 0)
        dow  = int(t.get("day_of_week", 0) or 0)

        if sess not in by_session:
            by_session[sess] = {"n": 0, "wins": 0, "pnl": 0.0}
        by_session[sess]["n"]    += 1
        by_session[sess]["wins"] += win
        by_session[sess]["pnl"]  += pnl

        by_hour[hr]["n"]    += 1
        by_hour[hr]["wins"] += win
        by_hour[hr]["pnl"]  += pnl

        if 0 <= dow <= 6:
            by_dow[dow]["n"]    += 1
            by_dow[dow]["wins"] += win
            by_dow[dow]["pnl"]  += pnl

    return {"by_session": by_session, "by_hour": by_hour, "by_dow": by_dow}


def _trade_frequency(trades: list, strategy: dict) -> dict:
    """Compute expected trade frequency statistics."""
    if not trades:
        return {}

    dates = pd.to_datetime([t["entry_time"] for t in trades if t.get("entry_time")])
    if len(dates) == 0:
        return {}

    total_days = max((dates.max() - dates.min()).days, 1)
    total_weeks  = max(total_days / 7, 1)
    total_months = max(total_days / 30.44, 1)
    n = len(trades)

    daily_counts = dates.to_series().dt.date.value_counts()
    weekly_dates = dates.to_series().dt.to_period("W").value_counts()

    return {
        "total": n,
        "per_day":    n / total_days,
        "per_week":   n / total_weeks,
        "per_month":  n / total_months,
        "max_per_day":  int(daily_counts.max()) if len(daily_counts) > 0 else 0,
        "min_per_day":  int(daily_counts.min()) if len(daily_counts) > 0 else 0,
        "days_with_trade": len(daily_counts),
        "total_days":  total_days,
        "date_start":  str(dates.min().date()),
        "date_end":    str(dates.max().date()),
    }


# ─────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────

def build_equity_chart(strategies_with_curves: list) -> str:
    if not HAS_PLOTLY or not strategies_with_curves:
        return "<p>No equity data available. Run training first.</p>"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=("Equity Curve (Balance $)", "Drawdown from Peak (%)"),
        vertical_spacing=0.06,
    )

    colors = ["#3498db","#e67e22","#2ecc71","#e74c3c",
              "#9b59b6","#1abc9c","#f39c12","#d35400",
              "#27ae60","#c0392b"]

    for i, item in enumerate(strategies_with_curves):
        sid   = item["strategy_id"]
        curve = item["curve"]
        color = colors[i % len(colors)]

        dates  = pd.to_datetime([r["date"] for r in curve])
        equity = [r["equity"]       for r in curve]
        dd     = [r["drawdown_pct"] for r in curve]

        fig.add_trace(go.Scatter(
            x=dates, y=equity, mode="lines",
            name=sid,
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{sid}</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dates, y=[-d for d in dd], mode="lines",
            name=f"{sid} DD",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False,
            hovertemplate=f"<b>{sid}</b><br>DD: %{{y:.1f}}%<extra></extra>",
        ), row=2, col=1)

    fig.add_hline(y=START_BALANCE, row=1, col=1,
                  line_dash="dash", line_color="#555", line_width=1,
                  annotation_text=f"Start ${START_BALANCE:,.0f}",
                  annotation_position="right")
    fig.add_hline(y=0, row=2, col=1,
                  line_dash="dash", line_color="#555", line_width=1)

    fig.update_layout(
        height=580, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#444"),
        margin=dict(l=70, r=20, t=40, b=20),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2d2d2d", tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="#2d2d2d", tickfont=dict(size=10))
    fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def build_drawdown_analysis(strategy_id: str, curve: list) -> str:
    """Detailed drawdown section: top 5 worst days, DD period, relative + absolute."""
    if not curve:
        return "<p>No equity data for drawdown analysis.</p>"

    dd = _compute_dd_periods(curve)
    if not dd:
        return "<p>Drawdown calculation failed.</p>"

    # Cards row
    recovered_txt = (
        f"Recovered in {dd['recovery_days']} days ({dd['recovery_dt']})"
        if dd["recovery_dt"] else
        "<span style='color:#e74c3c'>Not yet recovered</span>"
    )

    cards_html = f"""
    <div class="cards-row" style="margin-bottom:16px">
      <div class="card">
        <div class="card-label">Max Drawdown (Relative)</div>
        <div class="card-value" style="color:#e74c3c">{dd['max_dd_pct']:.1f}%</div>
        <div class="card-sub">Peak-to-trough as % of peak balance</div>
      </div>
      <div class="card">
        <div class="card-label">Max Drawdown (Absolute $)</div>
        <div class="card-value" style="color:#e74c3c">${dd['max_dd_money']:,.0f}</div>
        <div class="card-sub">Largest peak-to-trough in money</div>
      </div>
      <div class="card">
        <div class="card-label">DD Period</div>
        <div class="card-value" style="color:#e67e22">{dd['dd_duration_days']} days</div>
        <div class="card-sub">{dd['dd_start']} → trough {dd['dd_trough']}</div>
      </div>
      <div class="card">
        <div class="card-label">Recovery</div>
        <div class="card-value" style="color:{'#2ecc71' if dd['recovery_dt'] else '#e74c3c'}">
          {'Yes' if dd['recovery_dt'] else 'No'}</div>
        <div class="card-sub">{recovered_txt}</div>
      </div>
    </div>"""

    # Top 5 worst days table
    rows_html = ""
    for rank, day in enumerate(dd["worst_days"], 1):
        rows_html += f"""
        <tr>
          <td style="color:#888">#{rank}</td>
          <td>{day['date']}</td>
          <td style="color:#e74c3c">${day['drop_money']:,.0f}</td>
          <td style="color:#e74c3c">{abs(day['drop_pct']):.2f}%</td>
        </tr>"""

    worst_table = f"""
    <h4 style="color:#e74c3c;margin:16px 0 8px">Top 5 Worst Single Days — {strategy_id}</h4>
    <table class="strategy-table" style="max-width:500px">
      <thead>
        <tr><th>#</th><th>Date</th><th>Equity Drop ($)</th><th>Drop (%)</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="note">These are the 5 days with the largest single-day balance loss
    in the backtest. They help identify whether drawdowns are concentrated
    (crash-like) or spread across many small losses.</p>"""

    return cards_html + worst_table


def build_session_chart(strategy_id: str, trades: list) -> str:
    """Bar + line chart: trade count and win rate by trading session."""
    if not HAS_PLOTLY or not trades:
        return "<p>No trade data for session analysis. Run training first.</p>"

    stats = _session_stats(trades)
    by_sess = stats["by_session"]

    sessions  = [s for s in SESSION_ORDER if s in by_sess]
    counts    = [by_sess[s]["n"] for s in sessions]
    win_rates = [
        by_sess[s]["wins"] / by_sess[s]["n"] * 100
        if by_sess[s]["n"] > 0 else 0
        for s in sessions
    ]
    avg_pnl  = [
        by_sess[s]["pnl"] / by_sess[s]["n"]
        if by_sess[s]["n"] > 0 else 0
        for s in sessions
    ]
    colors   = [SESSION_COLORS.get(s, "#888") for s in sessions]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=sessions, y=counts,
        name="Trades",
        marker_color=colors,
        marker_opacity=0.8,
        hovertemplate="<b>%{x}</b><br>Trades: %{y}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=sessions, y=win_rates,
        name="Win Rate %",
        mode="lines+markers",
        line=dict(color="#f1c40f", width=2.5),
        marker=dict(size=9, color="#f1c40f", symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=50, secondary_y=True,
                  line_dash="dash", line_color="#555", line_width=1,
                  annotation_text="50%", annotation_position="right")

    fig.update_layout(
        title=dict(text=f"Session Analysis — {strategy_id}",
                   font=dict(color="#e0e0e0")),
        height=380,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", orientation="h", y=1.12),
        margin=dict(l=60, r=60, t=60, b=60),
        barmode="group",
    )
    fig.update_xaxes(gridcolor="#2d2d2d")
    fig.update_yaxes(title_text="Number of Trades", gridcolor="#2d2d2d",
                     secondary_y=False)
    fig.update_yaxes(title_text="Win Rate (%)", gridcolor="#2d2d2d",
                     range=[0, 100], secondary_y=True)

    chart_div = pyo.plot(fig, output_type="div", include_plotlyjs=False)

    # Session detail table
    total_n = sum(by_sess[s]["n"] for s in sessions) or 1
    rows_html = ""
    for s in sessions:
        if s not in by_sess:
            continue
        d   = by_sess[s]
        n   = d["n"]
        wr  = d["wins"] / n * 100 if n > 0 else 0
        pnl = d["pnl"]
        ap  = pnl / n if n > 0 else 0
        pct_of_trades = n / total_n * 100
        rows_html += f"""
        <tr>
          <td><span style="color:{SESSION_COLORS.get(s,'#888')};font-size:16px">&#9632;</span>
              <b>{s}</b></td>
          <td>{n:,} ({pct_of_trades:.0f}%)</td>
          <td style="color:{'#2ecc71' if wr>=50 else '#e74c3c'}">{wr:.1f}%</td>
          <td style="color:{_color(ap)}">{_fmt(ap, '+,.1f', '$')}</td>
          <td style="color:{_color(pnl)}">{_fmt(pnl, '+,.0f', '$')}</td>
        </tr>"""

    table_html = f"""
    <table class="strategy-table" style="margin-top:16px;max-width:700px">
      <thead>
        <tr><th>Session</th><th>Trades</th><th>Win Rate</th>
            <th>Avg P&L/Trade</th><th>Total P&L</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="note">
      Sessions defined by market hours (all times UTC):
      <b>Asian</b> 00:00–06:00 |
      <b>Pre-Market</b> 09:00–13:30 |
      <b>London</b> 08:00–17:00 |
      <b>NY/London Overlap</b> 13:30–17:00 |
      <b>New York</b> 13:30–21:00 (when London closed)
    </p>"""

    return chart_div + table_html


def build_hour_heatmap(strategy_id: str, trades: list) -> str:
    """Win rate and trade count by UTC hour of day."""
    if not HAS_PLOTLY or not trades:
        return ""

    stats   = _session_stats(trades)
    by_hour = stats["by_hour"]

    hours     = list(range(24))
    counts    = [by_hour[h]["n"] for h in hours]
    win_rates = [
        by_hour[h]["wins"] / by_hour[h]["n"] * 100
        if by_hour[h]["n"] > 0 else None
        for h in hours
    ]
    avg_pnl = [
        by_hour[h]["pnl"] / by_hour[h]["n"]
        if by_hour[h]["n"] > 0 else 0
        for h in hours
    ]

    # Colour bars by session membership (approx)
    def _hour_session_color(h):
        # UTC hours approximately:
        if 13 <= h < 17:   return SESSION_COLORS["NY/London Overlap"]
        if 8  <= h < 13:   return SESSION_COLORS["London"]
        if 13 <= h < 21:   return SESSION_COLORS["New York"]
        if 9  <= h < 13:   return SESSION_COLORS["Pre-Market"]
        if 0  <= h < 8:    return SESSION_COLORS["Asian"]
        return SESSION_COLORS["Other"]

    bar_colors = [_hour_session_color(h) for h in hours]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=[f"{h:02d}:00" for h in hours],
        y=counts, name="Trades",
        marker_color=bar_colors, marker_opacity=0.75,
        hovertemplate="<b>%{x} UTC</b><br>Trades: %{y}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=[f"{h:02d}:00" for h in hours],
        y=win_rates, name="Win Rate %",
        mode="lines+markers",
        line=dict(color="#f1c40f", width=2),
        marker=dict(size=6, color="#f1c40f"),
        connectgaps=True,
        hovertemplate="<b>%{x} UTC</b><br>Win Rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.add_hline(y=50, secondary_y=True,
                  line_dash="dash", line_color="#555", line_width=1)

    fig.update_layout(
        title=dict(text=f"Trade Distribution by Hour (UTC) — {strategy_id}",
                   font=dict(color="#e0e0e0")),
        height=340,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", orientation="h", y=1.12),
        margin=dict(l=60, r=60, t=60, b=50),
    )
    fig.update_xaxes(gridcolor="#2d2d2d", tickangle=-45, tickfont=dict(size=9))
    fig.update_yaxes(title_text="Trades", gridcolor="#2d2d2d", secondary_y=False)
    fig.update_yaxes(title_text="Win %", range=[0, 100], gridcolor="#2d2d2d",
                     secondary_y=True)

    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def build_frequency_section(strategy: dict, trades: list) -> str:
    """Trade frequency stats — answers 'is it normal to have no trades for X hours?'"""
    freq = _trade_frequency(trades, strategy)
    if not freq:
        return ""

    per_day = freq["per_day"]
    per_wk  = freq["per_week"]
    per_mo  = freq["per_month"]
    n       = freq["total"]

    # Expected hours between trades
    avg_hrs_between = 24.0 / per_day if per_day > 0 else 999

    # Guidance text
    if per_day < 0.5:
        cadence = f"~{1/per_day:.0f} trading days per trade — very selective strategy"
    elif per_day < 2:
        cadence = f"~{avg_hrs_between:.0f}h between trades on average — low frequency"
    elif per_day < 5:
        cadence = f"~{avg_hrs_between:.0f}h between trades on average — medium frequency"
    else:
        cadence = f"~{avg_hrs_between:.1f}h between trades on average — higher frequency"

    return f"""
    <div class="info-box" style="margin-top:20px">
      <h3>Trade Frequency — {strategy.get('strategy_id','?')}</h3>
      <div class="cards-row" style="margin:12px 0">
        <div class="card">
          <div class="card-label">Total Backtest Trades</div>
          <div class="card-value" style="color:#3498db">{n:,}</div>
          <div class="card-sub">{freq['date_start']} → {freq['date_end']}</div>
        </div>
        <div class="card">
          <div class="card-label">Avg per Day</div>
          <div class="card-value" style="color:#3498db">{per_day:.2f}</div>
          <div class="card-sub">trading days with activity: {freq['days_with_trade']:,}</div>
        </div>
        <div class="card">
          <div class="card-label">Avg per Week</div>
          <div class="card-value" style="color:#3498db">{per_wk:.1f}</div>
        </div>
        <div class="card">
          <div class="card-label">Avg per Month</div>
          <div class="card-value" style="color:#3498db">{per_mo:.0f}</div>
        </div>
        <div class="card">
          <div class="card-label">Max in One Day</div>
          <div class="card-value" style="color:#e67e22">{freq['max_per_day']}</div>
        </div>
      </div>
      <p style="font-size:13px;color:#ccc;margin-top:8px">
        <b>Cadence:</b> {cadence}<br>
        <b>Patience tip:</b> seeing no trade for
        {avg_hrs_between:.0f}+ hours is completely normal for this strategy
        — the ML only signals when conditions meet its confidence threshold.
        Forced trading below threshold is worse than waiting.
      </p>
    </div>"""


def build_timing_summary(best: dict) -> str:
    """Show what timing parameters the ML picked and why they matter."""
    if not best:
        return ""

    entry_tf = best.get("entry_tf", "?")
    htf_tf   = best.get("htf_tf", 0)
    conf     = best.get("confidence", 0) or 0
    sl_atr   = best.get("sl_atr", 0) or 0
    rr       = best.get("rr", 0) or 0
    be_r     = best.get("be_r", 0)

    htf_desc = f"{htf_tf}m HTF confirmation" if htf_tf else "No HTF filter (entry TF only)"

    tf_desc = {
        1:  "1m bars — ultra-short term, very frequent signals, tight spreads matter",
        3:  "3m bars — short term, good balance of frequency and signal quality",
        5:  "5m bars — standard intraday, most reliable for US30 institutional flow",
        10: "10m bars — medium term, fewer but higher quality signals",
        15: "15m bars — swing-intraday hybrid, strongest HTF alignment",
    }.get(entry_tf, f"{entry_tf}m bars")

    htf_tf_desc = {
        15: "15m — short-term trend confirmation",
        30: "30m — medium-term trend, filters out noise",
        60: "60m — hourly regime filter, strong bias signal",
        0:  "None — entry TF signal only, no additional filter",
    }.get(htf_tf, f"{htf_tf}m")

    be_desc = f"Move SL to break-even after +{be_r}R profit (protects capital)" if be_r else "Break-even not used"

    return f"""
    <div class="info-box">
      <h3>ML-Selected Timing Parameters — {best.get('strategy_id','?')}</h3>
      <table class="info-table">
        <tr>
          <td>Entry Timeframe</td>
          <td><b style="color:#3498db">{entry_tf}m bars</b> — {tf_desc}</td>
        </tr>
        <tr>
          <td>HTF Confirmation</td>
          <td><b style="color:#3498db">{htf_tf_desc}</b> — {htf_desc}</td>
        </tr>
        <tr>
          <td>Signal Confidence</td>
          <td><b style="color:#3498db">{conf:.2f}</b> — ensemble threshold (XGB+RF average).
              Only signals with probability ≥ {conf:.2f} (long) or ≤ {1-conf:.2f} (short) fire.</td>
        </tr>
        <tr>
          <td>Stop Loss</td>
          <td><b style="color:#e67e22">{sl_atr:.2f}× ATR(14)</b> — dynamically sized to
              current volatility. Tighter in quiet markets, wider in volatile.</td>
        </tr>
        <tr>
          <td>Risk:Reward</td>
          <td><b style="color:#2ecc71">1:{rr:.2f}</b> — TP placed at {rr:.2f}× the SL distance.
              Minimum expectancy requires win rate > {100/(1+rr):.0f}%.</td>
        </tr>
        <tr>
          <td>Break-Even</td>
          <td><b style="color:#9b59b6">{be_desc}</b></td>
        </tr>
      </table>
      <p class="note" style="margin-top:8px">
        These parameters were discovered by Genetic Algorithm + Optuna Bayesian search
        across {os.getenv('PER_TF_TRIALS','150')} trials per timeframe, optimising Sharpe ratio,
        then ranked by Efficiency Ratio = (peak profit / max drawdown $) × {ER_MULTIPLIER}.
      </p>
    </div>"""


def build_monthly_heatmap(strategy_id: str, monthly_pnl: list) -> str:
    if not HAS_PLOTLY or not monthly_pnl:
        return "<p>No monthly data available.</p>"

    df     = pd.DataFrame(monthly_pnl)
    years  = sorted(df["year"].unique())
    months = list(range(1, 13))
    z = []; text = []
    for yr in years:
        row_z = []; row_t = []
        yr_df = df[df["year"] == yr]
        for mo in months:
            mo_row = yr_df[yr_df["month"] == mo]
            if mo_row.empty:
                row_z.append(None); row_t.append("")
            else:
                val = float(mo_row["pnl_money"].iloc[0])
                wr  = float(mo_row["win_rate"].iloc[0])
                nt  = int(mo_row["n_trades"].iloc[0])
                row_z.append(val)
                row_t.append(f"${val:+,.0f}\n{nt}T {wr:.0f}%")
        z.append(row_z); text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z, x=MONTH_NAMES, y=[str(yr) for yr in years],
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#c0392b"],[0.45,"#922b21"],[0.5,"#2d2d2d"],
                    [0.55,"#1a5276"],[1,"#2ecc71"]],
        zmid=0, showscale=True,
        hovertemplate="<b>%{y} %{x}</b><br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Monthly P&L — {strategy_id}  ($ | trades | win%)",
                   font=dict(color="#e0e0e0")),
        height=max(250, len(years) * 38 + 100),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
        margin=dict(l=60, r=20, t=55, b=40),
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def _trade_freq_from_n(n_trades: int, backtest_start: str) -> dict:
    """Compute per-day/week/month from total trade count and backtest window."""
    try:
        start = pd.Timestamp(backtest_start)
        end   = pd.Timestamp.now()
        total_days   = max((end - start).days, 1)
        trading_days = max(total_days * 5 / 7, 1)   # approximate trading days
        total_weeks  = max(total_days / 7, 1)
        total_months = max(total_days / 30.44, 1)
        per_day   = n_trades / trading_days
        per_week  = n_trades / total_weeks
        per_month = n_trades / total_months
        avg_hours = 24.0 / per_day if per_day > 0 else 9999
        return {
            "per_day": per_day, "per_week": per_week,
            "per_month": per_month, "avg_hours": avg_hours,
        }
    except Exception:
        return {}


def build_strategy_table(strategies: list) -> str:
    if not strategies:
        return "<p>No strategies in database. Run training first.</p>"

    rows_html = ""
    for s in sorted(strategies, key=lambda x: (x.get("tf", 0), x.get("rank", 99))):
        sid  = s.get("strategy_id", "?")
        er   = s.get("efficiency_ratio")
        pf   = s.get("profit_factor")
        wr   = s.get("win_rate")
        sh   = s.get("sharpe")
        dd   = s.get("max_dd_pct")
        ddm  = s.get("max_dd_money")
        tp   = s.get("total_profit")
        nt   = s.get("n_trades", 0)
        exp  = s.get("expectancy")
        be   = s.get("be_r", 0)

        # Trade frequency (computed from n_trades, no per-trade data needed)
        freq = _trade_freq_from_n(nt, BACKTEST_START)
        per_day  = freq.get("per_day", 0)
        per_week = freq.get("per_week", 0)
        avg_hrs  = freq.get("avg_hours", 9999)
        if avg_hrs < 1:
            freq_str = f"{1/avg_hrs:.1f}/hr"
        elif avg_hrs < 24:
            freq_str = f"~{avg_hrs:.0f}h gap"
        else:
            freq_str = f"~{avg_hrs/24:.1f}d gap"

        active_badge = (
            '<span style="color:#2ecc71;font-weight:bold"> ★</span>'
            if s.get("is_active") else ""
        )
        rows_html += f"""
        <tr>
          <td><b>{sid}</b>{active_badge}</td>
          <td>{s.get('tf','?')}m</td>
          <td>{s.get('rank','?')}</td>
          <td>{s.get('entry_tf','?')}m / {s.get('htf_tf',0)}m</td>
          <td>{_fmt(s.get('sl_atr'),'.3f')}×ATR</td>
          <td>1:{_fmt(s.get('rr'),'.2f')}</td>
          <td>{_fmt(s.get('confidence'),'.2f')}</td>
          <td>{'OFF' if not be else f'+{be}R'}</td>
          <td style="color:{_color(er)}">{_fmt(er,'.2f')}</td>
          <td style="color:{_color(sh)}">{_fmt(sh,'.2f')}</td>
          <td style="color:{_color(wr and wr-50)}">{_fmt(wr,'.1f','%')}</td>
          <td style="color:{_color(pf and pf-1)}">{_fmt(pf,'.2f')}</td>
          <td style="color:{_color(tp)}">${_fmt(tp,',.0f')}</td>
          <td style="color:{_color(dd,False)}">{_fmt(dd,'.1f','%')}</td>
          <td style="color:#e74c3c">${_fmt(ddm,',.0f')}</td>
          <td>{nt:,}</td>
          <td title="{per_day:.2f}/day  {per_week:.1f}/week"
              style="color:#3498db">{freq_str}</td>
          <td style="color:{_color(exp)}">{_fmt(exp,'+.3f')}R</td>
        </tr>"""

    return f"""
    <table class="strategy-table">
      <thead>
        <tr>
          <th>Strategy</th><th>TF</th><th>Rank</th>
          <th>Entry/HTF</th><th>SL</th><th>R:R</th>
          <th>Conf</th><th>BE</th>
          <th>ER ↓</th><th>Sharpe</th><th>Win%</th>
          <th>PF</th><th>Profit $</th>
          <th>MaxDD%</th><th>MaxDD$</th>
          <th>Trades</th><th title="Avg gap between trades">Freq</th><th>Expect</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class="note">
      ★ = running live &nbsp;|&nbsp;
      ER = (peak profit / max DD $) × {ER_MULTIPLIER} &nbsp;|&nbsp;
      MaxDD% = relative (peak-to-trough as % of peak) &nbsp;|&nbsp;
      MaxDD$ = absolute money &nbsp;|&nbsp;
      Freq = avg gap between trades (hover for per-day/week) &nbsp;|&nbsp;
      Sorted by TF then rank. All stats from {BACKTEST_START}.
    </p>"""


def build_feature_importance_chart(symbol: str, tf: int) -> str:
    if not HAS_PLOTLY:
        return ""
    pairs = _load_feature_importance(symbol, tf)
    if not pairs:
        return "<p>Feature importance not available. Run training first.</p>"

    names  = [p[0] for p in reversed(pairs)]
    values = [p[1] for p in reversed(pairs)]
    max_v  = max(values) if values else 1

    def _feat_color(n):
        institutional = ["vwap","vp_","cvd","orb","poc","val","vah","hvn","lvn",
                         "regime","session","htf","imbalance","absorption"]
        session_kws   = ["is_us","is_london","is_asian","is_premarket","uk_hour",
                         "us_hour","hour_sin","hour_cos","dow_sin","dow_cos"]
        if any(k in n for k in institutional):   return "#3498db"
        if any(k in n for k in session_kws):     return "#e74c3c"
        if any(k in n for k in ["stoch","macd","ema","rsi","bb","atr","willr"]):
            return "#2ecc71"
        return "#e67e22"

    colors = [_feat_color(n) for n in names]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
    ))
    fig.add_annotation(
        x=max_v * 0.02, y=len(names) - 1,
        text="&#x1F535; Institutional  &#x1F7E2; Technical  &#x1F534; Session/Time  &#x1F7E0; Other",
        showarrow=False, font=dict(size=10, color="#aaa"), xanchor="left",
    )
    fig.update_layout(
        title=dict(text=f"Top 15 Features — {symbol} {tf}m (XGBoost)",
                   font=dict(color="#e0e0e0")),
        height=430,
        plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0", size=11),
        margin=dict(l=200, r=90, t=50, b=20),
        xaxis=dict(gridcolor="#2d2d2d", title="Importance"),
        yaxis=dict(gridcolor="#2d2d2d"),
    )
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)


def build_live_trades_table(trades: list) -> str:
    if not trades:
        return "<p>No live trades recorded yet.</p>"
    rows_html = ""
    for t in sorted(trades, key=lambda x: x.get("opened_at",""), reverse=True)[:50]:
        pnl    = t.get("pnl")
        be     = "BE" if t.get("be_done") else "—"
        status = t.get("status","open")
        rows_html += f"""
        <tr>
          <td>{t.get('opened_at','?')[:16]}</td>
          <td>{t.get('strategy_id','?')}</td>
          <td>{t.get('symbol','?')}</td>
          <td>{'LONG' if t.get('direction')==1 else 'SHORT'}</td>
          <td>{t.get('entry_tf','?')}m</td>
          <td>{t.get('confidence',0):.2f}</td>
          <td>${t.get('risk_amount',0):.0f}</td>
          <td style="color:{_color(pnl)}">{('$'+_fmt(pnl,',.2f')) if pnl is not None else '—'}</td>
          <td style="color:{'#2ecc71' if be=='BE' else '#888'}">{be}</td>
          <td>{status}</td>
          <td>{t.get('closed_at','—')[:16] if t.get('closed_at') else '—'}</td>
        </tr>"""
    return f"""
    <table class="strategy-table">
      <thead>
        <tr><th>Opened</th><th>Strategy</th><th>Symbol</th><th>Dir</th>
            <th>TF</th><th>Conf</th><th>Risk</th><th>P&L</th>
            <th>BE</th><th>Status</th><th>Closed</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ─────────────────────────────────────────────────────────────────────
# Main report builder
# ─────────────────────────────────────────────────────────────────────

def build_report():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"Building report... [{now_str}]")

    if not HAS_PLOTLY:
        print("ERROR: plotly required. pip install plotly")
        return

    strategies = get_all_strategies() if HAS_DB else []
    symbols    = sorted(set(s["symbol"] for s in strategies)) if strategies else []
    if not symbols:
        symbols = [p.stem.replace("_params","")
                   for p in PARAMS_DIR.glob("*_params.json")]

    # Load equity curves for top-10 by ER
    strategies_with_curves = []
    for s in sorted(strategies, key=lambda x: -(x.get("efficiency_ratio") or 0))[:10]:
        curve = get_equity_curve(s["strategy_id"]) if HAS_DB else []
        if curve:
            strategies_with_curves.append({**s, "curve": curve})

    # Identify best strategy overall
    best = strategies_with_curves[0] if strategies_with_curves else (
        strategies[0] if strategies else None
    )
    best_trades = []
    if best and HAS_DB:
        best_trades = get_backtest_trades(best["strategy_id"])

    # ── Summary cards ────────────────────────────────────────────────
    summary_cards = ""
    if best:
        er   = best.get("efficiency_ratio", 0) or 0
        wr   = best.get("win_rate", 0) or 0
        pf   = best.get("profit_factor", 0) or 0
        sh   = best.get("sharpe", 0) or 0
        dd   = best.get("max_dd_pct", 0) or 0
        ddm  = best.get("max_dd_money", 0) or 0
        tp   = best.get("total_profit", 0) or 0
        nt   = best.get("n_trades", 0) or 0
        sid  = best.get("strategy_id","?")
        tf   = best.get("entry_tf","?")
        htf  = best.get("htf_tf", 0)
        conf = best.get("confidence", 0) or 0
        live_trades = get_open_trades() if HAS_DB else []

        def card(label, val, color="#3498db", sub=""):
            return (f'<div class="card">'
                    f'<div class="card-label">{label}</div>'
                    f'<div class="card-value" style="color:{color}">{val}</div>'
                    + (f'<div class="card-sub">{sub}</div>' if sub else "")
                    + "</div>")

        htf_str = f"{htf}m" if htf else "None"
        summary_cards = f"""
        <div class="best-strategy-banner">
          Best strategy: <b>{sid}</b> &nbsp;|&nbsp;
          Entry: <b>{tf}m bars</b> &nbsp;|&nbsp;
          HTF: <b>{htf_str}</b> &nbsp;|&nbsp;
          Confidence threshold: <b>{conf:.2f}</b> &nbsp;|&nbsp;
          Trades: <b>{nt:,}</b> from {BACKTEST_START}
        </div>
        <div class="cards-row">
          {card("Efficiency Ratio", f"{er:.2f}",
                "#2ecc71" if er>1.5 else ("#e67e22" if er>1 else "#e74c3c"),
                f"Peak profit / max DD × {ER_MULTIPLIER}")}
          {card("Sharpe Ratio", f"{sh:.2f}",
                "#2ecc71" if sh>1 else ("#e67e22" if sh>0.5 else "#e74c3c"),
                ">1.0 good, >2.0 excellent")}
          {card("Win Rate", f"{wr:.1f}%",
                "#2ecc71" if wr>50 else "#e74c3c")}
          {card("Profit Factor", f"{pf:.2f}",
                "#2ecc71" if pf>1.5 else ("#e67e22" if pf>1 else "#e74c3c"),
                ">1.5 good, >2.0 excellent")}
          {card("Total Profit", f"${tp:,.0f}",
                "#2ecc71" if tp>0 else "#e74c3c",
                f"from ${START_BALANCE:,.0f} start")}
          {card("Max DD (Relative)", f"{dd:.1f}%",
                "#e74c3c" if dd>20 else ("#e67e22" if dd>10 else "#2ecc71"),
                "peak-to-trough as % of peak")}
          {card("Max DD (Absolute $)", f"${ddm:,.0f}",
                "#e74c3c" if ddm>2000 else ("#e67e22" if ddm>1000 else "#2ecc71"),
                "largest money drawdown ever")}
          {card("Open Positions", f"{len(live_trades)}", "#9b59b6", "currently live")}
        </div>"""

    # ── Equity chart ────────────────────────────────────────────────
    equity_div = build_equity_chart(strategies_with_curves)

    # ── Drawdown analysis for best strategy ─────────────────────────
    dd_analysis_div = ""
    if best and strategies_with_curves:
        best_curve = strategies_with_curves[0]["curve"]
        dd_analysis_div = build_drawdown_analysis(
            strategies_with_curves[0]["strategy_id"], best_curve
        )

    # ── Timing summary ──────────────────────────────────────────────
    timing_div = build_timing_summary(best)

    # ── Session + hour charts ────────────────────────────────────────
    session_div = ""
    hour_div    = ""
    freq_div    = ""
    if best_trades:
        session_div = build_session_chart(best["strategy_id"], best_trades)
        hour_div    = build_hour_heatmap(best["strategy_id"], best_trades)
        freq_div    = build_frequency_section(best, best_trades)

    # ── Strategy table ──────────────────────────────────────────────
    strategy_tbl = build_strategy_table(strategies)

    # ── Per-symbol monthly + feature importance ─────────────────────
    monthly_sections = ""
    for symbol in symbols:
        sym_strats = [s for s in strategies if s.get("symbol") == symbol]
        if not sym_strats:
            continue
        best_sym = sorted(sym_strats,
                          key=lambda s: s.get("efficiency_ratio") or 0,
                          reverse=True)[0]
        sid_best = best_sym.get("strategy_id","?")
        tf_best  = best_sym.get("entry_tf", 5)
        monthly  = get_monthly_pnl(sid_best) if HAS_DB else []
        monthly_sections += f"""
        <h2 style="color:#3498db;margin-top:40px">{symbol}</h2>
        <h3>Monthly P&L — {sid_best}</h3>
        {build_monthly_heatmap(sid_best, monthly)}
        <h3 style="margin-top:30px">Feature Importance — {symbol} {tf_best}m
            <span style="font-size:12px;color:#888;font-weight:normal">
            (what the ML uses most for decisions)</span></h3>
        {build_feature_importance_chart(symbol, tf_best)}"""

    # ── Optimisation criteria ────────────────────────────────────────
    optim_html = f"""
    <div class="info-box">
      <h3>Optimisation Criteria &amp; Search Space</h3>
      <table class="info-table">
        <tr><td>Objective</td>
            <td>Sharpe (GA + global Optuna) → Efficiency Ratio (final ranking)</td></tr>
        <tr><td>Method</td>
            <td>Genetic Algorithm (40 gen, pop=80) →
                per-TF Optuna Bayesian TPE ({os.getenv('PER_TF_TRIALS','150')} trials)</td></tr>
        <tr><td>Per-TF search</td>
            <td>Each TF optimised independently, top-{os.getenv('TOP_N_STRATEGIES','5')} saved</td></tr>
        <tr><td>Entry TF options</td><td>1m / 3m / 5m / 10m / 15m</td></tr>
        <tr><td>HTF options</td><td>None / 15m / 30m / 60m</td></tr>
        <tr><td>SL range</td><td>0.5 – 3.0 × ATR(14)</td></tr>
        <tr><td>R:R range</td><td>1.0 – 4.0</td></tr>
        <tr><td>Confidence</td><td>0.50 – 0.85 (ensemble threshold)</td></tr>
        <tr><td>Break-even</td><td>OFF / +1R / +2R trigger</td></tr>
        <tr><td>Efficiency Ratio</td>
            <td>ER = (peak profit $) / (max drawdown $) × {ER_MULTIPLIER}<br>
                Rewards high profit relative to worst drawdown.
                ER > 1.5 is good, > 2.5 is excellent.</td></tr>
        <tr><td>Backtest window</td><td>{BACKTEST_START} to present (~6 years)</td></tr>
        <tr><td>Risk mode</td>
            <td>{RISK_MODE} — {'{}% of balance/trade'.format(RISK_PCT)
                if RISK_MODE=='percent' else f'${FIXED_RISK_AMT:.0f} fixed/trade'}</td></tr>
        <tr><td>Risk cap</td>
            <td>Max {os.getenv('RISK_CAP_PCT','2.0')}% of balance across all open trades
                (BE trades are excluded — zero risk once SL at break-even)</td></tr>
      </table>
    </div>"""

    live_tbl = build_live_trades_table(get_open_trades() if HAS_DB else [])

    # ── Assemble HTML ────────────────────────────────────────────────
    risk_str = (_fmt(RISK_PCT,'.1f','%') if RISK_MODE=='percent'
                else f'${FIXED_RISK_AMT:.0f}')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML Trading Report — {now_str}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{
      font-family:'Segoe UI',sans-serif;
      background:#0f0f23; color:#e0e0e0; padding:24px;
    }}
    h1 {{ color:#3498db; font-size:22px; margin-bottom:6px; }}
    h2 {{ color:#3498db; font-size:18px; margin:28px 0 12px; }}
    h3 {{ color:#aaa; font-size:14px; margin:16px 0 8px; }}
    h4 {{ color:#ccc; font-size:13px; margin:12px 0 6px; }}
    .subtitle {{ color:#888; font-size:13px; margin-bottom:24px; }}

    .best-strategy-banner {{
      background:#1a1a2e; border-left:4px solid #3498db;
      padding:10px 16px; border-radius:4px;
      margin-bottom:16px; font-size:14px; color:#ccc;
    }}

    .cards-row {{
      display:flex; flex-wrap:wrap; gap:12px; margin-bottom:24px;
    }}
    .card {{
      background:#1a1a2e; border:1px solid #2d2d4e;
      border-radius:8px; padding:14px 18px;
      min-width:120px; flex:1;
    }}
    .card-label {{ font-size:10px; color:#888; text-transform:uppercase; letter-spacing:.5px; }}
    .card-value {{ font-size:22px; font-weight:bold; margin:4px 0; }}
    .card-sub   {{ font-size:10px; color:#666; }}

    .strategy-table {{
      width:100%; border-collapse:collapse; font-size:12px; margin-bottom:8px;
    }}
    .strategy-table th {{
      background:#1a1a2e; color:#888; padding:8px 10px;
      text-align:left; border-bottom:1px solid #2d2d4e; white-space:nowrap;
    }}
    .strategy-table td {{
      padding:7px 10px; border-bottom:1px solid #1a1a2e; white-space:nowrap;
    }}
    .strategy-table tr:hover td {{ background:#1a1a2e; }}
    .strategy-table tr:nth-child(even) td {{ background:#0d0d1f; }}

    .info-box {{
      background:#1a1a2e; border:1px solid #2d2d4e;
      border-radius:8px; padding:16px 20px; margin:20px 0;
    }}
    .info-box h3 {{ color:#3498db; margin-bottom:10px; font-size:15px; }}
    .info-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    .info-table td {{ padding:7px 10px; border-bottom:1px solid #2d2d4e; }}
    .info-table td:first-child {{ color:#888; width:200px; white-space:nowrap; }}

    .section {{ margin-bottom:30px; }}
    .divider {{ border:none; border-top:1px solid #2d2d4e; margin:30px 0; }}
    .note {{ color:#888; font-size:11px; margin-top:8px; line-height:1.5; }}
  </style>
</head>
<body>

<h1>ML Trading System — Performance Report</h1>
<div class="subtitle">
  Generated: {now_str} &nbsp;|&nbsp;
  Backtest from: {BACKTEST_START} &nbsp;|&nbsp;
  Risk: {RISK_MODE} ({risk_str}/trade) &nbsp;|&nbsp;
  Starting balance: ${START_BALANCE:,.0f}
</div>

<div class="section">
  {summary_cards if summary_cards else
   '<p style="color:#888">Run training first: python train.py</p>'}
</div>

<hr class="divider">
<h2>ML-Selected Timing Parameters</h2>
<div class="section">{timing_div}</div>

<hr class="divider">
<h2>Equity Curve &amp; Drawdown</h2>
<div class="note" style="margin-bottom:8px">
  Top 10 strategies by efficiency ratio. Each line = one parameter set.
  Use the legend to toggle individual strategies on/off.
</div>
<div class="section">{equity_div}</div>

<hr class="divider">
<h2>Drawdown Analysis</h2>
<div class="section">{dd_analysis_div if dd_analysis_div else
  '<p style="color:#888">No equity data yet. Run training first.</p>'}</div>

<hr class="divider">
<h2>Session Timing Analysis</h2>
<div class="note" style="margin-bottom:12px">
  Based on backtest trades from {BACKTEST_START}.
  Session classification uses market hours — timing is crucial in trading
  because institutional flow concentrates in specific windows.
</div>
<div class="section">{session_div if session_div else
  '<p style="color:#888">No trade data yet. Run training first.</p>'}</div>

<hr class="divider">
<h2>Trade Distribution by Hour (UTC)</h2>
<div class="section">{hour_div if hour_div else
  '<p style="color:#888">No trade data yet. Run training first.</p>'}</div>

{freq_div}

<hr class="divider">
<h2>All Strategies — Top {os.getenv('TOP_N_STRATEGIES','5')} per Timeframe</h2>
<div class="section">{strategy_tbl}</div>

<hr class="divider">
<h2>Monthly P&L &amp; Feature Importance</h2>
{monthly_sections if monthly_sections else
 '<p style="color:#888">No per-symbol data yet.</p>'}

<hr class="divider">
{optim_html}

<hr class="divider">
<h2>Live Trades (last 50)</h2>
<div class="section">{live_tbl}</div>

<div class="note" style="margin-top:30px;text-align:center;padding:20px">
  ML Trading System &nbsp;|&nbsp; Report generated {now_str} &nbsp;|&nbsp;
  Regenerate: <code>python report.py</code>
</div>

</body>
</html>"""

    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"\nReport saved: {OUTPUT_FILE}")
    print("Open it in any browser (double-click the file).")
    print()

    # ── Quick text summary ───────────────────────────────────────────
    print("=" * 70)
    print("  QUICK SUMMARY")
    print("=" * 70)
    if strategies:
        best_s = sorted(strategies,
                        key=lambda x: -(x.get("efficiency_ratio") or 0))
        print(f"  {'Strategy ID':<25} {'ER':>5} {'Sh':>5} {'WR':>6} {'PF':>5} "
              f"{'DD%':>6} {'DD$':>8} {'Profit':>10}")
        print("  " + "-" * 68)
        for s in best_s[:15]:
            er  = s.get("efficiency_ratio") or 0
            sh  = s.get("sharpe") or 0
            wr  = s.get("win_rate") or 0
            pf  = s.get("profit_factor") or 0
            dd  = s.get("max_dd_pct") or 0
            ddm = s.get("max_dd_money") or 0
            tp  = s.get("total_profit") or 0
            act = "★" if s.get("is_active") else " "
            print(f"  {act}{s['strategy_id']:<24} {er:>5.2f} {sh:>5.2f} "
                  f"{wr:>5.1f}% {pf:>5.2f} {dd:>5.1f}% ${ddm:>7,.0f} "
                  f"${tp:>9,.0f}")

        print()
        if best_trades:
            stats = _session_stats(best_trades)
            print("  SESSION BREAKDOWN (best strategy):")
            for sess in SESSION_ORDER:
                d = stats["by_session"].get(sess)
                if not d or d["n"] == 0:
                    continue
                wr_s = d["wins"] / d["n"] * 100
                print(f"    {sess:<22} {d['n']:>5} trades  "
                      f"WR: {wr_s:>5.1f}%  P&L: ${d['pnl']:>+9,.0f}")
    else:
        print("  No strategies found. Run: python train.py")
    print("=" * 70)


if __name__ == "__main__":
    build_report()
