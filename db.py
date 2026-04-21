"""
db.py — SQLite persistence layer for the ML trading system.

Schema:
  strategy_params  — top-5 param sets per symbol/TF, ranked by efficiency ratio
  equity_curves    — daily equity snapshots per strategy backtest
  live_trades      — all live trade records with BE tracking
  optuna_trials    — raw trial results from per-TF Optuna runs
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", r"F:\trading_ml"))
DB_PATH  = BASE_DIR / "trading.db"


# ─────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────

def _conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")   # safe concurrent reads
    c.execute("PRAGMA foreign_keys=ON")
    return c


# ─────────────────────────────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist. Safe to call every startup."""
    with _conn() as c:
        c.executescript("""
        -- Top-N param sets per symbol/TF (ranked by efficiency ratio)
        CREATE TABLE IF NOT EXISTS strategy_params (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id      TEXT    UNIQUE NOT NULL,  -- e.g. US30_3m_rank1
            symbol           TEXT    NOT NULL,
            tf               INTEGER NOT NULL,
            rank             INTEGER NOT NULL,         -- 1 = best ER
            entry_tf         INTEGER NOT NULL,
            htf_tf           INTEGER NOT NULL,
            sl_atr           REAL,
            rr               REAL,
            tp_mult          REAL,
            confidence       REAL,
            htf_weight       REAL,
            be_r             INTEGER,
            -- backtest stats
            sharpe           REAL,
            efficiency_ratio REAL,   -- (peak-start) / max_dd * ER_MULT
            win_rate         REAL,
            profit_factor    REAL,
            max_dd_pct       REAL,
            max_dd_money     REAL,
            total_profit     REAL,   -- peak - start in account currency
            n_trades         INTEGER,
            expectancy       REAL,
            sortino          REAL,   -- Sharpe using only downside deviation
            calmar           REAL,   -- annualised return / max DD%
            haircut_sharpe   REAL,   -- Sharpe deflated for multiple testing (n_trials)
            -- Monte Carlo robustness (1000 bootstrap resamples of trade sequence)
            mc_sharpe_p5     REAL,   -- worst 5% Sharpe across simulations
            mc_sharpe_p50    REAL,   -- median Sharpe
            mc_sharpe_p95    REAL,   -- best 5% Sharpe
            mc_dd_p95        REAL,   -- worst-case drawdown (95th percentile)
            mc_profit_p5     REAL,   -- worst 5% final profit %
            mc_pass          INTEGER DEFAULT 0,  -- 1 = profitable in 95% of MC runs
            -- Parameter sensitivity score (0-100, higher = more robust to param nudges)
            sensitivity_score REAL,
            robust            INTEGER DEFAULT 0, -- 1 = score >= 50
            -- SPA bootstrap edge test: p-value for H0: expected trade P&L <= 0
            -- p < 0.05 → statistically significant positive edge
            spa_p_value       REAL,
            -- Expected performance baselines (from backtest at training time).
            -- Used by _SmartKillSwitch to compare live metrics against training expectations
            -- rather than firing on absolute thresholds regardless of strategy character.
            expected_sharpe         REAL,
            expected_win_rate       REAL,
            expected_trades_per_day REAL DEFAULT 0.0,
            -- metadata
            updated_at       TEXT,
            is_active        INTEGER DEFAULT 0   -- 1 = running live
        );

        -- Daily equity snapshots per strategy (for charting)
        CREATE TABLE IF NOT EXISTS equity_curves (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id  TEXT    NOT NULL,
            date         TEXT    NOT NULL,
            equity       REAL,
            balance      REAL,
            drawdown_pct REAL,
            UNIQUE(strategy_id, date)
        );

        -- Monthly P&L per strategy
        CREATE TABLE IF NOT EXISTS monthly_pnl (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id  TEXT    NOT NULL,
            year         INTEGER NOT NULL,
            month        INTEGER NOT NULL,
            pnl_r        REAL,      -- total R gained that month
            pnl_money    REAL,      -- total $ gained that month
            n_trades     INTEGER,
            win_rate     REAL,
            UNIQUE(strategy_id, year, month)
        );

        -- All live trades (open and closed)
        CREATE TABLE IF NOT EXISTS live_trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id  TEXT    NOT NULL,
            symbol       TEXT    NOT NULL,
            direction    INTEGER NOT NULL,   -- 1=long, -1=short
            entry_price  REAL,
            sl_price     REAL,
            tp_price     REAL,
            lot          REAL,
            confidence   REAL,
            entry_tf     INTEGER,
            htf_tf       INTEGER,
            ticket       INTEGER,            -- MT5 position ticket
            risk_amount  REAL,              -- $ at risk when opened
            pnl          REAL,
            be_done      INTEGER DEFAULT 0,  -- 1 = SL moved to BE (no capital at risk)
            data_source  TEXT,
            opened_at    TEXT,
            closed_at    TEXT,
            status       TEXT DEFAULT 'open'  -- open / closed
        );

        -- Per-trade backtest results (for session/timing analysis)
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id  TEXT    NOT NULL,
            entry_time   TEXT,
            close_time   TEXT,
            session      TEXT,
            direction    INTEGER,
            pnl_r        REAL,
            pnl_money    REAL,
            win          INTEGER,
            hour_utc     INTEGER,
            day_of_week  INTEGER
        );

        -- Raw Optuna trial results per symbol/TF run
        CREATE TABLE IF NOT EXISTS optuna_trials (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT    NOT NULL,
            tf           INTEGER NOT NULL,
            trial_number INTEGER NOT NULL,
            params_json  TEXT,               -- JSON of param dict
            sharpe       REAL,
            run_date     TEXT
        );
        """)

        # ── Schema migrations (safe to run on existing DB) ────────────
        _add_column_if_missing(c, "strategy_params", "sortino",           "REAL")
        _add_column_if_missing(c, "strategy_params", "calmar",            "REAL")
        _add_column_if_missing(c, "strategy_params", "haircut_sharpe",    "REAL")
        _add_column_if_missing(c, "strategy_params", "mc_sharpe_p5",      "REAL")
        _add_column_if_missing(c, "strategy_params", "mc_sharpe_p50",     "REAL")
        _add_column_if_missing(c, "strategy_params", "mc_sharpe_p95",     "REAL")
        _add_column_if_missing(c, "strategy_params", "mc_dd_p95",         "REAL")
        _add_column_if_missing(c, "strategy_params", "mc_profit_p5",      "REAL")
        _add_column_if_missing(c, "strategy_params", "mc_pass",           "INTEGER DEFAULT 0")
        _add_column_if_missing(c, "strategy_params", "sensitivity_score", "REAL")
        _add_column_if_missing(c, "strategy_params", "robust",            "INTEGER DEFAULT 0")
        _add_column_if_missing(c, "strategy_params", "spa_p_value",       "REAL")
        _add_column_if_missing(c, "strategy_params", "expected_sharpe",         "REAL")
        _add_column_if_missing(c, "strategy_params", "expected_win_rate",       "REAL")
        _add_column_if_missing(c, "strategy_params", "expected_trades_per_day", "REAL")


def _add_column_if_missing(conn, table: str, column: str, col_type: str):
    """ALTER TABLE ADD COLUMN — silently skips if column already exists."""
    existing = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


# ─────────────────────────────────────────────────────────────────────
# strategy_params CRUD
# ─────────────────────────────────────────────────────────────────────

def upsert_strategy(row: dict):
    """Insert or replace a strategy_params row."""
    # Round floats to 2 dp for readability in logs, DB, and reports.
    # Trading params (sl_atr, tp_mult, confidence, htf_weight) are kept at 4 dp
    # since they feed live signal generation directly.
    _4dp = {"sl_atr", "tp_mult", "confidence", "htf_weight"}
    for k, v in row.items():
        if isinstance(v, float):
            row[k] = round(v, 4 if k in _4dp else 2)

    row.setdefault("updated_at", datetime.now().isoformat())
    row.setdefault("rr",                None)   # removed param — kept in DB for history
    # Default MC / sensitivity columns so old callers without them still work
    row.setdefault("sortino",           None)
    row.setdefault("calmar",            None)
    row.setdefault("haircut_sharpe",    None)
    row.setdefault("mc_sharpe_p5",      None)
    row.setdefault("mc_sharpe_p50",     None)
    row.setdefault("mc_sharpe_p95",     None)
    row.setdefault("mc_dd_p95",         None)
    row.setdefault("mc_profit_p5",      None)
    row.setdefault("mc_pass",           0)
    row.setdefault("sensitivity_score", None)
    row.setdefault("robust",            0)
    row.setdefault("spa_p_value",       None)
    row.setdefault("expected_sharpe",         None)
    row.setdefault("expected_win_rate",       None)
    row.setdefault("expected_trades_per_day", None)
    with _conn() as c:
        c.execute("""
            INSERT INTO strategy_params
              (strategy_id, symbol, tf, rank, entry_tf, htf_tf,
               sl_atr, rr, tp_mult, confidence, htf_weight, be_r,
               sharpe, efficiency_ratio, win_rate, profit_factor,
               max_dd_pct, max_dd_money, total_profit, n_trades, expectancy,
               sortino, calmar, haircut_sharpe,
               mc_sharpe_p5, mc_sharpe_p50, mc_sharpe_p95,
               mc_dd_p95, mc_profit_p5, mc_pass,
               sensitivity_score, robust, spa_p_value,
               expected_sharpe, expected_win_rate, expected_trades_per_day,
               updated_at, is_active)
            VALUES
              (:strategy_id,:symbol,:tf,:rank,:entry_tf,:htf_tf,
               :sl_atr,:rr,:tp_mult,:confidence,:htf_weight,:be_r,
               :sharpe,:efficiency_ratio,:win_rate,:profit_factor,
               :max_dd_pct,:max_dd_money,:total_profit,:n_trades,:expectancy,
               :sortino,:calmar,:haircut_sharpe,
               :mc_sharpe_p5,:mc_sharpe_p50,:mc_sharpe_p95,
               :mc_dd_p95,:mc_profit_p5,:mc_pass,
               :sensitivity_score,:robust,:spa_p_value,
               :expected_sharpe,:expected_win_rate,:expected_trades_per_day,
               :updated_at,:is_active)
            ON CONFLICT(strategy_id) DO UPDATE SET
              entry_tf=excluded.entry_tf, htf_tf=excluded.htf_tf,
              sl_atr=excluded.sl_atr, rr=excluded.rr,
              tp_mult=excluded.tp_mult, confidence=excluded.confidence,
              htf_weight=excluded.htf_weight, be_r=excluded.be_r,
              sharpe=excluded.sharpe,
              efficiency_ratio=excluded.efficiency_ratio,
              win_rate=excluded.win_rate,
              profit_factor=excluded.profit_factor,
              max_dd_pct=excluded.max_dd_pct,
              max_dd_money=excluded.max_dd_money,
              total_profit=excluded.total_profit,
              n_trades=excluded.n_trades,
              expectancy=excluded.expectancy,
              sortino=excluded.sortino,
              calmar=excluded.calmar,
              haircut_sharpe=excluded.haircut_sharpe,
              mc_sharpe_p5=excluded.mc_sharpe_p5,
              mc_sharpe_p50=excluded.mc_sharpe_p50,
              mc_sharpe_p95=excluded.mc_sharpe_p95,
              mc_dd_p95=excluded.mc_dd_p95,
              mc_profit_p5=excluded.mc_profit_p5,
              mc_pass=excluded.mc_pass,
              sensitivity_score=excluded.sensitivity_score,
              robust=excluded.robust,
              spa_p_value=excluded.spa_p_value,
              expected_sharpe=excluded.expected_sharpe,
              expected_win_rate=excluded.expected_win_rate,
              expected_trades_per_day=excluded.expected_trades_per_day,
              updated_at=excluded.updated_at
        """, row)


def get_strategy(strategy_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM strategy_params WHERE strategy_id=?",
            (strategy_id,)
        ).fetchone()
        return dict(row) if row else None


def get_top_strategy(symbol: str) -> dict | None:
    """Return the single best strategy for a symbol (rank 1, highest ER)."""
    with _conn() as c:
        row = c.execute("""
            SELECT * FROM strategy_params
            WHERE symbol=? AND efficiency_ratio IS NOT NULL
            ORDER BY efficiency_ratio DESC LIMIT 1
        """, (symbol,)).fetchone()
        return dict(row) if row else None


def get_all_strategies(symbol: str = None) -> list[dict]:
    with _conn() as c:
        if symbol:
            rows = c.execute(
                "SELECT * FROM strategy_params WHERE symbol=? ORDER BY tf, rank",
                (symbol,)
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM strategy_params ORDER BY symbol, tf, rank"
            ).fetchall()
        return [dict(r) for r in rows]


def set_strategy_active(strategy_id: str, active: bool):
    with _conn() as c:
        c.execute(
            "UPDATE strategy_params SET is_active=? WHERE strategy_id=?",
            (1 if active else 0, strategy_id)
        )


def delete_strategies_not_in_tfs(symbol: str, active_tfs: list):
    """
    Remove DB rows for TFs no longer in PARAM_SEEDS entry_tf_options.
    Called automatically at the start of each per-TF optimisation run so
    stale rows (e.g. old 1m / 30m entries) never pollute strategy selection.
    """
    placeholders = ",".join("?" * len(active_tfs))
    with _conn() as c:
        deleted = c.execute(
            f"DELETE FROM strategy_params "
            f"WHERE symbol=? AND tf NOT IN ({placeholders})",
            [symbol] + list(active_tfs)
        ).rowcount
    if deleted:
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"[DB] Removed {deleted} stale strategy row(s) for {symbol} "
            f"(TFs not in {active_tfs})"
        )


# ─────────────────────────────────────────────────────────────────────
# equity_curves
# ─────────────────────────────────────────────────────────────────────

def save_equity_curve(strategy_id: str, equity_df):
    """
    Save equity curve. equity_df must have columns:
    date (str YYYY-MM-DD), equity, balance, drawdown_pct
    """
    rows = [
        (strategy_id, str(r.date), r.equity, r.balance, r.drawdown_pct)
        for r in equity_df.itertuples()
    ]
    with _conn() as c:
        c.execute("DELETE FROM equity_curves WHERE strategy_id=?", (strategy_id,))
        c.executemany("""
            INSERT INTO equity_curves (strategy_id, date, equity, balance, drawdown_pct)
            VALUES (?,?,?,?,?)
        """, rows)


def get_equity_curve(strategy_id: str):
    """Returns list of dicts: date, equity, balance, drawdown_pct."""
    with _conn() as c:
        rows = c.execute("""
            SELECT date, equity, balance, drawdown_pct
            FROM equity_curves WHERE strategy_id=?
            ORDER BY date
        """, (strategy_id,)).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# monthly_pnl
# ─────────────────────────────────────────────────────────────────────

def save_monthly_pnl(strategy_id: str, monthly_rows: list[dict]):
    with _conn() as c:
        c.execute("DELETE FROM monthly_pnl WHERE strategy_id=?", (strategy_id,))
        c.executemany("""
            INSERT INTO monthly_pnl
              (strategy_id, year, month, pnl_r, pnl_money, n_trades, win_rate)
            VALUES
              (:strategy_id,:year,:month,:pnl_r,:pnl_money,:n_trades,:win_rate)
        """, [{**r, "strategy_id": strategy_id} for r in monthly_rows])


def get_monthly_pnl(strategy_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT year, month, pnl_r, pnl_money, n_trades, win_rate
            FROM monthly_pnl WHERE strategy_id=?
            ORDER BY year, month
        """, (strategy_id,)).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# live_trades
# ─────────────────────────────────────────────────────────────────────

def log_live_trade(trade: dict):
    trade.setdefault("opened_at", datetime.now().isoformat())
    trade.setdefault("status", "open")
    trade.setdefault("be_done", 0)
    with _conn() as c:
        c.execute("""
            INSERT INTO live_trades
              (strategy_id, symbol, direction, entry_price, sl_price, tp_price,
               lot, confidence, entry_tf, htf_tf, ticket, risk_amount,
               data_source, opened_at, status, be_done)
            VALUES
              (:strategy_id,:symbol,:direction,:entry_price,:sl_price,:tp_price,
               :lot,:confidence,:entry_tf,:htf_tf,:ticket,:risk_amount,
               :data_source,:opened_at,:status,:be_done)
        """, trade)


def update_trade_be(ticket: int, be_done: bool):
    with _conn() as c:
        c.execute(
            "UPDATE live_trades SET be_done=? WHERE ticket=? AND status='open'",
            (1 if be_done else 0, ticket)
        )


def close_live_trade(ticket: int, pnl: float):
    with _conn() as c:
        c.execute("""
            UPDATE live_trades
            SET status='closed', pnl=?, closed_at=?
            WHERE ticket=?
        """, (pnl, datetime.now().isoformat(), ticket))


def get_open_trades() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM live_trades WHERE status='open'"
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_live_trades(strategy_id: str, n: int = 20) -> list[dict]:
    """Return the last n closed live trades for a strategy, newest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM live_trades WHERE strategy_id=? AND status='closed' "
            "ORDER BY closed_at DESC LIMIT ?",
            (strategy_id, n)
        ).fetchall()
        return [dict(r) for r in rows]


def get_capital_at_risk() -> float:
    """Sum of risk_amount for open trades where SL not at BE."""
    with _conn() as c:
        row = c.execute("""
            SELECT COALESCE(SUM(risk_amount), 0)
            FROM live_trades
            WHERE status='open' AND be_done=0
        """).fetchone()
        return float(row[0])


# ─────────────────────────────────────────────────────────────────────
# backtest_trades
# ─────────────────────────────────────────────────────────────────────

def save_backtest_trades(strategy_id: str, trades: list[dict]):
    """Save per-trade backtest results for session/timing analysis."""
    with _conn() as c:
        c.execute("DELETE FROM backtest_trades WHERE strategy_id=?", (strategy_id,))
        c.executemany("""
            INSERT INTO backtest_trades
              (strategy_id, entry_time, close_time, session, direction,
               pnl_r, pnl_money, win, hour_utc, day_of_week)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, [
            (strategy_id,
             t.get("entry_time"), t.get("close_time"), t.get("session"),
             t.get("direction"), t.get("pnl_r"), t.get("pnl_money"),
             t.get("win"), t.get("hour_utc"), t.get("day_of_week"))
            for t in trades
        ])


def get_backtest_trades(strategy_id: str) -> list[dict]:
    """Return all per-trade records for a strategy."""
    with _conn() as c:
        rows = c.execute("""
            SELECT entry_time, close_time, session, direction,
                   pnl_r, pnl_money, win, hour_utc, day_of_week
            FROM backtest_trades WHERE strategy_id=?
            ORDER BY entry_time
        """, (strategy_id,)).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# optuna_trials
# ─────────────────────────────────────────────────────────────────────

def save_optuna_trials(symbol: str, tf: int, trials: list[dict]):
    """Save all trial results for one per-TF optimization run."""
    run_date = datetime.now().isoformat()
    with _conn() as c:
        # Remove old trials for this symbol/tf
        c.execute(
            "DELETE FROM optuna_trials WHERE symbol=? AND tf=?",
            (symbol, tf)
        )
        c.executemany("""
            INSERT INTO optuna_trials
              (symbol, tf, trial_number, params_json, sharpe, run_date)
            VALUES (?,?,?,?,?,?)
        """, [
            (symbol, tf, t["number"], json.dumps(t["params"]), t["sharpe"], run_date)
            for t in trials
        ])


def get_optuna_trials(symbol: str, tf: int) -> list[dict]:
    with _conn() as c:
        rows = c.execute("""
            SELECT trial_number, params_json, sharpe
            FROM optuna_trials
            WHERE symbol=? AND tf=?
            ORDER BY sharpe DESC
        """, (symbol, tf)).fetchall()
        return [
            {"number": r["trial_number"],
             "params": json.loads(r["params_json"]),
             "sharpe": r["sharpe"]}
            for r in rows
        ]
