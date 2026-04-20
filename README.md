# ML Trading System — US30 CFD on MT5

**Instrument:** US30 CFD | **Broker:** IC Markets SC (demo) | **Data:** Dukascopy tick CSV to MT5 execution
**Last updated:** 2026-04-19 | **Analysis version:** v7

---

## System Overview

A four-module institutional-grade ML system for intraday CFD trading. Training is fully offline; the live loop loads saved artifacts and never blocks on heavy computation.

```
tick_pipeline.py  -> data/           Dukascopy CSV -> multi-TF OHLCV Parquet + microstructure
institutional_features.py            VWAP, volume profile, order flow delta, regime, liquidity
train.py          -> models/ params/ Walk-forward XGB+RF -> calibration -> GA+Optuna -> robustness
live.py           <- models/ params/ Signal generation -> risk gates -> MT5 orders -> kill switch
```

---

## File Map

| File | Role |
|------|------|
| `tick_pipeline.py` | Streams Dukascopy CSV in 500k-row chunks; builds 1m/3m/5m/10m/15m/30m/1H OHLCV with spread, delta, tick velocity, VWAP |
| `institutional_features.py` | Session VWAP + anchored VWAP + bands/slope; volume profile (POC/VAH/VAL/HVN/LVN); order flow delta + CVD; swing levels, equal H/L, stop-hunt detection; market regime |
| `phase2_adaptive_engine.py` | XGBoost + RF ensemble; Triple Barrier label grid; GA (DEAP) + Optuna; get_signal(); ga_fitness() with full execution realism; _check_leakage() |
| `pipeline.py` | Shared data load (load_symbol_data, engineer_full_features); used by both train.py and live.py to guarantee feature parity |
| `train.py` | Training entry point — heavy computation, run once |
| `live.py` | Live trading loop — lightweight, runs continuously |
| `backtest_engine.py` | Vectorised backtest with synthetic spread, lognormal noise, execution latency |
| `db.py` | SQLite: strategies, equity curves, monthly PnL, backtest trades, live trades |
| `report.py` | Interactive Plotly HTML report from DB |
| `.env` | All configuration lives here — never edit .py files |

---

## Architecture

### Labels — 75-Column Triple Barrier Grid

Generated at training time for every bar:

- `sl` in [1.5, 2.0, 2.5, 3.0, 3.5] x ATR14 (5 SL distances)
- `tp` in [1, 2, 3, 4, 5] x sl_dist (5 TP multiples)
- `be` in [0, 1, 2] x R (3 breakeven levels — 0=off, 1=move SL to entry at +1R, 2=at +2R)

Column name format: `target_sl{sl}_tp{tp}_be{be}` — 5x5x3 = 75 columns total.

GA selects the best (sl, tp, be) combination. The production model retrains on that exact label column, fixing the calibration mismatch from training on the default be=0 label.

### ML Models

| Model | Role |
|-------|------|
| XGBoost | Primary classifier; `seed=GLOBAL_SEED` |
| Random Forest | Ensemble diversity; `random_state=GLOBAL_SEED` |
| Isotonic calibrator | Maps raw ensemble probability to actual win rate; fitted on stacked OOS folds |

### Walk-Forward Validation

3-fold anchored expanding window (year boundaries):

- Fold 1: train up to year[-3], OOS = year[-2]
- Fold 2: train up to year[-2], OOS = year[-1]
- Fold 3: train up to year[-1], OOS = current year

10-bar purge gap at each boundary prevents look-ahead leakage. Falls back to 70/30 split if fewer than 3 years of data.

### GA + Optuna Parameter Search

Genome: `[entry_tf_idx, htf_idx, sl_atr, tp_mult, confidence, htf_weight, be_r_idx]`

1. **Global GA** — DEAP, 150 population x 80 generations = 12,000 evaluations across all TFs simultaneously
2. **Global Optuna** — 500 trials (TPE sampler, seed=GLOBAL_SEED), seeded with GA best genome
3. **Per-TF Optuna** — 150 trials per TF (PER_TF_TRIALS in .env); saves top-5 param sets per TF to DB

GA fitness function includes: synthetic spread, lognormal noise, slippage, fill probability, 30% next-bar latency (EXEC_DELAY_PROB), CVaR-95 penalty, MIN_CREDIBLE=100 trades floor, ATR spike filter.

### Execution Realism Model

Applied inside ga_fitness() and backtest_engine.py:

| Component | Detail |
|-----------|--------|
| Synthetic spread | (SPREAD_BASE_PTS + ATR14 x SPREAD_ATR_COEFF) x session_mult x lognormal(sigma=0.15). Active when raw spread_mean < 0.1 pts (Dukascopy near-zero spreads). |
| Session open | First SPREAD_OPEN_BARS bars get x SPREAD_OPEN_MULT |
| Fill probability | Based on sl_dist / spread ratio |
| Execution latency | EXEC_DELAY_PROB (30%) of fills delayed to next-bar open. Gap > MT5_DEVIATION_PTS = fill rejected (same as IOC rejection in live) |
| Per-genome RNG | Each genome gets its own default_rng(seed) — fill noise is deterministic per genome, not per run |

### Robustness Gates (per strategy, after backtest)

1. SPA bootstrap test — p < 0.05 rejects H0: no edge
2. Haircut Sharpe — deflated for number of Optuna trials (multiple testing correction)
3. Monte Carlo — 1,000 trade-order shuffles; mc_pass_rate > 60%
4. Sensitivity score — nudge each param by small step; score = fraction of variants within 20% of base Sharpe

### Live Trading Flow

```
wake at candle close
  -> refresh bars from MT5
  -> drift PSI check on top-10 features  [DRIFT]
  -> kill switch check                   [KILLSWITCH]
  -> get_signal()
       XGB + RF predict -> prob_raw
       isotonic calibrator -> prob_cal
       HTF alignment nudge -> prob_final (htf_adj = delta)
       compare vs confidence_threshold -> [SIGNAL] PASS or SKIP
  -> session gate (20:30-01:00 London)
  -> risk gate (daily loss, drawdown, position count cap)
  -> risk cap check (total capital at risk across all positions)
  -> place_order()                        [TRADE] OPEN
  -> _monitor_open_positions()
       BE trigger detected               [TRADE] BE
       position disappeared from MT5     [TRADE] CLOSE (PnL from history_deals_get)
  -> hot-reload check (new model files from train.py?)
```

### Kill Switch

Strategy-specific, loaded from DB at startup (`expected_sharpe`, `expected_win_rate`, `expected_trades_per_day`). Evaluates last 20 live trades:

- WARN: rolling Sharpe or win rate < 50% of expected
- REDUCE: < 40% of expected — risk halved on new trades
- DISABLE: < 30% of expected — no new entries until manually re-enabled
- Frequency check: warns if actual trades/day < 30% of expected after at least 5 elapsed days

### Drift Monitor

PSI (Population Stability Index) on top-10 XGB features vs training histogram:

- PSI < 0.10 — STABLE
- 0.10 to 0.25 — WARN
- > 0.25 — ALERT (risk halved, independent of kill switch)

---

## Configuration (.env)

Never edit .py files for configuration. Every parameter is in `.env`.

### Paths

```ini
BASE_DIR=F:\trading_ml
# Move the whole system to another drive by changing BASE_DIR alone.
# DATA_DIR / MODEL_DIR / LOG_DIR / PARAMS_DIR default to BASE_DIR subfolders.
```

### Tick Data

```ini
TICK_FILE_US30=F:\trading_ml\tick_data\US30_ticks.csv
TICK_SYMBOLS=US30
TICK_CHUNK_SIZE=500000    # rows per chunk; reduce to 200000 for 8 GB RAM
```

### MT5 Connection

```ini
MT5_LOGIN=52613516
MT5_PASSWORD=...
MT5_SERVER=ICMarketsSC-Demo
MT5_PATH=C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe
```

### Broker Symbol Names

```ini
SYMBOL_US30=US30      # verify exact name in MT5 Market Watch
SYMBOL_DE40=DE40
SYMBOL_EURUSD=EURUSD
ACTIVE_SYMBOLS=US30   # comma-separated canonical keys; empty = all
```

### Training Control

```ini
FORCE_RETRAIN=false       # true = always retrain from scratch; false = resume from disk
TOP_N_STRATEGIES=5        # top param sets saved per TF to DB
PER_TF_TRIALS=150         # Optuna trials per TF in per-TF optimisation
BACKTEST_START_DATE=2020-01-02
ER_MULTIPLIER=1.25        # efficiency ratio bonus for low drawdown strategies
GLOBAL_SEED=42            # reproducibility: XGB, RF, Optuna TPE, numpy, random
```

### Risk Limits (never changed by ML)

```ini
MAX_DAILY_LOSS_PCT=10.0
MAX_DRAWDOWN_PCT=35
MAX_OPEN_POSITIONS=10
RISK_PER_TRADE_PCT=1.0

RISK_MODE=fixed           # fixed = exact dollar amount; percent = % of balance
FIXED_RISK_AMOUNT=100     # dollars per trade when RISK_MODE=fixed
RISK_CAP_PCT=2.0          # max total capital at risk as % of balance
RISK_CAP_AMOUNT=400       # max total capital at risk in dollars (fixed mode)
```

### Execution Realism Model

```ini
ATR_SPIKE_FILTER_MULT=3.0    # bars where ATR > 3x median excluded from GA fitness

SPREAD_BASE_PTS=1.5          # synthetic spread floor in points
SPREAD_ATR_COEFF=0.04        # spread scales with ATR: total += ATR x 0.04
SPREAD_OPEN_MULT=2.0         # session-open spread multiplier
SPREAD_OPEN_BARS=5           # how many bars count as session open

EXEC_DELAY_PROB=0.30         # 30% of fills delayed to next-bar open
MT5_DEVIATION_PTS=20         # gap > 20 pts on delayed fill = missed trade (IOC reject)
```

### Multi-Strategy

```ini
EXTRA_STRATEGIES=            # e.g. US30_3m_rank2,US30_5m_rank1
                             # system auto-activates best by ER; add extras here
```

---

## Setup (First Time)

### 1. Folder structure

```
F:\trading_ml\
  venv\
  .env                        <- your config
  tick_data\
    US30_ticks.csv            <- Dukascopy export (EET timezone)
  *.py files
```

Auto-created on first run: `data\`, `models\`, `params\`, `logs\`, `trading.db`

### 2. Install dependencies

```bash
venv\Scripts\activate
pip install MetaTrader5 pandas numpy scikit-learn xgboost
pip install deap optuna pyarrow joblib scipy python-dotenv plotly
```

### 3. Configure MT5

- Open MT5 and log in to demo account
- Tools > Options > Expert Advisors > tick **Allow algorithmic trading**
- Right-click Market Watch > Show All > find your exact symbol name
- Update `SYMBOL_US30=` in `.env` if it differs from `US30`

### 4. Build Parquet from tick CSV (one time, ~30-60 min)

```bash
python tick_pipeline.py
```

Streams CSV in 500k-row chunks, validates every tick (price bounds, spread, volume), builds OHLCV bars for all TFs with microstructure, saves to `data\`.

Done when you see a data quality table with coverage > 90% and microstructure = YES on every TF.

### 5. Train (Terminal 1, ~1-3 hours)

```bash
python train.py              # skips TFs already on disk
python train.py --force      # full retrain from scratch
python train.py --report     # train then open HTML report
python train.py --symbol US30
```

Done when you see: `TRAINING COMPLETE — Start live trading: python live.py`

### 6. Start live trading (Terminal 2)

```bash
python live.py --dry         # signals logged, no orders placed (recommended first)
python live.py --test-trade  # place one micro-lot order and close it immediately
python live.py               # full live trading
```

`live.py` checks for updated model files every hour and hot-reloads automatically. Open positions are never interrupted by model reloads.

---

## Training Log Tags — Quick Reference

| Tag | When | What to verify |
|-----|------|----------------|
| `[SEED]` | Startup | Value matches GLOBAL_SEED in .env |
| `[DATA]` | After parquet load | duplicate_ts=0 (hard stop); missing_sessions < 3%; thin_sessions < 10 |
| `[PARITY]` | After feature engineering | max_diff < 1e-4; result = PASS — FAIL means do not deploy |
| `[LEAKAGE]` | Inside engineer_features | Only swing features in known look-ahead list; 0 unexpected leaks |
| `[LABELS]` | End of engineer_features | 75 columns; TP rates fall with tp_mult; be=1 rate <= be=0 rate |
| `[REGIME]` | After fold construction | drift < 0.30 per fold pair; ATR CV < 0.25 |
| `[FOLD]` | Per fold | XGB/RF AUC 0.53-0.62; AUC stable across folds; train_bars expanding |
| `[IMPORTANCE]` | Per fold + cross-fold | No single feature > 0.40 importance; cross-fold Jaccard > 0.60 |
| `[CALIBRATION]` | After isotonic fit | brier_after < brier_before; improvement 0.005-0.025 |
| `[PRED]` | After calibration | std 0.06-0.12 = NORMAL; std < 0.04 = COLLAPSED (no signal) |
| `[CAL-STABILITY]` | After calibration | All 3 ATR tertiles improve; high_vol has highest Brier |
| `[ENSEMBLE]` | Per fold | XGB-RF correlation < 0.90; disagree_rate > 12% |
| `[GA]` | After GA | best_score 1.0-3.0; sl_atr >= 2.0; n_trades >= 100; conf 0.55-0.75 |
| `[EXECUTION]` | After GA, rank-1 genome | avg_spread >= 1.5 pts; fill_rate 75-90%; delay_rate approx 30% |
| `[OPTUNA]` | After Optuna | Score >= GA score (Optuna refines GA seed) |
| `[SPA]` | After optimization | p_value < 0.05 = real edge confirmed |
| `[ROBUST]` | After SPA | DSR > 0; MC pass_rate > 60%; sensitivity_score >= 50 |
| `[STABILITY]` | Confidence sweep +-0.03 | SMOOTH shape; max drop < 15% across sweep |
| `[CLASS]` | Before retrain | Positives 15-50%; delta vs raw TP rate < +-3% |
| `[RETRAIN]` | Final model train | Label matches rank-1 params; Brier improves |
| `[BASELINES]` | After DB save | expected_sharpe, win_rate, trades_per_day all non-zero |

## Live Log Tags — Quick Reference

| Tag | What |
|-----|------|
| `[CONFIG]` | Risk gates loaded at startup |
| `[KILLSWITCH]` | Baselines loaded; state changes ACTIVE to WARN to REDUCE to DISABLE |
| `[SIGNAL]` | Every signal: prob_raw, prob_cal, htf_adj, prob_final, PASS or SKIP with conf_thresh |
| `[TRADE] OPEN` | Order placed: entry, sl, tp, sl_pts, tp_pts, risk, prob |
| `[TRADE] BE` | Break-even triggered: price moved to entry |
| `[TRADE] CLOSE` | Position closed: exit, actual PnL from MT5 deal history, R-multiple, duration, reason (TP/SL) |
| `[DRIFT]` | PSI check on top-10 features vs training distribution |

---

## Retraining While Live

```bash
# Terminal 2: live.py running — do NOT stop it
# Terminal 1:
python train.py --force
# live.py detects new model files each hour and hot-reloads automatically
# Open positions are never interrupted
```

---

## Database (trading.db)

| Table | Contents |
|-------|----------|
| `strategies` | One row per strategy: all params, Sharpe, ER, win rate, MC results, sensitivity, SPA p-value, expected baselines, is_active flag |
| `equity_curves` | Daily equity point per strategy |
| `monthly_pnl` | Monthly P&L per strategy |
| `backtest_trades` | Every backtest trade (entry/exit/pnl/label) |
| `live_trades` | Every live order: ticket, entry, SL, TP, PnL, BE flag, risk amount |
| `optuna_trials` | All Optuna trial results per symbol/TF |

---

## Known Approximations (flagged, not bugs)

| Item | Detail |
|------|--------|
| Swing look-ahead | SWING_LOOKBACK=10 bars; flagged as [LEAKAGE] KNOWN at training |
| Session volume profile | Full-day POC/VAH/VAL assigned to all intraday bars (causal approximation) |
| SPA test | Bootstrap Sharpe > 0, not full Hansen statistic (deferred to v8) |
| Synthetic spread | Active only when raw spread_mean < 0.1 pts; real broker spread used otherwise |
| Close reason detection | Heuristic from deal history: pnl > 0 = TP, pnl < 0 = SL |

---

## "Our London Time" — Time Reference Standard

The US and UK change their clocks on **different dates** each spring and autumn, creating two mismatch windows (~3 weeks in spring, ~1 week in autumn) where the London–NY gap is 4 h instead of the normal 5 h. During these windows US market events (NFP, FOMC, CPI) appear 1 h earlier on the raw London wall clock.

**"Our London Time"** = London wall clock + **+1 h bump** applied when:
- The mismatch is active: `(London UTC offset) − (NY UTC offset) == 4 h`, AND
- The London wall hour is between **12 and 20 inclusive**

### Where it is used

| Location | Uses "Our London Time" | Reason |
|----------|----------------------|--------|
| `live.py _is_session_blocked()` | Yes — `_our_london_time()` | Gate always fires at same time relative to NYSE close |
| Future event filter (NFP, FOMC, holidays) | Yes — call `_our_london_time()` | Event times will be stored in "Our London Time"; comparison must use the same reference |

### Where it is NOT used (intentionally)

| Feature | Uses plain London wall clock | Reason |
|---------|------------------------------|--------|
| `is_london`, `uk_hour`, `hour_sin/cos` | Yes — `tz_convert("Europe/London")` | ML features must be consistent between training history and live; changing the clock reference would break parity |
| `is_ny`, `is_overlap`, `is_us_open` | NY timezone directly | Computed from `tz_convert("America/New_York")` — always correct regardless of mismatch |

### Adding future event times

When adding high-impact events or holidays, store all times in "Our London Time":
- NFP always at **13:30** (not 12:30 during mismatch weeks)
- FOMC always at **19:00** or **20:00** as applicable
- In `live.py`, compare stored event times against `_our_london_time(datetime.now(tz=timezone.utc))`

### Mismatch windows

| Window | Start | End | Duration |
|--------|-------|-----|----------|
| Spring | 2nd Sunday of March (US springs forward) | Last Sunday of March (UK springs forward) | ~3 weeks |
| Autumn | Last Sunday of October (UK falls back) | 1st Sunday of November (US falls back) | ~1 week |

---

## Hardcoded Market Physics (never overridden by ML)

- **Session gate:** 20:30 to 01:00 in "Our London Time" — no new entries outside this window
- **Max drawdown:** 35% of account (account survival limit)
- **Fixed risk:** $100 per trade (ensures ER computation is accurate across all strategies)
