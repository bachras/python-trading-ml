# Quick Start — Tick Data to Live Trade

**Prerequisite:** MT5 open and logged in to demo account. Python venv activated.

---

## Step 1 — Install

```bash
venv\Scripts\activate
pip install MetaTrader5 pandas numpy scikit-learn xgboost deap optuna
pip install pyarrow joblib scipy python-dotenv plotly
```

---

## Step 2 — Configure `.env`

Minimum required changes from the template:

```ini
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Demo
MT5_PATH=C:\Program Files\MetaTrader 5 ...\terminal64.exe

TICK_FILE_US30=F:\trading_ml\tick_data\US30_ticks.csv
SYMBOL_US30=US30          # verify in MT5 Market Watch
ACTIVE_SYMBOLS=US30
```

Risk limits (start conservative, never changed by ML):

```ini
RISK_MODE=fixed
FIXED_RISK_AMOUNT=100     # $100 per trade
RISK_CAP_AMOUNT=400       # max $400 at risk across all open positions
MAX_DRAWDOWN_PCT=35
```

---

## Step 3 — Tick CSV → Parquet (once, ~30–60 min)

Place your Dukascopy export at `tick_data\US30_ticks.csv`  
(EET timezone, columns: `time, ask, bid, ask_volume, bid_volume`)

```bash
python tick_pipeline.py
```

**What it does:** Streams CSV in 500 k-row chunks → cleans ticks → builds 1m/3m/5m/10m/15m/30m/1H OHLCV + spread/delta/VWAP microstructure → saves to `data\`.

**Done when you see:** A data quality table with coverage > 90% and microstructure = YES.

---

## Step 4 — Train (once, ~1–3 hours)

```bash
python train.py
```

**What it does, in order:**

| Stage | Log tag | What to verify |
|-------|---------|----------------|
| Data integrity | `[DATA]` | `duplicate_ts=0`; `missing_sessions < 3%` |
| Feature sanity | `[FEATURES]` | no frozen or exploding features |
| Feature parity | `[PARITY]` | result = PASS (train ↔ live pipelines match) |
| Walk-forward folds | `[FOLD]` | XGB/RF AUC 0.53–0.62 per fold |
| Feature importance | `[IMPORTANCE]` | no single feature dominates (> 0.40) |
| Calibration | `[CALIBRATION]` | brier improves after isotonic fit |
| Probability shape | `[PRED]` | shape = NORMAL (std 0.06–0.12) |
| GA optimization | `[GA]` | best_score 1.0–3.0; n_trades ≥ 100 |
| Execution check | `[EXECUTION]` | avg_spread ≥ 1.5 pts; fill_rate 75–90% |
| Optuna refinement | `[OPTUNA]` | score ≥ GA score |
| Edge test | `[SPA]` | p_value < 0.05 |
| Robustness | `[ROBUST]` | DSR > 0; MC pass > 60% |
| Class balance | `[CLASS]` | positives 15–50%; delta vs raw TP rate < ±3% |
| Final retrain | `[RETRAIN]` | label matches rank-1 params; brier improves |
| DB save | `[BASELINES]` | expected_sharpe / win_rate / trades_per_day saved |

**Done when you see:** `TRAINING COMPLETE — Start live trading: python live.py`

---

## Step 5 — Dry-run live (sanity check, 1 session)

```bash
python live.py --dry
```

Signals are logged but no orders are placed. Watch for:

```
[CONFIG] Risk gates loaded: max_drawdown_pct=35.0 fixed_risk_amt=100.0
[KILLSWITCH] Loaded baselines: expected_sharpe=1.92 expected_trades_per_day=3.4
[SIGNAL] US30/1m | prob_raw=0.643 prob_cal=0.631 | htf=BULL htf_adj=+0.021 | prob_final=0.652 | conf_thresh=0.668 → SKIP
[SIGNAL] US30/1m | prob_raw=0.694 prob_cal=0.681 | htf=BULL htf_adj=+0.024 | conf_thresh=0.668 → PASS | direction=LONG
```

**Red flags in dry-run:**
- `expected_sharpe=0.0` or `expected_trades_per_day=0.0` → DB save failed, re-run training
- All signals SKIP → confidence threshold too high; check `[PRED]` std (collapsed model)
- `[PARITY] FAIL` in training log → do not go live

---

## Step 6 — Verify full pipeline with test trade

```bash
python live.py --test-trade
```

Places one real micro-lot trade and immediately closes it. Verifies: MT5 order placement → DB logging → BE monitoring → close with actual PnL from deal history.

All 6 steps should show `[PASS]`. If any shows `[FAIL]`, fix before going live.

---

## Step 7 — Go live (paper trading first)

```bash
python live.py
```

**Expected live log flow:**

```
[CONFIG] Risk gates loaded ...
[KILLSWITCH] Loaded baselines: expected_sharpe=1.92 ...
Sleeping 47s to next candle close...

[SIGNAL] US30/1m | prob_raw=0.701 prob_cal=0.688 | htf=BULL htf_adj=+0.023 | prob_final=0.711 | conf_thresh=0.668 → PASS | direction=LONG
[TRADE] OPEN  | id=4821 US30 LONG | entry=44821.5 sl=44771.5 tp=44971.5 | sl_pts=50.0 tp_pts=150.0 | risk=$100 | prob=0.711

[TRADE] BE    | id=4821 | price=44871.5 (+50.0 pts = +1R) → SL moved to 44822.5

[TRADE] CLOSE | id=4821 | exit=44971.5 | pnl=+200.00 (+2.00R) | duration=47m | reason=TP
```

---

## Common Commands

```bash
python train.py                   # train (skip TFs already on disk)
python train.py --force           # full retrain from scratch
python train.py --report          # train then open HTML report

python live.py                    # live trading
python live.py --dry              # signals only, no orders
python live.py --test-trade       # full pipeline smoke test
python live.py --signal-check     # print current probabilities vs threshold

python report.py                  # HTML performance report (no retrain)
```

---

## Retraining While Live

```bash
# Terminal 1: live.py running (do NOT stop it)
# Terminal 2:
python train.py --force
# live.py detects new model files each hour and hot-reloads
# open positions are never interrupted
```

---

## Monitoring Checklist (daily)

| Check | Where | Healthy |
|-------|-------|---------|
| Kill switch state | `live.log` → `[KILLSWITCH]` | ACTIVE |
| Signal frequency | `live.log` → `[KILLSWITCH]` freq warning | no warning after day 5 |
| Trade PnL | `live.log` → `[TRADE] CLOSE` | pnl ≠ 0.0 (if all zeros → bug not fixed) |
| Drift | `live.log` → `[DRIFT]` | STABLE ≥ 8/10 features |
| Spread model active | `live.log` → `[EXECUTION]` at training | avg_spread ≥ 1.5 pts |

---

## Key `.env` Knobs After Going Live

```ini
# To run a second strategy alongside the auto-selected best:
EXTRA_STRATEGIES=US30_3m_rank2

# To widen/tighten the execution latency model:
EXEC_DELAY_PROB=0.30        # fraction of fills delayed to next bar
MT5_DEVIATION_PTS=20        # gap > this → fill rejected (like IOC)

# To force a full retrain on next train.py run:
FORCE_RETRAIN=true
```
