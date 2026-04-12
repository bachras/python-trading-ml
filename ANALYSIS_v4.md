# ML Trading System — Implementation Audit & Verification Guide (v4)
**Date:** 2026-04-12
**Analyst:** Independent AI Review — Institutional Desk Standard
**Instrument:** US30 CFD (Dukascopy tick data → MT5 execution)
**Purpose:** Post-implementation audit of all v3 planned changes, gap list, logging requirements, and verification checklist for confirming each technique produced the expected outcome.

---

## 1. IMPLEMENTATION AUDIT

### 1A. Layer 1 — Internal Consistency Fixes

| # | Item | File | Status | Notes |
|---|------|------|--------|-------|
| L1-1 | Spread + slippage in `ga_fitness()` | `phase2_adaptive_engine.py` | ✅ DONE | `spread_cost = spread_arr[i]/2`, `slip_cost = atr[i]*0.1` |
| L1-2 | 75 label columns (5×5×3 grid) | `phase2_adaptive_engine.py` | ✅ DONE | `TB_SL_GRID`, `TB_TP_GRID`, `TB_BE_GRID` loops; `target_sl{sl}_tp{tp}_be{be}` naming |
| L1-3 | `_select_label_col()` with be_r | `phase2_adaptive_engine.py` | ✅ DONE | be_r=3 → be=2 mapping; fallback to default label |
| L1-4 | BE labels: be=0 vectorized, be=1/2 per-trade loop | `phase2_adaptive_engine.py` | ✅ DONE | Correct split per Q3 decision |
| L1-5 | Binary labels (1.0=TP hit, 0.0=else) | `phase2_adaptive_engine.py` | ✅ DONE | XGBoost/RF binary classifier |
| L1-6 | Final model retrained on best label after optimization | `train.py` | ✅ DONE | `_select_label_col()` → swap `df["target"]` → `train_ensemble()` |
| L1-7 | HTF nudge applied in `ga_fitness()` | `phase2_adaptive_engine.py` | ✅ DONE | `adjusted_prob` with bear/bull HTF scaling when `htf_w > 0` |
| L1-8 | Daily equity Sharpe in `ga_fitness()` via numpy | `phase2_adaptive_engine.py` | ⚠️ PARTIAL | Uses only bars with non-zero daily PnL (`mask = daily_pnl != 0`), not full calendar series — see Gap G1 |
| L1-9 | Minimum trade count penalty in `ga_fitness()` | `phase2_adaptive_engine.py` | ✅ DONE | `MIN_CREDIBLE=100`; `fitness = sharpe * min(n_trades/100, 1.0)`; hard floor at n<10 → −999 |
| L1-10 | `run_genetic_algo()` returns `best_score` | `phase2_adaptive_engine.py` | ✅ DONE | `best["best_score"] = float(hof[0].fitness.values[0])` |
| L1-11 | `GA_BOUNDS` narrowed to sl_atr [1.0,4.0], tp_mult [0.5,6.0] | `phase2_adaptive_engine.py` | ✅ DONE | |
| L1-12 | Daily trade cap removed | `backtest_engine.py`, `live.py` | ✅ DONE | Absent from both files |
| L1-13 | Consecutive loss cooldown removed | `backtest_engine.py`, `live.py` | ✅ DONE | Absent from both files |
| L1-14 | Vol-sizing removed from backtest; toggled in live | `backtest_engine.py`, `live.py` | ✅ DONE | `risk = fixed_amt` always in backtest; `VOL_SIZE_ENABLED` env toggle in live (default false) |
| L1-15 | Spread filter (2× median) removed from backtest | `backtest_engine.py` | ⚠️ PARTIAL | Old 2×spread filter gone; replaced by ATR > 3×median filter — see Gap G2 |
| L1-16 | Fixed risk = 100.0 in backtest | `backtest_engine.py` | ✅ DONE | `risk = fixed_amt` (not `balance * risk_pct / 100`) |
| L1-17 | `bfill()` removed from feature engineering | `institutional_features.py` | ✅ DONE | Replaced with `ffill` + `fillna(0)` |
| L1-18 | Session-level volume profile (groupby date) | `institutional_features.py` | ✅ DONE | Per-calendar-day cumulative intraday segments; matches VWAP pattern |

### 1B. Layer 2 — Edge Reliability Fixes

| # | Item | File | Status | Notes |
|---|------|------|--------|-------|
| L2-1 | Probability calibration (isotonic regression on OOS) | `train.py` | ✅ DONE | `IsotonicRegression` on stacked OOS fold predictions; saved as `calibrator_{key}.pkl` |
| L2-2 | SPA edge validation test after optimization | `train.py` | ⚠️ PARTIAL | `_spa_bootstrap_test()` present; bootstrap on trade PnLs. Not full Hansen SPA — see Gap G3 |
| L2-3 | PSI concept drift detection | `live.py` | ✅ DONE | `_DriftMonitor`; PSI on top features; periodic monitoring |
| L2-4 | Smart kill switch (graduated response) | `live.py` | ⚠️ PARTIAL | `_SmartKillSwitch`: last 20 trades, states warn/reduce/disable. Uses trade-level Sharpe, not annualized vs expected — see Gap G4 |

### 1C. Critical Configuration Gaps (Not Code — Environment)

| # | Item | Status | Fix Required |
|---|------|--------|-------------|
| C1 | Max DD hardcoded to 35% | ❌ NOT SET | `HARD_LIMITS["max_drawdown_pct"]` defaults to **10.0** from env. Must set `MAX_DRAWDOWN_PCT=35` in `.env` file. Code is in place, value is not. |
| C2 | `label_col` not persisted in `best_params` | ⚠️ PARTIAL | Label is re-derived at runtime via `_select_label_col(sl, tp, be_r)` — functionally correct but not explicit in the saved params dict. Low risk but makes audit harder. |
| C3 | Env var name: `VOL_SIZE_ENABLED` vs `VOL_SIZING_ENABLED` | ⚠️ NAMING | Minor mismatch vs analysis docs. Check `.env` uses the actual variable name from `live.py` (`VOL_SIZE_ENABLED`). |

---

## 2. REMAINING GAPS TO FIX

### G1. Sharpe Denominator Mismatch — Trading Days Only vs Full Calendar

**File:** `phase2_adaptive_engine.py` — `ga_fitness()` daily equity section

**Problem:** `ga_fitness()` computes Sharpe only on bars where `daily_pnl != 0` (i.e., days with at least one trade). Days with no trades are excluded from the std calculation. `backtest_engine` uses full calendar daily equity returns including flat days (zero return days). This inflates `ga_fitness()` Sharpe vs the reporting engine because it removes zero-return days from the denominator, artificially lowering variance.

**Example:**
```
ga_fitness:      std([+2R, -1R, +1.5R, +2R])   — 4 trading days only
backtest_engine: std([+2R, -1R, 0, +1.5R, 0, +2R, 0])   — 7 calendar days
```
`backtest_engine` Sharpe will be lower because zero days add to variance estimate. The optimizer therefore favours strategies with fewer, more selective trades (which have fewer zero-return days and thus artificially higher Sharpe in ga_fitness).

**Fix direction:** Include zero-return days in the daily equity series inside `ga_fitness()`. Track the first and last trade date, fill the full calendar range with zeros for no-trade days, then compute std/mean on the complete series.

---

### G2. ATR Volatility Filter is Not the Same as Spread Filter

**File:** `backtest_engine.py`

**Problem:** The v3 plan said to remove the 2×median spread filter because spread cost is already priced into entry. What remains in `backtest_engine.py` is an `ATR > 3×ATR_rolling_median` filter — this blocks entries during abnormally high volatility, which is a different and valid concept (not double-counting spread). However, it was not explicitly planned in v3, so it should be confirmed as intentional:

- **If intentional:** Document it explicitly. High ATR outliers are a valid exclusion (news spikes, opening auction chaos aligns with I4 execution regime awareness).
- **If unintentional:** Assess whether it hurts or helps. High-volatility entries may be exactly when the signal is strongest (breakout regime).

**Action:** Confirm intent, add to `.env` as `ATR_SPIKE_FILTER_MULT=3.0` so it is adjustable/removable.

---

### G3. SPA Test is Bootstrap-Only, Not Full Hansen SPA

**File:** `train.py` — `_spa_bootstrap_test()`

**Problem:** Current implementation bootstraps trade PnLs and checks if the 5th percentile Sharpe > 0. This tests "is Sharpe positive at 95% confidence?" — a useful but weaker test than the formal Hansen (2005) Superior Predictive Ability test, which:
1. Tests against a benchmark (not just zero)
2. Accounts for correlations between strategies in the comparison set
3. Uses the studentised maximum statistic across all tested strategies

For the current pipeline (hundreds of Optuna trials), the bootstrapped Sharpe > 0 test is directionally correct but may pass strategies that a proper SPA would reject.

**For now:** The current test is sufficient for initial deployment. Document this as a known limitation and upgrade to full SPA in Layer 3 if initial live performance is poor.

---

### G4. Kill Switch Uses Raw Trade-Level Sharpe, Not Annualised vs Expected

**File:** `live.py` — `_SmartKillSwitch`

**Problem:** Current implementation computes Sharpe from last 20 trade R-multiples directly. The v3 plan specified rolling annualised Sharpe compared against the backtest expected Sharpe, enabling statements like "live Sharpe < 50% of expected for 2 weeks → disable."

Without the expected baseline comparison:
- A strategy that was always weak cannot be distinguished from one that was strong and degraded
- The `reduce`/`disable` thresholds are absolute, not relative to the strategy's own baseline

**Fix direction:** When `train.py` saves the best strategy, store `expected_sharpe` and `expected_win_rate` in the DB alongside other metrics. `_SmartKillSwitch` reads these at init and uses them as thresholds:
```python
if rolling_sharpe < 0.5 * self.expected_sharpe and n_trades >= 20:
    state = "reduce"
if rolling_win_rate < self.expected_win_rate - 0.10 and n_trades >= 50:
    state = "disable"
```

---

## 3. LOGGING REQUIREMENTS

The following logs must be present to confirm each technique is working correctly during and after a training run.

### 3A. Label Grid (75 columns)

**Where to log:** At end of `engineer_features()` in `phase2_adaptive_engine.py`

**What to log:**
```
[LABELS] Generated 75 label columns. TP hit rates:
  target_sl1.5_tp1_be0 : 62.3%  (base rate, tight barrier)
  target_sl2.0_tp3_be0 : 38.1%
  target_sl3.5_tp5_be0 : 19.2%  (wide barrier, harder)
  target_sl1.5_tp1_be1 : 61.8%  (be=1, tp=1 → BE irrelevant)
  target_sl2.0_tp3_be1 : 35.4%  (be=1 causes some TP-bound trades to exit early via BE SL)
  target_sl2.0_tp3_be2 : 37.1%  (be=2, fewer early exits vs be=1)
```

**What to verify:**
1. TP hit rate decreases as `tp_mult` increases (harder prediction task for wider TP) ✓
2. TP hit rate slightly lower for be=1 vs be=0 for the same sl/tp (dynamic SL stops some TP-bound trades) ✓
3. For tp=1: TP rates for be=0, be=1, be=2 are nearly identical (BE irrelevant at tp=1) ✓
4. No column has 0% or 100% TP rate (would indicate a labeling bug) ✓

**Red flag:** If be=1 TP rate is *higher* than be=0 for the same sl/tp, the BE trigger logic has the direction wrong.

---

### 3B. ga_fitness() — Spread/Slippage Costs Applied

**Where to log:** Inside `ga_fitness()`, once per genome evaluation (or as a summary after full GA run)

**What to log:**
```
[GA] Genome (sl=2.0, tp=3, be=1, conf=0.68): n_trades=187, spread_cost_mean=3.2pts, 
     slip_cost_mean=4.1pts, total_cost_mean=7.3pts, as_pct_of_sl=3.7%
```

**What to verify after full GA run:**
1. Best genome's `sl_atr` should be **wider** than the pre-fix default (1.5) — costs punish tight SLs ✓
2. Best genome's `tp_mult` should be **lower or similar** to pre-fix (optimizer learned costs) ✓
3. `best_score` (Sharpe) should be **lower** than it would be with zero-cost evaluation — if it's the same, costs are not being applied ✓

**Red flag:** If best `sl_atr` is still at the grid minimum (1.5) after many generations AND `tp_mult` is at maximum (6), the spread costs are not influencing the optimizer — check that `spread_arr` is non-zero in the evaluation window.

---

### 3C. HTF Nudge in ga_fitness()

**Where to log:** Summary after GA run

**What to log:**
```
[GA] HTF alignment: best htf_weight=0.42, best htf_tf=60min
     HTF signal distribution: bullish=48.3%, bearish=31.2%, neutral=20.5%
     Signal adjustments: 287 boosted (aligned), 219 suppressed (counter-trend)
```

**What to verify:**
1. `htf_weight` in best params is non-zero (if 0, HTF adds no value — valid finding but worth noting) ✓
2. Number of adjusted signals is >0 (confirms HTF nudge is executing) ✓
3. If `htf_weight` is consistently near 0 across multiple runs, HTF alignment provides no signal on this instrument — update analysis accordingly

---

### 3D. Daily Equity Sharpe vs Per-Trade Sharpe

**Where to log:** In `ga_fitness()` during best-genome evaluation, log both Sharpe variants

**What to log:**
```
[GA] Best genome Sharpe comparison:
     Daily equity Sharpe (√252 annualised): 1.34
     Per-trade R-multiple Sharpe (√trades/yr): 1.87
     Backtest engine Sharpe (same params):    1.29
     ga_fitness vs backtest delta: +3.9% (target: <10%)
```

**What to verify:**
1. `ga_fitness()` Sharpe and `backtest_engine` Sharpe should agree within 10% ✓
2. If they diverge >10%, log which components differ (trade count, daily equity construction, spread model)
3. The per-trade Sharpe should be higher than the daily equity Sharpe for any strategy (confirms the daily version is the conservative, correct one)

---

### 3E. Label Selection per Genome

**Where to log:** In `_select_label_col()` — log when called for the best genome

**What to log:**
```
[LABEL_SEL] genome (sl=2.5, tp=3, be=1) → closest grid: (sl=2.5, tp=3, be=1) → column: target_sl2.5_tp3_be1
[LABEL_SEL] genome (sl=1.8, tp=2.7, be=0) → closest grid: (sl=2.0, tp=3, be=0) → column: target_sl2.0_tp3_be0 
             (distance: 0.36 — genome is 0.2 from sl=2.0, 0.3 from tp=3)
```

**What to verify:**
1. Closest column is always a valid key in the DataFrame ✓
2. Distance from genome to nearest grid point is logged — flag if distance > 1.0 (genome is far from any label) ✓
3. After final optimization, log the label column selected for the production model retrain

**Red flag:** If every genome maps to the same column (e.g., always `target_sl1.5_tp2_be0`), the GA is not exploring the search space — check `GA_BOUNDS` are correctly applied.

---

### 3F. Final Model Retrain on Best Label

**Where to log:** In `train.py` after per-TF optimization

**What to log:**
```
[RETRAIN] TF=5m best params: sl=2.5, tp=3, be=1 → label=target_sl2.5_tp3_be1
          TP rate in training data: 36.8%  (class balance for final model)
          Model retrained. OOS accuracy (on best label): 58.4%
          Calibrator fitted. Brier score before calibration: 0.231, after: 0.218
          Model saved: models/xgb_US30_5m.pkl, calibrator: models/calibrator_xgb_US30_5m.pkl
```

**What to verify:**
1. The label column name in the log matches `target_sl{best_sl}_tp{best_tp}_be{best_be}` exactly ✓
2. OOS accuracy on the correct label is reported (not on default `target_sl1.5_tp2_be0`) ✓
3. Brier score improves after calibration (lower = better calibrated) ✓
4. Calibrator file is saved alongside the model ✓

**Red flag:** If `label_col == "target_sl1.5_tp2_be0"` after every optimization run, the optimizer is always landing on the default grid point — either the search space is too constrained or spread costs are dominating too aggressively.

---

### 3G. Probability Calibration

**Where to log:** In `train.py` after fitting `IsotonicRegression`

**What to log:**
```
[CALIBRATION] Model: xgb_US30_5m
  Calibration curve (predicted_prob → actual_win_rate):
    [0.50–0.55]: predicted=0.52, actual=0.47  (delta: -0.05)
    [0.55–0.60]: predicted=0.57, actual=0.54  (delta: -0.03)
    [0.60–0.65]: predicted=0.62, actual=0.61  (delta: -0.01)
    [0.65–0.70]: predicted=0.67, actual=0.65  (delta: -0.02)
    [0.70–0.75]: predicted=0.72, actual=0.71  (delta: -0.01)
    [0.75+]:     predicted=0.79, actual=0.74  (delta: -0.05)
  Brier score: before=0.231, after=0.218
  Max bin delta before calibration: 0.08  (acceptable if <0.10)
```

**What to verify:**
1. After calibration, `predicted` and `actual` in each bin should be within 0.03 of each other ✓
2. Brier score should decrease after calibration ✓
3. Before calibration, XGBoost typically over-estimates probabilities — the `delta` should be negative in most bins (model is overconfident). If deltas are positive, the model is underconfident (unusual but possible with high regularization)
4. The calibrator must be applied at inference time — verify `get_signal()` calls `calibrator.predict()` before threshold comparison

**Red flag:** If Brier score increases after calibration, isotonic regression was fit on too few samples (fewer than 500 OOS predictions is unreliable). Check the number of OOS fold rows used.

---

### 3H. SPA Bootstrap Test

**Where to log:** In `train.py` after `_spa_bootstrap_test()`

**What to log:**
```
[SPA] TF=5m best strategy bootstrap edge validation:
      n_trades=234, mean_R=0.31, Sharpe=1.42
      Bootstrap (n=10000 resamples): 5th pct Sharpe=0.67, p-value=0.003
      PASS: edge distinguishable from zero at p<0.05
      
[SPA] TF=1m best strategy:
      n_trades=89, mean_R=0.18, Sharpe=0.94
      Bootstrap: 5th pct Sharpe=0.11, p-value=0.082
      WARN: edge not statistically significant at p<0.05 — marginal (n_trades may be too low)
```

**What to verify:**
1. Any strategy with p > 0.05 should be flagged and not deployed without explicit override ✓
2. p-value should be lower (more significant) for strategies with more trades ✓
3. If all strategies fail the SPA test after multiple optimization runs, either the data is insufficient or the edge is not real — do not deploy, investigate

---

### 3I. PSI Drift Detection (Live)

**Where to log:** In `live.py` — `_DriftMonitor` — each monitoring cycle (e.g., daily)

**What to log:**
```
[DRIFT] 2026-04-12 20:30 UTC — PSI check on top-10 features:
  vwap_dist_atr:     PSI=0.04  (OK)
  quote_cvd_norm:    PSI=0.09  (OK)
  atr_ratio:         PSI=0.18  (OK)
  hv20_pct:          PSI=0.23  (WARNING — approaching threshold 0.25)
  regime_sin:        PSI=0.11  (OK)
  ... (top 10 total)
  Max PSI: 0.23 (hv20_pct) — state: WARN
```

**What to verify after first week of live trading:**
1. PSI values should be low (< 0.10) for most features in a stable regime ✓
2. If PSI spikes immediately after going live, the live data distribution differs from training data — investigate feature computation in `pipeline.py` vs `tick_pipeline.py`
3. Log which feature triggered each state change (WARN/REDUCE/DISABLE)
4. Verify `_DriftMonitor` is actually being called on schedule (check timestamps in logs)

**Red flag:** If PSI is always near zero for every feature, the monitoring window may be too short or the baseline distribution may be recomputed on live data (defeating the purpose). Verify baseline is from training data only.

---

### 3J. Smart Kill Switch (Live)

**Where to log:** In `live.py` — `_SmartKillSwitch` — after each trade closes

**What to log:**
```
[KILLSWITCH] After trade #23 close:
  Rolling window: last 20 trades
  Win rate: 12/20 = 60.0%  (expected from backtest: 54.2%)
  Rolling Sharpe (trade-level): 1.18
  State: NORMAL (was: NORMAL)
  
[KILLSWITCH] After trade #41 close:
  Win rate: 8/20 = 40.0%  (expected: 54.2% — delta: -14.2pp, threshold: -10pp)
  Rolling Sharpe: 0.43
  State: REDUCE  (was: NORMAL) ← STATE CHANGE — risk halved to $50/trade
  Alert sent.
```

**What to verify:**
1. State transitions (NORMAL → WARN → REDUCE → DISABLE) are logged with timestamps ✓
2. The expected baselines (win rate, Sharpe) must be loaded from the strategy DB, not hardcoded — verify the DB row is read at init ✓
3. After a state change to REDUCE, confirm `trade_risk` in subsequent trades is halved ✓
4. After DISABLE state, confirm no new orders are placed until manual re-enable

---

### 3K. Max Drawdown = 35% Verification

**Where to verify:** Check `.env` file before any live run

**What to check:**
```bash
# In .env:
MAX_DRAWDOWN_PCT=35
```

**And log at startup in `live.py`:**
```
[CONFIG] Risk gates loaded:
  max_drawdown_pct: 35.0%  ← must be 35, not 10
  fixed_risk_amt:   100.0
  session_gate:     20:30–01:00 London (DST-aware)
  vol_size_enabled: false
```

**Red flag:** If log shows `max_drawdown_pct: 10.0`, the `.env` is missing `MAX_DRAWDOWN_PCT=35`. The circuit breaker will fire at 10% — far too early, killing potentially recoverable drawdowns.

---

### 3L. BE Labeling Correctness Spot-Check

**Where to verify:** One-time check after first feature engineering run

**What to check:**

For a known price path in the training data, manually verify the label is correct. Select a long trade that:
1. Crossed be=1 trigger (entry + sl_dist) — call it bar B1
2. Then crossed the BE SL level (entry + 1pt) — call it bar B2
3. Never reached TP

Expected: `target_sl{X}_tp{Y}_be1[that_trade_idx] = 0.0`, `target_sl{X}_tp{Y}_be0[that_trade_idx] = 0.0`

The R-multiples will differ at simulation time (be=0: −1R, be=1: ~0R), but both labels are 0.0 (not TP hit).

Also find a trade that:
1. Crossed BE trigger at bar B1
2. Reversed slightly (bar B2: low between entry and entry+sl_dist — i.e., dipped below BE SL but above original SL)
3. Then continued to TP at bar B3

Expected: `be=0 label = 1.0` (trade survived, reached TP), `be=1 label = 0.0` (BE exit stopped trade at bar B2 before TP)

If both labels are 1.0, the BE logic has a bug (dynamic SL not being applied).

---

## 4. POST-RUN VERIFICATION CHECKLIST

After the first full training run with all changes, verify the following before paper trading:

### 4A. Optimizer Behaviour Changed as Expected

| Check | Expected result | How to verify |
|-------|----------------|---------------|
| Best `sl_atr` wider than pre-fix | sl_atr ≥ 2.0 (was 1.5 default) | Compare best params to old params file |
| Best `tp_mult` realistic | tp_mult ≤ 4 (not hitting grid max of 6) | Check best params |
| GA and Optuna produce comparable scores | `ga_best_score` vs `opt_best_score` within 20% | Log from comparison in `train.py` |
| `ga_fitness` Sharpe ≈ `backtest_engine` Sharpe | Delta < 10% | Run backtest on best params, compare |
| HTF weight is non-zero | `htf_weight > 0.1` | Check best params |
| `be_r` is non-zero in at least some candidates | Various be values explored | Log from `_select_label_col()` calls |

### 4B. Calibration Worked

| Check | Expected result | How to verify |
|-------|----------------|---------------|
| Brier score improved | After < before | Calibration log |
| Max bin delta after calibration < 0.03 | Tight calibration | Calibration curve log |
| Calibrator file exists on disk | `calibrator_*.pkl` present | Check `models/` directory |
| `get_signal()` uses calibrated probs | Calibrator is loaded and applied | Add log line in `get_signal()` confirming calibrator loaded |

### 4C. Label Grid Sensible

| Check | Expected result | How to verify |
|-------|----------------|---------------|
| TP rate decreases with tp_mult | tp=1 > tp=2 > tp=3 > tp=4 > tp=5 for same sl | Label summary log |
| be=1 TP rate ≤ be=0 for same sl/tp (tp ≥ 2) | Dynamic SL should stop some TP-bound trades | Label summary log |
| tp=1: be=0, be=1, be=2 TP rates nearly identical | BE irrelevant at tp=1 | Label summary log |
| No column has 0% or 100% TP rate | Both classes present | Label summary log |

### 4D. System Consistency

| Check | Expected result | How to verify |
|-------|----------------|---------------|
| `ga_fitness` Sharpe vs `backtest_engine` Sharpe | < 10% difference | Re-run backtest with best params |
| Trade count in backtest ≥ 100 | Statistical credibility | Backtest output |
| Most recent walk-forward fold OOS Sharpe not significantly worse than earlier folds | < 30% degradation | Per-fold log from `train.py` |
| SPA test passes (p < 0.05) | Edge is statistically real | SPA log |
| Max DD in `.env` is 35% | Critical config check | Live startup log |

### 4E. Before First Paper Trade

1. Run `live.py` in `DRY_RUN=true` mode for at least 3 trading sessions
2. Confirm signals fire at expected frequency (compare to backtest signal count per day)
3. Confirm calibrated probability is logged per signal (should be in 0.52–0.85 range; rarely above 0.85)
4. Confirm `_DriftMonitor` check runs on schedule and logs PSI values
5. Confirm `_SmartKillSwitch` initialises with `expected_win_rate` and `expected_sharpe` from DB (not defaults)
6. Confirm no orders are placed in `DRY_RUN` mode (check MT5 terminal — order count = 0)

---

## 5. ADDITIONAL INSTITUTIONAL GAPS (New — Post-Implementation Review)

These items were identified after the Layer 1 + 2 implementation. They are ordered by priority. All are compatible with the fixed $100/trade risk model and the "let ML discover" philosophy.

---

### A1. Live Calibration Tracker — Online Predicted vs Actual (CRITICAL)

**What exists:** Offline calibration (isotonic regression fitted on OOS training data, stored as `calibrator_*.pkl`). PSI monitoring on features.

**What is missing:** The most important early-warning signal — whether the model's probability outputs still predict actual outcomes in live trading. PSI says "features look similar to training." It does not say "model predictions are still correct." These are independent: features can be stable while the relationship between features and outcomes shifts.

**What to add — `live.py` + `db.py`:**

For every live trade, store `predicted_probability` (the calibrated probability at signal time) and `actual_outcome` (1=TP hit, 0=SL/BE exit) when the trade closes. Then compute a rolling calibration curve:

```
[LIVE_CAL] Calibration check (last 100 closed trades):
  Bin [0.50–0.55]: predicted=0.52, actual=0.48  Δ=−0.04  n=18  (OK)
  Bin [0.55–0.60]: predicted=0.57, actual=0.53  Δ=−0.04  n=24  (OK)
  Bin [0.60–0.65]: predicted=0.62, actual=0.55  Δ=−0.07  n=21  (WARN)
  Bin [0.65–0.70]: predicted=0.67, actual=0.49  Δ=−0.18  n=19  (ALERT — model overconfident)
  Bin [0.70+]:     predicted=0.74, actual=0.41  Δ=−0.33  n=18  (CRITICAL — edge may be gone)
  Max bin delta: 0.33 → ALERT
```

**Thresholds (suggested starting points — adjust after 3 months of data):**
- Δ > 0.05 in any bin → log WARNING
- Δ > 0.10 in any bin → reduce to $50/trade (same as kill switch reduce state)
- Δ > 0.15 in any bin for 2 consecutive checks → disable trading, trigger retrain

**Why this is the most important gap:**
PSI may show "features stable" while the model's predictive relationship has shifted. This calibration tracker is the **earliest possible detection** of edge degradation — earlier than win rate drift, earlier than Sharpe degradation, earlier than PSI. A model that is systematically overconfident (predicts 0.70 but wins 40%) is losing money on every high-confidence trade. This is detectable within 50–80 trades, before P&L damage becomes severe.

**Minimum viable version:** One new column `predicted_probability REAL` in `live_trades` table. One function that groups closed trades into probability bins and reports Δ. Run daily.

**Files:** `db.py` (schema), `live.py` (log at signal, update at close), `report.py` (calibration curve chart).

---

### A2. Fitness Function Tail Risk Penalty — CVaR Floor

**What exists:** Sharpe + trade count penalty in `ga_fitness()`. ER in the gating system.

**What is missing:** Two strategies can have identical Sharpe and ER but very different loss distribution shapes. The optimizer will pick the one with better Sharpe, which may be the one with the hidden crash risk.

**Concrete example at US30 intraday level:**

| Strategy | Sharpe | ER | Worst 5% of trades | Max single loss |
|---|---|---|---|---|
| A | 1.4 | 4.2 | −3.8R average | −5.1R (gap event) |
| B | 1.3 | 4.0 | −1.0R average | −1.0R (capped) |

Strategy A wins under current fitness. Strategy B is clearly safer and worth the 0.1 Sharpe haircut.

**What to add to `ga_fitness()` in `phase2_adaptive_engine.py`:**

```python
# After computing trade R-multiples (r_arr)
sorted_r = np.sort(r_arr)
cvar_pct = int(len(r_arr) * 0.05)
if cvar_pct >= 3:
    cvar_95 = sorted_r[:cvar_pct].mean()   # mean of worst 5% — negative number
    # Penalty: CVaR worse than -2R gets a fitness haircut
    cvar_penalty = max(0.0, (-cvar_95 - 2.0) / 2.0)   # 0 if CVaR ≥ -2R, grows beyond
    fitness = sharpe * trade_penalty * max(0.5, 1.0 - cvar_penalty)
```

Also compute and log skewness of returns — prefer positive skew (occasional large wins, frequent small losses) over negative skew (frequent small wins, occasional large losses):

```python
skewness = scipy.stats.skew(r_arr)
# Optional: slight reward for positive skew
fitness *= (1.0 + 0.05 * np.clip(skewness, -1.0, 1.0))
```

**Log during GA:**
```
[GA] Best genome fitness breakdown:
  Sharpe=1.42, trade_penalty=1.0, cvar_95=-1.3R (OK), skewness=+0.21
  CVaR penalty=0.00, skew bonus=+0.01 → final fitness=1.43
```

**Alignment with philosophy:** This does not hardcode a behavioral constraint. The optimizer is still free to choose any parameters. It just receives a gradient signal: "setups with fat left tails are penalised proportionally." The optimizer will discover whether tight SL + wide TP (which creates bad left tails) is worth the theoretical Sharpe gain.

---

### A3. Parameter Stability Memory Across Training Runs

**What exists:** Best params saved to `params/US30_params.json` and DB after each training run.

**What is missing:** Comparison across runs. If `sl_atr` jumps from 2.5 to 1.5 to 3.5 across three consecutive training runs, the edge is unstable — either the optimizer is fitting noise (overfitting) or the market regime has genuinely shifted.

**What to add to `train.py` and `db.py`:**

After saving best params, load the previous run's params and compute deltas:
```
[PARAM_STABILITY] Training run comparison:
  Previous: sl_atr=2.5, tp_mult=3, confidence=0.65, be_r=1, htf_weight=0.42
  Current:  sl_atr=2.0, tp_mult=3, confidence=0.67, be_r=1, htf_weight=0.39
  Deltas:   Δsl=−0.5, Δtp=0, Δconf=+0.02, Δbe=0, Δhtf=−0.03
  Stability: STABLE (max delta < 1.0 on any parameter)
  
[PARAM_STABILITY] WARNING:
  Previous: sl_atr=2.0, tp_mult=3, confidence=0.65
  Current:  sl_atr=3.5, tp_mult=1, confidence=0.82
  Deltas:   Δsl=+1.5, Δtp=−2, Δconf=+0.17  ← LARGE JUMP
  Stability: UNSTABLE — parameters changed significantly. 
  Action: do not auto-deploy. Compare OOS metrics across runs before switching.
```

**Stability thresholds (suggested):**
- `|Δsl_atr| > 1.0` → WARN
- `|Δtp_mult| > 2` → WARN
- `|Δconfidence| > 0.10` → WARN
- All three WARNing simultaneously → require manual review before deployment

**DB change:** Add a `param_history` table or store all training runs in `strategy_params` with a timestamp. The comparison is then a simple SELECT of the two most recent rows per symbol/TF.

**Alignment with philosophy:** This is not constraining the optimizer — it is monitoring the optimizer's output over time. The optimizer retains full freedom to change parameters. The stability check is an alarm, not a constraint.

---

### A4. Time-Based Kill Switch Window (Not Just Last N Trades)

**What exists:** `_SmartKillSwitch` monitors last 20 trades. Graduated response: warn → reduce → disable.

**What is missing:** Time dimension. Last 20 trades could span 2 days of high-frequency entries or 2 weeks of selective entries. A strategy degrading slowly over 10 trading days may never trigger the 20-trade window if it fires only once per day.

**What to add — rolling time window alongside trade window:**

```python
# Check both: last N trades AND last N days
rolling_20_trade_sharpe = compute_sharpe(last_20_trades)
rolling_10_day_sharpe   = compute_sharpe(trades_in_last_10_calendar_days)

# Use the more conservative (lower) signal
effective_sharpe = min(rolling_20_trade_sharpe, rolling_10_day_sharpe)
```

**Log:**
```
[KILLSWITCH] Dual-window check:
  Last 20 trades (spans 8 days):  Sharpe=0.92, WR=55%  → NORMAL
  Last 10 calendar days:          Sharpe=0.61, WR=48%  → WARN  ← time window worse
  Effective state: WARN (worst of two windows)
```

**The kill switch state is driven by whichever window is worse.** This catches both:
- Fast degradation (20-trade window triggers)
- Slow degradation (time window triggers when trade frequency is low)

**Minimum viable version:** Add a `_last_N_days_trades(n=10)` query to `db.py` that returns closed trades from the last N calendar days. Already partially supported by the live trade log.

---

### A5. Signal Rejection Tracking ("Missed Trades" Log)

**What exists:** Executed trades logged. Rejected signals are silently discarded.

**What is missing:** Tracking which signals were generated but filtered out — and whether those rejected signals would have been profitable.

**What to add to `live.py`:**

For every signal generated by `get_signal()`, log whether it was accepted or rejected, and why:

```
[SIGNAL] 2026-04-12 14:32 UTC — US30 5m
  Raw probability: 0.71 → calibrated: 0.68
  Direction: LONG
  Confidence threshold: 0.65 → PASS
  HTF alignment: BEARISH, htf_weight=0.42 → adjusted_prob=0.58 → FAIL (below threshold)
  Decision: REJECTED (HTF filter)
  SL=2.0 ATR, TP=3.0×, entry=40112
  
[SIGNAL] 2026-04-12 15:18 UTC — US30 5m  
  Calibrated prob: 0.71 → PASS
  HTF: BULLISH → adjusted: 0.74 → PASS
  Risk gate: PASS (DD=4.2%, daily_loss=1.1%)
  Decision: ACCEPTED — order placed
```

Store rejected signals in a `rejected_signals` DB table with: timestamp, direction, calibrated_prob, rejection_reason, and — when the equivalent price path is known — the hypothetical outcome (did price hit TP or SL?).

**Why this matters:** If rejected signals (especially HTF-filtered ones) consistently hit TP at a higher rate than accepted ones, the HTF filter is removing edge, not protecting it. This is detectable within 4–8 weeks of live data and provides direct evidence for adjusting `htf_weight` in the next retrain.

**Also detects over-filtering drift:** If rejection rate climbs from 30% to 70% of signals over 2 months without parameter changes, feature distribution has shifted (PSI-detectable) or the confidence threshold is now too high for current market conditions.

---

### A6. Edge Concentration Risk — PnL Distribution by Context

**What exists:** Per-trade P&L logged. Strategy-level metrics (Sharpe, ER, win rate).

**What is missing:** Decomposing where the P&L actually comes from. A system that makes 80% of its profit in 2 specific hours of the day is fragile — those hours may change character without warning.

**What to add to `report.py` and `backtest_engine.py`:**

PnL breakdown tables computed from backtest trades (and later from live trades):

```
[EDGE_CONCENTRATION] PnL by hour (US30 5m, 6yr backtest):
  Hour (UTC) | Trades | Win% | Mean R | Total R | % of Total Profit
  13:00–14:00|   312  | 61%  |  +0.42 |  +131.0 |  28.4%  ← highest concentration
  14:00–15:00|   287  | 58%  |  +0.31 |   +88.9 |  19.3%
  15:00–16:00|   241  | 52%  |  +0.18 |   +43.4 |   9.4%
  16:00–17:00|   198  | 49%  |  +0.04 |    +7.9 |   1.7%
  ...
  Top 2 hours: 47.7% of total profit

[EDGE_CONCENTRATION] PnL by regime:
  Regime 2 (bull trend):    58% WR, mean R=+0.38 → 61% of profit
  Regime 0 (low vol range): 52% WR, mean R=+0.12 → 18% of profit
  Regime 3 (bear trend):    44% WR, mean R=−0.11 → −14% of profit (net drag)
```

**Concentration risk thresholds:**
- Top 2 hours > 60% of profit → FRAGILE (flag in report)
- Single regime > 70% of profit → FRAGILE
- Any regime shows negative total R with > 30 trades → consider regime filter

**This is not a constraint** — the optimizer is free to pick any parameters. This analysis tells you whether the discovered edge is robust (spread across hours/regimes) or concentrated (dependent on specific conditions). A concentrated edge is not necessarily bad, but it must be monitored more closely.

**Add to `report.py`:** Two new chart types — hour heatmap of P&L (not just trade count) and regime P&L bar chart. Data already exists in `backtest_trades` table.

---

### A7. HTF Nudge — Document Linearity Limitation, Plan ML Replacement

**What exists:** `prob *= (1 ± htf_weight * 0.3)` — linear, symmetric, constant-factor nudge applied when HTF is bullish or bearish.

**Known limitation:** The HTF effect is nonlinear and regime-dependent in practice. In a trending market, a bearish HTF may suppress a long signal by 30% → correct. In a ranging market, the same bearish HTF signal may have zero predictive value → incorrectly suppresses 30% of signal. The linear formula cannot distinguish these cases.

**Current status:** Acceptable for initial deployment. The optimizer can set `htf_weight=0` to effectively disable HTF if the linear nudge hurts performance, so the system is self-correcting.

**Future replacement (Layer 3):** Replace the linear nudge with a learnable interaction. Add HTF state as a feature to the primary model (e.g., `htf_composite_score`, `htf_bullish_flag`, `htf_bearish_flag`). Let XGBoost/RF learn the non-linear interaction between signal probability and HTF context. Remove the explicit `htf_weight` parameter from the optimizer's search space — the model learns it implicitly.

**Log current HTF nudge behaviour for evidence:**
```
[HTF_NUDGE] Statistics for this training period:
  Signals with HTF bullish: 412 → avg prob boost: +0.028 → accepted rate: 71%
  Signals with HTF bearish: 389 → avg prob reduction: −0.031 → accepted rate: 43%
  Signals with HTF neutral: 201 → no adjustment → accepted rate: 58%
  HTF filter rejection rate: 34% of all signals
```

If HTF-filtered signals (bearish HTF on long) have a significantly worse actual win rate than HTF-aligned ones, the nudge is working. If they have a similar win rate, HTF adds no value and `htf_weight` will naturally converge to 0 after the next retrain.

---

### A8. Ensemble Diversity Check — XGBoost vs Random Forest Agreement

**What exists:** `prob = (xgb_prob + rf_prob) / 2` — equal-weight average of two models.

**What is missing:** Measurement of whether the two models are actually providing diverse perspectives or essentially duplicating each other.

**What to add to `train.py` after fitting both models on the same fold:**

```python
# On OOS fold predictions
xgb_preds = xgb_model.predict_proba(X_oos)[:,1]
rf_preds  = rf_model.predict_proba(X_oos)[:,1]
corr = np.corrcoef(xgb_preds, rf_preds)[0,1]
disagreement_rate = np.mean(np.abs(xgb_preds - rf_preds) > 0.10)
```

**Log:**
```
[ENSEMBLE] OOS fold diversity check:
  XGB vs RF prediction correlation: 0.87  (target: <0.90)
  Disagreement rate (|diff|>0.10):  23%   (target: >15%)
  Assessment: ADEQUATE diversity — ensemble adds value
  
[ENSEMBLE] OOS fold diversity check:
  XGB vs RF prediction correlation: 0.96  (target: <0.90)
  Disagreement rate (|diff|>0.10):   7%   (target: >15%)
  Assessment: LOW diversity — ensemble behaving as single model
  Consider: different feature subsets, different max_depth, or add a third diverse model
```

**If correlation > 0.92 consistently:**
- Reduce RF `max_depth` or `max_features` to force more randomization
- Or use XGB alone (averaging two near-identical models adds noise without signal)
- Or add a third model with a deliberately different architecture (e.g., LightGBM with different hyperparams)

**Why this matters for the fixed $100/trade model:** With fixed risk and a fixed signal threshold, signal quality is the only variable that determines P&L. If the ensemble adds no diversity, it's just adding noise around a single model's output without improving accuracy.

---

### A9. Trade Quality as Soft Optimizer Penalty (Mean R per Trade)

**What exists:** Trade count penalty (`fitness *= min(n_trades/100, 1.0)`). Sharpe (mean/std of returns).

**What is missing:** Explicit penalty for low average R per trade. The optimizer could select a strategy that fires 200 trades at mean R = +0.05 (barely above spread cost) — high trade count, decent Sharpe due to consistency, but extremely fragile to any increase in spread or slippage.

**What to add to `ga_fitness()` in `phase2_adaptive_engine.py`:**

```python
mean_r = r_arr.mean()
# Soft penalty: mean R below 0.15 per trade gets increasingly penalised
# 0.15R on $100 risk = $15 per trade average — above typical spread+slippage ($7–10)
if mean_r < 0.15:
    quality_penalty = mean_r / 0.15   # scales from 0 (at mean_R=0) to 1.0 (at mean_R=0.15)
    fitness *= quality_penalty
```

**Log:**
```
[GA] Best genome quality check:
  n_trades=187, mean_R=+0.31, quality_penalty=1.0 (above threshold)
  
[GA] Candidate genome quality check:
  n_trades=312, mean_R=+0.08, quality_penalty=0.53 (below threshold — low mean R)
  Adjusted fitness: 1.24 → 0.66 (penalised)
```

**The threshold 0.15R is not arbitrary:** With typical US30 CFD spread + slippage = ~$7–10 per $100 at risk, a trade with mean R = 0.10 earns $10 on average and costs $8–10 in execution — net near zero. Only when mean R significantly exceeds execution costs does the strategy have genuine net profit. `0.15R = $15 avg net` provides a margin above execution cost while remaining achievable.

**This is aligned with fixed $100 risk:** Every mean_R threshold is directly in dollars at $100 risk. `mean_R = 0.15` means the average trade earns $15. Clear, measurable, independent of account size.

---

## 6. LAYER 3 — DEFERRED (After Live is Stable)

These items remain planned but are not in scope until live trading is validated:

| # | Item | File | Priority | Notes |
|---|------|------|----------|-------|
| L3-1 | Meta-labeling (two-model architecture) | `phase2_adaptive_engine.py`, `train.py` | High | Primary model → meta-model → trade/reject. Biggest remaining edge boost. |
| L3-2 | Block bootstrap Monte Carlo | `backtest_engine.py` | Medium | Preserve autocorrelation; replace IID bootstrap in `run_monte_carlo()` |
| L3-3 | Full Hansen SPA test | `train.py` | Medium | Upgrade `_spa_bootstrap_test()` to full studentised maximum statistic |
| L3-4 | Execution regime features | `institutional_features.py` | Medium | `execution_quality_score = f(spread_pct, tick_velocity_z, quote_update_freq)` as ML feature |
| L3-5 | Dynamic capital allocation | `live.py` | Medium | Tighten confidence threshold when rolling live Sharpe < 70% of expected |
| L3-6 | Regime-conditional models | `train.py`, `phase2_adaptive_engine.py` | Low | Separate XGB/RF per regime cluster |
| L3-7 | Kill switch vs expected baseline (G4 fix) | `live.py`, `db.py` | High | Store `expected_sharpe` + `expected_win_rate` in DB; kill switch reads at init |
| L3-8 | HTF nudge replaced by learnable ML feature (A7) | `institutional_features.py`, `phase2_adaptive_engine.py` | Medium | Add HTF state as model feature; remove linear nudge |

---

## 6. KNOWN REMAINING INCONSISTENCY (LOW RISK)

**HTF alignment in backtest_engine:**

`get_signal()` in live trading applies the HTF probability nudge (±30% × htf_weight). `backtest_engine.run_backtest()` uses raw ensemble probabilities without the HTF nudge. This means:
- Backtest ER/Sharpe is computed without HTF filtering
- Live signals will be filtered by HTF alignment
- Live win rate may be **higher** than backtest (if HTF correctly filters bad signals) or **lower** (if HTF alignment has residual noise)

This is a known gap but lower priority than the items above because:
1. The optimizer now includes HTF nudge in `ga_fitness()` — so the selected params account for HTF
2. `backtest_engine` is used for reporting, not parameter selection
3. The direction of the bias is likely positive (HTF should improve signal quality)

Fix: apply the same HTF nudge in `backtest_engine` as in `get_signal()` to get accurate reported metrics. Defer to after paper trading confirms the direction.

---

## 7. ESTIMATED STATUS SUMMARY

| Layer | Items | Done | Partial | Remaining |
|-------|-------|------|---------|-----------|
| Layer 1 (internal consistency) | 18 | 16 | 2 | 0 |
| Layer 2 (edge reliability) | 4 | 2 | 2 | 0 |
| Config/env | 3 | 0 | 1 | 2 |
| Additional gaps (Section 5) | 9 | 0 | 0 | 9 |
| Layer 3 (deferred) | 8 | 0 | 0 | 8 |

### Priority order for Section 5 items:

| # | Item | Priority | Blocking live? | Effort |
|---|------|----------|----------------|--------|
| A1 | Live calibration tracker (predicted vs actual) | **Critical** | Not blocking but highest signal value | Small — 1 DB column, 1 function |
| A3 | Parameter stability memory across runs | **High** | No | Small — DB query + delta log |
| A2 | CVaR / tail risk fitness penalty | **High** | No | Small — 10 lines in `ga_fitness()` |
| A4 | Time-based kill switch window | **High** | No | Small — add `last_N_days_trades()` query |
| A5 | Signal rejection tracking | **Medium** | No | Medium — new DB table + log in signal path |
| A6 | Edge concentration analysis | **Medium** | No | Medium — new report charts, backtest decomposition |
| A8 | Ensemble diversity check | **Medium** | No | Small — add to `train.py` after fold training |
| A9 | Trade quality soft penalty | **Medium** | No | Small — 5 lines in `ga_fitness()` |
| A7 | HTF nudge ML replacement | **Low (deferred)** | No | Large — moved to Layer 3 |

### Immediate action items before first training run:
1. **Set `MAX_DRAWDOWN_PCT=35` in `.env`** — hard blocker, circuit breaker fires at 10% otherwise
2. **Confirm env var name:** `VOL_SIZE_ENABLED` (not `VOL_SIZING_ENABLED`) in `live.py`
3. **Add startup config log** in `live.py` to print all risk gate values at init
4. **Fix Sharpe denominator** in `ga_fitness()` (G1 — include zero-return calendar days)
5. **Add `expected_sharpe` + `expected_win_rate`** to DB strategy save (enables G4 kill switch baseline)
6. **Add CVaR penalty** to `ga_fitness()` (A2 — 10 lines, highest impact per line of code)
7. **Add ensemble diversity check** to `train.py` (A8 — no cost, pure diagnostic)
8. **Add trade quality penalty** to `ga_fitness()` (A9 — 5 lines, guards against spread-eating strategies)

### Before going live (paper trade → live capital):
9. **Implement live calibration tracker** (A1) — most important ongoing monitoring signal
10. **Implement signal rejection log** (A5) — detects over-filtering after 4–8 weeks of data
11. **Add parameter stability check** (A3) — required before any model update in live

**The system is in a deployable state for paper trading after items 1–5 above and one training run. Items 6–8 are small improvements that take under an hour each and materially improve the optimizer's output quality. Items 9–11 are live-phase additions that grow in value as trade history accumulates.**
