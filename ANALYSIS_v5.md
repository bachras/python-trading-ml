# ML Trading System — Final Pre-Training Audit (v5)
**Date:** 2026-04-12
**Instrument:** US30 CFD (Dukascopy tick data → MT5 execution)
**Purpose:** Current implementation status against all v4 planned items, plus six new pre-training requirements (R5–R6 critical, S1–S3 strongly recommended) identified in post-v4 review. This is the go/no-go checklist before the first full training run.

**GO / NO-GO Summary:**
| Tier | Items | Action |
|------|-------|--------|
| 🔴 BLOCKING | R1–R6 | Must fix before training |
| 🟠 STRONGLY RECOMMENDED | S1–S3 (Section 4) | Fix before paper trading |
| 🟡 DEFERRED | P1–P3 (paper), L3-1–L3-9 | After live capital stable |

---

## 1. COMPLETE IMPLEMENTATION STATUS

### Layer 1 — Internal Consistency

| # | Item | Status |
|---|------|--------|
| L1-1 | Spread + slippage in `ga_fitness()` | ✅ |
| L1-2 | 75 label columns (5×5×3 grid), `target_sl{sl}_tp{tp}_be{be}` naming | ✅ |
| L1-3 | `_select_label_col()` with be_r; be_r=3→be=2 mapping + fallback | ✅ |
| L1-4 | be=0 vectorized, be=1/2 per-trade Python loop | ✅ |
| L1-5 | Binary labels (1.0=TP hit, 0.0=else) | ✅ |
| L1-6 | Final production model retrained on best-matching label after optimization | ✅ |
| L1-7 | HTF nudge applied in `ga_fitness()` (`adjusted_prob` with bear/bull scaling) | ✅ |
| L1-8 | Daily equity Sharpe in `ga_fitness()` via numpy zero-padding | ✅ Fixed: `np.zeros(n_days_span)` + `np.add.at`; first→last trade date range |
| L1-9 | Minimum trade count penalty (`MIN_CREDIBLE=100`; hard floor n<10→−999) | ✅ |
| L1-10 | `run_genetic_algo()` returns `best_score` | ✅ |
| L1-11 | `GA_BOUNDS` narrowed: sl_atr [1.0,4.0], tp_mult [0.5,6.0] | ✅ |
| L1-12 | Daily trade cap removed from `backtest_engine` + `live.py` | ✅ |
| L1-13 | Consecutive loss cooldown removed from `backtest_engine` + `live.py` | ✅ |
| L1-14 | Vol-sizing removed from backtest; `VOL_SIZE_ENABLED=false` toggle in live | ✅ |
| L1-15 | 2×spread filter removed; ATR spike filter configurable via `ATR_SPIKE_FILTER_MULT` env | ✅ Code done |
| L1-16 | Fixed `risk = fixed_amt` (not `balance * risk_pct / 100`) in backtest | ✅ |
| L1-17 | `bfill()` removed; replaced with `ffill` + `fillna(0)` | ✅ |
| L1-18 | Session-level volume profile (groupby date, not rolling window) | ✅ |

### Layer 2 — Edge Reliability

| # | Item | Status |
|---|------|--------|
| L2-1 | Probability calibration — `IsotonicRegression` on stacked OOS fold predictions | ✅ |
| L2-2 | SPA bootstrap test after optimization (`_spa_bootstrap_test()`) | ✅ Partial — bootstrap Sharpe>0, not full Hansen SPA (G3, deferred) |
| L2-3 | PSI concept drift detection (`_DriftMonitor`) | ✅ |
| L2-4 | Smart kill switch — loads `expected_sharpe` + `expected_win_rate` from DB; graduated warn/reduce/disable | ✅ Partial — `expected_trades_per_day` missing (see R1 below) |

### Section 5 Immediate Items (A2, A8, A9)

| # | Item | Status |
|---|------|--------|
| A2 | CVaR tail risk penalty in `ga_fitness()` | ✅ `sorted_r`, `cvar_95`, `cvar_penalty` multiplied into fitness |
| A8 | Ensemble diversity check in `train.py` (XGB vs RF OOS correlation) | ✅ `[ENSEMBLE]` log with `np.corrcoef` |
| A9 | Trade quality penalty — `mean_r < 0.15` progressive penalty in `ga_fitness()` | ✅ |

### Configuration

| # | Item | Status |
|---|------|--------|
| C1 | `MAX_DRAWDOWN_PCT=35` in `.env` | ✅ Confirmed at `.env` line 132 |
| C2 | `ATR_SPIKE_FILTER_MULT=3.0` in `.env` | ❌ Missing — code reads from env with default 3.0, but not in `.env` |
| C3 | `VOL_SIZE_ENABLED` effective value | ✅ Not in `.env` → defaults to `false` in `live.py` — correct |

### Logging

| # | Item | Status |
|---|------|--------|
| LOG-A | Label TP rate summary at end of `engineer_features()` | ❌ No `[LABELS]` log present |
| LOG-B | Final retrain: label col used + Brier before/after | ⚠️ `[RETRAIN]` logs label col + single Brier; no before/after pair for final retrain path |
| LOG-C | Kill switch state-change log `[KILLSWITCH]` on transitions | ✅ `live.py` logs every `prev_state → new_state` change |
| LOG-D | Startup `[CONFIG]` log (max DD, fixed risk, session gate) | ✅ `live.py` at startup |

---

## 2. REMAINING ITEMS — MUST FIX BEFORE FIRST TRAINING RUN

Six items are not yet complete. R1–R4 are small (< 30 lines each). R5–R6 are new critical additions identified after the v4 audit.

### R1. `expected_trades_per_day` Missing from Full Pipeline

**Status:** `expected_sharpe` and `expected_win_rate` are stored in DB and loaded by `_SmartKillSwitch`. The `expected_trades_per_day` field is absent from all three components needed.

**Why it matters:** The time-based kill switch window (A4 from v4) requires `expected_trades_per_day` to calibrate "last 10 calendar days" correctly. Without it, the time-window check cannot compare against a strategy-specific baseline. Also needed to detect signal frequency drops (e.g., model stops firing due to silent miscalibration).

**Fix — three small changes:**

1. **`db.py`** — Add column to `CREATE TABLE strategy_params` and migration:
```sql
expected_trades_per_day REAL DEFAULT 0.0,
```

2. **`train.py`** — Compute and store after backtest:
```python
n_trading_days = (bt_result["equity_curve"].index[-1] - bt_result["equity_curve"].index[0]).days * (252/365)
params_to_save["expected_trades_per_day"] = round(len(trades) / max(n_trading_days, 1), 2)
```

3. **`live.py` `_SmartKillSwitch.__init__()`** — Load and use:
```python
self.expected_trades_per_day = strategy.get("expected_trades_per_day", 2.0)
# In time-window check:
if n_days_elapsed >= 5 and n_live_trades < 0.3 * self.expected_trades_per_day * n_days_elapsed:
    log.warning("[KILLSWITCH] Signal frequency below 30% of expected — model may have stopped firing")
```

---

### R2. LOG-A — Label TP Rate Summary Missing

**Status:** `engineer_features()` generates all 75 label columns but emits no log output about their composition. Without this, you cannot verify the label grid is correct after a training run without manually inspecting the DataFrame.

**Why it matters:** If a BE labeling bug exists (e.g., be=1 TP rate > be=0 for the same sl/tp at tp≥2), it is undetectable without this log. The log also serves as a sanity check that all 75 columns were created and contain valid class distributions.

**Fix — add at end of `engineer_features()` in `phase2_adaptive_engine.py`:**
```python
label_cols = [c for c in d.columns if c.startswith("target_sl")]
log.info(f"[LABELS] Generated {len(label_cols)} label columns. Sample TP rates:")
for col in sorted(label_cols)[:6]:   # log first 6 as sample
    rate = d[col].mean()
    log.info(f"  {col}: {rate:.1%}")
# Flag any degenerate columns
degenerate = [c for c in label_cols if d[c].mean() < 0.05 or d[c].mean() > 0.95]
if degenerate:
    log.warning(f"[LABELS] DEGENERATE columns (near 0% or 100% TP rate): {degenerate}")
# Verify be=1 TP rate ≤ be=0 for same sl/tp at tp≥2
for sl in [1.5, 2.0, 2.5, 3.0, 3.5]:
    for tp in [2, 3, 4, 5]:
        c0 = f"target_sl{sl}_tp{tp}_be0"
        c1 = f"target_sl{sl}_tp{tp}_be1"
        if c0 in d.columns and c1 in d.columns:
            if d[c1].mean() > d[c0].mean() + 0.02:
                log.warning(f"[LABELS] BE BUG? {c1} ({d[c1].mean():.1%}) > {c0} ({d[c0].mean():.1%})")
log.info(f"[LABELS] All label columns verified.")
```

---

### R3. LOG-B — Final Retrain Brier Before/After Missing

**Status:** `[RETRAIN]` block in `train.py` logs the label column used and a single Brier score. It does not log Brier before calibration, making it impossible to confirm calibration improved on the final production model (as opposed to just the walk-forward OOS folds where the before/after pair is logged).

**Why it matters:** The final model is what gets deployed. If calibration failed silently on the retrain (e.g., too few samples in the retrain fold, or the best label column has a very different class balance), it's undetectable without the before/after comparison.

**Fix — update the `[RETRAIN]` logging block in `train.py`:**
```python
# After training final model on best label:
raw_probs  = final_model.predict_proba(X_retrain_oos)[:,1]
brier_before = brier_score_loss(y_retrain_oos, raw_probs)
# After fitting calibrator:
cal_probs   = calibrator.predict(raw_probs)
brier_after  = brier_score_loss(y_retrain_oos, cal_probs)
log.info(f"[RETRAIN] label={best_label_col}, n_train={len(X_train)}, "
         f"n_oos={len(X_retrain_oos)}, "
         f"brier_before={brier_before:.4f}, brier_after={brier_after:.4f}, "
         f"improvement={brier_before - brier_after:.4f}")
if brier_after >= brier_before:
    log.warning("[RETRAIN] Calibration did NOT improve Brier on final retrain — "
                "check OOS sample size and class balance")
```

---

### R4. `ATR_SPIKE_FILTER_MULT` Not in `.env`

**Status:** `backtest_engine.py` reads `ATR_SPIKE_FILTER_MULT` from env with a hardcoded default of `3.0`. The `.env` file does not contain this key. The behaviour is correct (default 3.0 applies), but the filter is invisible in the config file.

**Why it matters:** Silent parameters that affect trade filtering should be visible in `.env` so they can be reviewed and adjusted without editing code. If high-ATR entries are worth investigating (breakout regime), this parameter must be findable.

**Fix — add one line to `.env`:**
```
# ── BACKTEST FILTERS ────────────────────────────────────────────
# Skip entry signals where current ATR > this multiple of rolling ATR median.
# Set to 99.0 to effectively disable. Market-physics filter — not a behavioral cap.
ATR_SPIKE_FILTER_MULT=3.0
```

---

### R5. Train ↔ Live Feature Parity Check — CRITICAL

**Status:** Not implemented. This is the most common cause of ML trading systems failing silently in production.

**The problem:** `institutional_features.py` computes features for training. `pipeline.py` + `tick_pipeline.py` compute features for live signals. These are two separate code paths. Even tiny differences — VWAP computed with slightly different window alignment, NaN handling that differs between paths, rolling window off-by-one — cause silent failure:

```
Backtest Sharpe: 1.4
Live Sharpe:     0.3   ← no error, no exception, just "bad performance"
```

PSI drift detection (already implemented) only fires *after* live features diverge from training data distribution. It does not detect that the live computation *produces different values from the same input* as the training computation. These are two different failure modes.

**Fix — add a `[PARITY]` check to `train.py` (one-time, runs at the end of feature engineering):**

```python
def _check_feature_parity(df_train: pd.DataFrame, n_sample: int = 500) -> None:
    """
    Take last N rows of training data, recompute features via the live pipeline path,
    and assert they match the training pipeline output within floating-point tolerance.
    """
    sample_idx = df_train.index[-n_sample:]
    # Features already computed by training pipeline
    train_feats = df_train.loc[sample_idx, get_feature_cols(df_train)].values

    # Recompute via live pipeline path (pipeline.py entry point)
    raw_sample  = load_raw_bars_for_index(sample_idx)   # raw OHLCV without features
    live_feats  = compute_live_features(raw_sample).loc[sample_idx[-200:],
                                                        get_feature_cols(df_train)].values
    # Use last 200 rows (avoid warmup difference at start of sample)
    diff = np.abs(train_feats[-200:] - live_feats)
    max_diff     = float(diff.max())
    mean_diff    = float(diff.mean())
    worst_col_idx = int(diff.max(axis=0).argmax())
    worst_col    = get_feature_cols(df_train)[worst_col_idx]

    log.info(f"[PARITY] Feature parity check: max_diff={max_diff:.8f}, "
             f"mean_diff={mean_diff:.8f}, worst_feature={worst_col}")

    if max_diff > 1e-4:
        log.error(f"[PARITY] FAIL — training and live pipelines produce different values "
                  f"for the same input. Worst: {worst_col} diff={max_diff:.6f}. "
                  f"DO NOT DEPLOY until resolved.")
        raise RuntimeError("Feature parity check failed — train/live pipeline mismatch")

    log.info("[PARITY] PASS — training and live features match within tolerance ✅")
```

**Expected output when correct:**
```
[PARITY] Feature parity check: max_diff=0.00000000, mean_diff=0.00000000, worst_feature=vwap_dist_atr
[PARITY] PASS — training and live features match within tolerance ✅
```

**If it fails:** The log will name the specific feature(s) causing the mismatch. Fix the divergence in the relevant pipeline function before proceeding. Common causes: session boundary handling in VWAP, timezone conversion difference, rolling window `min_periods` mismatch.

**Blocking:** Yes. Do not train without this check passing. A model trained on features that live cannot reproduce exactly will degrade regardless of how well it performs in backtest.

---

### R6. Data Leakage Guard — CRITICAL

**Status:** Not implemented as an explicit assertion.

**What is currently relied upon:** Walk-forward with `PURGE_BARS=10` eliminates the most obvious train/test leakage. `bfill()` was removed (fixed in L1-17). But there is no *assertion* that features are causal — that no feature at bar `t` uses information from bar `t+1` or later.

**Where leakage can silently enter:**
- VWAP using full session total (which includes future bars) instead of running cumulative up to current bar
- Volume profile `groupby(date)` computing the full-day POC then assigning it to all bars in that day — the morning bars "know" where the afternoon volume will concentrate
- Rolling statistics with `closed='right'` vs `closed='left'` producing subtle look-ahead
- Label generation touching future bars (already correct via triple barrier, but worth asserting)

**What leakage does:**
```
Backtest: AMAZING (model appears to predict perfectly)
Live:     DEAD     (model was predicting the present, not the future)
```

**Fix — add a `[LEAKAGE]` temporal integrity check in `engineer_features()` in `phase2_adaptive_engine.py`:**

The key insight: if you shuffle the *future* portion of the data and recompute features, the features at bar `t` should **not change** if they are truly causal (because they only look backward). If they do change, the feature uses future information.

```python
def _check_leakage(df: pd.DataFrame, n_check: int = 200) -> None:
    """
    Shuffle future rows and verify that features for past bars do not change.
    A causal feature at bar t must depend only on data at bars <= t.
    """
    if len(df) < n_check * 2:
        return

    pivot = len(df) - n_check
    df_test = df.copy()
    # Shuffle the last n_check rows (future relative to pivot)
    future_idx = df_test.index[pivot:]
    df_test.loc[future_idx] = df_test.loc[future_idx].sample(frac=1).values

    # Recompute features on the shuffled dataframe
    df_reshuffled = add_institutional_features(df_test)

    # Features at bars BEFORE the pivot must be identical
    feat_cols = get_feature_cols(df)
    original  = df.loc[df.index[pivot-50:pivot], feat_cols].values
    reshuffled = df_reshuffled.loc[df.index[pivot-50:pivot], feat_cols].values

    diff = np.abs(original - reshuffled)
    if diff.max() > 1e-8:
        bad_cols = [feat_cols[i] for i in np.where(diff.max(axis=0) > 1e-8)[0]]
        log.error(f"[LEAKAGE] DETECTED in features: {bad_cols}. "
                  f"These features change when future data is shuffled — forward-looking!")
        raise RuntimeError(f"Data leakage detected in features: {bad_cols}")

    log.info("[LEAKAGE] Temporal integrity check PASSED — no forward-looking features detected ✅")
```

**Note on volume profile:** The per-session groupby approach (already implemented in L1-18) is potentially leaky for intraday bars — bars at 09:45 will have a POC computed from the full day's volume. This is a known design trade-off: true real-time VP requires bar-by-bar computation. **The leakage check will flag this** if present. If flagged, either accept it as a known approximation (with reduced confidence in VP features) or switch to cumulative intraday VP.

**Blocking:** Yes. A leaky feature makes backtest results meaningless. The check takes < 5s.

---

## 3. DEFERRED — PAPER TRADE PHASE (Items 13–15 from v4)

These require live trade data to be useful. Implement when paper trading begins.

| # | Item | Trigger | Notes |
|---|------|---------|-------|
| P1 | Live calibration tracker (A1) | After 50+ closed paper trades | Store `predicted_probability` per trade in `live_trades` table; daily calibration curve by probability bin; Δ > 0.10 per bin → reduce sizing |
| P2 | Signal rejection log (A5) | From day 1 of paper trading | Log every generated signal with rejection reason; `rejected_signals` DB table; monitor rejection rate over time |
| P3 | Parameter stability check (A3) | Before first live model update | Compare `sl_atr`, `tp_mult`, `confidence`, `be_r` deltas across consecutive training runs; UNSTABLE flag if any delta exceeds thresholds |

---

## 4. STRONGLY RECOMMENDED — NOT BLOCKING BUT HIGH VALUE

These three items do not block the training run or paper trading. Implement after R1–R6 are done, before or during paper trading.

### S1. Execution Fill Realism in `ga_fitness()` — Missed Fills

**Current state:** Every signal in `ga_fitness()` always results in a filled trade. In reality, fast-moving US30 CFD markets cause order rejections (price moves beyond the 20-deviation tolerance), partial fills, and missed entries during news events.

**Effect on optimizer:** Without missed fill modelling, the optimizer favours tight-entry strategies that rely on precise execution — setups that look great in backtest but fail to fill 5–15% of the time in live.

**Suggested implementation — dynamic fill probability in `ga_fitness()`:**
```python
# Fill probability as a function of spread relative to SL distance
# Wide SL relative to spread → easier to fill; tight SL → fill risk higher
sl_pts    = sl_atr * atr[i]
fill_prob = np.clip(1.0 - (spread_arr[i] / sl_pts) * 0.5, 0.80, 1.0)
if np.random.rand() > fill_prob:
    continue   # missed fill — skip this trade
```

This model: at sl_atr=1.5 (tight) with typical spread = 3pts and ATR = 50pts → sl_pts=75, fill_prob = 1 − (3/75)×0.5 = 0.98. At sl_atr=1.0 (very tight), fill_prob drops toward 0.93. Tighter SLs are slightly penalised for execution uncertainty, consistent with reality.

**Important:** Introduce a fixed random seed *per genome evaluation* (not global) so the fill simulation is deterministic within an evaluation but varies across different genomes — otherwise noise from random misses contaminates the Sharpe comparison.

---

### S2. Confidence Threshold Stability Check

**Current state:** Sensitivity analysis nudges `sl_atr`, `tp_mult`, and `confidence`. But the nudges are ±fixed steps, and the result is only used for the `robust` gate (pass/fail), not reported as a smooth curve.

**The risk:** The optimizer may find a confidence threshold that creates a narrow performance spike rather than a robust plateau:
```
threshold 0.66 → Sharpe 0.81
threshold 0.67 → Sharpe 1.34  ← optimizer picks this
threshold 0.68 → Sharpe 0.79
```
This strategy is fragile — a tiny calibration shift kills it in live. The existing sensitivity `robust` flag helps, but the curve shape is invisible.

**Add to `train.py` after final backtest, log the curve:**
```python
[STABILITY] Confidence threshold sensitivity (best params, sl=2.5, tp=3):
  conf=0.63 → Sharpe=1.12, ER=3.8
  conf=0.64 → Sharpe=1.19, ER=4.1
  conf=0.65 → Sharpe=1.28, ER=4.4
  conf=0.66 → Sharpe=1.34, ER=4.7  ← selected
  conf=0.67 → Sharpe=1.31, ER=4.5
  conf=0.68 → Sharpe=1.24, ER=4.2
  Shape: SMOOTH ✅ (max drop from peak < 15% across ±0.03)
```

**Flat/smooth curve → robust edge. Spike → fragile edge, consider using a slightly lower-Sharpe but smoother threshold.**

This is distinct from the existing sensitivity analysis because it logs the full curve, not just a pass/fail score.

---

### S3. Global Random Seed Control

**Current state:** GA uses DEAP with default seeding. Optuna uses `TPESampler` with auto-generated seed. Walk-forward XGB/RF use default sklearn seeds. Bootstrap MC uses `np.random` with no seed.

**Effect:** Two training runs on identical data can produce different models, different best params, different Sharpe scores. This makes debugging and regression testing unreliable.

**Fix — add to `train.py` startup and log it:**
```python
GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
# Pass to XGBoost:   xgb_params["random_state"] = GLOBAL_SEED
# Pass to RF:        rf_params["random_state"]   = GLOBAL_SEED
# Pass to Optuna:    TPESampler(seed=GLOBAL_SEED)
# Pass to DEAP GA:   random.seed(GLOBAL_SEED) before tools.register(...)
log.info(f"[SEED] Global seed set to {GLOBAL_SEED}")
```

**Add `GLOBAL_SEED=42` to `.env`.**

Note: Fixed seeds for reproducibility. For production, use a different seed per training run (e.g., current date as seed) to avoid seed-specific overfitting, but keep the value logged so any run can be reproduced.

---

## 5. DEFERRED — LAYER 3 (After Live Capital is Stable)

| # | Item | Priority | Notes |
|---|------|----------|-------|
| L3-1 | Meta-labeling (primary → meta-model → trade/reject) | High | Biggest remaining edge boost; ~30–50% DD reduction in published research |
| L3-2 | Block bootstrap Monte Carlo (preserve autocorrelation) | Medium | Replace IID bootstrap in `run_monte_carlo()` |
| L3-3 | Full Hansen SPA test | Medium | Upgrade `_spa_bootstrap_test()` from Sharpe>0 to studentised maximum statistic |
| L3-4 | Execution regime features (`execution_quality_score`) | Medium | Add to `institutional_features.py`; let ML learn when to avoid poor fill conditions |
| L3-5 | Dynamic capital allocation | Medium | Tighten confidence threshold when rolling live Sharpe < 70% of expected |
| L3-6 | Regime-conditional models | Low | Separate XGB/RF per regime; highest compute cost |
| L3-7 | HTF nudge → learnable ML feature | Medium | Add HTF state as model input; remove linear ±30% nudge; let XGBoost learn the interaction |
| L3-8 | Edge concentration analysis in report (A6) | Medium | PnL by hour + regime charts in `report.py`; data already in `backtest_trades` table |
| L3-9 | Time-based kill switch window (A4) | High | Rolling 10-calendar-day window alongside 20-trade window; requires R1 (`expected_trades_per_day`) first |

---

## 6. PRE-TRAINING RUN CHECKLIST

Complete all six remaining items (R1–R6), then verify this checklist before starting training:

### Code changes (R1–R6):
- [ ] R1: Add `expected_trades_per_day` to `db.py` schema + `train.py` + `live.py` kill switch
- [ ] R2: Add `[LABELS]` TP rate log at end of `engineer_features()`
- [ ] R3: Add Brier before/after pair to `[RETRAIN]` block in `train.py`
- [ ] R4: Add `ATR_SPIKE_FILTER_MULT=3.0` to `.env`
- [ ] R5: Implement `[PARITY]` train/live feature parity check in `train.py`
- [ ] R6: Implement `[LEAKAGE]` temporal integrity check in `engineer_features()`

### Environment:
- [ ] `MAX_DRAWDOWN_PCT=35` — confirmed ✅ (`.env` line 132)
- [ ] `RISK_MODE=fixed` — confirmed ✅ (`.env` line 140)
- [ ] `FIXED_RISK_AMOUNT=100` — confirmed ✅ (`.env` line 141)
- [ ] `VOL_SIZE_ENABLED` not set → defaults `false` — confirmed ✅
- [ ] `ATR_SPIKE_FILTER_MULT=3.0` added to `.env` — pending R4

### After training run — spot-check these logs:
- [ ] `[PARITY]` — max feature diff = 0.00000000; "PASS" message present ✅
- [ ] `[LEAKAGE]` — "Temporal integrity check PASSED" message present ✅
- [ ] `[LABELS]` — 75 columns generated; no degenerate columns; be=1 TP rate ≤ be=0 for same sl/tp (tp≥2)
- [ ] `[ENSEMBLE]` — XGB vs RF correlation logged; confirm < 0.92
- [ ] `[CALIBRATION]` — Brier before > after in walk-forward OOS
- [ ] `[RETRAIN]` — label column used + Brier before/after; confirm improvement
- [ ] `[SPA]` — p-value < 0.05 for best strategy
- [ ] `[GA]` best genome — `sl_atr` ≥ 2.0 (wider than pre-fix default 1.5); `cvar_95` logged; `mean_r` > 0.15
- [ ] `[STABILITY]` — confidence threshold sensitivity curve logged; shape is smooth (no isolated spike)
- [ ] `ga_fitness` Sharpe vs `backtest_engine` Sharpe for same params — delta < 10%
- [ ] Walk-forward fold OOS: most recent fold Sharpe not > 30% worse than earlier folds

### Before first paper trade:
- [ ] Run `DRY_RUN=true` for 3 trading sessions; confirm signals fire, no orders placed
- [ ] `[CONFIG]` startup log shows `max_drawdown_pct: 35.0`, `fixed_risk_amt: 100.0`
- [ ] `[KILLSWITCH]` init log confirms `expected_sharpe` and `expected_win_rate` loaded from DB (not defaults)
- [ ] `[DRIFT]` PSI check fires on schedule
- [ ] Begin paper-trade items P1 (live calibration tracker) and P2 (signal rejection log) from day 1

---

## 7. SYSTEM ARCHITECTURE SUMMARY (What the System Does Now)

For reference: this is the complete pipeline as implemented.

```
TRAINING (train.py + phase2_adaptive_engine.py):
  Dukascopy tick data → OHLCV bars (tick_pipeline.py)
  → 73 institutional features (institutional_features.py)
      - Session VWAP, Volume Profile (session-level), Order Flow, Regime
      - bfill() removed → ffill + fillna(0)
  → 75 triple barrier label columns (5×5×3: sl×tp×be)
      - be=0: vectorized numpy scan
      - be=1/2: per-trade Python loop with dynamic SL
      - Binary: 1.0=TP hit, 0.0=else
  → Walk-forward (3 folds, PURGE_BARS=10) + XGB+RF training
      - Default label: target_sl1.5_tp2_be0 for fold search
      - Isotonic calibration on stacked OOS predictions
  → GA (12,000 eval) + Optuna (500 trials) parameter search
      - ga_fitness() includes: spread+slippage, HTF nudge, daily equity Sharpe,
        trade count penalty, CVaR penalty, trade quality penalty
      - Each genome uses _select_label_col() to match its (sl,tp,be) to nearest grid
      - GA returns best_score (fixed)
  → Tiered robustness gating: DSR > 0 → MC pass → Sensitivity ≥ 50 → ER rank
  → SPA bootstrap test (p < 0.05 required)
  → Final production model retrained on best-matching label column
  → Save: model pkl, calibrator pkl, params to DB

LIVE TRADING (live.py):
  MT5 data → pipeline.py → 73 features → get_signal()
  → calibrated probability (isotonic calibrator applied)
  → Session gate (20:30–01:00 London, DST-aware) ← hardcoded market physics
  → Confidence threshold (from optimizer)
  → HTF nudge (adjusted_prob = raw_prob ± htf_weight × 0.3)
  → Risk gates: max DD 35%, daily loss %, max positions, fixed $100/trade
  → MT5 order (TRADE_ACTION_DEAL, IOC, 20-deviation)
  → DB logging (open/close lifecycle, BE detection)
  → _DriftMonitor: PSI on top features, periodic
  → _SmartKillSwitch: rolling 20-trade Sharpe+WR vs expected baseline,
      graduated warn/reduce/disable
  → Hot-reload: only if newer model files AND no open positions
```

**Intentional hardcoded constraints (market physics or account survival):**
- Session gate: 20:30–01:00 London — US30 underlying not trading; not a behavioral assumption
- Max DD 35%: account survival circuit breaker
- Fixed $100/trade: ER accuracy by design; vol-sizing code available via toggle for future scaling

**Risk model:** Fixed $100/trade → linear equity curve → ER numerator and denominator in same unit → order-independent, scale-independent capital efficiency measurement
