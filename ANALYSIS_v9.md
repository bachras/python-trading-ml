# ML Trading System — Analysis v9
**Date:** 2026-04-22
**Instrument:** US30 CFD (Dukascopy tick data → MT5 execution)
**Log analyzed:** `logs/training_20260421_1958.log`
**Run:** Full retrain, 2026-04-21 19:58 → 21:45 (107 min)
**Purpose:** Structured implementation spec for next training run. Every item has a precise file/function location, exact change, and verification criterion. Implement in priority order.

---

## 0. EXECUTIVE SUMMARY: DO NOT TRUST CURRENT RESULTS

The second training run completed successfully and produced richer data than the first. However, the core problems identified in v8 were **not implemented** before this run. All four timeframes (3m, 5m, 10m, 15m) show identical pathological patterns:

| Signal | Value | Status |
|--------|-------|--------|
| Fill rate | 97–99% | ❌ ABORT triggered on all TFs |
| Win rate | 94–98% | ❌ Impossible (realistic: 45–65%) |
| Sharpe | 7–12 | ❌ Not trustable |
| ER | 850–3700 | ❌ All flagged OVERFIT |
| Haircut Sharpe | Negative on all 20 strategies | ❌ Selection bias confirmed |
| SPA p-value | 1.000 on all 20 strategies | ❌ Cannot reject H0 |
| tp_mult (GA/Optuna) | 0.50 on all TFs | ❌ Below minimum viable TP |

**The optimizer is finding the same pathological solution across all TFs: tiny TP + near-perfect fills = fake high win rate = fake high Sharpe.**

---

## 1. WHAT THE SECOND RUN REVEALED

### New log evidence vs v8:

| Symptom | Where in log | Root cause | Fix |
|---------|-------------|-----------|-----|
| `fill_rate=97–99%` → ABORT on all 4 TFs | `[EXECUTION]` | v8 F1 not implemented | G1 (revised) |
| `tp_mult=0.50` selected by both GA and Optuna | params line 489–492 | No TP floor in fitness | G2 |
| `SPA edge WEAK (p=1.000)` on all 20 strategies | every `[BASELINES]` block | Fake edge from execution bias | G1+G2 |
| `Haircut Sharpe` negative on all 20 strategies | strategy summary tables | Selection bias; tiny TP inflates OOS | G2 |
| Ensemble `corr=0.92–0.95` on ALL folds, ALL TFs | `[ENSEMBLE]` warnings | XGB and RF using identical signal | G3 |
| `is_swing_high`/`is_swing_low` = top features BUT marked as known lookahead on 10m/15m | `[IMPORTANCE]` + `[LEAKAGE]` | Signal contamination risk | G4 |
| 15m has 16 degenerate label columns (vs 2–6 for other TFs) | `[LABELS]` 15m | TP dist too tight at 15m ATR scale | G5 |
| Jaccard avg=0.54 (LOW) for 5m and 15m | `[IMPORTANCE]` | Unstable feature set fold-to-fold | G6 |
| GA selected entry_tf=15m + htf_tf=15m (same TF) | params line 489 | HTF context not working | G7 |
| `vwap_dist_pct`, `vwap_anch_dist_pct`, `quote_delta_pct` low-variance on ALL TFs | `[FEATURE]` warnings | These features carry no signal | G8 |
| Calibration max bin delta 0.106–0.116 ALERT on all TFs | `[CALIBRATION]` | Underfitting at high-confidence bins | G9 |
| Regime drift HIGH (fold_0→fold_1) on all TFs | `[REGIME]` | ATR doubled 2020→2024 period | G10 |

---

## 2. CRITICAL PROBLEMS (MUST FIX BEFORE NEXT RUN)

---

### G1. Execution Realism — Still Broken 🚨 CRITICAL (v8 F1 not applied)

**Log evidence:**
```
[EXECUTION] US30/3m  → fill_rate=97.1%  gap_skip_rate=1.4%  delay_rate=39.7%
[EXECUTION] US30/5m  → fill_rate=98.1%  gap_skip_rate=1.1%  delay_rate=38.4%
[EXECUTION] US30/10m → fill_rate=98.3%  gap_skip_rate=1.1%  delay_rate=39.2%
[EXECUTION] US30/15m → fill_rate=98.8%  gap_skip_rate=1.3%  delay_rate=41.4%
[EXECUTION] ABORT — fill_rate=9x.x%. Execution filter not working.  ← repeated 4×
```

**Target after fix:** `fill_rate: 80–90%`, `gap_skip_rate: 3–8%`

**Changes required:**

**File: `.env`**
```ini
EXEC_DELAY_PROB=0.55      # was 0.40 (still too low — fill rate still 97%+)
MT5_DEVIATION_PTS=10      # was 15 (must tighten further; 15 was insufficient)
```

**File: `phase2_adaptive_engine.py` — inside `ga_fitness()`, in the delayed-fill block**

The ATR-move filter from v8 F1 was not applied. Apply it now:
```python
# Reject delayed fill if price moved more than 35% of ATR during delay
price_move = abs(next_open - closes[i])
if price_move > 0.35 * atr[i]:
    n_gap_miss += 1
    continue
```

Also add a hard abort guard inside `ga_fitness()` itself (not just post-hoc in `train.py`):
```python
# After trade loop — abort genome if execution is unrealistic
fill_rate_check = n_trades / max(raw_signals, 1)
if fill_rate_check > 0.92:
    return (-999.0,)   # hard penalty — GA will avoid these genomes
```

**Why 0.92 floor (not 0.95 as in v8):** The v8 threshold of 0.97 was too permissive — fill rates of 97.1–98.8% still triggered ABORT. Lower the genome penalty threshold to 0.92 so the GA actively avoids high-fill regimes during evolution, not just post-hoc.

**Verification:**
```
[EXECUTION] fill_rate=87.4%   ← target 80–90%
[EXECUTION] gap_skip_rate=4.2%  ← target 3–8%
No ABORT lines in [EXECUTION] after fix
```

---

### G2. TP Minimum Floor — GA Still Exploiting tiny TP 🚨 CRITICAL (v8 F2 partially applied)

**Log evidence:**
```
GA best genome: sl_atr=2.0, tp_mult=0.50   ← line 489
Optuna best:    sl_atr=2.65, tp_mult=0.5026  ← line 492
[GRID] best combo: sl2.5_tp1_be1 on all 4 TFs  ← tp=1x smallest grid point
sensitivity tp_mult-0.3: Sharpe increases by +2 to +3 on every TF
```

The optimizer is locked in a pathological basin: TP ≈ 0.5R with avg costs of 4–7 pts means the model wins by having near-zero SL losses (because fills are almost perfect). This is not a real edge.

**Changes required:**

**File: `phase2_adaptive_engine.py` — inside `ga_fitness()`, at the top (before trade loop)**

```python
# Hard TP floor — tp_mult < 1.5 is not viable when avg_cost ≈ 4–7 pts
if tp_mult < 1.5:
    return (-999.0,)

# Hard SL floor (from v8 F2 — verify this is actually applied)
if sl_atr < 2.0:
    return (-999.0,)
```

**File: GA parameter bounds in `phase2_adaptive_engine.py`**

```python
"tp_mult": (1.5, 5.0)   # was (0.3, 5.0) or similar — remove sub-1.5 space
"sl_atr":  (2.0, 4.0)   # was (1.0, 4.0) per v8 — verify this is applied
```

**File: Optuna search space (wherever Optuna suggest_float for tp_mult)**

```python
tp_mult = trial.suggest_float("tp_mult", 1.5, 5.0)   # was 0.3 or 0.5 lower bound
```

**Expected effect:** All GA/Optuna results will show `tp_mult ≥ 1.5`. Win rates will drop from 95%+ to realistic 50–65%. Sharpe values will fall to 1–3 range — this is expected and correct. ER will also become realistic (< 50).

**Verification:**
```
GA best genome: tp_mult=2.1   ← always ≥ 1.5
[BASELINES] expected_win_rate=0.58   ← 50–65% range
```

---

### G3. Ensemble Diversity — XGB and RF Behaving as Single Model 🚨 CRITICAL

**Log evidence (every fold, every TF):**
```
[ENSEMBLE] Fold 1 diversity LOW: corr=0.940 (target<0.90), disagree>10%=9.0%
[ENSEMBLE] Fold 2 diversity LOW: corr=0.946 (target<0.90), disagree>10%=8.9%
[ENSEMBLE] Fold 3 diversity LOW: corr=0.936 (target<0.90), disagree>10%=9.9%
... (same for 5m, 10m, 15m — all 12 fold messages are LOW)
```

The ensemble is providing no diversification benefit. Both models learn the same patterns because they are trained on the same label with similar hyperparameters.

**Changes required:**

**File: `phase2_adaptive_engine.py` or `train.py` — XGB hyperparameters**

Force XGB and RF to learn differently:
```python
# XGB: reduce depth, increase regularization
xgb_params = {
    "max_depth": 4,          # was likely 6; force shallower trees
    "subsample": 0.7,        # was 0.8; more stochastic
    "colsample_bytree": 0.6, # was 0.8; fewer features per tree → diversity
    "reg_lambda": 2.0,       # increase L2 regularization
    "min_child_weight": 30,  # was lower; require more evidence per leaf
}

# RF: increase contrast vs XGB
rf_params = {
    "max_features": 0.3,     # use 30% of features per split (was 'sqrt' ≈ 0.08; try higher value for diff signal)
    "min_samples_leaf": 50,  # force RF to generalize more
    "max_depth": 12,         # let RF grow deeper than XGB
}
```

**File: `train.py` — after `[ENSEMBLE]` correlation computed**

Apply the v8 F10 ensemble discount (carry over from v8):
```python
corr = float(np.corrcoef(p_xgb_oos, p_rf_oos)[0, 1])
if corr > 0.90:
    log.warning(f"[ENSEMBLE] HIGH CORRELATION ({corr:.3f}) — Sharpe discounted 20%.")
    oos_sharpe_adjusted = oos_sharpe * 0.80   # increased from 0.85 in v8
else:
    oos_sharpe_adjusted = oos_sharpe
```

**Verification:**
```
[ENSEMBLE] Fold 1 diversity OK: corr=0.81   ← target < 0.90
[ENSEMBLE] disagree>10%=18.4%   ← target > 15%
```

---

### G4. Lookahead Feature Contamination in Top Importance 🚨 CRITICAL (NEW)

**Log evidence:**
```
[LEAKAGE] US30/10m: known lookahead features changed under permutation (accepted approximation):
  ['is_swing_low', 'equal_low']
[LEAKAGE] US30/15m: known lookahead features changed under permutation (accepted approximation):
  ['is_swing_high', 'dist_swing_high', 'equal_high']
```

And simultaneously:
```
[IMPORTANCE] US30/10m fold=1: is_swing_high=0.1175, is_swing_low=0.1167  ← top 2
[IMPORTANCE] US30/15m fold=1: is_swing_high=0.1137, is_swing_low=0.0797  ← top 2
```

**The #1 and #2 most important features are also flagged as potential lookahead features on 10m and 15m.** Even though the leakage check accepts them as an "approximation", having them dominate importance at 12–17% each means the model may be partially predicting using future information.

**This is a critical integrity issue.**

**Root cause:** Swing high/low detection typically requires future bars to confirm (e.g., a swing high at bar N requires bars N+1 and N+2 to be lower). If the lookahead bars are within the same label window, this creates leakage.

**Investigation required:**

**File: wherever `is_swing_high` / `is_swing_low` are computed (likely `institutional_features.py` or `features.py`)**

Check the calculation:
```python
# If this is something like:
is_swing_high = (high[i] > high[i-1]) & (high[i] > high[i+1])
#                                                    ^^^^^^^^ FUTURE BAR = LEAKAGE
```

**Fix options:**
1. **Strict fix:** Recompute using only past bars: `is_swing_high = (high[i] > high[i-1]) & (high[i] > high[i-2])` — zero lookahead but slightly noisier
2. **Pragmatic fix:** Keep 1-bar lookahead (only `i+1`) but mark explicitly; ensure label window starts at `i+2` not `i+1`
3. **Diagnostic step:** Run feature importance on a version where `is_swing_high`/`is_swing_low` are explicitly zeroed out. If AUC drops significantly (> 0.01), leakage is real.

**Verification:**
```
[LEAKAGE] US30/10m: leakage check passed (147 features checked)  ← no lookahead warning
[IMPORTANCE] is_swing_high importance drops below 0.08  ← when using past-only definition
```

---

## 3. IMPORTANT PROBLEMS (HIGH VALUE, FIX BEFORE NEXT RUN IF POSSIBLE)

---

### G5. Label Grid Degenerate Columns — Especially 15m 🟠 IMPORTANT

**Log evidence:**
```
[LABELS] 3m  — 6  degenerate columns (TP rate < 2% or > 98%)
[LABELS] 5m  — 2  degenerate columns
[LABELS] 10m — 4  degenerate columns
[LABELS] 15m — 16 degenerate columns  ← much worse than others
```

15m degenerate labels include `tp4` and `tp5` combos with multiple `be` variants. At 15m candles, ATR is ~60 pts; a TP of 5R means ~300 pts target — almost never reached in the backtest window.

**Changes required:**

**File: `phase2_adaptive_engine.py` or wherever label grid is defined**

Narrow the TP grid for 15m:
```python
if tf == 15:
    tp_values = [1, 2, 3]       # was [1, 2, 3, 4, 5]; remove tp4 and tp5 for 15m
    sl_values = [2.0, 2.5, 3.0] # was [1.5, 2.0, 2.5, 3.0, 3.5]; remove sl1.5 and sl3.5 for 15m
```

This reduces the 15m label grid from 75 combos (5×5×3) to 27 combos (3×3×3), eliminating the 16 near-zero columns.

**Also implement v8 F5 (spread-aware entry) now** — this additionally reduces TP rates by 1–3% across all TFs, consistent with realistic execution.

**Verification:**
```
[LABELS] 15m — 0 degenerate columns   ← no near-zero TP rate columns
[LABELS] 15m TP rate range [0.032, 0.385]   ← tighter, more usable range
```

---

### G6. Feature Stability — 5m and 15m Use Different Features Per Fold 🟠 IMPORTANT

**Log evidence:**
```
[IMPORTANCE] US30/3m  cross-fold Jaccard: avg=0.82 (OK)
[IMPORTANCE] US30/5m  cross-fold Jaccard: avg=0.54 (LOW — model uses different features per fold)
[IMPORTANCE] US30/10m cross-fold Jaccard: avg=0.82 (OK)
[IMPORTANCE] US30/15m cross-fold Jaccard: avg=0.54 (LOW — model uses different features per fold)
```

Low Jaccard means the model discovers different explanations of the data in each time period. This is a generalization risk — the "consistent" features in 5m and 15m are only 7 features (vs 9 for 3m/10m).

**Changes required:**

**File: `train.py` — after cross-fold Jaccard is computed**

Add a warning gate:
```python
if jaccard_avg < 0.65:
    log.warning(
        f"[IMPORTANCE] {symbol}/{tf}m — LOW feature stability (Jaccard={jaccard_avg:.2f}). "
        f"Model may not generalize. Consider: (1) increase L2 reg, (2) add feature selection, "
        f"(3) increase min_child_weight."
    )
    # Discount OOS Sharpe by 10% for unstable models
    oos_sharpe_adjusted *= 0.90
```

This does not block training but penalizes unstable models in ranking. Combined with G3 (diversity hyperparameters), the RF and XGB changes may naturally improve feature stability.

**Verification:**
```
[IMPORTANCE] US30/5m cross-fold Jaccard: avg=0.71 (OK)   ← target ≥ 0.65
```

---

### G7. GA HTF Degenerate — Same TF for Entry and HTF 🟠 IMPORTANT (NEW)

**Log evidence:**
```
GA best genome: {'entry_tf': 15, 'htf_tf': 15, ...}   ← line 489
```

The GA selected entry_tf=15m AND htf_tf=15m. This means HTF context = same data as entry = zero multi-timeframe benefit. The HTF nudge (F7 in v8) is being applied using the same candle the signal comes from.

**Changes required:**

**File: `phase2_adaptive_engine.py` — inside `ga_fitness()`, at the top**

```python
# Hard constraint: HTF must be at least 2x the entry TF
if htf_tf <= entry_tf:
    return (-999.0,)
```

**File: GA bounds for `htf_tf`**

Ensure the bounds prevent same-TF selection:
```python
# If entry_tf=3m, valid htf_tf = [5, 10, 15, 60, 240]
# If entry_tf=5m, valid htf_tf = [10, 15, 60, 240]
# If entry_tf=10m, valid htf_tf = [15, 60, 240]
# If entry_tf=15m, valid htf_tf = [60, 240]
```

This forces the GA to use genuine higher timeframe context.

**Verification:**
```
GA best genome: {'entry_tf': 10, 'htf_tf': 60, ...}   ← htf > entry always
```

---

### G8. Permanently Low-Variance Features — Remove from Training 🟠 IMPORTANT

**Log evidence:**
```
[FEATURE] Low-variance institutional columns detected (tf=3m):  ['vwap_dist_pct', 'vwap_anch_dist_pct', 'quote_delta_pct']
[FEATURE] Low-variance institutional columns detected (tf=5m):  ['vwap_dist_pct', 'vwap_anch_dist_pct', 'quote_delta_pct']
[FEATURE] Low-variance institutional columns detected (tf=10m): ['vwap_dist_pct', 'vwap_anch_dist_pct', 'quote_delta_pct']
[FEATURE] Low-variance institutional columns detected (tf=15m): ['vwap_dist_pct', 'vwap_anch_dist_pct', 'quote_delta_pct']
```

These three features are low-variance on **every timeframe, every run**. They are not frozen (std > 0.01) but they carry near-zero predictive information. Keeping them in training wastes model capacity and may introduce noise.

**Changes required:**

**File: wherever features are filtered before model training (likely `train.py` or `phase2_adaptive_engine.py`)**

```python
# Permanently exclude known low-variance columns before training
LOW_VARIANCE_DROP = [
    'vwap_dist_pct',
    'vwap_anch_dist_pct',
    'quote_delta_pct',
]
X_train = X_train.drop(columns=[c for c in LOW_VARIANCE_DROP if c in X_train.columns])
```

**Note:** This also applies inside `ga_fitness()` if features are passed directly. Ensure the live pipeline (`live.py`) also excludes these columns from inference.

**Verification:**
```
[FEATURE] Low-variance institutional columns detected: none   ← no warning on 3m/5m/10m/15m
```

---

### G9. Calibration — Max Bin Delta Still Above Alert Threshold 🟠 IMPORTANT

**Log evidence:**
```
[CALIBRATION] US30_3m  — max bin delta=0.106 ← ALERT  (Brier 0.2367→0.2352)
[CALIBRATION] US30_5m  — max bin delta=0.110 ← ALERT  (Brier 0.2366→0.2351)
[CALIBRATION] US30_10m — max bin delta=0.116 ← ALERT  (Brier 0.2372→0.2359)
[CALIBRATION] US30_15m — max bin delta=0.106 ← ALERT  (Brier 0.2368→0.2352)
```

The calibration improves (Brier decreases) on all TFs — so the v8 F6 guard would not trigger. However, the calibrated model still has large residual errors at high-confidence bins (0.80–0.90 range). At these bins, the model predicts 0.87 but the actual win rate is 0.97 — a 10+ point gap.

**Root cause:** Isotonic regression has too few samples in high-confidence bins (n=200–1400). With these sample sizes, isotonic regression is unreliable.

**Changes required:**

**File: `train.py` — calibration section**

```python
# Alert gate: if max bin delta > 0.10, try Platt scaling as fallback
if max_bin_delta > 0.10:
    log.warning(
        f"[CALIBRATION] max bin delta={max_bin_delta:.3f} > 0.10. "
        f"Isotonic may be overfitting sparse high-confidence bins. Trying Platt scaling."
    )
    # Fallback: fit logistic regression calibrator
    from sklearn.calibration import CalibratedClassifierCV
    platt_cal = LogisticRegression(C=1.0).fit(
        oos_probs_all.reshape(-1, 1), oos_labels_all
    )
    platt_brier = brier_score_loss(oos_labels_all, platt_cal.predict_proba(oos_probs_all.reshape(-1,1))[:,1])
    if platt_brier < brier_after:
        log.info(f"[CALIBRATION] Platt scaling better ({platt_brier:.4f} < {brier_after:.4f}). Using Platt.")
        calibrator = platt_cal
        brier_after = platt_brier
```

**Verification:**
```
[CALIBRATION] max bin delta=0.063   ← target < 0.08
[CALIBRATION] Brier: before=0.2368 → after=0.2331   ← larger improvement
```

---

## 4. MEDIUM PRIORITY (IMPLEMENT AFTER CRITICAL/IMPORTANT FIXES)

---

### G10. Regime Drift — Adjust Walk-Forward Weighting 🟡 MEDIUM

**Log evidence:**
```
[REGIME] US30/3m  fold_0→fold_1 drift=5.968 (HIGH) | atr_mean: 15.2→22.2, kurtosis: 32→607
[REGIME] US30/5m  fold_0→fold_1 drift=5.512 (HIGH) | atr_mean: 19.9→29.0, kurtosis: 25→437
[REGIME] US30/10m fold_0→fold_1 drift=3.281 (HIGH) | atr_mean: 28.5→41.6, kurtosis: 21→231
[REGIME] US30/15m fold_0→fold_1 drift=2.639 (HIGH) | atr_mean: 35.3→51.7, kurtosis: 19→167
```

All TFs show HIGH drift from fold 0 (2020–2023) to fold 1 (2024). ATR roughly doubled and kurtosis exploded (COVID/2022 volatility regime change). However, fold 1→fold 2 drift is OK on all TFs.

This means the model trained on fold 0 data (low-ATR) may not generalize to fold 1+ (high-ATR). The 3-fold walk-forward overweights 2020–2023 conditions.

**Changes required:**

**File: `train.py` — walk-forward fold construction**

Weight OOS evaluation by recency:
```python
# Apply recency weighting when computing final OOS Sharpe
fold_weights = [0.20, 0.35, 0.45]  # fold 3 (most recent) weighted highest
oos_sharpe_weighted = sum(s * w for s, w in zip(fold_sharpes, fold_weights))
```

Also add an ATR-normalized training option: when drift is HIGH, consider scaling features by the rolling ATR before training:
```python
# Normalize price-dependent features by ATR to reduce regime sensitivity
for col in ATR_SENSITIVE_FEATURES:
    X_train[col] = X_train[col] / X_train['atr14']
```

**Verification:**
```
[REGIME] fold_0→fold_1 drift=5.9 (HIGH) — recency weight applied (0.20/0.35/0.45)
```

---

### G11. Sensitivity Asymmetry — TP Reduction Always Helps 🟡 MEDIUM (NEW INSIGHT)

**Log evidence (5m as example, pattern identical across all TFs):**
```
sensitivity tp_mult-0.3: Sharpe=9.84 ER=2772    ← reducing TP increases Sharpe by +2.5
sensitivity tp_mult+0.3: Sharpe=6.62 ER=1268    ← increasing TP drops Sharpe
```

The sensitivity analysis is showing that TP reduction monotonically increases Sharpe. This is a direct signature of execution exploitation — the optimizer is rewarded for smaller TP because fills are near-perfect. After G1 (execution fix) and G2 (TP floor), this asymmetry should disappear. If it persists after those fixes, it indicates a deeper structural issue.

**Action:** No new code change required. This is a diagnostic indicator.

After implementing G1 and G2, recheck sensitivity:
```
# Target after fixes:
sensitivity tp_mult-0.3: Sharpe=2.1    ← reducing TP below 1.5 is penalized
sensitivity tp_mult+0.3: Sharpe=2.8    ← increasing TP slightly improves or is neutral
```

If tp sensitivity is still strongly asymmetric after G1+G2, revisit label generation.

---

## 5. CARRY-OVER FROM v8 (NOT YET IMPLEMENTED)

The following items from v8 were not applied before this run. Apply them in addition to new items:

| v8 Item | Status | Priority for next run |
|---------|--------|----------------------|
| F1 — Execution realism | ❌ Not applied | Superceded by G1 (tighter params) |
| F2 — SL floor + cost ratio penalty | ❌ Not applied (sl_atr=2.0 in GA but Optuna still found 2.65 — check bounds) | Superceded by G2 |
| F3 — Execution regime penalties in GA fitness | ❌ Not applied | Included in G1 |
| F4 — Drop 1m from optimization | ✅ Applied — 1m not in this run | Done |
| F5 — Spread-aware entry in labels | ❌ Not applied | Apply with G5 |
| F6 — Calibration robustness guard | ✅ Guard not triggered (Brier improved) | Keep; enhance with G9 |
| F7 — HTF nudge cap ±0.05 | ❌ Not applied | Apply still relevant |
| F8 — Trade frequency sanity constraint | ❌ Not applied | Apply; trades_per_day currently 3.65–9.59 (3m runs at 9.59 borderline) |
| F9 — Execution hard-fail validation | ✅ Partially — ABORT fires but training continues | Strengthen: make it actually stop training |
| F10 — Ensemble redundancy penalty | ❌ Not applied | Superceded by G3 |

---

## 6. WHAT IS WORKING WELL (DO NOT CHANGE)

| Component | Evidence | Status |
|-----------|----------|--------|
| 1m removal | Not in this run, no OOM | ✅ Keep dropped |
| 147 features (TICK+INSTITUTIONAL) | No frozen/exploding features on any TF | ✅ Good |
| PARITY check | PASS on all 4 TFs (max_diff < 1e-5) | ✅ Reliable |
| Probability distribution shape | mean≈0.507, std≈0.122, NORMAL on all TFs | ✅ Model not collapsed |
| Christmas filter | Applied correctly (16k, 9.7k, 4.9k, 3.2k bars removed by TF) | ✅ Working |
| Leakage check | Passes on 3m/5m; warns correctly on 10m/15m | ✅ Working (G4 addresses warning) |
| Walk-forward fold structure | 3 folds, OOS by year (2024/2025/2026) | ✅ Keep |
| [STABILITY] sensitivity | All TFs: SMOOTH shape | ✅ Good structure |
| Monte Carlo | All 20 strategies: MC PASS | ✅ Robust in backtest universe |
| Calibration direction | Brier improves on all TFs (guard not triggered) | ✅ Calibration working |
| Spread/slippage realism | 2.85–3.07 pts spread, 2.41–5.32 pts slippage | ✅ Realistic for US30 |

---

## 7. `.env` CHANGES REQUIRED

```ini
# ── EXECUTION LATENCY MODEL (tighten further vs v8) ──────────────────────────
EXEC_DELAY_PROB=0.55        # was 0.40 in v8 (still produced 97%+ fill rate)
MT5_DEVIATION_PTS=10        # was 15 in v8 (still insufficient)

# ── TIMEFRAMES ────────────────────────────────────────────────────────────────
TIMEFRAMES=3,5,10,15        # keep current 4 TFs; 1m stays dropped
```

---

## 8. IMPLEMENTATION ORDER

Implement in this exact sequence:

1. **G1** — Tighten `.env` params AND add ATR-move filter AND add fill_rate hard abort inside `ga_fitness()`. This is the single most important change.
2. **G2** — Add `tp_mult ≥ 1.5` hard floor in `ga_fitness()` AND tighten GA/Optuna bounds for `tp_mult`. Without this, optimizer will find the same pathological solution even with better execution.
3. **G4** — Investigate `is_swing_high`/`is_swing_low` lookahead definition. Run diagnostic (zero out these features → check AUC drop). This is a data integrity issue that must be resolved before trusting any result.
4. **G7** — Add HTF ≠ entry_tf constraint in `ga_fitness()`. Quick single-line fix.
5. **G3** — Adjust XGB and RF hyperparameters to improve ensemble diversity. Apply the 20% ensemble correlation discount.
6. **F7 (v8)** — Apply HTF nudge cap ±0.05 in both `ga_fitness()` and `live.py`.
7. **F8 (v8)** — Apply trade frequency penalty. Check 3m trades_per_day=9.59 (near overtrading at 10/day limit).
8. **G5** — Narrow 15m label grid + apply spread-aware entry (F5 from v8).
9. **G8** — Drop `vwap_dist_pct`, `vwap_anch_dist_pct`, `quote_delta_pct` from training features.
10. **G6** — Add Jaccard stability penalty (10% discount for avg < 0.65).
11. **G9** — Add Platt scaling fallback when isotonic max bin delta > 0.10.
12. **G10** — Add recency-weighted OOS Sharpe (0.20/0.35/0.45 fold weights).

---

## 9. VERIFICATION CHECKLIST FOR NEXT TRAINING RUN

Run through this after training completes before accepting any result.

### Execution model:
- [ ] `[EXECUTION]` `fill_rate` in 80–90% range on all TFs — no ABORT lines
- [ ] `[EXECUTION]` `gap_skip_rate` ≥ 3% on all TFs
- [ ] No `[EXECUTION] ABORT` messages anywhere in log

### Optimization:
- [ ] `GA best genome: tp_mult ≥ 1.5` on ALL TF combinations
- [ ] `Optuna best: tp_mult ≥ 1.5` — no 0.5x values
- [ ] `GA best genome: htf_tf > entry_tf` — not the same TF
- [ ] `GA best genome: sl_atr ≥ 2.0` — confirmed
- [ ] `[BASELINES] expected_win_rate` in 0.45–0.65 range — NOT 0.95+
- [ ] `[BASELINES] expected_sharpe` in 1.5–4.0 range — NOT 7–12
- [ ] `[BASELINES] ER` in 5–50 range — NOT 800–3700
- [ ] SPA p-value < 0.10 on at least one TF — NOT p=1.000 across the board
- [ ] Haircut Sharpe positive on rank-1 strategies — NOT negative
- [ ] `[BASELINES] expected_trades_per_day` in 1.5–9.0 range

### Ensemble:
- [ ] `[ENSEMBLE]` correlation < 0.90 on all folds — no LOW diversity warnings
- [ ] `disagree>10%` > 15% on all folds

### Labels:
- [ ] `[LABELS] 15m` — 0 or ≤ 4 degenerate columns (was 16)
- [ ] `[LABELS] TP rate range` slightly lower than current run (spread-aware entry effect)

### Features:
- [ ] No `[FEATURE] Low-variance` warning for `vwap_dist_pct`, `vwap_anch_dist_pct`, `quote_delta_pct`
- [ ] `[IMPORTANCE]` — `is_swing_high` importance < 0.09 on 10m/15m (post lookahead fix)

### Calibration:
- [ ] `[CALIBRATION]` max bin delta < 0.08 on all TFs
- [ ] `[CALIBRATION]` Brier improves on all TFs — no GUARD triggered

### Final audit:
- [ ] Zero `[OVERFIT ER]` red flags in FINAL STRATEGY AUDIT
- [ ] Zero `[NEG HC_SR]` red flags
- [ ] At least one strategy without `[ZERO DD]` flag

### Red flags — stop and investigate immediately if seen:
- `fill_rate > 92%` → G1 ATR-move filter not applied; check `ga_fitness()` block
- `tp_mult < 1.5` in any genome → G2 floor not applied; check bounds and early return path
- `htf_tf == entry_tf` → G7 constraint not applied
- `WR% > 80%` on any rank-1 strategy → execution model still broken despite G1
- `SPA p=1.000` on all strategies → tp_mult constraint not working; re-examine G2
- `[ENSEMBLE] LOW` on any fold → G3 hyperparameter changes insufficient; tighten further
- `[IMPORTANCE] is_swing_high > 0.15` on 10m/15m → lookahead still present; re-examine G4

---

## 10. KEY NUMBERS COMPARISON: v8 RUN vs EXPECTED AFTER FIXES

| Metric | v8 (first run) | v9 (this run) | Expected after fixes |
|--------|---------------|--------------|---------------------|
| fill_rate | 95–99% | 97–99% | 80–90% |
| win rate | 95–98% | 94–98% | 45–65% |
| Sharpe (rank-1) | 7–8 | 7–9 | 1.5–4.0 |
| ER | 500–2400 | 850–3700 | 5–50 |
| SPA p-value | 1.000 | 1.000 | < 0.10 |
| Haircut Sharpe | Negative | Negative | Positive |
| tp_mult (best) | 0.5 | 0.5 | ≥ 1.5 |
| Ensemble corr | Not measured | 0.92–0.95 | < 0.90 |
| Degenerate labels (15m) | N/A | 16 | 0–4 |
| Calibration max bin delta | 0.106 | 0.106–0.116 | < 0.08 |

---

## 11. KNOWN ISSUES DEFERRED (NOT IN SCOPE FOR NEXT RUN)

| Issue | Deferred reason |
|-------|----------------|
| `vp_poc`/ORB all-NaN on 240m | Insufficient bars for session-level groupby; acceptable for HTF context |
| Regime ATR normalization (G10 advanced part) | Requires feature engineering changes; implement after execution/TP fixes stabilize results |
| `quote_delta_divergence` low-variance on 3m/5m (not all TFs) | Monitor; if consistently low add to G8 drop list |

---

## 12. SIMPLE SUMMARY

| Component | Status | Action |
|-----------|--------|--------|
| Model (AUC 0.62–0.63) | ✅ Healthy | No change |
| Features (147, no frozen/exploding) | ✅ Good | Drop 3 low-variance + investigate swing lookahead |
| Pipeline (parity, leakage, stability) | ✅ Solid | Keep; fix HTF same-TF bug |
| Execution model | ❌ Still broken | G1 — tighten EXEC_DELAY_PROB + MT5_DEVIATION_PTS + ATR filter |
| TP constraint | ❌ Not applied | G2 — tp_mult ≥ 1.5 hard floor |
| Results (Sharpe/WR/ER/SPA) | ❌ Not trustworthy | Will normalize after G1+G2 |
| Ensemble diversity | ❌ Broken | G3 — hyperparameter changes |
| Lookahead integrity | ⚠️ Suspected | G4 — must audit before trusting importance |

**Do not retrain until G1 and G2 are implemented. Retraining without the TP floor will produce identical results.**
