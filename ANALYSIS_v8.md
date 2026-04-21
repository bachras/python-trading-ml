# ML Trading System — Post-First-Training Analysis v8
**Date:** 2026-04-19
**Instrument:** US30 CFD (Dukascopy tick data → MT5 execution)
**Purpose:** Structured implementation spec for a second AI agent. Based on first full training run results. Every item has a precise file/function location, exact change, and verification criterion. No ambiguity. Implement in priority order.

---

## 0. WHAT THE FIRST TRAINING RUN REVEALED

The first training run completed. The following problems were observed in the logs and output metrics. Each problem maps to one or more fixes in Section 1.

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| `fill_rate > 95%` in `[EXECUTION]` | `MT5_DEVIATION_PTS=30` too permissive; `EXEC_DELAY_PROB=0.30` too low; no ATR-move filter during delay | F1 |
| `gap_skip_rate < 1%` in `[EXECUTION]` | Same — gap rejection almost never firing | F1 |
| `sl_atr < 2.0` selected by GA | No hard floor; cost as fraction of SL is too high on tight SLs | F2 |
| `avg_total_cost > 12% of SL` | SL too tight relative to spread+slippage | F2 |
| GA not penalising unrealistic execution regimes | Fill/gap stats observed but not fed back into fitness | F3 |
| 1m timeframe consuming RAM, skipping institutional features, polluting optimizer | 2.1M bars → OOM on parity; no institutional features → weak signal | F4 |
| Labels assume perfect zero-cost entry | `entry_price = closes[i]`; spread not added to entry | F5 |
| Calibration may silently degrade | No rejection rule if `brier_after >= brier_before` | F6 |
| HTF nudge can move probability by ±0.09 or more | `htf_weight * 0.3` uncapped; can dominate model output | F7 |
| GA selects strategies with `< 1.5` or `> 10` trades/day | No frequency constraint in fitness | F8 |
| `[EXECUTION]` values only logged, not validated | Unrealistic spread/slippage silently passes | F9 |
| `[ENSEMBLE]` correlation `> 0.92` — ensemble adds no value | Correlation observed but not penalised | F10 |

---

## 1. IMPLEMENTATION ITEMS (PRIORITY ORDER)

---

### F1. Execution Realism — Fix Fill Rate 🚨 CRITICAL

**Target after fix:**
- `fill_rate`: 85%–92%
- `gap_skip_rate`: 2%–6%

**Current state:** `MT5_DEVIATION_PTS=30` allows most delayed fills through. `EXEC_DELAY_PROB=0.30` only delays 30% of trades. No ATR-based move filter — even if a delayed bar opens far from signal price, it still fills.

**Changes required:**

**File: `.env`**
```ini
MT5_DEVIATION_PTS=15       # tighten from 30 → 15 points
EXEC_DELAY_PROB=0.40       # increase from 0.30 → 0.40
```

**File: `phase2_adaptive_engine.py` — inside `ga_fitness()` trade loop, in the delayed-fill block**

Current delayed-fill block (approximately):
```python
if rng.random() < EXEC_DELAY_PROB and i + 1 < n_bars:
    next_open = opens[i + 1]
    gap_pts   = abs(next_open - closes[i])
    if gap_pts > MT5_DEVIATION_PTS:
        n_gap_miss += 1
        continue
    entry_price = next_open
    extra_slip  = gap_pts
```

Add an **ATR-based price-move rejection** inside the delayed block, after the `gap_pts > MT5_DEVIATION_PTS` check:
```python
    # NEW: reject delayed fill if price moved more than 40% of ATR during delay
    # This models price running away before MT5 order reaches the exchange
    price_move = abs(next_open - closes[i])
    if price_move > 0.4 * atr[i]:
        n_gap_miss += 1
        continue
```

This single rule rejects fills on fast-moving candles (news, session opens) where in live trading the order would either be rejected or would fill at a deeply unfavourable price that the backtest would not capture. The `0.4 * ATR` threshold means: if the bar moves more than 40% of its own average range during the delay, skip the trade.

**Why this works:** On average US30 ATR of ~50 points, this rejects fills when price moves > 20 points during the delay window. Combined with `MT5_DEVIATION_PTS=15`, this produces realistic 85–92% fill rates.

**Verification (`[EXECUTION]` log after fix):**
```
fill_rate     = 87.4%      ← target 85–92%
gap_skip_rate = 4.2%       ← target 2–6%
delay_rate    = 39.8%      ← near EXEC_DELAY_PROB=0.40
```
⚠️ Red flag: `fill_rate > 95%` after fix → ATR-move filter not applied; check variable name `atr[i]` matches the array in `ga_fitness()`
⚠️ Red flag: `gap_skip_rate = 0%` → `n_gap_miss` counter not being incremented in new block

---

### F2. Minimum SL Floor + Cost Ratio Penalty 🚨 CRITICAL

**Problem:** GA selects `sl_atr < 2.0` because tight SLs produce more trades which inflates the fitness denominator. Cost (spread + slippage) as a fraction of the SL distance becomes unrealistically high.

**Changes required:**

**File: `phase2_adaptive_engine.py` — inside `ga_fitness()`, at the top of the function body (before the trade loop)**

Add a hard floor that returns immediately with a penalty score:
```python
# Hard SL floor — sl_atr < 2.0 is not viable with the spread model
# Cost (~3.5 pts) vs SL distance (sl_atr * ATR) must stay below 15%
if sl_atr < 2.0:
    return (-np.inf,)   # or however ga_fitness returns to DEAP
```

Add a cost-ratio penalty **inside the trade loop, after `total_cost` is computed**, accumulated per trade:
```python
# Track cost-to-SL ratio per trade
cost_ratio_sum += total_cost / max(sl_dist, 1e-9)
```

Then **after the trade loop**, add the penalty to fitness:
```python
avg_cost_ratio = cost_ratio_sum / max(n_trades, 1)
if avg_cost_ratio > 0.15:
    # Cost consuming > 15% of each SL on average — not viable live
    fitness *= 0.70
```

**File: `GA_BOUNDS` in `phase2_adaptive_engine.py`**

Tighten the lower bound of `sl_atr` to match the hard floor:
```python
"sl_atr": (2.0, 4.0)   # was (1.0, 4.0)
```

This prevents GA from wasting evaluations in the sub-2.0 zone that is now rejected.

**Verification (`[GA]` log after fix):**
```
[GA] best genome: sl_atr=2.4  ← always ≥ 2.0 after fix
```
⚠️ Red flag: `sl_atr=1.x` still appearing → hard floor not applied; check `ga_fitness()` receives the correct parameter key

---

### F3. Execution Regime Penalties in GA Fitness 🚨 CRITICAL

**Problem:** Execution stats (fill rate, gap rate) are computed inside `ga_fitness()` and logged in `[EXECUTION]`, but do not feed back into the fitness score. GA can select strategies with unrealistic execution (e.g., `fill_rate=99%`) which will never be achievable live.

**Changes required:**

**File: `phase2_adaptive_engine.py` — inside `ga_fitness()`, **after** the trade loop, before the final fitness calculation**

Add execution regime penalties using the counters already being tracked:
```python
# --- Execution regime fitness penalties ---
# fill_rate: fraction of signals that actually filled
fill_rate     = n_trades / max(raw_signals, 1)   # raw_signals = total signals before fill_prob
gap_skip_rate = n_gap_miss / max(n_delayed, 1)   # n_delayed = trades that entered delay branch

# Penalty 1: fill_rate too high → model is living in a zero-friction fantasy
if fill_rate > 0.95:
    fitness *= 0.80   # -20%: almost nothing is being rejected; unrealistic

# Penalty 2: fill_rate too low → strategy is not tradable (too selective/costly)
if fill_rate < 0.75:
    fitness *= 0.85   # -15%: too many rejections; probably pathological SL/spread combo

# Penalty 3: gap_skip_rate too low → ATR-move filter and gap rejection not triggering
# This means the strategy only fires on very calm bars (may not generalise to live)
if gap_skip_rate < 0.01:
    fitness *= 0.90   # -10%: suspiciously low rejection; news/open candles not filtered
```

**Requires adding two new counters to `ga_fitness()` setup:**
```python
raw_signals = 0   # increment once per bar where a signal is generated (before fill_prob)
n_delayed   = 0   # increment once per trade that enters the EXEC_DELAY_PROB branch
```

Increment `raw_signals` at each signal bar (before fill_prob check), `n_delayed` when `rng.random() < EXEC_DELAY_PROB` branch is entered.

**Verification (`[EXECUTION]` log):**
After fix, the GA will select genomes with `fill_rate` in the 85–92% range naturally. Log should show these penalties reduced fitness for genomes outside the band.

---

### F4. Drop 1-Minute Timeframe from Optimization 🚨 HIGH

**Problem:**
- 1m data has 2.1M bars → institutional features skipped (> 750k threshold) → only 81 tick features, no VWAP/volume profile/order flow → weakest signal quality
- Parity check causes OOM on 1m (allocating 1.1 GB for institutional features on 2M rows)
- 1m GA is optimizing on tick features only, which is a different feature set than 3m/5m; results are not comparable and pollute the TF selection

**Changes required:**

**File: `train.py` (or wherever the timeframe list is defined)**

Remove 1m from the optimization loop:
```python
TIMEFRAMES = [3, 5]   # was [1, 3, 5]; remove 1
```

If the list is env-driven (`TIMEFRAMES` in `.env`), update `.env`:
```ini
TIMEFRAMES=3,5
```

**Alternative (less disruptive):** Keep 1m data loading for HTF context (used as a higher-resolution input to 3m/5m features) but exclude from GA optimization loop:
```python
OPTIMIZE_TIMEFRAMES = [3, 5]   # only these enter the GA/Optuna loop
```

**Why not just fix the 1m OOM?** The skip of institutional features on > 750k bars means 1m will always train on a weaker feature set (81 columns vs 161). The resulting model would be compared against 3m/5m models that have VWAP, volume profile, order flow — a fundamentally unfair comparison. Remove from optimization; 1m can re-enter when the 750k-bar institutional feature skip is resolved.

**Verification:** Training log should show only `US30/3m` and `US30/5m` in optimization output. No `[PARITY]` OOM. Total training time drops significantly.

---

### F5. Spread-Aware Entry in Triple-Barrier Labels 🟠 IMPORTANT

**Problem:** Labels are generated as: signal at `closes[i]` → check if price reaches TP/SL within MAX_HOLD bars. Entry is assumed at `closes[i]` with zero cost. In live trading, actual entry is `closes[i] + spread/2` (long) or `closes[i] - spread/2` (short). This makes labels systematically optimistic — TP is computed from a better entry than live will achieve.

**Changes required:**

**File: `phase2_adaptive_engine.py` — inside `engineer_features()`, in the triple-barrier label generation, for both `be=0` (vectorized) and `be=1/2` (loop) variants**

For the vectorized `be=0` block, adjust TP/SL thresholds to account for spread at entry:
```python
# Spread-aware entry adjustment
# Use SPREAD_BASE_PTS as a proxy for half-spread cost at entry
entry_spread_cost = SPREAD_BASE_PTS / 2.0   # half-spread on entry

# Adjust: TP is harder to reach (spread works against you)
# SL is easier to hit (spread works against you from entry)
tp_distance_adj = tp_distance + entry_spread_cost    # TP is further away
sl_distance_adj = sl_distance - entry_spread_cost    # SL is closer
sl_distance_adj = max(sl_distance_adj, sl_distance * 0.5)  # safety floor
```

For the `be=1/2` per-trade loop, adjust `tp_price` and `sl_price`:
```python
# In the loop, after computing entry_price:
half_spread = SPREAD_BASE_PTS / 2.0
if direction == 1:   # long
    effective_entry = entry_price + half_spread
else:                 # short
    effective_entry = entry_price - half_spread
# Recompute TP/SL from effective_entry instead of entry_price
```

**Expected effect:** TP rates across the 75-column grid will decrease by ~1–3% (especially for low tp_mult columns where the spread cost is a significant fraction of the TP distance). This brings labels closer to live reality.

**Verification (`[LABELS]` log after fix):**
```
[LABELS] TP rate range [0.006, 0.489]   ← slightly lower than pre-fix range [0.008, 0.502]
```
⚠️ Red flag: TP rates unchanged → `SPREAD_BASE_PTS` variable not accessible in `engineer_features()` scope; pass it as a parameter or read from env directly

---

### F6. Calibration Robustness Guard 🟠 IMPORTANT

**Problem:** If `brier_after >= brier_before`, the current code still uses the calibrator (isotonic regression overfit or too few samples). A degraded calibrator actively harms live trading by distorting probability estimates.

**Changes required:**

**File: `train.py` — in the calibration block, after computing `brier_after`**

```python
# Guard 1: reject calibrator if it does not improve Brier
if brier_after >= brier_before:
    log.warning(
        f"[CALIBRATION] GUARD — calibrator degraded OOS Brier "
        f"({brier_before:.4f} → {brier_after:.4f}). "
        f"Skipping calibration. Raw model probabilities will be used."
    )
    calibrator = None   # live.py already handles None calibrator
    use_calibrator = False
else:
    use_calibrator = True

# Guard 2: warn if OOS sample is too small for reliable isotonic fit
if len(oos_probs_all) < 10_000:
    log.warning(
        f"[CALIBRATION] Small OOS pool ({len(oos_probs_all)} samples). "
        f"Isotonic regression may be unstable. Consider 3+ folds."
    )
```

**Ensure `use_calibrator` flag flows through to model save:** If `use_calibrator = False`, save a sentinel (e.g., `None` or a pass-through calibrator) so `live.py` uses raw probabilities without crashing.

**Verification (`[CALIBRATION]` log):**
```
[CALIBRATION] GUARD — calibrator degraded ... Skipping calibration.   ← only if Brier worsened
```
Normal case: no GUARD message, calibration proceeds as before.

---

### F7. HTF Nudge Hard Cap 🟠 IMPORTANT

**Problem:** HTF adjustment is `prob * (1 ± htf_weight * 0.3)`. With `htf_weight = 0.4` (within optimizer range), the adjustment can reach `±0.12`, shifting a `0.60` probability to `0.72` or `0.48`. This can override a near-threshold signal entirely, making the confidence threshold parameter meaningless.

**Changes required:**

**File: `phase2_adaptive_engine.py` — in `ga_fitness()` and in `get_signal()` in `live.py`** (both places where HTF nudge is applied)

Reduce the scale factor and apply a hard clip:
```python
# Change: 0.3 → 0.2 (reduce maximum nudge magnitude)
raw_adjustment = htf_weight * 0.2 * htf_direction   # htf_direction = +1 or -1

# Hard cap: HTF cannot move probability by more than ±0.05 in absolute terms
raw_adjustment = np.clip(raw_adjustment, -0.05, +0.05)

adjusted_prob = base_prob + raw_adjustment
adjusted_prob = np.clip(adjusted_prob, 0.0, 1.0)   # always keep in valid range
```

**Why ±0.05 cap:** With a typical confidence threshold of ~0.67, a maximum nudge of 0.05 means HTF can move a borderline 0.65 signal to 0.70 (pass) or a borderline 0.69 signal to 0.64 (skip). This is meaningful context — not a decision override.

**Verification (`[SIGNAL]` log in live):**
```
htf_adj=+0.031 ← always in range (-0.05, +0.05)
```
⚠️ Red flag: `htf_adj > 0.06` → clip not applied; check both `ga_fitness()` and `get_signal()` were both updated

---

### F8. Trade Frequency Sanity Constraint in GA 🟠 IMPORTANT

**Problem:** GA can select strategies with < 1.5 trades/day (not practically deployable; kill switch has no data) or > 10 trades/day (overtrading; signal quality degrades). Currently only MIN_CREDIBLE (100 total trades) guards this, which does not enforce a per-day rate.

**Changes required:**

**File: `phase2_adaptive_engine.py` — inside `ga_fitness()`, after the trade loop, alongside other penalty calculations**

```python
# Trade frequency penalty
n_days = max(n_days_span, 1)
trades_per_day = n_trades / n_days

if trades_per_day < 1.5:
    # Too infrequent: kill switch cannot accumulate data; ER estimate unreliable
    tpd_penalty = 0.70
elif trades_per_day > 10.0:
    # Too frequent: signal density too high; likely low-quality threshold
    tpd_penalty = 0.70
elif trades_per_day > 7.0:
    # Approaching overtrading: soft warning
    tpd_penalty = 0.90
else:
    tpd_penalty = 1.0

fitness *= tpd_penalty
```

**Note:** `n_days_span` is already computed for the daily Sharpe calculation (Section L1-8) — reuse it. Do not recompute.

**Verification (`[GA]` + `[BASELINES]` logs):**
```
[BASELINES] expected_trades_per_day=3.40   ← always in 1.5–10 range
```
⚠️ Red flag: `expected_trades_per_day < 1.5` → constraint not applied; or `n_days_span` computing incorrectly

---

### F9. Execution Hard-Fail Validation After GA 🟠 IMPORTANT

**Problem:** `[EXECUTION]` summary is currently informational. If spread is near-zero, the spread model is not working but training proceeds and produces a model trained on a friction-free world.

**Changes required:**

**File: `train.py` — immediately after the `[EXECUTION]` log is emitted (after GA completes)**

```python
# Hard validation of execution realism before accepting GA result
exec_stats = _get_execution_stats(best_genome_result)   # dict returned from ga_fitness internals

if exec_stats["avg_spread"] < 1.0:
    raise RuntimeError(
        f"[EXECUTION] ABORT — avg_spread={exec_stats['avg_spread']:.2f} pts. "
        f"Spread model not active. Check SPREAD_BASE_PTS in .env."
    )

if exec_stats["fill_rate"] > 0.97:
    raise RuntimeError(
        f"[EXECUTION] ABORT — fill_rate={exec_stats['fill_rate']:.1%}. "
        f"Execution model not filtering. Check EXEC_DELAY_PROB and MT5_DEVIATION_PTS."
    )

if exec_stats["avg_slippage"] < 0.3:
    raise RuntimeError(
        f"[EXECUTION] ABORT — avg_slippage={exec_stats['avg_slippage']:.2f} pts. "
        f"Slippage model not active."
    )
```

**Implementation note:** `ga_fitness()` currently does not return a dict of execution stats — it returns the fitness scalar to DEAP. Two options:
- **Option A (preferred):** After GA completes, re-run `ga_fitness()` on the best genome once more with a `diagnostic=True` flag that returns a full stats dict instead of just the scalar. Re-evaluation of a single genome is fast (~1ms).
- **Option B:** Add a module-level dict (e.g., `_last_exec_stats = {}`) that `ga_fitness()` populates on each call. After GA, `_last_exec_stats` holds the stats from the best genome's last evaluation.

**Verification:** If spread model is broken, training now stops with a clear error instead of silently producing an overfit model.

---

### F10. Ensemble Redundancy Penalty 🟡 MEDIUM

**Problem:** If XGB and RF OOS predictions are highly correlated (`> 0.92`), the ensemble average `(p_xgb + p_rf) / 2` adds almost no value over a single model, but the pipeline presents it as a two-model ensemble. The fitness score should reflect this.

**Changes required:**

This is a training-time penalty, not a `ga_fitness()` penalty. It applies to the walk-forward evaluation score.

**File: `train.py` — in the section after `[ENSEMBLE]` correlation is computed, before the strategy is accepted for robustness gating**

```python
corr = float(np.corrcoef(p_xgb_oos, p_rf_oos)[0, 1])

if corr > 0.92:
    log.warning(
        f"[ENSEMBLE] HIGH CORRELATION ({corr:.3f}) — models nearly identical. "
        f"Ensemble Sharpe will be discounted by 15%."
    )
    # Apply discount to the OOS Sharpe used for strategy ranking
    oos_sharpe_adjusted = oos_sharpe * 0.85
else:
    oos_sharpe_adjusted = oos_sharpe

# Use oos_sharpe_adjusted downstream for robustness gating and strategy comparison
```

**Note:** This does not change the deployed model (both XGB and RF are still used). It only penalises strategies discovered during training where the two models happened to converge to the same solution. Strategies with diverse models get a small advantage in selection.

**Verification (`[ENSEMBLE]` log):**
```
[ENSEMBLE] Diversity: GOOD (correlation=0.78)   ← no discount applied
[ENSEMBLE] HIGH CORRELATION (0.94) — discounted by 15%   ← discount applied
```

---

### BONUS: Net Edge Per Trade Metric — `[EDGE]` 🟢 LOW EFFORT / HIGH INSIGHT

**Add everywhere:** This single metric summarises whether the strategy has any edge after all friction is accounted for.

```python
# After the trade loop in ga_fitness()
avg_sl_pts = np.mean([sl_atr * atr[i] for i in trade_indices])   # or use sl_dist array
net_edge   = mean_r - (avg_total_cost / avg_sl_pts)

log.debug(f"[EDGE] {symbol}/{tf}m net_edge_per_trade={net_edge:.3f}R")
```

**Add to `[EXECUTION]` summary log in `train.py`:**
```
[EXECUTION] ...
  net_edge_per_trade = 0.18R   ← target > 0.10R
```

**Thresholds:**
| Value | Meaning |
|-------|---------|
| `> 0.15R` | Strong net edge — viable live |
| `0.10–0.15R` | Marginal — acceptable with stable execution |
| `0.05–0.10R` | Weak — likely noise after live friction |
| `< 0.05R` | No edge — do not deploy |

---

## 2. `.env` CHANGES REQUIRED

All changes to make in `.env` before the next training run:

```ini
# ── EXECUTION LATENCY MODEL (tightened) ──────────────────────────────────────
EXEC_DELAY_PROB=0.40        # was 0.30
MT5_DEVIATION_PTS=15        # was 30

# ── TIMEFRAMES ────────────────────────────────────────────────────────────────
TIMEFRAMES=3,5              # remove 1m from optimization
```

---

## 3. SUMMARY TABLE — ALL ITEMS

| ID | Priority | File(s) | Change type | Target metric |
|----|----------|---------|-------------|---------------|
| F1 | 🚨 Critical | `.env`, `phase2_adaptive_engine.py` | Tighten deviation, add ATR-move filter | `fill_rate` 85–92%, `gap_skip_rate` 2–6% |
| F2 | 🚨 Critical | `phase2_adaptive_engine.py` | Hard SL floor + cost ratio penalty | `sl_atr` always ≥ 2.0 |
| F3 | 🚨 Critical | `phase2_adaptive_engine.py` | Execution regime penalties in fitness | GA selects realistic fill regimes |
| F4 | 🚨 High | `train.py`, `.env` | Remove 1m from optimization loop | No OOM; no parity failure on 1m |
| F5 | 🟠 Important | `phase2_adaptive_engine.py` | Spread-adjusted entry in labels | TP rates drop 1–3%; labels more realistic |
| F6 | 🟠 Important | `train.py` | Calibration guard + min OOS check | Degraded calibrator skipped automatically |
| F7 | 🟠 Important | `phase2_adaptive_engine.py`, `live.py` | HTF nudge cap ±0.05, scale 0.2 | `htf_adj` always in (−0.05, +0.05) |
| F8 | 🟠 Important | `phase2_adaptive_engine.py` | Trade frequency penalty in fitness | `expected_trades_per_day` always 1.5–10 |
| F9 | 🟠 Important | `train.py` | Hard-fail validation after GA | Training aborts if spread model broken |
| F10 | 🟡 Medium | `train.py` | Ensemble correlation discount | High-corr ensembles discounted 15% in ranking |
| BONUS | 🟢 Low effort | `phase2_adaptive_engine.py`, `train.py` | `[EDGE]` net_edge_per_trade metric | Visible in every `[EXECUTION]` log |

---

## 4. IMPLEMENTATION ORDER FOR THE AGENT

Implement in this exact sequence to avoid dependency issues:

1. **F4 first** — drop 1m from optimization. This eliminates OOM, parity failure, and the weakest TF from polluting results. Fastest win.
2. **F1** — tighten `.env` values and add ATR-move filter in `ga_fitness()`. This is the biggest single change to execution realism.
3. **F2** — add hard SL floor and update `GA_BOUNDS`. Prevents the optimizer from exploring sub-2.0 SL territory entirely.
4. **F3** — add execution regime penalties using counters already tracked in F1.
5. **F7** — HTF nudge cap in both `ga_fitness()` and `live.py`. Quick, affects both paths.
6. **F8** — trade frequency penalty. Reuses `n_days_span` already computed.
7. **F9** — execution hard-fail checks. Requires deciding Option A vs B for stats return.
8. **F6** — calibration guard. Add after existing `brier_after` computation.
9. **F5** — spread-aware labels. Most careful change; test that TP rates shift slightly downward.
10. **F10** — ensemble discount. Cosmetic change to ranking; implement last.
11. **BONUS** — `[EDGE]` metric. Add at end; no logic changes.

---

## 5. VERIFICATION CHECKLIST FOR NEXT TRAINING RUN

After implementing all items, verify these in the training logs before accepting results:

### Training startup:
- [ ] Log shows `TIMEFRAMES = [3, 5]` — no 1m
- [ ] `[DATA]` emitted for 3m and 5m only — no OOM
- [ ] `[PARITY]` passes for 3m and 5m

### Feature / label sanity:
- [ ] `[FEATURES]` frozen list: `vwap_dist_atr`, `atr14`, `volume_z` should NOT be frozen (these are normalized correctly). Price-level features (`vwap`, `ema*`) will have large p99 — this is expected and should not block training.
- [ ] `[LABELS]` TP rate range slightly lower than previous run (spread-aware entry effect)
- [ ] No degenerate label columns with `be=0` at `tp_mult=1` or `2` (those should be 20%+)

### Optimization:
- [ ] `[GA]` best genome: `sl_atr ≥ 2.0` — confirmed
- [ ] `[EXECUTION]` after GA: `avg_spread ≥ 1.5 pts`, `fill_rate 85–92%`, `gap_skip_rate 2–6%`
- [ ] `[EXECUTION]` hard-fail checks pass without RuntimeError
- [ ] `[EDGE]` `net_edge_per_trade > 0.10R`
- [ ] `[ENSEMBLE]` correlation `< 0.92` (no discount applied)
- [ ] `[STABILITY]` confidence curve is smooth (no spike)
- [ ] `[BASELINES]` `expected_trades_per_day` in range 1.5–10

### Calibration:
- [ ] `[CALIBRATION]` `brier_after < brier_before` — no GUARD message
- [ ] `[PRED]` `std` in 0.06–0.12 (not collapsed, not overconfident)
- [ ] `[CAL-STABILITY]` no `WEAK` tertile

### Red flags — stop and investigate if seen:
- `[EXECUTION]` `fill_rate > 95%` → F1 ATR-move filter not applied
- `[GA]` `sl_atr < 2.0` → F2 hard floor not reached; check `ga_fitness()` return path
- `[EXECUTION]` `avg_spread < 1.0` → spread model broken; check `.env` and SPREAD_BASE_PTS read
- `[PRED]` `std < 0.04` → model collapsed; check feature set quality for 3m/5m
- `[CALIBRATION]` GUARD message → calibration failed; model will use raw probabilities
- `[ENSEMBLE]` HIGH CORRELATION → discount applied; may want to tune model hyperparameters for diversity
- `[BASELINES]` `expected_trades_per_day < 1.5` → frequency constraint not applied

---

## 6. KNOWN ISSUES FROM FIRST RUN — NOT IN SCOPE FOR THIS ITERATION

These were observed but are deferred:

| Issue | Deferred reason |
|-------|----------------|
| `thin_sessions=335` in `[DATA]` for 1m | 1m removed from optimization (F4); will re-evaluate when 1m re-enters |
| `is_london` parity failure | Timezone mismatch between `institutional_features.py` and `pipeline.py`; fix in separate session as it requires comparing both session-flag implementations |
| Duplicate log lines in training startup | Logger configured with multiple handlers; fix by ensuring `logging.basicConfig` is called once; low priority vs correctness issues |
| `[FEATURES]` false positives (frozen/exploding) for price-level features | The `std < 0.01` threshold flags normalized features; the `p99 > 1000` threshold flags raw price features. Both are false positives for US30. Fix thresholds: frozen → `std < 1e-6`; exploding → use `(p99 - p1) / (std + 1e-9) > 20` relative measure |
| `vp_poc` and `ORB` all-NaN on 240m | Insufficient bars for session-level groupby; acceptable for HTF context; suppress warning or skip VP/ORB features for TFs < 30 bars/session |

The `is_london` parity fix and `[FEATURES]` threshold fix should be implemented **before** going live, but do not block the second training run.
