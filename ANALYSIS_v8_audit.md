# v8 Implementation Audit Report
**Date:** 2026-04-19
**Audited against:** ANALYSIS_v8.md

---

## QUICK STATUS

| ID | Item | Status | Notes |
|----|------|--------|-------|
| F1 | Execution realism — fill rate fix | ✅ DONE | MT5_DEVIATION_PTS=15, EXEC_DELAY_PROB=0.40 in .env; ATR-move filter (`gap_pts > 0.4 * atr[i]`) at lines 1100–1104 of phase2. Counter names differ from spec (`_ts_delayed`, `_ts_n_signals`) but logic is equivalent |
| F2 | Minimum SL floor + GA_BOUNDS | ✅ DONE | Hard floor at lines 1057–1059; `GA_BOUNDS sl_atr = (2.0, 4.0)` |
| F3 | Execution regime penalties in fitness | ✅ DONE | fill_rate > 0.95 → ×0.80; fill_rate < 0.75 → ×0.85; gap_skip_rate < 0.01 → ×0.90 at lines 1208–1217 |
| F4 | Remove 1m from optimization | ✅ DONE | `entry_tf_options = [3, 5, 10, 15]` — 1m excluded |
| F4b | 30m also excluded (intentional) | ✅ DONE | Same list; comment explicitly says "30m excluded: too few trades/day" |
| F5 | Spread-aware entry in labels | ❌ NOT DONE | Labels still use raw `closes[i]` as entry — zero-cost assumption |
| F6 | Calibration robustness guard | ✅ DONE | If brier_after ≥ brier_before → calibrator .pkl deleted (live.py uses raw probs); OOS < 10k → warning |
| F7 | HTF nudge cap ±0.05, scale 0.2 | ✅ DONE | Clipped in `ga_fitness()` lines 1014–1018 AND in `get_signal()` lines 1588–1596; live.py imports `get_signal` so both paths use same logic |
| F8 | Trade frequency penalty | ✅ DONE | < 1.5 or > 10 trades/day → ×0.70; > 7/day → ×0.90 at lines 1219–1225 |
| F9 | Execution hard-fail validation | ✅ DONE | RuntimeError if avg_spread < 1.0 or fill_rate > 0.97 at train.py lines 1277–1288; uses `track_stats=True` re-evaluation of best genome |
| F10 | Ensemble redundancy penalty | ❌ NOT DONE | Correlation logged with `LOW`/`ADEQUATE` tag but no Sharpe discount applied when corr > 0.92 |
| BONUS | net_edge_per_trade metric | ❌ NOT DONE | No `net_edge` anywhere in codebase |

**Score: 8 / 11 items fully implemented.**

---

## DETAIL ON THE 3 MISSING ITEMS

### F5 — Spread-aware label entry (not done)
`engineer_features()` still uses `entry = closes_arr[i]` with no spread offset. TP/SL distances are computed from this perfect-entry price.

Impact: moderate. The spread effect is captured in `ga_fitness()` and `backtest_engine.py` during simulation — so the optimizer sees friction costs. However, the *labels* themselves (used to train XGBoost/RF) are optimistic about entry quality. A label marked `1` (TP hit) may have only barely reached TP from the real entry (`close + half_spread`). This slightly inflates the positive class quality the model learns.

Fix when ready: In `engineer_features()`, shift `tp_distance` up and `sl_distance` down by `SPREAD_BASE_PTS / 2.0` before computing barrier levels.

### F10 — Ensemble redundancy penalty (not done)
The `[ENSEMBLE]` block logs diversity as `ADEQUATE`/`LOW` but does not apply a Sharpe discount. The OOS Sharpe used for strategy ranking is unchanged regardless of ensemble correlation.

Impact: low in practice — if correlation is already < 0.92 consistently, the penalty would never trigger anyway. Implement if you observe HIGH CORRELATION warnings persisting after a training run.

Fix when ready: In `train.py`, after the `[ENSEMBLE]` block, multiply `oos_sharpe_for_ranking` by `0.85` when `corr > 0.92`.

### BONUS — net_edge_per_trade (not done)
No `net_edge` calculation exists anywhere. This is a diagnostic-only metric; missing it does not affect training correctness.

Fix when ready: After the trade loop in `ga_fitness()`:
```python
net_edge = mean_r - (np.mean(_ts_total_cost) / (sl_atr * np.mean(atr[valid_bars])))
```
Add to `[EXECUTION]` summary log in `train.py`.

---

## DEFERRED ISSUES — CURRENT STATE

These were not in scope for v8 but the audit checked their status anyway.

| Issue | Current state | Action needed |
|-------|--------------|---------------|
| `[FEATURES]` frozen threshold | `std < 0.01` but now has a `_small_ok` exclusion list that protects normalized features (pct returns, distances etc.) — false positives likely reduced | Verify `_small_ok` list covers all the frozen features seen in the first run: `vwap_dist`, `ret1`, `p_vs_21`, `r_lag*`, `htf_bullish`, etc. |
| `[FEATURES]` exploding threshold | Changed from `p99 > 1000` to `abs(p99) > 500_000` — price-level features (`vwap`, `ema*`) no longer false-positive | ✅ Effectively fixed; docstring still says `p99>1000` (cosmetic only) |
| `is_london` parity failure | `phase2_adaptive_engine.py` uses DST-aware London minutes (correct). `phase1_mt5_data.py` still uses UTC hour 8–16 (different rule) — mismatch if that file feeds the same pipeline | Check if `phase1_mt5_data.py` is in the live or training data path; if yes, align its `is_london` definition |
| Duplicate log lines | `phase2_adaptive_engine.py` calls `basicConfig` at import; `train.py` calls it again with `force=True` — consolidates handlers but still fires twice during import chain | Low priority; `force=True` means the second call wins and handlers are not doubled. Lines appear duplicate because both modules log at startup before `force=True` runs |

---

## ACTIVE TIMEFRAMES

```
entry_tf_options = [3, 5, 10, 15]
```

- **Excluded by design:** 1m (2.1M bars → no institutional features, OOM risk), 30m (comment: "too few trades/day for kill switch and calibration reliability")
- **Active:** 3m, 5m, 10m, 15m — all have institutional features (< 750k bar threshold), all will have full `[PARITY]`, `[LEAKAGE]`, and `[LABELS]` checks

---

## BEFORE NEXT TRAINING RUN — REMAINING CHECKLIST

```
Required before training:
[ ] None — all blocking items (F1–F4, F6–F9) are implemented

Recommended before deploying to live:
[ ] F5: Spread-aware label entry (moderate impact on label quality)
[ ] is_london: Align phase1_mt5_data.py definition with phase2 DST-aware version
[ ] Verify _small_ok list in _log_feature_sanity() covers all normalized features
     to eliminate false FROZEN warnings

Optional / low priority:
[ ] F10: Ensemble Sharpe discount
[ ] BONUS: net_edge_per_trade metric
[ ] Fix docstring in _log_feature_sanity() (still says p99>1000)
```

---

## EXPECTED LOG IMPROVEMENTS VS FIRST RUN

After the v8 changes, the second training run should show:

| Log tag | First run | Expected now |
|---------|-----------|--------------|
| `[EXECUTION] fill_rate` | `> 95%` (problem) | `85%–92%` |
| `[EXECUTION] gap_skip_rate` | `< 1%` (problem) | `2%–6%` |
| `[GA] sl_atr` | possibly `< 2.0` | always `≥ 2.0` |
| `[BASELINES] expected_trades_per_day` | potentially `< 1.5` or `> 10` | `1.5–10` |
| `[GA] best_score` (Sharpe) | inflated by fake fills | ~10–25% lower, more realistic |
| `[FEATURES] FROZEN` warning | 25 features flagged | reduced by `_small_ok` exclusions |
| `[FEATURES] EXPLODING` warning | 6 price-level features | likely none (threshold now 500k) |
| `[DATA]` 1m block | OOM on parity check | absent (1m removed from loop) |
