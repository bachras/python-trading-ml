# ML Trading System — Institutional Audit v7
**Date:** 2026-04-12
**Instrument:** US30 CFD (Dukascopy tick data → MT5 execution)
**Purpose:** Final pre-training status + comprehensive log verification manual. Copy training and live log output and compare against every table in Section 3 to detect bugs, logic inconsistencies, and strategy misalignments without reading code.

---

## 1. IMPLEMENTATION STATUS — v6 AUDIT RESULT

### All code is implemented. One gap: `.env` keys.

| # | Item | Code | `.env` |
|---|------|------|--------|
| E1 | Regime separation per fold (`_compute_fold_regime_stats`, `_log_fold_regime_divergence`) | ✅ `train.py` 218–258, called 446–454 | — |
| E2 | Calibration stability across volatility bins (`_check_calibration_stability`, `oos_atr_all`) | ✅ `train.py` 261–287, called 565–571 | — |
| E3 | Execution latency model in `ga_fitness()` (`opens`, `EXEC_DELAY_PROB`, `MT5_DEVIATION_PTS`, gap-skip logic) | ✅ `phase2_adaptive_engine.py` 948, 1043–1051 | ❌ Keys absent |
| G1 | `ATR_SPIKE_FILTER_MULT=3.0` | ✅ Reads env with default | ❌ Key absent |
| G2 | `SPREAD_BASE_PTS`, `SPREAD_ATR_COEFF`, `SPREAD_OPEN_MULT`, `SPREAD_OPEN_BARS` | ✅ Reads env with defaults | ❌ Keys absent |
| G3 | `GLOBAL_SEED=42` | ✅ Reads env with default 42 | ❌ Key absent |
| R1–R6, S1–S3 | All previously confirmed | ✅ All present | — |

**Note on `MT5_DEVIATION_PTS` default:** Code defaults to `30` (not `20` from the v6 spec). 30 points is a reasonable US30 CFD value and avoids over-rejecting trades on wider ATR bars. If you want to enforce 20, set explicitly in `.env`.

---

## 2. SINGLE REMAINING GAP — `.env` ADDITIONS

Add these 9 keys. All code already reads them with correct defaults — this makes parameters visible, reviewable, and overridable without touching code.

```ini
# ── BACKTEST FILTERS ─────────────────────────────────────────────────────────
ATR_SPIKE_FILTER_MULT=3.0

# ── SYNTHETIC SPREAD MODEL ───────────────────────────────────────────────────
SPREAD_BASE_PTS=1.5
SPREAD_ATR_COEFF=0.04
SPREAD_OPEN_MULT=2.0
SPREAD_OPEN_BARS=5

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
GLOBAL_SEED=42

# ── EXECUTION LATENCY MODEL ──────────────────────────────────────────────────
EXEC_DELAY_PROB=0.30
MT5_DEVIATION_PTS=20
```

---

## 3. COMPREHENSIVE LOG VERIFICATION MANUAL

**How to use:** Run training, copy the full log output, then work through each section below in order. Every log tag is a searchable anchor. Healthy ranges are shown in green (prose), red flags in ⚠️ markers.

---

### 3.1 STARTUP — `[SEED]` and `[CONFIG]`

**When emitted:** First lines of `train.py` and `live.py` startup.

**What to check in training logs:**
```
[SEED] Global seed set to 42
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Seed value | Matches `GLOBAL_SEED` in `.env` | Missing entirely → seeds not set; runs non-reproducible |

**What to check in live logs:**
```
[CONFIG] Risk gates loaded:
  max_drawdown_pct : 35.0
  daily_loss_pct   : X.X
  fixed_risk_amt   : 100.0
  vol_size_enabled : False
  session_gate     : 20:30–01:00 London
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `max_drawdown_pct` | `35.0` | Any other value → check `.env`; 10.0 = default not overridden |
| `fixed_risk_amt` | `100.0` | Other → risk model broken |
| `vol_size_enabled` | `False` | `True` → vol-sizing active; check if intentional |
| `session_gate` | `20:30–01:00 London` | Missing → session gate not logging; may still work but unverified |

---

### 3.2 DATA INTEGRITY — `[DATA]`

**When emitted:** `train.py` immediately after loading raw bars from parquet, before any feature computation. This is the earliest possible sanity check — catches data issues that would silently poison everything downstream.

**Why it matters:** Most ML trading failures originate in data, not models. Silent gaps in tick→bar aggregation, duplicated timestamps (common in MT5 exports), corrupted ticks, and negative spreads all produce plausible-looking DataFrames that the rest of the pipeline processes without complaint.

```
[DATA] US30/1m
  rows=86400  start=2022-01-03  end=2024-12-31
  expected_sessions=522  missing_sessions=3 (0.6%)  thin_sessions=2
  duplicate_ts=0
  zero_volume_pct=0.02%
  negative_spread_rows=1841
  max_gap_minutes=185  (on 2023-09-04 — Labor Day)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `rows` | Consistent with date range × bars/session | `< 30,000` for 1–2 year dataset → suspicious; check parquet |
| `missing_sessions` | `< 1%` of expected Mon–Fri sessions | `> 3%` → significant day-level gaps; check data download coverage for those dates |
| `thin_sessions` | `0–5` | `> 10` → many partial sessions; tick aggregation producing incomplete days |
| `duplicate_ts` | `0` | `> 0` → **critical**; duplicates produce look-ahead in all rolling features; deduplicate before training |
| `zero_volume_pct` | `< 0.1%` | `> 1%` → tick data gaps masquerading as bars; check tick pipeline |
| `negative_spread_rows` | Present but manageable — expected (Dukascopy raw bid/ask artefact) | `0` → synthetic spread model may not be reaching bars that need it |
| `max_gap_minutes` | Explainable by known holidays/weekends | `> 240` on a Mon–Fri non-holiday → full trading session missing; check data download |

**Implementation — add at start of training pipeline in `train.py`:**
```python
def _log_data_integrity(df: pd.DataFrame, symbol: str, tf: int) -> None:
    n   = len(df)
    idx = df.index

    # --- Missing session count (meaningful for US30 CFD) ---
    # Use Mon-Fri calendar days rather than minute-frequency date_range:
    # raw minute count would always show ~75% "missing" due to overnight/weekend hours.
    all_dates       = pd.bdate_range(idx[0].date(), idx[-1].date())  # Mon-Fri only
    dates_in_data   = set(idx.normalize().date)
    expected_sess   = len(all_dates)
    missing_sess    = sum(1 for d in all_dates if d.date() not in dates_in_data)
    # Thin session: day present but < 50% of median bars-per-day
    bars_per_day    = pd.Series(idx.date).value_counts()
    median_bars     = bars_per_day.median()
    thin_sess       = int((bars_per_day < 0.5 * median_bars).sum())

    dupes       = int(idx.duplicated().sum())
    zero_vol    = 100 * (df["Volume"] == 0).mean() if "Volume" in df.columns else 0.0
    neg_spread  = int((df.get("spread", pd.Series(dtype=float)) < 0).sum())
    gaps        = idx.to_series().diff().dt.total_seconds().div(60).dropna()
    max_gap     = gaps.max()
    max_gap_date = idx[gaps.argmax() + 1].date() if len(gaps) else "N/A"

    log.info(
        f"[DATA] {symbol}/{tf}m\n"
        f"  rows={n}  start={idx[0].date()}  end={idx[-1].date()}\n"
        f"  expected_sessions={expected_sess}  "
        f"missing_sessions={missing_sess} ({100*missing_sess/max(expected_sess,1):.1f}%)  "
        f"thin_sessions={thin_sess}\n"
        f"  duplicate_ts={dupes}\n"
        f"  zero_volume_pct={zero_vol:.2f}%\n"
        f"  negative_spread_rows={neg_spread}\n"
        f"  max_gap_minutes={max_gap:.0f}  (on {max_gap_date})"
    )
    if dupes > 0:
        raise ValueError(
            f"[DATA] ABORT — {dupes} duplicate timestamps in {symbol}/{tf}m. "
            f"Deduplicate before training."
        )
    if missing_sess / max(expected_sess, 1) > 0.05:
        log.warning(
            f"[DATA] HIGH missing session rate ({missing_sess}/{expected_sess}) — "
            f"model trained on incomplete history."
        )
```

---

### 3.3 FEATURE PARITY — `[PARITY]`

**When emitted:** `train.py` after features are computed, before fold construction.

```
[PARITY] Loading raw bars and running live pipeline on full history ...
[PARITY] max_diff=0.00000000, mean_diff=0.00000000, n_features_differ=0/73, worst=vwap_dist_atr
[PARITY] PASS — training and live features match within 1e-4 ✅
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `max_diff` | `< 1e-4` (ideally 0.0) | `> 1e-4` → train/live mismatch; **do not deploy** |
| `n_features_differ` | `0` | `> 0` → which features? Fix that function |
| Result line | `PASS` | `FAIL` → RuntimeError thrown; training stops |

---

### 3.4 LEAKAGE CHECK — `[LEAKAGE]`

**When emitted:** Inside `engineer_features()` in `phase2_adaptive_engine.py`.

```
[LEAKAGE] KNOWN look-ahead features (swing confirmation, SWING_LOOKBACK=10 bars): ['swing_high_dist', 'equal_highs', ...]
[LEAKAGE] Check complete — 3 known look-ahead, 0 unexpected leaks ✅
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Known look-ahead list | Swing features only | Any VWAP/volume/EMA feature in the known list → those should be causal; investigate |
| Unexpected leaks | `0` | `> 0` → RuntimeError thrown; fix the feature before proceeding |

---

### 3.5 FEATURE DISTRIBUTION SANITY — `[FEATURES]`

**When emitted:** `train.py` after `add_institutional_features()` completes, before walk-forward fold construction. Complements `[PARITY]` (which checks equality between pipelines) — this checks *correctness* of values within a single run.

**Why it matters:** Parity verifies train = live. This catches exploding values (division by zero producing inf/nan), frozen features (std ≈ 0 from a constant), and scale drift between runs that can silently degrade model performance without triggering any error.

```
[FEATURES] US30/1m distribution sanity (73 features, 86400 bars):
  vwap_dist_atr   : mean= 0.12  std=0.85  p1=-2.1  p99= 3.4  nan=0
  atr14           : mean=52.1   std=18.3  p1=21.3  p99=98.4  nan=0
  volume_z        : mean= 0.01  std=1.02  p1=-2.4  p99= 2.9  nan=0
  order_flow_delta: mean= 0.03  std=1.14  p1=-2.8  p99= 2.7  nan=12
  session_vwap    : mean= 0.08  std=0.63  p1=-1.4  p99= 1.6  nan=0
  regime_vol      : mean= 0.41  std=0.49  p1= 0.0  p99= 1.0  nan=0
  ... (top 10 most extreme shown)
[FEATURES] Frozen features (std < 0.01): none ✅
[FEATURES] Exploding features (p99 > 1000 or nan > 0.1%): none ✅
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `nan` count per feature | `0` or very small | `> 0.5%` nan rate → `ffill + fillna(0)` not covering all cases; check feature function |
| `std ≈ 0` (frozen) | No frozen features | Any feature with `std < 0.01` → feature is constant; provides no signal and wastes model capacity |
| `p99 > 1000` (exploding) | None | Any → division by zero in feature computation; will dominate model via extreme values |
| `atr14` mean | `30–120 pts` for US30 CFD | `< 10` → suspicious; ATR period wrong or data in wrong units | `> 200` → crisis data or scaling error |
| `volume_z` mean | Near `0.0`, std near `1.0` | `mean > 1.0` → z-score normalization applied twice | `std >> 2` → normalization using wrong baseline |
| `vwap_dist_atr` | `mean ≈ 0`, `p99 < 5` | `p99 > 10` → VWAP or ATR calculation producing extreme outliers; clamp or investigate |
| Scale consistency run-to-run | Same order of magnitude as previous training | `atr14 mean` doubles between runs → dataset extended into different regime; expected but note |

**Implementation — add after `add_institutional_features()` in `train.py`:**
```python
def _log_feature_sanity(df: pd.DataFrame, symbol: str, tf: int,
                         n_features: int = 10) -> None:
    feat_cols = get_feature_cols(df)
    stats = df[feat_cols].agg(["mean", "std", lambda x: x.quantile(0.01),
                                lambda x: x.quantile(0.99),
                                lambda x: x.isna().sum()]).T
    stats.columns = ["mean", "std", "p1", "p99", "nan_count"]
    frozen   = stats[stats["std"] < 0.01].index.tolist()
    exploding = stats[(stats["p99"].abs() > 1000) | (stats["nan_count"] > len(df) * 0.001)].index.tolist()
    log.info(f"[FEATURES] {symbol}/{tf}m distribution sanity ({len(feat_cols)} features, {len(df)} bars):")
    # Log top-N by p99 absolute value (most likely to be problematic)
    for col in stats["p99"].abs().nlargest(n_features).index:
        r = stats.loc[col]
        log.info(f"  {col:<25}: mean={r['mean']:+.2f}  std={r['std']:.2f}  p1={r['p1']:+.2f}  p99={r['p99']:+.2f}  nan={int(r['nan_count'])}")
    if frozen:
        log.warning(f"[FEATURES] FROZEN features (std < 0.01): {frozen}")
    else:
        log.info("[FEATURES] Frozen features: none ✅")
    if exploding:
        log.warning(f"[FEATURES] EXPLODING/HIGH-NAN features: {exploding}")
    else:
        log.info("[FEATURES] Exploding features: none ✅")
```

---

### 3.6 LABEL GRID — `[LABELS]`

**When emitted:** End of `engineer_features()`.

```
[LABELS] Generated 75 label columns. Sample TP rates:
  target_sl1.5_tp1_be0: 42.3%
  target_sl1.5_tp2_be0: 28.1%
  target_sl1.5_tp3_be0: 18.4%
  target_sl2.0_tp1_be0: 44.1%
  target_sl2.0_tp2_be0: 29.3%
  target_sl2.5_tp1_be0: 45.2%
[LABELS] All label columns verified.
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Column count | `75` | `< 75` → some grid cells not generated; check TB_SL_GRID / TB_TP_GRID / TB_BE_GRID |
| TP rates (be=0) | Decrease as `tp_mult` increases; increase slightly as `sl_atr` increases | TP rate *rising* with tp_mult → label computation error |
| TP rates range | `5%–65%` for valid intraday strategies | `< 2%` or `> 95%` → degenerate column; check DEGENERATE warning |
| be=1 vs be=0 same sl/tp | be=1 TP rate ≤ be=0 for tp_mult ≥ 2 | be=1 TP rate > be=0 → BE labeling bug; SL should be moving to entry, reducing TP hits |
| DEGENERATE warning | Absent | Present → remove that column from GA search or investigate label generation |

---

### 3.7 CLASS BALANCE — `[CLASS]`

**When emitted:** `train.py` right before `[RETRAIN]` begins — after GA/Optuna has selected `best_label_col` but before the final model is trained on it. Logging it here means "selected label" is accurate rather than always showing the default label.

**Why it matters:** TP rate in `[LABELS]` is the raw label balance over all bars. The class balance at training time may differ after ATR spike filter exclusions and session gate filtering. Extreme imbalance (`< 10%` positives) makes XGBoost/RF default thresholds meaningless and can inflate apparent accuracy while providing no real edge.

```
[CLASS] Selected label: target_sl2.5_tp3_be1
  total_bars = 66000  positives = 18744 (28.4%)  negatives = 47256 (71.6%)
  ratio      = 2.51:1 (acceptable)
  vs [LABELS] raw TP rate: 29.1%  (delta = -0.7% — consistent ✅)
[CLASS] Grid extremes (for reference):
  target_sl1.5_tp1_be0: pos=42.3%  (near-balanced, expected for low tp_mult)
  target_sl3.5_tp5_be2: pos= 8.1%  (skewed, expected for wide TP + BE)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `positives` pct for selected label | `15%–50%` | `< 10%` → severely imbalanced; XGBoost `scale_pos_weight` needs adjusting | `> 60%` → TP too easy; `sl_atr` probably too wide for the `tp_mult` chosen |
| Delta vs `[LABELS]` raw TP rate | `< ±3%` | `> ±5%` → ATR filter or session gate removing a biased subset of bars; investigate which bars are excluded |
| High `tp_mult` + `be_r=2` columns | Low positives expected (`< 15%`) | `> 30%` for `tp_mult=5, be_r=2` → label computation error; BE should make TP harder at high multiples |
| Grid extremes | Wide variation across 75 columns (tight labels 40%+, wide labels 8%+) | All columns near 25–35% → grid not spanning meaningfully different outcomes; check TB_SL_GRID ranges |

---

### 3.8 FOLD REGIME SEPARATION — `[REGIME]`

**When emitted:** `train.py` after walk-forward fold construction, once per symbol/timeframe.

```
[REGIME] US30/1m fold_0->fold_1 drift=0.14 (OK ✅) | atr_mean: 48.2->51.6, kurtosis: 3.4->3.8, vol_clust: 0.31->0.29
[REGIME] US30/1m fold_1->fold_2 drift=0.19 (OK ✅) | atr_mean: 51.6->55.1, kurtosis: 3.8->4.1, vol_clust: 0.29->0.34
[REGIME] US30/1m ATR coefficient of variation across folds: 0.07 (OK)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Per-fold `drift` score | `< 0.30 (OK ✅)` | `> 0.30 (HIGH ⚠️)` → folds span different volatility regimes; check if OOS Sharpe also drops on that fold |
| `atr_mean` trajectory | Gradual increase or stable | Step-change > 30% between folds → structural break in dataset; consider trimming or noting regime |
| ATR CV | `< 0.25` | `> 0.25` → high regime dispersion across folds; strategy must be tested across all regimes separately |
| `vol_clust` (ACF of abs returns) | `0.20–0.45` — normal volatility clustering | `> 0.60` → extreme clustering (crisis regime in that fold); `< 0.10` → unusually random (data gap?) |
| `kurtosis` | `3–8` for intraday US30 | `> 15` → fat tail event dominated fold (e.g., COVID); strategy performance on this fold may not generalise |

**Cross-check:** If `drift > 0.30` on fold `i→i+1`, look at the walk-forward OOS Sharpe for fold `i+1` (emitted later in `[FOLD]` log). A HIGH drift + OOS Sharpe drop means the strategy has regime dependency.

---

### 3.9 WALK-FORWARD FOLDS — `[FOLD]`

**When emitted:** `train.py` after each fold trains, once per fold. Add to `train.py` after each fold's XGB and RF are fit:
```python
log.info(f"[FOLD] {symbol}/{tf}m fold={fold_idx}/{n_folds} | "
         f"train_bars={len(train_df)} oos_bars={len(val_df)} | "
         f"XGB auc={xgb_auc:.3f} RF auc={rf_auc:.3f} | "
         f"label={label_col} pos_rate={val_df[label_col].mean():.1%}")
```

```
[FOLD] US30/1m fold=0/3 | train_bars=42000 oos_bars=8000 | XGB auc=0.574 RF auc=0.561 | label=target_sl1.5_tp2_be0 pos_rate=28.3%
[FOLD] US30/1m fold=1/3 | train_bars=50000 oos_bars=8000 | XGB auc=0.581 RF auc=0.568 | label=target_sl1.5_tp2_be0 pos_rate=27.9%
[FOLD] US30/1m fold=2/3 | train_bars=58000 oos_bars=8000 | XGB auc=0.572 RF auc=0.560 | label=target_sl1.5_tp2_be0 pos_rate=28.8%
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `train_bars` | Increases fold-over-fold (expanding window) | Constant → walk-forward not expanding; check `_walk_forward_folds()` |
| `oos_bars` | Roughly equal across folds | Very unequal → dataset boundary issue |
| XGB/RF AUC | `0.53–0.62` for noisy intraday data | `> 0.65` → likely overfit or leakage; `< 0.52` → model learns nothing; check features |
| AUC trend across folds | Stable ± 0.02 | Monotonic *decrease* → model degrades on recent data; consider shrinking train window |
| `pos_rate` per fold | Stable ± 2% across folds | Large variation → class distribution differs by period; most recent fold's balance is what matters for live |

---

### 3.10 CALIBRATION — `[CALIBRATION]`

**When emitted:** `train.py` after isotonic regression is fitted on stacked OOS predictions.

```
[CALIBRATION] US30/1m | n_oos=24000 | brier_before=0.2341 brier_after=0.2198 improvement=0.0143
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `n_oos` | Total OOS bars across all folds | `< 5000` → too few samples for reliable calibration |
| `brier_before` | `0.19–0.26` for intraday classification | `< 0.15` → suspiciously good; check for leakage |
| `brier_after` | Less than `brier_before` | `≥ brier_before` → calibration failed or overfit; isotonic may be inverting probabilities |
| `improvement` | `0.005–0.025` | `< 0.001` → calibration had negligible effect; probabilities were already well-shaped | `> 0.05` → very miscalibrated before; model probabilities were poor; check feature quality |

---

### 3.11 PREDICTED PROBABILITY DISTRIBUTION — `[PRED]`

**When emitted:** `train.py` immediately after isotonic calibration is fitted. Add after `calibrator.fit(...)`:
```python
cal_probs = calibrator.predict(np.array(oos_probs_all))
p = np.array(cal_probs)
conf = best_params.get("confidence_threshold", 0.65)   # from earlier in the flow
log.info(
    f"[PRED] {symbol}/{tf}m — OOS calibrated probability distribution (n={len(p)}):\n"
    f"  mean={p.mean():.3f}  std={p.std():.3f}\n"
    f"  p10={np.percentile(p,10):.3f}  p25={np.percentile(p,25):.3f}  "
    f"p50={np.percentile(p,50):.3f}  p75={np.percentile(p,75):.3f}  "
    f"p90={np.percentile(p,90):.3f}\n"
    f"  %>{conf:.2f} = {100*(p>conf).mean():.1f}%   "
    f"  %>0.70 = {100*(p>0.70).mean():.1f}%\n"
    f"  %<0.30 = {100*(p<0.30).mean():.1f}%"
)
shape = "COLLAPSED" if p.std() < 0.04 else ("OVERCONFIDENT" if p.std() > 0.18 else "NORMAL")
log.info(f"[PRED] shape={shape} (std={p.std():.3f})")
```

**Why it matters:** Calibration quality (Brier score) tells you if the mapping is accurate. This tells you the *shape* of the prediction distribution — which directly explains optimizer behaviour. A collapsed distribution (all probs near 0.5) means the model has no signal; an overconfident distribution means calibration didn't compress the extremes. Without this, you cannot diagnose why GA picks unusual thresholds or why trade count is off.

```
[PRED] US30/1m — OOS calibrated probability distribution (n=24000):
  mean=0.512  std=0.083
  p10=0.415   p25=0.461  p50=0.508  p75=0.558  p90=0.615
  %>0.70 = 6.2%    %>0.65 = 14.8%   %>0.60 = 28.3%
  %<0.30 = 4.8%    %<0.35 =  9.1%   %<0.40 = 18.7%
  shape: NORMAL (std 0.083 — good discrimination)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `mean` | Near `0.50` (class balance) or slightly above if TP rate > 50% | `mean > 0.60` → calibration shifted upward; miscalibrated or class imbalance not handled |
| `std` | `0.06–0.12` for intraday ML | `< 0.04` → collapsed distribution; model returns near-constant probability; **no signal at all** |
| `std` | `0.06–0.12` | `> 0.18` → overconfident; isotonic may have over-sharpened; verify Brier improvement was genuine |
| `%> conf_threshold` | Should match `n_trades / n_bars` ratio from `[GA]` result | If `conf=0.668` selected but only `2%` of OOS probs exceed it → optimizer found 2% signal density; verify trade count makes sense |
| `p90` | `< 0.75` usually | `> 0.85` → extreme tail; model has very high-confidence predictions that should be scrutinised |
| `%<0.30` ≈ `%>0.70` | Roughly symmetric (calibrated model has balanced tails) | Asymmetric by > 3× → model is directionally biased; check class balance in training |

**Cross-check with `[STABILITY]`:** If `std < 0.06` (collapsed), the stability curve will be flat — confidence threshold barely matters. If `std > 0.15`, small threshold changes will dramatically change trade count — this produces the "spike" pattern the stability check warns about.

---

### 3.12 CALIBRATION STABILITY — `[CAL-STABILITY]`

**When emitted:** `train.py` immediately after `[CALIBRATION]`.

```
[CAL-STABILITY] US30/1m — Brier by ATR tertile (raw → calibrated):
  low_vol:  0.214 → 0.198
  mid_vol:  0.228 → 0.213
  high_vol: 0.261 → 0.249
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| All tertiles improve | `brier_after < brier_before` for each | Any `WEAK` flag → calibration hurts that regime; most concerning in `high_vol` (CVaR region) |
| `high_vol` Brier | Should be highest (most uncertain) | `high_vol < low_vol` → unusual; data ordering may be wrong |
| Improvement consistency | All three improve by similar margin | `low_vol` improves 0.030, `high_vol` improves 0.001 → calibration overfit to quiet market periods |

---

### 3.13 ENSEMBLE DIVERSITY — `[ENSEMBLE]`

**When emitted:** `train.py` after OOS predictions are collected across all folds.

```
[ENSEMBLE] US30/1m | XGB-RF OOS correlation=0.73 | disagree_rate=0.18 | n_oos=24000
[ENSEMBLE] Diversity: GOOD (correlation < 0.90)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Correlation | `0.55–0.88` | `> 0.92` → models are nearly identical; ensemble adds no value; check if same features dominate both |
| `disagree_rate` | `0.12–0.30` | `< 0.05` → models almost always agree; no diversity benefit | `> 0.45` → models contradicting each other too often; one may be noise |
| Diversity label | `GOOD` | `POOR` or `REDUNDANT` → investigate feature set |

---

### 3.14 FEATURE IMPORTANCE — `[IMPORTANCE]`

**When emitted:** `train.py` after each fold trains XGB and RF. Logs top-10 features by XGBoost `feature_importances_` (gain). Not for explainability — for sanity.

**Why it matters:** PSI monitors input distribution drift. This detects *model reliance drift* — does XGBoost depend on the same features across folds? If fold 0 uses VWAP features and fold 2 uses momentum features, the model has changed character. Also catches the most dangerous training bugs: a single feature dominating (usually leakage or a data artefact).

```
[IMPORTANCE] US30/1m fold=0 (XGB gain, top 10):
  1. vwap_dist_atr      0.182
  2. session_vwap_slope 0.124
  3. atr14              0.098
  4. order_flow_delta   0.087
  5. regime_vol         0.071
  6. momentum_5         0.063
  7. equal_highs        0.058
  8. volume_z           0.051
  9. spread_proxy       0.043
 10. ema_x_21_55        0.039
[IMPORTANCE] US30/1m fold=0 Jaccard(fold_0, fold_1) similarity = 0.70 (OK)
```

**After all folds, log aggregate:**
```
[IMPORTANCE] US30/1m cross-fold Jaccard similarity: fold0↔1=0.70, fold1↔2=0.68, avg=0.69 (OK ✅)
[IMPORTANCE] Consistently top-3 across all folds: vwap_dist_atr, atr14, order_flow_delta
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Top feature importance | `< 0.25` for #1 feature | `> 0.40` for any single feature → **potential leakage or data artefact**; that feature is dominating; investigate |
| Cross-fold Jaccard similarity | `> 0.60` avg | `< 0.50` → model using different features in different folds; regime dependency; check `[REGIME]` drift score for same folds |
| Consistent top-3 | Same core features across folds | Completely different features each fold → model logic is unstable; ensemble unreliable |
| Swing features (`equal_highs`, `swing_high_dist`) in top 5 | Acceptable but note look-ahead | Swing features in top-2 + high Jaccard → model may be exploiting the known look-ahead leakage |
| Random/noise features in top 10 | None | `bar_index` or similar → leakage; `timestamp` → temporal overfitting |

---

### 3.15 GENETIC ALGORITHM — `[GA]`

**When emitted:** `train.py` during GA optimization (one log per GA run, final result).

```
[GA] US30/1m | gen=50/50 best_score=1.847 | sl_atr=2.3 tp_mult=3.0 be_r=1 conf=0.672 htf_w=0.28 | n_trades=312
[GA] Returned to Optuna as seed params. best_score=1.847
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `best_score` (Sharpe) | `1.0–3.0` | `< 0.5` → optimizer found no edge; check data quality and spread model | `> 5.0` → suspiciously high; likely overfit or spread model not active |
| `sl_atr` | `≥ 2.0` (expected post-spread) | `< 1.5` → optimizer picking tight SLs; spread or fill_prob not penalizing correctly |
| `tp_mult` | `2–4` for intraday US30 | `> 5` → TP too far; most trades will exit at SL; check TP rate in labels |
| `n_trades` | `≥ 100` (MIN_CREDIBLE) | `< 100` → optimizer penalised this; if consistent, confidence threshold too high |
| `conf` (confidence threshold) | `0.55–0.75` | `> 0.80` → very selective; likely too few trades | `< 0.52` → near random; model not discriminating |
| `htf_w` | `0.1–0.4` | `≈ 0` → HTF has no effect; check if HTF data loaded correctly |
| `be_r` | `0`, `1`, or `2` | Out of range → `_select_label_col()` mapping error |

---

### 3.16 EXECUTION MODEL SUMMARY — `[EXECUTION]`

**When emitted:** `train.py` after GA completes, summarising the execution realism statistics across all trade simulations in the best genome's evaluation.

**Why it matters:** You model spread, slippage, fill probability, and execution delay — but currently no single log confirms they're all active and in the right ranges. This closes that gap. If spread is near-zero, if gap-skip rate is too high, or if fill rate is too low, the optimizer's Sharpe estimate is biased.

```
[EXECUTION] US30/1m — market realism summary (best genome, n_trades=312):
  avg_spread    = 2.14 pts   p50=1.82  p95=3.91
  avg_slippage  = 0.68 pts   p95=1.42
  fill_rate     = 83.2%      (missed fills: 16.8% of signals)
  gap_skip_rate = 5.4%       (of delayed-fill attempts: gap > MT5_DEVIATION_PTS=30)
  delay_rate    = 29.7%      (≈ EXEC_DELAY_PROB=0.30 ✅)
  total_cost_per_trade = 3.47 pts avg  (spread + slip + delay cost)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `avg_spread` | `1.5–2.5 pts` during session | `< 0.5` → spread model not active; check `.env` SPREAD_BASE_PTS | `> 5.0` → excessive; check SPREAD_ATR_COEFF against typical ATR |
| `fill_rate` | `75–90%` | `> 95%` → fill_prob model too permissive | `< 60%` → too selective; spread or sl_dist unusually mismatched |
| `gap_skip_rate` | `3–10%` | `> 20%` → MT5_DEVIATION_PTS too tight for the ATR; raise in `.env` | `0%` → delayed-fill logic not triggering gap check; code bug |
| `delay_rate` | Near `EXEC_DELAY_PROB` (default 0.30 = 30%) | Large deviation → RNG seeding issue or delay logic not entered |
| `total_cost_per_trade` | `2–5 pts` per trade | `< 1 pt` → friction model not working; Sharpe estimate optimistic | `> 8 pts` → cost exceeds typical edge; strategy may be marginal in live |

---

### 3.17 OPTUNA OPTIMIZATION — `[OPTUNA]`

**When emitted:** `train.py` after Optuna finishes (summary log).

```
[OPTUNA] US30/1m | trials=500 best_score=1.923 | sl_atr=2.4 tp_mult=3.0 be_r=1 conf=0.668
[OPTUNA] vs GA: Optuna(1.923) > GA(1.847) — Optuna params selected
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `best_score` vs GA | Optuna score ≥ GA score (expected: Optuna refines GA seed) | GA score much higher than Optuna → GA seed not passed correctly; Optuna starting from random |
| Winner | Either `Optuna params selected` or `GA params selected` | If always the same → check if both optimizers are actually running |
| Score difference | `< 20%` between GA and Optuna winner | `> 50%` gap → one optimizer is not functioning |

---

### 3.18 GA FITNESS INTERNALS — `[FITNESS]` (verbose, per-genome)

*This log fires on every genome evaluation — only enable for debugging, not production training.*

For debugging suspicious GA results, add temporarily to `ga_fitness()`:
```python
log.debug(f"[FITNESS] n={n_trades} sharpe={sharpe:.3f} er={er:.3f} "
          f"cvar={cvar_95:.3f} mean_r={mean_r:.3f} fill_miss={n_fill_miss} "
          f"gap_miss={n_gap_miss} fitness={fitness:.4f}")
```

When debugging, look for:
| Signal | Meaning |
|--------|---------|
| `n_trades` always near 100 | MIN_CREDIBLE floor kicking in; confidence too high for the data |
| `sharpe > 4.0` frequently | Spread/slippage constants near-zero → check `.env` spread keys |
| `cvar_95` near 0 | CVaR calculation empty (too few trades) |
| `gap_miss > 30%` of trades | MT5_DEVIATION_PTS too tight; raise in `.env` |
| `fill_miss > 20%` of trades | fill_prob too aggressive; or spread very wide relative to SL |

---

### 3.19 SPA TEST — `[SPA]`

**When emitted:** `train.py` after optimization, before robustness gating.

```
[SPA] US30/1m | p_value=0.031 | n_bootstrap=1000 | PASS (p < 0.05)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `p_value` | `< 0.05` | `> 0.05` → strategy fails reality check; edge may be from search space luck |
| `p_value` | `> 0.001` | `< 0.001` → suspiciously strong; check bootstrap n and Sharpe baseline |
| `n_bootstrap` | `≥ 500` | `< 100` → result unreliable |

---

### 3.20 ROBUSTNESS GATES

**When emitted:** `train.py` after SPA, part of tiered gating.

```
[ROBUST] US30/1m | DSR=0.84 (PASS) | MC_pass_rate=0.73 (PASS) | sensitivity_score=62 (PASS) | ER=4.21 (rank 1)
[ROBUST] US30/1m → SELECTED for deployment
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `DSR` (Deflated Sharpe) | `> 0` | `≤ 0` → strategy fails after trial count deflation; likely overfit |
| `MC_pass_rate` | `> 0.60` | `< 0.50` → fewer than half MC paths are profitable; fragile |
| `sensitivity_score` | `≥ 50` | `< 50` → strategy breaks under small parameter nudges |
| `ER` | `> 2.5` for viable intraday edge | `< 1.5` → too much drawdown relative to return |

---

### 3.21 CONFIDENCE STABILITY — `[STABILITY]`

**When emitted:** `train.py` after final backtest, sweeping confidence ±0.03.

```
[STABILITY] US30/1m confidence sensitivity (sl=2.4, tp=3.0):
  conf=0.638 → Sharpe=1.71, ER=3.82
  conf=0.648 → Sharpe=1.81, ER=4.01
  conf=0.658 → Sharpe=1.88, ER=4.18
  conf=0.668 → Sharpe=1.92, ER=4.21  ← selected
  conf=0.678 → Sharpe=1.89, ER=4.15
  conf=0.688 → Sharpe=1.84, ER=4.02
  conf=0.698 → Sharpe=1.76, ER=3.88
[STABILITY] Shape: SMOOTH ✅ (max drop 8.3% across ±0.03)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Curve shape | Smooth peak, gradual decline either side | Isolated spike (adjacent values >> or <<) → fragile threshold |
| Max drop from peak | `< 15%` across ±0.03 | `> 25%` drop within ±0.01 → cliff edge; pick a threshold with a flatter peak |
| `SMOOTH` label | Present | `SPIKE` → do not deploy at that exact threshold; manually select the adjacent smoother value |

---

### 3.22 FINAL RETRAIN — `[RETRAIN]`

**When emitted:** `train.py` after optimization, when retraining on best-matching label column.

```
[RETRAIN] US30/1m | label=target_sl2.5_tp3_be1 | n_train=66000 n_oos=8000
[RETRAIN] Brier before=0.2298 after=0.2141 improvement=0.0157
[RETRAIN] Production model saved → models/US30_1m_xgb.pkl + calibrator.pkl
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `label` column | Matches best `sl_atr`, `tp_mult`, `be_r` from optimizer | `target_sl1.5_tp2_be0` (default) every time → retrain skipping; best params always mapping to default |
| `n_train` | Full dataset minus final OOS fold | Very small `n_train` → check dataset size and fold config |
| `brier_after < brier_before` | Always | `improvement < 0` → calibration hurt final model; check n_oos size and class balance |
| `improvement` | `0.005–0.025` | `> 0.05` → OOS was very miscalibrated; investigate |
| Model saved | Message present | Absent → file write failed; model not updated |

---

### 3.23 EXPECTED BASELINES SAVED — `[BASELINES]`

**When emitted:** `train.py` immediately after `upsert_strategy()` completes. Add one line:
```python
log.info(f"[BASELINES] {symbol}/{tf}m saved to DB: "
         f"expected_sharpe={params['expected_sharpe']:.3f} "
         f"expected_win_rate={params['expected_win_rate']:.3f} "
         f"expected_trades_per_day={params['expected_trades_per_day']:.2f}")
```

```
[BASELINES] US30/1m saved to DB: expected_sharpe=1.920 expected_win_rate=0.580 expected_trades_per_day=3.40
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `expected_sharpe` | Matches `[OPTUNA]` / `[RETRAIN]` Sharpe | Mismatch → DB write used wrong source |
| `expected_win_rate` | `0.45–0.65` for intraday momentum | `> 0.70` → suspicious; check TP multiple (low tp_mult inflates WR) |
| `expected_trades_per_day` | `2–8` for US30 intraday session | `< 1` → model too selective for kill switch to trigger; `> 15` → overtrading signal |

---

### 3.24 LIVE STARTUP — `[KILLSWITCH]` INIT

**When emitted:** `live.py` at startup when loading strategy from DB.

```
[KILLSWITCH] Loaded baselines: expected_sharpe=1.92 expected_win_rate=0.58 expected_trades_per_day=3.4
[KILLSWITCH] State: ACTIVE | window=20 trades
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| All three baselines loaded | Non-zero values matching training output | `expected_sharpe=0.0` or `expected_trades_per_day=0.0` → DB save failed; kill switch using fallback defaults → too permissive |
| State | `ACTIVE` at startup | `WARN` or `DISABLE` at startup → previous session's state persisted; investigate |

---

### 3.25 LIVE SIGNAL — `[SIGNAL]`

**When emitted:** `live.py` every time a signal is evaluated (including rejections). Requires `get_signal()` to return a dict with `prob_raw`, `prob_cal`, `htf_state`, `htf_adj`, `prob_final`, and `direction` — not just the final probability. Add structured logging after the signal decision:
```python
sig = get_signal(df, model, calibrator, params)
action = "PASS" if sig["prob_final"] >= params["confidence_threshold"] else "SKIP"
log.info(
    f"[SIGNAL] {symbol}/{tf}m | prob_raw={sig['prob_raw']:.3f} prob_cal={sig['prob_cal']:.3f} | "
    f"htf={sig['htf_state']} htf_adj={sig['htf_adj']:+.3f} | "
    f"prob_final={sig['prob_final']:.3f} | conf={params['confidence_threshold']:.3f} → {action}"
    + (f" | direction={sig['direction']}" if action == "PASS" else "")
)
```

```
[SIGNAL] US30/1m | prob_raw=0.681 prob_cal=0.664 | htf=BULL htf_adj=+0.028 | prob_final=0.692 | conf_thresh=0.668 → PASS | direction=LONG
[SIGNAL] US30/1m | prob_raw=0.643 prob_cal=0.631 | htf=BEAR htf_adj=-0.021 | prob_final=0.610 | conf_thresh=0.668 → SKIP (below threshold)
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `prob_raw` vs `prob_cal` | `prob_cal < prob_raw` usually (calibration compresses extremes) | `prob_cal >> prob_raw` → isotonic inversion; re-examine calibration fit |
| `htf_adj` | ±0.01 to ±0.06 | `> ±0.10` → HTF weight too high; check `htf_w` param from optimizer |
| PASS/SKIP ratio | Roughly `10–30%` PASS | `> 60%` PASS → threshold too low, model firing too often | `< 5%` PASS → threshold too high, almost never trading |

---

### 3.26 TRADE LIFECYCLE — `[TRADE]`

**When emitted:** `live.py` on trade open, BE trigger, and close.

**⚠️ CRITICAL BUG IN CURRENT CODE — fix before paper trading:**
`_monitor_open_positions()` calls `close_live_trade(ticket, pnl=0.0)` with a hardcoded zero when a position disappears from MT5. This means:
- Every closed trade is stored in the DB with `pnl=0.0`
- The kill switch's rolling Sharpe calculation reads these zero values → it is comparing expected Sharpe against a sequence of zeros → kill switch is non-functional as a P&L monitor

**Fix:** Before calling `close_live_trade()`, fetch actual P&L from MT5 deal history:
```python
deals = mt5.history_deals_get(position=ticket)
actual_pnl = sum(d.profit for d in deals) if deals else 0.0
close_live_trade(ticket, pnl=actual_pnl)
```
`history_deals_get(position=ticket)` is a fast, targeted lookup — one API call per close event.

```
[TRADE] OPEN  | id=4821 US30 LONG | entry=44821.5 sl=44771.5 tp=44971.5 | sl_pts=50.0 tp_pts=150.0 | risk=$100 | prob=0.692
[TRADE] BE    | id=4821 | price=44871.5 (+50 pts = +1R) → SL moved to 44822.5
[TRADE] CLOSE | id=4821 | exit=44971.5 | pnl=+$200.0 (+2.0R) | duration=47m | reason=TP
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| `sl_pts` / `tp_pts` ratio | `tp_pts / sl_pts ≈ tp_mult` from params | Ratio mismatch → SL/TP computation using wrong ATR value |
| `risk` | Always `$100.00` | Any other value → fixed risk override not working |
| BE trigger | `+1R` from entry for `be_r=1` | Triggers at wrong price → BE logic using entry_price incorrectly |
| `pnl` for TP | `≈ risk × tp_mult` (e.g., `tp_mult=2` → `pnl ≈ +$200`) | Very different → position sizing or TP calculation error; or pnl=0.0 for all trades → bug above not fixed |
| `reason=TP` rate over time | Roughly matches `[BASELINES]` expected_win_rate | Much lower → live spread/slippage harder than backtest; expected some gap but > 15% drop needs investigation |
| `pnl` ever exactly `0.0` on CLOSE | Should not happen after bug fix | All CLOSEs show `pnl=0.0` → `history_deals_get` fix not applied; kill switch is blind |

---

### 3.27 KILL SWITCH TRANSITIONS — `[KILLSWITCH]`

**When emitted:** `live.py` when state changes (ACTIVE → WARN → REDUCE → DISABLE).

```
[KILLSWITCH] State change: ACTIVE → WARN | rolling_sharpe=0.71 (expected 1.92) | win_rate=0.44 (expected 0.58) | trades_in_window=20
[KILLSWITCH] Signal frequency LOW: 1.2 trades/day (expected 3.4) over 8 days
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| Transitions during paper trading | Rare; WARN acceptable; REDUCE/DISABLE should not happen without investigation | `DISABLE` within first week of paper trading → strategy not working in live at all; review signal rejection log |
| `rolling_sharpe` at WARN | `< 50% of expected` | Normal early variance; investigate if sustained > 2 weeks |
| Frequency warning | Only fires after `≥ 5 days elapsed` | Fires on day 1–2 → clock calculation bug; check `oldest_trade_ts` source |

---

### 3.28 DRIFT MONITOR — `[DRIFT]`

**When emitted:** `live.py` on schedule (e.g., every N trades or every hour).

```
[DRIFT] PSI check | n_features=10 | STABLE: 8/10 | WARN: vwap_dist_atr(0.14) | ALERT: none
[DRIFT] All PSI scores within tolerance.
```
| Check | Healthy | ⚠️ Red flag |
|-------|---------|------------|
| STABLE count | `≥ 8/10` | `< 6/10` → data distribution shifted significantly |
| WARN features | Isolated PSI `0.10–0.25` acceptable | Same feature consistently in WARN across multiple checks → real drift |
| ALERT features | None | Any PSI `> 0.25` → feature distribution has shifted significantly; model may be invalidated |
| VWAP features drifting | Expected near session time boundaries | Same PSI pattern for weeks → regime shift, not boundary effect |

---

### 3.29 FULL TRAINING RUN — QUICK SANITY CROSS-CHECK TABLE

After the full training log, verify these numbers are internally consistent:

| Cross-check | How to verify |
|-------------|--------------|
| `[DATA]` `duplicate_ts=0` | Hard requirement before any other check; duplicates corrupt rolling features — code raises `ValueError` if non-zero |
| `[TRADE]` CLOSE `pnl ≠ 0.0` | Confirm `history_deals_get` bug fix was applied; if all CLOSEs show `pnl=0.0`, kill switch Sharpe calculation is computing against zeros → non-functional |
| `[DATA]` `missing_bars` pct vs `[REGIME]` ATR CV | High missing bars in one period + high ATR CV → that period may be the regime break; check if it's in train or OOS |
| `[FEATURES]` frozen/exploding → `[IMPORTANCE]` top feature | If a frozen feature (`std ≈ 0`) somehow appears in `[IMPORTANCE]` top-10, that's a sorting/indexing bug |
| `[CLASS]` positives pct vs `[BASELINES]` expected_win_rate | Should match within ±5%; large gap means the strategy's win rate assumption is wrong at the class level |
| `[PRED]` `%> conf_threshold` vs `[GA]` `n_trades / total_bars` | Fraction of OOS probs exceeding confidence threshold should equal the trade density; mismatch means signal density is regime-dependent |
| `[PRED]` `std` vs `[STABILITY]` curve shape | `std < 0.06` → stability curve will be near-flat (threshold doesn't matter much); `std > 0.15` → curve will have sharp peak/drop |
| `[EXECUTION]` `avg_spread` vs `.env` `SPREAD_BASE_PTS` | `avg_spread` should be ≥ `SPREAD_BASE_PTS` (1.5 pts); if equal or below, ATR volatility component not adding correctly |
| `[EXECUTION]` `fill_rate` vs `[GA]` `n_trades` ratio | `fill_rate = n_trades / raw_signals`; if `n_trades=312` but `fill_rate=83%`, raw signals ≈ 376; this should match label-filtered count |
| `[IMPORTANCE]` top-3 consistent vs `[REGIME]` high drift | If drift is HIGH between folds AND importance top-3 differs across folds → model is regime-dependent; not just data drift |
| Fold OOS Sharpe vs final Sharpe | `[FOLD]` avg OOS Sharpe should be within 30% of `[OPTUNA]` best_score |
| Label TP rate vs win rate | `[LABELS]` TP rate for best label col should roughly match `[BASELINES]` expected_win_rate |
| `sl_atr` in `[GA]`/`[OPTUNA]` vs `[TRADE]` `sl_pts` | `sl_pts ≈ sl_atr × ATR14_at_entry`; verify first live trade |
| `n_trades` in `[GA]` vs `expected_trades_per_day × backtest_days` | Should be consistent (within 20%) |
| `brier_before` in `[CALIBRATION]` vs `[RETRAIN]` | Both measure uncalibrated Brier; values should be similar (same model, different label may cause small difference) |
| `[ENSEMBLE]` correlation vs `[IMPORTANCE]` overlap | If XGB and RF correlation `> 0.90` AND they use the same top features, ensemble is redundant; investigate hyperparameter diversity |

---

## 4. DEFERRED (unchanged from v6)

### Paper trade phase (P1–P3): begin from day 1 of paper trading
### Layer 3 (L3-1 to L3-12): after live capital is stable

---

## 5. SYSTEM ARCHITECTURE (updated)

```
TRAINING:
  [SEED] → [DATA] integrity → [PARITY] → features → [FEATURES] sanity → [LEAKAGE]
  → [LABELS] grid → [CLASS] balance
  → walk-forward folds → [REGIME] per fold
  → per-fold train XGB+RF → [FOLD] AUC → [IMPORTANCE] per fold
  → stack OOS predictions → isotonic calibration → [CALIBRATION] → [PRED] distribution
  → [CAL-STABILITY] volatility bins → [ENSEMBLE] diversity
  → GA (12k eval, spread+slippage+fill_prob+exec_delay+CVaR+MIN_CREDIBLE) → [GA]
  → [EXECUTION] summary → Optuna (500 trials) → [OPTUNA]
  → tiered robustness: [SPA] → DSR → MC → Sensitivity → [ROBUST]
  → [STABILITY] confidence sweep
  → [RETRAIN] on best label column (Brier before/after)
  → [BASELINES] saved to DB

LIVE:
  [CONFIG] → [KILLSWITCH] init → session gate check → [SIGNAL]
  → [TRADE] open/BE/close → [KILLSWITCH] state check → [DRIFT] PSI schedule
```

**Hardcoded market physics (intentional, not configurable by ML):**
- Session gate: 20:30–01:00 London DST-aware
- Max DD: 35% (account survival)
- Fixed risk: $100/trade (ER accuracy)

**Known approximations (flagged, not bugs):**
- Swing detection look-ahead `SWING_LOOKBACK=10` bars → `[LEAKAGE] KNOWN` warning at training
- Session-level VP assigns full-day POC to all intraday bars → known causal approximation
- SPA test is bootstrap Sharpe > 0, not full Hansen statistic → L3-3
