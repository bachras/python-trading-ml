# ML Trading System — Pre-Training Audit (v6)
**Date:** 2026-04-12
**Instrument:** US30 CFD (Dukascopy tick data → MT5 execution)
**Purpose:** Full audit of codebase against all v5 planned items. This is the final go/no-go checkpoint before the first training run.

**GO / NO-GO:**
| Tier | Status |
|------|--------|
| 🟢 **FUNCTIONALLY READY** | All code logic is implemented. System can train correctly. |
| 🔴 **3 minor gaps** | Six `.env` keys missing. Parameters use correct defaults but are invisible in config. Add before training. |
| 🟠 **1 architecture note** | R7 spread model is inline (not in `tick_pipeline.py`) — acceptable, documented below. |
| 🟠 **3 edge survival gaps** | Regime separation, calibration stability, and execution latency — not blocking first training run but required before scaling capital (Section 4). |

**System strengths (institutional-grade, rare in retail quant stacks):**
Leakage detection · CVaR penalty · Ensemble correlation control · Triple-barrier labeling (75-column grid) · GA + Optuna hybrid search · Isotonic calibration on stacked OOS · Smart kill switch with DB-loaded baselines · Synthetic spread + fill probability modelling · Feature parity enforcement between train and live paths

*This system already answers "does it work historically?" — the remaining gaps address "does it work across different market identities?"*

---

## 1. COMPLETE IMPLEMENTATION STATUS (v6 Audit)

### Blocking Items (R1–R7 from v5)

| # | Item | v5 Status | v6 Audit Result |
|---|------|-----------|-----------------|
| R1 | `expected_trades_per_day` — `db.py` schema, `train.py` compute, `live.py` kill switch | ❌ Missing | ✅ **DONE** — `db.py` lines 87–89 (column + migration); `train.py` lines 701–708; `live.py` `_SmartKillSwitch` line 308 + frequency check lines 399–421 using oldest `live_trades` record |
| R2 | `[LABELS]` TP rate summary in `engineer_features()` | ❌ Missing | ✅ **DONE** — `phase2_adaptive_engine.py` lines 621–635: TP rate range, degenerate check (< 0.02 or > 0.98), BE-bug warning for same sl/tp |
| R3 | `[RETRAIN]` Brier before/after for final model | ⚠️ Partial | ✅ **DONE** — `train.py` lines 774–802 log `brier_before` + `brier_after` when calibrator loads. Minor caveat: no-calibrator branch (lines 809–813) logs single Brier only — acceptable; no-calibrator should not occur in normal training |
| R4 | `ATR_SPIKE_FILTER_MULT=3.0` in `.env` | ❌ Missing | ❌ **STILL MISSING** — `backtest_engine.py` line 41 reads it from env with default `"3.0"`, correct behaviour, but key absent from `.env` |
| R5 | `[PARITY]` train/live feature parity check in `train.py` (full parquet, last 200 rows) | ❌ Missing | ✅ **DONE** — `train.py` `def _check_feature_parity()` at line 122; `[PARITY]` logs lines 137–185; called at line 366 |
| R6 | Two-tier `[LEAKAGE]` check — known look-ahead (swing) → WARNING, unexpected → RuntimeError | ❌ Missing | ✅ **DONE** — `phase2_adaptive_engine.py` `KNOWN_LOOKAHEAD_FEATURES` at line 237; `def _check_leakage()` at line 680; `[LEAKAGE]` logs at lines 692, 737, 742, 746 |
| R7 | Synthetic spread model + `.env` spread keys | ❌ Missing | ⚠️ **PARTIAL** — See note below |

**R7 Architecture Note:** The original v5 spec called for `build_synthetic_spread()` in `tick_pipeline.py`. What was implemented instead is spread computed **inline** at simulation time:
- `phase2_adaptive_engine.py` lines 229–232 and 994–995: `SPREAD_BASE_PTS`, `SPREAD_ATR_COEFF`, `SPREAD_OPEN_MULT`, `SPREAD_OPEN_BARS` read from env; applied per trade in `ga_fitness()`
- `backtest_engine.py` lines 42–45 and 200–201: same constants; synthetic spread applied per bar

This approach is **architecturally correct** for training/backtesting: the Dukascopy zero-spread bias is resolved in both the optimizer and the backtest. Live trading uses MT5 real bid-ask spread directly, so no synthetic model is needed there. The only remaining gap is that the four `.env` keys are absent — the code applies correct defaults silently.

**Remaining R7 gap:** Add four lines to `.env` (see Section 2).

---

### Strongly Recommended Items (S1–S3 from v5)

| # | Item | v5 Status | v6 Audit Result |
|---|------|-----------|-----------------|
| S1 | Execution fill realism in `ga_fitness()` — dynamic `fill_prob` per trade | ❌ Missing | ✅ **DONE** — `phase2_adaptive_engine.py` lines 1034–1036: `fill_prob` computed from spread/SL distance; `if rng.random() > fill_prob: continue` |
| S2 | `[STABILITY]` confidence threshold sensitivity curve | ❌ Missing | ✅ **DONE** — `train.py` lines 830–861: sweeps `conf_base + d` for `d in np.arange(-0.03, 0.04, 0.01)` (±0.03, 7 steps); `[STABILITY]` logs 854–859. Range is wider than the ±0.02 in the spec — better coverage |
| S3 | Global seed — `np.random.seed`, `random.seed`, `GLOBAL_SEED` env | ❌ Missing | ⚠️ **PARTIAL** — `train.py` lines 112–117: `GLOBAL_SEED = int(os.getenv("GLOBAL_SEED", "42"))`, both seeds set, `[SEED]` log present. `GLOBAL_SEED` key absent from `.env` |

---

### All Previously Confirmed Items (spot-check confirms still present)

| # | Item | Status |
|---|------|--------|
| L1-1 | Spread + slippage in `ga_fitness()` | ✅ `spread_arr` line 984–997; `spread_cost`/`slip_cost` lines 1030–1031 |
| L1-2 | 75 label columns (5×5×3 grid) | ✅ |
| L1-3 | `_select_label_col()` with be_r mapping | ✅ |
| L1-4 | be=0 vectorized, be=1/2 per-trade loop | ✅ |
| L1-5 | Binary labels | ✅ |
| L1-6 | Final model retrained on best-matching label | ✅ |
| L1-7 | HTF nudge in `ga_fitness()` | ✅ |
| L1-8 | Daily equity Sharpe with `np.zeros` zero-padding | ✅ `daily_pnl = np.zeros(n_days_span)` line 1094; `np.add.at` line 1095 |
| L1-9 | `MIN_CREDIBLE=100` trade count penalty | ✅ lines 1081–1082 |
| L1-10 | `run_genetic_algo()` returns `best_score` | ✅ |
| L1-11 | `GA_BOUNDS` narrowed | ✅ |
| L1-12 | Daily trade cap removed | ✅ |
| L1-13 | Consecutive loss cooldown removed | ✅ |
| L1-14 | `VOL_SIZE_ENABLED=false` toggle | ✅ |
| L1-15 | ATR spike filter configurable via env | ✅ |
| L1-16 | Fixed `risk = fixed_amt` in backtest | ✅ |
| L1-17 | `bfill()` removed, `ffill + fillna(0)` | ✅ |
| L1-18 | Session-level volume profile (groupby date) | ✅ per-day loop in `add_volume_profile_features()` |
| L2-1 | `IsotonicRegression` calibration on OOS stack | ✅ `train.py` lines 447–448 |
| L2-2 | SPA bootstrap test `_spa_bootstrap_test()` | ✅ Partial — Sharpe > 0 bootstrap, not full Hansen SPA (deferred L3-3) |
| L2-3 | `_DriftMonitor` PSI drift detection | ✅ `live.py` class 217 |
| L2-4 | `_SmartKillSwitch` with expected baselines | ✅ `live.py` class 282 |
| A2 | CVaR penalty in `ga_fitness()` | ✅ `cvar_95`, `cvar_penalty` lines 1115–1117 |
| A8 | `[ENSEMBLE]` diversity `np.corrcoef` | ✅ `train.py` line 423, log lines 429–436 |
| A9 | Trade quality penalty `mean_r < 0.15` | ✅ |

---

## 2. REMAINING GAPS — THREE `.env` ADDITIONS

All remaining gaps are a single `.env` file edit — no code changes needed. The system is **functionally complete**; these make implicit defaults explicit and visible.

### G1. `ATR_SPIKE_FILTER_MULT` — (R4, 1 line)

```
# ── BACKTEST FILTERS ────────────────────────────────────────────────────────
# Skip entry signals where ATR > this multiple of rolling ATR median.
# Set to 99.0 to effectively disable. Market-physics filter, not a behavioral cap.
ATR_SPIKE_FILTER_MULT=3.0
```

---

### G2. Synthetic Spread Model Parameters — (R7, 4 lines)

The code in `phase2_adaptive_engine.py` and `backtest_engine.py` already reads these via `os.getenv` with correct defaults. Adding them makes the spread model visible and adjustable without touching code.

```
# ── SYNTHETIC SPREAD MODEL (Dukascopy tick data has zero/negative raw spread) ──
# Applied at simulation time in ga_fitness() and backtest_engine.
# Base floor (points) — US30 CFD minimum realistic spread during active session.
SPREAD_BASE_PTS=1.5
# Spread widens with volatility: spread += ATR14 * coeff
SPREAD_ATR_COEFF=0.04
# Session-open multiplier applied to first N bars of each trading session
SPREAD_OPEN_MULT=2.0
SPREAD_OPEN_BARS=5
```

**Verification after training:** Check `[SPREAD]` log at training startup for `mean ≈ 1.8–2.2 pts`, `p95 ≈ 3–4 pts`. If mean is near 0, the env key was not picked up.

---

### G3. `GLOBAL_SEED` — (S3, 1 line)

The seed logic is in `train.py` lines 112–117 and works correctly with default 42. Adding to `.env` makes each training run's seed explicit in the config record.

```
# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
# Seed passed to numpy, random, XGBoost, RF, Optuna TPESampler, and DEAP GA.
# Change per training cycle to avoid seed-specific overfitting; keep logged for replay.
GLOBAL_SEED=42
```

---

## 3. EDGE SURVIVAL — IMPLEMENT BEFORE SCALING CAPITAL

These three items are not blocking the first training run or paper trading. They close the remaining "market identity" blind spots — the difference between a system that survives one regime and one that survives across regimes. Implement before committing meaningful capital.

---

### E1. Regime Separation Report per Fold — 🔴 High Priority

**The blind spot:** Walk-forward CV assumes that folds are statistically representative of each other. US30 CFD has structural break regimes (COVID volatility expansion, 2022 compression, 2023 normalization, spread regime shifts with broker liquidity changes). If fold 0 is low-ATR compression and fold 2 is high-ATR expansion, GA can overfit a Sharpe structure that only works in the training regime — without any warning being raised.

None of the current checks (`[LEAKAGE]`, `[PARITY]`, PSI) test this: they verify correctness and stability within a deployment, not regime diversity across training folds.

**Fix — add `[REGIME]` fold divergence log to `train.py` after fold construction, before training:**

```python
def _compute_fold_regime_stats(df_fold: pd.DataFrame) -> dict:
    """Compute regime fingerprint for a fold."""
    rets = df_fold["close"].pct_change().dropna()
    atr  = (df_fold["high"] - df_fold["low"]).rolling(14).mean().dropna()
    acf1 = float(np.corrcoef(np.abs(rets[1:]), np.abs(rets[:-1]))[0, 1])  # volatility clustering
    return {
        "atr_mean":  float(atr.mean()),
        "atr_std":   float(atr.std()),
        "kurtosis":  float(rets.kurt()),
        "vol_clust": acf1,
        "n_bars":    len(df_fold),
    }

def _log_fold_regime_divergence(fold_stats: list[dict]) -> None:
    """Log KL-like divergence between adjacent folds using ATR distribution."""
    for i in range(len(fold_stats) - 1):
        a, b = fold_stats[i], fold_stats[i + 1]
        # Simplified divergence: normalised distance in regime fingerprint space
        drift = (
            abs(a["atr_mean"] - b["atr_mean"]) / (a["atr_mean"] + 1e-9) +
            abs(a["kurtosis"] - b["kurtosis"]) / (abs(a["kurtosis"]) + 1) +
            abs(a["vol_clust"] - b["vol_clust"])
        ) / 3.0
        level = "HIGH ⚠️" if drift > 0.30 else "OK ✅"
        log.info(f"[REGIME] fold_{i} → fold_{i+1} drift={drift:.3f} ({level}) | "
                 f"atr_mean: {a['atr_mean']:.1f}→{b['atr_mean']:.1f}, "
                 f"kurtosis: {a['kurtosis']:.2f}→{b['kurtosis']:.2f}, "
                 f"vol_clust: {a['vol_clust']:.3f}→{b['vol_clust']:.3f}")
    # Overall spread across all folds
    atr_cv = np.std([s["atr_mean"] for s in fold_stats]) / (np.mean([s["atr_mean"] for s in fold_stats]) + 1e-9)
    log.info(f"[REGIME] ATR coefficient of variation across folds: {atr_cv:.3f} "
             f"({'HIGH — regime shift present' if atr_cv > 0.25 else 'OK'})")
```

**What a healthy log looks like:**
```
[REGIME] fold_0 → fold_1 drift=0.14 (OK ✅) | atr_mean: 48.2→51.6, kurtosis: 3.4→3.8, vol_clust: 0.31→0.29
[REGIME] fold_1 → fold_2 drift=0.19 (OK ✅) | atr_mean: 51.6→55.1, kurtosis: 3.8→4.1, vol_clust: 0.29→0.34
[REGIME] ATR coefficient of variation across folds: 0.07 (OK)
```

**Red flag:** `drift > 0.30` (HIGH) on any adjacent pair. This does not stop training but **requires investigation**: does the OOS Sharpe drop sharply on the high-drift fold? If yes, the strategy is regime-specific and may fail in live when regime shifts.

**Action if HIGH:** Either extend the dataset to include more regime diversity, or note the specific regime condition (e.g., ATR > 60) as a live circuit breaker until the strategy proves it can survive it.

---

### E2. Calibration Stability Across Volatility Bins — 🟠 Medium Priority

**The blind spot:** Isotonic calibration is fitted on the full stacked OOS pool. It can silently overfit the mid-probability region (where most trades cluster) while degrading at the tails — exactly where CVaR lives. The current Brier before/after checks confirm average improvement but cannot detect this.

**Fix — split OOS pool into three volatility bins after calibration fitting, log Brier per bin:**

```python
def _check_calibration_stability(oos_probs_raw: np.ndarray,
                                  oos_probs_cal: np.ndarray,
                                  oos_labels: np.ndarray,
                                  oos_atr: np.ndarray) -> None:
    """
    Split OOS trades into low/mid/high ATR tertiles.
    Compute Brier improvement per tertile.
    A healthy calibration improves (or is neutral) across all three.
    """
    tertiles = np.percentile(oos_atr, [33, 67])
    bin_names = ["low_vol", "mid_vol", "high_vol"]
    masks = [
        oos_atr <= tertiles[0],
        (oos_atr > tertiles[0]) & (oos_atr <= tertiles[1]),
        oos_atr > tertiles[1],
    ]
    log.info("[CAL-STABILITY] Brier score by volatility tertile (raw → calibrated):")
    for name, mask in zip(bin_names, masks):
        if mask.sum() < 10:
            log.warning(f"[CAL-STABILITY] {name}: too few samples ({mask.sum()}) to evaluate")
            continue
        b_raw = brier_score_loss(oos_labels[mask], oos_probs_raw[mask])
        b_cal = brier_score_loss(oos_labels[mask], oos_probs_cal[mask])
        flag = " ⚠️ WEAK" if b_cal >= b_raw else ""
        log.info(f"  {name}: {b_raw:.4f} → {b_cal:.4f}{flag} (n={mask.sum()})")
```

**What a healthy log looks like:**
```
[CAL-STABILITY] Brier score by volatility tertile (raw → calibrated):
  low_vol:  0.108 → 0.094
  mid_vol:  0.114 → 0.101
  high_vol: 0.131 → 0.119
```

**Red flag:** Any tertile shows `b_cal >= b_raw` (marked `WEAK`). This means calibration hurts performance in that volatility regime. In high-vol cases this directly impairs CVaR accuracy. Action: check isotonic sample size for that tertile; consider separate calibrators per regime if high-vol degradation persists.

---

### E3. Execution Latency Realism in `ga_fitness()` — 🟠 Medium Priority

**The blind spot:** `ga_fitness()` assumes instant execution: signal fires at bar close → fill at next bar open. In practice, the MT5 order pathway (Python → MT5 terminal → broker → exchange) takes 200–800ms. For US30 CFD:

- Session open bars: price can move 5–15 points in 500ms
- News candles: 20–50 points in 200ms
- Normal bars: usually fine (price stable between bars)

The current `fill_prob` model (S1, already implemented) captures whether an order fills at all. This is about *when* it fills — a filled order on the next bar instead of the current bar changes the entry price and therefore the realised R.

**Gap threshold decision — use MT5 deviation limit, not % of SL:**

The live system already answers the "extreme gap" question: `live.py` places orders with `IOC` fill policy and `20-deviation`. If the market moves more than 20 points from the requested price, MT5 **rejects the order entirely** — it does not fill at a worse price. Therefore:

- Filling at any next-bar open (Option A) is *more optimistic than live* for large gaps — it fills trades MT5 would reject
- Using `sl_dist * 0.50` as threshold (Option B) creates asymmetric optimizer incentives: wide-SL genomes tolerate 87-point gaps while tight-SL genomes skip anything over 30 points, even though the live MT5 behaviour is identical for both

**The physically correct model** uses the same fixed deviation limit as the live system. The `MT5_DEVIATION_PTS` env var (default 20, matching live `20-deviation`) is the threshold. No % of SL needed.

**Implementation — add to `ga_fitness()` trade loop, after `fill_prob` passes:**

```python
# Execution latency model:
# EXEC_DELAY_PROB fraction of trades experience a processing delay (MT5 pathway ~200-800ms).
# Delayed trades fill at next bar open instead of current bar close.
# If the gap between close and next open exceeds MT5_DEVIATION_PTS, MT5 would reject
# the IOC order in live — model as a missed fill (same as fill_prob rejection).
if rng.random() < EXEC_DELAY_PROB and i + 1 < n_bars:
    next_open   = opens[i + 1]          # opens[] precomputed array alongside closes[]
    gap_pts     = abs(next_open - entry_price)
    if gap_pts > MT5_DEVIATION_PTS:
        continue                         # gap too large → order rejected, missed fill
    # Delayed fill: entry moves to next open, SL/TP recalculate from new entry
    entry_price = next_open             # SL/TP are direction * distance from entry,
                                        # so they shift automatically with entry_price
    extra_slip  = gap_pts               # full actual slippage, no artificial cap
else:
    extra_slip = 0.0

total_cost = spread_cost + slip_cost + extra_slip
```

**Required setup additions in `ga_fitness()`:**
```python
# Alongside existing: closes = df["Close"].values, atr = df["ATR14"].values
opens          = df["Open"].values                                 # add this
EXEC_DELAY_PROB = float(os.getenv("EXEC_DELAY_PROB", "0.30"))    # module-level constant
MT5_DEVIATION_PTS = float(os.getenv("MT5_DEVIATION_PTS", "20"))  # module-level constant
```

**Expected effect on optimizer:** Strategies requiring pinpoint entry (tight breakout setups) lose more trades to gap rejection than wide-SL strategies — correctly penalising them. The `sl_atr ≥ 2.0` preference already seen in optimizer output is reinforced. Genomes that cluster near news candles (high-ATR bars) will have higher missed-fill rates, consistent with live experience.

**Configuration — add to `.env`:**
```
# ── EXECUTION LATENCY MODEL ──────────────────────────────────────────────────
# Fraction of trades subject to next-bar-open fill (MT5 processing delay ~200-800ms)
EXEC_DELAY_PROB=0.30
# Max gap (points) between signal-bar close and next-bar open before order is rejected.
# Matches live.py IOC order with 20-deviation. US30 CFD: 20 pts is typical broker limit.
MT5_DEVIATION_PTS=20
```

**Verification after implementation:** The `[GA]` best genome's `n_trades` should drop ~5–10% vs pre-E3 (delayed fills that become gap-rejections). If it drops more than 20%, `MT5_DEVIATION_PTS` may be too tight for the dataset's ATR range — raise to 25 or 30.

---

## 4. DEFERRED — PAPER TRADE PHASE

These require live trade data. Implement from day 1 of paper trading.

| # | Item | Trigger |
|---|------|---------|
| P1 | Live calibration tracker — store `predicted_probability` per trade; daily calibration curve by bin; Δ > 0.10 → reduce sizing | After 50+ closed paper trades |
| P2 | Signal rejection log — log every signal with rejection reason; `rejected_signals` DB table | From day 1 |
| P3 | Parameter stability check — compare `sl_atr`, `tp_mult`, `confidence`, `be_r` across consecutive training runs; UNSTABLE flag if delta exceeds thresholds | Before first live model update |

---

## 5. DEFERRED — LAYER 3 (After Live Capital is Stable)

| # | Item | Priority |
|---|------|----------|
| L3-1 | Meta-labeling (primary → meta-model → trade/reject) | High — ~30–50% DD reduction in published research |
| L3-2 | Block bootstrap Monte Carlo (preserves return autocorrelation) | Medium |
| L3-3 | Full Hansen SPA test (upgrade from Sharpe > 0 bootstrap) | Medium |
| L3-4 | Execution regime features (`execution_quality_score`) | Medium |
| L3-5 | Dynamic capital allocation (tighten threshold when live Sharpe decays) | Medium |
| L3-6 | Regime-conditional models (separate XGB/RF per regime) | Low |
| L3-7 | HTF nudge → learnable ML feature (replace ±30% linear nudge) | Medium |
| L3-8 | Edge concentration analysis in `report.py` (PnL by hour + regime) | Medium |
| L3-9 | Time-based kill switch window (10 calendar-day window; requires R1) | High |
| L3-10 | Full causal swing detection (replace `SWING_LOOKBACK` confirmation) | Medium — removes known look-ahead flag in `[LEAKAGE]` |
| L3-11 | Feature importance drift tracking — SHAP top-K Jaccard similarity across folds; `[FEATURE-DRIFT]` log; flag < 0.60 | Medium — detects silent regime overfitting even when PSI says data is stable |
| L3-12 | Equity curve stationarity test — rolling Sharpe (50-trade window) or KPSS/ADF on equity slope; `[EQUITY-STABILITY] stationary=TRUE/FALSE` | Medium — distinguishes stable-edge system from lucky regime run with same mean Sharpe |

---

## 6. PRE-TRAINING RUN CHECKLIST

### Before training (required):
- [ ] Add G1, G2, G3 to `.env` (6 lines total — documented in Section 2 above)

### Before scaling capital (Section 3 items):
- [ ] E1: Add `[REGIME]` fold divergence report to `train.py` (~25 lines)
- [ ] E2: Add `[CAL-STABILITY]` volatility-bin calibration check to `train.py` (~30 lines)
- [ ] E3: Add execution latency model to `ga_fitness()` + `EXEC_DELAY_PROB=0.30` + `MT5_DEVIATION_PTS=20` to `.env` (~15 lines). Gap > `MT5_DEVIATION_PTS` → missed fill; gap ≤ limit → fill at next open with full slippage, no cap

### Environment verification:
- [x] `MAX_DRAWDOWN_PCT=35` ✅
- [x] `RISK_MODE=fixed` ✅
- [x] `FIXED_RISK_AMOUNT=100` ✅
- [x] `VOL_SIZE_ENABLED` absent → defaults `false` ✅
- [ ] `ATR_SPIKE_FILTER_MULT=3.0` — add (G1)
- [ ] `SPREAD_BASE_PTS=1.5` + 3 spread keys — add (G2)
- [ ] `GLOBAL_SEED=42` — add (G3)

### After training run — verify these logs are present and healthy:

| Log tag | What to check |
|---------|--------------|
| `[SEED]` | `Global seed set to 42` — confirms reproducibility |
| `[SPREAD]` | `mean ≈ 1.8–2.2 pts`, `p95 ≈ 3–4`, `p99 < 10` — confirms spread model active |
| `[PARITY]` | `n_features_differ=0`, "PASS" — train and live pipelines match |
| `[LEAKAGE]` | `0 unexpected leaks` — only known look-ahead (swing) in WARNING |
| `[LABELS]` | `75 label columns`; no degenerate; be=1 TP rate ≤ be=0 for same sl/tp (tp≥2) |
| `[CALIBRATION]` | `brier_before > brier_after` in walk-forward OOS |
| `[ENSEMBLE]` | XGB vs RF correlation `< 0.92` |
| `[RETRAIN]` | `brier_before=X.XXXX brier_after=Y.YYYY improvement=Z.ZZZZ`; improvement > 0 |
| `[SPA]` | `p_value < 0.05` for best strategy |
| `[STABILITY]` | Smooth curve (max Sharpe drop < 15% across ±0.03 from best threshold); no isolated spike |
| `[GA]` best genome | `sl_atr ≥ 2.0`; `cvar_95` logged; `mean_r > 0.15` |
| `[KILLSWITCH]` init | `expected_sharpe` and `expected_win_rate` loaded from DB (not fallback defaults) |
| `[REGIME]` *(after E1)* | Fold drift scores logged; no `HIGH ⚠️` flag without investigation; ATR CV < 0.25 |
| `[CAL-STABILITY]` *(after E2)* | No tertile shows `WEAK`; high-vol Brier improves or is neutral |

### Red flags — investigate before paper trading if seen:
- `[SPREAD]` mean ≈ 0 → spread model not applied
- `[PARITY]` FAIL → train/live feature mismatch; do not deploy
- `[LEAKAGE]` unexpected features → new look-ahead introduced; investigate
- `[RETRAIN]` `improvement < 0` → calibration failed on final model; check OOS sample size
- `[STABILITY]` isolated Sharpe spike → fragile threshold; pick adjacent smoother value
- `[ENSEMBLE]` correlation > 0.95 → XGB and RF are essentially identical; check feature set
- Walk-forward most-recent fold Sharpe > 30% worse than earlier folds → regime shift
- `[REGIME]` `drift > 0.30 (HIGH)` on any fold pair + OOS Sharpe drops on that fold → strategy is regime-specific; add ATR-based live circuit breaker before scaling
- `[CAL-STABILITY]` high-vol tertile shows `WEAK` → calibration degrades under volatility; investigate isotonic sample size for that regime

### Before first paper trade:
- [ ] `DRY_RUN=true` for 3 sessions — signals fire, no orders placed
- [ ] `[CONFIG]` log: `max_drawdown_pct: 35.0`, `fixed_risk_amt: 100.0`
- [ ] `[KILLSWITCH]` init log: baselines loaded from DB
- [ ] `[DRIFT]` PSI fires on schedule
- [ ] Activate P1 (live calibration tracker) and P2 (signal rejection log) from day 1

---

## 7. SYSTEM ARCHITECTURE SUMMARY

```
TRAINING (train.py + phase2_adaptive_engine.py):
  Dukascopy tick data → OHLCV bars (tick_pipeline.py)
  → Synthetic spread applied inline at simulation time
      - SPREAD_BASE_PTS + ATR14*SPREAD_ATR_COEFF + session-open widening
      - Lognormal noise to avoid mechanical-looking series
      - Hard floor: SPREAD_BASE_PTS (default 1.5 pts)
  → [PARITY] check: both pipelines produce identical features on full history
  → 73 institutional features (institutional_features.py)
      - Session VWAP, Volume Profile (per-day groupby), Order Flow, Regime
      - ffill + fillna(0); bfill() removed
  → [LEAKAGE] check: two-tier (known look-ahead → WARNING; unexpected → RuntimeError)
  → 75 triple barrier label columns (5×5×3: sl_atr × tp_mult × be)
      - be=0: vectorized numpy; be=1/2: per-trade Python loop with dynamic SL
      - Binary labels: 1.0=TP hit, 0.0=else
  → [LABELS] log: TP rates, degenerate check, BE-bug assertion
  → Walk-forward (3 folds, PURGE_BARS=10) + XGB+RF training
      - Default label: target_sl1.5_tp2_be0 for fold search
      - IsotonicRegression calibration on stacked OOS predictions
  → GA (12,000 eval) + Optuna (500 trials) parameter search
      - ga_fitness(): synthetic spread+slippage, fill_prob (missed fills),
        HTF nudge, daily equity Sharpe (np.zeros padding), MIN_CREDIBLE=100,
        CVaR penalty, trade quality penalty (mean_r < 0.15)
      - _select_label_col() matches genome (sl,tp,be) to nearest grid column
      - GA returns best_score
  → [STABILITY]: confidence threshold swept ±0.03 in 0.01 steps; smooth curve required
  → Tiered robustness gating: DSR > 0 → MC pass → Sensitivity ≥ 50 → ER rank
  → SPA bootstrap test (p < 0.05)
  → Final model retrained on best-matching label column
      - [RETRAIN] logs brier_before + brier_after + improvement
  → [ENSEMBLE]: XGB vs RF OOS correlation check
  → Save: model pkl, calibrator pkl, params + expected baselines to DB

LIVE TRADING (live.py):
  MT5 real-time data → pipeline.py → 73 features → get_signal()
  → calibrated probability (IsotonicRegression calibrator)
  → Session gate: 20:30–01:00 London, DST-aware (hardcoded market physics)
  → Confidence threshold (from optimizer)
  → HTF nudge (adjusted_prob ± htf_weight × 0.3)
  → Risk gates: max DD 35%, daily loss %, max positions, fixed $100/trade
  → MT5 order (TRADE_ACTION_DEAL, IOC, 20-deviation)
  → DB logging (open/close lifecycle, BE detection)
  → _DriftMonitor: PSI on top features, periodic schedule
  → _SmartKillSwitch: rolling 20-trade Sharpe+WR vs expected baselines
      - expected_sharpe, expected_win_rate, expected_trades_per_day from DB
      - Frequency check: n_live_trades vs expected rate since oldest live trade
      - Graduated response: warn → reduce sizing → disable
  → Hot-reload: newer model files + no open positions required
```

**Intentional hardcoded constraints (market physics or account survival):**
- Session gate: 20:30–01:00 London — US30 underlying inactive outside these hours
- Max DD 35%: account survival circuit breaker (separate from kill switch)
- Fixed $100/trade: ER accuracy by design; vol-sizing toggle available for future scaling

**Known design trade-offs (documented, not bugs):**
- Swing detection features (`swing_high_dist`, `equal_highs`, etc.) are confirmed forward-looking by ≤10 bars due to `SWING_LOOKBACK` confirmation window — flagged by `[LEAKAGE]` as WARNING; remediation deferred to L3-10
- Session-level volume profile (full-day POC assigned to all intraday bars) is a known approximation — same trade-off as above; causal real-time VP deferred

**Risk model:** Fixed $100/trade → linear equity curve → ER numerator and denominator in same unit → order-independent, scale-independent capital efficiency measurement
