# ML Trading System — Institutional Deep Audit (v3)
**Date:** 2026-04-11  
**Analyst:** Independent AI Review — Institutional Desk Standard  
**Instrument:** US30 CFD (Dukascopy tick data → MT5 execution)  
**Prior reviews:** ANALYSIS.md (critical analysis), ANALYSIS_v2.md (implementation audit)  
**Scope:** Full codebase review of 12 Python modules (≈3,400 lines of core logic), SQLite schema, and all prior review recommendations  
**Note:** `US30_params.json` is stale (pre-ANALYSIS.md). All parameter references below are to the codebase defaults and search ranges, not to that file.

---

## Design Philosophy (as stated by the system architect)

The intended optimization pipeline is a **two-stage process:**

1. **Stage 1 — Unconstrained exploration:** Give the ML models and parameter optimizer (GA + Optuna) maximum freedom. Session hours, TP/SL multipliers, HTF alignment weight, break-even triggers, confidence thresholds, timeframes — these are **all searchable parameters**, not hardcoded human assumptions. The optimizer discovers what works from data.

2. **Stage 2 — Physics constraints only:** Add real-world execution costs (spread, slippage) that cannot be optimized away because they are market physics. Re-optimize with these costs included. The result goes to live.

**What is NOT the philosophy:** Hardcoding restrictions based on human intuition before ML has had a chance to discover the optimal values. As Mark Douglas observed: you cannot know which of the next N trades will be profitable. Arbitrary limits like "max 6 trades/day" cut potential winners as often as losers. If those constraints are truly optimal, **the optimizer will discover them** — the session features, hour encoding, and trade-frequency metrics are all in the feature set.

**What is hardcoded intentionally (and correctly):**
- **Session gate (20:30–01:00 London, DST-aware):** US30 CFD is essentially a closed market during these hours. The underlying Dow Jones index is not trading, spreads widen to 10–30+ points, and liquidity evaporates. This is market physics, not a behavioral assumption — the same category as spread and slippage.
- **Max drawdown circuit breaker (35%):** Hard account-survival limit. No strategy edge is worth risking more than 35% of capital in drawdown.
- **Fixed $100 risk per trade:** Starting capital $10,000 → $100 = 1% of initial capital, fixed. As the account grows, risk stays at $100. This is a deliberate design choice for ER accuracy (see G11 below).

This philosophy is correct and aligns with how the best quantitative firms operate. The analysis below is written through this lens.

---

## Executive Summary

This project has evolved significantly since the first review. The three critical gaps identified in ANALYSIS.md (#4 labeling, #5 walk-forward, #6 execution modeling) are now fixed. The system architecture is sound: real Dukascopy tick data → institutional-grade features → XGB+RF ensemble → GA+Optuna parameter optimization → tiered robustness gating → MT5 live execution.

The deep audit reveals three categories of remaining issues:
1. **Internal inconsistencies** between the optimizer (`ga_fitness`), the backtest engine, and the live signal pipeline — each computes things slightly differently, meaning the optimizer maximizes a metric that doesn't match what gets reported or traded.
2. **Hardcoded behavioral constraints that should be ML-discoverable** — daily trade caps, consecutive loss cooldowns, and volatility-scaling bands are fixed human assumptions that should either be in the optimizer's search space or removed. (Session gate, max drawdown, and fixed $100/trade risk are intentional and correct — see Design Philosophy.)
3. **Institutional-level gaps in edge validation and survival** — probability calibration, statistical edge validation (SPA test), concept drift detection, meta-labeling, tail risk awareness, and smart kill switches. These separate "a good backtest" from "a system you can trust with capital."

**Bottom line:** The foundation is institutional-grade. The remaining work is (a) making modules internally consistent, (b) moving hardcoded constraints into the optimizer's domain, and (c) adding the edge reliability and survival layers that prove the edge is real and detect when it stops working.

---

## THE GOOD — What Puts This System Ahead of 95% of Retail

### G1. Triple Barrier Labeling — Correctly Implemented
**File:** `phase2_adaptive_engine.py` lines 452–513

The vectorized forward-scanning triple barrier is the single most important methodological improvement since ANALYSIS.md. It correctly:
- Uses `TB_SL_ATR=1.5`, `TB_TP_MULT=2.0`, `MAX_HOLD=50` bars
- Implements "first touch wins" semantics via an `unlabelled` boolean mask
- Awards TP when both barriers are touched in the same bar (conservative)
- Falls back to next-bar direction only if ATR is unavailable
- Purges the last `MAX_HOLD` rows via `dropna()`

The ML objective now directly matches the trading objective: "given this entry, does TP get hit before SL within 50 bars?" This eliminates the #1 critical gap from v1.

### G2. Walk-Forward with Boundary Purging
**File:** `train.py` lines 108–172

Three anchored expanding folds with `PURGE_BARS=10` at each train/OOS boundary. The implementation correctly:
- Builds folds from calendar years (OOS = most recent year per fold)
- Drops 10 bars from the end of train AND start of OOS at each boundary
- Falls back to 70/30 split if < 3 years of data
- Reports per-fold OOS accuracy for both XGB and RF
- Trains final production model on full dataset after fold evaluation

This eliminates the "12,750 evaluations on the same 15%" data-mining problem from v1.

### G3. Tiered Robustness Gating for Strategy Selection
**File:** `train.py` lines 592–673

The five-tier strategy activation system is exactly what was recommended:
- **Tier 1:** DSR > 0 + MC pass + Sensitivity ≥ 50 + robust flag
- **Tier 2:** DSR > 0 + MC pass (relaxes sensitivity)
- **Tier 3:** DSR > 0 only
- **Tier 4:** ER-filter only (drops robustness)
- **Tier 5:** Last resort fallback

ER remains the final ranker within the selected tier — capital efficiency ranks candidates, robustness gates filter candidates. This is the correct architecture.

### G4. Spread + Slippage in Backtest
**File:** `backtest_engine.py` lines 186–325

Half-spread at entry + `0.1 × ATR` slippage. Uses real `spread_mean` from tick data when available. Entry cost is directional (longs pay above close, shorts receive below). The `SPREAD_MAX = 2 × spread_median` filter rejects entries during abnormally wide spreads. This alone removes the largest source of backtest-to-live P&L gap for US30 CFD.

### G5. Institutional Feature Engineering Remains Best-in-Class
**File:** `institutional_features.py` — 877 lines, ~73 features

The tick-derived feature set is genuinely institutional:
- Session VWAP with proper daily reset, 3σ bands, anchored VWAP from pivot volume
- Volume profile (POC/VAH/VAL/HVN/LVN) with correct 70% value area expansion
- Quote-based order flow (renamed from "delta" to "quote_delta" — honest about CFD data)
- Absorption, failed auction, stacked imbalance detection
- 6-regime classifier (ADX, EMA slope, ATR ratio, HV20 percentile)
- Swing-based stop hunt detection, equal high/low detection
- Pin bar and wick analysis in ATR units

The rename from `delta`→`quote_delta` and `cvd`→`quote_cvd` shows intellectual honesty about CFD tick volume limitations. This is good practice.

### G6. Composite HTF Alignment
**File:** `pipeline.py` lines 141–175

Replaces the old single EMA55 crossover with a 5-factor weighted composite:
```
htf_composite = ema_dir(0.25) + vwap_dir(0.25) + delta_dir(0.20) + poc_dir(0.15) + regime_dir(0.15)
```
Patched into `phase2_adaptive_engine` via monkey-patch so `get_signal()` uses it transparently. Much more robust than a single moving average crossover.

### G7. Live Risk Management Stack
**File:** `live.py` — complete gate chain

Comprehensive risk gates are implemented:
- Session gate (20:30–01:00 London, DST-aware) — correctly hardcoded (market physics, not assumption)
- Equity-based daily loss limit (not balance — correctly includes floating P&L)
- Max drawdown circuit breaker (**35% equity** — hard account-survival limit)
- Max open positions (3)
- BE-aware risk capital cap (BE'd trades don't count toward exposure)
- Fixed $100/trade risk (1% of $10K starting capital, stays fixed as account grows)
- Volatility-adjusted sizing and consecutive loss cooldown (see U8 below — these are currently hardcoded but should be ML-discoverable)

The session gate, max DD breaker, daily loss limit, and fixed $100 risk are correctly **hard limits**. The behavioral patterns (cooldowns, vol-scaling band, daily trade cap) are a different matter — see THE UGLY section.

### G8. Safe Hot-Reload
**File:** `live.py` lines 312–349

Model files are reloaded only when:
1. Disk files are newer than last load
2. No open positions exist for that symbol

This prevents mid-trade model switches that could create inconsistent signal/risk states.

### G9. Sharpe from Daily Equity Returns
**File:** `backtest_engine.py` lines 399–407

Cross-TF strategy comparison now uses daily equity returns annualized with `√252`, not per-trade R-multiples. This makes Sharpe ratios comparable regardless of trade frequency — a strategy trading 3× daily and one trading 2× weekly are on the same denominator.

### G11. Fixed $100/Trade Risk — ER Accuracy By Design

The risk model uses **fixed $100 per trade** on a $10,000 starting account (= 1% of initial capital). Critically, this amount stays at $100 even as the account grows. This is the correct choice for ER measurement for two reasons:

**1. ER accuracy:** ER = total profit / max drawdown. If you compound (risk % of equity), winning early makes positions larger later, which makes drawdown dollars larger later, which distorts the ER denominator. A strategy that made $3,000 early then drew down $800 looks worse under compounding than it should, because those $800 drawdown dollars came from a larger position. Fixed sizing removes this noise — every trade contributes the same dollar stake to the ER calculation regardless of when it occurred.

**2. Comparable across time:** With fixed $100/trade, a trade in month 1 and a trade in month 36 have identical dollar stakes. This makes the ER denominator (max drawdown in $) directly comparable to the ER numerator (total profit in $), both measured in the same "unit" — the initial risk stake.

**The implication:** As the account grows, the effective risk % per trade shrinks (from 1% at $10K to 0.5% at $20K). This is intentional conservative behaviour — the absolute risk stays constant, the relative risk declines with success. If the account goal is to scale up, the $100 can be periodically reset (e.g., to 1% of current balance at the start of each quarter), which keeps the ER calculation clean within each period.

**ER formula correctness:** The `backtest_engine.py` ER calculation already uses `r100 = r * 100` normalization (each R = $100) — this is consistent with the fixed $100/trade model. Both the optimizer and the reporting engine should use `risk = 100.0` (fixed), not `balance * risk_pct / 100`.

### G10. Data Pipeline Quality
**File:** `tick_pipeline.py` — robust engineering

- Dukascopy CSV auto-detection (delimiter, header, timestamp format)
- EET→UTC timezone conversion with DST handling (`ambiguous="NaT"`, `nonexistent="shift_forward"`)
- Streaming chunks (500K ticks) prevent RAM issues on 16GB+ CSV files
- Price/spread/volume filters with rejection statistics
- Bid-price OHLCV (consistent with CFD execution price)
- Microstructure features computed at bar level from raw ticks

---

## THE BAD — Internal Inconsistencies That Undermine Optimization

### B1. Triple Barrier Label Parameters ≠ Trading Parameters (CRITICAL)

**The disconnect:**
- **Labeling** uses hardcoded `TB_SL_ATR=1.5`, `TB_TP_MULT=2.0` (`phase2_adaptive_engine.py` lines 461–462)
- **Optimization** searches `sl_atr ∈ [0.3, 5.0]`, `tp_mult ∈ [0.5, 8.0]` (`GA_BOUNDS`, lines 621–628)

The optimizer is free to select any `sl_atr` and `tp_mult` within its search range — which is correct per the design philosophy. But the ML models are **always** trained on labels generated with fixed 1.5/2.0 barriers. So the model learns to predict "will TP be hit at 3.0 ATR (1.5 × 2.0) before SL at 1.5 ATR?"

When the optimizer selects different values (e.g., `sl_atr=0.8`, `tp_mult=4.0`), the actual trade targets TP at 3.2 ATR and SL at 0.8 ATR — a completely different setup than what the model was trained to predict. The model's confidence score was calibrated for the 1.5/2.0 question, not the 0.8/4.0 question.

**Severity:** High. The wider the optimizer drifts from the labeling parameters, the worse the model's calibration becomes. A model that is 60% accurate at predicting the 1.5/2.0 outcome may be only 40% accurate at predicting a 0.8/4.0 outcome.

**Suggested fix direction — aligned with the two-stage philosophy:**
The optimizer should be free to search `sl_atr` and `tp_mult`. But the **labels** need to match what each genome actually trades. Three options:
- **(A) Per-genome relabeling (ideal but expensive):** During `ga_fitness()`, re-label the target column using the genome's own `sl_atr`/`tp_mult`. This makes the model's training objective match the genome's trading parameters. Cost: adds ~2s per genome evaluation (the vectorized labeling loop is fast).
- **(B) Multi-label columns (practical):** Pre-compute labels for several `sl_atr`/`tp_mult` combinations (e.g., 3–5 pairs spanning the search range). Each genome picks the closest pre-computed label column. Cost: one-time during feature engineering.
- **(C) Train multiple models per label set:** Train separate XGB/RF models for each label configuration, let the optimizer select which model + which params to use together. Most aligned with the "let ML decide" philosophy but highest compute cost.

Option (B) is the best balance of correctness and compute cost.

### B2. `ga_fitness()` Sharpe Formula ≠ `backtest_engine` Sharpe Formula

**`ga_fitness()` (phase2, lines 769–777):**
```python
trades_per_year = len(returns) / (n_days / 252.0)
sharpe = r_arr.mean() / (r_arr.std() + 1e-10) * np.sqrt(trades_per_year)
```

**`backtest_engine` (lines 399–407):**
```python
daily_ret = daily_eq.pct_change().dropna()
sharpe = daily_ret.mean() / (daily_ret.std() + 1e-10) * np.sqrt(252)
```

The optimizer selects parameters by maximizing one Sharpe formula (per-trade R-multiples × √trades_per_year). The reporting system evaluates the same parameters using a completely different Sharpe formula (daily equity returns × √252). These can give very different rankings for the same strategy — especially when trade frequency varies.

A strategy that trades once per day gets `√252 ≈ 15.87` annualization from both formulas (coincidentally aligned). But a strategy that trades 3× per day gets `√756 ≈ 27.5` from the optimizer but still `√252 ≈ 15.87` from the reporting engine. The optimizer will favor higher-frequency strategies disproportionately.

**Severity:** Medium-High. The optimizer's ranking of candidates does not match how they'll be scored and selected in the tiered gating system.

**Suggested fix direction:** Use daily equity returns in `ga_fitness()` as well, or at minimum use consistent annualization. The daily-equity version in `backtest_engine` is the more correct one.

### B3. `ga_fitness()` Has No Spread/Slippage Model (the Stage 2 gap)

**`ga_fitness()` (phase2, line 726):**
```python
entry = closes[i]   # zero cost entry
```

**`backtest_engine` (lines 318–322):**
```python
spread_cost = spread_arr[i] / 2.0
slip_cost   = atr_arr[i] * SLIP_FACTOR
total_cost  = spread_cost + slip_cost
entry       = closes[i] + direction * total_cost
```

This is the **key missing piece of the two-stage philosophy.** Stage 1 (unconstrained exploration) exists. But Stage 2 (re-optimize with real execution physics) does not. The `backtest_engine` includes spread+slippage for reporting, but the optimizer (`ga_fitness`) never sees those costs.

Spread and slippage are not human assumptions — they are **market physics** that cannot be optimized away. A US30 CFD trade physically costs spread + slippage regardless of what parameters you choose. These costs belong in the optimizer.

For US30 with typical ATR ~200 points: a tight `sl_atr=0.7` gives SL distance ≈ 140 points. Typical spread+slippage ≈ 23 points = **16% of the SL** is consumed at entry. Without seeing this, the optimizer will systematically prefer tighter SL strategies that look great with free execution but bleed to death in reality.

**Severity:** High. This is the single most impactful fix — it implements Stage 2 of the design philosophy.

**Suggested fix direction:** Add the same spread+slippage model from `backtest_engine` into `ga_fitness()`. The `spread_mean` column already exists in the dataframe. Three lines of code.

### B4. GA Never Wins vs Optuna (Bug)
**File:** `train.py` lines 304–316, `phase2_adaptive_engine.py` lines 827–829

`run_genetic_algo` returns `decode_genome(hof[0])` — a dict with trading params but **no `best_score` key**. The comparison logic:
```python
best = (opt_params if opt_params.get("best_score", -999) >
        ga_params.get("best_score", -999) else ga_params)
```

`ga_params.get("best_score", -999)` is always `-999`. Optuna's result is always chosen unless Optuna itself returns a score below -999 (impossible since the fitness floor is -999). The GA runs ~12,000 evaluations but its result is **never used**. This is wasted compute.

**Severity:** Medium. The GA explores a different search topology (population-based) that can find solutions Optuna's TPE sampler misses. Losing that diversity reduces optimization quality.

**Suggested fix direction:** Add `best["best_score"] = hof[0].fitness.values[0]` to `run_genetic_algo` before returning.

### B5. `ga_fitness()` Does Not Apply HTF Alignment

**`get_signal()` (phase2, lines 1105–1113):** Applies HTF probability nudge (±30% × htf_weight).

**`ga_fitness()` (phase2, lines 698–717):** Uses raw `(p_xgb + p_rf) / 2` with no HTF adjustment.

The optimizer searches `htf_weight ∈ [0.0, 1.0]` and `htf_tf ∈ [0, 15, 30, 60, 240]` — giving ML full freedom to decide whether and how much HTF alignment matters. This is the correct design. But the fitness evaluation never actually **applies** the HTF nudge to probabilities, so the optimizer has no feedback on whether `htf_weight=0.7` is better or worse than `htf_weight=0.1`. The HTF weight parameter is effectively invisible to the optimizer — it gets optimized based on noise.

Then in live execution, `get_signal()` does apply the HTF nudge, creating a signal pipeline the optimizer never evaluated.

**Severity:** Medium. The optimizer can't learn the value of HTF alignment if it never sees its effect.

**Suggested fix direction:** Apply the same HTF probability adjustment in `ga_fitness()`. This lets the optimizer truly discover whether HTF helps and how much weight it should get — consistent with "let ML decide."

### B6. Break-Even Stop = `entry ± 1.0` (Hardcoded Price Offset)

**`backtest_engine.py` line 237, `ga_fitness()` line 732:**
```python
t["sl"] = t["entry"] + t["dir"] * 1.0
```

The BE stop moves to entry + 1 price unit. For US30 at ~40,000, 1.0 point is reasonable. But this is a hardcoded value that doesn't adapt. The optimizer already searches `be_r ∈ [0, 1, 2, 3]` (whether and when to trigger BE), which is great — ML decides if BE is used at all. But the **offset after BE triggers** is fixed at 1.0 regardless of spread, ATR, or instrument.

**Severity:** Low for US30-only. The optimizer can set `be_r=0` (no BE) if BE isn't helpful, so this doesn't trap the system.

**Suggested fix direction:** Make the BE offset proportional to spread or a small fraction of ATR instead of a fixed constant, so it scales naturally with instrument and volatility.

### B7. `rr` Ghost Key in Stale Params File

`decode_genome()` no longer includes `rr` — correctly removed. The old params file still contains it but is stale (pre-ANALYSIS.md). Next training run will regenerate the file cleanly.

**Severity:** Non-issue once a fresh training run is done. Noted only for completeness.

---

## THE UGLY — Deeper Structural Concerns

### U1. The Three-System Divergence Problem

The optimizer (`ga_fitness`), the reporting backtest (`backtest_engine`), and the live signal (`get_signal`) should ideally be three views of the same system. Currently they diverge:

| Aspect | ga_fitness() (optimizer) | backtest_engine (reporting) | get_signal() (live) |
|--------|--------------------------|----------------------------|---------------------|
| Spread/slippage | None | Half-spread + 0.1×ATR | Real market spread |
| HTF alignment | Not applied | Not applied | Applied (±30% × htf_weight) |
| Sharpe formula | Per-trade R × √(trades/yr) | Daily equity × √252 | N/A |
| Signal ensemble | XGB + RF average | XGB + RF average | XGB + RF average |
| Label barriers | Fixed 1.5/2.0 | N/A (uses model predictions) | N/A (uses model predictions) |

**What should be in the optimizer (market physics):**
- Spread and slippage — these are real costs, not optional
- HTF probability nudge — this is part of the signal pipeline the optimizer's parameters control

**What is correctly hardcoded (market physics or account survival):**
- Session gate (20:30–01:00 London) — US30 is a closed market during these hours, not a behavioral assumption
- Max drawdown circuit breaker (35%) — hard account-survival limit
- Fixed $100/trade risk — intentional, for ER accuracy (see G11)

**What should NOT be hardcoded (human assumptions the optimizer should discover):**
- Daily trade cap → the optimizer controls trade frequency via confidence threshold
- Consecutive loss cooldown → this is a behavioural assumption; if the edge is real, the next trade after 4 losses has the same EV as any other (Mark Douglas)
- Fixed vol-sizing band → the optimizer can discover whether vol-scaling helps through its risk metrics

**The fix:** Split the current constraints into three categories:
1. **Physics (must be in optimizer):** spread, slippage, HTF nudge → add to `ga_fitness()`
2. **Hard limits (never optimized):** session gate, max DD breaker (35%), max daily loss %, max open positions, fixed $100/trade → keep exactly as-is
3. **Behavioral (currently hardcoded, should be ML-discoverable or removed):** daily cap, cooldown, vol-sizing band → either add as optimizer search dimensions or remove from the pipeline

### U2. Walk-Forward Folds Are Calendar Years, Not Market Regimes

The walk-forward implementation splits by calendar year. This is correct as a baseline. But for US30 specifically, market regimes don't align with calendar boundaries:

- 2020: COVID crash + recovery (extreme regime transition mid-March)
- 2021: Low-volatility bull run
- 2022: Bear market (FOMC tightening)
- 2023: Rangebound → bull transition
- 2024: Bull market, low vol
- 2025: Mixed (tariff uncertainty)
- 2026: Ongoing

A fold that trains on 2020–2023 and tests on 2024 is testing "does a model trained through a crash, bull, bear, and range work in a pure bull?" The answer may be yes, but it tells you nothing about what happens in the next bear.

**The deeper issue:** With only 3 folds and annual OOS periods, you get 3 data points about out-of-sample performance. The statistical significance of "average OOS Sharpe across 3 folds" is low. One anomalous year can dominate.

**Suggested improvement (no code change needed — analysis guidance):**
- Report per-fold OOS results prominently (not just the average)
- If fold 3 (most recent OOS) shows degradation vs folds 1–2, that's a deployment red flag
- Consider 5 folds with 6-month OOS periods for more data points (requires ≥ 5 years of data, which you have)

### U3. Minimum Trade Count as a Statistical Reality Check

The optimizer searches `confidence ∈ [0.50, 0.92]`. It is free to pick any value — high confidence = selective + high win rate + few trades; low confidence = more trades + lower win rate. This is correct: let ML decide.

**The concern is statistical, not philosophical.** If the optimizer selects a very high confidence (say 0.85+) combined with a wide TP multiplier, the resulting strategy may produce very few trades (< 30–50) over the full 6-year backtest. With < 50 trades:
- Monte Carlo bootstrap has too few data points to produce meaningful percentile estimates
- DSR correction is unreliable (skewness/kurtosis estimates are noisy with small N)
- Sensitivity analysis on 6 backtests of 30 trades each has high variance
- You cannot distinguish "genuinely selective model" from "model that fired 20 times and happened to win 15 by luck"

**The optimizer itself doesn't know this.** It optimizes Sharpe, which can look great on 20 trades if 15 are winners at 5:1 RR. But the confidence interval around that Sharpe is enormous.

**Suggested approach (consistent with "let ML decide"):** Don't hardcode a confidence cap. Instead, add a **minimum trade count penalty** to the fitness function:
```
if n_trades < MIN_CREDIBLE_TRADES:
    sharpe *= (n_trades / MIN_CREDIBLE_TRADES)  # progressive penalty
```

This lets the optimizer freely explore high-confidence strategies but naturally penalizes ones that produce too few trades to be statistically credible. The optimizer will find the balance point itself. De Prado suggests ~100 minimum; 200+ is preferable.

This is different from hardcoding `confidence_max=0.80` — it doesn't restrict the search space, it just tells the optimizer that few-trade strategies need to be proportionally more impressive to win.

### U4. The `bfill()` in Feature Engineering Introduces Lookahead

**File:** `institutional_features.py` line 781–782 (inside `add_institutional_features`):
```python
df.ffill(inplace=True)
df.bfill(inplace=True)
```

After computing all institutional features, NaN values are forward-filled (correct) then **backward-filled** (lookahead). `bfill` propagates a future valid value into earlier rows. For features that have NaN at the start of the series (VWAP, volume profile, rolling statistics that need warmup), this means the first N rows get values computed from future data.

In walk-forward training, the early rows of each fold's training set may contain backward-filled values from later in the series. The model learns from features that wouldn't exist at that point in real time.

**Severity:** Medium. Affects primarily the first 50–200 rows of each series (warmup period). In a 6-year dataset with millions of rows, this is a small fraction. But it violates causal ordering.

**Suggested fix direction:** Replace `bfill` with explicit handling: either drop NaN rows (since they're warmup rows with incomplete features anyway), or fill with neutral values (0 for z-scores, median for levels).

### U5. Regime Detection Overlap Creates Ambiguous Labels

**File:** `institutional_features.py` lines 624–630

The regime assignment uses sequential pandas boolean indexing where **last assignment wins**:
```python
regime = 0 (default)
regime[is_contracting]              = 5  # set
regime[is_expanding]                = 4  # overwrites 5 where both true
regime[is_bear_trend]               = 3  # overwrites 4 where both true
regime[is_bull_trend]               = 2  # overwrites 3 where both true
regime[~is_trending & is_high_vol]  = 1
regime[~is_trending & is_low_vol]   = 0
```

A bar can be simultaneously `is_expanding` (ATR slope > 0.02) and `is_bull_trend` (ADX > 25 + EMA21 > EMA55 + slope > 0). In that case, regime = 2 (bull trend overwrites expanding). This means **regime 4 (expanding) and 5 (contracting) can only appear when ADX < 25 (not trending)**. The docstring says regime 4 = "expanding volatility" but in practice it means "expanding volatility in a ranging market" — a much narrower condition.

This isn't necessarily wrong, but it means the regime feature carries less information than intended. "Trending + expanding vol" (a strong breakout) gets the same regime label as "trending + normal vol."

**Severity:** Low. The model receives `adx_norm`, `atr_ratio`, `atr_slope`, and `hv20_pct` as separate features, so it can reconstruct the nuance. But the `regime_sin`/`regime_cos` encoding loses this information.

### U6. Volume Profile Is Session-Window Rolling, Not True Session-Level

**File:** `institutional_features.py` — `add_volume_profile_features`

The volume profile is computed as a **rolling window** of `session_bars` (default 390) for every bar. This means:
- Bar at 10:00 AM: profile covers 9:30 AM ← 6.5 hours back (crosses into previous session)
- Bar at 3:30 PM: profile covers 10:00 AM ← same session, good
- Bar at 9:45 AM (near open): profile is 95% yesterday's afternoon data

**True institutional volume profile** resets at session open. The current rolling window creates a smoothed, time-lagging version that blends sessions. At session open (where you most need a clean profile), you're looking at yesterday's end-of-day volume distribution.

**Severity:** Medium. The feature still captures "where was volume concentrated recently" but it's not the same as the standard TPO/Market Profile that institutional desks use.

**Suggested fix direction:** Use session-grouped profiles (one per trading day), forward-filled within the session. The `add_vwap_features` function already does this for VWAP via `groupby(dates)` — the same approach should be applied to volume profile.

### U7. Sensitivity Analysis Uses Only Sharpe, Not ER

**File:** `backtest_engine.py` lines 590–654

The sensitivity scorer nudges confidence/sl_atr/tp_mult and measures **Sharpe change**:
```python
change = abs(st["sharpe"] - base_sharpe) / (abs(base_sharpe) + 1e-10)
score  = float(np.clip(1.0 - change, 0.0, 1.0))
```

But strategy **selection** uses ER (efficiency ratio). A strategy can have stable Sharpe across parameter nudges (high sensitivity score) but unstable ER (because drawdown is volatile across nudges). The gate passes it, but the ranking metric would have flagged it.

**Suggested fix direction:** Either score ER stability alongside Sharpe stability, or change the sensitivity test to use the same metric that ultimately selects strategies (ER).

### U8. Hardcoded Behavioral Constraints That Should Be ML-Discoverable

Several values in `backtest_engine.py` and `live.py` are hardcoded human assumptions that contradict the "let ML decide" philosophy. Note: the session gate (20:30–01:00 London), max drawdown (35%), and fixed $100/trade are **intentional and correct** — they are not listed here.

| Constraint | Current Value | File | Why It Should Be ML-Discoverable |
|-----------|---------------|------|----------------------------------|
| Daily trade cap | 6 trades/day | `backtest_engine.py` line 199, `live.py` line 193 | The optimizer controls trade frequency via `confidence` threshold. A high-confidence filter naturally limits trades. Hardcoding 6 means a strategy that legitimately wants 8 trades on a high-volatility day is blocked. |
| Consecutive loss cooldown | 4 losses → 0.5% risk for 6 trades | `backtest_engine.py` lines 199–202, `live.py` lines 190–192 | Mark Douglas: each trade is independent given the edge. If the model has a positive expected value, the 5th trade after 4 losses has the same EV as any other. Cooldowns also change the risk amount away from the fixed $100, which undermines ER accuracy. |
| Vol-sizing band | 0.5%–2.0% inverse ATR | `backtest_engine.py` lines 333–334, `live.py` lines 503–504 | With fixed $100/trade, vol-sizing is irrelevant — risk is already fixed. This code path no longer applies to the intended risk model. |
| Spread filter (2× median) | `SPREAD_MAX = spread_median * 2.0` | `backtest_engine.py` line 195 | Wide spread is already priced in at entry via the spread cost model. A separate 2× median filter double-counts it. The optimizer, seeing real spread costs, will naturally avoid entries during extreme spread events. |

**What must remain hardcoded:**
- Session gate (20:30–01:00 London) — market physics for US30
- Max drawdown circuit breaker (35%) — account survival
- Max daily loss % — single-day wipeout prevention
- Max open positions (3) — uncontrolled exposure cap
- Fixed $100/trade — ER accuracy by design

**Suggested approach:**
1. **Daily cap:** Remove the hard `6 trades/day` limit or raise it to a non-binding value (e.g., 50). Let the confidence threshold control frequency.
2. **Cooldown:** Remove the consecutive loss risk reduction. With fixed $100/trade, every trade has identical sizing — cooldowns have no meaning in this model. If you want protection after losses, the max drawdown breaker (35%) handles the extreme case.
3. **Vol-sizing band:** Remove entirely — superseded by fixed $100/trade model.
4. **Spread filter:** Remove the 2× median hard block. The spread cost is already in the entry price; let the ER metric discard bad-spread strategies naturally through the optimization.

**Alternative (optimizer-searchable):**
Add remaining behavioral constraints to `GA_BOUNDS` so ML discovers the values:
```
daily_cap              ∈ [4, 50]          (50 = effectively unlimited)
cooldown_after_losses  ∈ [0, 3, 5, 8]    (0 = no cooldown — likely wins)
```

---

## PARAMETER OPTIMIZATION ANALYSIS

### Search Space Design — Well Structured

The current `PARAM_SEEDS` / `GA_BOUNDS` give the optimizer genuine freedom:

| Parameter | Search Range | What ML Decides |
|-----------|-------------|-----------------|
| `entry_tf` | [1, 3, 5, 10, 15, 30] minutes | Which timeframe has the most exploitable signal |
| `htf_tf` | [0, 15, 30, 60, 240] (0 = none) | Whether HTF helps and which one |
| `sl_atr` | [0.3, 5.0] | Tight scalp SL vs wide swing SL |
| `tp_mult` | [0.5, 8.0] | Scalp (quick profit) vs runner (wide TP) |
| `confidence` | [0.50, 0.92] | Aggressive (many trades, lower WR) vs selective (few trades, higher WR) |
| `htf_weight` | [0.0, 1.0] | HTF influence (0 = ignore completely) |
| `be_r` | [0, 1, 2, 3] | Never/+1R/+2R/+3R trigger for moving stop to BE |

This is the correct approach: wide ranges, ML discovers the sweet spot. The `htf_tf=0` option and `be_r=0` option are particularly important — they let the optimizer say "HTF alignment doesn't help" or "break-even isn't worth it" rather than forcing these features on.

### What the Next Training Run's Params Will Reveal

After fixing B3 (adding spread/slippage to `ga_fitness()`), expect the optimizer to shift toward:
- **Wider `sl_atr`:** Tight SLs are punished more heavily when spread+slippage costs are visible
- **Lower `tp_mult`:** Realistic execution costs favor more achievable TP targets
- **Potentially lower `confidence`:** More trades can compensate for the per-trade cost haircut, if win rate supports it
- **Potentially different `entry_tf`:** Higher TFs have proportionally lower spread cost relative to move size

These shifts are healthy — they reflect reality. The current stale params (pre-ANALYSIS.md) optimized on free execution will look very different from params optimized with real costs.

### ER Calculation — Correctly Designed and Consistent with Fixed Risk Model

**File:** `backtest_engine.py` lines 423–431

```python
r100     = r * 100.0              # normalize each R = $100 at risk
eq100    = start_balance + np.cumsum(r100)
peak100  = np.maximum.accumulate(eq100)
dd100    = (peak100 - eq100).max()
profit100= eq100[-1] - start_balance
er       = (profit100 / (dd100 + 1e-10)) * er_multiplier
```

This is **profit per unit of max drawdown** — the correct capital efficiency metric. The `r100` normalization maps every trade to a fixed $100 stake, which is **exactly consistent with the fixed $100/trade risk model** (see G11). The backtest ER and the live ER are therefore measuring the same thing — no compounding distortion.

**Why this is the right design:** With a fixed $100/trade, the equity curve grows linearly (each win adds exactly RR × $100, each loss subtracts $100 × SL fraction). The ER numerator (total profit) and denominator (max drawdown) are both in the same unit — dollars at $100/trade. A strategy that earns $3,000 while drawing down $600 has ER = 5.0 regardless of when those trades occurred. This is order-independent and scale-independent, which is exactly what you want for comparing strategies across timeframes and testing periods.

`ER_MULTIPLIER=1.25` (from env) applies a constant scaling factor across all strategies equally — it shifts the absolute ER value but does not change relative rankings.

---

## LIVE-READINESS ASSESSMENT

### What Will Work

| Component | Assessment |
|-----------|-----------|
| MT5 order execution | Correct: bid for short, ask for long, IOC fill, 20-deviation |
| Session gate (20:30–01:00 London) | Correct: DST-aware, matches US30 market physics |
| Account survival gates (max DD 35%, daily loss, max positions) | Correct: equity-based, hard limits |
| Fixed $100/trade risk | Correct: consistent with ER accuracy design |
| Hot-reload | Correct: safe (no reload with open positions) |
| BE detection | Correct: monitors MT5 SL vs stored entry |
| DB logging | Correct: open/close lifecycle tracked |
| Consecutive loss cooldown | Misaligned with fixed $100/trade — changes risk away from $100, see U8 |
| Volatility-adjusted sizing | Superseded by fixed $100/trade model — see U8 |

### What Will Likely Diverge from Backtest (Before Fixes)

| Component | Expected Divergence | Root Cause |
|-----------|--------------------|----|
| Win rate | Lower than backtest by 5–15% | Triple barrier labels (1.5/2.0) don't match optimizer's chosen sl_atr/tp_mult (B1) |
| Spread cost impact | Understated in optimization | `ga_fitness()` has zero execution costs (B3) |
| HTF alignment | Not evaluated during optimization | `ga_fitness()` doesn't apply HTF nudge (B5) |
| Trade count | Potentially lower than optimizer saw | Daily cap (6 trades) in backtest_engine restricts trades the optimizer never saw blocked (U8) |

### What Will Likely Converge After Fixes

Once B1 (label alignment), B3 (spread in optimizer), and B5 (HTF in optimizer) are fixed and a fresh training run is done:
- The optimizer will have seen real costs → selected params will be cost-aware
- The model will predict the same barrier setup that's actually traded → calibrated win rates
- HTF weight will reflect actual HTF value → the optimizer can properly set it (or ignore it)
- **Expected backtest-to-live gap shrinks from 25–60% to 10–20%** (the residual is from real-time data quality vs historical tick data)

### Minimum Requirements Before Live Deployment

**Before training (code changes):**
1. **Apply fixes B3 + B1 + B5** (spread in optimizer, label alignment, HTF in optimizer)
2. **Add probability calibration (I2):** Isotonic regression on OOS predictions — makes confidence threshold meaningful
3. **Remove behavioral constraints from backtest_engine (Fix 8):** daily cap, cooldown, vol-sizing

**After training:**
4. **Run SPA test on top strategies (I1):** Reject any strategy that cannot beat the null at p < 0.05
5. **Verify trade count on new params:** If < 100 trades over the full backtest, add trade count penalty (U3)
6. **Check per-fold degradation:** If the most recent walk-forward fold shows significantly worse OOS, that's a deployment red flag
7. **Compare `ga_fitness()` output vs `backtest_engine` output** for the same params — should agree within 10%

**Before live capital:**
8. **Implement drift detection (I7):** PSI on top features + auto-disable on degradation
9. **Implement smart kill switch (I10):** Rolling Sharpe vs expected, graduated response
10. **Paper trade for 2–4 weeks:** Run in `DRY_RUN` mode, log `predicted_probability` with each trade. Compare signal frequency, win rate, and calibration curve against backtest expectations

---

## SUGGESTED CODE FIXES (NO CODE CHANGES — DESCRIPTION ONLY)

### Fix Priority 1: Add Spread/Slippage to `ga_fitness()` — STAGE 2 IMPLEMENTATION
**File:** `phase2_adaptive_engine.py` — `ga_fitness()` around line 726
**What:** Add the same spread+slippage cost model from `backtest_engine`. Load `spread_mean` from the dataframe (already present), compute `total_cost = spread_arr[i]/2 + atr[i]*0.1`, adjust entry price.
**Why:** This is the missing Stage 2 of the two-stage philosophy. Spread and slippage are market physics — the optimizer must see them. Without this, every parameter set is evaluated on a fantasy where execution is free.
**Impact:** Highest priority. Will fundamentally change what the optimizer selects. Expect wider SLs, more achievable TPs, and better cost-adjusted parameters.

### Fix Priority 2: Align Triple Barrier Labels with Optimizer-Selected Parameters
**File:** `phase2_adaptive_engine.py` — `engineer_features()` lines 460–462
**What:** Pre-compute labels for 3–5 `sl_atr`/`tp_mult` combinations spanning the search range. Store as separate columns (e.g., `target_tight`, `target_mid`, `target_wide`). Each genome in the optimizer picks the closest label column based on its `sl_atr`/`tp_mult` values.
**Why:** The ML model must predict the same setup it will actually trade. A model trained on 1.5/2.0 barriers cannot reliably predict outcomes for 0.8/4.0 barriers.
**Impact:** High — directly fixes the model calibration problem. The optimizer is free to search the full range; it just gets labels that match its genome.

### Fix Priority 3: Add HTF Nudge to `ga_fitness()`
**File:** `phase2_adaptive_engine.py` — `ga_fitness()` between lines 700 and 712
**What:** Apply the same HTF probability adjustment as `get_signal()`: if HTF is bearish and signal is long, `prob *= (1 - htf_w * 0.3)`. Requires loading the HTF dataframe for the genome's `htf_tf`.
**Why:** The optimizer searches `htf_weight` and `htf_tf` — it's free to decide if HTF helps. But without seeing the effect, it's optimizing those parameters blindly. This fix lets the optimizer **actually evaluate** whether HTF alignment improves or hurts performance.
**Impact:** Medium-High — enables the optimizer to make an informed decision about HTF.

### Fix Priority 4: Return `best_score` from GA
**File:** `phase2_adaptive_engine.py` — `run_genetic_algo()` line 827–829
**What:** Add `best["best_score"] = hof[0].fitness.values[0]` before returning.
**Why:** Currently GA result is never used regardless of quality (bug). The GA explores a population-based search topology that can find solutions Optuna's TPE misses.
**Impact:** Low-Medium — gives the system two diverse optimization approaches that compete fairly.

### Fix Priority 5: Remove `bfill()` from Feature Engineering
**File:** `institutional_features.py` — `add_institutional_features()` line 782
**What:** Replace `df.bfill(inplace=True)` with `df.fillna(0, inplace=True)` or simply drop NaN rows (warmup rows with incomplete features).
**Why:** `bfill` propagates future values into past rows (lookahead). Violates causal ordering.
**Impact:** Low — affects only warmup rows, but is a methodological correctness issue.

### Fix Priority 6: Align Sharpe Formulas Between Optimizer and Reporter
**File:** `phase2_adaptive_engine.py` — `ga_fitness()` lines 769–777
**What:** Build a simple daily equity curve from the simulated trades, compute Sharpe from daily returns with `√252` — matching the `backtest_engine` formula.
**Why:** The optimizer selects parameters by maximizing one Sharpe formula but they're evaluated and ranked by a different one. This creates rank-order disagreements.
**Impact:** Medium — ensures the optimizer's ranking matches the reporting system's ranking.

### Fix Priority 7: Add Minimum Trade Count Penalty to Fitness
**File:** `phase2_adaptive_engine.py` — `ga_fitness()` around line 766
**What:** Instead of a hard floor of 10 trades, add a progressive penalty: `if n_trades < 100: sharpe *= (n_trades / 100)`. This soft-penalizes low-trade strategies without hardcoding a trade cap.
**Why:** Small sample sizes produce unreliable statistics. The optimizer should prefer strategies with enough trades to be statistically credible, but this should be a gradient (not a cliff) so the optimizer can balance selectivity vs sample size.
**Impact:** Medium — prevents the optimizer from finding corner solutions with 15 trades at 12:1 RR.

### Fix Priority 8: Remove Hardcoded Behavioral Constraints from `backtest_engine`
**File:** `backtest_engine.py` — lines 199 (daily cap), 199–202 (cooldown), 333–334 (vol-sizing band), 195 (spread filter)
**What:** Remove daily cap, consecutive loss cooldown, vol-sizing band, and the 2× median spread filter from `backtest_engine.py`. Keep the session gate (20:30–01:00 London) and the 35% max drawdown circuit breaker — these are legitimate hard limits. Also replace the `balance * risk_pct / 100` risk calculation with fixed `risk = 100.0` to match the intended model.
**Why:** The backtest reports metrics with behavioral constraints the optimizer never saw. If the optimizer produces params that generate 10 trades/day and the backtest caps at 6, reported ER and Sharpe are wrong. The session gate stays because the optimizer's training data already excludes those hours (the gate applies to live data only), and the fixed $100 risk is the correct calculation for ER accuracy.
**Impact:** Medium-High — makes the reporting backtest consistent with the optimizer's view of reality and with the fixed-risk model.

### Fix Priority 9: Session-Level Volume Profile
**File:** `institutional_features.py` — `add_volume_profile_features()`
**What:** Group by trading session (calendar date) instead of rolling window. Compute one profile per session, forward-fill to each bar within that session.
**Why:** Rolling window blends sessions; institutional VP resets at session open. The `add_vwap_features` function already does session-level grouping via `groupby(dates)` — apply the same pattern.
**Impact:** Medium — improves feature quality, especially for morning entries where the current rolling window is mostly yesterday's data.

### Fix Priority 10: Report Missing Visualizations (from ANALYSIS_v2.md)
**File:** `report.py`
**What:** All 8 items from ANALYSIS_v2.md Section 4 remain unimplemented:
- Robustness traffic lights in strategy table
- Monte Carlo fan chart
- Return distribution histogram
- Rolling 90-day performance chart
- Day-of-week chart
- Sortino/Calmar/Max consec loss cards
- Live vs backtest comparison
- CFD data integrity note

The data for all of these exists in SQLite. These are pure visualization additions.
**Impact:** Low (doesn't affect trading performance) but high for operational confidence and monitoring.

### Fix Priority 11: Make Behavioral Constraints Optimizer-Searchable (Advanced)
**File:** `phase2_adaptive_engine.py` — `GA_BOUNDS` and `decode_genome()`
**What:** Add optional search dimensions for constraints that are currently hardcoded (session gate and max DD are excluded — they are intentional hard limits):
```
daily_cap             ∈ [4, 50]        (50 = effectively unlimited for intraday)
cooldown_after_losses ∈ [0, 3, 5, 8]  (0 = no cooldown — likely wins given fixed $100/trade)
```
**Why:** Instead of humans guessing "6 trades/day", let the optimizer discover the right value from data. With fixed $100/trade, cooldown (which changes risk sizing) is arguably redundant — but let the optimizer confirm this rather than hardcoding it.
**Impact:** Medium — expands the optimizer's freedom. But increases search space width, so may need more Optuna trials to converge well.

---

## QUANTITATIVE RISK ASSESSMENT

### Expected Backtest-to-Live Degradation

**Before fixes (current state — stale params, optimizer doesn't see costs):**

| Factor | Estimated Impact on Sharpe | Estimated Impact on ER |
|--------|---------------------------|------------------------|
| Spread/slippage not in optimizer | -15% to -30% | -10% to -20% |
| Triple barrier mismatch | -10% to -25% (win rate) | -10% to -20% |
| HTF alignment not in optimizer | -5% to -15% | -5% to -10% |
| Real-time data quality vs tick-derived | -5% to -10% | -5% to -10% |
| **Cumulative (before fixes)** | **-25% to -60%** | **-20% to -45%** |

**After fixes (spread in optimizer, label alignment, HTF in optimizer, fresh training):**

| Factor | Estimated Impact on Sharpe | Estimated Impact on ER |
|--------|---------------------------|------------------------|
| Residual spread model imprecision | -5% to -10% | -3% to -8% |
| Real-time data quality vs tick-derived | -5% to -10% | -5% to -10% |
| Regime shift since training end | Variable (±20%) | Variable (±15%) |
| **Cumulative (after fixes)** | **-10% to -20%** | **-8% to -18%** |

The fixes dramatically reduce the expected gap. A 10–20% Sharpe degradation is **normal and acceptable** for any trading system transitioning from backtest to live. Below 10% is unrealistic to expect.

### Capital Requirement Analysis (ER-Based)

ER tells you directly: **for every dollar of max drawdown endured, you earn ER dollars of profit.**

With **fixed $100/trade** and **35% max drawdown**, the capital requirement is straightforward:

| Scenario | Expected Live ER | Max Drawdown $ (35% of account) | Annual Profit = ER × Max DD $ | Starting Account |
|----------|-----------------|--------------------------------|-------------------------------|-----------------|
| $10K account | 3.0 | $3,500 | ~$10,500 | $10,000 |
| $10K account | 5.0 | $3,500 | ~$17,500 | $10,000 |
| $25K account | 5.0 | $8,750 | ~$43,750 | $25,000 |

**Important note on the fixed $100 model:** The drawdown is bounded by the 35% circuit breaker, but the profit potential grows with the number of trades per year (more trades × $100 × avg R-multiple = more profit). With a $10,000 account and fixed $100/trade, a 200-trade year at 0.5R average = $10,000 profit (100% annual return). The 35% max drawdown ($3,500) is the price you pay for this. An ER of ~2.9 would describe that.

**The key insight:** ER tells you the capital efficiency regardless of account size. An ER of 5 means for every $1 of max drawdown you experience, you earn $5 of profit. With fixed $100/trade this is a clean, account-size-independent measurement. To scale up: increase the $100 fixed risk per trade (e.g., to $200 = 2% of $10K or 1% of $20K), rerun ER — it should stay stable if the edge is real.

---

## WHAT MAKES THIS SYSTEM DIFFERENT FROM TYPICAL RETAIL

### Genuinely Strong

1. **Tick-derived features** — CVD, absorption, stacked imbalance, VWAP from real tick data (not broker OHLCV)
2. **Triple barrier labeling** — ML objective matches trading objective
3. **Walk-forward with purging** — proper out-of-sample evaluation
4. **Tiered robustness gating** — DSR/MC/sensitivity as hard filters before ER ranking
5. **ER as primary metric** — capital efficiency, not raw Sharpe
6. **BE-aware risk management** — protected positions don't count toward exposure cap
7. **"Let ML decide" philosophy** — wide optimizer search ranges, minimal hardcoded assumptions
8. **Two-stage optimization intent** — unconstrained exploration → physics-constrained refinement
9. **Fixed $100/trade risk** — ER remains accurate as account grows; clean linear equity curve
10. **Session gate (20:30–01:00)** — correctly hardcoded as market physics, not assumption

### Where It Falls Short of True Institutional

**Internal consistency (fix now — blocks everything else):**
1. Stage 2 not yet implemented — spread/slippage not in optimizer (fix priority 1)
2. Label-parameter misalignment — fixed barrier labels vs free optimizer search (fix priority 2)
3. Cooldown and daily cap contradict fixed $100/trade model and ML-decides philosophy (fix priority 8, 11)

**Edge reliability (fix before live):**
4. No probability calibration — confidence threshold operates on uncalibrated scale (I2)
5. No edge validation test (SPA/Reality Check) — cannot prove edge is real, not lucky (I1)
6. No concept drift detection — model will degrade silently in live (I7)
7. No smart kill switch — only the 35% DD breaker, no early warning (I10)

**Edge amplification (fix after live is stable):**
8. No meta-labeling — primary + meta model architecture (I6)
9. No tail risk awareness in optimization — Sharpe/ER miss left-tail crash risk (I3)
10. No regime-conditional models — one model for all 6 regimes

---

## INSTITUTIONAL-LEVEL GAPS — Edge Reliability & Survival

The preceding sections cover internal consistency and the two-stage philosophy. This section addresses a different question: **is the edge real, and will you know when it stops being real?** These are the gaps that separate a well-built backtest from a system an institutional desk would trust with capital. Each item is evaluated through the project's philosophy: let ML discover, fix only physics, use fixed $100/trade, US30 CFD spot.

### I1. No Edge Validation Beyond DSR (White's Reality Check / SPA Test)

**What exists:** Deflated Sharpe Ratio with a `√log(n_trials)` multi-testing penalty. This is good but insufficient given the search scale.

**The gap:** The system runs GA (12,000 evaluations) + Optuna (500 trials) + per-TF Optuna (150 trials × N timeframes). Across this search, the best result is almost certainly overfit to some degree — even with walk-forward and DSR. DSR applies a single penalty based on trial count. It does not test whether the best result is significantly better than what you'd expect from random parameter selection.

**What institutions add:**
- **White's Reality Check (2000):** Tests the null hypothesis "the best strategy is no better than a benchmark (e.g., buy-and-hold or zero return)" by bootstrapping the full set of strategy returns. If the best strategy cannot reject this null at p < 0.05, it's likely a data-mining artifact.
- **Hansen's SPA test (Superior Predictive Ability):** A more powerful version of Reality Check that controls for dependencies between strategies (your strategies share features and data — they're not independent).

**How it fits the philosophy:** This doesn't restrict the optimizer's freedom. It evaluates the optimizer's output after the fact. The pipeline becomes: `optimize → robustness gates → edge validation → deploy`. Edge validation is a statistical checkpoint, not a behavioral constraint.

**Practical implementation:** Run SPA test on the top-N strategies from each optimization run. If none reject the null, the optimization found noise, not edge — don't deploy. The `scipy.stats` bootstrap functions handle the heavy lifting. Cost: minutes of compute after each training run.

**Priority:** High — this is the difference between "looks good in backtest" and "statistically distinguishable from luck."

### I2. No Probability Calibration (CRITICAL for Threshold-Based Systems)

**What exists:** Raw ensemble probabilities `prob = (p_xgb + p_rf) / 2` compared against a confidence threshold.

**The gap:** XGBoost and Random Forest output probabilities that are almost never calibrated out of the box. A model that outputs `p=0.70` does not mean "70% chance of TP hit." The probability is an ordinal ranking (higher = more confident), not a calibrated frequency. This means:
- The confidence threshold the optimizer selects (e.g., 0.65) has no interpretable meaning
- Two different model retrains may produce completely different probability scales for the same edge
- The optimizer is tuning a threshold on an uncalibrated scale — it's finding "which raw score works" rather than "what confidence level is profitable"

**What institutions add:**
- **Platt scaling:** Fit a logistic regression on the model's outputs vs actual outcomes on a held-out calibration set. Fast, well-understood.
- **Isotonic regression:** Non-parametric calibration — maps outputs to observed frequencies without assuming a functional form. Better for ensemble models that may have non-monotonic calibration curves.

**How it fits the philosophy:** Calibration is not a constraint — it's a correction that makes the probability scale meaningful. The optimizer is still free to choose any confidence threshold. But after calibration, `p=0.70` actually means "70% of the time this model said 0.70, the trade hit TP." The threshold parameter now has real meaning instead of being an arbitrary cutoff on a distorted scale.

**Implementation detail:** After training XGB+RF on training data, hold out the OOS fold from walk-forward. Fit isotonic regression on `(predicted_proba, actual_outcome)` pairs from OOS. Store the calibrator alongside the model. Apply at inference time: `calibrated_prob = calibrator.predict(raw_prob)`.

**Cost:** Negligible — `sklearn.calibration.CalibratedClassifierCV` or manual `IsotonicRegression` in 10 lines.

**Priority:** High — this is a hidden leak. The entire signal chain depends on `prob >= threshold`, and that comparison is only meaningful if the probabilities mean what they claim.

### I3. No PnL Tail Risk Awareness in Optimization

**What exists:** Optimization on Sharpe (mean/std) and ranking on ER (profit/max_drawdown).

**The gap:** Sharpe treats upside and downside volatility identically. ER captures max drawdown but not the shape of the loss distribution. Two strategies with identical Sharpe and ER can have very different tail risk:

| | Strategy A | Strategy B |
|---|---|---|
| Sharpe | 1.8 | 1.8 |
| ER | 5.0 | 5.0 |
| Return skewness | -1.2 (fat left tail) | +0.3 (slight right skew) |
| Worst 5% of trades | -3.2R average | -1.1R average |
| Max single trade loss | -4.8R | -1.0R |

Strategy A has a hidden crash risk that Sharpe and ER don't see. Strategy B has clean, bounded losses.

**What institutions add to the fitness function or gating:**
- **Skewness:** Positive skew = occasional large wins, frequent small losses (trend-following profile). Negative skew = frequent small wins, occasional large losses (hidden blowup risk). Prefer positive or near-zero skewness.
- **Tail ratio:** `mean(top 5% of returns) / |mean(bottom 5%)|`. Values > 1.0 = right tail larger than left tail = good.
- **Expected Shortfall / CVaR (95%):** "What is the average loss in the worst 5% of trades?" This bounds the catastrophic scenario.

**How it fits the philosophy:** These are not constraints — they are additional metrics the optimizer or the gating system can use. Two approaches:
1. **In the fitness function:** `fitness = sharpe * (1 + 0.1 * skewness) * min(tail_ratio, 2.0)` — rewards positive skew and penalizes left-tail-heavy strategies without hardcoding a skewness minimum.
2. **In the gating system:** Add a tail risk gate alongside DSR/MC/sensitivity: reject strategies with CVaR(95%) worse than -2R or negative skewness worse than -1.5.

**Priority:** Medium-High — particularly important for US30 CFD where gap risk (weekend gaps, news spikes) creates left-tail events that don't appear in intraday bar data.

### I4. No Execution Regime Awareness

**What exists:** Static spread+slippage model (`spread_mean/2 + 0.1×ATR`).

**The gap:** Execution quality for US30 CFD varies dramatically by market condition. The static model treats a calm Tuesday afternoon and a CPI release minute identically. In reality:

| Condition | Spread (typical) | Slippage (typical) | Fill quality |
|-----------|-----------------|-------------------|--------------|
| Normal NY session | 1–3 pts | 0–2 pts | Good |
| London open (first 15 min) | 3–8 pts | 2–5 pts | Fair |
| NFP / CPI / FOMC (±5 min) | 10–30+ pts | 5–20+ pts | Poor, possible rejection |
| Overnight (session gate blocks this) | 5–15 pts | 3–10 pts | N/A (blocked) |

**What institutions add:** An execution quality score as a **feature** (not a hard filter — consistent with "let ML decide"):

```
execution_quality = f(spread_percentile, tick_velocity_z, quote_update_frequency)
```

This feature goes into the ML model's input. The model can learn to lower its confidence during poor execution conditions. The optimizer can discover whether execution quality matters by observing its importance.

**For US30 CFD specifically:** A simple approach — add `spread_pct_of_atr` (already partially computed) and `tick_velocity_z` (already in `tick_pipeline.py`) as explicit features. The model can learn that signals during high `spread_pct_of_atr` periods are less reliable.

**News events (NFP, CPI, FOMC):** Create a binary feature `is_high_impact_news_window` from a static economic calendar (released months in advance). This is not prediction — it's known schedule data. The model decides whether to avoid these windows or embrace them.

**Priority:** Medium — the static spread model handles 80% of this. The remaining 20% (news events, auction chaos) matters for live resilience but won't dramatically change backtest results.

### I5. Trade Return Autocorrelation (Not IID)

**What exists:** Monte Carlo resampling with IID assumption (bootstrap with replacement on per-trade R-multiples).

**The gap:** The MC simulation assumes each trade's outcome is independent. In reality, trade returns cluster — winning streaks and losing streaks occur more often than IID predicts because:
- Market regimes persist (trending markets produce consecutive trend-following wins)
- Volatility clusters (GARCH-like behaviour in US30 — high-vol day is followed by high-vol day)
- Feature autocorrelation (the same conditions that generated signal N often persist for signal N+1)

**Impact:** IID Monte Carlo is slightly optimistic. It underestimates the probability of prolonged drawdowns (because it randomly scatters losses) and overestimates recovery speed (because it randomly scatters wins). The `mc_pass` gate may pass strategies that would fail under realistic autocorrelated sequences.

**What institutions add:**
- **Block bootstrap:** Resample in blocks of 5–10 consecutive trades instead of individual trades. This preserves local autocorrelation structure while still randomizing the overall sequence.
- **Return autocorrelation check:** Compute `autocorr(lag=1)` on trade returns. If significantly positive, the IID MC is meaningfully biased. If near zero, IID is fine.

**How it fits the philosophy:** This is a methodological improvement to an existing tool, not a new constraint. Replace the IID bootstrap in `run_monte_carlo()` with a block bootstrap. The optimizer's freedom is unchanged.

**Priority:** Medium — matters more for strategies with high trade frequency where autocorrelation has more data points to manifest. For strategies trading 2–3 times daily on US30, this is relevant.

### I6. Meta-Labeling (Two-Model Architecture)

Recommended in ANALYSIS.md, still not implemented. This is the single largest potential improvement to the signal chain.

**Current:** `features → XGB+RF ensemble → probability → threshold → trade`

**Institutional:** `features → primary model (direction) → meta-model (should we trade this signal?) → trade/reject`

**Why it matters for this system specifically:** The primary XGB+RF model is trained to predict direction (TP vs SL). It does not learn *when it is wrong*. A meta-model trained on the primary model's mistakes learns exactly this — it identifies the conditions under which the primary model's signals are unreliable and filters them out.

**How it fits the philosophy:** The meta-model is ML deciding whether to trust ML. No hardcoded rules, no human assumptions. The meta-model receives: primary confidence, regime, spread conditions, recent model accuracy, feature stability. It outputs a single probability: "will this specific trade succeed?" The optimizer's confidence threshold then applies to the meta-model's output, not the primary model's output.

**Implementation path:**
1. Train primary XGB+RF as now
2. Generate primary predictions on training data (OOS fold from walk-forward)
3. Build meta-features: `[primary_prob, regime, spread_pct_of_atr, recent_5_trade_wr, feature_stability]`
4. Train meta-model (RF or logistic regression) on: "did this specific primary signal actually hit TP?"
5. In live: primary fires → meta-model evaluates → if meta says >0.45 → trade

**Expected impact:** Typical meta-labeling results in published research show 30–50% drawdown reduction with moderate Sharpe improvement. The filtered signal set is smaller but higher quality.

**Priority:** High — this is the biggest edge boost remaining after the internal consistency fixes.

### I7. No Concept Drift Detection (CRITICAL for Live Survival)

**What exists:** No formal drift detection. The system will trade the same model indefinitely until manually retrained.

**The gap:** Every ML model eventually degrades because the data distribution shifts. For US30 CFD:
- Fed policy regime changes (2022 hiking → 2024 pausing → ? cutting)
- Volatility regime shifts (VIX 12 → VIX 30 changes everything)
- Market structure changes (algorithmic participation increases, spread dynamics shift)
- Broker-specific changes (Dukascopy quoting algorithm updates change the tick volume features)

Without drift detection, the model silently becomes unreliable. By the time you notice from P&L, you've already given back weeks of profit.

**What institutions add:**
- **Population Stability Index (PSI):** Compare the distribution of each key feature (top 10 by importance) over the last 30 days vs the training period. PSI > 0.2 = significant shift. PSI > 0.5 = population has fundamentally changed.
- **KL divergence:** Same concept, information-theoretic measure. Works well for continuous features.
- **Prediction confidence distribution shift:** If the model's average output probability drifts (e.g., from mean 0.55 in training to mean 0.48 in recent live), the model's internal landscape has shifted.

**Automatic response (aligned with philosophy — no human judgment needed):**
1. PSI > 0.2 on 3+ important features → log warning, continue trading
2. PSI > 0.5 on any top-5 feature → reduce to 50% sizing ($50/trade), log alert
3. Rolling 30-trade live win rate < 35% (when backtest expected 50%) → auto-disable, trigger retrain
4. After retrain, require paper-trade validation before re-enabling

**Priority:** Critical for live deployment — without this, every model eventually dies silently.

### I8. No Dynamic Capital Allocation Layer

**What exists:** Single strategy → fixed $100/trade.

**What institutions add:** Even with a single strategy, capital allocation adjusts based on edge confidence:
- **Scale up** during periods of demonstrated edge stability (rolling Sharpe > expected for 30+ trades)
- **Scale down** during underperformance (rolling Sharpe < 50% of expected)

**How it fits the fixed $100 model:** Rather than changing $100 per trade (which would break ER accuracy), the allocation layer controls **whether to trade at all**:
- Full confidence: trade every signal ($100/trade)
- Reduced confidence: trade only signals above a higher threshold (e.g., top 50% of predictions)
- Degraded: stop trading, trigger retrain

This is conceptually a **dynamic confidence threshold** that tightens or loosens based on recent live performance. The optimizer sets the baseline threshold; the allocation layer adjusts it in real-time based on observed edge quality.

**This is distinct from the cooldown mechanism** (which was correctly identified as a behavioral assumption in U8). Cooldowns trigger on consecutive losses (which are random). The allocation layer triggers on **statistical degradation** (which is systematic).

**Priority:** Medium — important for live longevity, but the drift detection in I7 handles the extreme case. This adds the gradual response in between.

### I9. No Latency / Missed Fill Modeling

**What exists:** Slippage model (`0.1 × ATR` per entry).

**The gap:** For US30 CFD on M1/M3 timeframes, the gap between "bar closes, signal fires" and "order reaches MT5 server" is typically 500ms–2s. During fast-moving markets (spike bars, news reactions), price can move 10–30+ points in that window. The static slippage model doesn't capture:
- **Execution delay:** The 1–2s between signal and fill
- **Missed fills:** During spikes, the order may be rejected or filled at a significantly worse price
- **Requotes:** CFD brokers can requote during high volatility

**Realistic impact for US30 CFD spot:** At normal conditions, the 0.1×ATR slippage model is sufficient. During fast moves, actual slippage can be 3–5× the model. This affects maybe 5–10% of trades but those are often the high-conviction signals (because volatility triggers confidence in the model).

**Practical approach (no code change — analysis):** After paper trading begins, log actual fill price vs expected price for every trade. Compute the distribution of `actual_slippage / modeled_slippage`. If the ratio is consistently > 1.5, increase `SLIP_FACTOR` from 0.1 to a value that matches observed reality. This is calibration from live data, not a guess.

**Priority:** Low for now — the static model is adequate for initial deployment. Calibrate from live data after paper trading.

### I10. Kill Switch Should Be Smarter Than Max DD

**What exists:** Max drawdown circuit breaker at 35%.

**The gap:** 35% drawdown is a catastrophic last resort. By the time it triggers, the system has lost a third of the account. Institutional kill switches trigger much earlier based on **statistical deviation from expected behavior**:

| Signal | Threshold | Action |
|--------|-----------|--------|
| Rolling 20-trade Sharpe < 50% of backtest expected | 2 weeks of underperformance | Log warning |
| Rolling 50-trade win rate < backtest expected - 10pp | Statistically significant (binomial test p < 0.05) | Reduce to paper-trade mode |
| 3+ consecutive days with zero signals (when 2+/day expected) | Model may be broken | Alert, check model health |
| Max drawdown hits 15% | Halfway to circuit breaker | Automatic sizing reduction to $50/trade |
| Max drawdown hits 35% | Circuit breaker | Full stop |

**How it fits the philosophy:** These are not behavioral constraints — they are statistical tests against the model's own expected performance. The thresholds are derived from the backtest (expected win rate, expected Sharpe, expected signal frequency), not from human intuition. They answer: "is the model still performing within its own historical distribution?"

**Priority:** High for live deployment — this is the early warning system. The 35% breaker is the airbag; this is the seatbelt.

### I11. Feature Redundancy and Importance Stability

**What exists:** ~73 features fed to XGB+RF. XGB feature importance computed and displayed in report.

**The gap:** With 73 features, many are likely correlated (e.g., `vwap_dist_atr` and `vwap_dist_pct` measure the same thing in different units). Redundant features:
- Inflate the effective dimensionality → more overfitting risk
- Make feature importance unreliable (importance splits arbitrarily between correlated pairs)
- Slow down training and inference

**What institutions add:**
- **Correlation pruning:** Remove one of any pair with `|corr| > 0.90`. This typically removes 10–20% of features with zero loss in model quality.
- **SHAP importance stability across walk-forward folds:** If a feature is important in fold 1 but irrelevant in fold 3, it's regime-dependent and potentially unreliable. Stable importance across folds = robust feature.

**How it fits the philosophy:** Feature pruning happens pre-training, not during optimization. It reduces the noise the ML model has to sort through. This doesn't constrain what the model learns — it removes garbage from its input.

**Priority:** Medium — meaningful for model quality but not blocking. Can be done in a separate pass after the main fixes.

### I12. No Trade Quality Feedback Loop

**What exists:** Live trades logged to SQLite with entry/exit/pnl.

**The gap:** The system stores outcomes but doesn't track calibration over time. For each live trade, you should store:
- `predicted_probability` (what the model said)
- `actual_outcome` (1 = TP hit, 0 = SL hit)

Then continuously compute a **live calibration curve:** group predictions into bins (0.5–0.6, 0.6–0.7, 0.7–0.8, etc.) and compare predicted probability vs actual win rate in each bin.

**Why this matters:** If the model says 0.75 confidence and the actual win rate for trades in the 0.70–0.80 bin is 0.45, the model is overconfident. This is drift that you can detect much earlier than from raw P&L — because even a miscalibrated model can be profitable for a while if the ER is high enough. By the time P&L shows the problem, the calibration drift has been present for weeks.

**How it fits the philosophy:** This is pure observation — no constraint, no behavioral rule. It's the data that tells you when the edge is degrading. Combined with I7 (drift detection), it creates a complete monitoring system.

**Priority:** Medium — implement when live trading begins. Requires only logging `predicted_probability` alongside each trade (a one-line DB change).

---

### Institutional Gap Priority Summary

| Priority | Item | Category | Impact |
|----------|------|----------|--------|
| **Critical** | I2. Probability calibration | Edge quality | Fixes the meaning of confidence threshold — every other optimization depends on this scale |
| **Critical** | I7. Concept drift detection | Edge survival | Without this, every model dies silently in live |
| **High** | I1. Edge validation (SPA test) | Edge reality | Answers "is this edge real or lucky?" |
| **High** | I6. Meta-labeling | Edge boost | 30–50% DD reduction, Sharpe improvement |
| **High** | I10. Smart kill switch | Account survival | Early warning before 35% DD triggers |
| **High** | I3. Tail risk metrics | Risk awareness | Prevents selecting strategies with hidden crash risk |
| **Medium** | I5. Block bootstrap MC | Methodology | Makes MC more realistic for autocorrelated returns |
| **Medium** | I8. Dynamic capital allocation | Live adaptation | Scales confidence threshold based on live performance |
| **Medium** | I11. Feature pruning / stability | Model quality | Reduces overfitting risk from redundant features |
| **Medium** | I12. Trade quality feedback loop | Monitoring | Detects calibration drift before P&L shows it |
| **Low** | I4. Execution regime features | Realism | Adds execution quality as ML feature |
| **Low** | I9. Latency / fill modeling | Realism | Calibrate from live data after paper trading |

### The Core Principle These Items Share

> **Institutional systems don't just optimize the edge — they optimize the reliability of the edge, the survival of the edge, and the detection of when the edge disappears.**

The system currently optimizes the edge (stage 1) and is building the reliability layer (robustness gates, walk-forward). The items above complete the picture: I2 makes the edge measurement accurate, I1 proves it's real, I6 amplifies it, I7/I10/I12 detect when it dies, and I3/I5 ensure you survive the bad scenarios along the way.

---

## FINAL VERDICT

**The system's design philosophy is correct: give ML maximum freedom to discover optimal parameters, then layer on only real-world physics.** This is how the best quantitative firms operate. The tick data pipeline, institutional features, triple barrier labeling, walk-forward validation, tiered robustness gating, and ER-based ranking are all genuinely institutional-grade implementations.

**The remaining work falls into three layers:**

**Layer 1 — Internal consistency (fix now, 2–3 days):**
1. Stage 2 implementation — spread+slippage in `ga_fitness()` (highest single impact)
2. Label alignment — multi-label columns matching optimizer's sl_atr/tp_mult search range
3. HTF visibility — apply HTF nudge in `ga_fitness()` so optimizer can evaluate it
4. Remove behavioral constraints (daily cap, cooldown, vol-sizing) from `backtest_engine`

**Layer 2 — Edge reliability (fix before live, 3–5 days):**
5. Probability calibration (I2) — isotonic regression on OOS predictions, makes confidence threshold meaningful
6. Edge validation (I1) — SPA test after optimization, proves edge is real not lucky
7. Concept drift detection (I7) — PSI on top features, auto-disable on degradation
8. Smart kill switch (I10) — rolling performance vs expected, graduated response before 35% DD

**Layer 3 — Edge amplification (fix after live is stable):**
9. Meta-labeling (I6) — two-model architecture, biggest remaining edge boost
10. Tail risk metrics (I3) — skewness/CVaR in fitness function or gating
11. Block bootstrap MC (I5) — preserves autocorrelation in Monte Carlo
12. Trade quality feedback loop (I12) — live calibration curve over time

**Estimated timeline:**
- Layer 1: 2–3 days of code changes + 1 day training
- Layer 2: 3–5 days of code changes (can overlap with paper trading)
- Paper trading: 2–4 weeks
- Layer 3: ongoing improvements after live is stable
- Total to credible live: 4–6 weeks

**The foundation is strong. The three intentional hardcoded constraints (session gate, 35% max DD, fixed $100/trade) are correctly placed and well-reasoned. The core philosophy — let ML discover, constrain only physics — is correct. The remaining work is (a) making the optimization pipeline honest, (b) proving the edge is real and not lucky, and (c) knowing when it stops working.**

**The system currently optimizes the edge. After these additions, it will optimize the edge, prove the edge, and protect the edge — which is what separates a good backtest from a deployable trading system.**
