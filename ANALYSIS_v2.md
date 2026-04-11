# ML Trading System — Implementation Audit (v2)
**Date:** 2026-04-10  
**Based on:** ANALYSIS.md recommendations vs current code state  
**Purpose:** What is done, what is partial, what is still missing

---

## Status Key
- ✅ Fully implemented and correct
- ⚠️ Partially implemented — gap described
- ❌ Not yet implemented
- 🔵 Future feature (agreed to defer)

---

## SECTION 1 — Critical Architecture Fixes

### 1.1 LSTM + PPO Removed from Live Signal ✅
**Confirmed in:** `phase2_adaptive_engine.py` lines 1084–1094

`get_signal()` now uses only XGB + RF:
```python
p_xgb, p_rf = 0.5, 0.5
# ... load models ...
prob = (p_xgb + p_rf) / 2.0
```
No `p_lstm`, no `p_rl`, no PPO anywhere in the signal path. Backtest and live now use the identical signal. This is the most important fix and it is done correctly.

---

### 1.2 Triple Barrier Labeling ✅
**Confirmed in:** `phase2_adaptive_engine.py` lines 452–507

Fully implemented with vectorized forward-scanning:
- `TB_SL_ATR = 1.5`, `TB_TP_MULT = 2.0`, `MAX_HOLD = 50`
- Correct "first touch wins" semantics
- "Both touched same bar" → TP wins (conservative)
- Falls back to next-bar direction only if ATR is unavailable
- `dropna()` correctly purges the last MAX_HOLD rows that have no label

This replaces the old `(c.shift(-1) > c).astype(int)` target. The ML objective now directly matches the trading objective.

---

### 1.3 `rr` Parameter Partially Removed ⚠️
**Confirmed in:** `train.py` (removed from upsert/logging), `live.py` (params dict has no `rr`), `backtest_engine.py` (TP uses `tp_mult`)

**What is done:**
- `live.py` params dict uses only `tp_mult` — no `rr` passed to live trades
- `train.py` logs only `tp_mult`, upserts only `tp_mult` to DB
- `backtest_engine.py` TP calculation: `tp = entry + direction * sl_dist * tp_mult` ✅

**What is still broken:**
`backtest_engine.py` still reads `rr` from params and uses it for P&L calculation:
```python
# backtest_engine.py
rr     = params.get("rr", 2.0)      # reads rr ...
# ...
elif bh >= t["tp"]:
    pnl_r = t["rr"]                 # ... and uses it for reward
```

And `phase2_adaptive_engine.py` `ga_fitness()` still uses `params["rr"]` for reward:
```python
if bar_high >= tp: pnl = risk * params["rr"]   # should be tp_mult
```

**And `decode_genome()` still decodes both:**
```python
"rr":         float(np.clip(genome[3], GA_BOUNDS[3][0], GA_BOUNDS[3][1])),
"tp_mult":    float(np.clip(genome[4], GA_BOUNDS[4][0], GA_BOUNDS[4][1])),
```

**The inconsistency:** TP distance is set by `tp_mult` but the reward when TP is hit is calculated using `rr`. These are independent Optuna parameters that can diverge (e.g. `tp_mult=3.0` sets TP 3R away, but `rr=2.0` records the win as only 2R profit). The optimization fitness is wrong by construction.

**Fix required:**
In `backtest_engine.py` and `ga_fitness()`: replace `pnl = risk * params["rr"]` with the actual achieved R-multiple:
```python
# When TP is hit, actual R = tp_dist / sl_dist = tp_mult
pnl_r = params["tp_mult"]   # replaces t["rr"]
```
Then remove `rr` from `PARAM_SEEDS`, `GA_BOUNDS`, and `decode_genome()` entirely.

---

### 1.4 Walk-Forward 3-Fold Validation ✅
**Confirmed in:** `train.py` lines 111–172

Correctly implemented:
- 3 anchored expanding folds (OOS = most recent year per fold)
- `PURGE_BARS = 10` dropped at each train/OOS boundary (prevents label leakage)
- Falls back to 70/30 split if < 3 years of data
- Per-fold OOS accuracy reported for XGB and RF
- Final production model trained on full dataset after fold evaluation

Walk-forward OOS accuracy is logged per TF so you can see if performance degrades on recent folds vs older ones.

---

### 1.5 Boundary Purging (CPCV Partial) ✅
**Confirmed in:** `train.py` line 108, lines 153–154

`PURGE_BARS = 10` is applied at both ends of each fold boundary. This is the most important part of CPCV and it is done. Full CPCV (combinatorial paths) remains a future feature.

---

## SECTION 2 — Backtest Quality

### 2.1 Spread + Slippage Model in Backtest ✅
**Confirmed in:** `backtest_engine.py` lines 187–189, 318–322

```python
spread_arr  = df["spread_mean"].values if "spread_mean" in df.columns else np.zeros(len(df))
SLIP_FACTOR = 0.1
# ...
spread_cost = spread_arr[i] / 2.0
slip_cost   = atr_arr[i] * SLIP_FACTOR
```

Half-spread on entry + 0.1×ATR slippage. For US30 this is realistic for normal conditions. For high-volatility periods (news events) the `spread_mean` from tick data will naturally capture the wider spread. Correctly not applied on the exit side (SL/TP are limit-order equivalents).

---

### 2.2 Sharpe from Daily Equity Returns ✅
**Confirmed in:** `backtest_engine.py` lines 395–407

```python
daily_ret = daily_eq.pct_change().dropna()
if len(daily_ret) > 1:
    sharpe = float(daily_ret.mean() / (daily_ret.std() + 1e-10) * np.sqrt(252))
```

Correct. Daily equity returns annualized with `sqrt(252)`. Makes cross-TF Sharpe comparisons valid since all strategies are now on the same daily-return denominator.

**Remaining issue in `ga_fitness()`:** The optimization fitness function still uses per-trade R-multiples with `sqrt(252)`:
```python
# phase2_adaptive_engine.py ga_fitness()
sharpe = r_arr.mean() / (r_arr.std() + 1e-10) * np.sqrt(252)
```
This is the wrong formula for the same reasons described in ANALYSIS.md. The optimizer is selecting parameters against an incorrectly-annualized Sharpe. The backtest and the optimizer are now using different Sharpe formulas, which means Optuna is optimizing for a metric that doesn't match what gets reported and selected.

**Fix:** In `ga_fitness()`, replace per-trade R Sharpe with a simple trade-frequency-adjusted version:
```python
# In ga_fitness() — replace the sharpe calculation
if len(returns) < 10:
    return (-999.0,)
r_arr  = np.array(returns)
# Annualise by trade frequency rather than calendar days
# n_trades per year = len(returns) / (backtest_days / 252)
n_days = max((df.index[-1] - df.index[0]).days, 1)
trades_per_year = len(returns) / (n_days / 252.0)
sharpe = r_arr.mean() / (r_arr.std() + 1e-10) * np.sqrt(trades_per_year)
```

---

### 2.3 Sortino + Calmar Computed and Saved ✅
**Confirmed in:** `backtest_engine.py` and `train.py` lines 447–448

`sortino` and `calmar` are computed in `run_backtest()` and saved to the DB via `upsert_strategy()`. They appear in the training log summary table. Not yet in the HTML report (see Section 4).

---

## SECTION 3 — Risk Management & Live Trading

### 3.1 DSR + Monte Carlo + Sensitivity Gates ✅
**Confirmed in:** `train.py` lines 624–673 (`_activate_best_strategy()`)

Four-tier gating correctly implemented:
- **Tier 1:** DSR > 0 + MC pass + Sensitivity ≥ 50 + robust = 1
- **Tier 2:** DSR > 0 + MC pass (drops sensitivity)
- **Tier 3:** DSR > 0 only
- **Tier 4:** ER-filter only (drops all robustness)
- **Tier 5:** Last resort fallback (no ER cap)

Logs which tier was used and why. ER is the final ranker among eligible candidates — exactly as recommended. This is fully correct.

---

### 3.2 Session Gate ✅
**Confirmed in:** `live.py` lines 119–135 (`_is_session_blocked()`)

Blocks entries 20:30–01:00 **London time** (DST-correct via `pytz` / `America/New_York` equivalent using `Europe/London`). Correctly covers post-NYSE-close overnight hours for US30.

---

### 3.3 Daily Trade Cap ✅
**Confirmed in:** `live.py` line 193 (`_FilterState.DAILY_TRADE_CAP = 6`)

6 trades maximum per calendar day per symbol. Resets at midnight London time. Correct.

---

### 3.4 Consecutive Loss Cooldown ✅
**Confirmed in:** `live.py` lines 190–192, 218–230

- Triggers after **4 consecutive losses** (`CONSEC_LOSS_CAP = 4`)
- Reduces risk to **0.5%** (`CONSEC_COOL_RISK_PCT = 0.5`)
- Lasts for **6 trades** (`CONSEC_COOL_TRADES = 6`)

Matches exactly what was agreed in Q8. Win resets the counter. Cooldown decrements per trade opened (not per bar).

---

### 3.5 Volatility-Adjusted Position Sizing ✅
**Confirmed in:** `live.py` lines 501–517

```python
norm_atr = cur_atr / med_atr   # current ATR vs 100-bar median
base_pct = float(np.clip(RISK_PCT / norm_atr, VOL_SIZE_MIN, VOL_SIZE_MAX))
# VOL_SIZE_MIN = 0.5,  VOL_SIZE_MAX = 2.0
```

Inverse ATR scaling: high vol → smaller size, low vol → larger size. Capped at 0.5%–2% band. Consecutive loss cooldown overrides this (the stricter cap wins).

---

### 3.6 Composite HTF Alignment ✅
**Confirmed in:** `pipeline.py` lines 146–180 (`add_htf_alignment_full()`)

Replaces the old single EMA55 crossover with a 5-factor composite:
```python
htf_composite = (ema_dir*0.25 + vwap_dir*0.25 + delta_dir*0.20
                 + poc_dir*0.15 + regime_dir*0.15)
```
- EMA direction (price vs EMA55)
- VWAP direction (price vs session VWAP)
- CVD slope (cumulative volume delta trend)
- POC distance (price position vs volume profile POC)
- Regime (bull/bear regime flags)

Patched into `phase2_adaptive_engine` via `p2.add_htf_alignment = add_htf_alignment_full` so `get_signal()` uses the upgraded version transparently.

---

### 3.7 Equity-Based Daily Loss Calculation ✅
**Confirmed in:** `phase2_adaptive_engine.py` line 1184

```python
daily_loss_pct = max(dd_pct, 0.0)   # equity drawdown from session start
```

Uses `account_info().equity` drawdown instead of balance. Open position losses now count toward the daily limit. Fixes the original bug where open losing trades were invisible to the daily loss gate.

---

## SECTION 4 — Report / HTML Missing Items

These are all in ANALYSIS.md as recommended additions to `report.py`. None have been implemented yet.

### 4.1 Robustness Traffic Lights in Strategy Table ❌
**Status: Not implemented**

The strategy table in `report.py` still does not show DSR, MC pass/fail, or sensitivity score columns. The data is in SQLite (`haircut_sharpe`, `mc_pass`, `sensitivity_score` columns). Adding the three columns is a low-effort, high-visibility improvement that makes the robustness gating transparent in the HTML.

**Add to `build_strategy_table()`:**
```python
dsr  = s.get("haircut_sharpe", None)
mc   = s.get("mc_pass", None)
sens = s.get("sensitivity_score", None)

dsr_cell  = (f'<span style="color:#2ecc71">{dsr:.2f}</span>'
             if dsr and dsr > 0
             else f'<span style="color:#e74c3c">{dsr:.2f}</span>'
             if dsr is not None else "—")
mc_cell   = ('<span style="color:#2ecc71">✓</span>'
             if mc else '<span style="color:#e74c3c">✗</span>'
             if mc is not None else "—")
sens_cell = (f'<span style="color:#2ecc71">{sens:.0f}</span>'
             if sens and sens >= 50
             else f'<span style="color:#e74c3c">{sens:.0f}</span>'
             if sens is not None else "—")
```

---

### 4.2 Monte Carlo Fan Chart ❌
**Status: Not implemented**

`mc_pass` boolean is shown nowhere in the report. The fan chart (p5/p25/p50/p75/p95 equity paths from 500 bootstrap resamples) shows whether strategy performance is robust to trade ordering or depends on lucky sequencing. Full code in ANALYSIS.md section "Report & HTML — What Should Be Added."

---

### 4.3 Return Distribution Histogram ❌
**Status: Not implemented**

Per-trade R-value histogram showing skewness and kurtosis. Reveals hidden tail risk that Sharpe hides. `backtest_trades` table has the per-trade data. Full code in ANALYSIS.md.

---

### 4.4 Rolling 90-Day Performance Chart ❌
**Status: Not implemented**

The single most important chart for the live deployment decision. Shows if the last 90 days of backtest have a degrading Sharpe or win rate — if so, don't go live until you retrain on more recent data. Full code in ANALYSIS.md.

---

### 4.5 Day-of-Week Performance Chart ❌
**Status: Not implemented**

`_session_stats()` already computes `by_dow` but it is never rendered in the HTML. 15 lines to add. Full code in ANALYSIS.md.

---

### 4.6 Sortino + Calmar + Max Consecutive Losses in Summary Cards ❌
**Status: Not implemented**

All three are computed in `backtest_engine.py` and stored in the DB. None appear in the HTML summary card row. Specifically:
- `sortino` — downside-only volatility penalty (if Sortino >> Sharpe, upside volatility is inflating Sharpe)
- `calmar` — annualised return / max DD% (regime-normalised)
- `max_consec_loss` — "if you see 5+ consecutive losses live, that's outside 6yr backtest range — investigate"

Add 3 cards to `build_report()` next to the existing Sharpe card:
```python
{card("Sortino Ratio", f"{sortino:.2f}", "#2ecc71" if sortino>1.5 else "#e67e22",
      "Sharpe penalising only downside")}
{card("Calmar Ratio", f"{calmar:.2f}", "#2ecc71" if calmar>1.0 else "#e67e22",
      "Annualised return / max DD%")}
{card("Max Consec Losses", f"{max_consec}",
      "#e74c3c" if max_consec>6 else "#e67e22",
      "Longest losing streak in backtest")}
```

---

### 4.7 Live vs Backtest Comparison Section ❌
**Status: Not implemented (needs 30+ closed live trades to be useful)**

Once live trading accumulates ~30 closed trades, this section compares live win rate / profit factor / expectancy against backtest expectations. Flags statistically significant degradation. Full code in ANALYSIS.md. Add now so it appears automatically once data accumulates.

---

### 4.8 CFD Data Integrity Warning ❌
**Status: Not implemented**

A note below the feature importance chart warning that `cvd`, `delta`, `absorption`, and `stacked_imbalance` features are Dukascopy quote update counts, not real traded volume. Prevents misinterpreting high feature importance as "model learned real institutional order flow." Trivial to add.

---

## SECTION 5 — Dead Code Cleanup

### 5.1 `pipeline.py` Still Imports and Loads LSTM + PPO ⚠️
**Confirmed in:** `pipeline.py` lines 41–42, 338–355

```python
from tensorflow.keras.models import load_model as keras_load_model   # unused
from stable_baselines3 import PPO                                     # unused
# ...
lstm_path = MODEL_DIR / f"lstm_{key}.keras"
if lstm_path.exists():
    models_cache[f"lstm_{key}"] = keras_load_model(...)   # loaded but never used

ppo_path = MODEL_DIR / f"ppo_{symbol}.zip"
if ppo_path.exists():
    models_cache[f"ppo_{symbol}"] = PPO.load(...)          # loaded but never used
```

`get_signal()` no longer uses `lstm_` or `ppo_` keys from the cache. These imports add 10–30 seconds to `live.py` startup (TensorFlow and stable_baselines3 initialization). No harm but wastes startup time and creates confusion.

**Fix:** Remove the LSTM and PPO load blocks from `load_models_from_disk()`, and remove the two imports from the top of `pipeline.py`. The `.keras` and `.zip` files can remain on disk (they don't affect anything) — just stop loading them.

---

### 5.2 `phase2_adaptive_engine.py` Still Has `train_lstm()` Reference ⚠️
**Confirmed in:** `phase2_adaptive_engine.py` lines 1382–1385

`run_historical_training()` (the old monolithic pipeline) still references `train_lstm`. `train.py` no longer calls `run_historical_training()` — it uses its own `run_training()` function. So this is dead code that only runs if someone calls the old entry point directly.

Not a live issue but creates confusion. Should be removed or guarded with a deprecation warning.

---

## SECTION 6 — Future Features (Deferred by Agreement)

| Feature | Reason Deferred | Recommended Timing |
|---|---|---|
| Meta-labeling (two-model architecture) | Requires stable triple-barrier-trained primary model first | After first successful walk-forward retrain is validated |
| Fractional differentiation | Incremental improvement (~5-10%); bigger gains from fixes already done | After priorities 1-4 are stable |
| Sample uniqueness / concurrency weighting | Only meaningful with overlapping triple-barrier labels | Same sprint as meta-labeling |
| Full CPCV (combinatorial paths) | Purging is done; full CPCV adds statistical rigor at high compute cost | After walk-forward is stable and meta-labeling is done |
| HH/HL market structure labeling for HTF | HTF composite already upgraded significantly; this is marginal | Next retrain cycle |
| CME YM volume validation for CFD features | Informational — doesn't change model behavior | When performance drift is suspected |

---

## SECTION 7 — Priority Action List (Remaining Work)

### Immediate (before next retrain)

| # | Action | File | Effort |
|---|---|---|---|
| 1 | Remove `rr` from `ga_fitness()` and `decode_genome()` — replace with `tp_mult` | `phase2_adaptive_engine.py` | 30 min |
| 2 | Fix `ga_fitness()` Sharpe to use trade-frequency annualization | `phase2_adaptive_engine.py` | 15 min |
| 3 | Remove LSTM + PPO load from `pipeline.py` | `pipeline.py` | 10 min |

### Report (can be done independently, any time)

| # | Addition | File | Effort |
|---|---|---|---|
| 4 | Robustness columns (DSR/MC/Sens) in strategy table | `report.py` | 30 min |
| 5 | Sortino + Calmar + Max Consec Loss summary cards | `report.py` | 20 min |
| 6 | Rolling 90-day Sharpe + win rate chart | `report.py` | 45 min |
| 7 | Monte Carlo fan chart | `report.py` | 45 min |
| 8 | Return distribution histogram | `report.py` | 30 min |
| 9 | Day-of-week bar chart | `report.py` | 20 min |
| 10 | Live vs backtest comparison section | `report.py` | 45 min |
| 11 | CFD data note below feature importance | `report.py` | 5 min |

### After next retrain is validated

| # | Action | Depends on |
|---|---|---|
| 12 | Meta-labeling (primary + meta model architecture) | Stable triple-barrier labels |
| 13 | Sample uniqueness weighting | Meta-labeling in place |

---

## Summary

**Correctly implemented (confirmed in code):**
Walk-forward with purging, triple barrier labeling, LSTM/PPO removed from live, DSR/MC/Sensitivity gating with tier fallback, spread+slippage in backtest, daily Sharpe from equity returns, composite HTF alignment, session gate, daily trade cap, consecutive loss cooldown, volatility-adjusted sizing, equity-based daily loss, Sortino+Calmar computed.

**Three small remaining fixes before next retrain:**
1. Remove `rr` from optimization internals — replace with `tp_mult` everywhere
2. Fix `ga_fitness()` Sharpe annualization
3. Remove dead LSTM/PPO load code from `pipeline.py`

**Eight report improvements:**
All data already exists in the SQLite database. These are pure HTML/Plotly additions to `report.py` that make existing computed metrics visible.
