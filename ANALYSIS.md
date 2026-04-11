# ML Trading System — Critical Analysis
**Date:** 2026-04-10  
**Analyst:** AI Mentor Review (strict institutional-grade standard)

---

## Cross-Reference: 14 Institutional Suggestions vs Your Actual Code

Each suggestion rated: ✅ Done | ⚠️ Partial | ❌ Missing | 🔴 Critical Gap

---

### 1. Market Reality First — Session behavior, spread variability, liquidity holes
**Rating: ⚠️ Partial**

What you have:
- Session flags in feature engineering: `is_london`, `is_ny`, `is_overlap`, `hour_sin/cos`
- `spread_mean` captured from real tick data in `tick_pipeline.py`
- ORB (Opening Range Breakout) feature correctly identifies NY open range
- `--signal-check` mode warns about overnight low-probability windows

What is completely missing:
- **No variable spread applied in backtest.** The `spread_mean` column exists in your data but the backtest enters at `closes[i]` with zero spread cost. You have the data — you're just not using it.
- **No news/event filter.** FOMC, NFP, CPI cause instant regime switches. US30 spreads widen to 15–30 points during these. No detection, no block.
- **No rollover detection.** Index CFDs roll quarterly. Artificial price gaps at rollover dates corrupt tick data and can trigger false signals.
- **No "lunch drift" dead zone.** 12:00–13:30 UTC for US30 is consistently low-quality. It's mentioned in the guide but not enforced in code.

**Concrete fix:** Add a hard session gate and a spread filter to `live.py`:
```python
AVOID_HOURS_UTC = list(range(21, 24)) + list(range(0, 8))   # overnight
SPREAD_MAX_ATR_RATIO = 0.15   # skip if spread > 15% of ATR

def is_tradeable_conditions(symbol, df):
    hour = datetime.utcnow().hour
    if hour in AVOID_HOURS_UTC:
        return False, "outside trading hours"
    spread = df["spread_mean"].iloc[-1] if "spread_mean" in df.columns else 0
    atr    = df["atr14"].iloc[-1]
    if spread > atr * SPREAD_MAX_ATR_RATIO:
        return False, f"spread too wide ({spread:.1f} > {atr*SPREAD_MAX_ATR_RATIO:.1f})"
    return True, "ok"
```

---

### 2. Regime Detection as Core Layer
**Rating: ⚠️ Partial — detection exists, conditional routing does not**

What you have (genuinely good):
- 6-regime classifier in `add_regime_features()`: low-vol range, high-vol range, bull trend, bear trend, expanding, contracting
- ADX proxy, EMA slope, ATR ratio, HV20 percentile — all solid signals
- Regime fed as features (`regime_sin`, `regime_cos`, `adx_norm`) to ML models

What is critically missing:
- **One model runs in all regimes.** The regime label goes into XGB/RF as a feature, but there is no conditional routing. A single XGB model trained on all 6 regimes simultaneously is trying to be good at trend-following AND mean-reversion AND breakout at the same time. It will average out to mediocre in all of them.
- **No HMM or Bayesian switching.** Your regime is rule-based (ADX > 25 = trending). This produces noisy single-bar flips. ADX crossing 25 for one bar then back to 24 generates two regime changes in two bars.
- **No market structure labeling.** HH/HL (bullish structure) vs LH/LL (bearish structure) is not computed. This is the most reliable non-lagging trend label.

**Concrete fix — regime-conditional model routing:**
```python
# In signal generation:
current_regime = df["regime"].iloc[-1]

if current_regime in [2, 3]:       # trending
    model_key = f"xgb_{key}_trend"
elif current_regime in [0, 1]:     # ranging
    model_key = f"xgb_{key}_range"
else:                               # expanding/contracting
    model_key = f"xgb_{key}_breakout"

# Train separate models per regime group during training
```

**Add market structure labeling** (HH/HL sequence, no lookahead):
```python
def label_market_structure(df, lookback=20):
    # Bullish: last swing high > prior swing high AND last swing low > prior swing low
    # Bearish: last swing low < prior swing low AND last swing high < prior swing high
    # This is non-lagging (uses confirmed swings only)
```

---

### 3. Feature Engineering Beyond Indicators
**Rating: ✅ Done — this is your strongest area**

Your `institutional_features.py` covers essentially everything suggested:
- ✅ Candle imbalance → `absorption`, `failed_auction`, `wick_up_atr`, `wick_down_atr`, `pin_bar_bull/bear`
- ✅ Range expansion/contraction → `atr_ratio`, `atr_slope`, regime classifier
- ✅ Session-relative positioning → ORB features, session flags, hour encoding
- ✅ Previous high/low sweeps → `stop_hunt_up`, `stop_hunt_down`, `equal_high`, `equal_low`
- ✅ VWAP deviations → `vwap_dist_atr`, `vwap_dist_pct`, `vwap_band_pos`, `vwap_cross`
- ✅ Volatility compression before expansion → `regime_contracting`, `atr_slope`

Two things worth adding:
- **Spread as a real-time feature:** `spread_pct_of_atr` — wide spread = institutions not committed, signal quality drops
- **Time distance from session open:** `bars_since_open` normalized — setups near the open behave differently from midday setups
- **Market structure label** (HH/HL vs LH/LL) — the one structural label missing from your already strong feature set

---

### 4. Labeling — Triple Barrier / TP-SL Outcome
**Rating: 🔴 Critical Gap — this is your most important unfixed problem**

Current code:
```python
d["target"] = (c.shift(-1) > c).astype(int)   # next bar up or down
```

What the suggestion says (and what I independently found): this is the single most common reason ML trading systems fail backtesting but collapse in live trading. A model trained to predict next-bar direction has no idea whether a trade at a 1.5 ATR SL with a 2.0 ATR TP will win or lose. Those are completely different problems.

**The triple barrier method (Marcos Lopez de Prado, "Advances in Financial ML"):**
```python
def make_triple_barrier_labels(df, sl_atr=1.5, tp_mult=2.0, max_bars=50):
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    atrs   = df["atr14"].values
    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df) - max_bars):
        entry   = closes[i]
        sl_dist = atrs[i] * sl_atr
        tp_dist = sl_dist * tp_mult
        tp = entry + tp_dist   # long only for simplicity
        sl = entry - sl_dist

        for j in range(i + 1, min(i + max_bars, len(df))):
            if lows[j] <= sl:
                labels[i] = 0; break    # SL hit first
            if highs[j] >= tp:
                labels[i] = 1; break    # TP hit first
        # if neither hit: label = 0 (timeout = loss in expectation)

    df["target"] = labels
    return df
```

Use different `sl_atr` and `tp_mult` values per TF during labeling — these should match what the optimizer will later use.

---

### 5. Walk-Forward + Reality-Based Validation
**Rating: 🔴 Critical Gap**

Current code: fixed 70/15/15 split, all 12,750+ optimization evaluations on the same last 15%.

The suggestion's approach is exactly right. Suggested implementation for your system:

```
Fold 1: Train 2020-01 → 2022-12 | Validate 2023-01 → 2023-06 | Test 2023-07 → 2023-12
Fold 2: Train 2020-01 → 2023-06 | Validate 2023-07 → 2023-12 | Test 2024-01 → 2024-06
Fold 3: Train 2020-01 → 2023-12 | Validate 2024-01 → 2024-06 | Test 2024-07 → 2024-12
Fold 4: Train 2020-01 → 2024-06 | Validate 2024-07 → 2024-12 | Test 2025-01 → 2025-06
Fold 5: Train 2020-01 → 2024-12 | Validate 2025-01 → 2025-06 | Test 2025-07 → 2026-03
```

Report: average out-of-sample Sharpe across all 5 test folds, plus the distribution (min/max fold). If fold 3 has Sharpe 2.1 but fold 4 has -0.3, the strategy is regime-dependent and you need to know which regime you're currently in before deploying it.

---

### 6. Execution Modeling — Variable Spread, Slippage
**Rating: 🔴 Critical Gap for US30 specifically**

What you have: `spread_mean` in tick data (the raw data is there).
What the backtest uses: `closes[i]` with zero spread, zero slippage.

US30-specific facts this suggestion is correct about:
- Normal spread: 1–3 points
- London open (first 15 min): 3–8 points
- US news events: 10–30+ points
- Overnight/Asian session: 5–15 points

At 1% risk per trade with a 30-point SL, a 5-point spread = 16% of your theoretical risk is already lost at entry. Over 500 trades this is enormous.

**Concrete fix for `backtest_engine.py`:**
```python
# Add spread cost at entry
spread = df["spread_mean"].iloc[i] if "spread_mean" in df.columns else atr_arr[i] * 0.05
if direction == 1:
    entry = closes[i] + spread   # buy at ask
    sl    = entry - sl_dist
    tp    = entry + sl_dist * tp_mult
else:
    entry = closes[i] - spread   # sell at bid
    sl    = entry + sl_dist
    tp    = entry - sl_dist * tp_mult

# Add slippage model (volatility-dependent)
slippage = atr_arr[i] * 0.02 * np.random.uniform(0.5, 1.5)   # 0–3% of ATR
entry += direction * slippage
```

---

### 7. Risk Model > Entry Model
**Rating: ✅ Mostly done, one significant gap**

What you have (good):
- ✅ Daily loss limit (`MAX_DAILY_LOSS_PCT`)
- ✅ Max drawdown circuit breaker (`MAX_DRAWDOWN_PCT`)
- ✅ Max open positions
- ✅ BE-aware risk capital cap (your best risk feature)
- ✅ ATR-based SL (volatility-adjusted stop distance)
- ✅ Risk per trade as % of balance (compounds correctly)

What is missing:
- **Volatility-adjusted position sizing.** ATR-based SL adjusts the distance, but not the lot size for current volatility regime. In regime 4 (expanding volatility), ATR is 2–3x larger than regime 0. The SL distance grows but you still risk 1% of balance. Effective dollar risk stays the same, but the probability distribution of outcomes is different.
- **Max trades per session.** If 4 signals fire in the first 30 minutes of NY open (common during trend days), you can take all 4. A cooldown or session cap prevents overexposure to a single market narrative.
- **Consecutive loss cooldown.** After 3 consecutive losses, reduce size to 0.5% for the next 5 trades. Standard institutional practice.

---

### 8. Parameter Stability — Stable Regions, Not Peak Values
**Rating: ✅ Done — tools exist, but not used in strategy selection**

What you have:
- ✅ `run_sensitivity()` — nudges confidence/sl_atr/rr by ±step, scores stability 0–100
- ✅ `run_monte_carlo()` — bootstrap resampling, p5/p50/p95 Sharpe distribution
- ✅ `haircut_sharpe()` — multi-testing penalty (Bailey & Lopez de Prado)

The problem: **none of these are used in the actual strategy selection.** The active strategy is selected by `efficiency_ratio` (raw). The `sensitivity_score` and `mc_pass` metrics are computed in `backtest_engine.py` but are only displayed in the HTML report — they don't gate which strategy gets activated.

**Fix:** Add a robustness gate to strategy activation in `train.py`:
```python
# Only activate a strategy if it passes robustness checks
if result["sensitivity_score"] >= 50 and result["mc_pass"]:
    # eligible for activation
    ...
```
Currently you could activate a strategy with sensitivity_score=20 (cliff-edge parameters) simply because its ER is highest.

---

### 9. Ensemble / Multi-Model Approach
**Rating: ⚠️ Partial — wrong type of ensemble**

What you have: LSTM + XGB + RF + PPO — 4 model architectures.

The problem: all 4 models are trained on the **same target** (next-bar direction), from the **same features**, on the **same data**. They will be highly correlated in their predictions. This is model architecture diversity, not strategy diversity.

What the suggestion means by ensemble:
- **Model A:** Trend-following (trained on trending regime data, uses CVD slope, EMA slope)
- **Model B:** Mean-reversion (trained on ranging regime data, uses VWAP distance, absorption)
- **Model C:** Breakout (trained on contracting regime data, uses ATR compression, volume surge)
- **Meta-model:** Decides which sub-model to weight based on current regime

A true diverse ensemble produces uncorrelated signals. When Model A fails (wrong regime), Model B or C compensates. Your current ensemble will all fail together in the same market conditions because they were trained together.

---

### 10. Trade Filtering Layer — Alpha Preservation
**Rating: ❌ Missing as a hard layer**

What you have (soft filters only):
- Confidence threshold (soft — just a probability cutoff)
- HTF alignment check (soft — reduces prob but doesn't hard-block)
- Risk gate (hard — but this is risk management, not alpha filtering)

What is completely missing as **hard trade filters**:

| Filter | Status | Impact |
|---|---|---|
| Time-of-day gate (no trades 21:00–07:30 UTC) | ❌ Missing | High |
| Spread filter (skip if spread > X% of ATR) | ❌ Missing | High for US30 |
| News/event blackout (±30 min around FOMC, NFP) | ❌ Missing | High |
| Volatility filter (skip if ATR < 20th percentile — dead market) | ❌ Missing | Medium |
| Consecutive loss cooldown | ❌ Missing | Medium |
| "No trade zone" after max daily trades hit | ❌ Missing | Medium |

The suggestion's observation is exactly right: **removing 30–50% of trades (the low-quality ones) typically increases profitability.** Your confidence threshold is doing some of this work, but it's soft and regime-unaware. A 0.64 threshold in a dead overnight market is not the same quality gate as a 0.64 threshold during NY/London overlap.

---

### 11. Psychological Bias Removal
**Rating: ⚠️ Partial**

What you have:
- ✅ Max daily loss hard stop
- ✅ Max drawdown circuit breaker
- ✅ No manual override possible (all config in .env)

What is missing:
- ❌ **Max trades per session** — no limit on how many signals can fire per day
- ❌ **Cooldown after consecutive losses** — the system will keep firing at full risk after 5 straight losses
- ❌ **Abnormal condition detection** — if the model starts generating signals every minute (possible if confidence threshold is set too low and market is choppy), no circuit breaker stops it
- ❌ **Overtrading detection** — if `n_trades_today > 2 * expected_daily_frequency`, something is wrong; pause and alert

---

### 12. Data Quality & Granularity
**Rating: ✅ Done — strongest area of the project**

- ✅ Real Dukascopy tick data (not broker OHLCV)
- ✅ Accurate session timestamps with DST handling (`America/New_York`)
- ✅ Bid/ask volume separation (real order flow, not estimated)
- ✅ Data quality report (coverage %, rejection rate, microstructure presence)
- ✅ Chunk streaming (handles 16GB CSV without RAM issues)
- ✅ Incremental append (monthly updates without full rebuild)
- ✅ Live bar microstructure estimation with clear documentation of limitations

No significant gaps here. This is genuinely better than 95% of retail automated systems.

---

### 13. Monitoring & Adaptation Post-Deployment
**Rating: ⚠️ Partial — operational monitoring exists, performance drift detection does not**

What you have:
- ✅ Daily log files with structured output
- ✅ `--signal-check` mode (raw ML probabilities vs threshold)
- ✅ Hot-reload of new models without trade interruption
- ✅ SQLite database of all live trades with P&L
- ✅ HTML report on demand

What is missing:
- ❌ **Drift detection.** If the model's recent live win rate drops from 50% to 35% over 20 trades, the system keeps trading. No statistical test flags this.
- ❌ **Auto-disable on degradation.** No condition that automatically pauses live trading when performance deviates significantly from backtest expectations.
- ❌ **Feature drift monitoring.** If the distribution of key features (e.g., `cvd_z`, `vwap_dist_atr`) shifts significantly from training data, model predictions become unreliable. No detection exists.
- ❌ **Retraining triggers based on live performance** (as opposed to just time-based).

**Concrete drift detection to add:**
```python
def check_model_drift(recent_trades, expected_win_rate, min_trades=20):
    if len(recent_trades) < min_trades:
        return False
    actual_wr = sum(1 for t in recent_trades if t["pnl"] > 0) / len(recent_trades)
    # Binomial test: is actual win rate significantly below expected?
    from scipy.stats import binom_test
    p_value = binom_test(
        sum(1 for t in recent_trades if t["pnl"] > 0),
        n=len(recent_trades),
        p=expected_win_rate / 100,
        alternative="less"
    )
    return p_value < 0.05   # statistically significant degradation
```

---

### 14. The Real Edge — Conditional, Not Constant
**Rating: ⚠️ Partial — implicit but not architected**

What you have:
- Confidence threshold (soft conditional — only trade when ML is confident)
- HTF alignment (conditional — reduces confidence against HTF trend)
- Regime features fed to models (ML can learn regime-conditional behavior implicitly)

What is missing:
- **Explicit "strike zone" architecture.** Right now the system will attempt a signal evaluation on every closed candle in every session. The filtering is all probability-based (soft). A proper strike zone system has hard preconditions that must ALL be met before ML is even consulted:

```
Strike Zone = Session is active
           AND spread is normal
           AND regime is known (not transitioning)
           AND no news in past/next 30 min
           AND consecutive losses < threshold
           AND ML confidence > adaptive threshold for this regime/session
```

Only when all preconditions pass does the ML ensemble vote. This is the architecture described — not a single soft threshold, but a layered decision tree of hard and soft gates.

---

## Overall Cross-Reference Scorecard

| # | Suggestion | Status | Priority |
|---|---|---|---|
| 1 | Market reality (sessions, spreads, liquidity holes) | ⚠️ Partial | High |
| 2 | Regime detection as core routing layer | ⚠️ Partial | High |
| 3 | Feature engineering beyond indicators | ✅ Done | — |
| 4 | Triple barrier / TP-SL outcome labeling | 🔴 Critical Gap | Critical |
| 5 | Walk-forward validation | 🔴 Critical Gap | Critical |
| 6 | Execution modeling (spread, slippage) | 🔴 Critical Gap | High |
| 7 | Risk model > entry model | ✅ Mostly done | Low |
| 8 | Parameter stability (stable regions, not peaks) | ⚠️ Tools exist, not enforced | Medium |
| 9 | Ensemble of diverse strategy types | ⚠️ Architecture ensemble only | Medium |
| 10 | Trade filtering hard layer | ❌ Missing | High |
| 11 | Psychological bias removal | ⚠️ Partial | Medium |
| 12 | Data quality & granularity | ✅ Done | — |
| 13 | Monitoring & adaptation | ⚠️ Partial | Medium |
| 14 | Conditional edge architecture | ⚠️ Partial | Medium |

**3 Critical Gaps (fix before going live):** #4 (labeling), #5 (walk-forward), #6 (execution modeling)  
**4 High Priority gaps:** #1 (session/spread gates), #2 (regime routing), #10 (filter layer), fix already listed  
**Everything else:** medium priority improvements that increase robustness over time

---

---

## Architecture Overview

Tick data (Dukascopy) → Parquet → ~161 institutional features → LSTM + XGBoost + RandomForest + PPO ensemble → GA + Optuna parameter optimization → MT5 live execution.

**Files:**
| File | Role |
|---|---|
| `tick_pipeline.py` | Streams Dukascopy CSV → Parquet (OHLCV + microstructure) |
| `institutional_features.py` | VWAP, Volume Profile, Order Flow, Liquidity, Regime |
| `phase2_adaptive_engine.py` | ML models, RL agent, GA, Optuna, signal logic |
| `pipeline.py` | Shared data pipeline (train + live) |
| `train.py` | Training only (LSTM, XGB, RF, PPO, GA, Optuna) |
| `live.py` | Live trading loop (model inference + MT5 execution) |
| `backtest_engine.py` | Full backtesting with equity curves and stats |
| `db.py` | SQLite (strategies, equity curves, live trades) |
| `report.py` | HTML performance report |

---

## THE GOOD — Genuinely Strong Foundations

### 1. Tick-derived institutional features are real edge
`institutional_features.py` is the strongest part of the project. Session VWAP with proper daily reset, volume profile (POC/VAH/VAL/HVN/LVN) from real tick volumes, per-bar order flow delta (bid vs ask volume), cumulative volume delta (CVD), stacked imbalance detection (3:1 ratio over 3+ consecutive bars), absorption detection (high volume + small range = limit orders absorbing retail), failed auction signals, and stop hunt detection. These are concepts that genuinely separate institutional-grade systems from retail indicator stacking. Most retail algos cannot replicate these because they work from OHLCV only.

### 2. No future leakage in regime volatility ranking
```python
hv20_pct = hv20.rolling(252).rank(pct=True)
```
Explicitly uses a rolling rank window instead of ranking the full series. Many systems make this mistake and encode future volatility context into training features. This fix was clearly intentional and correct.

### 3. Memory-efficient LSTM streaming
`make_lstm_dataset()` using `tf.data` avoids materializing the full 3D sequence cube (which would be ~27 GB for 1m on 2.8M bars). The flat ~700 MB array + batch-level generation is sophisticated engineering that makes training on the full tick dataset feasible.

### 4. Deflated Sharpe Ratio with selection bias correction
```python
def haircut_sharpe(trades, n_trials=150):
    # Bailey & Lopez de Prado (2014) multi-testing penalty
    penalty = max(1.0, np.sqrt(np.log(max(n_trials, 2))))
    return float(sr_adj / penalty)
```
At 150 trials this applies a ~2.24x downward correction. Most retail algos publish raw Sharpe and call it a day. This is proper academic methodology.

### 5. Monte Carlo + Parameter sensitivity testing
`run_monte_carlo()` (bootstrap resampling) and `run_sensitivity()` (±nudge on confidence/sl_atr/rr) are proper robustness checks. The sensitivity scorer penalizes instability in **either** direction — a strategy that improves dramatically when nudged is also an overfitting signal.

### 6. Break-even aware risk capital calculation
Excluding positions where SL = break-even from the total risk cap means a strategy that moves trades to BE quickly can keep opening new positions. Most systems naively block on position count alone and miss valid entries while protected positions run.

### 7. Train/Live separation with safe hot-reload
Splitting `train.py` and `live.py` is architecturally clean. Hot-reload of new models only fires when no positions are open for that symbol. Positions are never interrupted by model updates.

### 8. DST-correct session open range
```python
_us_idx = (...).tz_convert("America/New_York")
is_open_bar = (_us_idx.hour == 9) & (_us_idx.minute >= 30)
```
Using `America/New_York` instead of hardcoded UTC hours handles DST automatically. Without this, the opening range would be 1 hour off for ~4 months per year.

### 9. Proper value area expansion (Auction Market Theory)
The POC → VAH/VAL expansion algorithm correctly expands toward whichever neighbour has more volume, exactly as TPO/AMT defines the 70% value area. Not a naive symmetric band.

### 10. Test trade pipeline verification
`python live.py --test-trade` places a real micro-lot order and verifies all 6 pipeline steps (placement → DB log → position verification → BE monitoring → risk cap → close). This is proper end-to-end smoke testing.

---

## THE BAD — Serious Methodological Problems

### 1. Backtest signal ≠ Live signal (CRITICAL FLAW)

The backtest engine uses only XGB + RF:
```python
# backtest_engine.py
prob = (p_xgb + p_rf) / 2.0
```

But live execution uses all 4 models:
```python
# phase2_adaptive_engine.py
prob = (p_lstm * 0.30 + p_xgb * 0.25 + p_rf * 0.25 + p_rl * 0.20)
```

**Impact:** The entire optimization pipeline (GA + Optuna, ~12,150 evaluations) is tuning parameters for a proxy signal that is never actually traded. The confidence threshold selected at e.g. 0.64 corresponds to `(XGB+RF)/2`, but in live mode the same 0.64 is applied to a completely different combined probability. You are selecting strategies based on a signal you never execute.

**Fix:** Either include LSTM + PPO in the backtest signal (hard — requires sequence building per bar), or remove them from the live ensemble and trade only XGB + RF (simpler and consistent).

---

### 2. The test set is fully data-mined

`ga_fitness()` evaluates every genome on the same fixed last 15% slice:
```python
df = df_full.iloc[int(n_full * 0.85):]
```

This same slice is used for:
- 12,000 GA genome evaluations (pop=150 × gen=80)
- 150 Optuna trials per TF × 5 TFs = 750 Optuna evaluations
- Top-5 selection per TF (25 strategies scored on same slice)

After 12,750+ evaluations on the identical data, the "out-of-sample" test is completely in-sample by selection. The `haircut_sharpe()` provides a theoretical correction but is not used in actual strategy selection — `efficiency_ratio` (raw, not deflated) is what selects the active strategy.

**Fix:** Walk-forward validation with at least 3–5 expanding windows. Each fold's parameters are evaluated on a truly unseen forward test period. Report the average out-of-sample Sharpe across folds.

---

### 3. ML target misaligns with trading objective

```python
d["target"] = (c.shift(-1) > c).astype(int)   # next bar up/down
```

Models are trained to predict whether the **next single bar closes up or down**. But trades hold for up to 50 bars with an ATR-based SL and TP. A model that is 52% accurate at predicting next-bar direction is not necessarily profitable at a 1:2 RR trade held for 10 bars — these are different problems.

**Fix:** Replace target with a forward-looking trade outcome label:
```python
def make_trade_target(df, sl_atr=1.5, rr=2.0, max_bars=50):
    # For each bar: scan forward max_bars bars
    # Return 1 if TP reached before SL, 0 if SL hit first
    # This directly matches what the trading system does
```
This target leaks future data during feature engineering — apply it only to the label column, never as an input feature.

---

### 4. HTF alignment is too simplistic

```python
# HTF trend: price above EMA55 = bullish
htf_dir = (htf_feat["Close"] > htf_feat["ema55"]).astype(int) * 2 - 1
```

A single EMA55 crossover as "institutional HTF alignment" is a 1990s indicator. For US30 specifically, institutions use:
- **Market structure:** consecutive Higher High / Higher Low sequence
- **Break of Structure (BOS):** price closes above/below a significant swing high/low
- **Daily VWAP position:** are institutions net long/short for the day?
- **Session high/low sweeps:** did price take out yesterday's high before reversing?

The institutional features you built for the entry TF exist. They should also be computed at HTF and merged down.

---

### 5. Volume Profile and Anchored VWAP are O(n²) — will take hours on 1m data

```python
# institutional_features.py
for i in range(session_bars, n):           # ~2.8M iterations on 1m
    profile = compute_volume_profile(...)   # 390-element slice each time
```

On 2.8M bars at 1m with `session_bars=390`, this is approximately 1.1 billion numeric operations per call. The anchored VWAP loop is similarly structured. This will take many hours on the full tick dataset and is likely the reason training is expected to take 60–120 minutes.

**Fix:** Pre-compute session-level volume profiles (one per trading day, not one per bar), then forward-fill the POC/VAH/VAL values to each bar within that session. This reduces from O(n × session_bars) to O(n_days × session_bars) — about 1,500× faster.

---

### 6. Sharpe annualization uses wrong factor

```python
sharpe = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
```

`np.sqrt(252)` is the correct annualization factor when `r` is a series of **daily returns**. Here `r` is an array of per-trade R-multiples with variable time gaps. A strategy taking 50 trades/year and one taking 500 trades/year will produce incomparable Sharpe ratios with this formula, making cross-TF strategy comparison (the main selection criterion) invalid.

**Fix:** Build daily returns from the equity curve, then compute Sharpe:
```python
daily_returns = equity_df["equity"].pct_change().dropna()
sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
```
The `equity_df` is already constructed — use it.

---

### 7. No slippage model in backtest

The backtest enters at the closing price with zero spread cost:
```python
entry = closes[i]
sl    = entry - direction * sl_dist
tp    = entry + direction * sl_dist * tp_mult
```

US30 spreads widen to 5–15+ points during news events (FOMC, NFP, CPI). On a 1m entry with a 30-point SL, a 10-point spread is a 33% immediate drawdown on the trade before it even starts. The backtest P&L is overstated by the average spread cost times the number of trades.

**Fix:** Add a spread model:
```python
spread_pts = df["spread_mean"].iloc[i] if "spread_mean" in df.columns else atr_arr[i] * 0.05
entry_long  = closes[i] + spread_pts / 2   # buy at ask
entry_short = closes[i] - spread_pts / 2   # sell at bid
```

---

## THE UGLY — Fundamental Design Issues

### 1. PPO is conceptually misused in the ensemble

```python
action, _ = rl_m.predict(obs, deterministic=True)
p_rl = 0.8 if action == 1 else (0.2 if action == 2 else 0.5)
```

A PPO agent outputs discrete **actions** (hold/long/short), not probabilities. Converting `action == 1 → 0.8` and averaging with classifier probabilities is mathematically meaningless. RL and supervised classifiers solve fundamentally different problems:
- **Classifiers** (XGB, RF, LSTM): optimize cross-entropy loss on next-bar direction
- **RL agent (PPO)**: optimizes long-term cumulative reward via policy gradient

Averaging their outputs assumes they're measuring the same quantity on the same scale. They are not. The RL "vote" of 0.8 is not a probability — it is an arbitrary constant assigned to a discrete action.

**Fix options:**
- Use PPO as a **hard gate**: only trade when RL action agrees with the classifier direction (action==1 and classifiers say long, or action==2 and classifiers say short)
- Use PPO **standalone** with its own signal threshold, independent of classifiers
- Replace PPO with a proper probabilistic RL approach (SAC with continuous action space outputting a trade size between -1 and +1)

---

### 2. The RL observation dimension bug

In `TradingEnv` (training):
```python
self.observation_space = spaces.Box(
    shape=(self.RL_OBS_BARS * self.n_features,),  # RL_OBS_BARS = 1
    # → shape = (n_features,) ≈ (146,)
)
```

In `get_signal()` (live inference):
```python
obs = scaled[feature_cols].tail(SEQ_LEN).values.flatten()
# SEQ_LEN = 60 → shape = (60 * n_features,) ≈ (8760,)
```

The model is trained on `(146,)` observations but receives `(8760,)` at inference. This will either:
- Crash with a dimension mismatch error (best case — visible failure)
- Be silently truncated by stable_baselines3 (worst case — garbage RL votes that look plausible)

**Fix:** Match the inference observation to training:
```python
obs = scaled[feature_cols].iloc[-1].values.astype(np.float32)  # single bar = (146,)
```

---

### 3. The `rr` vs `tp_mult` parameter redundancy

Both parameters represent "take profit distance as a multiple of stop loss distance":
- `rr`: used in backtest as `pnl = risk * rr`
- `tp_mult`: used in signal as `tp = close + direction * sl_dist * tp_mult`

They are semantically identical but optimized independently by Optuna. The optimizer wastes search budget exploring a fully redundant dimension. A genome where `rr=2.0, tp_mult=3.0` is internally inconsistent — the backtest says 2R profit but the live trade targets 3R.

**Fix:** Remove `rr`. Set `tp = close + direction * sl_dist * tp_mult` everywhere (including backtest), and derive the R-multiple from the actual exit price vs entry vs SL.

---

### 4. The "incremental live learning" is parameters-only, not true online learning

```python
def incremental_update(symbol, df_dict, models_cache, scalers_cache):
    new_params = run_optuna(...)   # re-tunes only parameters
    save_params(symbol, new_params)
    # XGB, RF, LSTM models are NEVER retrained here
```

After 20 live trades, the system "re-learns" — but only re-runs Optuna on the same stale models. The ML models themselves never see new live trade data. The system is not learning from experience; it's re-tuning parameters on historical simulations using models that were trained once and never updated.

**Fix:** Collect closed live trades into a small "experience buffer." After N trades, fine-tune XGB/RF on recent data (last 6 months + live trades). For LSTM, use a frozen base + fine-tune only the final layers.

---

### 5. Daily risk reset uses balance (not equity)

```python
def reset_daily(self):
    self.session_start_balance = get_account_balance()  # balance, not equity
```

MT5 `account_info().balance` reflects only closed trades. Open positions' floating P&L is not included. If you have a losing open trade from yesterday at -$500, today's risk reset starts from a balance that's $500 higher than the true account value. The daily loss limit will allow more loss than intended.

**Fix:** Use `account_info().equity` (balance + floating P&L) as the session start reference.

---

## MISSING FEATURES — Would Materially Improve Robustness

### 1. Walk-forward optimization (highest priority)
Divide 6 years into rolling folds. Example:
- Fold 1: Train 2020–2022, Test 2023
- Fold 2: Train 2020–2023, Test 2024
- Fold 3: Train 2020–2024, Test 2025
- Fold 4: Train 2020–2025, Test 2026

Report the **average out-of-sample Sharpe** across folds, not the best backtest result. The distribution of fold results shows regime sensitivity.

### 2. Regime-conditional models
You already detect 6 regimes. Train **separate XGB/RF models per regime**:
- Regime 0/1 (ranging): features like VWAP distance, mean-reversion signals get higher weight
- Regime 2/3 (trending): features like EMA slope, CVD trend get higher weight
- Regime 4/5 (expanding/contracting): volatility breakout features matter most

A single model trying to be good in all regimes typically becomes mediocre in all of them.

### 3. Trade-labeled ML targets
Replace next-bar binary classification with forward trade outcome:
```python
def label_forward_outcome(df, sl_atr=1.5, tp_mult=2.0, max_bars=50):
    """
    For each bar: given an entry at close, SL at close - sl_atr*ATR,
    TP at close + tp_mult*sl_atr*ATR — scan forward max_bars.
    Label 1 if TP reached first, 0 if SL reached first.
    """
```
This directly aligns the ML objective with the trading objective.

### 4. Session time filter (hard gate)
From backtest analysis: overnight signals (21:00–07:30 UTC) and early Asian session are predominantly noise for US30. Enforce a hard gate — no new entries outside active sessions:
```python
TRADE_HOURS_UTC = [(8, 0, 21, 0)]  # London open to NY close
def is_trading_hour(dt):
    h = dt.hour
    return 8 <= h < 21
```
This alone typically improves Sharpe by 0.3–0.5 by eliminating low-quality setups.

### 5. Volatility-adjusted position sizing
Fixed 1% per trade ignores regime. In expanding volatility (regime 4), 1% ATR-based SL is 2–3x wider than in low-vol ranging. Use volatility targeting:
```python
target_vol = 0.01   # 1% daily portfolio volatility target
current_vol = df["atr14"].iloc[-1] / df["Close"].iloc[-1]
size_scalar = target_vol / (current_vol + 1e-10)
risk_pct    = base_risk_pct * np.clip(size_scalar, 0.3, 2.0)
```

### 6. Correlation filter for multi-symbol trading
US30, USTEC, US500, and UK100 have pairwise correlations of 0.75–0.92 during US sessions. Opening all four on the same signal quadruples exposure without diversification. Skip new entries if an open position exists in a highly correlated instrument:
```python
CORRELATION_GROUPS = [
    {"US30", "USTEC", "US500"},    # US equity indices
    {"EURUSD", "GBPUSD"},          # USD pairs (moderate correlation)
]
def is_correlated_position_open(symbol, open_positions, threshold=0.75):
    ...
```

### 7. Proper commission / swap model
For index CFDs, overnight swap can be 2–5 points per lot per night. For a trade held 10 bars on 1m this is irrelevant. For a 15m trade held overnight it is significant. Add:
```python
SWAP_PER_LOT_PER_NIGHT = {
    "US30": -3.0,   # broker-specific, check swap specification
    "DE40": -2.5,
}
```

---

## Advanced Findings — 6 Additional Points (Lopez de Prado / Institutional ML)

Status per point: ✅ Already covered | ⚠️ Partially addressed | 🔴 Not in code | 🔵 Future feature (too complex for current phase)

---

### A. CFD Data Reality — Tick Volume Is Not Real Volume
**Status: 🔴 Not addressed — affects all order flow features**

This is the most important data-level finding and it directly undermines your strongest feature group.

**The problem in your code:**
`institutional_features.py` builds CVD, stacked imbalance, absorption, and delta features from `bid_volume` and `ask_volume` from Dukascopy tick data. The assumption is that these volumes represent actual traded contracts. They do not.

Dukascopy is a CFD/FX liquidity aggregator. Their "tick volume" = number of times the broker's pricing engine updated the quote, not the number of contracts that changed hands at that price. Two consequences:

1. **CVD is measuring quoting activity, not actual buying/selling pressure.** A period with 10,000 bid-side "volume" means the bid was requoted 10,000 times — it says nothing about whether real money was committed long or short.
2. **Absorption detection (`high volume + small range`) is detecting quote clustering, not institutional limit order activity.** This is a completely different (and much less meaningful) phenomenon.

The features still have some signal — quoting activity does correlate with real activity — but it is a noisy, broker-specific proxy. Your model is learning patterns that are partially artifacts of Dukascopy's quoting algorithm, not market structure.

**What to do:**

Short term (rename and reframe, no code change):
```python
# Rename to reflect reality — these are quote-based proxies, not real flow
"delta"     → "quote_delta"       # ask quotes - bid quotes per bar
"cvd"       → "quote_cvd"         # cumulative quote direction
"absorption"→ "quote_clustering"  # high quote density + small range
```
Renaming forces honest thinking about what the model is learning.

Medium term (blend real futures volume as validation):
- Fetch CME YM (Dow futures) real volume via a free API (Polygon.io free tier, Yahoo Finance futures, or Quandl)
- Compute the **correlation between your CFD quote delta and real YM volume delta** on the same bars
- If correlation > 0.6, your CFD proxies are valid signal. If < 0.3, they are noise specific to Dukascopy.
- Add real YM volume as a feature even though you execute on CFD:

```python
# In pipeline.py — fetch YM 1m bars from Yahoo Finance as ground truth volume
import yfinance as yf
ym = yf.download("YM=F", interval="1m", period="60d")
df["real_futures_volume"] = ym["Volume"].reindex(df.index, method="nearest")
df["real_futures_vol_ratio"] = df["real_futures_volume"] / df["real_futures_volume"].rolling(20).mean()
```

Long term (future feature): Replace Dukascopy volume entirely with CME Level 2 data via a paid feed (IQFeed, Rithmic, or Databento). This is the only way to get real order flow for US30/Dow. Cost: ~$50–150/month.

**Bottom line:** Your order flow features are your most creative work. Do not abandon them — but know they are CFD quote proxies, not real institutional flow. Label them accordingly and validate against futures volume when possible.

---

### B. Meta-Labeling — Two-Model Architecture
**Status: 🔴 Not in code — high value, implementable now**

This is Lopez de Prado's most actionable improvement over a standard single-model approach, and it fits naturally into your existing architecture.

**Current architecture (one model does everything):**
```
Features → XGB → probability → threshold → direction + size
```

**Meta-labeling architecture (two models, separated concerns):**
```
Stage 1 — Primary model (side only):
  Features → XGB_primary → BUY or SELL (binary, no probability needed)

Stage 2 — Meta model (filter + size):
  Features + primary_prediction + primary_confidence → RF_meta → P(trade wins)
  If P < 0.4 → skip trade entirely (size = 0)
  If P ≥ 0.4 → size = base_lot × P   (conviction-weighted sizing)
```

**Why this is better than a single model:**

The primary model is trained on direction prediction. The meta model is trained on a completely different question: "Given that the primary model said BUY and these market conditions exist, what is the probability this specific trade hits TP before SL?" These are different enough that a separate model trained on them outperforms a single model trying to do both simultaneously.

**Concrete implementation for your system:**

```python
# Step 1: train primary model (direction only, triple-barrier labels)
# This is your existing XGB — keep it as-is after the labeling fix

# Step 2: generate primary predictions on training data
train_pred_proba = xgb_primary.predict_proba(X_train)[:, 1]
train_pred_side  = (train_pred_proba >= 0.5).astype(int)

# Step 3: build meta-model training set
# Only include rows where primary model fired a signal
meta_mask  = (train_pred_proba >= 0.5) | (train_pred_proba <= 0.5)  # all bars
X_meta     = np.column_stack([X_train, train_pred_proba, train_pred_side])
y_meta     = y_train_triple_barrier   # 1 = TP hit, 0 = SL hit (forward outcome)

# Step 4: train meta model
rf_meta = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1)
rf_meta.fit(X_meta, y_meta)

# Step 5: live signal generation
def get_signal_metalabeled(features, xgb_primary, rf_meta, min_meta_prob=0.45):
    p_primary = xgb_primary.predict_proba(features)[0, 1]
    side      = 1 if p_primary >= 0.5 else -1

    meta_features = np.append(features, [p_primary, int(p_primary >= 0.5)])
    p_meta        = rf_meta.predict_proba(meta_features.reshape(1, -1))[0, 1]

    if p_meta < min_meta_prob:
        return {"direction": 0, "reason": f"meta filter rejected (p={p_meta:.2f})"}

    # Size proportional to meta-model conviction
    size_scalar = np.clip((p_meta - min_meta_prob) / (1.0 - min_meta_prob), 0.3, 1.0)
    return {"direction": side, "confidence": p_meta, "size_scalar": size_scalar}
```

**Key benefit:** The meta model acts as the ultimate noise filter. It learns to reject the primary model's signals when conditions are unfavorable (wrong regime, wide spread, low liquidity) without needing explicit hard-coded rules for each condition.

---

### C. Fractional Differentiation for Stationarity
**Status: 🔵 Future feature — conceptually important, complex to implement correctly**

**The problem it solves:**
- Raw price is non-stationary (trends indefinitely) → ML models trained on it learn spurious long-term correlations
- Standard returns (`pct_change()`) are stationary but destroy all price memory — the model cannot know whether price is at a 5-year high or a 5-year low
- Fractional differentiation finds the minimum `d` (0 < d < 1) that makes the series just barely stationary while retaining maximum memory

**Your current approach:** All price-derived features use standard integer differences (returns, EMAs, ATR). This is the standard approach and not wrong — it just sacrifices some information.

**Why it's a future feature, not now:**
- Requires computing the ADF (Augmented Dickey-Fuller) test for each feature series at each `d` value to find the minimum stationary `d`
- Adds ~30–60 min of compute to preprocessing per symbol
- Benefits are real but incremental (~5–10% model improvement estimated)
- Your bigger gaps (labeling, walk-forward, spread model) will give much larger improvements per hour of work

**When to add it:** After priorities 1–4 are fixed and you have a stable baseline. Then fractional differentiation is a meaningful marginal improvement.

```python
# Sketch of fractional differentiation (reference only)
# Full implementation: see mlfinlab library or Lopez de Prado Ch. 5
def frac_diff(series, d, thres=1e-5):
    """Apply fractional differencing with weight cutoff."""
    w = [1.0]
    for k in range(1, len(series)):
        w.append(-w[-1] * (d - k + 1) / k)
        if abs(w[-1]) < thres:
            break
    w = np.array(w[::-1])
    output = pd.Series(index=series.index, dtype=float)
    for i in range(len(w) - 1, len(series)):
        output.iloc[i] = np.dot(w, series.iloc[i - len(w) + 1: i + 1])
    return output.dropna()
```

---

### D. Sample Uniqueness and Concurrency Weighting
**Status: 🔵 Future feature — critical with triple barrier labels, complex**

**The problem this solves (and why it matters for your system):**

Once you switch to triple barrier labeling (priority #4), your training samples will overlap in time. A trade that opens at 10:00 and closes at 10:30 overlaps with a trade that opens at 10:15 and closes at 10:45. Both labels share the 10:15–10:30 price data. Standard ML treats every row as independent. When rows share data, the model double-counts those periods — usually the most volatile ones (because volatility creates more signals) — and overfits to them.

**The fix concept:**
```python
# For each training sample i, calculate what fraction of its bars
# are unique (not shared with any other concurrent open trade)
def get_uniqueness_weights(label_df):
    """
    label_df: DataFrame with columns [open_time, close_time, label]
    Returns: array of weights, one per sample.
    Samples with more concurrent overlapping trades get lower weights.
    """
    weights = []
    for i, row in label_df.iterrows():
        t0, t1 = row["open_time"], row["close_time"]
        # Count how many other trades were open during [t0, t1]
        concurrent = label_df[
            (label_df["open_time"] < t1) & (label_df["close_time"] > t0)
        ]
        # Uniqueness = 1 / avg concurrent count across this trade's duration
        uniqueness = 1.0 / len(concurrent)
        weights.append(uniqueness)
    return np.array(weights)

# Use in XGBoost:
xgb_model.fit(X_train, y_train, sample_weight=uniqueness_weights)
```

**Why it's a future feature:**
- Only becomes relevant after triple barrier labeling is implemented
- Computing pairwise concurrency on 2.8M bars at 1m is expensive (O(n²) naively, needs interval tree optimization)
- Add this in the same sprint as triple barrier labeling, not before

---

### E. Combinatorial Purged Cross-Validation (CPCV)
**Status: 🔵 Future feature — better than walk-forward, significantly more complex**

**What it adds over walk-forward:**
Walk-forward tests a single linear path through history (fold 1 → fold 2 → fold 3). CPCV generates many alternative histories by combining different subsets of data folds, giving a full distribution of out-of-sample P&L paths — not just one.

**The purging component (this part you should add even without full CPCV):**

When you split training and test data, the bars immediately before the test start and immediately after the training end are contaminated by autocorrelation. A trade that opened in the last day of training and closed in the first day of testing leaks future information into the training label.

**Purging is simple and should be added to your walk-forward immediately:**
```python
def purge_boundary(train_df, test_df, purge_bars=10):
    """
    Remove the last `purge_bars` bars from training data
    (the period closest to the test set) to prevent label leakage
    at the train/test boundary.
    """
    cutoff = test_df.index[0]
    train_df = train_df[train_df.index < cutoff]
    # Remove last purge_bars rows (closest to test boundary)
    return train_df.iloc[:-purge_bars]
```

Add `purge_bars=10` to your walk-forward fold construction immediately — it takes 3 lines of code and prevents a subtle but real source of overfitting.

**Full CPCV:** implement after walk-forward is stable and you want to further tighten the statistical validity of reported results.

---

### F. Deflated Sharpe Ratio — Existing Implementation vs Full DSR
**Status: ⚠️ Implemented but incomplete — and not used in strategy selection**

---

#### Why ER Must Stay as the Primary Ranking Metric

Before explaining the DSR fix, it is important to be clear: **Efficiency Ratio (ER) is the correct primary ranking metric for this system and should not be replaced by Sharpe or DSR.**

Here is why. Consider two strategies with identical Sharpe ratios and identical total profit:

| Strategy | Total Profit | Max Drawdown | ER |
|---|---|---|---|
| A | $4,000 | $2,000 | 2.0 |
| B | $4,000 | $500 | 8.0 |

Sharpe treats these as equivalent because the distribution of trade returns looks the same. But they are not equivalent in the real world. Strategy B earned the same $4,000 while only putting $500 at risk at its worst moment. This means:
- You can run Strategy B at **4× the position size** as Strategy A for the same max drawdown budget
- At 4× size, Strategy B generates $16,000 of real profit vs Strategy A's $4,000 — from the same capital, same risk tolerance
- A prop firm or serious account always sizes to max drawdown capacity, not to Sharpe

**ER directly measures capital efficiency: how many dollars of profit did you generate per dollar of maximum pain endured.** This is what institutional performance evaluation cares about. Keep ER as the final ranking criterion.

---

#### The Problem: All Robustness Tools Are Built But None Are Wired as Gates

Your code already computes:
- `haircut_sharpe()` — DSR approximation (selection bias penalty)
- `run_monte_carlo()` — bootstrap P&L distribution, `mc_pass = True/False`
- `run_sensitivity()` — parameter nudge stability score (0–100)

But the actual strategy selection in `train.py` ignores all three:
```python
# What actually happens — raw ER, no robustness gates
valid.sort(key=lambda s: s["efficiency_ratio"], reverse=True)
active_strategy = valid[0]   # highest ER wins, regardless of DSR/MC/sensitivity
```

This means a strategy with:
- DSR = -0.4 (statistically a data mining mirage)
- mc_pass = False (collapses under reordered trade sequences)
- sensitivity_score = 18 (tiny parameter nudge destroys performance)

...can still be selected as the active live trading strategy simply because its ER is highest. The robustness tools exist in the code but have zero influence on what actually trades.

---

#### The Fix: DSR, MC, and Sensitivity as Gates — ER as the Ranker

The correct architecture is a two-stage filter:

**Stage 1 — Robustness gates (pass/fail):**
Three questions that must all pass before a strategy is eligible for selection at all.

**Stage 2 — ER ranking (best wins):**
Among all strategies that passed Stage 1, rank by ER. Highest ER is activated.

```
Selection logic should be:

  Step 1: Is DSR > 0?
          → No  → DISCARD (statistical mirage — lucky backtest, not real edge)
          → Yes → continue

  Step 2: Did Monte Carlo pass? (p5 Sharpe > 0)
          → No  → DISCARD (performance depends on lucky trade ordering)
          → Yes → continue

  Step 3: Is sensitivity_score ≥ 50?
          → No  → DISCARD (cliff-edge parameters — tiny change = collapse)
          → Yes → ELIGIBLE

  Step 4: Among all ELIGIBLE strategies → rank by Efficiency Ratio
          → Highest ER is activated for live trading
```

**What each gate catches:**

- **DSR gate** catches strategies that only look good because Optuna ran 150 trials and picked the luckiest one. DSR < 0 means the strategy's Sharpe is not statistically distinguishable from what you would expect by chance from that many trials.
- **Monte Carlo gate** catches strategies whose backtest profit depended on the specific ordering of wins and losses. If you shuffle the trade sequence 1,000 times and 5%+ of those shuffles go negative, the strategy is fragile to timing luck.
- **Sensitivity gate** catches strategies where the optimized parameters sit on a narrow peak. If nudging confidence by ±0.05 or SL by ±0.2 ATR causes a major performance drop, you have found an overfit parameter set, not a robust edge.
- **ER ranking** then selects the most capital-efficient strategy from the ones that survived all three gates.

---

#### Concrete Implementation

```python
# In train.py — replace current selection logic with this

def select_active_strategy(strategies: list[dict], all_trials: list[dict]) -> dict | None:
    """
    Two-stage selection:
      Stage 1: DSR > 0, MC pass, sensitivity >= 50  (robustness gates)
      Stage 2: rank by Efficiency Ratio              (capital efficiency)
    """
    eligible = []

    for s in strategies:
        strategy_id = s.get("strategy_id", "?")

        # Gate 1: Deflated Sharpe > 0
        dsr = s.get("haircut_sharpe", None)
        if dsr is None:
            # Compute from stored Optuna trials for this strategy's TF
            tf_trials = [t for t in all_trials if t.get("tf") == s.get("entry_tf")]
            dsr = haircut_sharpe(s.get("trades", []), n_trials=len(tf_trials) or 150)
        if dsr <= 0:
            log.info(f"  {strategy_id}: REJECTED by DSR gate (DSR={dsr:.3f} ≤ 0)")
            continue

        # Gate 2: Monte Carlo pass (p5 Sharpe > 0)
        mc = s.get("mc_pass", None)
        if mc is False:
            log.info(f"  {strategy_id}: REJECTED by Monte Carlo gate")
            continue

        # Gate 3: Parameter sensitivity ≥ 50
        sens = s.get("sensitivity_score", 100.0)   # default 100 = not computed = don't block
        if sens < 50.0:
            log.info(f"  {strategy_id}: REJECTED by sensitivity gate (score={sens:.1f})")
            continue

        eligible.append(s)
        log.info(f"  {strategy_id}: ELIGIBLE | ER={s.get('efficiency_ratio', 0):.2f} "
                 f"DSR={dsr:.2f} MC={'pass' if mc else '?'} Sens={sens:.0f}")

    if not eligible:
        log.warning("  No strategy passed all robustness gates — falling back to best raw ER")
        # Fallback: if ALL strategies fail gates, pick least-bad by ER
        # (better than no strategy at all, but logs a warning for review)
        return max(strategies, key=lambda s: s.get("efficiency_ratio", 0)) if strategies else None

    # Stage 2: rank eligible strategies by Efficiency Ratio
    eligible.sort(key=lambda s: s.get("efficiency_ratio", 0), reverse=True)
    winner = eligible[0]
    log.info(f"\n  ACTIVE STRATEGY SELECTED: {winner.get('strategy_id')}")
    log.info(f"    ER={winner.get('efficiency_ratio', 0):.2f}  "
             f"DSR={winner.get('haircut_sharpe', 0):.2f}  "
             f"Sharpe={winner.get('sharpe', 0):.2f}  "
             f"Max DD={winner.get('max_dd_pct', 0):.1f}%")
    return winner
```

**What changes in the HTML report:**
Add a "Robustness" column to the strategy table showing DSR / MC / Sensitivity as a traffic light (🟢 all pass / 🟡 partial / 🔴 failed). This makes it immediately visible which strategies are statistically clean vs which ones only look good because of data mining.

---

## Summary of New Points

| Point | Status | Action |
|---|---|---|
| A. CFD volume is quote activity, not real flow | 🔴 Not addressed | Rename features + validate against YM futures volume |
| B. Meta-labeling (primary model + meta filter model) | 🔴 Not in code | Add after triple barrier labeling is done |
| C. Fractional differentiation | 🔵 Future feature | Add after priorities 1–4 are fixed |
| D. Sample uniqueness / concurrency weighting | 🔵 Future feature | Add in same sprint as triple barrier labeling |
| E. CPCV — add purging now, full CPCV later | ⚠️ Add purging now | 3-line fix for walk-forward boundary leakage |
| F. Deflated Sharpe — exists but unused in selection | ⚠️ Partial | Wire DSR gate into strategy activation logic |

---

---

## Report & HTML — What Is There vs What Is Missing

### What `report.py` Currently Renders (confirmed from code)

| Section | Status | Notes |
|---|---|---|
| Summary cards (ER, Sharpe, WR, PF, Profit, DD%, DD$, Open positions) | ✅ | Good |
| ML timing parameters table (TF, HTF, confidence, SL, RR, BE) | ✅ | Good |
| Equity curve + drawdown chart (top 10 strategies, Plotly) | ✅ | Good |
| Drawdown analysis (max DD%, DD$, period, recovery, top 5 worst days) | ✅ | Good |
| Session analysis (win rate + trade count by session + table) | ✅ | Good |
| Hour-of-day trade distribution heatmap (UTC) | ✅ | Good |
| Trade frequency section (per day/week/month, avg gap) | ✅ | Good |
| Strategy comparison table (all TFs, all ranks) | ✅ | Good |
| Monthly P&L heatmap (green/red grid per symbol) | ✅ | Good |
| Feature importance chart (XGBoost top 15, colour-coded by group) | ✅ | Good |
| Optimisation criteria explanation | ✅ | Good |
| Live trades table (last 50) | ✅ | Good |
| Robustness scores (DSR, MC pass, sensitivity) | ❌ | Computed but never displayed |
| Monte Carlo fan chart | ❌ | Missing entirely |
| Return distribution histogram | ❌ | Missing entirely |
| Rolling performance over time | ❌ | Missing entirely |
| Regime performance breakdown | ❌ | Missing entirely |
| Day-of-week performance | ❌ | `by_dow` computed but never rendered |
| Live vs backtest comparison | ❌ | Data exists in DB, not rendered |
| Sortino / Calmar cards | ❌ | Computed in backtest_engine but not shown |
| Consecutive loss stats | ❌ | Computed but not in report |

---

### What Should Be Added to the HTML Report

#### 1. Robustness Traffic Light Column in Strategy Table
**Priority: High — connect the work already done**

The strategy table shows ER, Sharpe, Win Rate, PF, DD. It does not show the three robustness metrics that should gate strategy selection. Add three columns:

| DSR | MC | Sens |
|---|---|---|
| 🟢 1.4 | 🟢 Pass | 🟢 72 |
| 🔴 -0.2 | 🔴 Fail | 🟡 48 |

```python
# In build_strategy_table(), add to each row:
dsr   = s.get("haircut_sharpe", None)
mc    = s.get("mc_pass", None)
sens  = s.get("sensitivity_score", None)

dsr_cell  = (f'<span style="color:#2ecc71">{dsr:.2f}</span>' if dsr and dsr > 0
             else f'<span style="color:#e74c3c">{dsr:.2f}</span>' if dsr else "—")
mc_cell   = ('<span style="color:#2ecc71">✓ Pass</span>' if mc
             else '<span style="color:#e74c3c">✗ Fail</span>' if mc is False else "—")
sens_cell = (f'<span style="color:#2ecc71">{sens:.0f}</span>' if sens and sens >= 50
             else f'<span style="color:#e74c3c">{sens:.0f}</span>' if sens else "—")
```

This makes it immediately visible at a glance which strategies are statistically clean vs lucky. A strategy with high ER but red DSR and red MC is a warning sign you can see without reading the numbers.

---

#### 2. Monte Carlo Fan Chart
**Priority: High — most visually powerful robustness check**

Currently `mc_pass` (True/False) is computed but there is no visual. A fan chart showing the distribution of 1,000 resampled equity paths tells you far more than a single boolean:

- **Narrow fan** (p5 and p95 close together) = performance is consistent regardless of trade ordering → robust
- **Wide fan** (p5 near zero, p95 at 3×) = performance depends heavily on lucky ordering → fragile

```python
def build_monte_carlo_chart(strategy_id: str, trades: list, n_sims: int = 500) -> str:
    """
    Bootstrap resample trade sequence n_sims times.
    Plot p5 / p25 / p50 / p75 / p95 equity paths as shaded fan.
    """
    if not trades or not HAS_PLOTLY:
        return ""

    pnl_r  = np.array([t["pnl_r"] for t in trades])
    rng    = np.random.default_rng(42)
    curves = []
    for _ in range(n_sims):
        sim    = rng.choice(pnl_r, size=len(pnl_r), replace=True)
        equity = 10000 * np.cumprod(1 + sim * 0.01)
        curves.append(equity)

    curves  = np.array(curves)
    p5      = np.percentile(curves, 5,  axis=0)
    p25     = np.percentile(curves, 25, axis=0)
    p50     = np.percentile(curves, 50, axis=0)
    p75     = np.percentile(curves, 75, axis=0)
    p95     = np.percentile(curves, 95, axis=0)
    x       = list(range(len(p50)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p95)+list(p5[::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="p5–p95"))
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p75)+list(p25[::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.25)",
        line=dict(color="rgba(0,0,0,0)"), name="p25–p75"))
    fig.add_trace(go.Scatter(x=x, y=p50, name="Median",
        line=dict(color="#3498db", width=2)))
    fig.add_hline(y=10000, line_dash="dash", line_color="#555")
    fig.update_layout(
        title=f"Monte Carlo Fan — {strategy_id} ({n_sims} resampled paths)",
        height=380, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"))
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)
```

**Reading it:** If the p5 line (worst 5% of scenarios) stays above the starting balance, the strategy is positive in 95% of all possible trade orderings. If p5 dips below start, some orderings lose money — the strategy has sequence-of-returns risk.

---

#### 3. Return Distribution Histogram
**Priority: Medium — exposes hidden tail risk**

A histogram of per-trade R-values reveals things that Sharpe and win rate hide:

- **Normal-looking bell curve:** safe, symmetric outcomes
- **Heavy left tail:** rare but catastrophic losses (the system occasionally has a very large loser despite good average)
- **Right skew with many small losses + rare large wins:** classic trend-following profile (frustrating to live through but mathematically sound)
- **Positive skew with many small wins + rare large losses:** looks great until the large loss arrives (blowup risk)

```python
def build_return_distribution(strategy_id: str, trades: list) -> str:
    if not trades or not HAS_PLOTLY:
        return ""

    r = [t["pnl_r"] for t in trades]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=40,
        marker_color="#3498db", marker_line_color="#16213e", marker_line_width=1,
        name="Trade R-values",
        hovertemplate="R = %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    # Add vertical lines for key statistics
    mean_r = np.mean(r)
    fig.add_vline(x=0,       line_color="#e74c3c", line_dash="dash",
                  annotation_text="Zero", annotation_position="top")
    fig.add_vline(x=mean_r,  line_color="#2ecc71", line_dash="dot",
                  annotation_text=f"Mean {mean_r:+.2f}R", annotation_position="top right")
    fig.update_layout(
        title=f"Trade Return Distribution — {strategy_id}",
        xaxis_title="R-Multiple per trade",
        yaxis_title="Number of trades",
        height=320, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"))

    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt())
    note = f"""<p class="note" style="margin-top:8px">
        Skewness: {skew:+.2f} (negative = left tail risk, positive = occasional large wins) &nbsp;|&nbsp;
        Excess kurtosis: {kurt:+.2f} (>0 = fat tails = extreme outcomes more common than normal)
        </p>"""
    return pyo.plot(fig, output_type="div", include_plotlyjs=False) + note
```

---

#### 4. Rolling Performance Over Time (Drift Detection)
**Priority: High — most important for live deployment decision**

A 90-day rolling Sharpe and rolling win rate chart answers the most important live trading question: **is the system still working?** A strategy that was strong in 2021–2023 but has been degrading since 2024 will show clearly here before you commit live capital.

```python
def build_rolling_performance(strategy_id: str, trades: list, window: int = 90) -> str:
    """
    Plot rolling 90-day Sharpe and win rate across the backtest period.
    Flat or improving = strategy is stable.
    Declining into recent period = regime change / model decay.
    """
    if not trades or not HAS_PLOTLY:
        return ""

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.set_index("entry_time").sort_index()

    # Rolling 90-day window: compute Sharpe and win rate for each window
    dates, rolling_sharpe, rolling_wr = [], [], []
    all_dates = pd.date_range(df.index.min(), df.index.max(), freq="7D")

    for d in all_dates:
        window_trades = df[df.index >= d - pd.Timedelta(days=window)]
        window_trades = window_trades[window_trades.index <= d]
        if len(window_trades) < 5:
            continue
        r  = window_trades["pnl_r"].values
        sh = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
        wr = (r > 0).mean() * 100
        dates.append(d)
        rolling_sharpe.append(float(sh))
        rolling_wr.append(float(wr))

    if not dates:
        return ""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Rolling 90-day Sharpe", "Rolling 90-day Win Rate (%)"),
                        row_heights=[0.5, 0.5], vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=dates, y=rolling_sharpe, mode="lines",
        line=dict(color="#3498db", width=2), name="Sharpe"), row=1, col=1)
    fig.add_hline(y=0, row=1, col=1, line_color="#e74c3c", line_dash="dash")
    fig.add_hline(y=1, row=1, col=1, line_color="#2ecc71", line_dash="dot",
                  annotation_text="Good (1.0)")

    fig.add_trace(go.Scatter(x=dates, y=rolling_wr, mode="lines",
        line=dict(color="#e67e22", width=2), name="Win Rate %"), row=2, col=1)
    fig.add_hline(y=50, row=2, col=1, line_color="#555", line_dash="dash",
                  annotation_text="50%")

    fig.update_layout(
        title=f"Rolling Performance — {strategy_id} (90-day windows)",
        height=480, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"), showlegend=False)
    note = """<p class="note">Declining Sharpe or win rate in the most recent period
        signals regime change or model decay. If the last 90 days trend toward zero,
        consider retraining before going live.</p>"""
    return pyo.plot(fig, output_type="div", include_plotlyjs=False) + note
```

---

#### 5. Day-of-Week Performance Chart
**Priority: Low — `by_dow` is already computed, just not rendered**

The `_session_stats()` function already computes `by_dow` (Monday–Friday breakdown). It is never used in the HTML. Add a simple bar chart — 5 bars showing win rate per day of week. For US30, Friday often has worse performance due to position unwinding before weekend. If Friday win rate is 38% vs Tuesday at 56%, that's a hard filter worth adding.

```python
def build_dow_chart(strategy_id: str, trades: list) -> str:
    stats  = _session_stats(trades)
    by_dow = stats["by_dow"]
    days   = ["Mon","Tue","Wed","Thu","Fri"]
    counts = [by_dow[i]["n"]    for i in range(5)]
    wr     = [by_dow[i]["wins"] / by_dow[i]["n"] * 100
              if by_dow[i]["n"] > 0 else 0 for i in range(5)]
    colors = ["#2ecc71" if w >= 50 else "#e74c3c" for w in wr]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=days, y=counts, name="Trades",
        marker_color="#3498db", marker_opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=days, y=wr, name="Win Rate %",
        mode="lines+markers+text", text=[f"{w:.0f}%" for w in wr],
        textposition="top center", line=dict(color="#f1c40f", width=2),
        marker=dict(color=colors, size=10)), secondary_y=True)
    fig.add_hline(y=50, secondary_y=True, line_dash="dash", line_color="#555")
    fig.update_layout(title=f"Day-of-Week Performance — {strategy_id}",
        height=300, plot_bgcolor="#1a1a2e", paper_bgcolor="#16213e",
        font=dict(color="#e0e0e0"))
    return pyo.plot(fig, output_type="div", include_plotlyjs=False)
```

---

#### 6. Sortino + Calmar + Max Consecutive Loss Cards
**Priority: Medium — already computed, just not displayed**

`backtest_engine.py` computes `sortino`, `calmar`, and `max_consec_loss` in the stats dict. None appear in the HTML summary cards.

- **Sortino ratio:** like Sharpe but only penalises downside deviation, not upside volatility. A strategy with many small consistent wins and rare losses will have Sortino >> Sharpe. A strategy with symmetric volatility has Sortino ≈ Sharpe. If Sortino >> Sharpe, your upside volatility is inflating the Sharpe denominator — the strategy is actually safer than Sharpe suggests.
- **Calmar ratio:** annualised return divided by max DD%. Normalises for the holding period. Better for comparing strategies across different backtest lengths.
- **Max consecutive losses:** the most psychologically important number for live trading. "How many losses in a row should I expect before assuming the system is broken?" If max consecutive losses in 6 years of backtest is 7, and you hit 8 live losses in a row, that's a statistical anomaly worth investigating.

Add these three cards to the summary row:
```python
{card("Sortino Ratio", f"{sortino:.2f}",
      "#2ecc71" if sortino > 1.5 else "#e67e22",
      "Sharpe penalising only downside — higher = less tail risk")}
{card("Calmar Ratio", f"{calmar:.2f}",
      "#2ecc71" if calmar > 1.0 else "#e67e22",
      "Annualised return / max DD% — >1.0 good")}
{card("Max Consec. Losses", f"{max_consec}",
      "#e74c3c" if max_consec > 6 else "#e67e22",
      "Longest losing streak in 6yr backtest")}
```

---

#### 7. Live vs Backtest Comparison Section
**Priority: High once live trading starts**

Once live trades accumulate (minimum ~30), add a section comparing live performance against backtest expectations. This is your drift detection dashboard.

```python
def build_live_vs_backtest(best_strategy: dict, live_trades: list) -> str:
    """
    Compare live closed trades against backtest expected metrics.
    Highlights if live is performing significantly below expectations.
    """
    closed = [t for t in live_trades if t.get("status") == "closed" and t.get("pnl") is not None]
    if len(closed) < 10:
        return f"""<div class="info-box">
            <h3>Live vs Backtest Comparison</h3>
            <p style="color:#888">Accumulating data... {len(closed)}/10 closed trades needed.
            Check back after more trades close.</p></div>"""

    live_pnls  = [t["pnl"] for t in closed]
    live_risks = [t.get("risk_amount", 100) for t in closed]
    live_r     = [p / r for p, r in zip(live_pnls, live_risks)]

    live_wr    = sum(1 for r in live_r if r > 0) / len(live_r) * 100
    live_pf    = (sum(r for r in live_r if r > 0) /
                  (-sum(r for r in live_r if r < 0) + 1e-10))
    live_exp   = np.mean(live_r)

    bt_wr  = best_strategy.get("win_rate", 50)
    bt_pf  = best_strategy.get("profit_factor", 1.5)
    bt_exp = best_strategy.get("expectancy", 0)

    def _compare_row(label, live_val, bt_val, fmt=".1f", good_positive=True):
        delta = live_val - bt_val
        color = "#2ecc71" if (delta > 0) == good_positive else "#e74c3c"
        return f"""<tr>
            <td>{label}</td>
            <td style="color:{color}">{live_val:{fmt}}</td>
            <td>{bt_val:{fmt}}</td>
            <td style="color:{color}">{delta:+{fmt}}</td>
        </tr>"""

    return f"""<div class="info-box">
        <h3>Live vs Backtest — {best_strategy.get('strategy_id','?')}
            ({len(closed)} closed live trades)</h3>
        <table class="info-table">
            <thead><tr><th>Metric</th><th>Live</th>
                <th>Backtest Expected</th><th>Delta</th></tr></thead>
            <tbody>
                {_compare_row("Win Rate %", live_wr, bt_wr)}
                {_compare_row("Profit Factor", live_pf, bt_pf)}
                {_compare_row("Expectancy (R)", live_exp, bt_exp, ".3f")}
            </tbody>
        </table>
        <p class="note">If live win rate is more than 10pp below backtest over 30+ trades,
        consider retraining. Small samples (< 30 trades) can show large random variation
        even from a healthy strategy — do not over-react to early results.</p>
    </div>"""
```

---

#### 8. CFD Data Integrity Warning in Feature Importance Section
**Priority: Low — informational but prevents misinterpretation**

Add a note below the feature importance chart warning that volume-derived features are CFD quote proxies, not real institutional flow. Prevents interpreting a high `cvd_z` importance as "the model learned real order flow" when it actually learned "Dukascopy's quoting pattern."

```html
<p class="note" style="border-left:3px solid #e67e22; padding-left:8px; margin-top:12px">
  ⚠️ <b>CFD Data Note:</b> Features derived from bid/ask volume
  (cvd, delta, absorption, stacked_imbalance) are based on <em>quote update frequency</em>
  from Dukascopy's pricing engine, not real traded contracts.
  They correlate with real institutional activity but are broker-specific proxies.
  High importance for these features means the model found signal in quoting patterns —
  validate against real futures volume (CME YM) before concluding it reflects true order flow.
</p>
```

---

### Summary: Report Additions Ranked by Priority

| Addition | Priority | Effort | Impact |
|---|---|---|---|
| Robustness traffic lights (DSR/MC/Sensitivity columns in table) | 🔴 High | Low | Makes selection quality immediately visible |
| Rolling 90-day Sharpe + win rate chart | 🔴 High | Medium | Primary drift detection tool for live deployment |
| Monte Carlo fan chart | 🔴 High | Medium | Best visual for regime robustness |
| Live vs backtest comparison section | 🔴 High | Medium | Essential once live trades accumulate |
| Return distribution histogram | 🟡 Medium | Low | Exposes tail risk and skewness |
| Sortino / Calmar / Max consec loss cards | 🟡 Medium | Low | Already computed — 10 lines to add |
| Day-of-week performance chart | 🟢 Low | Low | `by_dow` already computed, just not rendered |
| CFD data warning below feature importance | 🟢 Low | Trivial | Prevents misinterpretation |

---

## Priority Fix List

| Priority | Fix | Impact |
|---|---|---|
| **1** | Align backtest signal with live signal (use same 4-model ensemble, or drop LSTM+PPO from live) | Eliminates strategy selection based on wrong proxy signal |
| **2** | Fix PPO observation dimension: `tail(SEQ_LEN)` → `iloc[-1]` (single bar) | Fixes a live inference bug — current code feeds 60x wrong-sized obs |
| **3** | Replace fixed test-slice optimization with walk-forward | Makes reported metrics actually predictive of live performance |
| **4** | Change ML target from next-bar direction to forward-TP-reached | Aligns ML objective with trading objective |
| **5** | Vectorize volume profile computation (session-level, not bar-level) | Makes training on 1m data feasible without hours of waiting |
| **6** | Fix Sharpe annualization — use daily equity returns, not per-trade R | Makes cross-TF strategy comparison valid |
| **7** | Fix daily risk reset to use equity, not balance | Correctly accounts for floating P&L in daily loss limit |
| **8** | Add realistic spread/slippage model to backtest | Makes P&L estimates credible |
| **9** | Remove `rr`/`tp_mult` redundancy | Reduces optimization search space, removes internal inconsistency |
| **10** | Add session time gate (no trades 21:00–07:30 UTC) | Removes majority of noise trades, improves Sharpe |
| **11** | Add correlation filter for multi-symbol mode | Prevents hidden exposure concentration |
| **12** | Rethink PPO role — gate filter or standalone, not averaged probability | Removes mathematically meaningless ensemble component |

---

## Summary Verdict

**The feature engineering layer (`institutional_features.py`) is genuinely institutional-grade and is the project's real competitive advantage.** Tick-derived volume profile, order flow delta, CVD, absorption, and stop hunt detection are features most retail systems cannot replicate. This foundation is strong.

**The ML training pipeline and optimization framework have serious methodological flaws** that will make backtest results look much better than live trading will actually be. The expected gap between backtest Sharpe and live Sharpe is large — likely 2–4x — primarily because:

1. The optimization signal (XGB+RF only) ≠ the live signal (4-model weighted average)
2. The 15% test set is data-mined by 12,000+ evaluations
3. The ML target (next-bar direction) misaligns with the trading objective (50-bar TP/SL outcome)
4. Zero slippage / spread costs in backtest

Fix priorities 1–4 and this becomes a serious, deployable system. The institutional feature foundation gives you something to build on — most projects fail because they never build this layer correctly. Yours did. Now the methodology around it needs to match that quality.
