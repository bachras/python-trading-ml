"""
debug_rf_oom.py — verify RF max_samples fix handles large datasets without OOM.

Simulates the exact scenario: RF fit on ~1.77M rows with old params (n_jobs=-1)
vs new params (n_jobs=4, max_samples=500_000). Uses synthetic data of same shape.
"""
import gc
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

GLOBAL_SEED = 42
N_ROWS = 1_770_000
N_FEATS = 80   # approximate feature count in training data

rng = np.random.default_rng(GLOBAL_SEED)
print(f"Generating synthetic data: {N_ROWS:,} rows × {N_FEATS} features ...")
X = rng.standard_normal((N_ROWS, N_FEATS), dtype=np.float32).astype(np.float64)
y = rng.integers(0, 3, size=N_ROWS)

# ── NEW config (fix) ──────────────────────────────────────────────────────────
print("\n[NEW] n_jobs=4, max_samples=500_000")
rf_new = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10,
    n_jobs=4, max_samples=500_000, random_state=GLOBAL_SEED,
)
t0 = time.perf_counter()
try:
    rf_new.fit(X, y)
    elapsed = time.perf_counter() - t0
    print(f"  [PASS] fit completed in {elapsed:.1f}s")
    acc = (rf_new.predict(X[:10_000]) == y[:10_000]).mean()
    print(f"  sample accuracy on first 10K rows: {acc:.3f}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {e}")
finally:
    del rf_new
    gc.collect()

print("\nDone.")
