"""
debug_session_alignment.py
===========================
Quick diagnostic for is_london timezone alignment.

Usage:
    python debug_session_alignment.py          # checks 1m parquet
    python debug_session_alignment.py --tf 5   # checks 5m parquet

Outputs:
  - Index timezone of the parquet
  - Current time in UTC / London / EET
  - Discrepancy count between OLD (UTC-naive) and NEW (DST-aware) is_london
  - Discrepancy count between stored parquet is_london and NEW computation
  - Sample rows around the first discrepancy
"""

import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Minimal path setup — avoid importing heavy training modules
from pathlib import Path
import os
DATA_DIR = Path(os.getenv("BASE_DIR", r"F:\trading_ml")) / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--tf", type=int, default=1, help="Timeframe in minutes (default: 1)")
parser.add_argument("--rows", type=int, default=500, help="Tail rows to inspect")
args = parser.parse_args()

tf   = args.tf
rows = args.rows

# ── 1. Current wall-clock in relevant zones ─────────────────────────────────
now_utc    = datetime.now(timezone.utc)
now_london = now_utc.astimezone(__import__("zoneinfo").ZoneInfo("Europe/London"))
now_eet    = now_utc.astimezone(__import__("zoneinfo").ZoneInfo("Europe/Helsinki"))

print("=" * 60)
print("SESSION ALIGNMENT DIAGNOSTIC")
print("=" * 60)
print(f"Current time  UTC   : {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Current time  London: {now_london.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"Current time  EET   : {now_eet.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print()

# ── 2. Load parquet tail ─────────────────────────────────────────────────────
sym    = "US30"
pq_path = DATA_DIR / f"{sym}_{tf}m_ticks.parquet"
if not pq_path.exists():
    print(f"[ERROR] Parquet not found: {pq_path}")
    sys.exit(1)

df = pd.read_parquet(pq_path).tail(rows)
print(f"Parquet: {pq_path.name}  ({len(df):,} rows tail)")
print(f"Index dtype : {df.index.dtype}")
print(f"Index tzinfo: {df.index.tzinfo}")
print(f"Index range : {df.index[0]}  to  {df.index[-1]}")
print()

# ── 3. Compute is_london three ways ──────────────────────────────────────────

# A) Stored in parquet (built by tick_pipeline.py)
stored = df["is_london"].astype(int) if "is_london" in df.columns else None

# B) OLD method: naive UTC hour (current phase2_adaptive_engine before fix)
hr_utc = df.index.hour
old_london = ((hr_utc >= 8) & (hr_utc < 16)).astype(int)

# C) NEW method: DST-aware London time (fixed phase2_adaptive_engine)
try:
    uk_idx = df.index.tz_convert("Europe/London")
except TypeError:
    uk_idx = df.index.tz_localize("UTC").tz_convert("Europe/London")
uk_mins    = uk_idx.hour * 60 + uk_idx.minute
new_london = ((uk_mins >= 480) & (uk_mins < 1020)).astype(int)

# ── 4. Discrepancy report ─────────────────────────────────────────────────────
print("--- is_london comparison ---")
if stored is not None:
    diff_old = int((stored != old_london).sum())
    diff_new = int((stored != new_london).sum())
    pct_old  = 100 * diff_old / len(df)
    pct_new  = 100 * diff_new / len(df)
    print(f"Stored vs OLD (UTC-naive) : {diff_old:4d}/{len(df)} ({pct_old:.1f}%)  diff")
    print(f"Stored vs NEW (DST-aware) : {diff_new:4d}/{len(df)} ({pct_new:.1f}%)  diff")
    if diff_new == 0:
        print("[OK] NEW method matches stored parquet perfectly")
    else:
        print("[WARN] NEW method still differs — investigate further")
else:
    print("[WARN] is_london not present in parquet — only comparing old vs new")

diff_methods = int((old_london != new_london).sum())
print(f"OLD vs NEW               : {diff_methods:4d}/{len(df)} ({100*diff_methods/len(df):.1f}%)  bars differ")
print()

# ── 5. Sample rows around the first discrepancy (old vs new) ─────────────────
mismatch_idx = np.where(old_london != new_london)[0]
if len(mismatch_idx) == 0:
    print("No mismatch between old and new methods in this tail slice.")
else:
    first_m = mismatch_idx[0]
    lo = max(0, first_m - 3)
    hi = min(len(df), first_m + 4)
    sample = df.iloc[lo:hi].copy()
    sample["utc_hour"]  = hr_utc[lo:hi]
    sample["uk_hour"]   = uk_idx[lo:hi].hour
    sample["is_london_stored"] = stored[lo:hi] if stored is not None else np.nan
    sample["is_london_old"]    = old_london[lo:hi]
    sample["is_london_new"]    = new_london[lo:hi]
    cols_show = ["utc_hour", "uk_hour", "is_london_stored", "is_london_old", "is_london_new"]
    print(f"Sample rows around first mismatch (iloc {lo}..{hi}):")
    print(sample[cols_show].to_string())
    print()
    # Show DST transition dates
    transition_dates = sorted(set(df.index[mismatch_idx].date))
    print(f"Mismatch dates ({len(transition_dates)} unique): {transition_dates[:10]}")
    print()

# ── 6. Session stats in this slice ────────────────────────────────────────────
print("--- Session coverage in tail slice ---")
total = len(df)
for label, col in [("OLD is_london", old_london), ("NEW is_london", new_london)]:
    pct = 100 * col.sum() / total
    print(f"  {label}: {col.sum():6,} bars in-session ({pct:.1f}%)")
if stored is not None:
    pct = 100 * stored.sum() / total
    print(f"  Stored     : {stored.sum():6,} bars in-session ({pct:.1f}%)")

print()
print("Done. If 'Stored vs NEW diff = 0' then the fix is correct.")
print("Run: python train.py  to verify [PARITY] PASS")
