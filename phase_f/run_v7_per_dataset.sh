#!/usr/bin/env bash
# run_v7_per_dataset.sh — orchestrate the v7 per-dataset routing resaver.
#
# Pre-flight steps run BEFORE any compute:
#   A. Verify all batch 1 files are uploaded
#   B. Run verify_router_resolution.py (config + adapters + PEFT version)
#   C. Run peft_adapter_switch_unittest.py (proves set_adapter actually switches)
#   D. Stamp router_v7.yaml with current git SHA
#
# Compute steps:
#   E. Run f4_v1c2_resave_PER_DATASET.py (~10 min)
#   F. Run f4_v1h_sort_quantiles.py against the v7 parquet (~5 sec)
#   G. Report v6 vs v7 pinball@0.9 per dataset side-by-side
#
# Usage:
#   bash run_v7_per_dataset.sh

set -e
cd /workspace/kubernetes-cpu-ensemble-thesis

V7_PARQUET="phase_f/data/f3_quantiles_per_dataset_h060_cad30.parquet"
V7_SORTED="phase_f/data/f3_quantiles_per_dataset_h060_cad30_sorted.parquet"
V6_SORTED="phase_f/data/f3_lora_quantiles_h060_cad30_sorted.parquet"

echo "=========================================="
echo "Step A: verify required files exist"
echo "=========================================="
for f in \
    phase_f/configs/router_v7.yaml \
    phase_f/models/f3_lora_rank8_no_log1p/transform_metadata.json \
    phase_f/scripts/f4_v1c2_resave_PER_DATASET.py \
    phase_f/scripts/transform_io.py \
    phase_f/scripts/f4_v1h_sort_quantiles.py \
    verify_router_resolution.py \
    peft_adapter_switch_unittest.py
do
    if [ ! -f "$f" ]; then
        echo "FAIL: missing $f"
        exit 1
    fi
done
echo "OK: all 7 prerequisite files present"

echo
echo "=========================================="
echo "Step B: router_v7.yaml + adapter integrity"
echo "=========================================="
python verify_router_resolution.py

echo
echo "=========================================="
echo "Step C: PEFT set_adapter unit test (guard vs PEFT #1802)"
echo "=========================================="
python peft_adapter_switch_unittest.py

echo
echo "=========================================="
echo "Step D: stamp router_v7.yaml with git SHA"
echo "=========================================="
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
echo "Current git SHA: $GIT_SHA"
# Update the router_v7.yaml field in-place. Use python to be safe vs sed quirks.
python - <<PYEOF
from pathlib import Path
import yaml
p = Path("phase_f/configs/router_v7.yaml")
with open(p) as f:
    cfg = yaml.safe_load(f)
cfg["router_sha"] = "$GIT_SHA"
with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
print(f"Updated router_sha in {p}")
PYEOF

echo
echo "=========================================="
echo "Step E: run per-dataset resaver (~10 min)"
echo "=========================================="
python phase_f/scripts/f4_v1c2_resave_PER_DATASET.py \
    --router phase_f/configs/router_v7.yaml \
    --horizons h060 --cadence-min 30 \
    2>&1 | tee phase_f/data/f4_v1c2_resave_v7.log

if [ ! -f "$V7_PARQUET" ]; then
    echo "FAIL: v7 parquet $V7_PARQUET not produced. See log for details."
    exit 1
fi
echo "OK: v7 parquet written"

echo
echo "=========================================="
echo "Step F: sort quantiles"
echo "=========================================="
# Patch: the sort script may have a hardcoded input path. We need to
# temporarily point it at our v7 parquet. The cleanest way: run it inline.
python - <<PYEOF
"""Sort v7 quantiles per row (CFG 2010 rearrangement)."""
import numpy as np
import pandas as pd
from pathlib import Path

INPUT  = "$V7_PARQUET"
OUTPUT = "$V7_SORTED"

print("=" * 70)
print("V7 SORT (Chernozhukov-Fernandez-Val-Galichon 2010)")
print("=" * 70)
print(f"input:  {INPUT}")
print(f"output: {OUTPUT}")

df = pd.read_parquet(INPUT)
print(f"  shape: {df.shape}")

q_cols = [c for c in df.columns if c.startswith("q_")]
q_cols_sorted = sorted(q_cols, key=lambda c: float(c.split("_")[1]))
print(f"  quantile columns (in order): {q_cols_sorted}")

q_arr = df[q_cols_sorted].to_numpy()
print(f"  quantile array: {q_arr.shape}")

n_inv_pre = int((np.diff(q_arr, axis=1) < 0).any(axis=1).sum())
worst_pre = float(np.diff(q_arr, axis=1).min())
print(f"\n--- BEFORE sorting ---")
print(f"  rows with inversions: {n_inv_pre} / {len(df)} ({100*n_inv_pre/len(df):.2f}%)")
print(f"  worst negative diff:  {worst_pre:.4f}")

q_sorted = np.sort(q_arr, axis=1)
df_out = df.copy()
for i, c in enumerate(q_cols_sorted):
    df_out[c] = q_sorted[:, i]

n_inv_post = int((np.diff(q_sorted, axis=1) < 0).any(axis=1).sum())
worst_post = float(np.diff(q_sorted, axis=1).min())
print(f"\n--- AFTER sorting ---")
print(f"  rows with inversions: {n_inv_post} / {len(df)} (should be 0)")
print(f"  worst negative diff:  {worst_post:.4f}")

# pinball@0.9 pre vs post
y = df["cpu_true"].values
valid = ~np.isnan(y)
q90_pre  = q_arr[:, q_cols_sorted.index("q_0.9")]
q90_post = q_sorted[:, q_cols_sorted.index("q_0.9")]
def pinball(y, q, tau=0.9):
    err = y - q
    return np.where(err >= 0, tau*err, (tau-1)*err).mean()
print(f"\n--- pinball loss at tau=0.9 (overall) ---")
print(f"  pre-sort:  {pinball(y[valid], q90_pre[valid]):.6f}")
print(f"  post-sort: {pinball(y[valid], q90_post[valid]):.6f}")

print(f"\n  writing to {OUTPUT} ...")
df_out.to_parquet(OUTPUT, index=False)
print(f"  done. shape: {df_out.shape}")
PYEOF

echo
echo "=========================================="
echo "Step G: v6 vs v7 pinball@0.9 side-by-side"
echo "=========================================="
python - <<PYEOF
import numpy as np
import pandas as pd

V6 = "$V6_SORTED"
V7 = "$V7_SORTED"

def pinball(y, q, tau=0.9):
    err = y - q
    return np.where(err >= 0, tau*err, (tau-1)*err).mean()

def cap_active_rate(q, cap=110.0):
    return float((np.abs(q - cap) < 0.01).mean())

print(f"{'='*70}")
print(f"V6 (uniform log1p+cap=110) vs V7 (per-dataset)")
print(f"{'='*70}")

df6 = pd.read_parquet(V6)
df7 = pd.read_parquet(V7)
print(f"v6 rows: {len(df6)}, v7 rows: {len(df7)}")

print(f"\n{'dataset':12s} {'pinball v6':>12s} {'pinball v7':>12s} {'delta':>10s} {'cap6':>8s} {'cap7':>8s}")
print(f"{'-'*72}")
for ds in ["alibaba", "bitbrains", "bytedance"]:
    s6 = df6[df6.dataset == ds]
    s7 = df7[df7.dataset == ds]
    s6v = s6[s6.cpu_true.notna()]
    s7v = s7[s7.cpu_true.notna()]
    p6 = pinball(s6v.cpu_true.values, s6v["q_0.9"].values)
    p7 = pinball(s7v.cpu_true.values, s7v["q_0.9"].values)
    c6 = cap_active_rate(s6v["q_0.95"].values) * 100
    c7 = cap_active_rate(s7v["q_0.95"].values) * 100
    delta = p7 - p6
    arrow = "DOWN" if delta < 0 else "UP"
    print(f"{ds:12s} {p6:>12.4f} {p7:>12.4f} {delta:>+10.4f} {c6:>7.2f}% {c7:>7.2f}%   {arrow}")

print(f"\nNotes:")
print(f"  pinball delta < 0 means v7 IMPROVED on that dataset")
print(f"  cap%: percentage of q_0.95 values exactly at 110.0 (cap activations)")
print(f"  v7 Alibaba should be ~0.42 (close to F3 no-log1p baseline)")
print(f"  v7 Bitbrains and ByteDance should be unchanged from v6")
PYEOF

echo
echo "=========================================="
echo "v7 RUN COMPLETE."
echo "=========================================="
echo "Outputs:"
echo "  $V7_PARQUET"
echo "  $V7_SORTED"
echo "  phase_f/data/router_resolved_v7.json"
echo "  phase_f/data/f4_v1c2_resave_v7.log"
echo
echo "Next: paste the 'V6 vs V7' table back to Claude for sanity check."
