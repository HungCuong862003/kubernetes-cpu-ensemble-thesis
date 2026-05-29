"""
f4_v1e_scope_planner.py — Decide how many origins F4 actually needs to predict
                           on, and estimate inference time + disk budget BEFORE
                           committing to a long batch run.

Dry-run output showed: 1,887,147 origins across 3 datasets if we predict at
every test row. That is too many. F4 doesn't need a quantile at every test
timestep -- it needs one at every MPC scheduling timestep. Subsampling drops
the origin count by ~6-12x.

This script:
  1. Inventories test-set timestamps per dataset
  2. Computes origin counts under three MPC cadences:
       (a) every 5 min   (1 step) - matches HPA simulator sync_steps=1
       (b) every 30 min  (6 steps for 5-min datasets, 3 steps for 10-min) - OptScaler default
       (c) every 60 min  (12 steps for 5-min, 6 steps for 10-min) - more conservative
  3. Estimates inference wallclock for each scenario
  4. Estimates output parquet size for each scenario

NO inference is run; just planning numbers.

Run on Vast:
    python phase_f/scripts/f4_v1e_scope_planner.py
"""

from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path("/workspace/kubernetes-cpu-ensemble-thesis")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATASETS = ["alibaba", "bitbrains", "bytedance"]

# Each dataset's intrinsic resolution
DATASET_RES_MIN = {
    "alibaba":   5,
    "bitbrains": 5,
    "bytedance": 10,
}

# F3 day-1 reported batched inference speeds (single-pass; LoRA-FT may differ)
# Alibaba: 5000 containers per horizon in ~3.4s -> 1470 containers/s for a
# single inference per container. For per-origin prediction we need a different
# rate estimate. Reasonable assumption: with batch_size=256 and GPU pinned,
# throughput is ~5000 origins/s on RTX 5070 Ti for 12-step H. Update with
# actual numbers after a 1000-origin micro-benchmark.
ASSUMED_ORIGINS_PER_SEC = 5000.0      # conservative; will calibrate later

# Output schema: 5 quantile columns + step_idx + container_id + time_stamp
# + cpu_true per row, ~50 bytes per row when stored as parquet with snappy.
BYTES_PER_ROW = 50
H_STEPS = 12      # for h=60


def load_test_per_container(dset):
    """Load test.parquet and return per-container row counts."""
    p = DATA_PROCESSED / dset / "test.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    id_col = "container_id" if "container_id" in df.columns else (
        "vm_id" if "vm_id" in df.columns else df.columns[0])
    counts = df.groupby(id_col).size()
    return counts


def origins_under_cadence(counts, cadence_steps):
    """Given per-container test row counts and an MPC cadence (in steps),
    compute total number of origins after subsampling.

    Subsampling rule: take every cadence_steps-th test row per container.
    So origins per container = ceil(n_test / cadence_steps)."""
    if counts is None:
        return 0
    n_per = np.ceil(counts.to_numpy() / cadence_steps).astype(int)
    return int(n_per.sum())


def main():
    print("=" * 70)
    print("F4 V1E: SCOPE + BUDGET PLANNER")
    print("=" * 70)

    print("\n--- inventory per dataset (test split only) ---")
    counts_by_dset = {}
    for dset in DATASETS:
        counts = load_test_per_container(dset)
        if counts is None:
            print(f"  {dset}: MISSING")
            continue
        counts_by_dset[dset] = counts
        res = DATASET_RES_MIN[dset]
        total_rows = int(counts.sum())
        n_cont = len(counts)
        p50 = int(counts.quantile(0.5))
        p95 = int(counts.quantile(0.95))
        # test duration in days
        median_dur_min = p50 * res
        median_dur_d = median_dur_min / 60 / 24
        print(f"  {dset:>10s}: containers={n_cont:>5d}  "
              f"total_test_rows={total_rows:>10d}  "
              f"per-container p50={p50:>5d} p95={p95:>5d}  "
              f"resolution={res}min  median_duration={median_dur_d:.2f} days")

    print("\n--- origin counts under three MPC cadences ---")
    cadences = [
        ("every 5 min  (sync_steps=1, matches HPA simulator)",     None),    # interpret per dataset
        ("every 30 min (OptScaler default h=30)",                  30),
        ("every 60 min (conservative)",                            60),
    ]
    summary_rows = []
    for label, cadence_min in cadences:
        print(f"\n  cadence: {label}")
        total = 0
        for dset, counts in counts_by_dset.items():
            res = DATASET_RES_MIN[dset]
            if cadence_min is None:
                steps = 1
            else:
                steps = max(1, cadence_min // res)
            n_orig = origins_under_cadence(counts, steps)
            total += n_orig
            print(f"    {dset:>10s}: cadence={steps:>2d} steps "
                  f"({steps * res:>2d} min), origins={n_orig:>10d}")
        print(f"    {'TOTAL':>10s}: {total:>10d}")
        summary_rows.append({"cadence": label,
                             "total_origins": total})

    print("\n--- inference time + disk estimates ---")
    print(f"  Assumed throughput: {ASSUMED_ORIGINS_PER_SEC:.0f} origins/sec")
    print(f"  (calibrate this with a 1000-origin micro-benchmark before commit)")
    print()
    print(f"  {'cadence':<55s}  {'origins':>12s}  {'wallclock':>10s}  {'disk':>8s}")
    for row in summary_rows:
        n = row["total_origins"]
        wallclock_s = n / ASSUMED_ORIGINS_PER_SEC
        if wallclock_s < 60:
            wc_str = f"{wallclock_s:.0f}s"
        elif wallclock_s < 3600:
            wc_str = f"{wallclock_s/60:.1f}m"
        else:
            wc_str = f"{wallclock_s/3600:.2f}h"
        disk_mb = n * H_STEPS * BYTES_PER_ROW / 1e6
        if disk_mb < 1000:
            disk_str = f"{disk_mb:.0f}MB"
        else:
            disk_str = f"{disk_mb/1024:.1f}GB"
        print(f"  {row['cadence']:<55s}  {n:>12d}  {wc_str:>10s}  {disk_str:>8s}")

    print("\n--- RECOMMENDATION ---")
    print()
    print("  For F4 alignment with OptScaler protocol: use the 30-min cadence.")
    print("  This matches OptScaler's h=30 default and gives a reasonable origin")
    print("  count (~300K). Inference fits in minutes, output parquet under 1 GB.")
    print()
    print("  The HPA simulator (task2_hpa_v2.py) runs at sync_steps=1 internally.")
    print("  This is fine -- the simulator executes the SCALING DECISION every")
    print("  step, but the MPC SOLVE is run only at scheduling timesteps. Between")
    print("  solves, the planned x_path[1] is held constant. F4's simulator wrap")
    print("  must implement this: solve at t in {0, 6, 12, 18, ...}, hold u^1")
    print("  constant for the intervening 5 steps.")
    print()
    print("  Implication for f4_v1c_resave_f3_quantiles.py: subsample origins to")
    print("  the 30-min cadence before predicting. Reduces inference cost ~6x.")


if __name__ == "__main__":
    main()
