"""
build_per_container_r2.py — reconstruct per-container R² for Alibaba.

Phase 0 documented `per_container_r2.csv` (19,684 rows = 4,921 containers × 4
horizons) but the file isn't on this machine. This script regenerates it from
the saved .npy prediction files and the raw test parquet.

Alignment rule (from Phase 0): per container, the saved prediction arrays
correspond to rows [h_steps : len], where h_steps = horizon_minutes / 5.
Equivalently, the trailing h_steps rows of each container have no future
label and were dropped before training/prediction.

Verification: aggregate R² recomputed under this rule must match
`results/thesis_figures/comparison_table.csv` to within 0.005. If it doesn't,
the alignment is wrong and we need to replicate the full feature-engineering
dropna instead of using the simple end-trim.

Output: per_container_r2.csv (current directory)
Schema: container_id, horizon, n_points, et_r2, naive_r2,
        et_minus_naive_pp, note
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- paths --------------------------------------------------------
PROJECT_ROOT = Path("E:/k8s-ensemble-forecast")
TEST_PARQUET = PROJECT_ROOT / "data" / "alibaba" / "test.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "alibaba"
OUTPUT_CSV = Path("per_container_r2.csv")  # written to wherever script runs

HORIZONS = ["10min", "30min", "60min", "120min"]
H_STEPS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}

# Ground-truth aggregate R² from comparison_table.csv (verified in /mnt/project)
EXPECTED_AGG_ET = {
    "10min": 0.9212,
    "30min": 0.8402,
    "60min": 0.7997,
    "120min": 0.7624,
}
EXPECTED_AGG_NAIVE = {
    "10min": 0.9188,
    "30min": 0.8361,
    "60min": 0.7878,
    "120min": 0.7178,
}

# c_38893 anchor from Phase 0 summary
ANCHOR_CONTAINER = "c_38893"
EXPECTED_ANCHORS = {
    "30min": 1.05,
    "120min": 47.12,
}


def compute_r2(y_true, y_pred):
    """
    Manual R² so we can detect constant y_true case explicitly.
    Returns (r2, note) where note is one of '', 'constant_y_true',
    'too_few_points'.
    """
    if len(y_true) < 2:
        return np.nan, "too_few_points"
    var = float(np.var(y_true))
    if var == 0.0:
        return np.nan, "constant_y_true"
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot, ""


def main():
    # ---------- 1. Load test parquet -------------------------------------
    print(f"Loading {TEST_PARQUET}")
    if not TEST_PARQUET.exists():
        print(f"ERROR: parquet not found at {TEST_PARQUET}")
        sys.exit(1)

    df = pd.read_parquet(TEST_PARQUET)
    print(f"  rows: {len(df):,}")
    print(f"  columns: {list(df.columns)}")

    # Defensive: confirm required columns
    for col in ("container_id", "time_stamp", "cpu_util_percent"):
        if col not in df.columns:
            print(f"ERROR: missing column {col!r} in test.parquet")
            print(f"  Available columns: {list(df.columns)}")
            sys.exit(1)

    n_containers = df["container_id"].nunique()
    print(f"  unique containers: {n_containers}")
    if n_containers != 4921:
        print(f"  WARNING: expected 4921 containers, got {n_containers}")

    # Sort once. Lexicographic on container_id, ascending on time_stamp.
    # This must match the order sprint1_Main.py used when writing the .npy
    # files. The aggregate R² check below confirms it does.
    print("  sorting by (container_id, time_stamp)...")
    df = df.sort_values(["container_id", "time_stamp"]).reset_index(drop=True)

    # Build per-container cpu series dict (one pass, O(n))
    print("  building per-container series dict...")
    container_cpu = {}
    for cid, group in df.groupby("container_id", sort=True):
        container_cpu[cid] = group["cpu_util_percent"].to_numpy(dtype=np.float64)
    container_ids = sorted(container_cpu.keys())
    print(f"  per-container series ready: {len(container_ids)} containers")

    # ---------- 2. For each horizon, align and compute R² ----------------
    rows_out = []
    alignment_ok = True

    for horizon in HORIZONS:
        print(f"\n--- {horizon} ---")
        h_steps = H_STEPS[horizon]

        pred_et_path = RESULTS_DIR / horizon / "pred_extratrees.npy"
        pred_naive_path = RESULTS_DIR / horizon / "pred_naive.npy"
        if not pred_et_path.exists() or not pred_naive_path.exists():
            print(f"  ERROR: missing prediction files at {RESULTS_DIR / horizon}")
            sys.exit(1)

        pred_et = np.load(pred_et_path).astype(np.float64).ravel()
        pred_naive = np.load(pred_naive_path).astype(np.float64).ravel()
        print(f"  pred_extratrees.npy: shape={pred_et.shape}")
        print(f"  pred_naive.npy     : shape={pred_naive.shape}")

        if len(pred_et) != len(pred_naive):
            print(f"  ERROR: pred_et and pred_naive length mismatch: "
                  f"{len(pred_et)} vs {len(pred_naive)}")
            sys.exit(1)

        # Build flat y_true by concatenating per-container shifted series
        # and record (start, end) offsets per container.
        y_true_chunks = []
        offsets = {}
        cursor = 0
        for cid in container_ids:
            cpu = container_cpu[cid]
            n = len(cpu)
            if n <= h_steps:
                # Container too short for this horizon — empty slice
                offsets[cid] = (cursor, cursor, n)  # (start, end, raw_len)
                continue
            y_true_c = cpu[h_steps:]  # row i's target is cpu_util at i+h_steps
            offsets[cid] = (cursor, cursor + len(y_true_c), n)
            y_true_chunks.append(y_true_c)
            cursor += len(y_true_c)

        y_true = np.concatenate(y_true_chunks) if y_true_chunks else np.array([])
        print(f"  reconstructed y_true length: {len(y_true):,}")

        # ---- alignment check ----
        if len(y_true) != len(pred_et):
            diff = len(y_true) - len(pred_et)
            print(f"  ALIGNMENT MISMATCH at {horizon}: "
                  f"y_true={len(y_true):,}, pred_et={len(pred_et):,}, "
                  f"diff={diff:,}")
            print(f"  The simple 'drop last h_steps per container' rule fails here.")
            print(f"  You need to replicate sprint1_Main.py's full feature "
                  f"engineering dropna step. Stopping.")
            alignment_ok = False
            break

        # ---- aggregate R² check vs comparison_table.csv ----
        agg_et, _ = compute_r2(y_true, pred_et)
        agg_naive, _ = compute_r2(y_true, pred_naive)
        exp_et = EXPECTED_AGG_ET[horizon]
        exp_naive = EXPECTED_AGG_NAIVE[horizon]
        print(f"  aggregate ET R²    : {agg_et:.4f}  (expected {exp_et:.4f})")
        print(f"  aggregate Naive R² : {agg_naive:.4f}  (expected {exp_naive:.4f})")

        if abs(agg_et - exp_et) > 0.005:
            print(f"  WARNING: ET aggregate R² off by {abs(agg_et - exp_et):.4f}")
            alignment_ok = False
        if abs(agg_naive - exp_naive) > 0.005:
            print(f"  WARNING: Naive aggregate R² off by {abs(agg_naive - exp_naive):.4f}")
            alignment_ok = False

        # ---- per-container R² ----
        anchor_delta = None
        for cid in container_ids:
            start, end, raw_len = offsets[cid]
            n_pts = end - start
            if n_pts < 2:
                rows_out.append({
                    "container_id": cid,
                    "horizon": horizon,
                    "n_points": n_pts,
                    "et_r2": np.nan,
                    "naive_r2": np.nan,
                    "et_minus_naive_pp": np.nan,
                    "note": "too_few_points",
                })
                continue

            yt = y_true[start:end]
            ye = pred_et[start:end]
            yn = pred_naive[start:end]
            et_r2, et_note = compute_r2(yt, ye)
            naive_r2, naive_note = compute_r2(yt, yn)
            note = et_note or naive_note  # both 'constant_y_true' if either fires
            if not np.isnan(et_r2) and not np.isnan(naive_r2):
                delta_pp = (et_r2 - naive_r2) * 100.0
            else:
                delta_pp = np.nan
            rows_out.append({
                "container_id": cid,
                "horizon": horizon,
                "n_points": n_pts,
                "et_r2": et_r2,
                "naive_r2": naive_r2,
                "et_minus_naive_pp": delta_pp,
                "note": note,
            })
            if cid == ANCHOR_CONTAINER:
                anchor_delta = delta_pp

        # ---- anchor check ----
        if anchor_delta is not None and horizon in EXPECTED_ANCHORS:
            exp = EXPECTED_ANCHORS[horizon]
            print(f"  {ANCHOR_CONTAINER} et_minus_naive_pp = {anchor_delta:.2f}  "
                  f"(expected ~{exp})")
            if abs(anchor_delta - exp) > max(0.5, abs(exp) * 0.05):
                print(f"  WARNING: anchor for {ANCHOR_CONTAINER} at {horizon} is off")

    # ---------- 3. Write CSV ---------------------------------------------
    if not alignment_ok:
        print("\n*** Alignment checks failed. NOT writing output. ***")
        print("    Investigate before proceeding.")
        sys.exit(1)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV.resolve()} with {len(out_df):,} rows "
          f"(expected {n_containers * len(HORIZONS):,})")

    # quick distributional summary
    for h in HORIZONS:
        sub = out_df[out_df["horizon"] == h]
        valid = sub.dropna(subset=["et_r2"])
        print(f"  {h}: {len(sub)} rows, "
              f"{len(valid)} with valid R², "
              f"median ET R² = {valid['et_r2'].median():.4f}")


if __name__ == "__main__":
    main()
