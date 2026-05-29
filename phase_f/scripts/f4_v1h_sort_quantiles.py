"""
f4_v1h_sort_quantiles.py — Apply per-row quantile sorting to the existing
                            f3_lora_quantiles_h060_cad30.parquet, eliminating
                            the 8% inversion rate.

Justification: Chernozhukov, Fernandez-Val & Galichon (2010, Econometrica 78,
1093-1125) prove that sorting quantile estimates is the projection onto the
monotone cone in L-infinity and cannot worsen the quantile estimation L^p
loss. Fakoor, Kim, Mueller, Smola & Tibshirani (2023, JMLR 24:162, Prop. 1)
prove the same for pinball loss specifically.

Output:
  phase_f/data/f3_lora_quantiles_h060_cad30_sorted.parquet

Run on Vast:
    python phase_f/scripts/f4_v1h_sort_quantiles.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path("/workspace/kubernetes-cpu-ensemble-thesis")
IN_PATH  = PROJECT_ROOT / "phase_f" / "data" / "f3_lora_quantiles_h060_cad30.parquet"
OUT_PATH = PROJECT_ROOT / "phase_f" / "data" / "f3_lora_quantiles_h060_cad30_sorted.parquet"

Q_COLS = ["q_0.5", "q_0.7", "q_0.8", "q_0.9", "q_0.95"]


def count_inversions(arr):
    """Count rows where any consecutive pair is non-monotonic."""
    diffs = np.diff(arr, axis=1)
    bad_rows = (diffs < 0).any(axis=1)
    return int(bad_rows.sum()), float(diffs[diffs < 0].min()) if bad_rows.any() else 0.0


def main():
    print("=" * 70)
    print("F4 V1H: SORT QUANTILES PER ROW")
    print("=" * 70)
    print(f"input:  {IN_PATH}")
    print(f"output: {OUT_PATH}")

    print("\nloading parquet ...")
    df = pd.read_parquet(IN_PATH)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")

    # Extract quantile matrix
    q_arr = df[Q_COLS].to_numpy()
    print(f"\n  quantile array: {q_arr.shape}")

    # Pre-sort diagnostics
    print("\n--- BEFORE sorting ---")
    n_bad_pre, max_inv_pre = count_inversions(q_arr)
    print(f"  rows with inversions: {n_bad_pre} / {len(q_arr)} "
          f"({100.0 * n_bad_pre / len(q_arr):.2f}%)")
    print(f"  worst negative diff:  {max_inv_pre:.4f}")

    # Apply sort
    print("\n  applying per-row sort ...")
    q_sorted = np.sort(q_arr, axis=1)

    # Post-sort verification
    print("\n--- AFTER sorting ---")
    n_bad_post, max_inv_post = count_inversions(q_sorted)
    print(f"  rows with inversions: {n_bad_post} / {len(q_sorted)} "
          f"(should be 0)")
    print(f"  worst negative diff:  {max_inv_post:.4f}")

    if n_bad_post != 0:
        print("  ERROR: sorting failed to enforce monotonicity")
        return

    # Compute pinball loss change for the canonical q_0.9 (col index 3)
    # Only rows with truth
    has_truth = df["cpu_true"].notna()
    if has_truth.sum() > 0:
        truth = df.loc[has_truth, "cpu_true"].to_numpy()
        q90_pre = q_arr[has_truth.to_numpy(), 3]
        q90_post = q_sorted[has_truth.to_numpy(), 3]

        def pinball(y_true, q_pred, tau=0.9):
            err = y_true - q_pred
            return 2.0 * np.maximum(tau * err, (tau - 1.0) * err).mean()

        pin_pre  = pinball(truth, q90_pre)
        pin_post = pinball(truth, q90_post)
        print(f"\n--- pinball loss at tau=0.9 (overall) ---")
        print(f"  pre-sort:  {pin_pre:.6f}")
        print(f"  post-sort: {pin_post:.6f}")
        print(f"  delta:     {pin_post - pin_pre:+.6f} "
              f"({100*(pin_post - pin_pre)/pin_pre:+.3f}%)")
        if pin_post > pin_pre + 1e-6:
            print("  WARN: post-sort pinball is HIGHER than pre-sort.")
            print("        This contradicts CFG (2010); likely a numerical edge case.")
        else:
            print("  CFG (2010) holds: post-sort pinball cannot exceed pre-sort.")

    # Write back
    print("\n  writing sorted quantiles to output ...")
    for i, c in enumerate(Q_COLS):
        df[c] = q_sorted[:, i]
    df.to_parquet(OUT_PATH, index=False)
    print(f"  wrote: {OUT_PATH}")
    print(f"  size:  {OUT_PATH.stat().st_size / 1e6:.2f} MB")
    print(f"  shape: {df.shape}")

    print("\n" + "=" * 70)
    print("DONE. Use f3_lora_quantiles_h060_cad30_sorted.parquet for F4 MPC.")
    print("=" * 70)


if __name__ == "__main__":
    main()
