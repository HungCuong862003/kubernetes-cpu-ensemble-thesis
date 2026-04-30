"""
Task 0: Extract per-container test arrays from cached pipeline results.

Purpose: Foundation models (Chronos-Bolt, TimesFM) need per-container time
series, but sprint1_Main.py saves predictions as flat concatenated .npy arrays
without container boundary info. This script reconstructs the mapping.

What it does:
  1. Loads test.parquet from Drive (raw data)
  2. Re-runs feature engineering (fast, no model training) to reproduce test_fe
  3. Extracts container_id ordering after dropna (matches saved .npy arrays)
  4. Loads saved pred_*.npy and maps them to per-container slices
  5. Saves per-container arrays for foundation model inference

Usage on Colab:
  !python3 task0_extract_per_container.py \
      --data-dir /content/drive/MyDrive/sprint1\ v9\ results/data \
      --checkpoint-dir /content/drive/MyDrive/sprint1\ v9\ results/sprint1 \
      --output-dir /content/drive/MyDrive/thesis_upgrade/per_container \
      --horizons 10min 30min 60min 120min

Output structure:
  output_dir/
    10min/
      container_index.csv    (container_id, start_idx, end_idx, n_points)
      raw_cpu_series.npz     (key = container_id string → 1D numpy array)
      y_true.npy             (flat, same as pipeline)
      y_naive.npy            (flat, same as pipeline)
      pred_hetero_ensemble.npy (flat, same as pipeline)
    30min/ ...
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Minimal feature engineering (copied from sprint1_Main.py) ──────────────

def _compute_rolling_only(df):
    """Reproduce the exact rolling computation from sprint1_Main.py."""
    df = df.copy()
    df = df.sort_values(["container_id", "time_stamp"])
    g = df.groupby("container_id")

    for metric in ["cpu", "mem"]:
        col = f"{metric}_util_percent"
        pre = f"{metric}_roll"
        for window in [6, 12, 24, 48]:
            for stat in ["mean", "std", "min", "max"]:
                col_name = f"{pre}_{stat}_{window}"
                df[col_name] = g[col].transform(
                    lambda x, w=window, f=stat: getattr(
                        x.rolling(w, min_periods=1), f)()
                )

    hour = (df["time_stamp"] % 86400) / 3600
    dow  = (df["time_stamp"] // 86400) % 7
    df["hour_sin"]       = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"]       = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin"]        = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos"]        = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    df["business_hours"] = ((hour >= 9) & (hour <= 17)).astype(np.float32)

    return df


def create_features_minimal(df, horizon_steps):
    """Minimal feature engineering — just enough to reproduce the exact dropna
    behavior and container_id ordering from sprint1_Main.py."""
    min_lag = horizon_steps + 1
    base = _compute_rolling_only(df)
    g = base.groupby("container_id")

    # Lag features (need at least one for dropna to work on cpu_residual)
    for offset in [0, 1, 2, 3, 5, 11, 23, 47]:
        lag = min_lag + offset
        if lag <= 96:
            base[f"cpu_lag_{lag}"] = g["cpu_util_percent"].shift(lag)
            base[f"mem_lag_{lag}"] = g["mem_util_percent"].shift(lag)

    # Shift rolling features
    roll_cols = [c for c in base.columns if "_roll_" in c]
    for col in roll_cols:
        base[col] = g[col].shift(min_lag)

    # Same-time-yesterday (ppd=288 for Alibaba 5-min intervals)
    ppd = 288
    hist_lag = ppd + min_lag
    base["cpu_same_time_1d"] = g["cpu_util_percent"].shift(hist_lag)
    base["mem_same_time_1d"] = g["mem_util_percent"].shift(hist_lag)

    # Targets and naive baseline — these are what dropna checks
    base["cpu_target"]   = g["cpu_util_percent"].shift(-horizon_steps)
    base["mem_target"]   = g["mem_util_percent"].shift(-horizon_steps)
    base["naive_cpu"]    = base["cpu_util_percent"]
    base["naive_mem"]    = base["mem_util_percent"]
    base["cpu_residual"] = base["cpu_target"] - base["naive_cpu"]
    base["mem_residual"] = base["mem_target"] - base["naive_mem"]

    # Fill nans for std/volatility cols (same as pipeline)
    for pat in ["_std_", "volatility", "_cv", "_range", "accel", "jerk"]:
        cols = [c for c in base.columns if pat in c]
        if cols:
            base[cols] = base[cols].fillna(0)

    return base


# ── Main extraction logic ──────────────────────────────────────────────────

HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}


def extract_horizon(test_df, horizon_name, checkpoint_dir, output_dir):
    """Extract per-container arrays for one horizon."""
    horizon_steps = HORIZONS[horizon_name]
    hz_ckpt = os.path.join(checkpoint_dir, horizon_name)
    hz_out  = os.path.join(output_dir, horizon_name)
    os.makedirs(hz_out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {horizon_name} (steps={horizon_steps})")
    print(f"{'='*60}")

    # Step 1: Re-run feature engineering on test data
    print("  Building features (no model training)...")
    test_fe = create_features_minimal(test_df, horizon_steps)

    # Step 2: Apply exact same dropna as pipeline
    required = ["cpu_target", "cpu_residual", "naive_cpu",
                "mem_target", "mem_residual", "naive_mem"]
    test_fe = test_fe.dropna(subset=required).reset_index(drop=True)
    print(f"  test_fe after dropna: {len(test_fe):,} rows")

    # Step 3: Extract container ordering
    cids = test_fe["container_id"].values
    unique_cids = []
    boundaries = []
    start = 0
    current_cid = cids[0]
    for i in range(1, len(cids)):
        if cids[i] != current_cid:
            unique_cids.append(current_cid)
            boundaries.append((start, i))
            current_cid = cids[i]
            start = i
    unique_cids.append(current_cid)
    boundaries.append((start, len(cids)))

    print(f"  Containers: {len(unique_cids)}")

    # Step 4: Verify against saved predictions
    pred_naive_path = os.path.join(hz_ckpt, "pred_naive.npy")
    pred_ens_path   = os.path.join(hz_ckpt, "pred_hetero_ensemble.npy")

    if os.path.exists(pred_naive_path):
        saved_naive = np.load(pred_naive_path)
        if len(saved_naive) == len(test_fe):
            print(f"  ✓ pred_naive.npy length matches ({len(saved_naive):,})")
        else:
            print(f"  ✗ LENGTH MISMATCH: pred_naive.npy={len(saved_naive)}, "
                  f"test_fe={len(test_fe)}")
            print(f"    This means the dropna behavior differs. Check data version.")
            return False
    else:
        print(f"  ✗ pred_naive.npy not found at {pred_naive_path}")
        return False

    if os.path.exists(pred_ens_path):
        saved_ens = np.load(pred_ens_path)
        print(f"  ✓ pred_hetero_ensemble.npy length matches ({len(saved_ens):,})")
    else:
        print(f"  ! pred_hetero_ensemble.npy not found (may not be needed)")
        saved_ens = None

    # Step 5: Save container boundary index
    index_rows = []
    for cid, (s, e) in zip(unique_cids, boundaries):
        index_rows.append({
            "container_id": cid,
            "start_idx": s,
            "end_idx": e,
            "n_points": e - s
        })
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_csv(os.path.join(hz_out, "container_index.csv"), index=False)
    print(f"  Saved container_index.csv ({len(idx_df)} containers)")

    # Step 6: Save raw per-container CPU time series for foundation models
    # Foundation models need the RAW cpu values in temporal order, NOT features
    raw_cpu = test_fe["cpu_util_percent"].values.astype(np.float32)
    raw_series = {}
    for cid, (s, e) in zip(unique_cids, boundaries):
        raw_series[str(cid)] = raw_cpu[s:e]
    np.savez_compressed(os.path.join(hz_out, "raw_cpu_series.npz"), **raw_series)
    print(f"  Saved raw_cpu_series.npz")

    # Step 7: Save y_true, y_naive, pred_ensemble as flat arrays (for convenience)
    y_true  = test_fe["cpu_target"].values.astype(np.float32)
    y_naive = test_fe["naive_cpu"].values.astype(np.float32)
    np.save(os.path.join(hz_out, "y_true.npy"), y_true)
    np.save(os.path.join(hz_out, "y_naive.npy"), y_naive)
    if saved_ens is not None:
        np.save(os.path.join(hz_out, "pred_hetero_ensemble.npy"), saved_ens)
    print(f"  Saved y_true.npy, y_naive.npy")

    # Step 8: Print per-container stats
    lengths = [e - s for s, e in boundaries]
    print(f"\n  Per-container series length:")
    print(f"    min={min(lengths)}, median={np.median(lengths):.0f}, "
          f"max={max(lengths)}, mean={np.mean(lengths):.0f}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-container test arrays for foundation model inference")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with train.parquet, val.parquet, test.parquet")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Pipeline checkpoint directory (contains 10min/, 30min/, etc.)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to save per-container arrays")
    parser.add_argument("--horizons", nargs="+", default=list(HORIZONS.keys()),
                        choices=list(HORIZONS.keys()),
                        help="Which horizons to process (default: all 4)")
    args = parser.parse_args()

    # Load test data
    test_path = os.path.join(args.data_dir, "test.parquet")
    if not os.path.exists(test_path):
        test_path = os.path.join(args.data_dir, "test.csv")
    if not os.path.exists(test_path):
        print(f"ERROR: test data not found in {args.data_dir}")
        sys.exit(1)

    print(f"Loading test data from {test_path}...")
    if test_path.endswith(".parquet"):
        test_df = pd.read_parquet(test_path)
    else:
        test_df = pd.read_csv(test_path)

    test_df.sort_values(["container_id", "time_stamp"], inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    print(f"  {len(test_df):,} rows, {test_df['container_id'].nunique()} containers")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each horizon
    results = {}
    for hz in args.horizons:
        ok = extract_horizon(test_df, hz, args.checkpoint_dir, args.output_dir)
        results[hz] = ok

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for hz, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {hz}: {status}")
    print(f"\nOutput: {args.output_dir}")
    print(f"\nFor foundation model inference, use raw_cpu_series.npz:")
    print(f"  data = np.load('output_dir/10min/raw_cpu_series.npz')")
    print(f"  for cid in data.files:")
    print(f"      series = data[cid]  # 1D numpy array of CPU values")


if __name__ == "__main__":
    main()
