"""
f4_evaluation_harness.py — run F4 closed-loop simulations for v6 vs v7 comparison.

For each (dataset, pipeline, seed) in:
    datasets   = {alibaba, bitbrains, bytedance}
    pipelines  = {v6 (uniform), v7 (per-dataset)}
    seeds      = {42, 1729, 7331}

run the chance-constrained MPC autoscaler against the forecast quantile parquet
and save per-step trajectories. Output: phase_f/data/f4_trajectories.parquet

The bootstrap analysis is done by a separate script (f4_pareto_analysis.py) on
the saved trajectories. This script handles only data generation.

Pre-registration (f4_evaluation_plan.md) is committed to git BEFORE this script
runs. The metrics, datasets, seeds, and decision rule are fixed in that document.

Usage:
    python phase_f/scripts/f4_evaluation_harness.py \
        --v6-parquet phase_f/data/f3_lora_quantiles_h060_cad30_sorted.parquet \
        --v7-parquet phase_f/data/f3_quantiles_per_dataset_h060_cad30_sorted.parquet
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Local modules
from simulate_hpa_v3 import (
    simulate_hpa_with_desired_path,
    make_mpc_desired_path_fn,
    _import_v2,
)


PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "phase_f" / "data"

DATASETS = ["alibaba", "bitbrains", "bytedance"]
PIPELINES = ["v6", "v7"]
SEEDS = [42, 1729, 7331]
ALPHA = 0.95
MPC_MODE = "static_chance"      # use static for cheaper compute; cvxpy as ablation
CADENCE_MIN = 30
CPU_REQUEST = 0.5
MAX_REPLICAS = 1000
SAFETY_MARGIN = 1.0


# Dataset resolution (minutes per timestep)
DATASET_RES_MIN = {"alibaba": 5, "bitbrains": 5, "bytedance": 10}


def load_dataset_test(dataset):
    """Load only the test split of a dataset and return cpu series per container."""
    base = DATA_PROCESSED / dataset
    test_path = base / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} missing")
    df = pd.read_parquet(test_path)

    # Identify columns
    id_col = None
    for cand in ("container_id", "vm_id", "instance_id"):
        if cand in df.columns:
            id_col = cand
            break
    cpu_col = None
    for cand in ("cpu_util_percent", "cpu_target", "cpu", "cpu_percent"):
        if cand in df.columns:
            cpu_col = cand
            break
    if id_col is None or cpu_col is None:
        raise ValueError(f"cannot identify id/cpu in {df.columns.tolist()}")

    df = df.sort_values([id_col, "time_stamp"]).reset_index(drop=True)
    return df, id_col, cpu_col


def build_quantile_lookup_for_container(forecast_df, dataset, container_id, alpha):
    """
    Build a function (t -> q_alpha[1..H]) for a specific container, returning the
    nearest quantile path forecast issued at the most recent origin <= t.

    Forecast rows have columns: container_id, time_stamp (origin), step_idx (0..H-1),
    t_target, q_0.5..q_0.95, cpu_true.

    We pre-build a dict origin_ts -> {step_idx -> q_alpha}, sorted by origin_ts.

    The closest available quantile column to alpha is selected (e.g. for alpha=0.95
    we use q_0.95 directly).
    """
    sub = forecast_df[
        (forecast_df.dataset == dataset)
        & (forecast_df.container_id.astype(str) == str(container_id))
    ]
    if len(sub) == 0:
        return lambda t: None, None

    # Select the q column closest to alpha
    quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    q_chosen = min(quantile_levels, key=lambda q: abs(q - alpha))
    q_col = f"q_{q_chosen}"

    # Build origin -> array(H,) mapping
    origins = {}
    for origin_ts, g in sub.groupby("time_stamp"):
        g = g.sort_values("step_idx")
        origins[int(origin_ts)] = g[q_col].to_numpy(dtype=np.float32)

    origin_sorted = np.array(sorted(origins.keys()))

    def lookup(t_step):
        # t_step is a step index. The container's time_stamps are spaced by res_min*60 sec.
        # We need to know the time_stamp value at step index t_step; the caller passes step,
        # not seconds. So we need a step-to-timestamp map.
        # In this harness we use container-local step indices = 0, 1, 2, ...
        # Convert to original time_stamp via the container's first test timestamp + step*res_min*60.
        return None    # placeholder; replaced by caller closure
    return lookup, (origins, origin_sorted, q_col)


def run_single_simulation(dataset, container_id, container_cpu_series,
                          container_ts_array, forecast_table, alpha, seed,
                          res_min, max_replicas):
    """
    Run one container's HPA simulation under chance-constrained MPC.

    Returns dict with violation_rate, waste_rate, replica_trajectory etc.
    """
    HPAConfig, HPAState = _import_v2()

    cfg = HPAConfig()
    cfg.cpu_request = CPU_REQUEST
    cfg.max_replicas = max_replicas
    cfg.safety_margin = SAFETY_MARGIN

    # Find origins in the forecast table that match this container
    sub = forecast_table[
        (forecast_table.dataset == dataset)
        & (forecast_table.container_id.astype(str) == str(container_id))
    ]
    if len(sub) == 0:
        return None

    quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    q_chosen = min(quantile_levels, key=lambda q: abs(q - alpha))
    q_col = f"q_{q_chosen}"

    # Map origin time_stamp -> q-array(H,)
    origins = {}
    for origin_ts, g in sub.groupby("time_stamp"):
        g = g.sort_values("step_idx")
        origins[int(origin_ts)] = g[q_col].to_numpy(dtype=np.float32)

    if len(origins) == 0:
        return None

    # Convert container_ts_array (the actual timestamps) into a fast lookup:
    # step_idx -> ts. The simulation runs over the test CPU series with cadence_steps
    # equal to (CADENCE_MIN // res_min).
    cadence_steps = max(1, CADENCE_MIN // res_min)
    ts_to_step = {int(ts): i for i, ts in enumerate(container_ts_array)}

    # Precompute step -> q_array mapping using forecast origins
    step_to_q = {}
    for origin_ts, q_arr in origins.items():
        if int(origin_ts) in ts_to_step:
            step_idx = ts_to_step[int(origin_ts)]
            step_to_q[step_idx] = q_arr

    if len(step_to_q) == 0:
        return None

    sorted_decision_steps = sorted(step_to_q.keys())

    def quantile_lookup(t):
        """Return the q-path for the most recent origin <= t, or None if none."""
        # Find last decision_step <= t
        # Use binary search
        lo, hi = 0, len(sorted_decision_steps) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_decision_steps[mid] <= t:
                best = sorted_decision_steps[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            return None
        return step_to_q[best]

    desired_path_fn = make_mpc_desired_path_fn(
        quantile_lookup, alpha=alpha, mode=MPC_MODE,
        cpu_request=cfg.cpu_request, safety_margin=cfg.safety_margin,
        min_replicas=cfg.min_replicas, max_replicas=cfg.max_replicas,
    )

    np.random.seed(seed)
    result = simulate_hpa_with_desired_path(
        container_cpu_series, cfg, desired_path_fn,
        forecast_horizon=12, cadence_steps=cadence_steps,
    )
    # Drop large arrays to keep memory bounded; we only need scalar summaries
    # for the bootstrap analysis. The viol_mask is needed for block bootstrap.
    return {
        "dataset": dataset,
        "container_id": str(container_id),
        "violation_rate": result["violation_rate"],
        "waste_rate": result["waste_rate"],
        "violation_severity": result["violation_severity"],
        "replica_mean": float(result["replica_trajectory"].mean()),
        "replica_max":  int(result["replica_trajectory"].max()),
        "n_steps": len(container_cpu_series),
        "viol_mask": result["viol_mask"].astype(np.int8),
        "replica_traj": result["replica_trajectory"].astype(np.int32),
    }


def run_all(v6_parquet, v7_parquet, datasets, pipelines, seeds, max_containers=None,
            output_path=None, alpha=0.95):
    """Run the full 18-config evaluation grid."""
    print("=" * 70)
    print("F4 EVALUATION HARNESS")
    print("=" * 70)
    print(f"v6_parquet: {v6_parquet}")
    print(f"v7_parquet: {v7_parquet}")
    print(f"datasets:   {datasets}")
    print(f"pipelines:  {pipelines}")
    print(f"seeds:      {seeds}")
    print(f"alpha:      {alpha}")
    print(f"max_containers per dataset: {max_containers or 'ALL'}")

    print("\nLoading forecast parquets ...")
    forecast_v6 = pd.read_parquet(v6_parquet)
    forecast_v7 = pd.read_parquet(v7_parquet)
    forecast_v6["container_id"] = forecast_v6.container_id.astype(str)
    forecast_v7["container_id"] = forecast_v7.container_id.astype(str)
    print(f"  v6: {len(forecast_v6)} rows")
    print(f"  v7: {len(forecast_v7)} rows")

    all_results = []

    for dataset in datasets:
        print(f"\n{'='*60}\nDATASET: {dataset}\n{'='*60}")
        res_min = DATASET_RES_MIN[dataset]
        df, id_col, cpu_col = load_dataset_test(dataset)

        container_ids = df[id_col].unique()
        if max_containers is not None:
            container_ids = container_ids[:max_containers]
        print(f"  containers to evaluate: {len(container_ids)}")

        for pipeline in pipelines:
            forecast_table = forecast_v6 if pipeline == "v6" else forecast_v7
            for seed in seeds:
                print(f"\n  [{dataset}/{pipeline}/seed={seed}] running ...")
                t0 = time.time()

                container_results = []
                for i, cid in enumerate(container_ids):
                    cdf = df[df[id_col] == cid].sort_values("time_stamp")
                    cpu_series = cdf[cpu_col].to_numpy(dtype=np.float32)
                    ts_array  = cdf["time_stamp"].to_numpy(dtype=np.int64)
                    if len(cpu_series) < 24:
                        continue
                    r = run_single_simulation(
                        dataset, cid, cpu_series, ts_array,
                        forecast_table, alpha, seed, res_min, MAX_REPLICAS,
                    )
                    if r is None:
                        continue
                    container_results.append(r)
                    if (i + 1) % 200 == 0:
                        elapsed = time.time() - t0
                        print(f"     {i+1}/{len(container_ids)} containers, {elapsed:.0f}s")

                elapsed = time.time() - t0
                print(f"  done in {elapsed:.1f}s ({len(container_results)} containers had valid forecasts)")
                if not container_results:
                    print(f"  WARNING: no container had any forecasts for {dataset}/{pipeline}")
                    continue

                # Aggregate per-container summaries
                for r in container_results:
                    all_results.append({
                        "dataset": dataset,
                        "pipeline": pipeline,
                        "seed": seed,
                        "container_id": r["container_id"],
                        "violation_rate": r["violation_rate"],
                        "waste_rate": r["waste_rate"],
                        "violation_severity": r["violation_severity"],
                        "replica_mean": r["replica_mean"],
                        "replica_max": r["replica_max"],
                        "n_steps": r["n_steps"],
                    })

    out_df = pd.DataFrame(all_results)
    if output_path:
        out_df.to_parquet(output_path, index=False)
        print(f"\nWROTE: {output_path}")
        print(f"  shape: {out_df.shape}")
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6-parquet", type=str, required=True)
    ap.add_argument("--v7-parquet", type=str, required=True)
    ap.add_argument("--max-containers", type=int, default=None,
                    help="cap containers per dataset for smoke test; None=ALL")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--output", type=str,
                    default=str(OUT_DIR / "f4_trajectories.parquet"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_all(
        args.v6_parquet, args.v7_parquet, DATASETS, PIPELINES, SEEDS,
        max_containers=args.max_containers,
        output_path=args.output,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
