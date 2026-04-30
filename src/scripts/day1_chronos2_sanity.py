"""
Day 1 sanity check: Chronos-2 zero-shot on Alibaba, all four horizons.

Goal: decide if the A+ plan is viable. If Chronos-2 beats the existing NNLS
ensemble at any horizon on a fair evaluation, we proceed. If it loses on
all four, we stop and re-plan.

Baselines to beat (from comparison_table.csv, Hetero Ens R2):
    10min   0.92129
    30min   0.84041
    60min   0.80110
    120min  0.76420

Naive persistence R2 on the FULL test set (floor, same CSV):
    10min   0.91878
    30min   0.83611
    60min   0.78780
    120min  0.71780

Evaluation scheme
-----------------
The ensemble was evaluated at every valid (container, timestamp) row in
the test set as an h-step-ahead prediction. We cannot trivially do the
same for Chronos-2 without running one model call per row, which would
take hours. Instead we pick K=10 origins per container, evenly spread
across the test period, and evaluate only the h-step-ahead prediction at
each origin. Sample size is roughly 4920 * 10 = ~49,200 points per horizon,
which is enough for a binary sanity check.

Because subsampling could make comparisons unfair, we ALSO compute naive
persistence R2 on the exact same subsample and report it alongside the
hardcoded baseline. The fair Chronos-2 vs naive comparison is the
subsample-vs-subsample one.

This is NOT the final leaderboard number. Day 2's job is larger K or full
per-row evaluation.

Run on Vast.ai RTX 4090. Expected runtime: ~15-30 min for all four horizons
including Chronos-2 weight download on first run (~500 MB).

Written plainly with intermediate variables and prints so any failure is
easy to localize.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration - edit paths if your Vast.ai layout differs
# =============================================================================

DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/workspace/k8s-ensemble-forecast/data/alibaba",
))

# Filenames we try in order. First found at top level of DATA_ROOT wins.
TEST_FILE_CANDIDATES = [
    "test.parquet",
    "test.csv",
]

# 10 origins per container. Lower via --num_origins for quick smoke tests.
NUM_ORIGINS = 10

# Require at least 24 points of history before the first origin. Chronos
# technically handles shorter but it is noisy.
MIN_CONTEXT = 24

# How many containers to put into a single predict_df call. Caps GPU memory.
# 256 at ~1600 context points each = ~400k target tokens per call; fine on a
# 24GB 4090 in bf16. Lower to 128 if OOM, raise to 512 if you have headroom.
BATCH_SIZE = 256

# Quantile levels from Chronos-2. Today we only score R2 on the point
# forecast (mean) and p50, but we persist p10/p90 for Day 2 cost-aware use.
QUANTILE_LEVELS = [0.1, 0.5, 0.9]

# Column names. Override via env if your file is different.
ID_COL     = os.environ.get("ID_COL",     "container_id")
TIME_COL   = os.environ.get("TIME_COL",   "time_stamp")
TARGET_COL = os.environ.get("TARGET_COL", "cpu_util_percent")


# Per-dataset configuration. Values verified against the project's own
# CSV baselines and against Jimmy's runner scripts:
#   alibaba:   sprint1_Main.py lines 86-87  (SAMPLING_INTERVAL=300, 5-min)
#              comparison_table.csv        (naive and hetero_ens R2)
#   bitbrains: bitbrains_xval.py line 64   (same HORIZONS = 5-min steps)
#              bitbrains_summary_corrected.csv (BB_Naive_R2, BB_ML_R2_median)
#   bytedance: bytedance_runner.py line 50 (HORIZONS = 10-min steps)
#              results_summary__Bytedance.csv (naive_R2, hetero_ensemble_R2)
#
# Baselines key "hetero_ens" is the project's reported comparison target.
# For bitbrains, the project's reported number is BB_ML_R2_median which is
# the per-VM median of the ML R2 (different aggregation than alibaba/bytedance).
# We report it as-is and label it accordingly in the verdict.
DATASETS = {
    "alibaba": {
        "cadence_min": 5,
        "horizons": [
            ("10min",  2),
            ("30min",  6),
            ("60min", 12),
            ("120min", 24),
        ],
        "baselines": {
            "10min":  {"naive": 0.91878, "hetero_ens": 0.92129},
            "30min":  {"naive": 0.83611, "hetero_ens": 0.84041},
            "60min":  {"naive": 0.78780, "hetero_ens": 0.80110},
            "120min": {"naive": 0.71780, "hetero_ens": 0.76420},
        },
        "ens_label": "NNLS Ensemble",
    },
    "bitbrains": {
        "cadence_min": 5,
        "horizons": [
            ("10min",  2),
            ("30min",  6),
            ("60min", 12),
            ("120min", 24),
        ],
        # bitbrains_summary_corrected.csv: ML here is per-VM median, not
        # aggregate ensemble. BCF predicts ML fails on bitbrains and this
        # is exactly what the project numbers show (negative delta at
        # short horizons, tiny positive at 120min).
        "baselines": {
            "10min":  {"naive": 0.8113, "hetero_ens": 0.6344},
            "30min":  {"naive": 0.5620, "hetero_ens": 0.3611},
            "60min":  {"naive": 0.3014, "hetero_ens": 0.1199},
            "120min": {"naive": 0.0306, "hetero_ens": 0.0090},
        },
        "ens_label": "ML (per-VM median)",
    },
    "bytedance": {
        "cadence_min": 10,   # !!! 10-min cadence, not 5-min
        "horizons": [
            ("10min",   1),  # 10 / 10 = 1 step
            ("30min",   3),  # 30 / 10 = 3
            ("60min",   6),  # 60 / 10 = 6
            ("120min", 12),  # 120 / 10 = 12
        ],
        "baselines": {
            "10min":  {"naive": 0.7852, "hetero_ens": 0.8549},
            "30min":  {"naive": 0.7462, "hetero_ens": 0.8014},
            "60min":  {"naive": 0.7019, "hetero_ens": 0.7629},
            "120min": {"naive": 0.6471, "hetero_ens": 0.7601},
        },
        "ens_label": "NNLS Ensemble",
    },
}


def detect_dataset(data_root_path):
    # Try to guess the dataset from the path. User can override via --dataset.
    p = str(data_root_path).lower()
    for name in DATASETS:
        if name in p:
            return name
    return None


# =============================================================================
# Small helpers
# =============================================================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_r2(y_true, y_pred):
    # Plain R2, same formula as sklearn.metrics.r2_score. Written out so an
    # examiner can see exactly what we compute.
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def compute_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


# =============================================================================
# Data loading
# =============================================================================

def find_test_file():
    # We deliberately refuse to fall back to DATA_ROOT/<horizon>/test.parquet
    # because those per-horizon splits may have been truncated differently
    # during the ensemble's data prep. Using the 10min file to evaluate the
    # 120min horizon would silently corrupt results. One raw file shared
    # across all four horizons is what we need; the context/target slicing
    # happens later in evaluate_horizon.

    # Escape hatch if the user knows exactly which file to use.
    forced = os.environ.get("FORCE_TEST_FILE")
    if forced:
        path = Path(forced)
        if not path.exists():
            raise FileNotFoundError(
                f"FORCE_TEST_FILE={forced} does not exist"
            )
        return path

    for filename in TEST_FILE_CANDIDATES:
        top_level = DATA_ROOT / filename
        if top_level.exists():
            return top_level

    # Only per-horizon files found - refuse and explain.
    # Collect the union of horizon labels across all dataset configs so we
    # can surface any per-horizon files the user may have staged, regardless
    # of which dataset they were targeting.
    per_horizon_hits = []
    all_horizon_labels = {
        label for cfg in DATASETS.values() for label, _ in cfg["horizons"]
    }
    for horizon_label in all_horizon_labels:
        for filename in TEST_FILE_CANDIDATES:
            candidate = DATA_ROOT / horizon_label / filename
            if candidate.exists():
                per_horizon_hits.append(str(candidate))

    message = (
        f"No raw test.parquet or test.csv found directly under {DATA_ROOT}. "
        f"We need ONE file containing untransformed per-container series.\n"
    )
    if per_horizon_hits:
        message += (
            "\nFound per-horizon files but refusing to auto-pick one "
            "because they may be truncated differently per horizon:\n"
        )
        for hit in per_horizon_hits:
            message += f"  {hit}\n"
        message += (
            f"\nOptions:\n"
            f"  (a) copy the raw test file to {DATA_ROOT}/test.parquet\n"
            f"  (b) set DATA_ROOT to a directory containing it\n"
            f"  (c) if you are sure one of the above paths IS the raw "
            f"untruncated data, set FORCE_TEST_FILE=/full/path/to/file\n"
        )
    raise FileNotFoundError(message)


def load_test_long(path):
    # Load test data as a long-format DataFrame with the three columns we
    # need: container_id, time_stamp, cpu_util_percent.
    log(f"loading {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    needed = [ID_COL, TIME_COL, TARGET_COL]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}. "
            f"Available: {list(df.columns)}. "
            f"Override via ID_COL/TIME_COL/TARGET_COL env vars if needed."
        )

    df = df[needed].copy()
    df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    log(
        f"loaded {len(df):,} rows, "
        f"{df[ID_COL].nunique():,} unique containers"
    )
    return df


def build_per_container_series(df):
    # Group the flat DataFrame into {container_id: numpy array of target}.
    # Easier than pandas groupby for the context/target slicing we do later.
    series_by_id = {}
    for cid, sub in df.groupby(ID_COL, sort=False):
        series_by_id[cid] = sub[TARGET_COL].to_numpy(dtype=np.float64)
    return series_by_id


# =============================================================================
# Origin selection
# =============================================================================

def pick_origins(series_length, h_steps, k, min_context):
    # Origin = index of LAST observed point Chronos sees as context.
    # Target = series[origin + h_steps]. We need:
    #   origin + h_steps < series_length   (target in range)
    #   origin + 1       >= min_context    (enough history)
    lo = min_context - 1
    hi = series_length - h_steps - 1
    if hi < lo:
        return []
    if hi == lo:
        return [lo]
    origins = np.linspace(lo, hi, num=k).round().astype(int)
    origins = np.unique(origins)
    return origins.tolist()


# =============================================================================
# Chronos-2 runner
# =============================================================================

def load_chronos_pipeline():
    # Import and load the Chronos-2 pipeline. Fails loudly if missing.
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        log("FATAL: chronos-forecasting not installed. Run:")
        log("  pip install 'chronos-forecasting>=2.1.0' 'pandas[pyarrow]'")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log(f"loading Chronos-2 pipeline on {device} with dtype={dtype}")

    # bf16 on GPU roughly halves VRAM and is noticeably faster with
    # negligible accuracy cost for forecasting.
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=dtype,
    )
    return pipeline


def _build_context_df(series_by_id, origins_by_id, cadence_min):
    # Build one long-format DataFrame where each container contributes
    # rows [0..origin] of its own series. Returned with synthetic datetime
    # stamps spaced cadence_min minutes apart. Absolute calendar time does
    # not matter for Chronos-2 zero-shot, but RELATIVE spacing does: it
    # influences Chronos-2's inferred frequency and seasonal patterns.
    # Feeding it 5-min spacing when the real series is 10-min cadence
    # would make the model misidentify diurnal cycles.
    id_parts = []
    ts_parts = []
    tg_parts = []
    for cid, origin in origins_by_id.items():
        context = series_by_id[cid][: origin + 1]
        n = len(context)
        id_parts.append(np.full(n, cid, dtype=object))
        ts_parts.append(np.arange(n, dtype=np.int64))
        tg_parts.append(context)

    context_df = pd.DataFrame({
        "id":        np.concatenate(id_parts),
        "timestamp": np.concatenate(ts_parts),
        "target":    np.concatenate(tg_parts),
    })
    context_df["timestamp"] = pd.to_datetime(
        context_df["timestamp"] * cadence_min * 60,
        unit="s",
        origin="2020-01-01",
    )
    return context_df


def _extract_last_step(pred_df):
    # pred_df has one row per (id, future timestamp). We want the LAST of
    # the prediction_length rows per container = h-step-ahead.
    pred_df = pred_df.sort_values(["id", "timestamp"])
    last_step = pred_df.groupby("id").tail(1).set_index("id")

    # Point-forecast column name varies across chronos versions. Try in
    # order: "predictions" (current), "mean", "0.5".
    point_col = None
    for candidate in ("predictions", "mean", "0.5"):
        if candidate in last_step.columns:
            point_col = candidate
            break
    if point_col is None:
        raise RuntimeError(
            f"Could not find point-forecast column. "
            f"predict_df returned columns: {list(last_step.columns)}"
        )
    return last_step, point_col


def run_origin_batch_chunked(pipeline, series_by_id, origins_by_id,
                             h_steps, batch_size, cadence_min):
    # Split origins_by_id into chunks of batch_size containers, call
    # predict_df per chunk, merge. Prevents OOM at Alibaba scale.
    all_items = list(origins_by_id.items())
    result = {}

    num_batches = (len(all_items) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        batch_items = all_items[start : start + batch_size]
        batch_origins = dict(batch_items)

        context_df = _build_context_df(series_by_id, batch_origins, cadence_min)
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=h_steps,
            quantile_levels=QUANTILE_LEVELS,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        last_step, point_col = _extract_last_step(pred_df)

        has_p10 = "0.1" in last_step.columns
        has_p50 = "0.5" in last_step.columns
        has_p90 = "0.9" in last_step.columns
        for cid, row in last_step.iterrows():
            result[cid] = {
                "pred": float(row[point_col]),
                "p10": float(row["0.1"]) if has_p10 else float("nan"),
                "p50": float(row["0.5"]) if has_p50 else float("nan"),
                "p90": float(row["0.9"]) if has_p90 else float("nan"),
            }
    return result


# =============================================================================
# Per-horizon evaluation
# =============================================================================

def evaluate_horizon(pipeline, series_by_id, horizon_label, h_steps,
                     num_origins, batch_size, cadence_min):
    log(f"=== {horizon_label} (h_steps={h_steps}, cadence={cadence_min}min) ===")

    # Pick origins per container. Skip containers too short for this horizon.
    origin_lists = {}
    skipped_short = 0
    for cid, series in series_by_id.items():
        origins = pick_origins(len(series), h_steps, num_origins, MIN_CONTEXT)
        if len(origins) == 0:
            skipped_short += 1
            continue
        origin_lists[cid] = origins
    log(
        f"  {len(origin_lists):,} containers usable, "
        f"{skipped_short:,} skipped (too short)"
    )
    if len(origin_lists) == 0:
        return {"error": "no usable containers for this horizon"}

    # One origin-index at a time; for each, batch containers with batch_size.
    max_k = max(len(origins) for origins in origin_lists.values())

    y_true_all = []
    y_pred_mean_all = []
    y_pred_p50_all = []
    y_pred_p10_all = []
    y_pred_p90_all = []
    y_pred_naive_all = []   # same-subsample naive baseline, reviewer fix #1
    # Track (cid, origin) alongside each appended point so we can pair
    # Chronos-2 predictions against pred_hetero_ensemble.npy for DM testing.
    container_ids_all = []
    origins_all = []

    for k in range(max_k):
        batch = {}
        for cid, origins in origin_lists.items():
            if k < len(origins):
                batch[cid] = origins[k]
        if not batch:
            continue

        t0 = time.time()
        preds = run_origin_batch_chunked(
            pipeline, series_by_id, batch, h_steps, batch_size, cadence_min,
        )
        elapsed = time.time() - t0
        throughput = len(batch) / elapsed if elapsed > 0 else float("inf")
        log(
            f"  origin {k+1}/{max_k}: {len(batch):,} series in "
            f"{elapsed:.1f}s ({throughput:.0f} series/s)"
        )

        for cid, origin in batch.items():
            series = series_by_id[cid]
            y_true_all.append(float(series[origin + h_steps]))
            y_pred_mean_all.append(preds[cid]["pred"])
            y_pred_p50_all.append(preds[cid]["p50"])
            y_pred_p10_all.append(preds[cid]["p10"])
            y_pred_p90_all.append(preds[cid]["p90"])
            # Naive persistence: repeat the last observed value at origin.
            y_pred_naive_all.append(float(series[origin]))
            container_ids_all.append(str(cid))
            origins_all.append(int(origin))

    # Persist per-point arrays for downstream Diebold-Mariano testing
    # against the NNLS ensemble. Shapes must all equal len(y_true_all).
    perpoint_path = f"chronos2_perpoint_{horizon_label}.npz"
    np.savez(
        perpoint_path,
        y_true=np.array(y_true_all, dtype=np.float32),
        y_pred_mean=np.array(y_pred_mean_all, dtype=np.float32),
        y_pred_p50=np.array(y_pred_p50_all, dtype=np.float32),
        y_pred_p10=np.array(y_pred_p10_all, dtype=np.float32),
        y_pred_p90=np.array(y_pred_p90_all, dtype=np.float32),
        y_naive=np.array(y_pred_naive_all, dtype=np.float32),
        container_ids=np.array(container_ids_all, dtype=object),
        origins=np.array(origins_all, dtype=np.int64),
        horizon_label=horizon_label,
        h_steps=h_steps,
    )
    log(f"  saved per-point arrays to {perpoint_path} ({len(y_true_all):,} points)")

    return {
        "n_points": len(y_true_all),
        "r2_mean":  compute_r2(y_true_all, y_pred_mean_all),
        "r2_p50":   compute_r2(y_true_all, y_pred_p50_all),
        "mae_mean": compute_mae(y_true_all, y_pred_mean_all),
        "mae_p50":  compute_mae(y_true_all, y_pred_p50_all),
        # Same-subsample naive for fair comparison (fix #1).
        "r2_naive_subsample":  compute_r2(y_true_all, y_pred_naive_all),
        "mae_naive_subsample": compute_mae(y_true_all, y_pred_naive_all),
        # Quantile coverage statistics for Day 2. Not scored today.
        "mean_p10": float(np.mean(y_pred_p10_all)),
        "mean_p90": float(np.mean(y_pred_p90_all)),
        "pct_inside_p10_p90": float(np.mean(
            (np.asarray(y_true_all) >= np.asarray(y_pred_p10_all)) &
            (np.asarray(y_true_all) <= np.asarray(y_pred_p90_all))
        )),
    }


def print_verdict(horizon_label, results, baselines, ens_label):
    if "error" in results:
        print(f"\n=== {horizon_label} ===\n  ERROR: {results['error']}\n")
        return

    naive_full = baselines[horizon_label]["naive"]
    naive_sub  = results["r2_naive_subsample"]
    ens        = baselines[horizon_label]["hetero_ens"]
    r2         = results["r2_mean"]

    # Right-pad the ensemble label so the number column stays aligned.
    ens_line_label = f"  {ens_label} R2".ljust(27)

    print()
    print(f"=== {horizon_label} ===")
    print(f"  Naive R2 (full test set)   {naive_full:.4f}")
    print(f"  Naive R2 (same subsample)  {naive_sub:.4f}  <-- fair baseline")
    print(f"{ens_line_label}{ens:.4f}")
    print(f"  Chronos-2 R2 (mean)        {r2:.4f}")
    print(f"  Chronos-2 R2 (p50)         {results['r2_p50']:.4f}")
    print(f"  Chronos-2 MAE              {results['mae_mean']:.4f}")
    print(f"  p10-p90 coverage           {results['pct_inside_p10_p90']*100:.1f}%")
    print(f"  n points evaluated         {results['n_points']:,}")

    if r2 > ens:
        print(f"  VERDICT: beats {ens_label} by +{(r2 - ens) * 100:.2f}pp")
    elif r2 > naive_sub:
        gap_ens = (ens - r2) * 100
        gap_nv  = (r2 - naive_sub) * 100
        print(
            f"  VERDICT: beats subsample-naive by +{gap_nv:.2f}pp, "
            f"loses to {ens_label} by {gap_ens:.2f}pp"
        )
    else:
        print(
            f"  VERDICT: loses to subsample-naive by "
            f"{(naive_sub - r2) * 100:.2f}pp - investigate"
        )
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS.keys()), default=None,
        help="Which dataset config to use. If omitted, detected from "
             "DATA_ROOT path (looks for 'alibaba'/'bitbrains'/'bytedance').",
    )
    parser.add_argument(
        "--horizons", nargs="+", default=None,
        help="Subset of horizons to run. Default: all four for the dataset.",
    )
    parser.add_argument(
        "--output", default="chronos2_zero_shot_results.json",
        help="Where to write the final results JSON.",
    )
    parser.add_argument(
        "--num_origins", type=int, default=NUM_ORIGINS,
        help=f"Origins per container. Default {NUM_ORIGINS}.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Containers per predict_df call. Default {BATCH_SIZE}. "
             f"Lower to 128 if OOM on 4090, raise to 512 with more VRAM.",
    )
    args = parser.parse_args()

    # Resolve dataset config. Explicit --dataset wins over auto-detection.
    dataset_name = args.dataset or detect_dataset(DATA_ROOT)
    if dataset_name is None:
        log(
            f"FATAL: Could not detect dataset from DATA_ROOT={DATA_ROOT}. "
            f"Pass --dataset {{{'/'.join(sorted(DATASETS.keys()))}}} explicitly."
        )
        sys.exit(1)

    cfg = DATASETS[dataset_name]
    cadence_min = cfg["cadence_min"]
    horizons    = cfg["horizons"]
    baselines   = cfg["baselines"]
    ens_label   = cfg["ens_label"]

    # Horizon filter: default = all for this dataset.
    if args.horizons is None:
        selected_horizons = [h for h, _ in horizons]
    else:
        selected_horizons = args.horizons

    valid_labels = {h for h, _ in horizons}
    bad = [h for h in selected_horizons if h not in valid_labels]
    if bad:
        log(f"FATAL: unknown horizons {bad}. Valid: {sorted(valid_labels)}")
        sys.exit(1)

    log(f"Day 1 sanity check: Chronos-2 zero-shot on {dataset_name}")
    log(f"DATA_ROOT    = {DATA_ROOT}")
    log(f"dataset      = {dataset_name}  (cadence={cadence_min}min)")
    log(f"Python       = {sys.version.split()[0]}")
    log(f"num_origins  = {args.num_origins}")
    log(f"batch_size   = {args.batch_size}")

    # Load data once - same raw series used for all horizons.
    test_path = find_test_file()
    df = load_test_long(test_path)
    series_by_id = build_per_container_series(df)
    log(f"built per-container series dict with {len(series_by_id):,} entries")

    # Quick eyeball sanity on the first container.
    first_cid = next(iter(series_by_id))
    first_series = series_by_id[first_cid]
    log(
        f"sample container {first_cid}: length={len(first_series)}, "
        f"first 5 values={first_series[:5]}"
    )

    # Load Chronos-2 once. Weight download happens on first run (~500 MB).
    pipeline = load_chronos_pipeline()

    all_results = {}
    for horizon_label, h_steps in horizons:
        if horizon_label not in selected_horizons:
            continue
        try:
            results = evaluate_horizon(
                pipeline, series_by_id, horizon_label, h_steps,
                args.num_origins, args.batch_size, cadence_min,
            )
            all_results[horizon_label] = results
            print_verdict(horizon_label, results, baselines, ens_label)
        except Exception as err:
            log(f"ERROR on {horizon_label}: {type(err).__name__}: {err}")
            import traceback
            traceback.print_exc()
            all_results[horizon_label] = {"error": str(err)}

    # Final summary table.
    print()
    print("=" * 78)
    print(f"FINAL SUMMARY  dataset={dataset_name}  cadence={cadence_min}min")
    print("=" * 78)
    ens_col_header = ens_label if len(ens_label) <= 14 else ens_label[:14]
    print(
        f"{'Horizon':<8} {'Naive(full)':>12} {'Naive(sub)':>11} "
        f"{ens_col_header:>14} {'Chronos-2':>10} {'Verdict':>17}"
    )
    for horizon_label, _ in horizons:
        if horizon_label not in all_results:
            continue
        r = all_results[horizon_label]
        if "error" in r:
            print(
                f"{horizon_label:<8} {'-':>12} {'-':>11} "
                f"{'-':>14} {'ERROR':>10} {'':>17}"
            )
            continue
        naive_full = baselines[horizon_label]["naive"]
        naive_sub  = r["r2_naive_subsample"]
        ens        = baselines[horizon_label]["hetero_ens"]
        c2         = r["r2_mean"]
        if c2 > ens:
            verdict = f"beats {ens_label[:8]}"
        elif c2 > naive_sub:
            verdict = "beats sub-naive"
        else:
            verdict = "loses to naive"
        print(
            f"{horizon_label:<8} {naive_full:>12.4f} {naive_sub:>11.4f} "
            f"{ens:>14.4f} {c2:>10.4f} {verdict:>17}"
        )

    # Persist results for Day 2.
    out = {
        "dataset": dataset_name,
        "cadence_min": cadence_min,
        "baselines": baselines,
        "ens_label": ens_label,
        "chronos2_zero_shot": all_results,
        "num_origins": args.num_origins,
        "batch_size": args.batch_size,
        "min_context": MIN_CONTEXT,
        "data_root": str(DATA_ROOT),
        "test_file": str(test_path),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote results to {args.output}")


if __name__ == "__main__":
    main()