"""
day2_timesfm_sanity.py — TimesFM-2.5 zero-shot on Alibaba/Bitbrains/ByteDance.

Drop-in replacement for day1_chronos2_sanity.py that calls TimesFM 2.5 instead
of Chronos-2. Produces JSONs with the same schema so same_sample_*.py and the
merger pick them up with only a path change.

Origin-picking is VERBATIM from day1_chronos2_sanity.py. Do not edit without
updating that script too; the same-sample guarantee depends on exact match.

Baselines from same_sample_leaderboard_full.csv (Apr 24):
    Alibaba   ensemble_sub: 0.9224 0.8375 0.8029 0.7584    (K=20)
    Bitbrains (K=50 canonical, from k50_sensitivity_summary.csv):
              naive_sub:    0.8115 0.4769 0.1957 0.0914
              ensemble:     N/A (per-VM XGB, not subsampleable)
    ByteDance ensemble_sub: 0.8548 0.8055 0.7406 0.7668    (K=50)

Run on Vast.ai RTX 4090. Expected per-dataset runtime:
    Alibaba K=20   ~45 min (4920 containers × 20 origins × 4 horizons)
    Bitbrains K=50 ~30 min (156 VMs × 50 origins × 4 horizons)
    ByteDance K=50 ~15 min (93 containers × 50 origins × 4 horizons)
Plus ~5 min one-time weight download on first call.
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
# Config (override DATA_ROOT via env)
# =============================================================================

DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/workspace/k8s-ensemble-forecast/data/alibaba",
))

TEST_FILE_CANDIDATES = ["test.parquet", "test.csv"]

NUM_ORIGINS = 20         # default; overridden per-dataset or via --num_origins
MIN_CONTEXT = 24
MAX_CONTEXT = 1024       # TimesFM 2.5 max_context per HF docs
BATCH_SIZE = 512         # safe on 24 GB 4090, fp32

QUANTILE_LEVELS = [0.1, 0.5, 0.9]  # informational only

ID_COL     = os.environ.get("ID_COL",     "container_id")
TIME_COL   = os.environ.get("TIME_COL",   "time_stamp")
TARGET_COL = os.environ.get("TARGET_COL", "cpu_util_percent")


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
            "10min":  {"naive_sub": 0.9201, "ens_sub": 0.9224},
            "30min":  {"naive_sub": 0.8324, "ens_sub": 0.8375},
            "60min":  {"naive_sub": 0.7879, "ens_sub": 0.8029},
            "120min": {"naive_sub": 0.7085, "ens_sub": 0.7584},
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
        # K=50 naive_sub (canonical as of Apr 24). K=20 values deprecated.
        "baselines": {
            "10min":  {"naive_sub":  0.8115, "ens_sub": None},
            "30min":  {"naive_sub":  0.4769, "ens_sub": None},
            "60min":  {"naive_sub":  0.1957, "ens_sub": None},
            "120min": {"naive_sub":  0.0914, "ens_sub": None},
        },
        "ens_label": "N/A (per-VM XGB)",
    },
    "bytedance": {
        "cadence_min": 10,
        "horizons": [
            ("10min",  1),
            ("30min",  3),
            ("60min",  6),
            ("120min", 12),
        ],
        "baselines": {
            "10min":  {"naive_sub": 0.7829, "ens_sub": 0.8548},
            "30min":  {"naive_sub": 0.7529, "ens_sub": 0.8055},
            "60min":  {"naive_sub": 0.6721, "ens_sub": 0.7406},
            "120min": {"naive_sub": 0.6523, "ens_sub": 0.7668},
        },
        "ens_label": "NNLS Ensemble",
    },
}


# =============================================================================
# small helpers
# =============================================================================

def log(msg):
    print(msg, flush=True)


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def pick_origins(series_length, h_steps, k, min_context):
    """VERBATIM from day1_chronos2_sanity.py. Do not edit."""
    lo = min_context
    hi = series_length - h_steps - 1
    if hi < lo:
        return []
    if k == 1:
        return [int(round((lo + hi) / 2))]
    origins = np.linspace(lo, hi, k).round().astype(int).tolist()
    seen = set()
    out = []
    for o in origins:
        if o not in seen:
            seen.add(o)
            out.append(int(o))
    return out


def find_test_file(data_root):
    for name in TEST_FILE_CANDIDATES:
        p = data_root / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no test file in {data_root}. Tried: {TEST_FILE_CANDIDATES}")


def load_series_by_id(data_root):
    test_path = find_test_file(data_root)
    log(f"loading {test_path} ...")
    if str(test_path).endswith(".parquet"):
        df = pd.read_parquet(test_path)
    else:
        df = pd.read_csv(test_path)
    log(f"  rows: {len(df):,}   columns: {list(df.columns)}")

    for c in [ID_COL, TIME_COL, TARGET_COL]:
        if c not in df.columns:
            raise KeyError(
                f"column '{c}' not in {test_path}. "
                f"Columns present: {list(df.columns)}. "
                f"Override via env: ID_COL, TIME_COL, TARGET_COL."
            )

    df = df.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    series_by_id = {}
    for cid, g in df.groupby(ID_COL, sort=False):
        series_by_id[str(cid)] = g[TARGET_COL].to_numpy(dtype=np.float64)
    log(f"  containers: {len(series_by_id):,}")
    lens = [len(s) for s in series_by_id.values()]
    log(f"  series length: min={min(lens)}  median={int(np.median(lens))}  max={max(lens)}")
    return series_by_id


# =============================================================================
# TimesFM model wrapper
# =============================================================================

def load_timesfm(device="cuda"):
    """TimesFM 2.5 loader using manual safetensors bypass.

    Both from_pretrained() and load_checkpoint() are broken in the installed
    timesfm git HEAD as of Apr 24 2026. Workaround: download safetensors
    via hf_hub_download, load via load_state_dict. Verified clean.
    """
    import torch
    import timesfm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    log(f"loading TimesFM 2.5 200M on {device} (manual bypass)...")
    torch.set_float32_matmul_precision("high")

    log("  downloading/locating checkpoint...")
    ckpt_path = hf_hub_download(
        repo_id="google/timesfm-2.5-200m-pytorch",
        filename="model.safetensors",
    )

    log("  instantiating model...")
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)

    log("  loading state_dict...")
    state_dict = load_file(ckpt_path)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log(f"  WARN: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        log("  state_dict loaded cleanly")

    log(f"  moving to {device}...")
    model.model.to(device)

    log("  compiling ForecastConfig...")
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    log("  TimesFM loaded")
    return model

def forecast_batch(model, context_list, h_steps):
    """Run model.forecast on a batch of contexts.

    TimesFM 2.5 forecast output:
      point_forecast   shape (N, H)       — mean forecast
      quantile_forecast shape (N, H, 10)  — docs say: mean, then q10..q90
                                            We verify quantile ordering
                                            empirically after the run.

    Returns list of dicts, one per context, for the H-step-ahead prediction.
    """
    point_forecast, quantile_forecast = model.forecast(
        horizon=h_steps,
        inputs=context_list,
    )

    n = len(context_list)
    assert point_forecast.shape == (n, h_steps), (
        f"point_forecast shape {point_forecast.shape}, expected ({n}, {h_steps})")
    assert quantile_forecast.shape == (n, h_steps, 10), (
        f"quantile_forecast shape {quantile_forecast.shape}, "
        f"expected ({n}, {h_steps}, 10)")

    step = h_steps - 1   # last step is the H-ahead target
    out = []
    for i in range(n):
        out.append({
            "pred": float(point_forecast[i, step]),
            # HF docs convention: [mean, q10, q20, q30, q40, q50, q60, q70, q80, q90]
            "p10":  float(quantile_forecast[i, step, 1]),
            "p50":  float(quantile_forecast[i, step, 5]),
            "p90":  float(quantile_forecast[i, step, 9]),
        })
    return out


# =============================================================================
# Per-horizon evaluation
# =============================================================================

def run_one_horizon(
    model,
    series_by_id,
    horizon_label,
    h_steps,
    num_origins,
    batch_size,
    cadence_min,
):
    log("")
    log(f"=== {horizon_label}  (h_steps={h_steps}) ===")
    log(f"  cadence_min: {cadence_min}")
    log(f"  num_origins: {num_origins}")

    origin_lists = {}
    skipped_short = 0
    for cid, series in series_by_id.items():
        origins = pick_origins(len(series), h_steps, num_origins, MIN_CONTEXT)
        if len(origins) == 0:
            skipped_short += 1
            continue
        origin_lists[cid] = origins
    log(f"  {len(origin_lists):,} containers usable, "
        f"{skipped_short:,} skipped (too short)")
    if len(origin_lists) == 0:
        return {"error": "no usable containers for this horizon"}

    max_k = max(len(origins) for origins in origin_lists.values())

    y_true_all = []
    y_pred_mean_all = []
    y_pred_p50_all = []
    y_pred_p10_all = []
    y_pred_p90_all = []
    y_pred_naive_all = []
    container_ids_all = []
    origins_all = []

    for k in range(max_k):
        this_round = []
        for cid, origins in origin_lists.items():
            if k < len(origins):
                this_round.append((cid, origins[k]))
        if not this_round:
            continue

        t0 = time.time()

        round_preds = {}
        for start in range(0, len(this_round), batch_size):
            chunk = this_round[start : start + batch_size]
            contexts = []
            for cid, origin in chunk:
                series = series_by_id[cid]
                ctx = series[:origin]
                if len(ctx) > MAX_CONTEXT:
                    ctx = ctx[-MAX_CONTEXT:]
                contexts.append(ctx)

            preds = forecast_batch(model, contexts, h_steps)
            for (cid, _origin), pred in zip(chunk, preds):
                round_preds[cid] = pred

        elapsed = time.time() - t0
        throughput = len(this_round) / elapsed if elapsed > 0 else float("inf")
        log(f"  origin {k+1}/{max_k}: {len(this_round):,} series in "
            f"{elapsed:.1f}s ({throughput:.0f} series/s)")

        for cid, origin in this_round:
            series = series_by_id[cid]
            y_true_all.append(float(series[origin + h_steps]))
            pred = round_preds[cid]
            y_pred_mean_all.append(pred["pred"])
            y_pred_p50_all.append(pred["p50"])
            y_pred_p10_all.append(pred["p10"])
            y_pred_p90_all.append(pred["p90"])
            y_pred_naive_all.append(float(series[origin]))
            container_ids_all.append(str(cid))
            origins_all.append(int(origin))

    # persist per-point for downstream DM tests / Chapter 5
    perpoint_path = f"timesfm_perpoint_{horizon_label}.npz"
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

    # quantile ordering sanity: p10 < p50 < p90 must hold for most points
    p10_arr = np.array(y_pred_p10_all)
    p50_arr = np.array(y_pred_p50_all)
    p90_arr = np.array(y_pred_p90_all)
    order_ok = float(np.mean((p10_arr < p50_arr) & (p50_arr < p90_arr)))
    log(f"  quantile ordering sanity: p10<p50<p90 at {order_ok*100:.1f}% of points")
    if order_ok < 0.9:
        log("  *** quantile indices likely WRONG — swap in forecast_batch() ***")

    inside = (
        (np.asarray(y_true_all) >= np.asarray(y_pred_p10_all))
        & (np.asarray(y_true_all) <= np.asarray(y_pred_p90_all))
    )

    return {
        "n_points": len(y_true_all),
        "r2_mean":  compute_r2(y_true_all, y_pred_mean_all),
        "r2_p50":   compute_r2(y_true_all, y_pred_p50_all),
        "mae_mean": compute_mae(y_true_all, y_pred_mean_all),
        "mae_p50":  compute_mae(y_true_all, y_pred_p50_all),
        "r2_naive_subsample":  compute_r2(y_true_all, y_pred_naive_all),
        "mae_naive_subsample": compute_mae(y_true_all, y_pred_naive_all),
        "mean_p10": float(np.mean(y_pred_p10_all)),
        "mean_p90": float(np.mean(y_pred_p90_all)),
        "pct_inside_p10_p90": float(np.mean(inside)),
        "quantile_order_ok_pct": order_ok,
    }


# =============================================================================
# Verdict printing (parallels day1 script)
# =============================================================================

def print_verdict(horizon_label, results, baselines, ens_label):
    if "error" in results:
        print(f"\n=== {horizon_label} ===\n  ERROR: {results['error']}\n")
        return

    naive_sub = results["r2_naive_subsample"]
    r2        = results["r2_mean"]

    b = baselines[horizon_label]
    b_naive_sub = b["naive_sub"]
    b_ens_sub   = b["ens_sub"]

    print(f"\n--- {horizon_label} ---")
    print(f"  n_points              = {results['n_points']:,}")
    print(f"  TimesFM R2 (mean)     = {r2:.4f}")
    print(f"  TimesFM R2 (p50)      = {results['r2_p50']:.4f}")
    print(f"  TimesFM MAE (mean)    = {results['mae_mean']:.4f}")
    print(f"  same-sample naive R2  = {naive_sub:.4f}  (this run)")
    print(f"                        = {b_naive_sub:.4f}  (Chronos run, sanity)")
    diff_naive = (naive_sub - b_naive_sub) * 100
    flag = "[OK]" if abs(diff_naive) < 0.1 else "[CHECK]"
    print(f"      diff = {diff_naive:+.3f} pp  {flag}")
    if abs(diff_naive) >= 0.1:
        print("      ** origins differ between TimesFM and Chronos — "
              "samples are NOT aligned **")

    print(f"  delta TimesFM vs naive_sub = {(r2 - naive_sub)*100:+.2f} pp")
    if b_ens_sub is not None:
        print(f"  delta TimesFM vs ensemble  = {(r2 - b_ens_sub)*100:+.2f} pp   "
              f"(ensemble_sub = {b_ens_sub:.4f}, label='{ens_label}')")
    else:
        print(f"  ensemble: {ens_label} (no comparison)")

    print(f"  p10/p90 coverage    = {results['pct_inside_p10_p90']*100:.1f}%   "
          f"(target 80%)")
    print(f"  quantile ordering   = {results['quantile_order_ok_pct']*100:.1f}% "
          "(should be >90%)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        default=None,
                        help="which dataset (defaults to auto from DATA_ROOT name)")
    parser.add_argument("--num_origins", type=int, default=None,
                        help="override K (default: 20 for Alibaba, 50 for BD/BB)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.dataset is not None:
        dataset_name = args.dataset
    else:
        name = DATA_ROOT.name.lower()
        if name in DATASETS:
            dataset_name = name
        else:
            raise SystemExit(
                f"Cannot infer dataset from DATA_ROOT={DATA_ROOT}. "
                f"Pass --dataset alibaba|bitbrains|bytedance explicitly."
            )

    cfg = DATASETS[dataset_name]

    if args.num_origins is not None:
        K = args.num_origins
    else:
        # K=50 for non-stationary datasets (BD, BB); K=20 for Alibaba.
        # Bitbrains canonical bumped from K=20 to K=50 on Apr 24 based on
        # dense-pool drift check: K=50 drifts <2pp, K=20 up to 17pp.
        K = 50 if dataset_name in ("bytedance", "bitbrains") else 20

    if args.output is None:
        out_path = f"timesfm_k{K}_{dataset_name}.json"
    else:
        out_path = args.output

    log("=" * 64)
    log(f"TimesFM-2.5 zero-shot on {dataset_name.upper()}  (K={K})")
    log("=" * 64)
    log(f"DATA_ROOT:   {DATA_ROOT}")
    log(f"horizons:    {[h[0] for h in cfg['horizons']]}")
    log(f"cadence_min: {cfg['cadence_min']}")
    log(f"out_path:    {out_path}")

    series_by_id = load_series_by_id(DATA_ROOT)
    model = load_timesfm(device=args.device)

    all_results = {}
    for horizon_label, h_steps in cfg["horizons"]:
        all_results[horizon_label] = run_one_horizon(
            model=model,
            series_by_id=series_by_id,
            horizon_label=horizon_label,
            h_steps=h_steps,
            num_origins=K,
            batch_size=args.batch_size,
            cadence_min=cfg["cadence_min"],
        )

    payload = {
        "model":              "timesfm-2.5-200m",
        "dataset":            dataset_name,
        "cadence_min":        cfg["cadence_min"],
        "num_origins":        K,
        "min_context":        MIN_CONTEXT,
        "max_context":        MAX_CONTEXT,
        "ens_label":          cfg["ens_label"],
        "timesfm_zero_shot":  all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"\nwrote {out_path}")

    log("\n" + "=" * 64)
    log("VERDICT")
    log("=" * 64)
    for horizon_label, _ in cfg["horizons"]:
        print_verdict(horizon_label, all_results[horizon_label],
                      cfg["baselines"], cfg["ens_label"])


if __name__ == "__main__":
    main()