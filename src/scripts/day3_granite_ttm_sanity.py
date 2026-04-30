"""
day3_granite_ttm_sanity.py — Granite-TTM r2 zero-shot on Alibaba/Bitbrains/ByteDance.

Mirror of day2_timesfm_sanity.py, swapped to IBM's Tiny Time Mixer (Granite-TTM).
Origin picking is VERBATIM from day2_timesfm_sanity.py — NOT day1. Day1 uses a
different formula (lo = min_context-1, special case hi==lo, dedup via np.unique)
so day3 per-point arrays line up sample-for-sample with TimesFM (day2) but NOT
with Chronos-2 (day1). If you want same-sample DM tests against Chronos, you
need a separate alignment step at merge time.

What's different from day2:

  1. TTM has FIXED context_length and FIXED prediction_length. Default is the
     1024-96 variant ("1024-96-r2") to match TimesFM's MAX_CONTEXT=1024, but
     this is configurable via --ttm_revision (e.g. "512-96-r2" if you want a
     better fit for Alibaba's ~576-point series — see caveat below).
     Short series get front-padded with zeros + past_observed_mask=0 for
     those positions. Max h_steps in any dataset is 24 (Alibaba 120min),
     well under 96.

     Caveat on context length: for Alibaba (~576 points per container),
     1024-96 means ~44% of the input is zero-padding. TTM does scale with
     past_observed_mask but the patcher/mixer still attends to padded
     positions. Bitbrains (median 1713) and ByteDance (10-min cadence,
     longer windows) are mostly fine at 1024 — Bitbrains gets truncated to
     last 1024 instead. If TTM Alibaba R2 looks unreasonably bad, try
     --ttm_revision 512-96-r2.

  2. TTM r2 standard checkpoints DO NOT have a quantile head. They output
     point forecasts only. p10 / p50 / p90 are written as null in the JSON,
     pct_inside_p10_p90 and quantile_order_ok_pct are null too. The quantile
     sanity-check block is skipped when all values are NaN. Downstream
     mergers should treat these columns as N/A for TTM.

  3. TTM does its own RevIN scaling internally (config scaling="std").
     No manual normalization or per-series rescaling is needed.

  4. The forecast call is a normal model(past_values=..., past_observed_mask=...)
     under torch.no_grad() — there is no model.forecast() helper.

Baselines: naive_sub values come from the actual TimesFM run JSONs at the same
K (since pick_origins is identical between day2 and day3, the r2_naive_subsample
day3 produces will match TimesFM's exactly within float-rounding). The "Chronos
K=20" Alibaba values that day2 hardcoded are off by 0.5pp from what day2/day3
actually produce — that's why this script uses TimesFM-aligned values.
ens_sub values come from same_sample_leaderboard_full.csv (Apr 24).

Install on Colab (one of these in a setup cell):
    !pip install "transformers>=4.46"          # TTM is in main transformers
    # OR (official IBM library)
    !pip install granite-tsfm

Run on Colab Pro+. TTM is much smaller than TimesFM 200M (~5M params), so it
fits comfortably on any of the GPUs Colab assigns (T4 16GB, L4 22.5GB, A100
40GB). Per-dataset runtime is ~3-5x faster than the day2 TimesFM run because
of TTM's smaller parameter count, even on the slower Colab GPUs. Set
DATA_ROOT to wherever your test parquets live in Drive, e.g.:

    !DATA_ROOT=/content/drive/MyDrive/k8s-ensemble-forecast/data/alibaba \\
      python day3_granite_ttm_sanity.py --dataset alibaba
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Config (override DATA_ROOT via env)
# =============================================================================

# Default DATA_ROOT is Vast.ai-style. On Colab override via env var:
#   import os; os.environ["DATA_ROOT"] = "/content/drive/MyDrive/k8s-ensemble-forecast/data/alibaba"
# (or pass it on the command line shell).
DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    "/workspace/k8s-ensemble-forecast/data/alibaba",
))

TEST_FILE_CANDIDATES = ["test.parquet", "test.csv"]

NUM_ORIGINS = 20         # default; overridden per-dataset or via --num_origins
MIN_CONTEXT = 24
BATCH_SIZE = 512         # safe on Colab T4 16GB; TTM is tiny so it fits with
                         # plenty of room. Crank higher if you get an A100.
                         # 512 keeps parity with day2.

# TTM r2 default variant. Both context and prediction lengths are PARSED from
# the revision string at startup (e.g. "1024-96-r2" -> context=1024, pred=96).
# The user can override the revision via --ttm_revision.
TTM_REPO              = "ibm-granite/granite-timeseries-ttm-r2"
DEFAULT_TTM_REVISION  = "1024-96-r2"

# Filled in by main() after parsing the revision string.
TTM_CONTEXT_LENGTH    = None   # type: int
TTM_PREDICTION_LEN    = None   # type: int

ID_COL     = os.environ.get("ID_COL",     "container_id")
TIME_COL   = os.environ.get("TIME_COL",   "time_stamp")
TARGET_COL = os.environ.get("TARGET_COL", "cpu_util_percent")


# Hardcoded baselines.
#   - naive_sub: r2_naive_subsample expected from a same-K TimesFM run on this
#     dataset. day3's pick_origins is byte-identical to day2's, so day3's actual
#     r2_naive_subsample will match these values to ~4 decimals. NOT the same
#     as the Chronos K=20 r2_naive_subsample (those differ by 0.5–1.6pp on
#     Alibaba/Bitbrains because day1 has a different pick_origins formula).
#   - ens_sub: from same_sample_leaderboard_full.csv (Apr 24).
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
            # naive_sub from timesfm_k20_alibaba.json (verified Apr 28)
            "10min":  {"naive_sub": 0.9150, "ens_sub": 0.9224},
            "30min":  {"naive_sub": 0.8341, "ens_sub": 0.8375},
            "60min":  {"naive_sub": 0.7893, "ens_sub": 0.8029},
            "120min": {"naive_sub": 0.7083, "ens_sub": 0.7584},
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
        "baselines": {
            # naive_sub from timesfm_k50_bitbrains.json (verified Apr 28).
            # K=50 canonical as of Apr 24; K=20 values deprecated.
            "10min":  {"naive_sub":  0.8080, "ens_sub": None},
            "30min":  {"naive_sub":  0.4926, "ens_sub": None},
            "60min":  {"naive_sub":  0.1848, "ens_sub": None},
            "120min": {"naive_sub":  0.0981, "ens_sub": None},
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
            # naive_sub from timesfm_k50_bytedance.json (verified Apr 28)
            "10min":  {"naive_sub": 0.7829, "ens_sub": 0.8548},
            "30min":  {"naive_sub": 0.7529, "ens_sub": 0.8055},
            "60min":  {"naive_sub": 0.6721, "ens_sub": 0.7406},
            "120min": {"naive_sub": 0.6523, "ens_sub": 0.7668},
        },
        "ens_label": "NNLS Ensemble",
    },
}


# =============================================================================
# small helpers (duplicated from day2 — keep in sync)
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
    """VERBATIM from day2_timesfm_sanity.py.

    NOT verbatim from day1_chronos2_sanity.py — day1 differs in three places:
      * lo = min_context - 1 (vs min_context here)
      * special case is `if hi == lo: return [lo]` (vs `if k == 1: midpoint`)
      * dedup via np.unique (sorted) (vs seen-set, preserves linspace order)
    Same-sample alignment with TimesFM (day2) is exact; alignment with
    Chronos (day1) is NOT — handle that at merge time if you need it.
    """
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


def parse_ttm_revision(revision):
    """Parse revisions like "1024-96-r2" -> (1024, 96).

    Returns (context_length, prediction_length). Raises if the format isn't
    recognized. Lets the user pass a non-standard revision string and
    override context/prediction via separate flags if needed.
    """
    parts = revision.split("-")
    if len(parts) < 2:
        raise ValueError(
            f"can't parse TTM revision '{revision}'; "
            f"expected '{{context}}-{{prediction}}-r1|r2'. "
            f"Pass --ttm_context and --ttm_prediction to override."
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"can't parse integers from TTM revision '{revision}'. "
            f"Pass --ttm_context and --ttm_prediction to override."
        )


def nan_to_none(obj):
    """Recursively replace float NaN with None for strict-JSON output.

    json.dump emits NaN as the literal "NaN" which isn't valid JSON per
    RFC 7159 — Python's json.load accepts it but jq, JS JSON.parse, and
    Postgres JSONB will reject. Calling this before json.dump fixes that.
    """
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


# =============================================================================
# Granite-TTM model wrapper
# =============================================================================

def load_granite_ttm(revision, expected_ctx, expected_pred, device="cuda"):
    """Load Granite-TTM r2 (variant given by `revision`) for zero-shot inference.

    Tries the canonical transformers import first, falls back to the IBM
    granite-tsfm package if transformers doesn't have TinyTimeMixer yet.
    Either path produces the same model object — only the import differs.

    Asserts that the loaded config matches expected_ctx/expected_pred so we
    fail loud at load time, not deep inside the inference loop.
    """
    import torch

    log(f"loading Granite-TTM r2 (revision={revision}) on {device}...")
    torch.set_float32_matmul_precision("high")

    model = None
    try:
        from transformers import TinyTimeMixerForPrediction
        log("  using transformers.TinyTimeMixerForPrediction")
        model = TinyTimeMixerForPrediction.from_pretrained(
            TTM_REPO,
            revision=revision,
        )
    except (ImportError, AttributeError) as e:
        log(f"  transformers import failed ({e}); trying granite-tsfm")
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        log("  using tsfm_public.models.tinytimemixer.TinyTimeMixerForPrediction")
        model = TinyTimeMixerForPrediction.from_pretrained(
            TTM_REPO,
            revision=revision,
        )

    cfg = model.config
    cfg_ctx = getattr(cfg, "context_length", None)
    cfg_pred = getattr(cfg, "prediction_length", None)
    log(f"  config.context_length    = {cfg_ctx}")
    log(f"  config.prediction_length = {cfg_pred}")
    if cfg_ctx != expected_ctx:
        raise RuntimeError(
            f"TTM config.context_length={cfg_ctx} but expected {expected_ctx}. "
            f"Revision '{revision}' may not match --ttm_context override. "
            f"Either align them or pass --ttm_context {cfg_ctx}."
        )
    if cfg_pred != expected_pred:
        raise RuntimeError(
            f"TTM config.prediction_length={cfg_pred} but expected {expected_pred}. "
            f"Revision '{revision}' may not match --ttm_prediction override. "
            f"Either align them or pass --ttm_prediction {cfg_pred}."
        )

    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  TTM loaded ({n_params:,} params)")
    return model


def forecast_batch(model, context_list, h_steps, device="cuda"):
    """Run TTM forward pass on a batch of contexts, return list of dicts.

    TTM has FIXED context_length = TTM_CONTEXT_LENGTH. Short series get
    front-padded with zeros and the corresponding past_observed_mask entries
    set to 0 so TTM ignores them during normalization. Long series get
    truncated to the most recent TTM_CONTEXT_LENGTH points.

    TTM returns prediction_outputs of shape (N, TTM_PREDICTION_LEN, 1) for
    univariate. We take index h_steps-1 for the H-step-ahead prediction.

    Returns list of dicts (one per context), with quantile fields = NaN
    because TTM r2 standard checkpoints don't have a quantile head.
    """
    import torch

    n = len(context_list)
    ctx_len = TTM_CONTEXT_LENGTH

    # Build padded batch tensors. Right-align the actual context (most recent
    # values at the END), front-pad with zeros. past_observed_mask marks
    # observed=1, padded=0.
    past_values = torch.zeros((n, ctx_len, 1), dtype=torch.float32)
    past_observed_mask = torch.zeros((n, ctx_len, 1), dtype=torch.float32)

    for i, ctx in enumerate(context_list):
        ctx_arr = np.asarray(ctx, dtype=np.float32)
        L = min(len(ctx_arr), ctx_len)
        # Take the LAST L points if context is longer than ctx_len. Pad in
        # FRONT (left side) — TTM treats position 0 as oldest, position
        # ctx_len-1 as most recent.
        past_values[i, ctx_len - L:, 0] = torch.from_numpy(ctx_arr[-L:])
        past_observed_mask[i, ctx_len - L:, 0] = 1.0

    past_values = past_values.to(device)
    past_observed_mask = past_observed_mask.to(device)

    with torch.no_grad():
        outputs = model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
        )

    pred = outputs.prediction_outputs  # (N, TTM_PREDICTION_LEN, 1)
    expected_shape = (n, TTM_PREDICTION_LEN, 1)
    if tuple(pred.shape) != expected_shape:
        raise RuntimeError(
            f"TTM prediction_outputs shape {tuple(pred.shape)}, "
            f"expected {expected_shape}. Model variant mismatch?"
        )

    if h_steps < 1 or h_steps > TTM_PREDICTION_LEN:
        raise ValueError(
            f"h_steps={h_steps} out of range for TTM prediction length "
            f"{TTM_PREDICTION_LEN}"
        )

    step = h_steps - 1   # 0-indexed: last step in [0, h_steps) is at h_steps-1
    pred_step = pred[:, step, 0].detach().cpu().numpy()

    out = []
    for i in range(n):
        out.append({
            "pred": float(pred_step[i]),
            # TTM r2 standard: no quantile head. Keep keys for schema parity
            # with TimesFM/Chronos JSONs but write NaN. Downstream mergers
            # should detect and skip.
            "p10":  float("nan"),
            "p50":  float("nan"),
            "p90":  float("nan"),
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
    device,
    dataset_name,
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
                # Truncation also happens inside forecast_batch, but it doesn't
                # hurt to keep the slice short before sending to GPU.
                if len(ctx) > TTM_CONTEXT_LENGTH:
                    ctx = ctx[-TTM_CONTEXT_LENGTH:]
                contexts.append(ctx)

            preds = forecast_batch(model, contexts, h_steps, device=device)
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

    # persist per-point arrays for downstream DM tests / Chapter 5.
    # Filename includes dataset_name to prevent the cross-dataset clobber bug
    # that day2 has (running --dataset alibaba then --dataset bitbrains in
    # the same CWD overwrites the first dataset's per-point arrays silently).
    perpoint_path = f"granite_ttm_perpoint_{dataset_name}_{horizon_label}.npz"
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
        dataset=dataset_name,
    )
    log(f"  saved per-point arrays to {perpoint_path} ({len(y_true_all):,} points)")

    # Quantile sanity check — TTM has no quantile head, so all NaN. Skip if so.
    p10_arr = np.array(y_pred_p10_all, dtype=np.float64)
    p50_arr = np.array(y_pred_p50_all, dtype=np.float64)
    p90_arr = np.array(y_pred_p90_all, dtype=np.float64)

    all_nan = np.all(np.isnan(p10_arr)) and np.all(np.isnan(p50_arr)) and np.all(np.isnan(p90_arr))
    if all_nan:
        log("  TTM r2 has no native quantile head; quantile sanity skipped (NaN)")
        order_ok = float("nan")
        pct_inside = float("nan")
    else:
        # If a future TTM variant DOES populate quantiles, this branch handles it.
        order_ok = float(np.mean((p10_arr < p50_arr) & (p50_arr < p90_arr)))
        log(f"  quantile ordering sanity: p10<p50<p90 at {order_ok*100:.1f}% of points")
        if order_ok < 0.9:
            log("  *** quantile indices likely WRONG — check forecast_batch() ***")
        inside = (
            (np.asarray(y_true_all) >= p10_arr)
            & (np.asarray(y_true_all) <= p90_arr)
        )
        pct_inside = float(np.mean(inside))

    # NaN propagates through np.sum, so compute_r2/mae of all-NaN preds will
    # naturally return NaN. No explicit gating needed.
    return {
        "n_points": len(y_true_all),
        "r2_mean":  compute_r2(y_true_all, y_pred_mean_all),
        "r2_p50":   compute_r2(y_true_all, y_pred_p50_all),
        "mae_mean": compute_mae(y_true_all, y_pred_mean_all),
        "mae_p50":  compute_mae(y_true_all, y_pred_p50_all),
        "r2_naive_subsample":  compute_r2(y_true_all, y_pred_naive_all),
        "mae_naive_subsample": compute_mae(y_true_all, y_pred_naive_all),
        "mean_p10": float(np.mean(p10_arr)) if not all_nan else float("nan"),
        "mean_p90": float(np.mean(p90_arr)) if not all_nan else float("nan"),
        "pct_inside_p10_p90":   pct_inside,
        "quantile_order_ok_pct": order_ok,
    }


# =============================================================================
# Verdict printing (parallels day2 script, with TTM labels)
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
    print(f"  Granite-TTM R2 (mean) = {r2:.4f}")
    r2_p50 = results["r2_p50"]
    if math.isnan(r2_p50):
        print(f"  Granite-TTM R2 (p50)  = N/A (no quantile head)")
    else:
        print(f"  Granite-TTM R2 (p50)  = {r2_p50:.4f}")
    print(f"  Granite-TTM MAE       = {results['mae_mean']:.4f}")
    print(f"  same-sample naive R2  = {naive_sub:.4f}  (this run)")
    print(f"                        = {b_naive_sub:.4f}  (TimesFM run, sanity)")
    diff_naive = (naive_sub - b_naive_sub) * 100
    flag = "[OK]" if abs(diff_naive) < 0.1 else "[CHECK]"
    print(f"      diff = {diff_naive:+.3f} pp  {flag}")
    if abs(diff_naive) >= 0.1:
        print("      ** origins differ between TTM and TimesFM — "
              "samples are NOT aligned **")

    print(f"  delta TTM vs naive_sub = {(r2 - naive_sub)*100:+.2f} pp")
    if b_ens_sub is not None:
        print(f"  delta TTM vs ensemble  = {(r2 - b_ens_sub)*100:+.2f} pp   "
              f"(ensemble_sub = {b_ens_sub:.4f}, label='{ens_label}')")
    else:
        print(f"  ensemble: {ens_label} (no comparison)")

    pct_inside = results["pct_inside_p10_p90"]
    order_ok   = results["quantile_order_ok_pct"]
    if math.isnan(pct_inside):
        print(f"  p10/p90 coverage    = N/A (TTM r2 has no quantile head)")
    else:
        print(f"  p10/p90 coverage    = {pct_inside*100:.1f}%   (target 80%)")
        print(f"  quantile ordering   = {order_ok*100:.1f}% (should be >90%)")


# =============================================================================
# Main
# =============================================================================

def main():
    global TTM_CONTEXT_LENGTH, TTM_PREDICTION_LEN

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        default=None,
                        help="which dataset (defaults to auto from DATA_ROOT name)")
    parser.add_argument("--num_origins", type=int, default=None,
                        help="override K (default: 20 for Alibaba, 50 for BD/BB)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ttm_revision", default=DEFAULT_TTM_REVISION,
                        help=f"TTM revision tag on HF (default: {DEFAULT_TTM_REVISION}). "
                             f"For Alibaba consider '512-96-r2' to avoid ~50%% zero-padding.")
    parser.add_argument("--ttm_context", type=int, default=None,
                        help="override TTM context length; default parsed from --ttm_revision")
    parser.add_argument("--ttm_prediction", type=int, default=None,
                        help="override TTM prediction length; default parsed from --ttm_revision")
    args = parser.parse_args()

    # Resolve TTM shape config. If both overrides are provided we don't need
    # to parse the revision at all (this is the IBM-republished-under-"main"
    # fallback path). Otherwise we parse and let the overrides win where given.
    if args.ttm_context is not None and args.ttm_prediction is not None:
        TTM_CONTEXT_LENGTH = args.ttm_context
        TTM_PREDICTION_LEN = args.ttm_prediction
    else:
        parsed_ctx, parsed_pred = parse_ttm_revision(args.ttm_revision)
        TTM_CONTEXT_LENGTH = args.ttm_context if args.ttm_context is not None else parsed_ctx
        TTM_PREDICTION_LEN = args.ttm_prediction if args.ttm_prediction is not None else parsed_pred

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
        out_path = f"granite_ttm_k{K}_{dataset_name}.json"
    else:
        out_path = args.output

    log("=" * 64)
    log(f"Granite-TTM r2 zero-shot on {dataset_name.upper()}  (K={K})")
    log("=" * 64)
    log(f"DATA_ROOT:   {DATA_ROOT}")
    log(f"horizons:    {[h[0] for h in cfg['horizons']]}")
    log(f"cadence_min: {cfg['cadence_min']}")
    log(f"out_path:    {out_path}")
    log(f"TTM repo:    {TTM_REPO}  rev={args.ttm_revision}")
    log(f"TTM shapes:  context={TTM_CONTEXT_LENGTH}  prediction={TTM_PREDICTION_LEN}")

    series_by_id = load_series_by_id(DATA_ROOT)
    model = load_granite_ttm(
        revision=args.ttm_revision,
        expected_ctx=TTM_CONTEXT_LENGTH,
        expected_pred=TTM_PREDICTION_LEN,
        device=args.device,
    )

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
            device=args.device,
            dataset_name=dataset_name,
        )

    payload = {
        "model":                  f"granite-ttm-r2-{args.ttm_revision}",
        "dataset":                dataset_name,
        "cadence_min":            cfg["cadence_min"],
        "num_origins":            K,
        "min_context":            MIN_CONTEXT,
        "ttm_revision":           args.ttm_revision,
        "ttm_context_length":     TTM_CONTEXT_LENGTH,
        "ttm_prediction_length":  TTM_PREDICTION_LEN,
        "ens_label":              cfg["ens_label"],
        "has_quantile_head":      False,
        "granite_ttm_zero_shot":  all_results,
    }

    # Strip NaN -> None before serializing for strict-JSON consumers (jq, JS).
    with open(out_path, "w") as f:
        json.dump(nan_to_none(payload), f, indent=2)
    log(f"\nwrote {out_path}")

    log("\n" + "=" * 64)
    log("VERDICT")
    log("=" * 64)
    for horizon_label, _ in cfg["horizons"]:
        print_verdict(horizon_label, all_results[horizon_label],
                      cfg["baselines"], cfg["ens_label"])


if __name__ == "__main__":
    main()
