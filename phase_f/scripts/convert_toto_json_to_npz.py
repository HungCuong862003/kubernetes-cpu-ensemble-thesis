"""
convert_toto_json_to_npz.py
============================

Convert Toto foundation-model A2 JSON outputs to .npz format for
downstream F1 router consumption.

WHAT THE JSON CONTAINS
----------------------
A2_toto_serial.py (see that file for sampling protocol) writes a JSON
with per-horizon AGGREGATE statistics only:

    r2_mean, r2_p50, mae_mean, mae_p50,
    r2_naive_subsample, mae_naive_subsample,
    n_points, mean_p10, mean_p90, pct_inside_p10_p90

It does NOT save per-point arrays (y_true, y_pred, y_naive). Those
are accumulated locally inside evaluate_horizon() and discarded
after the aggregate is computed.

WHAT THIS SCRIPT PRODUCES
-------------------------
A .npz with named scalars per (horizon, metric) plus run metadata
and a copy of the JSON's baseline values for that dataset. Output
schema (one file per dataset):

  __model            : str   (e.g. 'toto-open-base-1.0')
  __dataset          : str   ('alibaba' / 'bitbrains' / 'bytedance')
  __cadence_min      : int32 (5 for ali/bb, 10 for byt)
  __num_origins      : int32 (20)
  __max_containers   : int32 (1000 for alibaba, -1 for unrestricted)
  __min_context      : int32
  __toto_max_context : int32
  __num_samples      : int32
  __seed             : int32
  __runtime_min      : float64
  __inference_mode   : str   ('serial')

  <horizon>__n_points              : int64
  <horizon>__r2_mean               : float64
  <horizon>__r2_p50                : float64
  <horizon>__mae_mean              : float64
  <horizon>__mae_p50               : float64
  <horizon>__r2_naive_subsample    : float64
  <horizon>__mae_naive_subsample   : float64
  <horizon>__baseline_naive        : float64
  <horizon>__baseline_hetero_ens   : float64

This schema matches the F1 router's expected input IF the router
decides at (dataset, horizon)-cell granularity. The pre-registered
F1 macro-F1 threshold (>= 0.55) over 11 cells implies cell-level
classification, which this schema supports.

If F1 later needs per-point arrays (y_true, y_pred, y_naive) for a
point-level classifier, A2_toto_serial.py must be modified to dump
arrays and re-run. See the note block at the bottom of this file.

USAGE
-----
Single file:
    python phase_f/scripts/convert_toto_json_to_npz.py \\
        --input  /workspace/.../foundation_comparison/toto_k20_alibaba.json \\
        --output /workspace/.../foundation_comparison/toto_k20_alibaba_npz.npz

Batch (all three datasets):
    python phase_f/scripts/convert_toto_json_to_npz.py --all

Roundtrip read example:
    import numpy as np
    d = np.load("toto_k20_alibaba_npz.npz", allow_pickle=False)
    r2_120 = float(d["120min__r2_mean"])
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(
    "/workspace/kubernetes-cpu-ensemble-thesis/results/foundation_comparison"
)

DATASETS = ["alibaba", "bitbrains", "bytedance"]

# Bytedance has no h10 by design (A2_toto_serial.py DATASETS dict lists
# only 30/60/120 for bytedance).
EXPECTED_HORIZONS = {
    "alibaba":   ["10min", "30min", "60min", "120min"],
    "bitbrains": ["10min", "30min", "60min", "120min"],
    "bytedance": [          "30min", "60min", "120min"],
}

# Anchor values from memory #23 (2026-05-21 TOTO ALIBABA full coverage,
# 4921 containers K=20 seed=42, n~98K/horizon).
# We allow +/- 0.5pp because in practice subsample selection at the
# 1000-container sweep introduced small drift; if drift is larger than
# this, the operator should investigate before trusting the npz.
ALIBABA_R2_ANCHORS = {
    "10min":  0.9173,
    "30min":  0.8513,
    "60min":  0.8205,
    "120min": 0.7586,
}
ANCHOR_TOL = 0.005

# Per-horizon metrics we lift from the JSON. Order matters only for
# log readability.
METRIC_KEYS = [
    "n_points",
    "r2_mean",
    "r2_p50",
    "mae_mean",
    "mae_p50",
    "r2_naive_subsample",
    "mae_naive_subsample",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_dataset_from_filename(path):
    """Try to figure out which dataset a JSON is from based on filename."""
    name = path.stem.lower()
    for ds in DATASETS:
        if ds in name:
            return ds
    return None


def convert_one(json_path, npz_path, dataset):
    """Load one toto_k20_*.json and write the corresponding npz."""
    log(f"converting {json_path.name}  ->  {npz_path.name}  (dataset={dataset})")

    with open(json_path) as f:
        blob = json.load(f)

    if "toto_zero_shot" not in blob:
        # If the file has a different top-level shape -- e.g. a Chronos-2
        # k20 file that uses chronos2_zero_shot -- we want a loud failure
        # rather than a silent miscategorisation.
        raise KeyError(
            f"{json_path}: missing 'toto_zero_shot' key. "
            f"Top-level keys: {list(blob.keys())}. "
            f"(Are you sure this is a Toto JSON and not Chronos-2 or another model?)"
        )

    horizons_in_json = list(blob["toto_zero_shot"].keys())
    log(f"  horizons in JSON: {horizons_in_json}")

    expected = EXPECTED_HORIZONS[dataset]
    missing = [h for h in expected if h not in horizons_in_json]
    extra   = [h for h in horizons_in_json if h not in expected]
    if missing:
        log(f"  WARN missing expected horizons for {dataset}: {missing}")
    if extra:
        log(f"  WARN unexpected extra horizons in JSON: {extra}")

    payload = {}

    # ----- Metadata (kept as 0-d arrays so the npz roundtrip is loss-free) -----
    # max_containers may be None in JSON (unrestricted run); store -1 in that case.
    raw_max = blob.get("max_containers")
    if raw_max is None:
        raw_max = -1

    payload["__model"]            = np.array(blob.get("model", "toto-open-base-1.0"))
    payload["__dataset"]          = np.array(blob.get("dataset", dataset))
    payload["__cadence_min"]      = np.array(blob.get("cadence_min", -1), dtype=np.int32)
    payload["__num_origins"]      = np.array(blob.get("num_origins", -1), dtype=np.int32)
    payload["__max_containers"]   = np.array(raw_max, dtype=np.int32)
    payload["__min_context"]      = np.array(blob.get("min_context", -1), dtype=np.int32)
    payload["__toto_max_context"] = np.array(blob.get("toto_max_context", -1), dtype=np.int32)
    payload["__num_samples"]      = np.array(blob.get("num_samples", -1), dtype=np.int32)
    payload["__seed"]             = np.array(blob.get("seed", -1), dtype=np.int32)
    payload["__runtime_min"]      = np.array(blob.get("runtime_min", -1.0), dtype=np.float64)
    payload["__inference_mode"]   = np.array(blob.get("inference_mode", "unknown"))

    # ----- Per-horizon Toto metrics -----
    n_horizon_rows_written = 0
    for h in horizons_in_json:
        m = blob["toto_zero_shot"][h]

        # If the horizon errored during the run we record an error marker
        # but skip writing numeric fields -- this preserves the failure
        # signal for downstream readers.
        if isinstance(m, dict) and "error" in m:
            log(f"  WARN {h}: horizon errored during run -> '{m['error']}'")
            payload[f"{h}__error"] = np.array(str(m["error"]))
            continue

        for key in METRIC_KEYS:
            v = m.get(key)
            if v is None:
                # mean_p10 / mean_p90 / pct_inside_p10_p90 are documented as
                # None for Toto runs because Toto's mean-forecast path
                # doesn't emit quantiles in our wrapper. Don't write them.
                continue
            dtype = np.int64 if key == "n_points" else np.float64
            payload[f"{h}__{key}"] = np.array(v, dtype=dtype)

        # ----- Per-horizon baselines (lifted from the JSON, not from memory) -----
        # We do this so the npz is self-contained: a downstream consumer
        # doesn't need to know about the canonical comparison numbers.
        baselines = blob.get("baselines", {}).get(h, {})
        for key in ["naive", "hetero_ens"]:
            v = baselines.get(key)
            if v is not None:
                payload[f"{h}__baseline_{key}"] = np.array(v, dtype=np.float64)

        n_horizon_rows_written += 1

    # ----- Write -----
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **payload)
    log(f"  wrote {len(payload)} keys ({n_horizon_rows_written} horizons) to {npz_path}")

    # ----- Sanity check Alibaba against memory #23 anchors -----
    if dataset == "alibaba":
        log(f"  sanity check vs memory #23 anchor (tol +/- {ANCHOR_TOL}):")
        any_drift = False
        for h, anchor in ALIBABA_R2_ANCHORS.items():
            key = f"{h}__r2_mean"
            if key not in payload:
                log(f"    {h}: MISSING in npz")
                any_drift = True
                continue
            got = float(payload[key])
            diff = got - anchor
            tag = "OK" if abs(diff) < ANCHOR_TOL else "DRIFT"
            if tag == "DRIFT":
                any_drift = True
            log(f"    {h}: got={got:.4f}  anchor={anchor:.4f}  "
                f"diff={diff:+.4f}  [{tag}]")
        if any_drift:
            log(f"  WARN drift exceeds tolerance; "
                f"investigate before using this npz in F1.")

    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        help="Path to a single toto_k20_*.json")
    parser.add_argument("--output", type=Path,
                        help="Path for output npz (use only with --input)")
    parser.add_argument("--all", action="store_true",
                        help="Convert all three datasets from --results_dir")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR,
                        help=f"For --all: where to find the JSON files "
                             f"(default: {RESULTS_DIR})")
    args = parser.parse_args()

    if args.all:
        if args.input or args.output:
            log("FATAL: --all is exclusive with --input/--output")
            sys.exit(1)

        n_done = 0
        n_skipped = 0
        for ds in DATASETS:
            json_path = args.results_dir / f"toto_k20_{ds}.json"
            npz_path  = args.results_dir / f"toto_k20_{ds}_npz.npz"
            if not json_path.exists():
                log(f"SKIP {ds}: {json_path} not found")
                n_skipped += 1
                continue
            try:
                convert_one(json_path, npz_path, ds)
                n_done += 1
            except Exception as err:
                log(f"ERROR on {ds}: {type(err).__name__}: {err}")
                import traceback
                traceback.print_exc()
                n_skipped += 1

        log("")
        log(f"summary: {n_done} converted, {n_skipped} skipped/failed")
        sys.exit(0 if n_skipped == 0 else 1)

    # ----- single-file mode -----
    if not (args.input and args.output):
        log("FATAL: pass --input AND --output, or use --all")
        sys.exit(1)

    ds = detect_dataset_from_filename(args.input)
    if ds is None:
        log(f"FATAL: cannot detect dataset from filename {args.input.name}. "
            f"Expected one of {DATASETS} in the filename.")
        sys.exit(1)

    convert_one(args.input, args.output, ds)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# Note: per-point arrays for F1
# ---------------------------------------------------------------------
#
# If F1's router needs per-point (y_true, y_pred, y_naive) arrays
# rather than the aggregate (R^2, MAE) lifted above, A2_toto_serial.py
# must be modified to save them. The minimal patch inside
# evaluate_horizon() is:
#
#     return {
#         "n_points":             len(y_true_all),
#         "r2_mean":              compute_r2(y_true_all, y_pred_all),
#         ...
#         "y_true":               y_true_all,        # NEW
#         "y_pred":               y_pred_all,        # NEW
#         "y_pred_naive":         y_pred_naive_all,  # NEW
#     }
#
# Storage cost (float64):
#   alibaba 1000-container subsample : ~98,000 pts/horizon x 4 horizons
#                                    : ~3 MB per array, ~9 MB per horizon
#   bitbrains                        : ~11,000 pts/horizon x 4 horizons (trivial)
#   bytedance                        : ~5,500 pts/horizon x 3 horizons (trivial)
#
# Runtime cost (re-run, RTX 4090, serial):
#   alibaba (1000 sub) ~ 1.5 hr
#   bitbrains          ~ 13 min
#   bytedance          ~  9 min
#
# This is deferred until the F1 router schema is finalised. Today's
# script captures what's actually in the JSON.
