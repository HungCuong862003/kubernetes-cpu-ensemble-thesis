"""
f4_v1c2_resave_PER_DATASET.py — per-dataset adapter routing resaver (v7).

Reads phase_f/configs/router_v7.yaml, loads both LoRA adapters via PEFT's
multi-adapter API (add_adapter + set_adapter), and produces a single parquet
where each dataset's rows came from inference with its routed adapter.

This is a SEPARATE script from the v6 single-adapter resaver
(f4_v1c2_resave_f3_quantiles.py). The v6 script and parquet are untouched.

Outputs:
  phase_f/data/f3_quantiles_per_dataset_h060_cad30.parquet  (unsorted)
  After running, run phase_f/scripts/f4_v1h_sort_quantiles.py with --input
  pointing to this parquet to produce the *_sorted.parquet for F4 consumption.

Run from project root:
    python phase_f/scripts/f4_v1c2_resave_PER_DATASET.py \
        --router phase_f/configs/router_v7.yaml \
        --horizons h060 --cadence-min 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


sys.path.insert(0, str(Path(__file__).parent))
from transform_io import (
    load_transform_metadata,
    apply_input_transform,
    apply_output_inverse,
)


# Module-level routing state. Populated at startup; switched per dataset
# by set_active_route().
PEFT_MODEL       = None    # peft.PeftModel wrapping the inner Chronos-2 model
PIPE             = None    # the BaseChronosPipeline (mutates inner)
ACTIVE_ROUTE     = None    # current dataset name
ROUTE_STATE      = {}      # dict[dataset_id -> route dict]

# Per-call state -- read by predict_quantiles_for_origins.
INPUT_TRANSFORM  = "identity"
OUTPUT_TRANSFORM = "identity"
ZERO_FLOOR       = False
MAX_OUTPUT       = None


# ── CONFIG ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR     = PROJECT_ROOT / "phase_f" / "data"

QUANTILE_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]
HORIZONS = {"h060": 12, "h010": 2, "h030": 6, "h120": 24}
DATASET_RES_MIN = {"alibaba": 5, "bitbrains": 5, "bytedance": 10}
N_CONTEXT  = 512
BATCH_SIZE = 256


# ── data loaders (identical to single-adapter resaver) ─────────────

def detect_id_col(df):
    for cand in ("container_id", "vm_id", "instance_id"):
        if cand in df.columns:
            return cand
    raise ValueError(f"no id column in {list(df.columns)}")


def detect_cpu_col(df):
    for cand in ("cpu_util_percent", "cpu_target", "cpu", "cpu_percent"):
        if cand in df.columns:
            return cand
    raise ValueError(f"no cpu column in {list(df.columns)}")


def load_dataset_concat(dataset):
    print(f"\n  loading {dataset} ...")
    base = DATA_PROCESSED / dataset
    parts = []
    for split in ("train", "val", "test"):
        p = base / f"{split}.parquet"
        if not p.exists():
            print(f"    {split}.parquet MISSING; skipping")
            continue
        df = pd.read_parquet(p)
        df["_split"] = split
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"no parquets for {dataset} under {base}")
    full = pd.concat(parts, ignore_index=True)
    id_col = detect_id_col(full)
    cpu_col = detect_cpu_col(full)
    full = full.sort_values([id_col, "time_stamp"]).reset_index(drop=True)
    print(f"    rows: {len(full):>10}  containers: {full[id_col].nunique():>6}")
    return full, id_col, cpu_col


# ── multi-adapter loading ──────────────────────────────────────────

def load_pipeline_with_routes(router_config):
    """
    Load Chronos-2 base, then load ALL adapters from router config into one
    PEFT model. Returns nothing; populates global PEFT_MODEL, PIPE, ROUTE_STATE.
    """
    global PEFT_MODEL, PIPE, ROUTE_STATE

    print("\n  loading Chronos-2 base ...")
    from chronos import BaseChronosPipeline
    PIPE = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )

    # Find inner model attribute
    inner_attr = None
    for cand in ("inner_model", "model", "_model", "base_model"):
        if hasattr(PIPE, cand):
            inner_attr = cand
            break
    if inner_attr is None:
        raise RuntimeError("Cannot find inner model attribute on pipeline.")
    inner = getattr(PIPE, inner_attr)

    # Build list of (dataset_id, adapter_name, adapter_path, sidecar_path)
    routes = router_config["routes"]
    print(f"\n  loading {len(routes)} adapters via PEFT add_adapter ...")

    from peft import PeftModel

    # Walk through routes. The FIRST adapter is loaded via PeftModel.from_pretrained;
    # subsequent adapters loaded via load_adapter. Adapters with the same physical path
    # are loaded once with the FIRST dataset's adapter_name as the canonical key.
    loaded_paths = {}    # adapter_path -> adapter_name (the one we registered with)
    first_load = True
    for dataset_id, route in routes.items():
        adapter_path = route["adapter_path"]
        adapter_name = route["adapter_name"]

        if adapter_path in loaded_paths:
            # Already loaded under a different name. Reuse it.
            canonical = loaded_paths[adapter_path]
            print(f"    {dataset_id}: '{adapter_name}' aliases existing '{canonical}' "
                  f"(same path: {adapter_path})")
            ROUTE_STATE[dataset_id] = {
                "adapter_name": canonical,    # use the registered name
                "adapter_path": adapter_path,
                "sidecar_path": route["transform_metadata"],
            }
            continue

        # New adapter, load it
        if first_load:
            print(f"    {dataset_id}: PeftModel.from_pretrained('{adapter_name}', {adapter_path})")
            PEFT_MODEL = PeftModel.from_pretrained(
                inner, adapter_path, adapter_name=adapter_name
            )
            setattr(PIPE, inner_attr, PEFT_MODEL)
            first_load = False
        else:
            print(f"    {dataset_id}: peft_model.load_adapter('{adapter_name}', {adapter_path})")
            PEFT_MODEL.load_adapter(adapter_path, adapter_name=adapter_name)

        loaded_paths[adapter_path] = adapter_name
        ROUTE_STATE[dataset_id] = {
            "adapter_name": adapter_name,
            "adapter_path": adapter_path,
            "sidecar_path": route["transform_metadata"],
        }

    print(f"\n  loaded adapters: {list(PEFT_MODEL.peft_config.keys())}")
    print(f"  routing table:")
    for dataset_id, state in ROUTE_STATE.items():
        print(f"    {dataset_id:10s} -> {state['adapter_name']}")

    # Pre-cache sidecars (one per unique adapter_path; ROUTE_STATE entries that share
    # a path will share the cached sidecar)
    sidecar_cache = {}
    for dataset_id, state in ROUTE_STATE.items():
        sp = state["sidecar_path"]
        if sp not in sidecar_cache:
            # transform_io.load_transform_metadata accepts a directory containing the sidecar,
            # not the sidecar path. So we pass the parent of the sidecar_path.
            sidecar_cache[sp] = load_transform_metadata(Path(sp).parent)
        state["transform"] = sidecar_cache[sp]
        print(f"    {dataset_id}: sidecar input={state['transform']['input_transform']}, "
              f"output={state['transform']['output_transform']}, "
              f"zero_floor={state['transform']['zero_floor']}, "
              f"max_output={state['transform']['max_output']}")


def set_active_route(dataset):
    """Switch PEFT to the routed adapter AND switch transform state for this dataset."""
    global ACTIVE_ROUTE, INPUT_TRANSFORM, OUTPUT_TRANSFORM, ZERO_FLOOR, MAX_OUTPUT

    if dataset not in ROUTE_STATE:
        raise KeyError(f"No route configured for dataset {dataset!r}. "
                       f"Available: {list(ROUTE_STATE.keys())}")

    state = ROUTE_STATE[dataset]

    if ACTIVE_ROUTE == dataset:
        return    # already active; no-op

    # PEFT set_adapter — the critical line
    PEFT_MODEL.set_adapter(state["adapter_name"])
    if PEFT_MODEL.active_adapter != state["adapter_name"]:
        raise RuntimeError(
            f"PEFT set_adapter silent failure: requested {state['adapter_name']!r}, "
            f"got {PEFT_MODEL.active_adapter!r}. "
            f"See GitHub PEFT issue #1802. Aborting."
        )

    INPUT_TRANSFORM  = state["transform"]["input_transform"]
    OUTPUT_TRANSFORM = state["transform"]["output_transform"]
    ZERO_FLOOR       = state["transform"]["zero_floor"]
    MAX_OUTPUT       = state["transform"]["max_output"]
    ACTIVE_ROUTE = dataset

    print(f"\n  route active: {dataset}")
    print(f"     adapter:    {state['adapter_name']}")
    print(f"     input:      {INPUT_TRANSFORM}")
    print(f"     output:     {OUTPUT_TRANSFORM}")
    print(f"     zero_floor: {ZERO_FLOOR}")
    print(f"     max_output: {MAX_OUTPUT}")


# ── inference helpers (parallel to single-adapter resaver) ─────────

def build_origins_with_cadence(df_full, id_col, cadence_min, dataset):
    """Subsample test origins at the MPC cadence."""
    res = DATASET_RES_MIN[dataset]
    cadence_steps = max(1, cadence_min // res)
    test = df_full[df_full["_split"] == "test"]
    origins_list = []
    for cid, g in test.groupby(id_col, sort=False):
        g_sorted = g.sort_values("time_stamp")
        sub = g_sorted.iloc[::cadence_steps]
        origins_list.append(sub[[id_col, "time_stamp"]])
    if not origins_list:
        return pd.DataFrame(columns=[id_col, "time_stamp"])
    origins = pd.concat(origins_list, ignore_index=True)
    print(f"    cadence={cadence_steps} steps ({cadence_min} min), "
          f"origins={len(origins)}")
    return origins


def predict_quantiles_for_origins(df_full, id_col, cpu_col, origins, H):
    """Batched inference. Returns (origins-out, quantiles-array)."""
    n = len(origins)
    print(f"\n  predicting quantiles: {n} origins, H={H} ...")
    print(f"    indexing containers ...")
    cid_to_rows = {}
    for cid, g in df_full.groupby(id_col, sort=False):
        cid_to_rows[cid] = g.sort_values("time_stamp").reset_index(drop=True)
    print(f"    indexed {len(cid_to_rows)} containers")

    contexts = []
    keep_origins = []
    for i, row in origins.iterrows():
        cid = row[id_col]
        ts = row["time_stamp"]
        ctx_df = cid_to_rows[cid]
        idx = ctx_df.index[ctx_df["time_stamp"] == ts]
        if len(idx) == 0:
            continue
        ctx_end = idx[0]
        ctx_start = max(0, ctx_end - N_CONTEXT + 1)
        ctx = ctx_df.iloc[ctx_start:ctx_end + 1][cpu_col].to_numpy(dtype=np.float32)
        if len(ctx) < 2:
            continue
        contexts.append(ctx)
        keep_origins.append((cid, ts))

    print(f"    contexts ready: {len(contexts)}, skipped: {n - len(contexts)}")

    # Apply input transform per the active route's setting
    contexts_t = []
    for c in contexts:
        c_t = apply_input_transform(c, INPUT_TRANSFORM)
        contexts_t.append(torch.from_numpy(c_t))

    print(f"    batched inference, batch_size={BATCH_SIZE} ...")
    t0 = time.time()
    q_all = []
    for batch_start in range(0, len(contexts_t), BATCH_SIZE):
        batch = contexts_t[batch_start:batch_start + BATCH_SIZE]
        with torch.inference_mode():
            # Positional first-arg call — robust to API rename (context vs inputs).
            out = PIPE.predict_quantiles(
                batch,
                prediction_length=H,
                quantile_levels=QUANTILE_LEVELS,
            )
        if isinstance(out, tuple):
            q = out[0]
        else:
            q = out
        if torch.is_tensor(q):
            q = q.detach().cpu().numpy()
        q_all.append(q)
        if batch_start % (BATCH_SIZE * 10) == 0 and batch_start > 0:
            elapsed = time.time() - t0
            rate = (batch_start + len(batch)) / elapsed
            print(f"        {batch_start + len(batch):>7d}/{len(contexts_t)}  "
                  f"({elapsed:.1f}s, {rate:.0f} origins/s)")

    elapsed = time.time() - t0
    rate = len(contexts_t) / max(elapsed, 1e-9)
    print(f"    inference done in {elapsed:.1f}s ({rate:.0f} origins/s)")

    q_arr = np.concatenate(q_all, axis=0)
    print(f"    quantiles shape: {q_arr.shape}")

    # Chronos-2 sometimes returns 4-D (n_origins, n_variates=1, H, n_quantiles)
    # for univariate input; squeeze the variate dim if present so the downstream
    # 3-D flatten logic works regardless of the inference path taken.
    if q_arr.ndim == 4 and q_arr.shape[1] == 1:
        q_arr = q_arr[:, 0, :, :]
        print(f"    squeezed n_variates=1 dim; new shape: {q_arr.shape}")

    # Apply output inverse transform per the active route's setting
    q_arr = apply_output_inverse(
        q_arr,
        OUTPUT_TRANSFORM,
        zero_floor=ZERO_FLOOR,
        max_output=MAX_OUTPUT,
    )

    keep_origins_df = pd.DataFrame(keep_origins, columns=[id_col, "time_stamp"])
    return keep_origins_df, q_arr


def flatten_to_long(keep_origins_df, q_arr, id_col, dataset, H, res_min):
    """Long-format DataFrame for parquet output."""
    n_origins, H_check, n_q = q_arr.shape
    assert H_check == H
    assert n_q == len(QUANTILE_LEVELS)

    print(f"    flattening to long-format DataFrame ...")
    rows = []
    for i, (cid, ts) in enumerate(zip(keep_origins_df[id_col], keep_origins_df["time_stamp"])):
        for step in range(H):
            t_target = ts + (step + 1) * res_min * 60   # seconds offset
            row = {
                "dataset": dataset,
                "container_id": str(cid),
                "time_stamp": int(ts),
                "step_idx": step,
                "t_target": int(t_target),
            }
            for qi, q_level in enumerate(QUANTILE_LEVELS):
                row[f"q_{q_level}"] = float(q_arr[i, step, qi])
            rows.append(row)

    out = pd.DataFrame(rows)
    print(f"    long-format DataFrame: {out.shape}")
    return out


def join_truth(out_df, df_full, id_col, cpu_col):
    """Join the cpu_true column from the original data via time_stamp."""
    truth = df_full[[id_col, "time_stamp", cpu_col]].rename(
        columns={id_col: "container_id", "time_stamp": "t_target", cpu_col: "cpu_true"}
    )
    truth["container_id"] = truth["container_id"].astype(str)
    joined = out_df.merge(truth, on=["container_id", "t_target"], how="left")
    n_joined = joined["cpu_true"].notna().sum()
    print(f"  joined truth: {n_joined}/{len(joined)} rows")
    return joined


# ── main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", type=str, default="phase_f/configs/router_v7.yaml")
    ap.add_argument("--horizons", nargs="+", default=["h060"])
    ap.add_argument("--cadence-min", type=int, default=30)
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("F4 V1C2 PER-DATASET ROUTING RESAVER (v7)")
    print("=" * 70)

    # Load router config
    router_path = Path(args.router)
    if not router_path.exists():
        print(f"FAIL: {router_path} not found")
        return 1
    with open(router_path) as f:
        config = yaml.safe_load(f)
    print(f"router_version: {config['router_version']}")
    print(f"router file:    {router_path}")

    datasets = args.datasets or list(config["routes"].keys())
    print(f"datasets:       {datasets}")
    print(f"horizons:       {args.horizons}")
    print(f"cadence:        every {args.cadence_min} min")
    print(f"quantiles:      {QUANTILE_LEVELS}")
    print(f"GPU available:  {torch.cuda.is_available()}")

    # Load both adapters
    load_pipeline_with_routes(config)

    # Iterate per horizon, per dataset
    for hz in args.horizons:
        H = HORIZONS[hz]
        print(f"\n{'='*60}\nHORIZON: {hz} (H={H} steps)\n{'='*60}")
        all_outs = []
        for dataset in datasets:
            print(f"\n--- {dataset} ---")
            set_active_route(dataset)
            df_full, id_col, cpu_col = load_dataset_concat(dataset)
            origins = build_origins_with_cadence(df_full, id_col, args.cadence_min, dataset)
            if len(origins) == 0:
                print(f"    no test origins for {dataset}; skipping")
                continue
            keep_origins_df, q_arr = predict_quantiles_for_origins(
                df_full, id_col, cpu_col, origins, H
            )
            res_min = DATASET_RES_MIN[dataset]
            out_df = flatten_to_long(keep_origins_df, q_arr, id_col, dataset, H, res_min)
            out_df = join_truth(out_df, df_full, id_col, cpu_col)
            all_outs.append(out_df)

        if not all_outs:
            print(f"  no outputs for horizon {hz}; skipping write")
            continue

        combined = pd.concat(all_outs, ignore_index=True)
        out_path = OUTPUT_DIR / f"f3_quantiles_per_dataset_{hz}_cad{args.cadence_min}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / 1e6
        print(f"\nWROTE: {out_path}  ({size_mb:.2f} MB)")
        print(f"  shape: {combined.shape}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
