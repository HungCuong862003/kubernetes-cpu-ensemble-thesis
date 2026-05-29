"""
f4_v1c_resave_f3_quantiles.py — Re-run F3 LoRA-FT eval saving PER-POINT
                                 quantile arrays in F4-consumable format.

PURPOSE: F3's existing eval file (f3_eval_lora_rank8.json, 30 KB) appears to
hold per-cell aggregate pinball loss stats, not per-(container, t_origin, h)
quantile arrays. F4's MPC needs the latter. This script re-runs LoRA-FT
inference and saves quantile arrays as .parquet for fast columnar access.

INPUTS (must exist on Vast):
  data/processed/<dataset>/train.parquet, val.parquet, test.parquet
  phase_f/models/f3_lora_rank8/                    (LoRA adapter directory)
  results/<dataset>/h060/predictions/test_spine.parquet  (test origin index)

OUTPUTS:
  phase_f/data/f3_lora_quantiles_h060.parquet      (canonical F4 input)
    columns: dataset, container_id, time_stamp, step_idx (0..H-1),
             q_0.5, q_0.7, q_0.8, q_0.9, q_0.95, cpu_true (joined from test)

  Optional per-horizon files for h10/h30/h120 if --all-horizons passed.

CALLING CONVENTION (verified in F3 Day-1 probe v4):
    quantile_preds_list, mean_preds_list = pipe.predict_quantiles(
        context_3d,                                         # (N, 1, n_context)
        prediction_length=H,
        quantile_levels=[0.5, 0.7, 0.8, 0.9, 0.95],
    )
  Each q_list element: (1, H, n_quantiles)
  Stack -> (n_series, 1, H, n_q), squeeze axis 1, moveaxis last->first
  -> (n_quantiles, n_series, H)

DATA LOADING (verified in F3 Day-1):
  Concat train+val+test per container for context (test alone has <512 obs).
  Use test_spine.parquet to define test ORIGINS (where to predict from).

This script is intentionally junior-programmer style with intermediate prints.
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


# ── CONFIG ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR    = PROJECT_ROOT / "results"
LORA_ADAPTER   = PROJECT_ROOT / "phase_f" / "models" / "f3_lora_rank8"
OUTPUT_DIR     = PROJECT_ROOT / "phase_f" / "data"

DATASETS = ["alibaba", "bitbrains", "bytedance"]

# F4 needs these quantile levels.
QUANTILE_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]

# Horizon to step count mapping. F4 primary = h060.
HORIZONS = {
    "h060": 12,    # 60 min / 5 min   - F4 PRIMARY
    "h010": 2,     # optional
    "h030": 6,     # optional
    "h120": 24,    # optional
}

N_CONTEXT = 512
BATCH_SIZE = 256


# ── helpers ────────────────────────────────────────────────────────

def detect_id_col(df):
    for cand in ("container_id", "vm_id", "instance_id"):
        if cand in df.columns:
            return cand
    raise ValueError(f"no id column in {list(df.columns)}")


def detect_cpu_col(df):
    # F3 day-1 doc: detect in this priority order
    for cand in ("cpu_util_percent", "cpu_target", "cpu", "cpu_percent"):
        if cand in df.columns:
            return cand
    raise ValueError(f"no cpu column in {list(df.columns)}")


def load_dataset_concat(dataset):
    """Load train+val+test parquets and concat per container, sorted by time.

    Per F3 day-1 doc: Alibaba test alone has ~340 obs/container, insufficient
    for N_CONTEXT=512. Concat with train+val gives enough context.
    Zero-shot/fine-tune inference has no leakage since train+val temporally
    precede test, so using them as context is standard time-series protocol.
    """
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
    print(f"    rows: {len(full):>10}  containers: {full[id_col].nunique():>6}  "
          f"id_col={id_col}  cpu_col={cpu_col}")
    return full, id_col, cpu_col


def load_test_spine(dataset, horizon_key):
    """Load test_spine.parquet to get the (container_id, time_stamp) test
    origins. F4's MPC will iterate over these origins."""
    p = RESULTS_DIR / dataset / horizon_key / "predictions" / "test_spine.parquet"
    if not p.exists():
        print(f"    WARN: {p} missing; will use all test rows as origins")
        return None
    spine = pd.read_parquet(p)
    return spine


def load_finetuned_pipeline():
    """Load Chronos-2 with LoRA-FT adapter applied. F3 Day-1's verified call
    chain is direct chronos-forecasting (no AutoGluon). PEFT adapter is loaded
    on top of the base pipeline."""
    print("\n  loading Chronos-2 base + LoRA adapter ...")
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    # Attach LoRA adapter. PEFT pattern:
    if LORA_ADAPTER.exists():
        try:
            from peft import PeftModel
            # NOTE: Chronos-2 internal model is at pipe.inner_model (this varies
            # by chronos-forecasting version; verify with introspection if it
            # fails). Probe v4 from F3 day-1 used pipe.predict_quantiles which
            # is the public interface.
            inner = pipe.inner_model if hasattr(pipe, "inner_model") else None
            if inner is None:
                # try other common attribute names
                for cand in ("model", "_model", "base_model"):
                    if hasattr(pipe, cand):
                        inner = getattr(pipe, cand)
                        break
            if inner is None:
                print("    WARNING: cannot find inner model attribute on pipeline.")
                print("    Pipeline attributes:", dir(pipe))
                print("    Aborting LoRA load; using zero-shot baseline.")
            else:
                wrapped = PeftModel.from_pretrained(inner, str(LORA_ADAPTER))
                # write back
                if hasattr(pipe, "inner_model"):
                    pipe.inner_model = wrapped
                else:
                    for cand in ("model", "_model", "base_model"):
                        if hasattr(pipe, cand):
                            setattr(pipe, cand, wrapped)
                            break
                print(f"    LoRA adapter loaded from {LORA_ADAPTER}")
        except Exception as e:
            print(f"    LoRA load FAILED: {e}")
            print(f"    Falling back to zero-shot. F4 evaluation will note this.")
    else:
        print(f"    NOTE: {LORA_ADAPTER} missing; running zero-shot only")
    return pipe


def predict_quantiles_for_origins(pipe, df_full, id_col, cpu_col,
                                   origins, H, n_context, batch_size):
    """For each origin (container_id, time_stamp), predict the H-step quantile
    path. Returns a long-format DataFrame:
      container_id, time_stamp, step_idx, q_0.5, q_0.7, q_0.8, q_0.9, q_0.95

    origins: DataFrame with columns [container_id (or vm_id etc.), time_stamp]
    """
    print(f"\n  predicting quantiles: {len(origins)} origins, H={H} ...")
    # Build a fast lookup from id -> sorted (time_stamp, cpu) arrays
    print("    indexing containers ...")
    grouped = df_full.groupby(id_col, sort=False)
    cpu_lookup = {}                  # id -> (ts_arr, cpu_arr)
    for cid, g in grouped:
        ts = g["time_stamp"].to_numpy()
        cv = g[cpu_col].to_numpy().astype(np.float32)
        cpu_lookup[cid] = (ts, cv)
    print(f"    indexed {len(cpu_lookup)} containers")

    # Group origins by container so we can prepare contexts efficiently.
    orig_id_col = origins.columns[0]   # by convention first col is the id
    contexts_to_predict = []           # list of (cid, t_origin, context_arr)
    skipped = 0
    for _, row in origins.iterrows():
        cid = row[orig_id_col]
        t_origin = row["time_stamp"]
        if cid not in cpu_lookup:
            skipped += 1
            continue
        ts, cv = cpu_lookup[cid]
        # index of t_origin in ts; we want context = cv[idx - n_context : idx]
        # so prediction starts at t_origin (the origin is the LAST observed point)
        idx = np.searchsorted(ts, t_origin)
        if idx < n_context:
            skipped += 1
            continue
        ctx = cv[idx - n_context: idx]
        contexts_to_predict.append((cid, t_origin, ctx))

    print(f"    contexts ready: {len(contexts_to_predict)}, skipped: {skipped}")
    if not contexts_to_predict:
        return pd.DataFrame()

    # Batch inference
    print(f"    batched inference, batch_size={batch_size} ...")
    all_q = []      # will collect arrays of shape (batch, H, n_q)
    all_meta = []   # parallel list of (cid, t_origin) per batch element
    t_start = time.time()

    for batch_start in range(0, len(contexts_to_predict), batch_size):
        batch = contexts_to_predict[batch_start: batch_start + batch_size]
        ctx_stack = np.stack([item[2] for item in batch])          # (B, n_context)
        # API: predict_quantiles wants (n_series, 1, n_context) per F3 day-1 probe v4
        ctx_tensor = torch.from_numpy(ctx_stack).unsqueeze(1).float()
        if torch.cuda.is_available():
            ctx_tensor = ctx_tensor.cuda()
        q_list, _ = pipe.predict_quantiles(
            ctx_tensor,
            prediction_length=H,
            quantile_levels=QUANTILE_LEVELS,
        )
        # q_list is a list of B tensors each (1, H, n_q). Stack -> (B, 1, H, n_q).
        q_stack = torch.stack(q_list, dim=0).squeeze(1)            # (B, H, n_q)
        q_arr = q_stack.detach().cpu().numpy().astype(np.float32)  # (B, H, n_q)
        all_q.append(q_arr)
        for item in batch:
            all_meta.append((item[0], item[1]))
        if (batch_start // batch_size) % 5 == 0:
            elapsed = time.time() - t_start
            done = batch_start + len(batch)
            print(f"      {done}/{len(contexts_to_predict)}  "
                  f"({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t_start
    print(f"    inference done in {elapsed:.1f}s")

    quantiles = np.concatenate(all_q, axis=0)                      # (N_origins, H, n_q)
    print(f"    quantiles shape: {quantiles.shape}")

    # Flatten to long format
    print("    flattening to long-format DataFrame ...")
    n_origins, H_chk, n_q = quantiles.shape
    assert H_chk == H, f"H mismatch: {H_chk} vs {H}"
    assert n_q == len(QUANTILE_LEVELS), f"n_q mismatch: {n_q} vs {len(QUANTILE_LEVELS)}"

    cids = np.array([m[0] for m in all_meta])
    ts_arr = np.array([m[1] for m in all_meta])
    step_idx = np.tile(np.arange(H), n_origins)
    cid_repeat = np.repeat(cids, H)
    ts_repeat = np.repeat(ts_arr, H)
    q_flat = quantiles.reshape(n_origins * H, n_q)

    df = pd.DataFrame({
        "container_id": cid_repeat,
        "time_stamp":   ts_repeat,
        "step_idx":     step_idx,
    })
    for i, ql in enumerate(QUANTILE_LEVELS):
        df[f"q_{ql}"] = q_flat[:, i]
    print(f"    long-format DataFrame: {df.shape}")
    return df


# ── main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", default=["h060"],
                    help="Which horizons to run. Default: h060 only.")
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    help="Which datasets to run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip inference; just verify loading works.")
    args = ap.parse_args()

    print("=" * 70)
    print("F4 V1C: RESAVE F3 QUANTILES AS PER-POINT ARRAYS")
    print("=" * 70)
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"LORA_ADAPTER = {LORA_ADAPTER}  (exists: {LORA_ADAPTER.exists()})")
    print(f"datasets:    {args.datasets}")
    print(f"horizons:    {args.horizons}")
    print(f"quantiles:   {QUANTILE_LEVELS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        pipe = load_finetuned_pipeline()

    for hkey in args.horizons:
        if hkey not in HORIZONS:
            print(f"unknown horizon {hkey}, skipping")
            continue
        H = HORIZONS[hkey]
        print(f"\n{'=' * 60}\nHORIZON: {hkey} (H={H} steps)\n{'=' * 60}")

        all_dfs = []
        for dset in args.datasets:
            print(f"\n--- {dset} ---")
            try:
                df_full, id_col, cpu_col = load_dataset_concat(dset)
            except FileNotFoundError as e:
                print(f"  skip: {e}")
                continue

            spine = load_test_spine(dset, hkey)
            if spine is None:
                # fallback: use last 500 observations per container as origins
                print("  building origins from data tail (no spine available)")
                origins_list = []
                for cid, g in df_full[df_full["_split"] == "test"].groupby(id_col):
                    g_sorted = g.sort_values("time_stamp")
                    if len(g_sorted) >= 1:
                        origins_list.append(g_sorted[[id_col, "time_stamp"]])
                if not origins_list:
                    print("  no test rows; skipping")
                    continue
                origins = pd.concat(origins_list, ignore_index=True)
            else:
                # spine columns include container_id and time_stamp
                origin_id_col = "container_id" if "container_id" in spine.columns else id_col
                origins = spine[[origin_id_col, "time_stamp"]].drop_duplicates()
                if origin_id_col != id_col:
                    origins = origins.rename(columns={origin_id_col: id_col})

            print(f"  origins to predict: {len(origins)}")

            if args.dry_run:
                print("  (dry run; skipping inference)")
                continue

            df_q = predict_quantiles_for_origins(
                pipe, df_full, id_col, cpu_col,
                origins, H, N_CONTEXT, BATCH_SIZE,
            )
            if df_q.empty:
                print(f"  WARN: empty quantiles for {dset}")
                continue
            df_q.insert(0, "dataset", dset)

            # Join ground-truth CPU at each forecasted step for evaluation.
            # We need cpu at (container_id, time_stamp + step_idx * resolution).
            # ByteDance uses 10-min resolution; others 5-min.
            interval_min = 10 if dset == "bytedance" else 5
            df_q["t_target"] = df_q["time_stamp"] + df_q["step_idx"] * interval_min * 60
            # join cpu_col from df_full
            join_keys = [id_col, "time_stamp"]
            truth = df_full[[id_col, "time_stamp", cpu_col]].rename(
                columns={cpu_col: "cpu_true", "time_stamp": "t_target",
                         id_col: "container_id"})
            df_q = df_q.merge(truth, on=["container_id", "t_target"], how="left")
            n_with_truth = df_q["cpu_true"].notna().sum()
            print(f"  joined truth: {n_with_truth}/{len(df_q)} rows have cpu_true")

            all_dfs.append(df_q)

        if not all_dfs:
            print(f"\nno data produced for {hkey}; skipping write")
            continue

        out_df = pd.concat(all_dfs, ignore_index=True)
        out_path = OUTPUT_DIR / f"f3_lora_quantiles_{hkey}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"\nWROTE: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
        print(f"  shape: {out_df.shape}")
        print(f"  columns: {list(out_df.columns)}")
        print(f"  by dataset:")
        for ds in args.datasets:
            sub = out_df[out_df["dataset"] == ds]
            print(f"    {ds}: {len(sub):>10}  origins: "
                  f"{sub.groupby(['container_id', 'time_stamp']).ngroups}")


if __name__ == "__main__":
    main()
