"""
f4_v1c2_resave_f3_quantiles.py — Re-run F3 LoRA-FT eval saving PER-POINT
                                  quantile arrays, with three bug fixes
                                  vs the v1 script.

Fixes vs f4_v1c_resave_f3_quantiles.py:

  FIX 1 -- pin_memory bug. Chronos-2's internal DataLoader tries to pin
           CPU tensors; passing a CUDA tensor breaks. Pass CPU tensor,
           the pipeline handles device placement.

  FIX 2 -- cadence subsample. Test-set total origins = 1.89M; predicting
           at every test row is excessive. F4 uses OptScaler-style 30-min
           cadence: predict only at scheduling timesteps. This drops
           origin count ~6x.

  FIX 3 -- micro-benchmark mode. New --benchmark flag runs inference on
           1000 origins only and reports throughput, so we have a real
           number for budgeting before committing to a long run.

Outputs (canonical):
  phase_f/data/f3_lora_quantiles_h060_cad30.parquet

Argparse:
  --horizons       which horizons (default: h060)
  --datasets       which datasets (default: all 3)
  --cadence-min    MPC scheduling cadence (default: 30 min)
  --benchmark      run 1000 origins for timing only
  --dry-run        skip inference, just verify data loading

Run on Vast:
    # 1) calibrate throughput first
    python phase_f/scripts/f4_v1c2_resave_f3_quantiles.py --benchmark

    # 2) if throughput acceptable, full h=60 run at 30-min cadence
    python phase_f/scripts/f4_v1c2_resave_f3_quantiles.py --horizons h060
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── transform_io helpers (sidecar pattern; reads adapter's transform_metadata.json)
# This module must live next to this script in phase_f/scripts/
sys.path.insert(0, str(Path(__file__).parent))
from transform_io import (
    load_transform_metadata,
    apply_input_transform,
    apply_output_inverse,
)

# Module-level holders; populated by load_finetuned_pipeline() once the
# adapter is loaded. Read by predict_quantiles_for_origins().
TRANSFORM_META   = None
INPUT_TRANSFORM  = "none"
OUTPUT_TRANSFORM = "none"
ZERO_FLOOR       = False


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
QUANTILE_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]

# Horizon to step count mapping. F4 primary = h060.
HORIZONS = {
    "h060": 12,
    "h010": 2,
    "h030": 6,
    "h120": 24,
}

DATASET_RES_MIN = {
    "alibaba":   5,
    "bitbrains": 5,
    "bytedance": 10,
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
    for cand in ("cpu_util_percent", "cpu_target", "cpu", "cpu_percent"):
        if cand in df.columns:
            return cand
    raise ValueError(f"no cpu column in {list(df.columns)}")


def load_dataset_concat(dataset):
    """Concat train+val+test per F3 day-1 convention."""
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


def load_finetuned_pipeline():
    """Chronos-2 base + LoRA-FT adapter + transform sidecar."""
    global TRANSFORM_META, INPUT_TRANSFORM, OUTPUT_TRANSFORM, ZERO_FLOOR
    print("\n  loading Chronos-2 base + LoRA adapter ...")
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    if LORA_ADAPTER.exists():
        try:
            from peft import PeftModel
            inner = None
            for cand in ("inner_model", "model", "_model", "base_model"):
                if hasattr(pipe, cand):
                    inner = getattr(pipe, cand)
                    break
            if inner is None:
                print("    WARNING: cannot find inner model attribute. Zero-shot only.")
            else:
                wrapped = PeftModel.from_pretrained(inner, str(LORA_ADAPTER))
                for cand in ("inner_model", "model", "_model", "base_model"):
                    if hasattr(pipe, cand):
                        setattr(pipe, cand, wrapped)
                        break
                print(f"    LoRA adapter loaded from {LORA_ADAPTER}")
        except Exception as e:
            print(f"    LoRA load FAILED: {e}; zero-shot only")
    else:
        print(f"    NOTE: {LORA_ADAPTER} missing; zero-shot only")

    # Load transform sidecar (transform_metadata.json next to the adapter).
    # If missing, defaults to identity transforms (no-op for old adapters).
    TRANSFORM_META = load_transform_metadata(LORA_ADAPTER)
    INPUT_TRANSFORM  = TRANSFORM_META["input_transform"]
    OUTPUT_TRANSFORM = TRANSFORM_META["output_transform"]
    ZERO_FLOOR       = TRANSFORM_META["zero_floor"]
    print(f"    resaver will apply: input={INPUT_TRANSFORM}, "
          f"output={OUTPUT_TRANSFORM}, zero_floor={ZERO_FLOOR}")
    return pipe


def build_origins_with_cadence(df_full, id_col, cadence_min, dataset):
    """Subsample test origins at the requested MPC cadence.

    For dataset with resolution R minutes and cadence_min C, take every
    (C/R)-th test row per container. Round up: at least one origin per
    container that has any test rows.
    """
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
    out = pd.concat(origins_list, ignore_index=True)
    print(f"    cadence={cadence_steps} steps ({cadence_steps * res} min), "
          f"origins={len(out)}")
    return out


def predict_quantiles_for_origins(pipe, df_full, id_col, cpu_col,
                                   origins, H, n_context, batch_size):
    """For each origin (container_id, time_stamp), predict H-step quantile path."""
    print(f"\n  predicting quantiles: {len(origins)} origins, H={H} ...")
    print("    indexing containers ...")
    grouped = df_full.groupby(id_col, sort=False)
    cpu_lookup = {}
    for cid, g in grouped:
        ts = g["time_stamp"].to_numpy()
        cv = g[cpu_col].to_numpy().astype(np.float32)
        cpu_lookup[cid] = (ts, cv)
    print(f"    indexed {len(cpu_lookup)} containers")

    orig_id_col = origins.columns[0]
    contexts_to_predict = []
    skipped = 0
    for _, row in origins.iterrows():
        cid = row[orig_id_col]
        t_origin = row["time_stamp"]
        if cid not in cpu_lookup:
            skipped += 1
            continue
        ts, cv = cpu_lookup[cid]
        idx = np.searchsorted(ts, t_origin)
        if idx < n_context:
            skipped += 1
            continue
        ctx = cv[idx - n_context: idx]
        contexts_to_predict.append((cid, t_origin, ctx))

    print(f"    contexts ready: {len(contexts_to_predict)}, skipped: {skipped}")
    if not contexts_to_predict:
        return pd.DataFrame()

    print(f"    batched inference, batch_size={batch_size} ...")
    all_q = []
    all_meta = []
    t_start = time.time()

    for batch_start in range(0, len(contexts_to_predict), batch_size):
        batch = contexts_to_predict[batch_start: batch_start + batch_size]
        ctx_stack = np.stack([item[2] for item in batch])

        # FIX 1: keep tensor on CPU. Chronos-2 pipeline pins CPU memory in its
        # internal DataLoader; passing a CUDA tensor triggers
        # "cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned".
        ctx_tensor = torch.from_numpy(ctx_stack).unsqueeze(1).float()

        # SIDECAR: forward transform (log1p when adapter was trained with log1p).
        # For no-log1p adapters, INPUT_TRANSFORM == "none" → identity.
        ctx_tensor = apply_input_transform(ctx_tensor, INPUT_TRANSFORM)

        q_list, _ = pipe.predict_quantiles(
            ctx_tensor,
            prediction_length=H,
            quantile_levels=QUANTILE_LEVELS,
        )
        q_stack = torch.stack(q_list, dim=0).squeeze(1)

        # SIDECAR: inverse transform back to raw scale (expm1 + zero floor when log1p).
        # For no-log1p adapters, OUTPUT_TRANSFORM == "none" → identity.
        q_stack = apply_output_inverse(q_stack, OUTPUT_TRANSFORM, zero_floor=ZERO_FLOOR)

        q_arr = q_stack.detach().cpu().numpy().astype(np.float32)
        all_q.append(q_arr)
        for item in batch:
            all_meta.append((item[0], item[1]))

        if (batch_start // batch_size) % 10 == 0:
            elapsed = time.time() - t_start
            done = batch_start + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            print(f"      {done:>7}/{len(contexts_to_predict)}  "
                  f"({elapsed:.1f}s, {rate:.0f} origins/s)")

    elapsed = time.time() - t_start
    final_rate = len(contexts_to_predict) / elapsed if elapsed > 0 else 0
    print(f"    inference done in {elapsed:.1f}s ({final_rate:.0f} origins/s)")

    quantiles = np.concatenate(all_q, axis=0)
    print(f"    quantiles shape: {quantiles.shape}")

    print("    flattening to long-format DataFrame ...")
    n_origins, H_chk, n_q = quantiles.shape
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


def run_benchmark(pipe, df_full, id_col, cpu_col, H, n_origins=1000):
    """Micro-benchmark: time inference on N synthetic origins."""
    print(f"\n  BENCHMARK MODE: timing {n_origins} origins at H={H} ...")
    test = df_full[df_full["_split"] == "test"].iloc[::10]    # spread
    if len(test) < n_origins:
        n_origins = len(test)
        print(f"    only {n_origins} test rows available; using all")
    sub = test.head(n_origins)
    origins = sub[[id_col, "time_stamp"]].reset_index(drop=True)
    t0 = time.time()
    df_q = predict_quantiles_for_origins(
        pipe, df_full, id_col, cpu_col, origins, H, N_CONTEXT, BATCH_SIZE,
    )
    elapsed = time.time() - t0
    rate = n_origins / elapsed
    print(f"\n  BENCHMARK RESULT: {n_origins} origins in {elapsed:.1f}s "
          f"= {rate:.0f} origins/s")
    return rate


# ── main ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", default=["h060"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--cadence-min", type=int, default=30,
                    help="MPC cadence in minutes (default 30, OptScaler default)")
    ap.add_argument("--benchmark", action="store_true",
                    help="Run 1000-origin micro-benchmark, then exit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip inference; verify data loading only.")
    args = ap.parse_args()

    print("=" * 70)
    print("F4 V1C2: RESAVE F3 QUANTILES (with FIXES)")
    print("=" * 70)
    print(f"datasets:      {args.datasets}")
    print(f"horizons:      {args.horizons}")
    print(f"cadence:       every {args.cadence_min} min")
    print(f"quantiles:     {QUANTILE_LEVELS}")
    print(f"GPU available: {torch.cuda.is_available()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run or args.benchmark:
        pass  # still need pipeline for benchmark
    if not args.dry_run:
        pipe = load_finetuned_pipeline()
    else:
        pipe = None

    # ── BENCHMARK MODE ─────────────────────────────────────────────
    if args.benchmark:
        print("\nrunning benchmark on alibaba h060 ...")
        df_full, id_col, cpu_col = load_dataset_concat("alibaba")
        H = HORIZONS["h060"]
        rate = run_benchmark(pipe, df_full, id_col, cpu_col, H, n_origins=1000)
        print()
        print("  At this throughput:")
        for n_orig in [10_000, 100_000, 300_000, 1_000_000]:
            secs = n_orig / rate
            if secs < 60:
                t_str = f"{secs:.0f}s"
            elif secs < 3600:
                t_str = f"{secs/60:.1f}m"
            else:
                t_str = f"{secs/3600:.2f}h"
            print(f"    {n_orig:>10d} origins -> {t_str}")
        return

    # ── PRODUCTION RUN ─────────────────────────────────────────────
    for hkey in args.horizons:
        if hkey not in HORIZONS:
            print(f"unknown horizon {hkey}, skipping")
            continue
        H = HORIZONS[hkey]
        print(f"\n{'='*60}\nHORIZON: {hkey} (H={H} steps)\n{'='*60}")

        all_dfs = []
        for dset in args.datasets:
            print(f"\n--- {dset} ---")
            try:
                df_full, id_col, cpu_col = load_dataset_concat(dset)
            except FileNotFoundError as e:
                print(f"  skip: {e}")
                continue

            # FIX 2: build origins with cadence subsample
            origins = build_origins_with_cadence(df_full, id_col,
                                                  args.cadence_min, dset)
            if origins.empty:
                print("  no origins; skipping")
                continue

            if args.dry_run:
                print("  (dry run; skipping inference)")
                continue

            df_q = predict_quantiles_for_origins(
                pipe, df_full, id_col, cpu_col,
                origins, H, N_CONTEXT, BATCH_SIZE,
            )
            if df_q.empty:
                continue
            df_q.insert(0, "dataset", dset)

            # join ground-truth CPU at each forecasted step
            interval_min = DATASET_RES_MIN[dset]
            df_q["t_target"] = df_q["time_stamp"] + df_q["step_idx"] * interval_min * 60
            truth = df_full[[id_col, "time_stamp", cpu_col]].rename(
                columns={cpu_col: "cpu_true", "time_stamp": "t_target",
                         id_col: "container_id"})
            df_q = df_q.merge(truth, on=["container_id", "t_target"], how="left")
            n_with_truth = df_q["cpu_true"].notna().sum()
            print(f"  joined truth: {n_with_truth}/{len(df_q)} rows")
            all_dfs.append(df_q)

        if not all_dfs:
            continue

        out_df = pd.concat(all_dfs, ignore_index=True)
        out_path = OUTPUT_DIR / f"f3_lora_quantiles_{hkey}_cad{args.cadence_min}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"\nWROTE: {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
        print(f"  shape: {out_df.shape}")


if __name__ == "__main__":
    main()
