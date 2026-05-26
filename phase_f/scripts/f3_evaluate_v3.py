"""
f3_evaluate_v3.py — Evaluate any LoRA/DoRA adapter on test set + held-out cells + asymmetric pinball

Adds three modes vs f3_evaluate.py:
  1. --heldout-frac F   Hold out the last F fraction of test series for replication check
  2. --asymmetric-cost-ratios r1,r2  Apply asymmetric pinball reweighting (e.g., 1,3,5,10)
  3. --adapter-dir PATH  Point at any saved LoRA/DoRA adapter

Usage:
    # Default (no hold-out, no asymmetric, evaluates main rank=8 LoRA)
    python phase_f/scripts/f3_evaluate_v3.py \
        --adapter-dir phase_f/models/f3_lora_rank8 \
        --tag lora_rank8

    # With hold-out and asymmetric pinball
    python phase_f/scripts/f3_evaluate_v3.py \
        --adapter-dir phase_f/models/f3_lora_rank8 \
        --tag lora_rank8 \
        --heldout-frac 0.3 \
        --asymmetric-cost-ratios 1,3,5,10

    # Evaluate DoRA adapter
    python phase_f/scripts/f3_evaluate_v3.py \
        --adapter-dir phase_f/models/f3_dora_rank8 \
        --tag dora_rank8
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from peft import PeftModel
from chronos import BaseChronosPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--adapter-dir", type=str, required=True)
parser.add_argument("--tag", type=str, required=True, help="Short label for this run (e.g. lora_rank4, dora_rank8, curriculum)")
parser.add_argument("--heldout-frac", type=float, default=0.0, help="Fraction of test series to hold out (last F of each dataset)")
parser.add_argument("--asymmetric-cost-ratios", type=str, default="", help="Comma-separated cost ratios for asymmetric pinball reweighting")
parser.add_argument("--out-json", type=str, default=None)
parser.add_argument("--out-csv", type=str, default=None)
args = parser.parse_args()

WORKSPACE  = Path("/workspace/kubernetes-cpu-ensemble-thesis")
DATA_DIR   = WORKSPACE / "data" / "processed"
PHASE_F    = WORKSPACE / "phase_f"
ADAPTER_DIR = Path(args.adapter_dir)

BASELINE_JSON = PHASE_F / "data" / "f3_zero_shot_baseline.json"
OUT_JSON = Path(args.out_json) if args.out_json else PHASE_F / "data" / f"f3_eval_{args.tag}.json"
OUT_CSV  = Path(args.out_csv) if args.out_csv else PHASE_F / "data" / f"f3_eval_{args.tag}.csv"

DATASETS     = ["alibaba", "bitbrains", "bytedance"]
INTERVAL_MIN = {"alibaba": 5, "bitbrains": 5, "bytedance": 10}
HORIZONS_MIN = [10, 30, 60, 120]
SKIP_CELLS   = {("bytedance", 10)}
TAUS         = [0.5, 0.7, 0.8, 0.9, 0.95]

N_CONTEXT  = 512
BATCH_SIZE = 256
PRIMARY_H_MIN = 60
PRIMARY_TAU   = 0.9

cost_ratios = [float(r) for r in args.asymmetric_cost_ratios.split(",") if r.strip()]
if not cost_ratios:
    cost_ratios = [1.0]  # symmetric only

print("=" * 70)
print(f"f3_evaluate_v3 — tag={args.tag}")
print(f"Adapter: {ADAPTER_DIR}")
print(f"Hold-out fraction: {args.heldout_frac}")
print(f"Cost ratios: {cost_ratios}")
print("=" * 70)

# -------------------------------------------------------------------------
# 1. Load model + adapter
# -------------------------------------------------------------------------

print("Step 1: Loading Chronos-2 + adapter")
pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu", dtype=torch.float32)
pipe.model = PeftModel.from_pretrained(pipe.model, str(ADAPTER_DIR))
pipe.model = pipe.model.merge_and_unload()
print(f"  Total params: {sum(p.numel() for p in pipe.model.parameters()):,}")
print()

# -------------------------------------------------------------------------
# 2. Load all series and split train/test (concat all parquets then split)
# -------------------------------------------------------------------------

def load_series(dataset):
    parts = []
    for split in ["train", "val", "test"]:
        p = DATA_DIR / dataset / f"{split}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    id_col  = next(c for c in ["container_id","vm_id","instance_id"] if c in df.columns)
    cpu_col = next(c for c in ["cpu_util_percent","cpu_target","cpu","cpu_percent"]
                   if c in df.columns)
    df = df.sort_values([id_col, "time_stamp"])
    out = {}
    for uid, grp in df.groupby(id_col):
        out[str(uid)] = grp[cpu_col].to_numpy(dtype=np.float32)
    return out, id_col

# -------------------------------------------------------------------------
# 3. Window builder with optional hold-out split
# -------------------------------------------------------------------------

def build_test_windows(series_dict, h_obs, use_heldout=False):
    """
    Build test windows. Each series uses its last (N_CONTEXT + h_obs) points.
    
    If use_heldout=True: split series alphabetically/numerically, return the last
    heldout_frac as held-out evaluation set.
    """
    sids = sorted(series_dict.keys())
    
    if use_heldout:
        cut = int(len(sids) * (1 - args.heldout_frac))
        sids_main = sids[:cut]
        sids_holdout = sids[cut:]
        groups = {"main": sids_main, "holdout": sids_holdout}
    else:
        groups = {"main": sids}
    
    results = {}
    for group_name, group_sids in groups.items():
        ctxs, tgts, valid_ids = [], [], []
        for sid in group_sids:
            arr = series_dict[sid]
            if len(arr) < N_CONTEXT + h_obs:
                continue
            ctx = arr[-(N_CONTEXT + h_obs):-h_obs]
            tgt = arr[-h_obs:]
            ctxs.append(ctx); tgts.append(tgt); valid_ids.append(sid)
        if not ctxs:
            results[group_name] = None
            continue
        ctx_t = torch.tensor(np.stack(ctxs), dtype=torch.float32).unsqueeze(1)
        results[group_name] = (ctx_t, np.stack(tgts), valid_ids)
    return results

# -------------------------------------------------------------------------
# 4. Loss functions
# -------------------------------------------------------------------------

def symmetric_pinball(y_true, y_pred, tau):
    diff = y_true - y_pred
    return float(np.where(diff >= 0, tau*diff, (tau-1)*diff).mean())

def asymmetric_pinball(y_true, y_pred, tau, cost_ratio_under_over):
    """
    Asymmetric pinball: under-prediction (y_true > y_pred) costs cost_ratio more.
    Cost ratio = 1 → symmetric pinball.
    Cost ratio = 5 → under-prediction costs 5x more (typical SLA setting).
    """
    diff = y_true - y_pred
    return float(np.where(
        diff >= 0,
        tau * cost_ratio_under_over * diff,    # under-prediction
        (tau - 1) * diff                       # over-prediction (negative)
    ).mean())

# -------------------------------------------------------------------------
# 5. Inference
# -------------------------------------------------------------------------

def run_inference(ctx_tensor, h_obs):
    n_series = ctx_tensor.shape[0]
    all_q = []
    for i in range(0, n_series, BATCH_SIZE):
        batch = ctx_tensor[i:i+BATCH_SIZE]
        with torch.no_grad():
            q_list, _ = pipe.predict_quantiles(
                batch, prediction_length=h_obs, quantile_levels=TAUS
            )
        q_stack = torch.stack(q_list, dim=0).squeeze(1)
        all_q.append(q_stack.numpy())
    return np.concatenate(all_q, axis=0)

# -------------------------------------------------------------------------
# 6. Load baseline
# -------------------------------------------------------------------------

with open(BASELINE_JSON) as f:
    bl = json.load(f)
baseline_primary = bl["metadata"]["pre_registration"]["baseline_primary"]
print(f"Baseline primary (h={PRIMARY_H_MIN} τ={PRIMARY_TAU}): {baseline_primary:.6f}")
print()

# -------------------------------------------------------------------------
# 7. Evaluation loop
# -------------------------------------------------------------------------

use_heldout = args.heldout_frac > 0.0
all_rows = []
all_cells = []

print("Step 3: Evaluation")
for ds in DATASETS:
    print(f"\n  Dataset: {ds}")
    series_dict, _ = load_series(ds)
    
    for h_min in HORIZONS_MIN:
        if (ds, h_min) in SKIP_CELLS:
            continue
        h_obs = h_min // INTERVAL_MIN[ds]
        
        groups = build_test_windows(series_dict, h_obs, use_heldout=use_heldout)
        
        for group_name, data in groups.items():
            if data is None:
                continue
            ctx_t, tgt_a, valid_ids = data
            n_valid = len(valid_ids)
            print(f"    h={h_min} ({h_obs} obs), {group_name}: {n_valid} series")
            
            q_preds = run_inference(ctx_t, h_obs)  # (N, H, n_q)
            
            cell = {
                "dataset": ds, "horizon_min": h_min, "horizon_obs": h_obs,
                "group": group_name, "n_valid": n_valid,
                "metrics": {}
            }
            
            for i, tau in enumerate(TAUS):
                q_i = q_preds[:, :, i]
                sym = symmetric_pinball(tgt_a, q_i, tau)
                cell["metrics"][f"sym_pinball_tau{tau:.2f}"] = sym
                
                # Asymmetric variants
                for cr in cost_ratios:
                    if cr == 1.0:
                        continue
                    asym = asymmetric_pinball(tgt_a, q_i, tau, cr)
                    cell["metrics"][f"asym_pinball_tau{tau:.2f}_cr{cr}"] = asym
                
                # Flat row
                all_rows.append({
                    "tag": args.tag,
                    "dataset": ds, "horizon_min": h_min,
                    "group": group_name, "tau": tau,
                    "sym_pinball": sym,
                    "n_valid": n_valid,
                })
            
            all_cells.append(cell)

# -------------------------------------------------------------------------
# 8. Compute primary metric (main group only)
# -------------------------------------------------------------------------

print()
print("=" * 70)
print("Primary metric: mean symmetric pinball at h=60, τ=0.9")
print("=" * 70)

primary_main = []
primary_holdout = []
per_dataset_main = {}
per_dataset_holdout = {}

for cell in all_cells:
    if cell["horizon_min"] != PRIMARY_H_MIN:
        continue
    val = cell["metrics"].get(f"sym_pinball_tau{PRIMARY_TAU:.2f}")
    if val is None: continue
    if cell["group"] == "main":
        primary_main.append(val)
        per_dataset_main[cell["dataset"]] = val
    elif cell["group"] == "holdout":
        primary_holdout.append(val)
        per_dataset_holdout[cell["dataset"]] = val

primary_main_mean = float(np.mean(primary_main)) if primary_main else float("nan")
improvement_main = (baseline_primary - primary_main_mean) / baseline_primary * 100

print(f"\nMain group:")
print(f"  Baseline       : {baseline_primary:.6f}")
print(f"  This adapter   : {primary_main_mean:.6f}")
print(f"  Improvement    : {improvement_main:+.2f}%")

if use_heldout and primary_holdout:
    primary_holdout_mean = float(np.mean(primary_holdout))
    improvement_holdout = (baseline_primary - primary_holdout_mean) / baseline_primary * 100
    replication_delta = abs(improvement_main - improvement_holdout)
    print(f"\nHold-out group:")
    print(f"  This adapter   : {primary_holdout_mean:.6f}")
    print(f"  Improvement    : {improvement_holdout:+.2f}%")
    print(f"  Replication Δ  : {replication_delta:.2f}pp (smaller = better)")

print()

# -------------------------------------------------------------------------
# 9. Asymmetric pinball ranking at primary cell
# -------------------------------------------------------------------------

if len(cost_ratios) > 1:
    print("Asymmetric pinball at h=60 τ=0.9 (main group):")
    print(f"  {'cost_ratio':>10}  {'mean_asym_pinball':>20}")
    for cr in cost_ratios:
        if cr == 1.0:
            vals = primary_main  # symmetric == cr=1
        else:
            vals = []
            for cell in all_cells:
                if cell["horizon_min"] != PRIMARY_H_MIN or cell["group"] != "main":
                    continue
                v = cell["metrics"].get(f"asym_pinball_tau{PRIMARY_TAU:.2f}_cr{cr}")
                if v is not None: vals.append(v)
        mean_v = float(np.mean(vals)) if vals else float("nan")
        print(f"  {cr:>10.1f}  {mean_v:>20.6f}")
    print()

# -------------------------------------------------------------------------
# 10. Save
# -------------------------------------------------------------------------

summary = {
    "tag": args.tag,
    "adapter_dir": str(ADAPTER_DIR),
    "primary_metric": {
        "baseline": baseline_primary,
        "main_group_finetune": primary_main_mean,
        "main_group_improvement_pct": round(improvement_main, 4),
    },
    "per_dataset_main": per_dataset_main,
    "cost_ratios_tested": cost_ratios,
    "cells": all_cells,
}

if use_heldout and primary_holdout:
    summary["primary_metric"]["holdout_group_finetune"] = float(np.mean(primary_holdout))
    summary["primary_metric"]["holdout_group_improvement_pct"] = round(improvement_holdout, 4)
    summary["primary_metric"]["replication_delta_pct"] = round(replication_delta, 4)
    summary["per_dataset_holdout"] = per_dataset_holdout

with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"Saved: {OUT_JSON}")

pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
print()
print(f"f3_evaluate_v3 DONE — tag={args.tag}")
