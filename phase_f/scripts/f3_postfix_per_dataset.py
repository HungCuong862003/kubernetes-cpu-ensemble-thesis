"""
f3_postfix_per_dataset.py — extract per-dataset main vs holdout for the post-fix lora_rank8

This is the critical question after the fix:
  - For Alibaba (the ONLY truly held-out dataset), what's the post-fix improvement
    on the holdout cohort?
  - Compare to pre-fix to see if the fix actually generalises or just overfits val.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PHASE_F = Path("/workspace/kubernetes-cpu-ensemble-thesis/phase_f/data")

# Load baseline
with open(PHASE_F / "f3_zero_shot_baseline.json") as f:
    bl = json.load(f)
per_ds_baseline = {}
for entry in bl.get("results", []):
    if entry.get("horizon_min") == 60:
        per_ds_baseline[entry["dataset"]] = entry["pinball_loss"]["0.90"]

# Load pre-fix and post-fix eval JSONs
with open(PHASE_F / "f3_eval_lora_rank8_prefix.json") as f:
    prefix = json.load(f)
with open(PHASE_F / "f3_eval_lora_rank8.json") as f:
    postfix = json.load(f)

print("=" * 72)
print("PER-DATASET MAIN vs HOLDOUT — pre-fix vs post-fix")
print("=" * 72)
print()

pre_main    = prefix.get("per_dataset_main", {})
pre_holdout = prefix.get("per_dataset_holdout", {})
post_main    = postfix.get("per_dataset_main", {})
post_holdout = postfix.get("per_dataset_holdout", {})

for ds in ['alibaba', 'bitbrains', 'bytedance']:
    bl_v = per_ds_baseline.get(ds)
    if bl_v is None: continue
    
    pre_m = pre_main.get(ds)
    pre_h = pre_holdout.get(ds)
    post_m = post_main.get(ds)
    post_h = post_holdout.get(ds)
    
    pre_m_imp = (bl_v - pre_m) / bl_v * 100 if pre_m else None
    pre_h_imp = (bl_v - pre_h) / bl_v * 100 if pre_h else None
    post_m_imp = (bl_v - post_m) / bl_v * 100 if post_m else None
    post_h_imp = (bl_v - post_h) / bl_v * 100 if post_h else None
    
    print(f"\n{ds.upper()} (baseline {bl_v:.4f}):")
    print(f"  Pre-fix:")
    print(f"    main_pinball    = {pre_m:.6f}  ({pre_m_imp:+.2f}%)")
    print(f"    holdout_pinball = {pre_h:.6f}  ({pre_h_imp:+.2f}%)")
    if pre_m_imp is not None and pre_h_imp is not None:
        print(f"    replication delta = {abs(pre_m_imp - pre_h_imp):.2f}pp")
    print(f"  Post-fix:")
    print(f"    main_pinball    = {post_m:.6f}  ({post_m_imp:+.2f}%)")
    print(f"    holdout_pinball = {post_h:.6f}  ({post_h_imp:+.2f}%)")
    if post_m_imp is not None and post_h_imp is not None:
        print(f"    replication delta = {abs(post_m_imp - post_h_imp):.2f}pp")
    print(f"  Delta (pre → post):")
    print(f"    main change    = {post_m - pre_m:+.6f}  ({(post_m_imp - pre_m_imp):+.2f}pp)")
    print(f"    holdout change = {post_h - pre_h:+.6f}  ({(post_h_imp - pre_h_imp):+.2f}pp)")
    
    if ds == 'alibaba':
        print(f"  ** ALIBABA is the ONLY truly held-out dataset (val cohort was 14% of main, 0% of holdout). **")
        print(f"     The holdout change above is the cleanest generalisation signal we have.")

# Specific Alibaba holdout question
print()
print("=" * 72)
print("KEY QUESTION: Did the fix improve Alibaba HOLDOUT generalisation?")
print("=" * 72)
ali_pre_h_imp = (per_ds_baseline['alibaba'] - pre_holdout['alibaba']) / per_ds_baseline['alibaba'] * 100
ali_post_h_imp = (per_ds_baseline['alibaba'] - post_holdout['alibaba']) / per_ds_baseline['alibaba'] * 100
print(f"  Alibaba holdout pre-fix:  {ali_pre_h_imp:+.2f}%")
print(f"  Alibaba holdout post-fix: {ali_post_h_imp:+.2f}%")
print(f"  Improvement: {ali_post_h_imp - ali_pre_h_imp:+.2f}pp")
print()
if ali_post_h_imp > ali_pre_h_imp:
    print("  VERDICT: Fix improved truly-held-out Alibaba generalisation — fine-tune learns transferable patterns.")
elif abs(ali_post_h_imp - ali_pre_h_imp) < 0.5:
    print("  VERDICT: Fix neutral on Alibaba holdout — improvements on main are within-distribution only.")
else:
    print("  VERDICT: Fix worsened Alibaba holdout — fine-tune is overfitting val/main, not generalising.")

