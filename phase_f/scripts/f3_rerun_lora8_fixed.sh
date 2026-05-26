#!/bin/bash
# f3_rerun_lora8_fixed.sh — re-run F3 lora_rank8 with the val-cohort fix
# This replaces the original lora_rank8 result for DECISION-015 purposes.
# Other variants (rank4/16/dora/curriculum) keep their existing results as
# documented robustness checks (with the pre-fix val cohort, disclosed).
#
# Total runtime estimate: ~1.5 hours GPU + ~5 min CPU eval.
#
# Run with:
#   cd /workspace/kubernetes-cpu-ensemble-thesis
#   nohup bash phase_f/scripts/f3_rerun_lora8_fixed.sh \
#       > phase_f/data/f3_rerun_lora8_log.txt 2>&1 &
#   echo "PID: $!"

set -e
set -u

cd /workspace/kubernetes-cpu-ensemble-thesis
source /venv/main/bin/activate

echo "=========================================="
echo "F3 lora_rank8 RE-RUN with adaptive val_frac fix"
echo "Start: $(date)"
echo "=========================================="
echo ""

# Backup the old adapter before overwriting
if [ -d "phase_f/models/f3_lora_rank8" ]; then
    echo "Backing up old adapter to phase_f/models/f3_lora_rank8_prefix"
    mv phase_f/models/f3_lora_rank8 phase_f/models/f3_lora_rank8_prefix
fi

# Backup old eval JSON
if [ -f "phase_f/data/f3_eval_lora_rank8.json" ]; then
    cp phase_f/data/f3_eval_lora_rank8.json phase_f/data/f3_eval_lora_rank8_prefix.json
fi
if [ -f "phase_f/data/f3_eval_lora_rank8.csv" ]; then
    cp phase_f/data/f3_eval_lora_rank8.csv phase_f/data/f3_eval_lora_rank8_prefix.csv
fi
if [ -f "phase_f/data/f3_training_rank8.csv" ]; then
    cp phase_f/data/f3_training_rank8.csv phase_f/data/f3_training_rank8_prefix.csv
fi

echo ""
echo "[1/3] TRAINING phase (with fix)"
echo ""

python phase_f/scripts/f3_v3_finetune_FIXED.py \
    --rank 8 \
    --out-dir phase_f/models/f3_lora_rank8 \
    --history-csv phase_f/data/f3_training_rank8.csv

echo ""
echo "[1/3] TRAINING DONE — $(date)"
echo ""

# Sanity check: confirm adapter was saved
if [ ! -d "phase_f/models/f3_lora_rank8" ]; then
    echo "ERROR: adapter directory not created"
    exit 1
fi

echo ""
echo "[2/3] EVALUATION phase"
echo ""

python phase_f/scripts/f3_evaluate_v3.py \
    --adapter-dir phase_f/models/f3_lora_rank8 \
    --tag lora_rank8 \
    --heldout-frac 0.3 \
    --asymmetric-cost-ratios 1,3,5,10

echo ""
echo "[2/3] EVALUATION DONE — $(date)"
echo ""

echo ""
echo "[3/3] Comparing pre-fix vs post-fix"
echo ""

# Run a quick comparison
python3 << 'PYEOF'
import json
from pathlib import Path

PHASE_F = Path("/workspace/kubernetes-cpu-ensemble-thesis/phase_f/data")

with open(PHASE_F / "f3_eval_lora_rank8_prefix.json") as f:
    prefix = json.load(f)
with open(PHASE_F / "f3_eval_lora_rank8.json") as f:
    postfix = json.load(f)

print("=" * 60)
print("PRE-FIX vs POST-FIX comparison for lora_rank8")
print("=" * 60)

p_pre = prefix.get("primary_metric", {})
p_post = postfix.get("primary_metric", {})

print(f"\nPrimary metric (mean pinball h=60, tau=0.9):")
print(f"  Baseline:                 {p_pre.get('baseline'):.6f} (same)")
print(f"  Pre-fix main:             {p_pre.get('main_group_finetune'):.6f}  →  improvement {p_pre.get('main_group_improvement_pct'):+.2f}%")
print(f"  Post-fix main:            {p_post.get('main_group_finetune'):.6f}  →  improvement {p_post.get('main_group_improvement_pct'):+.2f}%")
print(f"  Pre-fix holdout:          {p_pre.get('holdout_group_finetune'):.6f}  →  improvement {p_pre.get('holdout_group_improvement_pct'):+.2f}%")
print(f"  Post-fix holdout:         {p_post.get('holdout_group_finetune'):.6f}  →  improvement {p_post.get('holdout_group_improvement_pct'):+.2f}%")
print(f"  Pre-fix replication d.:   {p_pre.get('replication_delta_pct'):.2f}pp")
print(f"  Post-fix replication d.:  {p_post.get('replication_delta_pct'):.2f}pp")

# Per-dataset comparison
print(f"\nPer-dataset main pinball (h=60, tau=0.9):")
pre_ds = prefix.get("per_dataset_main", {})
post_ds = postfix.get("per_dataset_main", {})
for ds in ["alibaba", "bitbrains", "bytedance"]:
    pre_v = pre_ds.get(ds)
    post_v = post_ds.get(ds)
    if pre_v and post_v:
        delta = post_v - pre_v
        print(f"  {ds:12s}  pre-fix {pre_v:.6f}  →  post-fix {post_v:.6f}  (delta {delta:+.6f})")
PYEOF

echo ""
echo "=========================================="
echo "DONE — $(date)"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Backup of pre-fix adapter:    phase_f/models/f3_lora_rank8_prefix/"
echo "  Backup of pre-fix eval:       phase_f/data/f3_eval_lora_rank8_prefix.{json,csv}"
echo "  Backup of pre-fix training:   phase_f/data/f3_training_rank8_prefix.csv"
echo "  New adapter:                  phase_f/models/f3_lora_rank8/"
echo "  New eval:                     phase_f/data/f3_eval_lora_rank8.{json,csv}"
echo "  New training history:         phase_f/data/f3_training_rank8.csv"
echo ""
echo "Next step: re-run f3_compare_variants.py to update the variant table,"
echo "then update DECISION-015 with the new lora_rank8 numbers."
