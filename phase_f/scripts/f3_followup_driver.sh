#!/bin/bash
# f3_followup_driver.sh
# Run all F3 follow-up experiments: rank sweep (4, 16), DoRA, curriculum.
# Then evaluate each with hold-out and asymmetric pinball. Then build comparison table.
#
# Total GPU time estimate: ~5-7 hours on RTX A4000 (overnight).
# Total CPU eval time: ~30 minutes.
#
# Run with:
#   cd /workspace/kubernetes-cpu-ensemble-thesis
#   nohup bash phase_f/scripts/f3_followup_driver.sh \
#       > phase_f/data/f3_followup_log.txt 2>&1 &
#   echo "Driver PID: $!"

set -e   # exit on error
set -u   # exit on unset variable

cd /workspace/kubernetes-cpu-ensemble-thesis
source /venv/main/bin/activate

echo "=========================================="
echo "F3 follow-up driver — $(date)"
echo "=========================================="
echo ""

# ============================================================
# Phase 1: Training (GPU)
# ============================================================

echo "[1/3] TRAINING phase"
echo ""

# 1a. LoRA rank=4
echo "--- 1a. LoRA rank=4 ---"
python phase_f/scripts/f3_v3_finetune.py \
    --rank 4 \
    --out-dir phase_f/models/f3_lora_rank4 \
    --history-csv phase_f/data/f3_training_rank4.csv
echo ""

# 1b. LoRA rank=16
echo "--- 1b. LoRA rank=16 ---"
python phase_f/scripts/f3_v3_finetune.py \
    --rank 16 \
    --out-dir phase_f/models/f3_lora_rank16 \
    --history-csv phase_f/data/f3_training_rank16.csv
echo ""

# 1c. DoRA at rank=8
echo "--- 1c. DoRA rank=8 ---"
python phase_f/scripts/f3_v3_finetune.py \
    --rank 8 --dora \
    --out-dir phase_f/models/f3_dora_rank8 \
    --history-csv phase_f/data/f3_training_dora.csv
echo ""

# 1d. Curriculum learning (rank=8, mixed horizons per batch)
echo "--- 1d. Curriculum rank=8 ---"
python phase_f/scripts/f3_v3_finetune.py \
    --rank 8 --curriculum \
    --out-dir phase_f/models/f3_curriculum_rank8 \
    --history-csv phase_f/data/f3_training_curriculum.csv
echo ""

echo "[1/3] TRAINING DONE — $(date)"
echo ""

# ============================================================
# Phase 2: Evaluation (CPU, with hold-out + asymmetric pinball)
# ============================================================

echo "[2/3] EVALUATION phase"
echo ""

# Asymmetric cost ratios: under:over = 1, 3, 5, 10 (1 = symmetric)
COSTS="1,3,5,10"
HOLDOUT="0.3"

for variant in lora_rank4 lora_rank16 dora_rank8 curriculum_rank8; do
    echo "--- Evaluating: $variant ---"
    python phase_f/scripts/f3_evaluate_v3.py \
        --adapter-dir "phase_f/models/f3_${variant}" \
        --tag "$variant" \
        --heldout-frac "$HOLDOUT" \
        --asymmetric-cost-ratios "$COSTS"
    echo ""
done

# Also evaluate the original rank=8 with hold-out + asymmetric (re-uses existing adapter)
echo "--- Re-evaluating: lora_rank8 (original) with hold-out + asymmetric ---"
python phase_f/scripts/f3_evaluate_v3.py \
    --adapter-dir "phase_f/models/f3_lora_rank8" \
    --tag "lora_rank8" \
    --heldout-frac "$HOLDOUT" \
    --asymmetric-cost-ratios "$COSTS"
echo ""

echo "[2/3] EVALUATION DONE — $(date)"
echo ""

# ============================================================
# Phase 3: Aggregate comparison
# ============================================================

echo "[3/3] AGGREGATING comparison table"
echo ""

python phase_f/scripts/f3_compare_variants.py

echo ""
echo "[3/3] DONE — $(date)"
echo ""
echo "=========================================="
echo "F3 follow-up driver complete"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Training logs:       phase_f/data/f3_training_*.csv"
echo "  Adapters:            phase_f/models/f3_*"
echo "  Eval per variant:    phase_f/data/f3_eval_*.json + .csv"
echo "  Comparison table:    phase_f/data/f3_variant_comparison.csv + .md"
echo ""
echo "Read phase_f/data/f3_variant_comparison.md for the headline table."
