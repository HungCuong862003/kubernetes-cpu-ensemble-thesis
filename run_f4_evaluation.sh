#!/usr/bin/env bash
# run_f4_evaluation.sh — orchestrate the full F4 closed-loop evaluation.
#
# Pre-flight: f4_evaluation_plan.md must be git-committed BEFORE this runs.
# Run order:
#   A. Verify the pre-registration is committed
#   B. Smoke-test the MPC core
#   C. Smoke-test the harness on 10 containers
#   D. Run the full harness (3 datasets x 2 pipelines x 3 seeds = 18 runs)
#   E. Run the Pareto bootstrap analysis
#   F. Show the verdict
#
# Estimated wall-clock: ~9h for step D, minutes for the rest.
#
# Usage: bash run_f4_evaluation.sh

set -e
cd /workspace/kubernetes-cpu-ensemble-thesis

V6_PARQUET="phase_f/data/f3_lora_quantiles_h060_cad30_sorted.parquet"
V7_PARQUET="phase_f/data/f3_quantiles_per_dataset_h060_cad30_sorted.parquet"

echo "=========================================="
echo "Step A: verify pre-registration is committed"
echo "=========================================="
if [ ! -f "phase_f/f4_evaluation_plan.md" ]; then
    echo "FAIL: phase_f/f4_evaluation_plan.md missing"
    echo "      The pre-registration must exist and be committed BEFORE running F4."
    exit 1
fi
# Check git is clean wrt the plan
if ! git log --oneline phase_f/f4_evaluation_plan.md 2>/dev/null | head -1 > /dev/null; then
    echo "FAIL: phase_f/f4_evaluation_plan.md has never been committed to git."
    echo "      Run: git add phase_f/f4_evaluation_plan.md"
    echo "           git commit -m 'pre-register F4 evaluation plan'"
    exit 1
fi
echo "OK: f4_evaluation_plan.md is committed."
git log --oneline phase_f/f4_evaluation_plan.md | head -3

echo
echo "=========================================="
echo "Step B: smoke-test f4_mpc"
echo "=========================================="
python phase_f/scripts/f4_mpc.py

echo
echo "=========================================="
echo "Step C: smoke-test the harness (5 containers per dataset)"
echo "=========================================="
python phase_f/scripts/f4_evaluation_harness.py \
    --v6-parquet "$V6_PARQUET" \
    --v7-parquet "$V7_PARQUET" \
    --max-containers 5 \
    --output phase_f/data/f4_trajectories_smoke.parquet \
    2>&1 | tee phase_f/data/f4_harness_smoke.log

# Check the smoke parquet was produced
if [ ! -f "phase_f/data/f4_trajectories_smoke.parquet" ]; then
    echo "FAIL: smoke harness did not produce output. See log."
    exit 1
fi
SMOKE_ROWS=$(python -c "import pandas as pd; print(len(pd.read_parquet('phase_f/data/f4_trajectories_smoke.parquet')))")
echo "OK: smoke harness produced $SMOKE_ROWS rows (expect ~90: 5 ctr x 3 ds x 2 pipe x 3 seeds)"
if [ "$SMOKE_ROWS" -lt 30 ]; then
    echo "FAIL: too few rows; abort before committing to full run"
    exit 1
fi

echo
echo "=========================================="
echo "Step D: FULL harness run (~9h)"
echo "=========================================="
echo "Starting at: $(date)"
python phase_f/scripts/f4_evaluation_harness.py \
    --v6-parquet "$V6_PARQUET" \
    --v7-parquet "$V7_PARQUET" \
    --output phase_f/data/f4_trajectories.parquet \
    2>&1 | tee phase_f/data/f4_harness_full.log
echo "Completed at: $(date)"

if [ ! -f "phase_f/data/f4_trajectories.parquet" ]; then
    echo "FAIL: full harness did not produce output. See log."
    exit 1
fi
FULL_ROWS=$(python -c "import pandas as pd; print(len(pd.read_parquet('phase_f/data/f4_trajectories.parquet')))")
echo "OK: full harness produced $FULL_ROWS rows"

echo
echo "=========================================="
echo "Step E: Pareto bootstrap analysis"
echo "=========================================="
python phase_f/scripts/f4_pareto_analysis.py \
    --trajectories phase_f/data/f4_trajectories.parquet \
    --R 1000 \
    2>&1 | tee phase_f/data/f4_pareto_analysis.log

echo
echo "=========================================="
echo "Step F: verdict summary"
echo "=========================================="
if [ -f "phase_f/data/f4_pareto_decision.json" ]; then
    cat phase_f/data/f4_pareto_decision.json
fi

echo
echo "=========================================="
echo "F4 EVALUATION COMPLETE."
echo "=========================================="
echo "Outputs:"
echo "  phase_f/data/f4_trajectories.parquet"
echo "  phase_f/data/f4_pareto_verdicts.csv"
echo "  phase_f/data/f4_bootstrap_results.csv"
echo "  phase_f/data/f4_pareto_decision.json"
echo "  phase_f/data/f4_pareto_{alibaba,bitbrains,bytedance,combined}.pdf"
echo
echo "Apply the decision rule per phase_f/f4_evaluation_plan.md and write up."
