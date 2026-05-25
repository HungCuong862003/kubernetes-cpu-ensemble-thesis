"""
f1_verify.py  —  F1 result verification
Checks all F1 output CSVs against the canonical numbers reported in
f1_result_synthesis.md and D12/D13 journals.

Place in:  phase_f/scripts/
Run:       cd phase_f/scripts && python3 f1_verify.py

Expected result: ALL PASS (0 failures)
If any FAIL: number in CSV drifted from what was reported — investigate.
"""

import os
import numpy as np
import pandas as pd
from _paths import P

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(label, actual, expected, tol=0.0001):
    """Compare actual vs expected within tolerance. Record result."""
    ok = abs(float(actual) - float(expected)) <= tol
    status = PASS if ok else FAIL
    results.append((status, label, expected, actual))
    marker = "OK" if ok else "!!"
    print(f"  [{marker}] {label}")
    print(f"        expected={expected}  actual={round(float(actual),6)}")
    return ok

def check_exact(label, actual, expected):
    """Exact string/int match."""
    ok = str(actual).strip() == str(expected).strip()
    status = PASS if ok else FAIL
    results.append((status, label, expected, actual))
    marker = "OK" if ok else "!!"
    print(f"  [{marker}] {label}")
    print(f"        expected={expected!r}  actual={actual!r}")
    return ok

print("=" * 60)
print("F1 RESULT VERIFICATION")
print("=" * 60)

# ------------------------------------------------------------------
# 1. f1_feature_matrix.csv
# ------------------------------------------------------------------
print("\n--- f1_feature_matrix.csv ---")
fm_path = os.path.join(P['data'], "f1_feature_matrix.csv")
assert os.path.exists(fm_path), f"MISSING: {fm_path}"
fm = pd.read_csv(fm_path)

check_exact("row count (12 cells)", len(fm), 12)
check_exact("column count (6: dataset,horizon,4 features,label)", len(fm.columns), 7)

tally = fm['label'].value_counts().to_dict()
check_exact("Chronos-2 wins", tally.get('Chronos-2', 0), 6)
check_exact("TimesFM wins", tally.get('TimesFM', 0), 3)
check_exact("Granite-TTM wins", tally.get('Granite-TTM', 0), 2)
check_exact("NNLS wins", tally.get('NNLS', 0), 1)

# Spearman rho between acf_24h and cv must be -1.0 (dataset-constant finding)
from scipy.stats import spearmanr
rho_acf_cv, _ = spearmanr(fm['acf_24h'], fm['cv'])
check("Spearman rho(acf_24h, cv) = -1.0 (dataset-constant)", rho_acf_cv, -1.0, tol=0.001)

rho_acf_acf1h, _ = spearmanr(fm['acf_24h'], fm['acf_1h'])
check("Spearman rho(acf_24h, acf_1h) = -1.0 (dataset-constant)", rho_acf_acf1h, -1.0, tol=0.001)

# ------------------------------------------------------------------
# 2. f1_loo_cell_results.csv
# ------------------------------------------------------------------
print("\n--- f1_loo_cell_results.csv ---")
cell_path = os.path.join(P['data'], "f1_loo_cell_results.csv")
assert os.path.exists(cell_path), f"MISSING: {cell_path}"
cell = pd.read_csv(cell_path)

macro_row = cell[cell['class'] == 'MACRO'].iloc[0]
check("LOO-cell macro-F1 = 0.2532", macro_row['f1'], 0.2532, tol=0.0001)
check_exact("pre_reg_pass = False", macro_row['pre_reg_pass'], False)
check_exact("architecture = DecTree-depth3", macro_row['architecture'], "DecTree-depth3")
check("baseline_mf1 = 0.167", macro_row['baseline_mf1'], 0.167, tol=0.001)
check("uplift over baseline = 0.0862", macro_row['uplift'], 0.0862, tol=0.001)

per_class = cell[cell['class'] != 'MACRO'].set_index('class')['f1'].to_dict()
check("per-class F1: NNLS = 0.0000", per_class.get('NNLS', -1), 0.0, tol=0.001)
check("per-class F1: Chronos-2 = 0.7273", per_class.get('Chronos-2', -1), 0.7273, tol=0.001)
check("per-class F1: TimesFM = 0.2857", per_class.get('TimesFM', -1), 0.2857, tol=0.001)
check("per-class F1: Granite-TTM = 0.0000", per_class.get('Granite-TTM', -1), 0.0, tol=0.001)

# ------------------------------------------------------------------
# 3. f1_router_predictions.csv
# ------------------------------------------------------------------
print("\n--- f1_router_predictions.csv ---")
pred_path = os.path.join(P['data'], "f1_router_predictions.csv")
assert os.path.exists(pred_path), f"MISSING: {pred_path}"
preds = pd.read_csv(pred_path)

check_exact("prediction row count (12 folds)", len(preds), 12)
n_correct = int(preds['correct'].sum())
# From confusion matrix: 4 C2 + 1 TFM + 0 NNLS + 0 GTM = 6 correct... 
# Actually: C2:4, TFM:1, Granite:0, NNLS:0 = 5+1=6? Let me recount
# Confusion matrix: NNLS(0), C2(4), TFM(1), GTM(0) correct = 5+1=6 but wait
# C2 row: 4 correct. TFM row: 1 correct. Others: 0. Total = 5? 
# Actually: fold05(BB h10 C2 OK), fold08(BB h120 TFM OK), fold09-11(BD h10-h60 C2 OK) = 5
# Wait from output: OK=05,08,09,10,11 = 5 correct out of 12
check_exact("correct predictions count (5/12)", n_correct, 5)

# ------------------------------------------------------------------
# 4. f1_loo_dataset_results.csv
# ------------------------------------------------------------------
print("\n--- f1_loo_dataset_results.csv ---")
ds_path = os.path.join(P['data'], "f1_loo_dataset_results.csv")
assert os.path.exists(ds_path), f"MISSING: {ds_path}"
ds = pd.read_csv(ds_path)

check_exact("LOO-dataset row count (12)", len(ds), 12)
check("LOO-dataset macro-F1 = 0.1000", ds['macro_f1'].iloc[0], 0.1000, tol=0.001)
n_ds_correct = int(ds['correct'].sum())
check_exact("LOO-dataset correct count (1/12)", n_ds_correct, 1)

# ------------------------------------------------------------------
# 5. f1_dual_baseline_comparison.csv
# ------------------------------------------------------------------
print("\n--- f1_dual_baseline_comparison.csv ---")
dual_path = os.path.join(P['data'], "f1_dual_baseline_comparison.csv")
assert os.path.exists(dual_path), f"MISSING: {dual_path}"
dual = pd.read_csv(dual_path)

trivial   = dual[dual['baseline'] == 'trivial_always_chronos2'].iloc[0]
struct_   = dual[dual['baseline'] == 'structural_dataset_horizon_only'].iloc[0]
full_     = dual[dual['baseline'] == 'full_f1_router_4features'].iloc[0]

check("trivial baseline macro-F1 = 0.1670", trivial['macro_f1'], 0.167, tol=0.001)
check("structural baseline macro-F1 = 0.2253", struct_['macro_f1'], 0.2253, tol=0.001)
check("full router macro-F1 = 0.2532", full_['macro_f1'], 0.2532, tol=0.001)
check("full vs structural delta = +0.0280", full_['macro_f1'] - struct_['macro_f1'], 0.028, tol=0.001)
check("LOO-dataset macro-F1 stored = 0.1000", full_['loo_dataset_mf1'], 0.100, tol=0.001)
check_exact("architecture stored = DecTree-depth3", full_['architecture'], "DecTree-depth3")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

n_pass = sum(1 for r in results if r[0] == PASS)
n_fail = sum(1 for r in results if r[0] == FAIL)
total  = len(results)

print(f"\n  Total checks : {total}")
print(f"  PASS         : {n_pass}")
print(f"  FAIL         : {n_fail}")

if n_fail == 0:
    print(f"\n  ALL PASS — F1 numbers verified clean.")
else:
    print(f"\n  FAILURES ({n_fail}):")
    for status, label, expected, actual in results:
        if status == FAIL:
            print(f"    !! {label}")
            print(f"       expected={expected}  got={actual}")

print("\nf1_verify.py DONE")
