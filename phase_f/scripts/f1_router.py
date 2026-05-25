"""
f1_router.py  —  D12+D13 Tasks A+B
Architecture sweep, LOO-cell CV (pre-registered), LOO-dataset CV (robustness),
dual-baseline comparison.

Run f1_setup.py first.

Place in:  phase_f/
Run:       python3 f1_router.py

Outputs (all in phase_f/data/):
    f1_router_predictions.csv
    f1_loo_cell_results.csv
    f1_loo_dataset_results.csv
    f1_dual_baseline_comparison.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder

PHASE_F_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PHASE_F_DIR, "data")

CLASSES           = ["NNLS", "Chronos-2", "TimesFM", "Granite-TTM"]
PRE_REG_THRESHOLD = 0.55
BASELINE_MF1      = 0.167   # always-predict-Chronos-2 macro-F1

# ------------------------------------------------------------------
# Step 1: load feature matrix produced by f1_setup.py
# ------------------------------------------------------------------
print("=== Step 1: Load feature matrix ===")
fm = pd.read_csv(os.path.join(DATA_DIR, "f1_feature_matrix.csv"))
print(fm.to_string(index=False))

X_raw        = fm[["acf_24h", "horizon_min", "cv", "acf_1h"]].values
y            = fm["label"].values
datasets_arr = fm["dataset"].to_numpy(dtype=str)
print(f"\nN={len(y)}  distribution={dict(zip(*np.unique(y, return_counts=True)))}")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def loo_cell_cv(clf_fn, X, y, scale=False):
    preds = []
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        Xtr, Xte = X[mask], X[[i]]
        ytr, yte  = y[mask], y[i]
        if scale:
            sc  = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        clf = clf_fn()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)[0]
        preds.append((yte, pred))
        ok = "OK" if yte == pred else "--"
        print(f"  {ok} fold {i+1:02d}  ({fm.dataset.iloc[i]}, {fm.horizon.iloc[i]})"
              f"  actual={yte:<13}  pred={pred}")
    return preds


def metrics(preds):
    act = [p[0] for p in preds]
    prd = [p[1] for p in preds]
    mf1   = f1_score(act, prd, labels=CLASSES, average="macro",  zero_division=0)
    pcf   = f1_score(act, prd, labels=CLASSES, average=None,     zero_division=0)
    cm    = confusion_matrix(act, prd, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    return mf1, dict(zip(CLASSES, pcf)), cm_df


# ------------------------------------------------------------------
# Step 2: architecture sweep
# ------------------------------------------------------------------
print("\n=== Step 2: Architecture sweep (LOO-cell) ===\n")

archs = {
    "Logistic-L2":    (lambda: LogisticRegression(C=1.0, max_iter=2000,
                                                   solver="lbfgs", random_state=42), True),
    "kNN-k3":         (lambda: KNeighborsClassifier(n_neighbors=3, metric="euclidean"), True),
    "DecTree-depth2": (lambda: DecisionTreeClassifier(max_depth=2, random_state=42), False),
    "DecTree-depth3": (lambda: DecisionTreeClassifier(max_depth=3, random_state=42), False),
}

sweep  = {}
stored = {}

for name, (fn, scale) in archs.items():
    print(f"--- {name} ---")
    preds = loo_cell_cv(fn, X_raw, y, scale=scale)
    mf1, pcf, _ = metrics(preds)
    sweep[name]  = mf1
    stored[name] = preds
    flag = "PASS" if mf1 >= PRE_REG_THRESHOLD else "BELOW"
    print(f"  macro-F1={mf1:.4f}  [{flag}]  per-class={pcf}\n")

print("=== Sweep summary ===")
for nm, sc in sorted(sweep.items(), key=lambda x: -x[1]):
    flag = "PASS" if sc >= PRE_REG_THRESHOLD else "BELOW"
    print(f"  {nm:<20}: {sc:.4f}  [{flag}]")

best_name  = max(sweep, key=sweep.__getitem__)
best_preds = stored[best_name]
print(f"\nPICKED: {best_name}  macro-F1={sweep[best_name]:.4f}")

# ------------------------------------------------------------------
# Step 3: canonical LOO-cell result
# ------------------------------------------------------------------
print("\n=== Step 3: Canonical LOO-cell result ===")
mf1_cell, pcf_cell, cm_df = metrics(best_preds)
print(f"Macro-F1  : {mf1_cell:.4f}")
print(f"Threshold : {PRE_REG_THRESHOLD}")
print(f"Baseline  : {BASELINE_MF1}")
print(f"Uplift    : {mf1_cell - BASELINE_MF1:+.4f}")
print(f"PRE-REG   : {'PASS' if mf1_cell >= PRE_REG_THRESHOLD else 'NULL (BELOW THRESHOLD)'}")
print("\nPer-class F1:")
for cls, val in pcf_cell.items():
    print(f"  {cls:<15}: {val:.4f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(cm_df.to_string())

# save predictions
pred_rows = [{"fold": i+1,
               "dataset":   fm.dataset.iloc[i],
               "horizon":   fm.horizon.iloc[i],
               "actual":    a, "predicted": p,
               "correct":   int(a == p)}
             for i, (a, p) in enumerate(best_preds)]
pd.DataFrame(pred_rows).to_csv(os.path.join(DATA_DIR, "f1_router_predictions.csv"), index=False)
print(f"\nSaved: {DATA_DIR}/f1_router_predictions.csv")

res_rows = [{"class": c, "f1": pcf_cell[c]} for c in CLASSES]
res_rows.append({"class": "MACRO", "f1": mf1_cell})
res_df = pd.DataFrame(res_rows)
res_df["architecture"] = best_name
res_df["pre_reg_pass"] = mf1_cell >= PRE_REG_THRESHOLD
res_df["baseline_mf1"] = BASELINE_MF1
res_df["uplift"]       = round(mf1_cell - BASELINE_MF1, 4)
res_df.to_csv(os.path.join(DATA_DIR, "f1_loo_cell_results.csv"), index=False)
print(f"Saved: {DATA_DIR}/f1_loo_cell_results.csv")

# ------------------------------------------------------------------
# Step 4: LOO-dataset CV (post-hoc, 3 folds)
# ------------------------------------------------------------------
print("\n=== Step 4: LOO-dataset CV (3 folds, post-hoc) ===\n")

use_scale = best_name in ("Logistic-L2", "kNN-k3")
clf_fn    = archs[best_name][0]
loo_ds_preds = []

for held_ds in ["Alibaba", "Bitbrains", "ByteDance"]:
    test_mask = (datasets_arr == held_ds)
    Xtr, Xte  = X_raw[~test_mask], X_raw[test_mask]
    ytr, yte  = y[~test_mask], y[test_mask]
    hz_te     = fm.horizon.values[test_mask]
    if use_scale:
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    clf = clf_fn(); clf.fit(Xtr, ytr)
    y_preds = clf.predict(Xte)
    print(f"  Held-out: {held_ds}")
    for hz, act, prd in zip(hz_te, yte, y_preds):
        ok = "OK" if act == prd else "--"
        print(f"    {ok}  {hz:<6}  actual={act:<13}  pred={prd}")
    loo_ds_preds.extend(zip(yte, y_preds))

mf1_ds, _, _ = metrics(loo_ds_preds)
print(f"\nLOO-dataset macro-F1 : {mf1_ds:.4f}")
print(f"LOO-cell macro-F1    : {mf1_cell:.4f}")
print(f"Gap (cell - dataset) : {mf1_cell - mf1_ds:+.4f}")

loo_ds_rows = [{"actual": a, "predicted": p, "correct": int(a==p)} for a,p in loo_ds_preds]
loo_ds_df = pd.DataFrame(loo_ds_rows)
loo_ds_df["macro_f1"] = mf1_ds
loo_ds_df.to_csv(os.path.join(DATA_DIR, "f1_loo_dataset_results.csv"), index=False)
print(f"Saved: {DATA_DIR}/f1_loo_dataset_results.csv")

# ------------------------------------------------------------------
# Step 5: dual-baseline comparison
# ------------------------------------------------------------------
print("\n=== Step 5: Dual-baseline comparison ===\n")

ds_ohe   = OneHotEncoder(sparse_output=False, drop="first")
ds_enc   = ds_ohe.fit_transform(datasets_arr.reshape(-1, 1))
X_struct = np.hstack([ds_enc, X_raw[:, 1:2]])   # 2 dummies + horizon_min
print(f"Structural X shape: {X_struct.shape}  (2 dataset dummies + horizon_min)")

struct_preds = loo_cell_cv(
    lambda: DecisionTreeClassifier(max_depth=2, random_state=42),
    X_struct, y, scale=False
)
mf1_struct, _, _ = metrics(struct_preds)

print(f"\nDual-baseline summary:")
print(f"  Trivial (always Chronos-2)           : {BASELINE_MF1:.4f}")
print(f"  Structural (dataset + horizon only)  : {mf1_struct:.4f}  uplift={mf1_struct - BASELINE_MF1:+.4f}")
print(f"  Full F1 router (4 features)          : {mf1_cell:.4f}  uplift={mf1_cell - BASELINE_MF1:+.4f}")
print(f"  Structural -> Full delta             : {mf1_cell - mf1_struct:+.4f}")

dual_df = pd.DataFrame([
    {"baseline": "trivial_always_chronos2",        "macro_f1": BASELINE_MF1,
     "uplift_from_trivial": 0.0},
    {"baseline": "structural_dataset_horizon_only", "macro_f1": mf1_struct,
     "uplift_from_trivial": round(mf1_struct - BASELINE_MF1, 4)},
    {"baseline": "full_f1_router_4features",        "macro_f1": mf1_cell,
     "uplift_from_trivial": round(mf1_cell - BASELINE_MF1, 4)},
])
dual_df["architecture"]    = best_name
dual_df["loo_dataset_mf1"] = mf1_ds
dual_df.to_csv(os.path.join(DATA_DIR, "f1_dual_baseline_comparison.csv"), index=False)
print(f"Saved: {DATA_DIR}/f1_dual_baseline_comparison.csv")

# ------------------------------------------------------------------
# Step 6: FINAL block — paste this back
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("F1 ROUTER — FINAL CONSOLIDATED RESULTS")
print("=" * 60)
print(f"Architecture          : {best_name}")
print(f"LOO-cell macro-F1     : {mf1_cell:.4f}")
print(f"Pre-reg threshold     : {PRE_REG_THRESHOLD}")
print(f"Pre-reg outcome       : {'PASS' if mf1_cell >= PRE_REG_THRESHOLD else 'NULL (BELOW THRESHOLD)'}")
print(f"LOO-dataset macro-F1  : {mf1_ds:.4f}")
print(f"Structural baseline   : {mf1_struct:.4f}")
print(f"Trivial baseline      : {BASELINE_MF1:.4f}")
print(f"Full vs structural    : {mf1_cell - mf1_struct:+.4f}")
print()
print("Architecture sweep:")
for nm, sc in sorted(sweep.items(), key=lambda x: -x[1]):
    flag = "PASS" if sc >= PRE_REG_THRESHOLD else "BELOW"
    print(f"  {nm:<20}: {sc:.4f}  [{flag}]")
print()
print("Per-class F1 (canonical):")
for cls, val in pcf_cell.items():
    print(f"  {cls:<15}: {val:.4f}")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print(cm_df.to_string())
print("\nf1_router.py DONE")
