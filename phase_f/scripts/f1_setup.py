"""
f1_setup.py  —  D11 Task B
Build 12x4 feature matrix + honest structural inspection.

Place in:  phase_f/
Run:       python3 f1_setup.py

Outputs:
    phase_f/data/f1_feature_matrix.csv
"""

import os
import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank
from scipy.stats import spearmanr

# Self-locate: this script lives in phase_f/
PHASE_F_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_ROOT = os.path.dirname(PHASE_F_DIR)
OUT_DIR     = os.path.join(PHASE_F_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"PHASE_F_DIR : {PHASE_F_DIR}")
print(f"THESIS_ROOT : {THESIS_ROOT}")
print(f"OUT_DIR     : {OUT_DIR}")

# ------------------------------------------------------------------
# Step 1: omega_summary.csv at thesis root
# ------------------------------------------------------------------
print("\n=== Step 1: omega_summary.csv ===")
omega_path = os.path.join(THESIS_ROOT, "omega_summary.csv")
omega = pd.read_csv(omega_path)
print(omega.to_string())

dataset_features = {}
for _, row in omega.iterrows():
    ds = row["Dataset"]
    if "ByteDance" in ds or "IaaS" in ds:
        ds = "ByteDance"
    dataset_features[ds] = {
        "acf_24h": float(row["ACF@24h"]),
        "acf_1h":  float(row["ACF@1h"]),
        "cv":      float(row["CV_median"]),
    }

print("\nDataset feature lookup:")
for ds, f in dataset_features.items():
    print(f"  {ds}: ACF@24h={f['acf_24h']}, ACF@1h={f['acf_1h']}, CV={f['cv']}")

# ------------------------------------------------------------------
# Step 2: per-cell winner labels
# Canonical leaderboard_v1 tally: C2=6, TFM=3, Granite-TTM=2, NNLS=1
# Granite-TTM wins Alibaba h30 + h60 — only assignment satisfying the tally.
# ------------------------------------------------------------------
print("\n=== Step 2: Winner labels ===")

LABELS = {
    ("Alibaba",   "10min"):  "NNLS",
    ("Alibaba",   "30min"):  "Granite-TTM",
    ("Alibaba",   "60min"):  "Granite-TTM",
    ("Alibaba",   "120min"): "TimesFM",
    ("Bitbrains", "10min"):  "Chronos-2",
    ("Bitbrains", "30min"):  "Chronos-2",
    ("Bitbrains", "60min"):  "TimesFM",
    ("Bitbrains", "120min"): "TimesFM",
    ("ByteDance", "10min"):  "Chronos-2",
    ("ByteDance", "30min"):  "Chronos-2",
    ("ByteDance", "60min"):  "Chronos-2",
    ("ByteDance", "120min"): "Chronos-2",
}

HORIZON_MAP = {"10min": 10, "30min": 30, "60min": 60, "120min": 120}

tally = {}
for lbl in LABELS.values():
    tally[lbl] = tally.get(lbl, 0) + 1
print("Tally:", tally)
assert tally == {"NNLS": 1, "Granite-TTM": 2, "TimesFM": 3, "Chronos-2": 6}, \
    f"TALLY MISMATCH: {tally}"
print("Tally assertion OK")

# ------------------------------------------------------------------
# Step 3: build feature matrix
# ------------------------------------------------------------------
print("\n=== Step 3: Feature matrix (12 rows x 4 features) ===")

rows = []
for (ds, hz), lbl in LABELS.items():
    f = dataset_features[ds]
    rows.append({
        "dataset":     ds,
        "horizon":     hz,
        "acf_24h":     f["acf_24h"],
        "horizon_min": HORIZON_MAP[hz],
        "cv":          f["cv"],
        "acf_1h":      f["acf_1h"],
        "label":       lbl,
    })

fm = pd.DataFrame(rows)
print(fm.to_string(index=False))

X = fm[["acf_24h", "horizon_min", "cv", "acf_1h"]].values
feature_names = ["acf_24h", "horizon_min", "cv", "acf_1h"]

# ------------------------------------------------------------------
# Step 4: structural inspection
# ------------------------------------------------------------------
print("\n=== Step 4: Structural inspection ===")
print(f"Shape : {X.shape}")
print(f"Rank  : {matrix_rank(X)}")

print("\nPer-column distinct values and variance:")
for i, name in enumerate(feature_names):
    uniq = np.unique(X[:, i])
    print(f"  {name:<12}  var={np.var(X[:,i]):.6f}  n_distinct={len(uniq)}  vals={uniq}")

print("\nPearson correlation matrix:")
print(pd.DataFrame(X, columns=feature_names).corr(method="pearson").round(4).to_string())

print("\nSpearman correlation matrix:")
sp = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        rho, _ = spearmanr(X[:, i], X[:, j])
        sp[i, j] = rho
print(pd.DataFrame(sp, index=feature_names, columns=feature_names).round(4).to_string())

# ------------------------------------------------------------------
# Step 5: save
# ------------------------------------------------------------------
out_path = os.path.join(OUT_DIR, "f1_feature_matrix.csv")
fm.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("\nf1_setup.py DONE")
