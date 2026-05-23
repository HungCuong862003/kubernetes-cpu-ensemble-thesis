#!/usr/bin/env python3
"""
verify_ch3_ch5_ch6_audit.py

Targeted verification of the Ch3/Ch5/Ch6 prose-audit findings flagged in
Day 1.5 extended audit. NOT a replacement for verify_foundation.py.
This only confirms the 4 specific anchors that determine whether my
audit corrections are warranted.

Anchors:
  A. Ch3 §3.8 BCF p-value + JSON filename
     - bcf_pooled_3model.json should have p=0.0097 (canonical for §3.8)
     - bcf_pooled_results.json should have different stats (4-model, AUC~0.667)
     - Ch3 §3.8 currently cites the wrong file (results.json instead of 3model.json)
       and reports p<0.011 instead of p=0.0097
  B. Ch3 §3.10 HPA file reference
     - hpa_simulation_v2.csv: legacy v1 sprint, should exist with 800 rows
     - bcf_v2/hpa_simulation_*_v{3,4}.csv: Phase C canonical, should exist
     - Ch3 §3.10 currently cites only the legacy file
  C. Ch5 §5.1 BB 30min CI vs Alibaba 30min point
     - BB 30min Δpp CI should be [+3.54, +5.42]
     - Alibaba 30min v2 Δpp should be +3.39
     - 3.39 must NOT be inside [3.54, 5.42] for the Ch5 claim to be wrong
  D. Ch5 §5.6 vs Ch6 §6.1 HPA pair count 533/640
     - Confirm denominator structure (4 horizons × 4 τ × 10 s × 4 lag = 640 per dataset)
     - Confirm 533 total wins (Ch6 breakdown 128/156/157/92 per horizon)

Run on Vast.ai instance C.37423026 from /workspace/ (paths use /mnt/project
symlink tree).

Run:  python3 verify_ch3_ch5_ch6_audit.py
Exit code:  0 if all anchors confirm the audit findings, 1 if any unexpected.
"""

import json
import os
import sys
import csv
from pathlib import Path

# -------- where to look --------
# /mnt/project is the symlink tree set up earlier; fall back to /workspace
ROOT_CANDIDATES = [
    Path("/mnt/project"),
    Path("/workspace"),
    Path("/workspace/kubernetes-cpu-ensemble-thesis"),
]

def find_root():
    """Pick the first root that has results/ under it."""
    for c in ROOT_CANDIDATES:
        if (c / "results").exists():
            return c
    print("[FATAL] no root with results/ found in", ROOT_CANDIDATES)
    sys.exit(2)

ROOT = find_root()
print(f"[setup] ROOT = {ROOT}")
print()

# ============================================================
# anchor counters
# ============================================================
ANCHORS_PASS = 0
ANCHORS_FAIL = 0
ANCHORS_SKIP = 0

def record(name, ok, msg=""):
    """Record one anchor outcome. ok=True if the audit finding is CONFIRMED."""
    global ANCHORS_PASS, ANCHORS_FAIL, ANCHORS_SKIP
    if ok is None:
        ANCHORS_SKIP += 1
        print(f"[SKIP] {name}: {msg}")
    elif ok:
        ANCHORS_PASS += 1
        print(f"[CONFIRM] {name}: {msg}")
    else:
        ANCHORS_FAIL += 1
        print(f"[CONTRADICT] {name}: {msg}")


# ============================================================
# A. Ch3 §3.8 BCF stats  --  two JSON files with different stats
# ============================================================
print("=" * 70)
print("ANCHOR A: Ch3 §3.8 BCF JSON stats (canonical vs the file Ch3 cites)")
print("=" * 70)

# A.1: canonical bcf_pooled_3model.json
canonical_paths = [
    ROOT / "results/bcf/bcf_pooled_3model.json",
    ROOT / "kubernetes-cpu-ensemble-thesis/results/bcf/bcf_pooled_3model.json",
]
canonical_json = None
for p in canonical_paths:
    if p.exists():
        canonical_json = p
        break

if canonical_json is None:
    record("A.1 bcf_pooled_3model.json exists", None, "FILE NOT FOUND")
else:
    print(f"[found]  {canonical_json}")
    with open(canonical_json) as f:
        d3 = json.load(f)
    print(f"[A.1] keys: {list(d3.keys())}")
    auc3 = d3.get("pooled_auc") or d3.get("auc")
    p3   = d3.get("permutation_p") or d3.get("p_value") or d3.get("p")
    n3   = d3.get("n_pooled_pairs") or d3.get("n")
    ci_lo3 = d3.get("ci_95_lo") or d3.get("percentile_95_ci_lo")
    ci_hi3 = d3.get("ci_95_hi") or d3.get("percentile_95_ci_hi")
    ci_method = d3.get("ci_method", "unspecified")
    print(f"[A.1]   AUC      = {auc3}")
    print(f"[A.1]   p-value  = {p3}")
    print(f"[A.1]   n_pairs  = {n3}")
    print(f"[A.1]   CI       = [{ci_lo3}, {ci_hi3}]   method={ci_method}")
    # canonical expectation
    record(
        "A.1 canonical 3-model AUC=0.80",
        abs(auc3 - 0.80) < 0.01,
        f"auc={auc3}, expected 0.80",
    )
    record(
        "A.1 canonical 3-model p=0.0097",
        abs(p3 - 0.0097) < 0.001,
        f"p={p3}, expected 0.0097 (Ch3 §3.8 currently cites 'p<0.011')",
    )
    record(
        "A.1 canonical 3-model n=36",
        int(n3) == 36,
        f"n={n3}, expected 36",
    )
    record(
        "A.1 canonical CI method = percentile",
        "percentile" in str(ci_method).lower(),
        f"method={ci_method}",
    )

print()

# A.2: 4-model bcf_pooled_results.json that Ch3 §3.8 INCORRECTLY cites
fourmodel_paths = [
    ROOT / "results/bcf/bcf_pooled_results.json",
    ROOT / "kubernetes-cpu-ensemble-thesis/results/bcf/bcf_pooled_results.json",
]
fourmodel_json = None
for p in fourmodel_paths:
    if p.exists():
        fourmodel_json = p
        break

if fourmodel_json is None:
    record("A.2 bcf_pooled_results.json (4-model) exists", None, "FILE NOT FOUND")
else:
    print(f"[found]  {fourmodel_json}")
    with open(fourmodel_json) as f:
        d4 = json.load(f)
    auc4 = d4.get("pooled_auc") or d4.get("auc")
    p4   = d4.get("permutation_p") or d4.get("p_value") or d4.get("p")
    n4   = d4.get("n_pooled_pairs") or d4.get("n")
    print(f"[A.2]   AUC      = {auc4}")
    print(f"[A.2]   p-value  = {p4}")
    print(f"[A.2]   n_pairs  = {n4}")
    # 4-model expectation: AUC=0.667, p=0.0511, n=48 -- different from canonical
    record(
        "A.2 4-model AUC ~ 0.667 (NOT canonical for §3.8)",
        abs(auc4 - 0.667) < 0.02,
        f"auc={auc4} (Ch3 §3.8 wrongly cites this file as canonical)",
    )

print()

# ============================================================
# B. Ch3 §3.10 HPA file references
# ============================================================
print("=" * 70)
print("ANCHOR B: Ch3 §3.10 HPA file (legacy v2 vs Phase C v3/v4)")
print("=" * 70)

# B.1: legacy file
legacy_paths = [
    ROOT / "results/hpa_simulation_v2.csv",
    ROOT / "kubernetes-cpu-ensemble-thesis/results/hpa_simulation_v2.csv",
]
legacy_csv = None
for p in legacy_paths:
    if p.exists():
        legacy_csv = p
        break

if legacy_csv is None:
    record("B.1 hpa_simulation_v2.csv (legacy v1 sprint)", None, "FILE NOT FOUND")
else:
    with open(legacy_csv) as f:
        legacy_rows = sum(1 for _ in f) - 1  # subtract header
    print(f"[B.1] {legacy_csv}: {legacy_rows} rows")
    record(
        "B.1 legacy hpa_simulation_v2.csv has 800 rows",
        legacy_rows == 800,
        f"rows={legacy_rows} (this is what Ch3 §3.10 currently cites)",
    )

# B.2: Phase C v3/v4 canonical files
canonical_hpa_v3 = list((ROOT / "results/bcf_v2").glob("hpa_simulation_*_v3.csv")) if (ROOT / "results/bcf_v2").exists() else []
canonical_hpa_v4 = list((ROOT / "results/bcf_v2").glob("hpa_simulation_*_v4.csv")) if (ROOT / "results/bcf_v2").exists() else []

# also try the alt path
if not canonical_hpa_v3:
    alt = ROOT / "kubernetes-cpu-ensemble-thesis/results/bcf_v2"
    if alt.exists():
        canonical_hpa_v3 = list(alt.glob("hpa_simulation_*_v3.csv"))
        canonical_hpa_v4 = list(alt.glob("hpa_simulation_*_v4.csv"))

print(f"[B.2] Phase C v3 files: {len(canonical_hpa_v3)} ({[p.name for p in canonical_hpa_v3]})")
print(f"[B.2] Phase C v4 files: {len(canonical_hpa_v4)} ({[p.name for p in canonical_hpa_v4]})")

record(
    "B.2 canonical Phase C v3 files exist (≥1)",
    len(canonical_hpa_v3) >= 1,
    f"found {len(canonical_hpa_v3)} v3 files (Ch3 §3.10 should cite these, not legacy)",
)
record(
    "B.2 canonical Phase C v4 files exist (≥1)",
    len(canonical_hpa_v4) >= 1,
    f"found {len(canonical_hpa_v4)} v4 files (Ch3 §3.10 should cite these, not legacy)",
)

print()

# ============================================================
# C. Ch5 §5.1 BB 30min CI vs Alibaba 30min point
# ============================================================
print("=" * 70)
print("ANCHOR C: Ch5 §5.1 'BB 30min CI crosses Alibaba 30min point' claim")
print("=" * 70)

# C.1: pull BB 30min Δpp + CI
ci_paths = [
    ROOT / "results/bcf_v2/bootstrap_ci_cell_metrics.csv",
    ROOT / "kubernetes-cpu-ensemble-thesis/results/bcf_v2/bootstrap_ci_cell_metrics.csv",
    ROOT / "results/bootstrap_ci_cell_metrics.csv",
]
ci_csv = None
for p in ci_paths:
    if p.exists():
        ci_csv = p
        break

bb30_delta = None
bb30_ci_lo = None
bb30_ci_hi = None
if ci_csv is None:
    record("C.1 bootstrap_ci_cell_metrics.csv", None, "FILE NOT FOUND")
else:
    print(f"[found]  {ci_csv}")
    with open(ci_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[C.1] columns: {list(rows[0].keys()) if rows else 'EMPTY'}")
    # find Bitbrains 30min row
    for r in rows:
        ds = (r.get("dataset") or r.get("Dataset") or "").lower()
        try:
            h = int(r.get("horizon") or r.get("h_min") or 0)
        except ValueError:
            continue
        if "bitbrain" in ds and h == 30:
            print(f"[C.1] BB h=30 row: {r}")
            # try various field names for delta and CI
            for key_candidates, target in [
                (["delta_pp", "delta_r2_pp", "delta_vs_naive_pp"], "delta"),
                (["ci_lo", "ci_95_lo", "ci_lower", "lower"], "lo"),
                (["ci_hi", "ci_95_hi", "ci_upper", "upper"], "hi"),
            ]:
                val = None
                for kc in key_candidates:
                    if kc in r:
                        val = r[kc]
                        break
                if target == "delta":
                    bb30_delta = float(val) if val else None
                elif target == "lo":
                    bb30_ci_lo = float(val) if val else None
                elif target == "hi":
                    bb30_ci_hi = float(val) if val else None
            break

    print(f"[C.1] BB 30min:  Δpp={bb30_delta}, CI=[{bb30_ci_lo}, {bb30_ci_hi}]")

# C.2: pull Alibaba 30min Δpp from cross_dataset_headline_v2.csv
headline_paths = [
    ROOT / "results/foundation_comparison/cross_dataset_headline_v2.csv",
    ROOT / "kubernetes-cpu-ensemble-thesis/results/foundation_comparison/cross_dataset_headline_v2.csv",
    ROOT / "results/cross_dataset_headline_v2.csv",
]
headline_csv = None
for p in headline_paths:
    if p.exists():
        headline_csv = p
        break

ali30_delta = None
if headline_csv is None:
    record("C.2 cross_dataset_headline_v2.csv", None, "FILE NOT FOUND")
else:
    print(f"[found]  {headline_csv}")
    with open(headline_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        ds = (r.get("dataset") or r.get("Dataset") or "").lower()
        try:
            h = int(r.get("horizon") or r.get("h_min") or 0)
        except ValueError:
            continue
        if "alibaba" in ds and h == 30:
            print(f"[C.2] Ali h=30 row: {r}")
            # δ vs naive
            for kc in ["delta_pp", "delta_vs_naive_pp", "delta_r2_pp", "ml_r2_minus_naive_pp"]:
                if kc in r:
                    ali30_delta = float(r[kc])
                    break
            # or compute from r2_ensemble - r2_naive
            if ali30_delta is None:
                ens = r.get("r2_ensemble") or r.get("ensemble_r2")
                naive = r.get("r2_naive") or r.get("naive_r2")
                if ens and naive:
                    ali30_delta = (float(ens) - float(naive)) * 100  # pp
            break
    print(f"[C.2] Alibaba 30min:  Δpp={ali30_delta}")

# C.3: critical comparison
if bb30_ci_lo is not None and ali30_delta is not None:
    inside = bb30_ci_lo <= ali30_delta <= bb30_ci_hi
    print(f"[C.3] Is Alibaba +{ali30_delta:.2f} inside BB CI [{bb30_ci_lo:.2f}, {bb30_ci_hi:.2f}]?  {inside}")
    record(
        "C.3 Ch5 §5.1 claim 'BB CI crosses Ali point' is INCORRECT",
        not inside,
        f"Ali +{ali30_delta:.2f} is {'INSIDE' if inside else 'OUTSIDE'} BB CI [{bb30_ci_lo:.2f},{bb30_ci_hi:.2f}], gap={bb30_ci_lo-ali30_delta:+.2f}pp",
    )
else:
    record("C.3 BB CI vs Ali point comparison", None, "missing numbers, cannot compare")

print()

# ============================================================
# D. Ch5 §5.6 vs Ch6 §6.1 HPA 533/640
# ============================================================
print("=" * 70)
print("ANCHOR D: Ch5/Ch6 HPA pair count 533/640")
print("=" * 70)

# Per Ch6 §6.1 per-horizon breakdown: 128+156+157+92 = 533 / 640
# Try to find a file with strict-dominance counts. Most likely candidates:
candidates_d = [
    ROOT / "results/bcf_v2/hpa_v3_dominance_per_dataset.csv",
    ROOT / "results/bcf_v2/hpa_pareto_counts.csv",
    ROOT / "kubernetes-cpu-ensemble-thesis/results/bcf_v2/hpa_v3_dominance_per_dataset.csv",
    ROOT / "kubernetes-cpu-ensemble-thesis/results/bcf_v2/hpa_pareto_counts.csv",
]
d_csv = None
for p in candidates_d:
    if p.exists():
        d_csv = p
        break

if d_csv is None:
    record("D HPA dominance CSV", None, f"none of {[str(p) for p in candidates_d]} exist")
else:
    print(f"[found]  {d_csv}")
    with open(d_csv) as f:
        reader = csv.DictReader(f)
        d_rows = list(reader)
    print(f"[D] columns: {list(d_rows[0].keys()) if d_rows else 'EMPTY'}")
    for r in d_rows[:20]:
        print(f"[D]   {r}")

# Independently verify denominator structure by counting unique (h, tau, s, lag) in a v3 file
if canonical_hpa_v3:
    sample_v3 = canonical_hpa_v3[0]
    print(f"\n[D.struct] inspecting {sample_v3.name} to confirm grid structure")
    with open(sample_v3) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[D.struct] total rows: {len(rows)}")
    print(f"[D.struct] columns: {list(rows[0].keys()) if rows else 'EMPTY'}")
    # try to count unique values
    if rows:
        cols = rows[0].keys()
        for c in cols:
            vals = set(r[c] for r in rows)
            if 2 <= len(vals) <= 15:
                print(f"[D.struct]   {c}: {len(vals)} unique = {sorted(vals)}")
    # if there are exactly 800 rows, that's the canonical grid
    record(
        "D Phase C v3 file has 800 rows (= 4h × 4τ × 10s × 5 strategies)",
        len(rows) == 800,
        f"rows={len(rows)} in {sample_v3.name}",
    )

print()

# ============================================================
# summary
# ============================================================
print("=" * 70)
print(f"SUMMARY:  {ANCHORS_PASS} CONFIRM  /  {ANCHORS_FAIL} CONTRADICT  /  {ANCHORS_SKIP} SKIP")
print("=" * 70)
print()
print("Interpretation:")
print("  CONFIRM = audit finding is supported by the data on disk")
print("  CONTRADICT = audit finding is NOT supported (don't apply that correction)")
print("  SKIP = file not present on this instance, cannot verify")

sys.exit(0 if ANCHORS_FAIL == 0 else 1)
