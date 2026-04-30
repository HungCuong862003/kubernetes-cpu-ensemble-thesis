"""
bcf_generalization.py
---------------------
Does the BCF predicate  ( ACF@24h > 0.2 )  AND  ( horizon >= 30 min )
still separate "ML beats naive" from "ML doesn't" when we swap the
predictor? The predicate is a statement about the DATA (autocorrelation
is a workload property), so it shouldn't care which model is doing the
predicting. This script tests that.

Four "models" scored (Naive is used as a control row only):
  1. Naive       - control row (delta = 0 by construction, excluded from AUC)
  2. NNLS        - our Hetero NNLS ensemble; MIXED aggregation, see note below
  3. Chronos-2   - zero-shot foundation model, pooled subsample scoring
  4. TimesFM     - zero-shot foundation model, pooled subsample scoring

IMPORTANT: NNLS aggregation choice for Bitbrains
------------------------------------------------
The thesis's published table (boundary_condition_table_corrected.csv) reports
Bitbrains 120min as "+3.30pp" which comes from median(ml_r2 - naive_r2)
across 156 VMs. Pooled scoring on Bitbrains gives huge negative R² because
a handful of catastrophic VMs (ml_r2 goes negative) drag the pool down. To
stay consistent with the published narrative, NNLS uses:
    - Alibaba / ByteDance: pooled (ensemble_r2 - naive_r2)
    - Bitbrains:           median of per-VM deltas
Both choices are defensible. Flip NNLS_BITBRAINS_AGG to "pooled" to test
the alternative; result: NNLS Bitbrains-120min flips from win to loss and
pooled AUC shifts by ~0.03.

Fair delta (each model evaluated against the naive scored on the SAME
set of points as the model itself):
  NNLS Alibaba/ByteDance (ensemble_r2 - naive_r2) pooled  from chronos_benchmark_results.csv
  NNLS Bitbrains         median(ml_r2 - naive_r2)         from bitbrains_per_vm.csv
  Chronos-2              (r2_mean - r2_naive_subsample)   from k20_{ds}_v2.json
  TimesFM                (r2_mean - r2_naive_subsample)   from timesfm_*.json

For each (model, dataset, horizon) pair:
  predicate = 1 if (ACF@24h > 0.2) AND (horizon_min >= 30), else 0
  label     = 1 if delta_pp > 0, else 0

Then:
  - Per-model AUC
  - Pooled AUC across NNLS + Chronos-2 + TimesFM (3 x 3 x 4 = 36 pairs)
  - BCa bootstrap 95% CI on pooled AUC
  - Permutation p-value (label-shuffle, Phipson-Smyth convention)

Memory values have been wrong before. Everything is loaded from CSV/JSON
at runtime. No cached constants for R² numbers.

Outputs written to OUT_DIR:
    bcf_pairs.csv              - one row per (model, dataset, horizon)
    bcf_per_model_auc.csv      - AUC for each model individually
    bcf_pooled_results.json    - pooled AUC + BCa CI + permutation p
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

# ------------------------------- config ---------------------------------
# Change DATA_DIR to wherever your thesis files live.
#   Local:  Path("/mnt/project")
#   Colab:  Path("/content/drive/MyDrive/k8s-ensemble-forecast")
DATA_DIR = Path(r"E:\K8S-ENSEMBLE-FORECAST\bcf_stage")
OUT_DIR  = Path(r"E:\K8S-ENSEMBLE-FORECAST\bcf_stage\corrected\bcf_generalization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS    = ["10min", "30min", "60min", "120min"]
HORIZON_MIN = {"10min": 10, "30min": 30, "60min": 60, "120min": 120}

# Expected ACF@24h values (cross-checked against omega_summary.csv at runtime).
ACF_24H_EXPECTED = {
    "Alibaba":   0.316,
    "Bitbrains": 0.116,
    "ByteDance": 0.489,
}

# Predicate thresholds (from the original BCF).
ACF_THRESH     = 0.2
HORIZON_THRESH = 30   # minutes

# NNLS Bitbrains aggregation. Flip to "pooled" to cross-check.
NNLS_BITBRAINS_AGG = "per_vm_median"   # or "pooled"

N_BOOT    = 10_000
N_PERM    = 10_000
RNG_SEED  = 42

rng = np.random.default_rng(RNG_SEED)


# --------------------------- tiny helpers -------------------------------
def predicate(dataset, horizon_label, acf_map):
    """1 if ML expected to add value under BCF, else 0."""
    acf  = acf_map[dataset]
    hmin = HORIZON_MIN[horizon_label]
    return int(acf > ACF_THRESH and hmin >= HORIZON_THRESH)


def safe_auc(y_score, y_label):
    """roc_auc_score blows up if labels are all one class. Return nan."""
    y_score = np.asarray(y_score)
    y_label = np.asarray(y_label)
    if len(np.unique(y_label)) < 2:
        return np.nan
    return float(roc_auc_score(y_label, y_score))


def ds_normalize(name):
    """Map dataset strings from various CSVs into a canonical form."""
    if name.lower().startswith("alibaba"):   return "Alibaba"
    if name.lower().startswith("bitbrains"): return "Bitbrains"
    if name.lower().startswith("bytedance"): return "ByteDance"
    return None


# -------------------- sanity: reload ACF@24h from disk ------------------
# I don't trust hard-coded numbers -- verify against the CSV every run.
print("=== Verifying ACF@24h from omega_summary.csv ===")
omega = pd.read_csv(DATA_DIR / "omega_summary.csv")
print(omega[["Dataset", "ACF@24h"]])

acf_map = {}
for _, r in omega.iterrows():
    name = r["Dataset"]
    key  = "ByteDance" if name.startswith("ByteDance") else name
    acf_map[key] = float(r["ACF@24h"])

for k, v in ACF_24H_EXPECTED.items():
    got = acf_map.get(k)
    if got is None:
        raise RuntimeError(f"Missing ACF@24h for {k} in omega_summary.csv")
    if abs(got - v) > 1e-3:
        print(f"  WARNING: {k} ACF@24h drifted: expected {v}, got {got:.3f}")
    else:
        print(f"  OK: {k} ACF@24h = {got:.3f}")


# =======================================================================
# STEP 1: collect (model, dataset, horizon, delta_pp) rows from every source
# =======================================================================
rows = []


# -------------------- NAIVE (control, delta = 0) -----------------------
print("\n=== Loading NAIVE (control) ===")
for ds in ["Alibaba", "Bitbrains", "ByteDance"]:
    for h in HORIZONS:
        rows.append({
            "model":    "Naive",
            "dataset":  ds,
            "horizon":  h,
            "r2_model": np.nan,
            "r2_naive": np.nan,
            "delta_pp": 0.0,
        })
print("  12 rows emitted (delta = 0 everywhere)")


# -------------------- NNLS HETERO ENSEMBLE (mixed aggregation) ----------
# Alibaba / ByteDance: pooled full-test delta from chronos_benchmark_results.csv
# Bitbrains:           median of per-VM deltas from bitbrains_per_vm.csv
#                      (matches boundary_condition_table_corrected.csv)
print(f"\n=== Loading NNLS Hetero Ensemble (NNLS_BITBRAINS_AGG={NNLS_BITBRAINS_AGG}) ===")
cb = pd.read_csv(DATA_DIR / "chronos_benchmark_results.csv")
bb_vm = pd.read_csv(DATA_DIR / "bitbrains_per_vm.csv")
print(f"  chronos_benchmark_results.csv has {len(cb)} rows")
print(f"  bitbrains_per_vm.csv has {len(bb_vm)} rows, {bb_vm['vm_id'].nunique()} unique VMs")

for _, r in cb.iterrows():
    ds = ds_normalize(r["dataset"])
    if ds is None:
        print(f"  SKIP unknown dataset: {r['dataset']}")
        continue
    h = r["horizon"]
    if h not in HORIZONS:
        continue

    if ds == "Bitbrains" and NNLS_BITBRAINS_AGG == "per_vm_median":
        # swap in per-VM median
        sub = bb_vm[bb_vm["horizon"] == h]
        med_ml       = float(sub["ml_r2"].median())
        med_naive    = float(sub["naive_r2"].median())
        # median of per-VM deltas, not delta of medians (these differ)
        med_delta_pp = float(sub["r2_delta"].median()) * 100.0
        rows.append({
            "model":    "NNLS",
            "dataset":  ds,
            "horizon":  h,
            "r2_model": med_ml,
            "r2_naive": med_naive,
            "delta_pp": med_delta_pp,
        })
        tag = "per-VM median"
        print(f"  {ds:10s} {h:6s}  naive={med_naive:+.4f}  ens={med_ml:+.4f}  "
              f"Δ={med_delta_pp:+7.2f}pp  ({tag})")
    else:
        naive_r2    = float(r["naive_r2"])
        ensemble_r2 = float(r["ensemble_r2"])
        delta       = (ensemble_r2 - naive_r2) * 100.0
        rows.append({
            "model":    "NNLS",
            "dataset":  ds,
            "horizon":  h,
            "r2_model": ensemble_r2,
            "r2_naive": naive_r2,
            "delta_pp": delta,
        })
        tag = "pooled"
        print(f"  {ds:10s} {h:6s}  naive={naive_r2:+.4f}  ens={ensemble_r2:+.4f}  "
              f"Δ={delta:+7.2f}pp  ({tag})")


# -------------------- CHRONOS-2 (zero-shot) -----------------------------
# Fair delta: r2_mean - r2_naive_subsample (same subsample used for both).
print("\n=== Loading Chronos-2 (zero-shot) ===")
chronos_files = {
    "Alibaba":   "k20_alibaba_v2.json",
    "Bitbrains": "k20_bitbrains_v2.json",
    "ByteDance": "k20_bytedance_v2.json",
}
for ds, fname in chronos_files.items():
    with open(DATA_DIR / fname) as f:
        j = json.load(f)
    for h in HORIZONS:
        blk = j["chronos2_zero_shot"][h]
        r2_m    = float(blk["r2_mean"])
        r2_subn = float(blk["r2_naive_subsample"])
        delta   = (r2_m - r2_subn) * 100.0
        rows.append({
            "model":    "Chronos-2",
            "dataset":  ds,
            "horizon":  h,
            "r2_model": r2_m,
            "r2_naive": r2_subn,
            "delta_pp": delta,
        })
        print(f"  {ds:10s} {h:6s}  sub_naive={r2_subn:+.4f}  chronos={r2_m:+.4f}  Δ={delta:+7.2f}pp")


# -------------------- TIMESFM (zero-shot) -------------------------------
# Alibaba was k20, Bitbrains and ByteDance at k50. Doesn't matter for AUC,
# all we need is (r2_mean, r2_naive_subsample) per (ds, h).
print("\n=== Loading TimesFM (zero-shot) ===")
timesfm_files = {
    "Alibaba":   "timesfm_k20_alibaba.json",
    "Bitbrains": "timesfm_k50_bitbrains.json",
    "ByteDance": "timesfm_k50_bytedance.json",
}
for ds, fname in timesfm_files.items():
    with open(DATA_DIR / fname) as f:
        j = json.load(f)
    for h in HORIZONS:
        blk = j["timesfm_zero_shot"][h]
        r2_m    = float(blk["r2_mean"])
        r2_subn = float(blk["r2_naive_subsample"])
        delta   = (r2_m - r2_subn) * 100.0
        rows.append({
            "model":    "TimesFM",
            "dataset":  ds,
            "horizon":  h,
            "r2_model": r2_m,
            "r2_naive": r2_subn,
            "delta_pp": delta,
        })
        print(f"  {ds:10s} {h:6s}  sub_naive={r2_subn:+.4f}  timesfm={r2_m:+.4f}  Δ={delta:+7.2f}pp")


# =======================================================================
# STEP 2: compute predicate and label for each row
# =======================================================================
df = pd.DataFrame(rows)
df["predicate"] = df.apply(lambda r: predicate(r["dataset"], r["horizon"], acf_map), axis=1)
df["ml_wins"]   = (df["delta_pp"] > 0).astype(int)

print(f"\n=== All pairs loaded: {len(df)} total rows ===")
print(df.groupby("model").size().to_string())

pairs_out = OUT_DIR / "bcf_pairs.csv"
df.to_csv(pairs_out, index=False)
print(f"  wrote {pairs_out}")


# =======================================================================
# STEP 3: per-model AUC (note: "predicate_label_match" is NOT model accuracy)
# =======================================================================
print("\n=== Per-model AUC ===")
per_model_results = []
for model in ["Naive", "NNLS", "Chronos-2", "TimesFM"]:
    sub = df[df["model"] == model]
    if len(sub) == 0:
        print(f"  {model}: no rows, skipping")
        continue
    auc = safe_auc(sub["predicate"].values, sub["ml_wins"].values)
    n_total    = len(sub)
    n_pos      = int(sub["ml_wins"].sum())
    n_pred_pos = int(sub["predicate"].sum())
    match_rate = float((sub["predicate"] == sub["ml_wins"]).mean())
    per_model_results.append({
        "model":                  model,
        "n_pairs":                n_total,
        "n_ml_wins":              n_pos,
        "n_predicate_pos":        n_pred_pos,
        "auc":                    auc,
        "predicate_label_match":  match_rate,   # NOT model forecasting accuracy
    })
    auc_s = f"{auc:.3f}" if not np.isnan(auc) else "   n/a (degenerate labels)"
    print(f"  {model:12s}  n={n_total:3d}  ml_wins={n_pos:3d}  pred+={n_pred_pos:3d}  "
          f"AUC={auc_s}  match={match_rate:.3f}")

pm_df  = pd.DataFrame(per_model_results)
pm_out = OUT_DIR / "bcf_per_model_auc.csv"
pm_df.to_csv(pm_out, index=False)
print(f"  wrote {pm_out}")


# =======================================================================
# STEP 4: pooled AUC + BCa bootstrap CI + permutation p-value
# =======================================================================
# Pool everything except Naive. Naive rows have ml_wins=0 everywhere by
# construction (delta = 0), so they give no signal about the predicate.
#
# INDEPENDENCE CAVEAT: 36 pairs = 12 (dataset, horizon) cells observed
# under 3 models. Proper cluster-aware inference would block-bootstrap
# on (dataset, horizon), giving 12 clusters. We report pair-level CI/p
# here; disclose in chapter that this overstates precision.
print("\n=== Pooled AUC (excluding Naive) ===")
pool = df[df["model"] != "Naive"].copy()
y_score = pool["predicate"].values.astype(float)
y_label = pool["ml_wins"].values.astype(int)
print(f"  n_pooled = {len(pool)}  ml_wins = {int(y_label.sum())}  pred+ = {int(y_score.sum())}")

pooled_auc = safe_auc(y_score, y_label)
print(f"  pooled AUC = {pooled_auc:.4f}")


# ---- BCa bootstrap 95% CI ----
print(f"\n=== BCa bootstrap CI (n_boot={N_BOOT}) ===")
n = len(y_label)
idx_all = np.arange(n)
boot_aucs = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    boot_aucs[b] = safe_auc(y_score[idx], y_label[idx])
    if b < 3 or (b + 1) % 2500 == 0:
        print(f"    boot iter {b+1:5d}: auc = {boot_aucs[b]}")

boot_valid = boot_aucs[~np.isnan(boot_aucs)]
print(f"  {len(boot_valid)}/{N_BOOT} bootstrap draws valid (rest had single-class labels)")

# jackknife for BCa acceleration a_hat
jack_aucs = np.empty(n)
for i in range(n):
    mask = idx_all != i
    jack_aucs[i] = safe_auc(y_score[mask], y_label[mask])
jack_valid = jack_aucs[~np.isnan(jack_aucs)]
jack_mean  = jack_valid.mean()

num = np.sum((jack_mean - jack_valid) ** 3)
den = 6.0 * (np.sum((jack_mean - jack_valid) ** 2) ** 1.5)
a_hat = num / den if den > 0 else 0.0

# bias-correction z0
frac_below = float((boot_valid < pooled_auc).mean())
if 0.0 < frac_below < 1.0:
    z0 = norm.ppf(frac_below)
else:
    z0 = 0.0
    print(f"  NOTE: frac_below={frac_below:.3f} -> z0 set to 0 (bootstrap distribution degenerate)")

alpha  = 0.05
z_lo   = norm.ppf(alpha / 2)
z_hi   = norm.ppf(1 - alpha / 2)
lo_pct = norm.cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
hi_pct = norm.cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))
ci_lo  = float(np.quantile(boot_valid, lo_pct))
ci_hi  = float(np.quantile(boot_valid, hi_pct))
print(f"  BCa 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]   (z0={z0:+.3f}, a_hat={a_hat:+.4f})")
print(f"  Rounded for reporting: [{ci_lo:.2f}, {ci_hi:.2f}]")


# ---- Permutation test (Phipson-Smyth (count+1)/(n+1) convention) ----
# H0: predicate is uninformative about ml_wins.
# The +1 correction means p is never exactly 0; floor is 1/(n_perm+1).
print(f"\n=== Permutation test (n_perm={N_PERM}, Phipson-Smyth convention) ===")
perm_aucs = np.empty(N_PERM)
for b in range(N_PERM):
    y_shuf = rng.permutation(y_label)
    perm_aucs[b] = safe_auc(y_score, y_shuf)
    if b < 3 or (b + 1) % 2500 == 0:
        print(f"    perm iter {b+1:5d}: auc = {perm_aucs[b]}")

perm_valid  = perm_aucs[~np.isnan(perm_aucs)]
n_ge        = int((perm_valid >= pooled_auc).sum())
p_value     = (n_ge + 1) / (len(perm_valid) + 1)
print(f"  permutation p-value = {p_value:.4f}   (count_ge={n_ge}, n_valid={len(perm_valid)})")


# =======================================================================
# STEP 5: save pooled results
# =======================================================================
pooled_results = {
    "predicate":            f"(ACF@24h > {ACF_THRESH}) AND (horizon >= {HORIZON_THRESH}min)",
    "nnls_bitbrains_agg":   NNLS_BITBRAINS_AGG,
    "acf_24h_used":         {k: float(v) for k, v in acf_map.items()},
    "n_pooled_pairs":       int(len(pool)),
    "n_ml_wins":            int(y_label.sum()),
    "n_predicate_pos":      int(y_score.sum()),
    "pooled_auc":           float(pooled_auc),
    "bca_95_ci_lo":         float(ci_lo),
    "bca_95_ci_hi":         float(ci_hi),
    "bca_z0":               float(z0),
    "bca_a_hat":            float(a_hat),
    "permutation_p":        float(p_value),
    "permutation_method":   "Phipson-Smyth (count_ge + 1) / (n_perm + 1)",
    "n_boot":               int(N_BOOT),
    "n_perm":               int(N_PERM),
    "excluded":             ["Naive (all ml_wins=0 by construction)"],
    "caveats": [
        ("NNLS and foundation models use different naive references (NNLS: "
         "full-test pooled naive from chronos_benchmark_results.csv; Chronos-2 "
         "and TimesFM: k20/k50 same-subsample naive from sanity JSONs). Each "
         "delta is internally consistent (model vs naive on identical points) "
         "but reference sets differ across models."),
        ("NNLS Bitbrains uses median of per-VM deltas (NNLS_BITBRAINS_AGG="
         "'per_vm_median') to match boundary_condition_table_corrected.csv. "
         "Pooled scoring flips Bitbrains-120min from win to loss and shifts "
         "pooled AUC by ~0.03."),
        ("36 pooled pairs = 12 (dataset, horizon) cells under 3 models. "
         "Pairs are not iid; CI and p-value are pair-level and overstate "
         "precision relative to cluster-aware inference at n=12."),
        ("BCa CI precision is illusory beyond 2 decimals; ~200 unique AUC "
         "values in the bootstrap distribution. Report CI rounded."),
    ],
}
pooled_out = OUT_DIR / "bcf_pooled_results.json"
with open(pooled_out, "w") as f:
    json.dump(pooled_results, f, indent=2)
print(f"\n  wrote {pooled_out}")


# =======================================================================
# STEP 6: summary to stdout
# =======================================================================
print("\n" + "=" * 62)
print("BCF generalization summary")
print("=" * 62)
print(f"  NNLS_BITBRAINS_AGG = {NNLS_BITBRAINS_AGG}")
print("  Per-model AUC:")
for r in per_model_results:
    auc_s = f"{r['auc']:.3f}" if not np.isnan(r['auc']) else "n/a"
    print(f"    {r['model']:12s}  AUC = {auc_s}   n = {r['n_pairs']:3d}   "
          f"ml_wins = {r['n_ml_wins']}   pred+ = {r['n_predicate_pos']}")
print(f"\n  Pooled AUC (excl. Naive):  {pooled_auc:.3f}   (n = {len(pool)})")
print(f"  BCa 95% CI:                [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"  Permutation p-value:       {p_value:.4f}")
print("\nDone.")
