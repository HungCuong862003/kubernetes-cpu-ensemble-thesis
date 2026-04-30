# boundary_condition_stats.py
# Jimmy - thesis sprint - April 2026
#
# Purpose: harden the boundary condition framework with proper statistics.
# The rule we're defending is:
#   predict "ML beats naive" = (ACF@24h > 0.2) AND (horizon >= 30min)
#
# We've been claiming AUC=0.84 on 12 dataset-horizon cases with no CI, no
# p-value, no threshold sensitivity. Committee will destroy us on this.
# This script fixes that.
#
# Outputs (saved under corrected/boundary_stats/):
#   - boundary_eval_12cases.csv    : the 12-row evidence table
#   - boundary_stats_summary.csv   : accuracy, AUC, CIs, p-values
#   - threshold_sensitivity.csv    : AUC for ACF@24h cutoffs in [0.10, 0.30]
#   - boundary_null_dist.pdf       : permutation null distribution figure
#
# Run on Colab. Assumes standard Drive mount at /content/drive.


# -----------------------------------------------------------------------------
# Imports - keep it basic, we're on a deadline
# -----------------------------------------------------------------------------
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display on Colab
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix


# -----------------------------------------------------------------------------
# Config - edit these if paths differ
# -----------------------------------------------------------------------------
# NOTE: mount drive in colab first:
#   from google.colab import drive
#   drive.mount('/content/drive')
#
DRIVE_ROOT = "/content/drive/MyDrive/k8s-ensemble-forecast"

# Input paths (all CSVs - no .npy needed for this script)
PATH_ALIBABA_CMP = os.path.join(DRIVE_ROOT, "results/thesis_figures/comparison_table.csv")
PATH_BB_SUMMARY  = os.path.join(DRIVE_ROOT, "corrected/bitbrains_summary_corrected.csv")
PATH_BYTEDANCE   = os.path.join(DRIVE_ROOT, "results/bytedance/results_summary - Bytedance.csv")
PATH_OMEGA       = os.path.join(DRIVE_ROOT, "results/omega_results/omega_summary.csv")

# Output folder
OUT_DIR = os.path.join(DRIVE_ROOT, "corrected/boundary_stats")
os.makedirs(OUT_DIR, exist_ok=True)

# Analysis knobs
ACF_THRESHOLD   = 0.20       # the rule's ACF@24h cutoff
HORIZON_MIN     = 30         # the rule's horizon cutoff (minutes)
BOOTSTRAP_N     = 10000      # bootstrap iterations for CI
PERMUTATION_N   = 10000      # permutation test iterations
RANDOM_SEED     = 42         # reproducibility

np.random.seed(RANDOM_SEED)


# -----------------------------------------------------------------------------
# Step 1: build the 12-row evidence table
# -----------------------------------------------------------------------------
# We need one row per (dataset, horizon) with:
#   - ACF@24h (from omega_summary, dataset-level so same for all 4 horizons)
#   - delta_pp (hetero ensemble R² minus naive R², in percentage points)
#   - ml_wins (boolean: did ensemble actually beat naive?)
#   - rule_predicts_win (boolean: what our rule says)
#
# Three datasets × 4 horizons = 12 rows total. DSB excluded (abandoned).

print("="*70)
print("STEP 1: Building 12-row evidence table")
print("="*70)

# --- 1a. load ACF@24h per dataset ---
omega = pd.read_csv(PATH_OMEGA)
print("\nomega_summary.csv contents:")
print(omega)

# map dataset names to ACF@24h
# the omega file uses "Alibaba", "Bitbrains", "ByteDance IaaS"
acf_lookup = {}
for _, row in omega.iterrows():
    ds_name = row["Dataset"].strip()
    if "Alibaba" in ds_name:
        acf_lookup["Alibaba"] = float(row["ACF@24h"])
    elif "Bitbrains" in ds_name:
        acf_lookup["Bitbrains"] = float(row["ACF@24h"])
    elif "ByteDance" in ds_name:
        acf_lookup["ByteDance"] = float(row["ACF@24h"])

print(f"\nACF@24h lookup: {acf_lookup}")


# --- 1b. extract Alibaba delta_pp per horizon ---
# comparison_table.csv has a column "Δpp (Hetero vs Naive)"
alibaba = pd.read_csv(PATH_ALIBABA_CMP)
print(f"\nAlibaba comparison_table.csv has {len(alibaba)} rows")
print(alibaba[["Horizon", "Naive R²", "Hetero Ens R²", "Δpp (Hetero vs Naive)"]])

alibaba_rows = []
for _, row in alibaba.iterrows():
    horizon_str = row["Horizon"].strip()
    horizon_min = int(horizon_str.replace("min", ""))
    delta_pp = float(row["Δpp (Hetero vs Naive)"])
    alibaba_rows.append({
        "dataset": "Alibaba",
        "horizon_min": horizon_min,
        "acf_24h": acf_lookup["Alibaba"],
        "naive_r2": float(row["Naive R²"]),
        "ml_r2": float(row["Hetero Ens R²"]),
        "delta_pp": delta_pp,
    })


# --- 1c. extract Bitbrains delta_pp per horizon ---
# bitbrains_summary_corrected.csv has BB_Delta_pp (note: already in pp)
bitbrains = pd.read_csv(PATH_BB_SUMMARY)
print(f"\nBitbrains summary has {len(bitbrains)} rows")
print(bitbrains)

bb_rows = []
for _, row in bitbrains.iterrows():
    horizon_str = row["Horizon"].strip()
    horizon_min = int(horizon_str.replace("min", ""))
    # note: Bitbrains uses median ML R² across VMs (aggregate uses median aggregation
    # per thesis notes), and delta_pp is already in percentage points
    delta_pp = float(row["BB_Delta_pp"])
    bb_rows.append({
        "dataset": "Bitbrains",
        "horizon_min": horizon_min,
        "acf_24h": acf_lookup["Bitbrains"],
        "naive_r2": float(row["BB_Naive_R2"]),
        "ml_r2": float(row["BB_ML_R2_median"]),
        "delta_pp": delta_pp,
    })


# --- 1d. extract ByteDance delta_pp per horizon ---
# this one is trickier - the CSV has raw R² for naive and hetero_ensemble,
# we compute delta_pp ourselves
bytedance = pd.read_csv(PATH_BYTEDANCE)
print(f"\nByteDance results_summary has {len(bytedance)} rows")
# print just the columns we need
print(bytedance[["Horizon", "naive_R2", "hetero_ensemble_R2"]])

bd_rows = []
for _, row in bytedance.iterrows():
    horizon_str = row["Horizon"].strip()
    horizon_min = int(horizon_str.replace("min", ""))
    naive_r2 = float(row["naive_R2"])
    ml_r2 = float(row["hetero_ensemble_R2"])
    delta_pp = (ml_r2 - naive_r2) * 100.0  # convert to pp
    bd_rows.append({
        "dataset": "ByteDance",
        "horizon_min": horizon_min,
        "acf_24h": acf_lookup["ByteDance"],
        "naive_r2": naive_r2,
        "ml_r2": ml_r2,
        "delta_pp": delta_pp,
    })


# --- 1e. concatenate into the 12-row evidence table ---
all_rows = alibaba_rows + bb_rows + bd_rows
df = pd.DataFrame(all_rows)

# sanity check: should be exactly 12 rows
assert len(df) == 12, f"Expected 12 rows, got {len(df)}"
print(f"\nEvidence table has {len(df)} rows. Good.")

# add true label: did ML actually beat naive?
df["ml_wins"] = (df["delta_pp"] > 0).astype(int)

# add rule prediction: does our rule say ML should win?
df["rule_predicts_win"] = (
    (df["acf_24h"] > ACF_THRESHOLD) & (df["horizon_min"] >= HORIZON_MIN)
).astype(int)

# add whether rule was correct
df["rule_correct"] = (df["ml_wins"] == df["rule_predicts_win"]).astype(int)

# reorder columns for readability
col_order = ["dataset", "horizon_min", "acf_24h", "naive_r2", "ml_r2",
             "delta_pp", "ml_wins", "rule_predicts_win", "rule_correct"]
df = df[col_order]

print("\n" + "="*70)
print("12-row evidence table:")
print("="*70)
print(df.to_string(index=False))

# save it
evidence_path = os.path.join(OUT_DIR, "boundary_eval_12cases.csv")
df.to_csv(evidence_path, index=False)
print(f"\nSaved -> {evidence_path}")


# -----------------------------------------------------------------------------
# Step 2: binary classifier metrics + binomial sign test
# -----------------------------------------------------------------------------
# The rule is a binary classifier. For binary classifiers, the primary
# metric is accuracy, not AUC. We report both; AUC comes from using ACF@24h
# as a continuous score (step 3).

print("\n" + "="*70)
print("STEP 2: Binary rule evaluation + binomial test")
print("="*70)

y_true = df["ml_wins"].values
y_rule = df["rule_predicts_win"].values

n_total = len(y_true)
n_correct = int((y_true == y_rule).sum())
accuracy = n_correct / n_total

print(f"\nRule correctly classifies {n_correct}/{n_total} cases (accuracy = {accuracy:.4f})")

# confusion matrix
# rows are true, cols are predicted; [0][0]=TN, [0][1]=FP, [1][0]=FN, [1][1]=TP
cm = confusion_matrix(y_true, y_rule, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")

# sensitivity (recall) = TP / (TP + FN), only defined if there are positives
if (tp + fn) > 0:
    sensitivity = tp / (tp + fn)
else:
    sensitivity = float("nan")
# specificity = TN / (TN + FP), only defined if there are negatives
if (tn + fp) > 0:
    specificity = tn / (tn + fp)
else:
    specificity = float("nan")

print(f"\nSensitivity (recall on ML-wins cases): {sensitivity:.4f}")
print(f"Specificity (recall on naive-wins cases): {specificity:.4f}")

# binomial sign test
# H0: rule has no better than chance accuracy (p = 0.5)
# This is an exact test, no independence assumption beyond the outcomes.
binom_05 = stats.binomtest(n_correct, n_total, p=0.5, alternative="greater")
print(f"\nBinomial test vs p0=0.5 (chance):")
print(f"  exact one-tailed p-value = {binom_05.pvalue:.6f}")

# second null: beats a 60% baseline (more conservative)
binom_06 = stats.binomtest(n_correct, n_total, p=0.6, alternative="greater")
print(f"\nBinomial test vs p0=0.6 (conservative):")
print(f"  exact one-tailed p-value = {binom_06.pvalue:.6f}")


# -----------------------------------------------------------------------------
# Step 3: AUC treating ACF@24h as continuous score
# -----------------------------------------------------------------------------
# AUC quantifies how well ACF@24h *ranks* cases, independent of any threshold.
# This is our defense against the post-hoc threshold criticism - high AUC
# means ACF@24h is predictive regardless of the specific 0.2 cutoff.
#
# Caveat: AUC with ACF@24h alone doesn't incorporate horizon. We also compute
# AUC from a composite score.

print("\n" + "="*70)
print("STEP 3: AUC analysis")
print("="*70)

# check if we actually have both classes (AUC undefined if all one class)
n_pos = int(y_true.sum())
n_neg = n_total - n_pos
print(f"\nClass balance: {n_pos} positives, {n_neg} negatives")

if n_pos == 0 or n_neg == 0:
    print("WARNING: AUC undefined (only one class present)")
    auc_acf = float("nan")
    auc_composite = float("nan")
else:
    # AUC from ACF@24h alone
    auc_acf = roc_auc_score(y_true, df["acf_24h"].values)
    print(f"\nAUC using ACF@24h as score: {auc_acf:.4f}")

    # AUC from a simple composite score that includes horizon.
    # Just normalize both features to [0,1] and sum - not fancy, but
    # interpretable and doesn't overfit (no parameters estimated from data).
    acf_norm = df["acf_24h"].values / df["acf_24h"].max()
    horizon_norm = df["horizon_min"].values / df["horizon_min"].max()
    composite_score = acf_norm + horizon_norm
    auc_composite = roc_auc_score(y_true, composite_score)
    print(f"AUC using (ACF@24h + horizon) composite score: {auc_composite:.4f}")


# -----------------------------------------------------------------------------
# Step 4: stratified BCa bootstrap CI for AUC
# -----------------------------------------------------------------------------
# At n=12 with imbalanced classes, we need stratified resampling - otherwise
# some resamples end up with zero negatives and AUC is undefined.
#
# BCa (bias-corrected accelerated) is the gold standard per Carpenter & Bithell
# (2000), better than percentile for skewed statistics.
#
# Caveat: bootstrap CIs under-cover at small n. Nominal 95% coverage is
# actually ~88-91% in this regime. We state this explicitly in the write-up.

print("\n" + "="*70)
print("STEP 4: Bootstrap CI for AUC (stratified BCa)")
print("="*70)

def stratified_bootstrap_auc(y_true, scores, n_iter=10000, seed=42):
    """
    Stratified bootstrap: resample positives and negatives separately with
    replacement, so every resample has both classes. Returns the array of
    bootstrap AUC estimates.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot_aucs = []
    for i in range(n_iter):
        # resample each stratum with replacement, preserving stratum size
        pos_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pos_sample, neg_sample])

        y_boot = y_true[idx]
        s_boot = scores[idx]

        # if a resample somehow still degenerates (shouldn't with stratified,
        # but be defensive), skip it
        if len(np.unique(y_boot)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_boot, s_boot))

    return np.array(boot_aucs)


def bca_ci(boot_stats, observed_stat, alpha=0.05):
    """
    Bias-corrected accelerated (BCa) confidence interval.
    Efron (1987). Handles skewed sampling distributions better than percentile.
    """
    n_boot = len(boot_stats)
    # bias correction z0
    prop_below = (boot_stats < observed_stat).sum() / n_boot
    # clip to avoid inf when prop is exactly 0 or 1
    prop_below = np.clip(prop_below, 1/(2*n_boot), 1 - 1/(2*n_boot))
    z0 = stats.norm.ppf(prop_below)

    # acceleration: use jackknife-style approximation via the boot sample itself
    # (proper jackknife on small n is very small; we use empirical skewness-based
    # estimate which is common when jackknife is unstable)
    mean_b = np.mean(boot_stats)
    # skewness of bootstrap distribution
    numer = np.sum((mean_b - boot_stats) ** 3)
    denom = 6.0 * (np.sum((mean_b - boot_stats) ** 2) ** 1.5)
    if denom == 0:
        accel = 0.0
    else:
        accel = numer / denom

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    # adjusted percentiles
    alpha_lo = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - accel * (z0 + z_lo)))
    alpha_hi = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - accel * (z0 + z_hi)))

    ci_lo = np.quantile(boot_stats, alpha_lo)
    ci_hi = np.quantile(boot_stats, alpha_hi)
    return ci_lo, ci_hi


# run bootstrap on the ACF@24h score
if not np.isnan(auc_acf):
    print(f"\nRunning {BOOTSTRAP_N} stratified bootstrap iterations for AUC (ACF)...")
    boot_aucs_acf = stratified_bootstrap_auc(y_true, df["acf_24h"].values,
                                             n_iter=BOOTSTRAP_N, seed=RANDOM_SEED)
    print(f"  Got {len(boot_aucs_acf)} valid bootstrap samples")

    ci_lo_acf, ci_hi_acf = bca_ci(boot_aucs_acf, auc_acf, alpha=0.05)
    # also compute simple percentile CI for comparison
    ci_lo_pct = np.quantile(boot_aucs_acf, 0.025)
    ci_hi_pct = np.quantile(boot_aucs_acf, 0.975)

    print(f"\nAUC (ACF@24h) = {auc_acf:.4f}")
    print(f"  BCa 95% CI:       [{ci_lo_acf:.4f}, {ci_hi_acf:.4f}]")
    print(f"  Percentile 95% CI: [{ci_lo_pct:.4f}, {ci_hi_pct:.4f}]")
else:
    ci_lo_acf, ci_hi_acf = float("nan"), float("nan")
    ci_lo_pct, ci_hi_pct = float("nan"), float("nan")

# also bootstrap composite score AUC
if not np.isnan(auc_composite):
    boot_aucs_comp = stratified_bootstrap_auc(y_true, composite_score,
                                              n_iter=BOOTSTRAP_N, seed=RANDOM_SEED+1)
    ci_lo_comp, ci_hi_comp = bca_ci(boot_aucs_comp, auc_composite, alpha=0.05)
    print(f"\nAUC (composite) = {auc_composite:.4f}")
    print(f"  BCa 95% CI: [{ci_lo_comp:.4f}, {ci_hi_comp:.4f}]")
else:
    ci_lo_comp, ci_hi_comp = float("nan"), float("nan")


# -----------------------------------------------------------------------------
# Step 5: permutation test on AUC
# -----------------------------------------------------------------------------
# H0: ACF@24h has no relationship to ML-wins.
# We shuffle labels PERMUTATION_N times and count how often the shuffled AUC
# >= our observed AUC. That ratio is the p-value.
#
# At n=12 with 10:2 imbalance there are only C(12,2)=66 unique label
# permutations, so the p-value is quantized. We still run 10k iterations
# because it's cheap and gives a smoother null histogram for the figure.

print("\n" + "="*70)
print("STEP 5: Permutation test")
print("="*70)

if not np.isnan(auc_acf):
    print(f"\nRunning {PERMUTATION_N} permutations...")
    rng = np.random.default_rng(RANDOM_SEED + 100)
    perm_aucs = []
    acf_scores = df["acf_24h"].values
    for i in range(PERMUTATION_N):
        y_shuffled = rng.permutation(y_true)
        # skip degenerate (all 0 or all 1) - shouldn't happen with permutation
        # but be defensive
        if len(np.unique(y_shuffled)) < 2:
            continue
        perm_aucs.append(roc_auc_score(y_shuffled, acf_scores))
    perm_aucs = np.array(perm_aucs)

    # p-value = fraction of perms with AUC >= observed
    p_perm = (perm_aucs >= auc_acf).sum() / len(perm_aucs)
    print(f"  Observed AUC: {auc_acf:.4f}")
    print(f"  Permutation p-value: {p_perm:.6f}")
    print(f"  (at n=12 with 10 positives, only C(12,2)=66 unique perms exist)")
else:
    p_perm = float("nan")
    perm_aucs = np.array([])


# -----------------------------------------------------------------------------
# Step 6: threshold sensitivity analysis
# -----------------------------------------------------------------------------
# If our findings hold across thresholds in [0.10, 0.30], we're robust to
# the specific choice of 0.20. This is the standard defense against the
# "you picked the threshold after seeing the data" critique.

print("\n" + "="*70)
print("STEP 6: Threshold sensitivity")
print("="*70)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
sens_rows = []

for thresh in thresholds:
    # rule at this threshold
    rule_pred_t = ((df["acf_24h"] > thresh) & (df["horizon_min"] >= HORIZON_MIN)).astype(int).values
    acc_t = (y_true == rule_pred_t).sum() / n_total
    n_correct_t = int((y_true == rule_pred_t).sum())

    # binomial p at this threshold
    binom_t = stats.binomtest(n_correct_t, n_total, p=0.5, alternative="greater")

    sens_rows.append({
        "acf_threshold": thresh,
        "n_correct": n_correct_t,
        "n_total": n_total,
        "accuracy": acc_t,
        "binomial_p": binom_t.pvalue,
    })
    print(f"  threshold={thresh:.2f}: {n_correct_t}/{n_total} correct "
          f"(acc={acc_t:.3f}, p={binom_t.pvalue:.4f})")

sens_df = pd.DataFrame(sens_rows)
sens_path = os.path.join(OUT_DIR, "threshold_sensitivity.csv")
sens_df.to_csv(sens_path, index=False)
print(f"\nSaved -> {sens_path}")


# -----------------------------------------------------------------------------
# Step 7: summary CSV
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 7: Writing summary")
print("="*70)

summary_rows = [
    ("n_total", n_total),
    ("n_positives_ml_wins", n_pos),
    ("n_negatives_naive_wins", n_neg),
    ("rule_correct", n_correct),
    ("rule_accuracy", f"{accuracy:.4f}"),
    ("sensitivity", f"{sensitivity:.4f}"),
    ("specificity", f"{specificity:.4f}"),
    ("binomial_p_vs_0.5", f"{binom_05.pvalue:.6f}"),
    ("binomial_p_vs_0.6", f"{binom_06.pvalue:.6f}"),
    ("auc_acf_only", f"{auc_acf:.4f}"),
    ("auc_acf_bca_lo", f"{ci_lo_acf:.4f}"),
    ("auc_acf_bca_hi", f"{ci_hi_acf:.4f}"),
    ("auc_acf_pct_lo", f"{ci_lo_pct:.4f}"),
    ("auc_acf_pct_hi", f"{ci_hi_pct:.4f}"),
    ("auc_composite", f"{auc_composite:.4f}"),
    ("auc_composite_bca_lo", f"{ci_lo_comp:.4f}"),
    ("auc_composite_bca_hi", f"{ci_hi_comp:.4f}"),
    ("permutation_p", f"{p_perm:.6f}"),
    ("bootstrap_n_iter", BOOTSTRAP_N),
    ("permutation_n_iter", PERMUTATION_N),
]
summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
summary_path = os.path.join(OUT_DIR, "boundary_stats_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved -> {summary_path}")


# -----------------------------------------------------------------------------
# Step 8: null distribution figure
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("STEP 8: Plotting null distribution")
print("="*70)

if len(perm_aucs) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm_aucs, bins=30, color="steelblue", alpha=0.7,
            edgecolor="white", label="permuted labels (null)")
    ax.axvline(auc_acf, color="crimson", linewidth=2,
               label=f"observed AUC = {auc_acf:.3f}")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance (0.5)")
    ax.set_xlabel("AUC under permuted labels")
    ax.set_ylabel("frequency")
    ax.set_title(f"Permutation null distribution of AUC (n={n_total}, "
                 f"{PERMUTATION_N} perms, p={p_perm:.4f})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(OUT_DIR, "boundary_null_dist.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {fig_path}")


# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("DONE. Key numbers for the write-up:")
print("="*70)
print(f"  Rule accuracy:         {n_correct}/{n_total} = {accuracy:.3f}")
print(f"  Binomial p (vs 0.5):   {binom_05.pvalue:.4f}")
print(f"  AUC (ACF@24h):         {auc_acf:.3f}  [95% BCa: {ci_lo_acf:.3f}-{ci_hi_acf:.3f}]")
print(f"  AUC (composite):       {auc_composite:.3f}  [95% BCa: {ci_lo_comp:.3f}-{ci_hi_comp:.3f}]")
print(f"  Permutation p:         {p_perm:.4f}")
print(f"  Threshold sensitivity: see {sens_path}")
print(f"\nAll outputs in: {OUT_DIR}")
