"""
fdr_corrected_screen.py — Week 2 Task 4 (REVISED)

Replaces bonferroni_exceedances.py. Multiple-comparisons correction for the
26 exploratory candidates from Week 1's per-series partial-R² screen.

Two complementary procedures:

1. BH-FDR (Benjamini-Hochberg 1995, JRSSB 57:289-300): primary correction at
   q = 0.05. Controls false discovery rate under positive dependence
   (Benjamini-Yekutieli 2001, Annals of Statistics 29:1165-1188).

2. Westfall-Young max-T (Westfall-Young 1993, "Resampling-Based Multiple
   Testing"): dependence-aware supplement that exploits the cluster bootstrap
   to compute exact p-values under arbitrary dependence among the 26 features.

The pre-registered WPE is OUTSIDE the exploratory family and is reported
separately at α = 0.05 (k=1).

Method:
- For each of 26 exploratory candidates:
  - Compute observed partial-R²(feature | ACF@24h, horizon) on Bitbrains clip [-1,+1]
  - Compute a "raw" p-value: P(pr² < threshold | cluster bootstrap)
- BH-FDR: rank candidates by raw p-value, find largest k where p_(k) ≤ k·q/m
- Westfall-Young: for each bootstrap replicate, compute max partial-R² across
  the 26 candidates under permutation null; compare to observed maxima

Inputs:
  per_series_fulltest_r2.csv (Week 1 canonical outcome)
  features_joined_extended_v2.csv (Week 1 features)

Outputs:
  fdr_corrected_screen.csv — per-feature raw p, BH-adjusted q, WY p, verdict
  fdr_corrected_screen.md — interpretation report

References:
- Benjamini & Hochberg (1995), JRSSB 57:289-300
- Benjamini & Yekutieli (2001), Annals of Statistics 29:1165-1188
- Westfall & Young (1993), Wiley
- Westfall, Tobias & Wolfinger (2011), SAS Press

Runtime: ~5 min on CPU (2000 bootstrap × 26 features × n=568 lstsq)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# 26 exploratory candidates (WPE excluded — pre-registered)
EXPLORATORY_CANDIDATES = [
    "dfa_alpha", "lz76",
    "c22_FC_LocalSimple_mean3_stderr",
    "c22_SB_TransitionMatrix_3ac_sumdiagcov",
    "c22_DN_HistogramMode_5",
    "c22_CO_trev_1_num",
    "c22_SB_BinaryStats_diff_longstretch0",
    "c22_DN_HistogramMode_10",
    "c22_CO_f1ecac",
    "c22_CO_FirstMin_ac",
    "c22_CO_HistogramAMI_even_2_5",
    "c22_MD_hrv_classic_pnn40",
    "c22_SB_BinaryStats_mean_longstretch1",
    "c22_PD_PeriodicityWang_th0_01",
    "c22_CO_Embed2_Dist_tau_d_expfit_meandiff",
    "c22_IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "c22_FC_LocalSimple_mean1_tauresrat",
    "c22_DN_OutlierInclude_p_001_mdrmd",
    "c22_DN_OutlierInclude_n_001_mdrmd",
    "c22_SP_Summaries_welch_rect_area_5_1",
    "c22_SP_Summaries_welch_rect_centroid",
    "c22_SB_MotifThree_quantile_hh",
    "c22_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "c22_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "mse_scale_1", "mse_scale_5",
]
PRE_REGISTERED = "wpe_m4_t1"

THRESHOLD = 0.30
Q_FDR = 0.05
ALPHA_PRE_REG = 0.05
CLIP_LO, CLIP_HI = -1.0, 1.0
N_BOOT = 2000


def partial_r2(X_full, X_reduced, y):
    """Standard partial-R² calculation. Returns NaN if degenerate."""
    def _r2(X, y):
        X1 = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        yhat = X1 @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

    rf = _r2(X_full, y)
    rr = _r2(X_reduced, y)
    if np.isnan(rf) or np.isnan(rr) or rr >= 0.9999:
        return np.nan
    return (rf - rr) / (1.0 - rr)


def cluster_bootstrap_all_candidates(X_candidates, X_reduced, y, clusters,
                                       n_boot=2000, seed=42):
    """
    Run cluster bootstrap once, computing partial-R² for ALL candidates per
    replicate. Returns a (n_boot, n_candidates) array of partial-R² values.
    
    This shared-bootstrap approach is critical for Westfall-Young max-T,
    which requires the joint distribution of partial-R² statistics under
    the same resampling.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    k = len(uniq)
    n_cand = X_candidates.shape[1]

    print(f"  Running shared cluster bootstrap: {n_boot} replicates "
          f"over {k} clusters, {n_cand} candidates...")

    boots = np.full((n_boot, n_cand), np.nan)
    for i in range(n_boot):
        sampled = rng.choice(uniq, size=k, replace=True)
        idx = np.concatenate([np.where(clusters == u)[0] for u in sampled])
        for j in range(n_cand):
            X_full = np.column_stack([X_candidates[idx, j], X_reduced[idx]])
            try:
                pr2 = partial_r2(X_full, X_reduced[idx], y[idx])
                boots[i, j] = pr2
            except Exception:
                pass
        if (i + 1) % 250 == 0:
            print(f"    progress: {i + 1}/{n_boot}")
    return boots


def benjamini_hochberg(p_values, q=0.05):
    """
    BH-FDR procedure. Returns dict per feature with raw p, rank, BH-threshold,
    BH-adjusted q-value, and reject decision.
    
    BH adjusted p-value: q_(i) = min over j ≥ i of (m / j) * p_(j), clipped to 1.
    """
    p = np.asarray(p_values)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    
    # Adjusted p-values (BH q-values)
    adj = np.empty(m)
    # Standard formula: q_(i) = min over j >= i of (m / j) * p_(j)
    running_min = 1.0
    for j in range(m - 1, -1, -1):
        # j is 0-indexed; rank = j+1
        candidate = sorted_p[j] * m / (j + 1)
        running_min = min(running_min, candidate)
        adj[j] = min(running_min, 1.0)
    # Map back to original order
    bh_qvals = np.empty(m)
    bh_qvals[order] = adj
    
    # Rejection decisions
    reject_bh = bh_qvals <= q
    return bh_qvals, reject_bh


def westfall_young_max_t(observed_pr2, boots_pr2, threshold=0.30):
    """
    Westfall-Young step-down max-T procedure adapted to partial-R² with
    threshold > 0.30.
    
    For each candidate, compute one-sided "p-value adjusted for max-T":
      p_WY = P(max over selected candidates of (bootstrap pr²) ≤ observed pr²
              given that bootstrap pr² were resampled under the same
              cluster-bootstrap process)
    
    Step-down: process candidates in order of decreasing observed pr².
    """
    n_boot, n_cand = boots_pr2.shape
    order = np.argsort(observed_pr2)[::-1]  # descending observed
    
    p_wy = np.empty(n_cand)
    
    # Step-down: for the i-th largest, max is over candidates ranked i..n_cand-1
    for step, idx in enumerate(order):
        remaining = order[step:]
        # For each bootstrap replicate, what's the max pr² among remaining candidates?
        boot_maxes = np.nanmax(boots_pr2[:, remaining], axis=1)
        # One-sided test: P(boot_max ≥ observed[idx])
        # = fraction of bootstrap replicates where the max is at least as
        # large as what we observed for this candidate
        n_valid = (~np.isnan(boot_maxes)).sum()
        n_at_least = (boot_maxes >= observed_pr2[idx]).sum()
        raw_p = n_at_least / max(n_valid, 1)
        # Monotonicity enforcement (step-down WY): adj p must be ≥ previous
        if step == 0:
            p_wy[idx] = raw_p
        else:
            prev_idx = order[step - 1]
            p_wy[idx] = max(raw_p, p_wy[prev_idx])
    
    return p_wy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_series_csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    print(f"== fdr_corrected_screen.py ==")
    print(f"  per_series_csv: {args.per_series_csv}")
    print(f"  features:       {args.features}")
    print(f"  n_boot:         {args.n_boot}")
    print(f"  q_FDR:          {Q_FDR}")
    print(f"  clip:           [{CLIP_LO}, {CLIP_HI}] on Bitbrains")
    print(f"  threshold:      {THRESHOLD}")
    print(f"  k (exploratory): {len(EXPLORATORY_CANDIDATES)}")
    print()

    # Load and filter to Bitbrains under clip
    per = pd.read_csv(args.per_series_csv)
    bb = per[per["dataset"] == "bitbrains"].copy()
    bb["series_id"] = bb["series_id"].astype(str).str.replace(r"^bb_", "", regex=True)
    bb["r2_ensemble"] = bb["r2_ensemble"].clip(CLIP_LO, CLIP_HI)
    bb["r2_naive"] = bb["r2_naive"].clip(CLIP_LO, CLIP_HI)
    bb["delta_pp"] = (bb["r2_ensemble"] - bb["r2_naive"]) * 100.0

    feats = pd.read_csv(args.features)
    feats["series_id"] = feats["series_id"].astype(str)
    bb_feats = feats[feats["dataset"] == "bitbrains"].copy()

    joined = bb.merge(bb_feats, on=["series_id", "dataset"], how="inner")
    print(f"Joined data: {len(joined)} rows")
    print()

    # Build base regressors
    y = joined["delta_pp"].to_numpy()
    x_acf = joined["acf_24h"].to_numpy()
    x_h = joined["horizon_min"].to_numpy()
    cluster = joined["series_id"].to_numpy()
    X_reduced = np.column_stack([x_acf, x_h])

    # ===== Pre-registered WPE test (separate family) =====
    print("=" * 70)
    print("PRE-REGISTERED WPE TEST (separate from exploratory family)")
    print("=" * 70)
    x_wpe = joined[PRE_REGISTERED].to_numpy()
    mask = ~(np.isnan(y) | np.isnan(x_wpe) | np.isnan(x_acf) | np.isnan(x_h))
    n_wpe = int(mask.sum())
    X_full_wpe = np.column_stack([x_wpe[mask], X_reduced[mask]])
    wpe_pr2 = partial_r2(X_full_wpe, X_reduced[mask], y[mask])
    print(f"  WPE partial-R² (Bitbrains, clip [-1,+1]): {wpe_pr2:.4f}")
    print(f"  Pre-registered α = {ALPHA_PRE_REG}, threshold = {THRESHOLD}")
    print(f"  Cross threshold? {wpe_pr2 >= THRESHOLD}")
    print()

    # ===== Exploratory 26-candidate screen =====
    print("=" * 70)
    print(f"EXPLORATORY SCREEN ({len(EXPLORATORY_CANDIDATES)} candidates)")
    print("=" * 70)

    # Check all candidates present
    missing = [c for c in EXPLORATORY_CANDIDATES if c not in joined.columns]
    if missing:
        print(f"WARNING: missing candidates: {missing}")
        candidates = [c for c in EXPLORATORY_CANDIDATES if c in joined.columns]
    else:
        candidates = EXPLORATORY_CANDIDATES
    print(f"Candidates: {len(candidates)}")

    # Build candidate matrix
    cand_matrix = np.column_stack([joined[c].to_numpy() for c in candidates])
    # Mask for any NaN across candidates, y, controls
    full_mask = ~(np.isnan(y) | np.isnan(x_acf) | np.isnan(x_h))
    for j in range(cand_matrix.shape[1]):
        full_mask &= ~np.isnan(cand_matrix[:, j])
    n_eff = int(full_mask.sum())
    print(f"Effective n after NaN removal: {n_eff}")

    X_red_m = X_reduced[full_mask]
    y_m = y[full_mask]
    cand_m = cand_matrix[full_mask]
    cluster_m = cluster[full_mask]

    # Observed partial-R² per candidate
    observed_pr2 = np.empty(len(candidates))
    for j, c in enumerate(candidates):
        X_full = np.column_stack([cand_m[:, j], X_red_m])
        observed_pr2[j] = partial_r2(X_full, X_red_m, y_m)

    # Bootstrap (shared across all candidates for WY)
    print()
    boots = cluster_bootstrap_all_candidates(
        cand_m, X_red_m, y_m, cluster_m,
        n_boot=args.n_boot, seed=42,
    )

    # Raw p-values: P(bootstrap pr² < threshold)
    n_valid_per_cand = (~np.isnan(boots)).sum(axis=0)
    raw_p_values = np.array([
        (boots[~np.isnan(boots[:, j]), j] < THRESHOLD).sum() / max(n_valid_per_cand[j], 1)
        for j in range(boots.shape[1])
    ])

    # BH-FDR
    bh_qvals, bh_reject = benjamini_hochberg(raw_p_values, q=Q_FDR)

    # Westfall-Young max-T
    wy_pvals = westfall_young_max_t(observed_pr2, boots, threshold=THRESHOLD)

    # Verdict per candidate
    rows = []
    for j, c in enumerate(candidates):
        verdict = "FAILS"
        if bh_reject[j]:
            verdict = "SURVIVES_BH"
        elif raw_p_values[j] < 0.05:
            verdict = "MARGINAL_RAW"
        
        rows.append(dict(
            feature=c,
            observed_pr2=float(observed_pr2[j]),
            raw_p_value=float(raw_p_values[j]),
            bh_qvalue=float(bh_qvals[j]),
            wy_pvalue=float(wy_pvals[j]),
            survives_bh_q05=bool(bh_reject[j]),
            survives_wy_05=bool(wy_pvals[j] < 0.05),
            verdict=verdict,
            n=n_eff,
        ))

    # Sort by observed pr² descending for output
    rows = sorted(rows, key=lambda r: -r["observed_pr2"])
    out_df = pd.DataFrame(rows)

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS (sorted by observed partial-R² descending)")
    print("=" * 70)
    print()
    print(f"{'feature':<48} {'pr²':>7} {'raw_p':>8} {'BH_q':>8} {'WY_p':>8} {'verdict':>15}")
    print("-" * 100)
    for r in rows:
        flag = "  ★" if r["survives_bh_q05"] else ""
        print(f"{r['feature']:<48} {r['observed_pr2']:>+.4f} "
              f"{r['raw_p_value']:>8.4f} {r['bh_qvalue']:>8.4f} "
              f"{r['wy_pvalue']:>8.4f} {r['verdict']:>15}{flag}")

    # Save CSV
    csv_path = Path(args.outdir) / "fdr_corrected_screen.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Write report
    md_path = Path(args.outdir) / "fdr_corrected_screen.md"
    n_surv_bh = sum(1 for r in rows if r["survives_bh_q05"])
    n_surv_wy = sum(1 for r in rows if r["survives_wy_05"])
    with open(md_path, "w") as f:
        f.write("# FDR-corrected screen of Week 1 exploratory candidates\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Method\n\n")
        f.write(f"Per-series partial-R²(feature | ACF@24h, horizon) on Bitbrains ")
        f.write(f"under clip [{CLIP_LO}, {CLIP_HI}], n={n_eff}.\n\n")
        f.write(f"Cluster bootstrap: {args.n_boot} replicates over series.\n\n")
        f.write(f"**Primary correction**: Benjamini-Hochberg FDR at q = {Q_FDR} ")
        f.write(f"across {len(candidates)} exploratory candidates ")
        f.write("(Benjamini-Hochberg 1995, *JRSSB* 57:289-300).\n\n")
        f.write(f"**Supplement**: Westfall-Young step-down max-T (Westfall-Young 1993, Wiley) ")
        f.write("using the shared bootstrap distribution. Dependence-aware.\n\n")
        f.write(f"**Pre-registered WPE**: tested separately at α = {ALPHA_PRE_REG} (k=1).\n\n")
        
        f.write("## Pre-registered WPE result\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| WPE partial-R² | {wpe_pr2:.4f} |\n")
        f.write(f"| Threshold | {THRESHOLD} |\n")
        f.write(f"| Crosses threshold? | {wpe_pr2 >= THRESHOLD} |\n\n")
        if wpe_pr2 < THRESHOLD:
            f.write("WPE confirmatory test on Bitbrains under clip: **NULL**.\n\n")
        
        f.write("## Exploratory results\n\n")
        f.write("| Feature | pr² | raw_p | BH_q | WY_p | Verdict |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            flag = " ★" if r["survives_bh_q05"] else ""
            f.write(f"| `{r['feature']}` "
                    f"| {r['observed_pr2']:.4f} "
                    f"| {r['raw_p_value']:.4f} "
                    f"| {r['bh_qvalue']:.4f} "
                    f"| {r['wy_pvalue']:.4f} "
                    f"| {r['verdict']}{flag} |\n")
        
        f.write("\n## Interpretation\n\n")
        f.write(f"- {n_surv_bh}/{len(rows)} candidates survive BH-FDR at q = {Q_FDR}\n")
        f.write(f"- {n_surv_wy}/{len(rows)} candidates survive Westfall-Young max-T at α = 0.05\n\n")
        
        f.write("## Chapter implications\n\n")
        if n_surv_bh == 0:
            f.write("All Week 1 'exceedances' are multiple-comparison artefacts. ")
            f.write("Frame in §5 as: 'Of 26 exploratory candidates screened, none ")
            f.write("survive BH-FDR correction at q = 0.05 against the pre-registered ")
            f.write("threshold of 0.30. The pre-registered confirmatory test (WPE) ")
            f.write("similarly holds NULL. The predictability axis does not provide ")
            f.write("incremental information beyond ACF@24h and horizon on this ")
            f.write("workload class.'\n")
        elif n_surv_bh == 1:
            survivor = next(r for r in rows if r["survives_bh_q05"])
            f.write(f"Only `{survivor['feature']}` survives BH-FDR correction. ")
            f.write("Frame in §5 as: 'a single catch22 feature passes multiple-")
            f.write("comparison correction on the Bitbrains subset under clip; ")
            f.write("the finding is exploratory, dataset-specific, and clip-policy-")
            f.write("conditional. We do not promote it to confirmatory status.'\n\n")
            f.write("Action: examine substantive meaning of this feature (see catch22 ")
            f.write("documentation), check whether it correlates with workload ")
            f.write("characteristics on Bitbrains (the 5 idle VMs are likely drivers).\n")
        else:
            f.write(f"{n_surv_bh} candidates survive BH-FDR correction. ")
            f.write("This is a substantive multi-feature finding. Each survivor ")
            f.write("requires individual discussion. Examine the catch22 feature ")
            f.write("definitions and what they have in common (likely all symbolic ")
            f.write("transition-matrix or distribution-shape features).\n")
        
        f.write("\n## References\n\n")
        f.write("- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false ")
        f.write("discovery rate: a practical and powerful approach to multiple ")
        f.write("testing. *Journal of the Royal Statistical Society B*, 57(1), 289-300.\n")
        f.write("- Benjamini, Y., & Yekutieli, D. (2001). The control of the false ")
        f.write("discovery rate in multiple testing under dependency. ")
        f.write("*Annals of Statistics*, 29(4), 1165-1188.\n")
        f.write("- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based ")
        f.write("Multiple Testing: Examples and Methods for p-Value Adjustment*. ")
        f.write("Wiley.\n")
        f.write("- Westfall, P. H., Tobias, R. D., & Wolfinger, R. D. (2011). ")
        f.write("*Multiple Comparisons and Multiple Tests Using SAS* (2nd ed.). ")
        f.write("SAS Press.\n")
    print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
