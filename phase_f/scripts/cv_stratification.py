"""
cv_stratification.py — Week 2 Task 3 (NEW)

Replaces the clip-policy approach as the PRIMARY way to handle the 5 idle
Bitbrains VMs (bb_609-bb_613). Instead of clipping per-series R² to [-1, +1]
(which the research flagged as borderline data-dredging), we STRATIFY by
within-series coefficient of variation (CV) and report partial-R² within each
stratum.

Rationale (from methodological verification report):
  The 5 idle VMs drive the clip-policy difference. Clipping is post-hoc. The
  principled alternative is to pre-specify low-CV (idle/near-constant) vs
  high-CV (active) strata and report results within each. This is a standard
  subgroup analysis and avoids the appearance of clipping-to-find-effects.

Method:
  1. Compute per-VM within-series CV (std/mean of CPU on the series)
  2. Stratify Bitbrains VMs into CV bins (default: low / mid / high tertiles,
     plus an explicit "idle" flag for the 5 known idle VMs)
  3. Within each stratum, compute partial-R²(candidate | ACF@24h, horizon) for
     WPE + the 3 Week 1 exceedance features
  4. Report whether any threshold-crossing survives within an active-VM stratum
     (the production-relevant one)

This is now CONFIRMATORY ROBUSTNESS, not a fishing expedition: given that
the FDR screen (Task 4) found nothing surviving correction, we expect the
stratified analysis to show the same null and to attribute any raw exceedance
to the idle-VM stratum specifically.

Inputs:
  per_series_fulltest_r2.csv (Week 1 outcome)
  features_joined_extended_v2.csv (Week 1 features, has cv column)
  --cv_source (optional): omega_bitbrains.csv fallback for CV if not in features

Outputs:
  cv_stratification.csv — per (stratum, feature) partial-R²
  cv_stratification.md — interpretation report

Runtime: ~2 min (no clip; stratified subsets are small)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# 5 known idle Bitbrains VMs (stripped IDs)
KNOWN_IDLE_VMS = ["609", "610", "611", "612", "613"]

# Features to test in each stratum
FEATURES_TO_TEST = [
    "wpe_m4_t1",  # pre-registered
    "c22_SB_TransitionMatrix_3ac_sumdiagcov",  # Week 1 exceedance 1
    "c22_SP_Summaries_welch_rect_area_5_1",    # Week 1 exceedance 2
    "c22_DN_OutlierInclude_p_001_mdrmd",       # Week 1 exceedance 3
]

THRESHOLD = 0.30
N_BOOT = 1000


def partial_r2(X_full, X_reduced, y):
    """Standard partial-R²."""
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
        return np.nan, rf, rr
    return (rf - rr) / (1.0 - rr), rf, rr


def cluster_bootstrap_ci(X_full, X_reduced, y, clusters, n_boot=1000, seed=42):
    """Cluster bootstrap CI on partial-R²."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    k = len(uniq)
    if k < 3:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=k, replace=True)
        idx = np.concatenate([np.where(clusters == u)[0] for u in sampled])
        try:
            pr2, _, _ = partial_r2(X_full[idx], X_reduced[idx], y[idx])
            boots.append(pr2)
        except Exception:
            boots.append(np.nan)
    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]
    if len(boots) < 10:
        return np.nan, np.nan
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def get_cv_per_vm(joined, cv_source_path=None):
    """
    Get within-series CV per VM. Prefer the 'cv' column in the features-derived
    joined frame; fall back to an external omega_bitbrains.csv.
    """
    if "cv" in joined.columns and joined["cv"].notna().any():
        print("  Using 'cv' column from features file")
        cv_map = joined.groupby("series_id")["cv"].first().to_dict()
        return cv_map

    if cv_source_path and Path(cv_source_path).exists():
        print(f"  Loading CV from {cv_source_path}")
        cv_df = pd.read_csv(cv_source_path)
        cv_df["vm_id"] = cv_df["vm_id"].astype(str)
        return cv_df.set_index("vm_id")["cv"].to_dict()

    raise ValueError("No CV available: features file lacks 'cv' and no "
                     "--cv_source provided")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_series_csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--cv_source", default=None,
                    help="optional omega_bitbrains.csv for CV fallback")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--no_clip", action="store_true",
                    help="use no clip (default); stratification replaces clip")
    args = ap.parse_args()

    print("== cv_stratification.py ==")
    print(f"  per_series_csv: {args.per_series_csv}")
    print(f"  features:       {args.features}")
    print(f"  threshold:      {THRESHOLD}")
    print(f"  NO CLIP applied — stratification replaces clipping")
    print()

    # Load Bitbrains per-series, NO CLIP
    per = pd.read_csv(args.per_series_csv)
    bb = per[per["dataset"] == "bitbrains"].copy()
    bb["series_id"] = bb["series_id"].astype(str).str.replace(r"^bb_", "", regex=True)
    bb["delta_pp"] = (bb["r2_ensemble"] - bb["r2_naive"]) * 100.0
    print(f"Bitbrains rows (no clip): {len(bb)}")

    # Load features
    feats = pd.read_csv(args.features)
    feats["series_id"] = feats["series_id"].astype(str)
    bb_feats = feats[feats["dataset"] == "bitbrains"].copy()

    joined = bb.merge(bb_feats, on=["series_id", "dataset"], how="inner")
    print(f"Joined: {len(joined)} rows")

    # Get per-VM CV
    cv_map = get_cv_per_vm(joined, args.cv_source)
    joined["vm_cv"] = joined["series_id"].map(cv_map)
    n_missing_cv = joined["vm_cv"].isna().sum()
    if n_missing_cv > 0:
        print(f"  WARNING: {n_missing_cv} rows missing CV, dropping")
        joined = joined[joined["vm_cv"].notna()]
    print()

    # Mark idle VMs
    joined["is_idle"] = joined["series_id"].isin(KNOWN_IDLE_VMS)
    n_idle_rows = joined["is_idle"].sum()
    idle_present = sorted(set(joined[joined["is_idle"]]["series_id"]))
    print(f"Known idle VMs present: {idle_present} ({n_idle_rows} rows)")

    # Report CV distribution
    vm_cv = joined.groupby("series_id")["vm_cv"].first()
    print(f"\nCV distribution across {len(vm_cv)} VMs:")
    print(f"  min:    {vm_cv.min():.4f}")
    print(f"  q25:    {vm_cv.quantile(0.25):.4f}")
    print(f"  median: {vm_cv.median():.4f}")
    print(f"  q75:    {vm_cv.quantile(0.75):.4f}")
    print(f"  max:    {vm_cv.max():.4f}")
    print(f"  CV of the 5 idle VMs:")
    for vm in idle_present:
        print(f"    bb_{vm}: CV = {cv_map.get(vm, np.nan):.4f}")

    # Define strata: tertiles of CV + explicit idle exclusion variant
    cv_t1 = vm_cv.quantile(1/3)
    cv_t2 = vm_cv.quantile(2/3)
    print(f"\nCV tertile cuts: t1={cv_t1:.4f}, t2={cv_t2:.4f}")

    def assign_stratum(cv):
        if cv <= cv_t1:
            return "low_cv"
        elif cv <= cv_t2:
            return "mid_cv"
        else:
            return "high_cv"

    joined["cv_stratum"] = joined["vm_cv"].apply(assign_stratum)

    # Strata to analyse:
    #  - all (baseline, no clip)
    #  - low_cv, mid_cv, high_cv (tertiles)
    #  - active_only (exclude the 5 known idle VMs)
    strata = {
        "all_noclip": joined,
        "low_cv": joined[joined["cv_stratum"] == "low_cv"],
        "mid_cv": joined[joined["cv_stratum"] == "mid_cv"],
        "high_cv": joined[joined["cv_stratum"] == "high_cv"],
        "active_only": joined[~joined["is_idle"]],
        "idle_only": joined[joined["is_idle"]],
    }

    rows = []
    for stratum_name, sub in strata.items():
        n_vms = sub["series_id"].nunique()
        n_rows = len(sub)
        print()
        print("=" * 70)
        print(f"STRATUM: {stratum_name}  ({n_vms} VMs, {n_rows} rows)")
        print("=" * 70)

        if n_rows < 30 or n_vms < 5:
            print(f"  Too small (n_rows={n_rows}, n_vms={n_vms}), skipping inference")
            for feat in FEATURES_TO_TEST:
                rows.append(dict(
                    stratum=stratum_name, feature=feat, n_vms=n_vms, n_rows=n_rows,
                    partial_r2=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    crosses_threshold=False, note="stratum too small",
                ))
            continue

        y = sub["delta_pp"].to_numpy()
        x_acf = sub["acf_24h"].to_numpy()
        x_h = sub["horizon_min"].to_numpy()
        cluster = sub["series_id"].to_numpy()
        X_reduced = np.column_stack([x_acf, x_h])

        # r²_reduced for this stratum
        base_mask = ~(np.isnan(y) | np.isnan(x_acf) | np.isnan(x_h))
        if base_mask.sum() >= 5:
            _, _, r2_red = partial_r2(
                np.column_stack([np.zeros(base_mask.sum()), X_reduced[base_mask]]),
                X_reduced[base_mask], y[base_mask])
        else:
            r2_red = np.nan
        print(f"  r²_reduced(ACF+horizon) = {r2_red:.4f}")

        for feat in FEATURES_TO_TEST:
            if feat not in sub.columns:
                continue
            x_cand = sub[feat].to_numpy()
            mask = base_mask & ~np.isnan(x_cand)
            n = int(mask.sum())
            if n < 30:
                rows.append(dict(
                    stratum=stratum_name, feature=feat, n_vms=n_vms, n_rows=n,
                    partial_r2=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    crosses_threshold=False, note="too few after NaN",
                ))
                continue
            X_full = np.column_stack([x_cand[mask], x_acf[mask], x_h[mask]])
            X_red_m = X_reduced[mask]
            pr2, _, _ = partial_r2(X_full, X_red_m, y[mask])
            lo, hi = cluster_bootstrap_ci(X_full, X_red_m, y[mask],
                                           cluster[mask], n_boot=args.n_boot)
            crosses = bool(pr2 >= THRESHOLD) if not np.isnan(pr2) else False
            ci_excludes_threshold = (not np.isnan(lo)) and (lo > THRESHOLD)
            flag = "  ★ CROSSES" if crosses else ""
            ci_flag = "  [CI > 0.30]" if ci_excludes_threshold else ""
            print(f"    {feat:<48} pr²={pr2:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]{flag}{ci_flag}")
            rows.append(dict(
                stratum=stratum_name, feature=feat, n_vms=n_vms, n_rows=n,
                partial_r2=pr2, ci_lo=lo, ci_hi=hi,
                crosses_threshold=crosses,
                ci_excludes_threshold=ci_excludes_threshold,
                note="",
            ))

    # Save
    out_df = pd.DataFrame(rows)
    csv_path = Path(args.outdir) / "cv_stratification.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Report
    md_path = Path(args.outdir) / "cv_stratification.md"
    with open(md_path, "w") as f:
        f.write("# CV-stratified partial-R² (replaces clip-policy analysis)\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Rationale\n\n")
        f.write("The 5 idle Bitbrains VMs (bb_609-bb_613) drove the clip-policy ")
        f.write("difference in Week 1. Rather than clipping per-series R² to ")
        f.write("[-1, +1] (post-hoc, borderline data-dredging), this analysis ")
        f.write("stratifies by within-series CV and reports partial-R² within ")
        f.write("each stratum. The production-relevant question is whether any ")
        f.write("predictability feature crosses 0.30 in the **active-VM stratum** ")
        f.write("(excluding idle VMs), under NO clip.\n\n")
        f.write("## CV distribution\n\n")
        f.write(f"- Across {len(vm_cv)} VMs: min={vm_cv.min():.3f}, ")
        f.write(f"median={vm_cv.median():.3f}, max={vm_cv.max():.3f}\n")
        f.write(f"- Tertile cuts: low ≤ {cv_t1:.3f} < mid ≤ {cv_t2:.3f} < high\n")
        f.write(f"- Idle VMs (bb_609-613) CV: ")
        f.write(", ".join(f"bb_{vm}={cv_map.get(vm, float('nan')):.3f}"
                          for vm in idle_present))
        f.write("\n\n")
        f.write("## Results by stratum\n\n")
        f.write("| Stratum | Feature | n_VMs | n | pr² | CI | Crosses 0.30? |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            if pd.isna(r["partial_r2"]):
                continue
            cross = "**YES**" if r["crosses_threshold"] else "no"
            f.write(f"| {r['stratum']} | `{r['feature']}` | {r['n_vms']} "
                    f"| {r['n_rows']} | {r['partial_r2']:.3f} "
                    f"| [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {cross} |\n")
        f.write("\n## Interpretation\n\n")

        # Key question: does anything cross in active_only?
        active_crossings = [
            r for r in rows
            if r["stratum"] == "active_only" and r["crosses_threshold"]
        ]
        active_ci_robust = [
            r for r in rows
            if r["stratum"] == "active_only" and r.get("ci_excludes_threshold", False)
        ]
        if not active_crossings:
            f.write("**No feature crosses 0.30 in the active-VM stratum** (idle ")
            f.write("VMs excluded, no clip). This confirms that the Week 1 ")
            f.write("clip-policy exceedances were artefacts of the idle VMs. The ")
            f.write("predictability axis provides no incremental information on ")
            f.write("the production-relevant active workloads.\n\n")
            f.write("This is the structural confirmation of the FDR-screen result ")
            f.write("(no candidate survived multiple-comparison correction).\n")
        elif active_ci_robust:
            f.write(f"{len(active_ci_robust)} feature(s) cross 0.30 in the ")
            f.write("active-VM stratum WITH CI excluding the threshold. This would ")
            f.write("be a genuine finding requiring discussion:\n\n")
            for r in active_ci_robust:
                f.write(f"- `{r['feature']}`: pr²={r['partial_r2']:.3f} "
                        f"CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]\n")
        else:
            f.write(f"{len(active_crossings)} feature(s) have point estimates ")
            f.write("above 0.30 in the active stratum, but none with CI excluding ")
            f.write("0.30. Consistent with noise.\n")

        f.write("\n## Chapter framing\n\n")
        f.write("Use CV-stratification as the PRIMARY robustness analysis in §5, ")
        f.write("replacing the clip-policy dual-reporting. The clip analysis can ")
        f.write("be relegated to an appendix or a single sentence. Stratification ")
        f.write("is the principled subgroup analysis (no post-hoc data transform).\n")
    print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
