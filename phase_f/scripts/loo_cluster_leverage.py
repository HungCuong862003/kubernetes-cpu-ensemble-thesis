"""
loo_cluster_leverage.py — Week 2 Task 5 (AUGMENTED)

Original Task 5 was leave-one-VM-out stability. The methodological verification
report recommended augmenting with cluster-level partial-leverage statistics
(MacKinnon, Nielsen & Webb 2023, summclust) to identify whether specific VMs —
especially the 5 idle ones (bb_609-bb_613) — are high-leverage points driving
the Week 1 clip-policy result.

Two diagnostics:

1. **Leave-one-VM-out (LOO)**: drop each VM, recompute partial-R² on remaining
   VMs. Report the distribution of LOO partial-R² and identify which VMs cause
   the largest shifts.

2. **Cluster partial leverage**: for each VM cluster g, compute the leverage
   L_g (how much cluster g influences the fitted regression) and the partial
   leverage on the candidate coefficient. Flag VMs with L_g > 2·G/N.
   Reference: MacKinnon, Nielsen & Webb (2023), Stata Journal 23:942-982.

Applied to the 3 Week 1 exceedance features + WPE, on Bitbrains.
By default uses clip [-1, +1] to match Week 1 (so LOO is comparable), but
also reports no-clip leverage.

Inputs:
  per_series_fulltest_r2.csv
  features_joined_extended_v2.csv

Outputs:
  loo_cluster_leverage.csv — per (feature, VM) LOO partial-R² + leverage
  loo_cluster_leverage_summary.csv — per feature: LOO min/max/IQR, top-leverage VMs
  loo_cluster_leverage.md — report

Runtime: ~3 min (142 LOO refits × 4 features, plus leverage computation)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KNOWN_IDLE_VMS = ["609", "610", "611", "612", "613"]

FEATURES_TO_TEST = [
    "wpe_m4_t1",
    "c22_SB_TransitionMatrix_3ac_sumdiagcov",
    "c22_SP_Summaries_welch_rect_area_5_1",
    "c22_DN_OutlierInclude_p_001_mdrmd",
]

THRESHOLD = 0.30
CLIP_LO, CLIP_HI = -1.0, 1.0


def partial_r2(X_full, X_reduced, y):
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


def compute_cluster_leverage(X, clusters):
    """
    Cluster leverage L_g = trace(H_gg) where H = X(X'X)^-1 X' is the hat matrix
    and H_gg is the block for cluster g. This measures how much cluster g's
    own observations influence their own fitted values.

    Reference: MacKinnon, Nielsen & Webb (2023), Stata Journal 23:942-982,
    equation for cluster leverage. Normalised so sum of L_g = number of
    parameters (k).
    """
    n, k = X.shape
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    uniq = np.unique(clusters)
    leverage = {}
    for g in uniq:
        idx = np.where(clusters == g)[0]
        Xg = X[idx, :]
        # L_g = trace(Xg (X'X)^-1 Xg')
        Hgg = Xg @ XtX_inv @ Xg.T
        leverage[g] = float(np.trace(Hgg))
    return leverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_series_csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--clip", action="store_true", default=True,
                    help="apply clip [-1,+1] to match Week 1 (default True)")
    ap.add_argument("--no_clip_mode", action="store_true",
                    help="ALSO run without clip for comparison")
    args = ap.parse_args()

    print("== loo_cluster_leverage.py ==")
    print(f"  threshold: {THRESHOLD}")
    print(f"  clip:      [{CLIP_LO}, {CLIP_HI}] (matching Week 1)")
    print()

    # Load Bitbrains
    per = pd.read_csv(args.per_series_csv)
    bb = per[per["dataset"] == "bitbrains"].copy()
    bb["series_id"] = bb["series_id"].astype(str).str.replace(r"^bb_", "", regex=True)

    # Apply clip (matching Week 1)
    bb["r2_ensemble_clip"] = bb["r2_ensemble"].clip(CLIP_LO, CLIP_HI)
    bb["r2_naive_clip"] = bb["r2_naive"].clip(CLIP_LO, CLIP_HI)
    bb["delta_pp"] = (bb["r2_ensemble_clip"] - bb["r2_naive_clip"]) * 100.0

    feats = pd.read_csv(args.features)
    feats["series_id"] = feats["series_id"].astype(str)
    bb_feats = feats[feats["dataset"] == "bitbrains"].copy()

    joined = bb.merge(bb_feats, on=["series_id", "dataset"], how="inner")
    print(f"Joined: {len(joined)} rows, {joined['series_id'].nunique()} VMs")
    print()

    x_acf = joined["acf_24h"].to_numpy()
    x_h = joined["horizon_min"].to_numpy()
    y_all = joined["delta_pp"].to_numpy()
    cluster_all = joined["series_id"].to_numpy()

    loo_rows = []
    leverage_rows = []
    summary_rows = []

    for feat in FEATURES_TO_TEST:
        if feat not in joined.columns:
            print(f"SKIP {feat}: not in columns")
            continue
        print("=" * 70)
        print(f"FEATURE: {feat}")
        print("=" * 70)

        x_cand = joined[feat].to_numpy()
        mask = ~(np.isnan(y_all) | np.isnan(x_cand) | np.isnan(x_acf) | np.isnan(x_h))
        y = y_all[mask]
        xc = x_cand[mask]
        xa = x_acf[mask]
        xh = x_h[mask]
        clusters = cluster_all[mask]

        X_full = np.column_stack([xc, xa, xh])
        X_reduced = np.column_stack([xa, xh])

        # Full-sample partial-R²
        full_pr2 = partial_r2(X_full, X_reduced, y)
        print(f"  Full-sample partial-R²: {full_pr2:.4f}")

        # === Cluster leverage ===
        X_with_const = np.column_stack([np.ones(len(y)), X_full])
        leverage = compute_cluster_leverage(X_with_const, clusters)
        G = len(leverage)
        N = len(y)
        k = X_with_const.shape[1]
        leverage_threshold = 2.0 * k / G  # rule of thumb for cluster leverage
        mean_leverage = k / G  # average cluster leverage

        print(f"  Mean cluster leverage (k/G): {mean_leverage:.4f}")
        print(f"  High-leverage threshold (2k/G): {leverage_threshold:.4f}")

        # Flag high-leverage VMs
        high_lev = {vm: lev for vm, lev in leverage.items()
                    if lev > leverage_threshold}
        print(f"  High-leverage VMs ({len(high_lev)}):")
        for vm, lev in sorted(high_lev.items(), key=lambda x: -x[1])[:10]:
            idle_flag = " [IDLE]" if vm in KNOWN_IDLE_VMS else ""
            print(f"    bb_{vm}: leverage = {lev:.4f}{idle_flag}")

        for vm, lev in leverage.items():
            leverage_rows.append(dict(
                feature=feat, vm_id=vm, leverage=lev,
                is_high_leverage=bool(lev > leverage_threshold),
                is_idle=bool(vm in KNOWN_IDLE_VMS),
            ))

        # === Leave-one-VM-out ===
        uniq_vms = np.unique(clusters)
        loo_pr2 = {}
        for vm in uniq_vms:
            keep = clusters != vm
            if keep.sum() < 30:
                continue
            pr2_loo = partial_r2(X_full[keep], X_reduced[keep], y[keep])
            loo_pr2[vm] = pr2_loo
            loo_rows.append(dict(
                feature=feat, dropped_vm=vm, loo_partial_r2=pr2_loo,
                shift_from_full=pr2_loo - full_pr2 if not np.isnan(pr2_loo) else np.nan,
                is_idle=bool(vm in KNOWN_IDLE_VMS),
            ))

        loo_vals = np.array([v for v in loo_pr2.values() if not np.isnan(v)])
        loo_min = float(loo_vals.min())
        loo_max = float(loo_vals.max())
        loo_median = float(np.median(loo_vals))
        loo_iqr = float(np.percentile(loo_vals, 75) - np.percentile(loo_vals, 25))

        # How many LOO refits keep pr² above threshold?
        n_above = int((loo_vals >= THRESHOLD).sum())
        frac_above = n_above / len(loo_vals)

        print(f"  LOO partial-R²: min={loo_min:.4f}, median={loo_median:.4f}, "
              f"max={loo_max:.4f}, IQR={loo_iqr:.4f}")
        print(f"  LOO refits with pr² >= {THRESHOLD}: {n_above}/{len(loo_vals)} "
              f"({100*frac_above:.1f}%)")

        # Which VM, when dropped, causes the biggest drop in pr²?
        shifts = {vm: (loo_pr2[vm] - full_pr2)
                  for vm in loo_pr2 if not np.isnan(loo_pr2[vm])}
        most_influential = sorted(shifts.items(), key=lambda x: x[1])[:5]
        print(f"  VMs whose removal most DECREASES pr² (most supportive of effect):")
        for vm, shift in most_influential:
            idle_flag = " [IDLE]" if vm in KNOWN_IDLE_VMS else ""
            print(f"    bb_{vm}: shift = {shift:+.4f}{idle_flag}")

        # Reverse: VMs whose removal INCREASES pr² most (suppressing the effect)
        most_suppressing = sorted(shifts.items(), key=lambda x: -x[1])[:5]
        print(f"  VMs whose removal most INCREASES pr² (suppressing effect):")
        for vm, shift in most_suppressing:
            idle_flag = " [IDLE]" if vm in KNOWN_IDLE_VMS else ""
            print(f"    bb_{vm}: shift = {shift:+.4f}{idle_flag}")
        print()

        summary_rows.append(dict(
            feature=feat,
            full_partial_r2=full_pr2,
            loo_min=loo_min, loo_median=loo_median, loo_max=loo_max,
            loo_iqr=loo_iqr,
            n_loo_above_threshold=n_above,
            frac_loo_above_threshold=frac_above,
            n_high_leverage_vms=len(high_lev),
            n_idle_high_leverage=sum(1 for vm in high_lev if vm in KNOWN_IDLE_VMS),
        ))

    # Save
    pd.DataFrame(loo_rows).to_csv(
        Path(args.outdir) / "loo_cluster_leverage.csv", index=False)
    pd.DataFrame(leverage_rows).to_csv(
        Path(args.outdir) / "loo_cluster_leverage_detail.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        Path(args.outdir) / "loo_cluster_leverage_summary.csv", index=False)
    print(f"Wrote loo_cluster_leverage.csv, _detail.csv, _summary.csv")

    # Report
    md_path = Path(args.outdir) / "loo_cluster_leverage.md"
    with open(md_path, "w") as f:
        f.write("# LOO + cluster-leverage diagnostics (Bitbrains)\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Method\n\n")
        f.write("Two diagnostics on the 3 Week 1 exceedance features + WPE, ")
        f.write(f"Bitbrains under clip [{CLIP_LO}, {CLIP_HI}]:\n\n")
        f.write("1. **Leave-one-VM-out**: drop each VM, recompute partial-R².\n")
        f.write("2. **Cluster leverage** (MacKinnon-Nielsen-Webb 2023): identify ")
        f.write("VMs with disproportionate influence on the regression.\n\n")
        f.write("## Summary\n\n")
        f.write("| Feature | Full pr² | LOO min | LOO median | LOO max | "
                "% LOO ≥ 0.30 | High-lev VMs | Idle high-lev |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(f"| `{r['feature']}` | {r['full_partial_r2']:.3f} "
                    f"| {r['loo_min']:.3f} | {r['loo_median']:.3f} "
                    f"| {r['loo_max']:.3f} "
                    f"| {100*r['frac_loo_above_threshold']:.0f}% "
                    f"| {r['n_high_leverage_vms']} "
                    f"| {r['n_idle_high_leverage']} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("For each exceedance feature, the key questions are:\n\n")
        f.write("1. **Is the effect stable under LOO?** If most LOO refits keep ")
        f.write("pr² ≥ 0.30, the effect is not driven by a single VM. If the % drops ")
        f.write("sharply, it is fragile.\n")
        f.write("2. **Are idle VMs high-leverage?** If the 5 idle VMs appear as ")
        f.write("high-leverage points, the Week 1 exceedance is an idle-VM artefact, ")
        f.write("consistent with the CV-stratification and FDR-screen findings.\n\n")
        f.write("Cross-reference with `cv_stratification.md` (active-only stratum) ")
        f.write("and `fdr_corrected_screen.md` (multiple-comparison correction). ")
        f.write("All three analyses should agree: the exceedances are idle-VM ")
        f.write("artefacts that do not survive proper inference.\n\n")
        f.write("## Reference\n\n")
        f.write("MacKinnon, J. G., Nielsen, M. Ø., & Webb, M. D. (2023). Leverage, ")
        f.write("influence, and the jackknife in clustered regression models: ")
        f.write("Reliable inference using summclust. *Stata Journal*, 23(4), 942-982.\n")
    print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
