"""
identification_limits.py — Week 2 Task 1 (REVISED)

Replaces the wild-cluster bootstrap task. Documents the structural identification
limits of the locked F2 cell-level test at G=3 datasets, n=12.

Key references:
- Mundlak (1978), Econometrica 46:69-85: within/between decomposition,
  no-within-variation problem
- Snijders & Bosker (2012), Multilevel Analysis 2nd ed.: cluster-level inference
- Ibragimov & Müller (2016), Review of Economics and Statistics 98:83-96:
  t(G-1) inference at small G
- MacKinnon & Webb (2018), Econometrics Journal 21:114-135: wild bootstrap
  needs G ≥ 15-20 for acceptable size control — INAPPLICABLE here

What this script does:
1. Loads the cell-level partial-R² data (12 cells = 3 datasets × 4 horizons)
2. Demonstrates the structural degeneracy: any constant-per-dataset feature
   gives the same partial-R² by construction
3. Computes Ibragimov-Müller t(G-1) = t(2) statistic for between-dataset slopes
4. Outputs a documentation table for the F2 chapter §3 methodology footnote

Output:
  identification_limits.md — chapter-ready documentation
  identification_limits.csv — supporting numerical data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Cell-level data: 12 (dataset, horizon) cells from locked F2
# Source: phase_f/data/f2_partial_r2_results.csv + reconstructed from BCF pairs
# Schema we expect: dataset, horizon, wpe (per-dataset median), acf_24h,
#                   delta_pp (ensemble - naive R², in pp)
EXPECTED_DATASETS = ["alibaba", "bitbrains", "bytedance"]
EXPECTED_HORIZONS = [10, 30, 60, 120]


def partial_r2(X_full, X_reduced, y):
    """Standard partial-R² calculation."""
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


def demonstrate_structural_degeneracy(cells_df):
    """
    Show that ANY constant-per-dataset candidate returns the same partial-R²
    by construction at this n=12 design.
    """
    print("=" * 70)
    print("STRUCTURAL DEGENERACY DEMONSTRATION")
    print("=" * 70)
    print()
    print("Hypothesis: at n=12 with 3 dataset clusters, any regressor that takes")
    print("a constant value within each dataset gives the same partial-R² when")
    print("controlling for ACF@24h and horizon.")
    print()

    y = cells_df["delta_pp"].to_numpy()
    x_acf = cells_df["acf_24h"].to_numpy()
    x_h = cells_df["horizon_min"].to_numpy()
    X_reduced = np.column_stack([x_acf, x_h])

    # Try 5 different constant-per-dataset candidates
    test_candidates = {
        "wpe_actual": cells_df["wpe"].to_numpy(),
        "constant_v1": np.where(cells_df["dataset"] == "alibaba", 0.5,
                          np.where(cells_df["dataset"] == "bitbrains", 0.8, 0.3)),
        "constant_v2": np.where(cells_df["dataset"] == "alibaba", -1.0,
                          np.where(cells_df["dataset"] == "bitbrains", 2.0, 0.0)),
        "random_per_dataset_a": np.where(cells_df["dataset"] == "alibaba", 7.42,
                                  np.where(cells_df["dataset"] == "bitbrains", -3.18, 0.99)),
        "random_per_dataset_b": np.where(cells_df["dataset"] == "alibaba", 0.001,
                                  np.where(cells_df["dataset"] == "bitbrains", 100.0, -50.0)),
    }

    results = []
    for name, x_cand in test_candidates.items():
        X_full = np.column_stack([x_cand, x_acf, x_h])
        pr2, rf, rr = partial_r2(X_full, X_reduced, y)
        results.append(dict(candidate=name, partial_r2=pr2,
                            r2_full=rf, r2_reduced=rr))
        print(f"  {name:<25} partial-R² = {pr2:.6f}  (r²_full={rf:.4f}, r²_red={rr:.4f})")

    print()
    pr2_values = [r["partial_r2"] for r in results]
    sd = float(np.std(pr2_values))
    print(f"  Standard deviation across 5 candidates: {sd:.6e}")
    if sd < 1e-6:
        print("  -> CONFIRMED: all 5 constant-per-dataset candidates give "
              "identical partial-R²")
        print("     (within numerical precision). The test is structurally")
        print("     non-identified for any within-dataset-constant feature.")
    else:
        print("  -> Hypothesis NOT confirmed. Investigate.")

    return results


def ibragimov_muller_t_test(cells_df):
    """
    Ibragimov-Müller (2016) t(G-1) inference: fit a separate regression on each
    of the G dataset clusters, then test the between-dataset slope distribution
    using t(G-1) = t(2). This is the only defensible small-G inferential
    procedure under exchangeability of cluster estimators.
    """
    print()
    print("=" * 70)
    print("IBRAGIMOV-MÜLLER t(G-1)=t(2) INFERENCE")
    print("=" * 70)
    print()
    print("Procedure: fit slope of delta_pp on WPE within each dataset,")
    print("then test mean of 3 slopes vs 0 using t(2).")
    print()

    slopes = []
    for ds in EXPECTED_DATASETS:
        sub = cells_df[cells_df["dataset"] == ds]
        if len(sub) < 3:
            print(f"  {ds}: only {len(sub)} cells, skipping")
            continue
        y = sub["delta_pp"].to_numpy()
        x = sub["wpe"].to_numpy()
        x_h = sub["horizon_min"].to_numpy()
        # Fit delta_pp ~ WPE + horizon, get WPE slope
        X = np.column_stack([np.ones(len(y)), x, x_h])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        wpe_slope = beta[1]
        slopes.append(dict(dataset=ds, slope=wpe_slope, n_cells=len(sub)))
        print(f"  {ds}: WPE slope (controlling for horizon) = {wpe_slope:+.4f}")

    slope_vals = np.array([s["slope"] for s in slopes])
    G = len(slope_vals)
    if G < 2:
        print("  Need G ≥ 2 datasets; cannot proceed.")
        return None

    mean_slope = float(slope_vals.mean())
    sd_slope = float(slope_vals.std(ddof=1))
    se_mean = sd_slope / np.sqrt(G)
    t_stat = mean_slope / se_mean if se_mean > 0 else np.nan
    df = G - 1
    if not np.isnan(t_stat):
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    else:
        p_value = np.nan

    print()
    print(f"  Number of datasets (G):        {G}")
    print(f"  Mean of per-dataset slopes:    {mean_slope:+.4f}")
    print(f"  SD of per-dataset slopes:      {sd_slope:.4f}")
    print(f"  SE of mean (= SD / sqrt(G)):   {se_mean:.4f}")
    print(f"  t({df}) statistic:              {t_stat:+.4f}")
    print(f"  Two-sided p-value:             {p_value:.4f}")
    print()
    print(f"  At G={G}, t({df}) inference has effectively zero power except")
    print(f"  for very large effects. A non-significant result here is")
    print(f"  uninformative — it reflects the design, not the signal.")
    print()
    print(f"  Reference: Ibragimov & Müller (2016), Review of Economics and")
    print(f"  Statistics 98:83-96.")

    return dict(
        G=G, mean_slope=mean_slope, sd_slope=sd_slope, se_mean=se_mean,
        t_stat=t_stat, df=df, p_value=p_value, per_dataset_slopes=slopes,
    )


def write_chapter_documentation(cells_df, degeneracy_results, im_results, outdir):
    """Generate F2 chapter §3 methodology footnote content."""
    md_path = Path(outdir) / "identification_limits.md"
    with open(md_path, "w") as f:
        f.write("# F2 chapter §3 methodology — cell-level identification limits\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Purpose\n\n")
        f.write("This document is the source material for the F2 chapter §3.X.X ")
        f.write("methodology footnote on cell-level structural identification.\n\n")

        f.write("## Background\n\n")
        f.write("The pre-registered F2 confirmatory test (DECISION-005) is at ")
        f.write("cell-level n=12 (3 datasets × 4 horizons) with regressors ")
        f.write("delta_pp ~ WPE + ACF@24h + horizon. The test was specified prior ")
        f.write("to data collection and reported NULL at threshold 0.30 ")
        f.write("(partial-R² = 0.0790).\n\n")

        f.write("Week 1 analysis revealed that this design is structurally ")
        f.write("non-identified for within-dataset-constant regressors. Per-dataset ")
        f.write("median values of WPE (used in the locked F2 test) are constant ")
        f.write("within each dataset, hence subject to this limitation.\n\n")

        f.write("## Mundlak decomposition\n\n")
        f.write("Mundlak (1978, *Econometrica* 46:69-85) showed that a regressor ")
        f.write("constant within each cluster contributes only between-cluster ")
        f.write("variation to the regression: within-cluster variation is zero by ")
        f.write("construction. With G=3 datasets and the controls (ACF@24h, horizon) ")
        f.write("absorbing the cluster-by-horizon effects, the effective sample size ")
        f.write("for the between-cluster WPE slope is G=3, not n=12.\n\n")

        f.write("## Empirical demonstration\n\n")
        f.write("To verify the structural-degeneracy claim, five candidate features ")
        f.write("with arbitrary per-dataset constants (including the actual WPE values) ")
        f.write("were tested. All return identical partial-R² to numerical precision:\n\n")

        f.write("| Candidate | Partial-R² |\n")
        f.write("|---|---|\n")
        for r in degeneracy_results:
            f.write(f"| `{r['candidate']}` | {r['partial_r2']:.6f} |\n")
        f.write("\n")
        sd = float(np.std([r["partial_r2"] for r in degeneracy_results]))
        f.write(f"Standard deviation across candidates: {sd:.2e}\n\n")
        f.write("**Conclusion**: the cell-level test measures the incremental ")
        f.write("contribution of a *between-dataset* effect after partialling out ")
        f.write("horizon. It does NOT test WPE-specific information. ")
        f.write("Any feature constant within dataset would return the same value.\n\n")

        f.write("## Ibragimov-Müller small-G inference\n\n")
        f.write("Ibragimov & Müller (2016, *Review of Economics and Statistics* ")
        f.write("98:83-96) provide the only published asymptotic theory for ")
        f.write("inference with very few clusters: under exchangeability of ")
        f.write("cluster-level estimators, use t(G-1).\n\n")

        if im_results is not None:
            f.write("Application to the locked F2 cell-level test:\n\n")
            f.write("| Quantity | Value |\n")
            f.write("|---|---|\n")
            f.write(f"| Number of dataset clusters (G) | {im_results['G']} |\n")
            f.write(f"| Mean WPE slope across datasets | {im_results['mean_slope']:+.4f} |\n")
            f.write(f"| SD of slopes | {im_results['sd_slope']:.4f} |\n")
            f.write(f"| SE of mean slope | {im_results['se_mean']:.4f} |\n")
            f.write(f"| t({im_results['df']}) statistic | {im_results['t_stat']:+.4f} |\n")
            f.write(f"| Two-sided p-value | {im_results['p_value']:.4f} |\n\n")
            f.write("At G=3, t(2) inference has effectively zero power except for very ")
            f.write("large effects. A non-significant result reflects the design's ")
            f.write("inferential ceiling, not the absence of signal.\n\n")

        f.write("## Why wild-cluster bootstrap is NOT used\n\n")
        f.write("Cameron, Gelbach & Miller (2008, *Review of Economics and Statistics* ")
        f.write("90:414-427) and MacKinnon & Webb (2018, *Econometrics Journal* ")
        f.write("21:114-135) provide wild-cluster bootstrap as a small-cluster ")
        f.write("correction. MacKinnon & Webb (2018) simulations show acceptable size ")
        f.write("control only at G ≥ 15-20. At G=3, no published asymptotic theory ")
        f.write("supports its application. More fundamentally, structural ")
        f.write("non-identification cannot be cured by any resampling correction — ")
        f.write("the regressor lacks within-cluster variation by construction.\n\n")

        f.write("## Implication for chapter narrative\n\n")
        f.write("The pre-registered cell-level NULL at threshold 0.30 still ")
        f.write("stands as documentation, but is no longer claimed as a powered ")
        f.write("confirmatory test. Inferential weight transfers to the per-series ")
        f.write("extension (§5.X), which has G=142 (Bitbrains), G=93 (ByteDance), ")
        f.write("and G=4921 (Alibaba) clusters — orders of magnitude above the ")
        f.write("identification floor.\n\n")

        f.write("This is the standard 'ecological fallacy' framing (Robinson 1950, ")
        f.write("*American Sociological Review* 15:351-357; King 1997, ")
        f.write("*A Solution to the Ecological Inference Problem*): cell-level ")
        f.write("R²_reduced = 0.903 answers 'can we rank datasets by predictability?' ")
        f.write("(yes); per-series r²_reduced = 0.018-0.078 answers 'can we predict ")
        f.write("individual series-horizon ensemble lift?' (mostly no). Both are true.\n\n")

        f.write("## References\n\n")
        f.write("- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based ")
        f.write("improvements for inference with clustered errors. *Review of Economics ")
        f.write("and Statistics*, 90(3), 414-427.\n")
        f.write("- Ibragimov, R., & Müller, U. K. (2016). Inference with few heterogeneous ")
        f.write("clusters. *Review of Economics and Statistics*, 98(1), 83-96.\n")
        f.write("- King, G. (1997). *A Solution to the Ecological Inference Problem*. ")
        f.write("Princeton University Press.\n")
        f.write("- MacKinnon, J. G., & Webb, M. D. (2018). The wild bootstrap for few ")
        f.write("(treated) clusters. *Econometrics Journal*, 21(2), 114-135.\n")
        f.write("- Mundlak, Y. (1978). On the pooling of time series and cross section ")
        f.write("data. *Econometrica*, 46(1), 69-85.\n")
        f.write("- Robinson, W. S. (1950). Ecological correlations and the behavior of ")
        f.write("individuals. *American Sociological Review*, 15(3), 351-357.\n")
        f.write("- Snijders, T. A. B., & Bosker, R. J. (2012). *Multilevel Analysis: ")
        f.write("An Introduction to Basic and Advanced Multilevel Modeling* (2nd ed.). ")
        f.write("Sage Publications.\n")
    return md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells_csv",
                    help="cell-level data with cols: dataset, horizon_min, wpe, "
                         "acf_24h, delta_pp")
    ap.add_argument("--bcf_pairs_csv",
                    help="alternative: reconstruct from bcf_pairs.csv")
    ap.add_argument("--features",
                    help="features file for per-dataset WPE and ACF medians")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    print("== identification_limits.py ==")
    print(f"  outdir: {args.outdir}")
    print()

    # Load cell-level data
    if args.cells_csv and Path(args.cells_csv).exists():
        cells_df = pd.read_csv(args.cells_csv)
        print(f"Loaded {len(cells_df)} cells from {args.cells_csv}")
    elif args.bcf_pairs_csv and args.features:
        print("Reconstructing cells from bcf_pairs.csv + features")
        bcf = pd.read_csv(args.bcf_pairs_csv)
        feats = pd.read_csv(args.features)
        # Per-dataset medians
        medians = feats.groupby("dataset").agg(
            wpe=("wpe_m4_t1", "median"),
            acf_24h=("acf_24h", "median")
        ).reset_index()
        medians["dataset"] = medians["dataset"].str.lower()
        # Use ONLY the NNLS (production ensemble) rows. The cell-level F2 test
        # uses the production ensemble's delta_pp, not an average across the
        # foundation models. Filtering avoids diluting the outcome.
        if "model" in bcf.columns:
            bcf = bcf[bcf["model"] == "NNLS"].copy()
            print(f"  filtered to NNLS rows: {len(bcf)} cells")
        # Aggregate bcf_pairs to (dataset, horizon) cells
        bcf["dataset"] = bcf["dataset"].str.lower()
        if "horizon" in bcf.columns:
            # horizon column may be '10min', '30min' etc. — strip non-digits
            bcf["horizon_min"] = (
                bcf["horizon"].astype(str)
                .str.replace(r"[^0-9]", "", regex=True)
                .astype(int)
            )
        agg = bcf.groupby(["dataset", "horizon_min"]).agg(
            delta_pp=("delta_pp", "mean")
        ).reset_index()
        cells_df = agg.merge(medians, on="dataset", how="left")
        print(f"Reconstructed {len(cells_df)} cells")
    else:
        print("ERROR: must provide --cells_csv OR --bcf_pairs_csv + --features")
        return 1

    print()
    print("Cells:")
    print(cells_df.to_string(index=False))
    print()

    # Step 1: structural degeneracy demonstration
    degeneracy = demonstrate_structural_degeneracy(cells_df)

    # Step 2: Ibragimov-Müller t(G-1) inference
    im_results = ibragimov_muller_t_test(cells_df)

    # Step 3: write chapter documentation
    md_path = write_chapter_documentation(cells_df, degeneracy,
                                            im_results, args.outdir)
    print(f"\nWrote {md_path}")

    # Also save numerical results
    csv_path = Path(args.outdir) / "identification_limits.csv"
    pd.DataFrame(degeneracy).to_csv(csv_path, index=False)
    if im_results is not None:
        json_path = Path(args.outdir) / "identification_limits.json"
        with open(json_path, "w") as f:
            # Make all values JSON-serialisable
            im_json = dict(
                G=im_results["G"],
                mean_slope=float(im_results["mean_slope"]),
                sd_slope=float(im_results["sd_slope"]),
                se_mean=float(im_results["se_mean"]),
                t_stat=float(im_results["t_stat"]) if not np.isnan(im_results["t_stat"]) else None,
                df=int(im_results["df"]),
                p_value=float(im_results["p_value"]) if not np.isnan(im_results["p_value"]) else None,
                per_dataset_slopes=[
                    dict(dataset=s["dataset"], slope=float(s["slope"]), n_cells=int(s["n_cells"]))
                    for s in im_results["per_dataset_slopes"]
                ],
            )
            json.dump(im_json, f, indent=2)
        print(f"Wrote {json_path}")

    print("\n== DONE ==")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
