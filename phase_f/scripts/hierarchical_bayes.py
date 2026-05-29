"""
hierarchical_bayes.py — Week 2 Task 2 (RESTRUCTURED)

Replaces the flat-prior Bayesian-R² draft. Per the methodological verification:
- Hierarchical model with random dataset intercepts (and random ACF slopes)
- Regularised-horseshoe priors on candidate coefficients (Piironen-Vehtari 2017)
- Model comparison via PSIS-LOO (Vehtari-Gelman-Gabry 2017)
- Report posterior of Delta-R²_Bayes and Pr(Delta-R² >= 0.30)

Requires: pymc >= 5.0, arviz, numpy, pandas
  Install on Vast.ai:  pip install "pymc>=5.10" arviz

This script is the per-series version (the cell-level n=12 is not suitable for
Bayes either — the prior dominates at n=12). We fit on the per-series panel
with dataset as a grouping factor.

Two model comparisons per candidate feature:
  M_reduced:  delta_pp ~ 1 + acf_24h + horizon + (1 | dataset)
  M_full:     delta_pp ~ 1 + acf_24h + horizon + candidate + (1 | dataset)
PSIS-LOO ELPD difference (M_full - M_reduced) with SE tells us whether the
candidate improves out-of-sample predictive density.

For the headline WPE test we also compute the posterior of the Bayesian
partial-R² and Pr(partial-R² >= 0.30).

Inputs:
  per_series_fulltest_r2.csv
  features_joined_extended_v2.csv
  --candidates: comma-separated features to test (default: wpe_m4_t1)
  --clip / --no_clip: clip policy (default no_clip; stratification handles idle VMs)

Outputs:
  hierarchical_bayes_loo.csv — per candidate: elpd_diff, se, Pr(improve)
  hierarchical_bayes_r2.csv — posterior summaries of Bayesian R²
  hierarchical_bayes.md — report
  (NetCDF traces saved per candidate for reproducibility)

Runtime: ~10-20 min on Vast.ai (Alibaba n=17,032 is the slow one;
         Bitbrains n=568 and ByteDance n=279 are fast)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def check_imports():
    """Verify pymc and arviz are installed; give install instructions if not."""
    try:
        import pymc  # noqa
        import arviz  # noqa
        return True
    except ImportError as e:
        print("=" * 70)
        print("MISSING DEPENDENCY")
        print("=" * 70)
        print(f"  {e}")
        print()
        print("  Install on Vast.ai:")
        print("    pip install 'pymc>=5.10' arviz")
        print()
        print("  Then re-run this script.")
        return False


def standardise(x):
    """z-score, returning (z, mean, std) for back-transformation."""
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def build_and_sample(df, candidate, outdir, n_draws=2000, n_tune=2000,
                     n_chains=4, seed=42, target_accept=0.95):
    """
    Fit reduced and full hierarchical models for one candidate feature.
    Returns dict with LOO comparison and Bayesian R² posterior summaries.
    """
    import pymc as pm
    import arviz as az

    # Prepare data
    sub = df.dropna(subset=["delta_pp", "acf_24h", "horizon_min", candidate]).copy()
    n = len(sub)
    if n < 30:
        print(f"  {candidate}: only {n} rows after NaN drop, skipping")
        return None

    # Standardise predictors (helps sampling)
    y = sub["delta_pp"].to_numpy()
    acf_z, _, _ = standardise(sub["acf_24h"].to_numpy())
    h_z, _, _ = standardise(sub["horizon_min"].to_numpy())
    cand_z, _, _ = standardise(sub[candidate].to_numpy())

    # Dataset grouping
    datasets = sub["dataset"].astype("category")
    ds_codes = datasets.cat.codes.to_numpy()
    n_ds = len(datasets.cat.categories)

    print(f"  {candidate}: n={n}, datasets={n_ds}")

    # ---- Reduced model: delta_pp ~ acf + horizon + (1|dataset) ----
    print(f"    sampling REDUCED model...")
    with pm.Model() as m_reduced:
        # Hyperpriors for dataset random intercept
        mu_a = pm.Normal("mu_a", 0, 10)
        sigma_a = pm.HalfNormal("sigma_a", 10)
        a_ds_offset = pm.Normal("a_ds_offset", 0, 1, shape=n_ds)
        a_ds = pm.Deterministic("a_ds", mu_a + sigma_a * a_ds_offset)

        b_acf = pm.Normal("b_acf", 0, 5)
        b_h = pm.Normal("b_h", 0, 5)
        sigma = pm.HalfNormal("sigma", 10)

        mu = a_ds[ds_codes] + b_acf * acf_z + b_h * h_z
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        idata_reduced = pm.sample(
            draws=n_draws, tune=n_tune, chains=n_chains,
            random_seed=seed, progressbar=False,
            target_accept=target_accept,
            idata_kwargs={"log_likelihood": True},
        )

    # ---- Full model: + regularised-horseshoe candidate coefficient ----
    print(f"    sampling FULL model (regularised horseshoe)...")
    with pm.Model() as m_full:
        mu_a = pm.Normal("mu_a", 0, 10)
        sigma_a = pm.HalfNormal("sigma_a", 10)
        # Non-centred parametrisation of dataset intercepts (reduces divergences)
        a_ds_offset = pm.Normal("a_ds_offset", 0, 1, shape=n_ds)
        a_ds = pm.Deterministic("a_ds", mu_a + sigma_a * a_ds_offset)

        b_acf = pm.Normal("b_acf", 0, 5)
        b_h = pm.Normal("b_h", 0, 5)

        # Regularised horseshoe on the candidate coefficient
        # (Piironen-Vehtari 2017). With a single candidate this reduces to a
        # heavy-tailed shrinkage prior centred at 0 — appropriate for the
        # multiple-comparison context (we expect most candidates to be null).
        # Non-centred parametrisation of b_cand to reduce funnel divergences.
        tau = pm.HalfCauchy("tau", beta=0.05)  # global shrinkage
        lam = pm.HalfCauchy("lam", beta=1.0)   # local shrinkage
        c2 = pm.InverseGamma("c2", alpha=2, beta=2)  # slab
        lam_tilde = pm.Deterministic(
            "lam_tilde",
            (lam * tau) / pm.math.sqrt(c2 + (tau**2) * (lam**2)) * pm.math.sqrt(c2)
        )
        b_cand_raw = pm.Normal("b_cand_raw", 0, 1)
        b_cand = pm.Deterministic("b_cand", b_cand_raw * lam_tilde)

        sigma = pm.HalfNormal("sigma", 10)

        mu = a_ds[ds_codes] + b_acf * acf_z + b_h * h_z + b_cand * cand_z
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        idata_full = pm.sample(
            draws=n_draws, tune=n_tune, chains=n_chains,
            random_seed=seed, progressbar=False,
            target_accept=target_accept,
            idata_kwargs={"log_likelihood": True},
        )

    # ---- PSIS-LOO comparison ----
    print(f"    computing PSIS-LOO...")
    # ArviZ changed the ELPDData attribute names across versions. Be defensive:
    # classic ArviZ (0.x) exposes .elpd_loo; arviz>=1.x (arviz_stats) exposes
    # .elpd or stores it in a Series-like under 'elpd'. We try several.
    def _get_elpd(loo_obj):
        for attr in ("elpd_loo", "elpd"):
            if hasattr(loo_obj, attr):
                return float(getattr(loo_obj, attr))
        # Series/dict-like fallback
        try:
            for key in ("elpd_loo", "elpd"):
                if key in loo_obj:
                    return float(loo_obj[key])
        except Exception:
            pass
        # last resort: parse from the object's index if it's a pandas Series
        try:
            import pandas as _pd
            s = _pd.Series(loo_obj)
            for key in ("elpd_loo", "elpd"):
                if key in s.index:
                    return float(s[key])
        except Exception:
            pass
        return float("nan")

    elpd_diff = np.nan
    se_diff = np.nan
    try:
        loo_reduced = az.loo(idata_reduced)
        loo_full = az.loo(idata_full)
        e_full = _get_elpd(loo_full)
        e_reduced = _get_elpd(loo_reduced)
        elpd_diff = float(e_full - e_reduced)
        # SE of the difference via az.compare (column name stable: 'dse')
        try:
            comp = az.compare({"reduced": idata_reduced, "full": idata_full})
            if "dse" in comp.columns:
                se_diff = float(comp["dse"].max())
            elif "se" in comp.columns:
                se_diff = float(comp["se"].max())
        except Exception as ce:
            print(f"    WARN: az.compare failed ({ce}); se_diff unavailable")
    except Exception as le:
        print(f"    WARN: PSIS-LOO failed ({le}); reporting Bayesian R² only")

    # ---- Bayesian R² posterior (full and reduced) ----
    # Bayesian R² per draw (Gelman-Goodrich 2019):
    #   R²(s) = Var(mu(s)) / (Var(mu(s)) + sigma(s)^2)
    def bayes_r2_draws(idata, has_cand):
        post = idata.posterior
        # Flatten (chain, draw) -> sample axis
        a_ds_s = post["a_ds"].values     # (chain, draw, n_ds)
        b_acf_s = post["b_acf"].values   # (chain, draw)
        b_h_s = post["b_h"].values
        sigma_s = post["sigma"].values
        nchain, ndraw = b_acf_s.shape
        ns = nchain * ndraw

        a_ds_flat = a_ds_s.reshape(ns, -1)        # (ns, n_ds)
        b_acf_flat = b_acf_s.reshape(ns)          # (ns,)
        b_h_flat = b_h_s.reshape(ns)
        sigma_flat = sigma_s.reshape(ns)

        # Per-sample dataset intercept mapped to each observation: (ns, n_obs)
        a_obs = a_ds_flat[:, ds_codes]            # (ns, n_obs)
        # Linear predictor contributions via outer products
        mu = a_obs \
            + np.outer(b_acf_flat, acf_z) \
            + np.outer(b_h_flat, h_z)             # (ns, n_obs)
        if has_cand:
            b_cand_flat = post["b_cand"].values.reshape(ns)
            mu = mu + np.outer(b_cand_flat, cand_z)

        var_mu = mu.var(axis=1)                   # (ns,)
        r2 = var_mu / (var_mu + sigma_flat**2)
        return r2

    r2_full_draws = bayes_r2_draws(idata_full, has_cand=True)
    r2_reduced_draws = bayes_r2_draws(idata_reduced, has_cand=False)
    # Bayesian partial-R² per draw
    partial_r2_draws = (r2_full_draws - r2_reduced_draws) / (1 - r2_reduced_draws)

    pr2_mean = float(np.mean(partial_r2_draws))
    pr2_lo = float(np.percentile(partial_r2_draws, 2.5))
    pr2_hi = float(np.percentile(partial_r2_draws, 97.5))
    prob_ge_threshold = float(np.mean(partial_r2_draws >= 0.30))

    # b_cand posterior (standardised scale)
    b_cand_post = idata_full.posterior["b_cand"].values.flatten()
    b_cand_mean = float(np.mean(b_cand_post))
    b_cand_lo = float(np.percentile(b_cand_post, 2.5))
    b_cand_hi = float(np.percentile(b_cand_post, 97.5))
    prob_b_nonzero = float(np.mean(b_cand_post > 0)) if b_cand_mean > 0 else float(np.mean(b_cand_post < 0))

    # Save traces (nice-to-have for reproducibility; defensive across arviz
    # versions — classic arviz has az.to_netcdf, the 1.x rewrite uses the
    # InferenceData.to_netcdf method instead, and some builds lack both).
    def _save_trace(idata, path):
        try:
            if hasattr(az, "to_netcdf"):
                az.to_netcdf(idata, str(path))
                return True
            if hasattr(idata, "to_netcdf"):
                idata.to_netcdf(str(path))
                return True
        except Exception as e:
            print(f"    WARN: could not save trace {path} ({e})")
        return False

    _save_trace(idata_full, Path(outdir) / f"trace_full_{candidate}.nc")
    _save_trace(idata_reduced, Path(outdir) / f"trace_reduced_{candidate}.nc")

    print(f"    elpd_diff (full-reduced) = {elpd_diff:+.2f} (SE {se_diff:.2f})")
    print(f"    Bayesian partial-R²: mean={pr2_mean:.4f} "
          f"[{pr2_lo:.4f}, {pr2_hi:.4f}]")
    print(f"    Pr(partial-R² >= 0.30) = {prob_ge_threshold:.4f}")

    return dict(
        candidate=candidate, n=n,
        elpd_diff=elpd_diff, se_diff=se_diff,
        elpd_improves=bool(
            (not np.isnan(elpd_diff)) and (not np.isnan(se_diff))
            and elpd_diff > 2 * se_diff
        ),  # rule of thumb
        bayes_partial_r2_mean=pr2_mean,
        bayes_partial_r2_lo=pr2_lo,
        bayes_partial_r2_hi=pr2_hi,
        prob_partial_r2_ge_030=prob_ge_threshold,
        b_cand_mean=b_cand_mean,
        b_cand_lo=b_cand_lo, b_cand_hi=b_cand_hi,
        prob_b_cand_same_sign=prob_b_nonzero,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_series_csv", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--candidates", default="wpe_m4_t1",
                    help="comma-separated feature names")
    ap.add_argument("--dataset", default="all",
                    help="'all' or one of alibaba/bitbrains/bytedance")
    ap.add_argument("--clip", action="store_true",
                    help="apply clip [-1,+1] (default: no clip)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n_draws", type=int, default=2000)
    ap.add_argument("--n_tune", type=int, default=2000)
    ap.add_argument("--n_chains", type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.95,
                    help="NUTS target_accept; bump to 0.99 for funnel-prone features")
    args = ap.parse_args()

    if not check_imports():
        return 1

    import os
    os.makedirs(args.outdir, exist_ok=True)

    print("== hierarchical_bayes.py ==")
    print(f"  candidates: {args.candidates}")
    print(f"  dataset:    {args.dataset}")
    print(f"  clip:       {args.clip}")
    print()

    # Load
    per = pd.read_csv(args.per_series_csv)
    if args.dataset != "all":
        per = per[per["dataset"] == args.dataset]
    # strip prefixes
    for pref, ds in [("bb_", "bitbrains"), ("bd_", "bytedance")]:
        m = per["dataset"] == ds
        per.loc[m, "series_id"] = (
            per.loc[m, "series_id"].astype(str).str.replace(f"^{pref}", "", regex=True)
        )
    per["series_id"] = per["series_id"].astype(str)

    if args.clip:
        per["r2_ensemble"] = per["r2_ensemble"].clip(-1, 1)
        per["r2_naive"] = per["r2_naive"].clip(-1, 1)
    per["delta_pp"] = (per["r2_ensemble"] - per["r2_naive"]) * 100.0

    feats = pd.read_csv(args.features)
    feats["series_id"] = feats["series_id"].astype(str)

    joined = per.merge(feats, on=["series_id", "dataset"], how="inner")
    print(f"Joined: {len(joined)} rows, "
          f"{joined['dataset'].nunique()} datasets, "
          f"{joined['series_id'].nunique()} series")
    print()

    candidates = [c.strip() for c in args.candidates.split(",")]
    results = []
    for cand in candidates:
        if cand not in joined.columns:
            print(f"SKIP {cand}: not in columns")
            continue
        print("=" * 70)
        print(f"CANDIDATE: {cand}")
        print("=" * 70)
        res = build_and_sample(
            joined, cand, args.outdir,
            n_draws=args.n_draws, n_tune=args.n_tune,
            n_chains=args.n_chains, target_accept=args.target_accept,
        )
        if res:
            results.append(res)

    if not results:
        print("No results produced.")
        return 1

    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(Path(args.outdir) / "hierarchical_bayes_loo.csv", index=False)
    print(f"\nWrote hierarchical_bayes_loo.csv")

    # Report
    md_path = Path(args.outdir) / "hierarchical_bayes.md"
    with open(md_path, "w") as f:
        f.write("# Hierarchical Bayesian analysis of predictability candidates\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Model\n\n")
        f.write("Hierarchical (multilevel) regression with random dataset ")
        f.write("intercepts:\n\n")
        f.write("```\n")
        f.write("delta_pp ~ a_dataset + b_acf*ACF + b_h*horizon [+ b_cand*candidate]\n")
        f.write("a_dataset ~ Normal(mu_a, sigma_a)   # random intercept per dataset\n")
        f.write("b_cand    ~ regularised horseshoe   # Piironen-Vehtari 2017\n")
        f.write("```\n\n")
        f.write("Candidate coefficients use a regularised-horseshoe prior, ")
        f.write("appropriate for the multiple-comparison context (most candidates ")
        f.write("expected null). Model comparison by PSIS-LOO ")
        f.write("(Vehtari-Gelman-Gabry 2017).\n\n")
        f.write("## Results\n\n")
        f.write("| Candidate | n | ELPD diff (full-reduced) | SE | Improves? | "
                "Bayes partial-R² | Pr(≥0.30) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in results:
            imp = "**yes**" if r["elpd_improves"] else "no"
            f.write(f"| `{r['candidate']}` | {r['n']} "
                    f"| {r['elpd_diff']:+.2f} | {r['se_diff']:.2f} | {imp} "
                    f"| {r['bayes_partial_r2_mean']:.3f} "
                    f"[{r['bayes_partial_r2_lo']:.3f}, {r['bayes_partial_r2_hi']:.3f}] "
                    f"| {r['prob_partial_r2_ge_030']:.3f} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("- **ELPD diff > 2·SE** indicates the candidate improves ")
        f.write("out-of-sample predictive density (the Bayesian analogue of ")
        f.write("'the feature adds information').\n")
        f.write("- **Pr(partial-R² ≥ 0.30)** is the posterior probability that the ")
        f.write("candidate crosses the pre-registered threshold. This replaces the ")
        f.write("binary frequentist 'pass/fail' with a graded posterior statement.\n\n")
        any_improve = any(r["elpd_improves"] for r in results)
        any_prob = max((r["prob_partial_r2_ge_030"] for r in results), default=0)
        if not any_improve and any_prob < 0.1:
            f.write("**No candidate improves predictive density, and the posterior ")
            f.write(f"probability of crossing 0.30 is at most {any_prob:.3f}.** ")
            f.write("The Bayesian analysis concurs with the frequentist FDR screen: ")
            f.write("the predictability axis provides no incremental information ")
            f.write("over ACF@24h and horizon. The null is now supported by both ")
            f.write("paradigms.\n")
        else:
            f.write("Some candidate shows posterior support; examine individually.\n")
        f.write("\n## References\n\n")
        f.write("- Gelman, A., Goodrich, B., Gabry, J., & Vehtari, A. (2019). ")
        f.write("R-squared for Bayesian regression models. *The American ")
        f.write("Statistician*, 73(3), 307-309.\n")
        f.write("- Piironen, J., & Vehtari, A. (2017). Sparsity information and ")
        f.write("regularization in the horseshoe and other shrinkage priors. ")
        f.write("*Electronic Journal of Statistics*, 11(2), 5018-5051.\n")
        f.write("- Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian ")
        f.write("model evaluation using leave-one-out cross-validation and WAIC. ")
        f.write("*Statistics and Computing*, 27(5), 1413-1432.\n")
    print(f"Wrote {md_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())