"""
f2_partial_r2.py

F2 pre-registered test: partial-R^2(WPE | ACF@24h) >= 0.3
Pre-reg threshold from Dr. Ho written acceptance, 2026-05-22.

Two framings per DECISION-010 (2026-05-27):

  HEADLINE: per-cell pooled, n=12.
    Response: delta_pp from bcf_pairs.csv (NNLS rows, 4 horizons x 3 datasets)
    Predictors: dataset-median WPE, dataset-median ACF@24h, horizon_min
    Cluster-bootstrap 95% CI (cluster = dataset, k=3).

  ROBUSTNESS: Bitbrains per-VM pooled, n ~ 142 x 4 = 568.
    Response: r2_delta * 100 from bitbrains_per_vm.csv (in pp)
    Predictors: per-VM WPE (wpe_bitbrains.csv), per-VM ACF@24h
                (omega_bitbrains.csv), horizon_min
    Cluster-bootstrap 95% CI (cluster = vm_id).

Outputs:
  phase_f/data/f2_partial_r2_results.csv      one row per framing
  phase_f/data/per_series_deltas_bitbrains.csv  intermediate panel

Run from repo root:
  cd E:\\thesis\\kubernetes-cpu-ensemble-thesis
  python phase_f\\scripts\\f2_partial_r2.py 2>&1 |
    Tee-Object phase_f\\scripts\\f2_partial_r2.log
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# config
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

BCF_PAIRS    = REPO_ROOT / "results" / "bcf" / "bcf_pairs.csv"
WPE_ALI      = REPO_ROOT / "phase_f" / "data" / "wpe_alibaba.csv"
WPE_BB       = REPO_ROOT / "phase_f" / "data" / "wpe_bitbrains.csv"
WPE_BYT      = REPO_ROOT / "phase_f" / "data" / "wpe_bytedance.csv"
OMEGA_ALI    = REPO_ROOT / "results" / "alibaba" / "omega_alibaba.csv"
OMEGA_BB     = REPO_ROOT / "results" / "bitbrains" / "omega_bitbrains.csv"
STATS_BYT    = REPO_ROOT / "results" / "bytedance" / "bytedance_per_instance_stats.csv"
BB_PERVM     = REPO_ROOT / "results" / "bitbrains" / "diagnostics" / "bitbrains_per_vm.csv"

OUT_RESULTS  = REPO_ROOT / "phase_f" / "data" / "f2_partial_r2_results.csv"
OUT_BB_JOIN  = REPO_ROOT / "phase_f" / "data" / "per_series_deltas_bitbrains.csv"

N_BOOT       = 5000
SEED         = 20260527
THRESHOLD    = 0.30   # pre-reg

HORIZON_TO_MIN = {"10min": 10, "30min": 30, "60min": 60, "120min": 120}


# ============================================================
# helpers
# ============================================================

def parse_horizon(s):
    """'30min' -> 30."""
    return int(str(s).replace("min", ""))


def fit_partial_r2(df, y_col, full_predictors, drop_var):
    """
    Compute partial-R^2 of `drop_var` controlling for the remaining
    predictors.

    partial-R^2 = (R^2_full - R^2_reduced) / (1 - R^2_reduced)

    Returns: partial_r2, r2_full, r2_reduced, coef_dropvar, n
    """
    y = df[y_col].values

    X_full = sm.add_constant(df[full_predictors].values)
    X_red  = sm.add_constant(df[[c for c in full_predictors
                                 if c != drop_var]].values)

    m_full = sm.OLS(y, X_full).fit()
    m_red  = sm.OLS(y, X_red).fit()

    r2_full = m_full.rsquared
    r2_red  = m_red.rsquared

    if (1.0 - r2_red) < 1e-12:
        partial = float("nan")
    else:
        partial = (r2_full - r2_red) / (1.0 - r2_red)

    # coefficient of the dropped variable in the full model
    # add_constant prepends a const column, so the drop_var's index
    # in the X matrix is 1 + its index in full_predictors
    drop_idx = full_predictors.index(drop_var)
    coef = m_full.params[1 + drop_idx]

    return partial, r2_full, r2_red, coef, len(df)


def compute_vif(df, predictors):
    """
    Variance inflation factor for each predictor.
    VIF > 10 = problematic collinearity; > 5 = warning.
    """
    X = sm.add_constant(df[predictors].values)
    vifs = {}
    for i, p in enumerate(predictors):
        try:
            v = float(variance_inflation_factor(X, i + 1))
        except Exception:
            v = float("nan")
        vifs[p] = v
    return vifs


def cluster_bootstrap_partial_r2(df, y_col, full_predictors, drop_var,
                                  cluster_col, n_boot=5000, seed=20260527):
    """
    Cluster-bootstrap partial-R^2.

    Resample cluster IDs with replacement; for each chosen cluster take
    all rows. Compute partial-R^2 per resample. Return percentile 95%
    CI bounds plus a count of failed (non-finite) resamples.
    """
    rng = np.random.default_rng(seed)

    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    cluster_to_rows = {c: np.where(df[cluster_col].values == c)[0]
                       for c in clusters}

    boot_vals = np.empty(n_boot, dtype=np.float64)
    n_failed = 0

    for b in range(n_boot):
        picked = rng.choice(clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_rows[c] for c in picked])
        boot_df = df.iloc[idx]

        try:
            partial, _, _, _, _ = fit_partial_r2(
                boot_df, y_col, full_predictors, drop_var
            )
            boot_vals[b] = partial
        except Exception:
            boot_vals[b] = np.nan
            n_failed += 1

    finite = boot_vals[np.isfinite(boot_vals)]
    if len(finite) < 100:
        print(f"  WARNING: only {len(finite)} finite bootstrap values "
              f"(of {n_boot}); CI may be unreliable.")
        n_failed = n_boot - len(finite)

    ci_low  = float(np.percentile(finite, 2.5))
    ci_high = float(np.percentile(finite, 97.5))

    return ci_low, ci_high, boot_vals, n_failed


# ============================================================
# HEADLINE: per-cell pooled, n=12
# ============================================================

def run_headline():
    print("=" * 68)
    print("HEADLINE: per-cell pooled, n=12")
    print("=" * 68)

    # ---- step 1: load bcf_pairs.csv, keep NNLS rows only -----------
    print("\n[1] loading bcf_pairs.csv ...")
    pairs = pd.read_csv(BCF_PAIRS)
    print(f"    total rows: {len(pairs)}")
    print(f"    models present: {sorted(pairs['model'].unique())}")

    nnls = pairs[pairs["model"] == "NNLS"].copy()
    nnls["horizon_min"] = nnls["horizon"].map(parse_horizon)
    print(f"    NNLS rows: {len(nnls)}")
    print(f"    NNLS by dataset: {nnls['dataset'].value_counts().to_dict()}")

    has_byt_h10 = ((nnls["dataset"] == "ByteDance") &
                   (nnls["horizon"] == "10min")).any()
    print(f"    ByteDance h10 present? {has_byt_h10}  "
          f"(D2 incidental finding; expected True)")

    # ---- step 2: per-dataset WPE medians ---------------------------
    print("\n[2] per-dataset WPE medians (from D4 outputs) ...")
    wpe_ali_df = pd.read_csv(WPE_ALI).rename(columns={"wpe_m4_t1": "wpe"})
    wpe_bb_df  = pd.read_csv(WPE_BB).rename(columns={"wpe_m4_t1": "wpe"})
    wpe_byt_df = pd.read_csv(WPE_BYT).rename(columns={"wpe_m4_t1": "wpe"})
    print(f"    wpe_alibaba   cols={list(wpe_ali_df.columns)}  "
          f"rows={len(wpe_ali_df)}")
    print(f"    wpe_bitbrains cols={list(wpe_bb_df.columns)}  "
          f"rows={len(wpe_bb_df)}")
    print(f"    wpe_bytedance cols={list(wpe_byt_df.columns)}  "
          f"rows={len(wpe_byt_df)}")

    for name, dfw in [("alibaba", wpe_ali_df),
                      ("bitbrains", wpe_bb_df),
                      ("bytedance", wpe_byt_df)]:
        if "wpe" not in dfw.columns:
            raise SystemExit(f"wpe_{name}.csv missing 'wpe' column; "
                             f"got {list(dfw.columns)}")

    wpe_map = {
        "Alibaba":   float(wpe_ali_df["wpe"].median()),
        "Bitbrains": float(wpe_bb_df["wpe"].median()),
        "ByteDance": float(wpe_byt_df["wpe"].median()),
    }
    print(f"    WPE medians: {wpe_map}")

    # sanity vs D4 journal expected values
    expected = {"Alibaba": 0.5667, "Bitbrains": 0.7276, "ByteDance": 0.9545}
    for ds, want in expected.items():
        got = wpe_map[ds]
        flag = "OK" if abs(got - want) < 0.005 else "DRIFT"
        print(f"    sanity {ds}: got {got:.4f} vs D4 journal {want:.4f} "
              f"[{flag}]")

    # ---- step 3: per-dataset ACF@24h medians -----------------------
    print("\n[3] per-dataset ACF@24h medians (from omega/stats files) ...")
    omega_ali = pd.read_csv(OMEGA_ALI)
    omega_bb  = pd.read_csv(OMEGA_BB)
    stats_byt = pd.read_csv(STATS_BYT)
    print(f"    omega_alibaba                 rows={len(omega_ali)}")
    print(f"    omega_bitbrains               rows={len(omega_bb)}")
    print(f"    bytedance_per_instance_stats  rows={len(stats_byt)}")

    acf_map = {
        "Alibaba":   float(omega_ali["acf_24h"].median()),
        "Bitbrains": float(omega_bb["acf_24h"].median()),
        "ByteDance": float(stats_byt["acf_24h"].median()),
    }
    print(f"    ACF@24h medians: {acf_map}")

    # ---- step 4: join into per-cell design matrix ------------------
    print("\n[4] joining into per-cell design matrix ...")
    cell = nnls[["dataset", "horizon", "horizon_min", "delta_pp"]].copy()
    cell["wpe"]     = cell["dataset"].map(wpe_map)
    cell["acf_24h"] = cell["dataset"].map(acf_map)
    cell = cell.sort_values(["dataset", "horizon_min"]).reset_index(drop=True)
    print(f"    final design matrix: {len(cell)} rows")
    print(cell.to_string(index=False))

    print("\n[4b] cross-dataset gut check:")
    for ds in ["Alibaba", "Bitbrains", "ByteDance"]:
        sub = cell[cell["dataset"] == ds]
        print(f"    {ds}: wpe={sub['wpe'].iloc[0]:.4f}  "
              f"acf24h={sub['acf_24h'].iloc[0]:.4f}  "
              f"mean(delta_pp)={sub['delta_pp'].mean():+.2f}")

    # ---- step 5: fit full / reduced; compute partial-R^2 -----------
    print("\n[5] regression — full vs reduced ...")
    full_preds = ["wpe", "acf_24h", "horizon_min"]
    partial, r2_full, r2_red, coef_wpe, n = fit_partial_r2(
        cell, y_col="delta_pp", full_predictors=full_preds, drop_var="wpe"
    )
    print(f"    R^2 full (WPE + ACF + horizon):     {r2_full:.4f}")
    print(f"    R^2 reduced (ACF + horizon):        {r2_red:.4f}")
    print(f"    coef(WPE) in full model:            {coef_wpe:+.4f}")
    if coef_wpe < 0:
        sign_note = ("negative — higher entropy -> less ML benefit "
                     "(matches intuition)")
    else:
        sign_note = ("POSITIVE — higher entropy -> more ML benefit "
                     "(inverts the predictability story; disclose)")
    print(f"    sign of WPE coef:                   {sign_note}")
    print(f"    partial-R^2(WPE | ACF, horizon):    {partial:.4f}")
    print(f"    pre-reg threshold:                  {THRESHOLD}")
    print(f"    passes pre-reg?                     "
          f"{'YES' if partial >= THRESHOLD else 'NO'}")

    # ---- step 5b: sensitivity — drop horizon covariate -------------
    print("\n[5b] sensitivity — drop horizon covariate ...")
    partial_noh, r2_full_noh, r2_red_noh, coef_wpe_noh, _ = fit_partial_r2(
        cell, y_col="delta_pp",
        full_predictors=["wpe", "acf_24h"], drop_var="wpe"
    )
    print(f"    partial-R^2(WPE | ACF, no horizon): {partial_noh:.4f}")
    print(f"    coef(WPE) in this model:            {coef_wpe_noh:+.4f}")

    # ---- step 6: VIF -----------------------------------------------
    print("\n[6] VIF (collinearity diagnostic) ...")
    vifs = compute_vif(cell, full_preds)
    for p, v in vifs.items():
        flag = "OK" if v < 5 else ("WARN" if v < 10 else "PROBLEM")
        print(f"    VIF({p}) = {v:.2f}  [{flag}]")
    print("    NOTE: n=12 with WPE and ACF varying only across 3 datasets")
    print("    means high VIF is structural, not a bug. It is the reason")
    print("    partial-R^2 is bounded by the between-dataset variance.")

    # ---- step 7: cluster-bootstrap CI ------------------------------
    print(f"\n[7] cluster-bootstrap 95% CI (n_boot={N_BOOT}, "
          f"cluster=dataset, k=3) ...")
    ci_low, ci_high, _, n_failed = cluster_bootstrap_partial_r2(
        cell, y_col="delta_pp", full_predictors=full_preds, drop_var="wpe",
        cluster_col="dataset", n_boot=N_BOOT, seed=SEED
    )
    print(f"    failed resamples: {n_failed} / {N_BOOT}")
    print(f"    95% percentile CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print("    NOTE: k=3 clusters means many resamples will repeat one or")
    print("    two datasets; CI is correspondingly wide.")

    # ---- step 8: residual diagnostics per dataset ------------------
    print("\n[8] residual diagnostics (per-dataset bias) ...")
    X_full = sm.add_constant(cell[full_preds].values)
    m_full = sm.OLS(cell["delta_pp"].values, X_full).fit()
    resid = cell["delta_pp"].values - m_full.fittedvalues
    cell_r = cell.copy()
    cell_r["resid"] = resid
    for ds in ["Alibaba", "Bitbrains", "ByteDance"]:
        sub_r = cell_r[cell_r["dataset"] == ds]["resid"]
        print(f"    {ds}: mean(resid)={sub_r.mean():+.3f}  "
              f"sd(resid)={sub_r.std():.3f}  "
              f"max|resid|={np.abs(sub_r).max():.3f}")

    return {
        "framing":              "headline_per_cell",
        "n":                    n,
        "partial_r2":           partial,
        "ci_low":               ci_low,
        "ci_high":              ci_high,
        "r2_full":              r2_full,
        "r2_reduced":           r2_red,
        "coef_wpe":             coef_wpe,
        "vif_wpe":              vifs.get("wpe", float("nan")),
        "vif_acf_24h":          vifs.get("acf_24h", float("nan")),
        "passes_threshold":     bool(partial >= THRESHOLD),
        "partial_r2_nohorizon": partial_noh,
    }


# ============================================================
# ROBUSTNESS: Bitbrains per-VM pooled
# ============================================================

def run_bitbrains_pervm():
    print("\n" + "=" * 68)
    print("ROBUSTNESS: Bitbrains per-VM pooled")
    print("=" * 68)

    # ---- step 1: load per-VM r2 ------------------------------------
    print("\n[1] loading bitbrains_per_vm.csv ...")
    per_vm = pd.read_csv(BB_PERVM)
    per_vm["horizon_min"] = per_vm["horizon"].map(parse_horizon)
    print(f"    rows: {len(per_vm)}")
    print(f"    unique VMs: {per_vm['vm_id'].nunique()}")
    print(f"    horizons: {sorted(per_vm['horizon'].unique())}")

    # r2_delta is in fractional R^2 scale (ml_r2 - naive_r2)
    # multiply by 100 to get pp, matching bcf_pairs.csv delta_pp scale
    per_vm["delta_pp"] = per_vm["r2_delta"] * 100.0
    print(f"    delta_pp summary (pp): "
          f"min={per_vm['delta_pp'].min():+.2f} "
          f"med={per_vm['delta_pp'].median():+.2f} "
          f"max={per_vm['delta_pp'].max():+.2f}")

    # ---- step 2: load per-VM WPE -----------------------------------
    print("\n[2] loading wpe_bitbrains.csv ...")
    wpe_bb = pd.read_csv(WPE_BB).rename(columns={"wpe_m4_t1": "wpe", "container_id": "vm_id"})
    # coerce: omega has int vm_id; wpe came from str container_id
    # strip "bb_" prefix that D4 WPE script added to container_id
    wpe_bb["vm_id"] = (
        wpe_bb["vm_id"].astype(str)
                       .str.replace("bb_", "", regex=False)
                       .astype(int)
    )
    print(f"    rows: {len(wpe_bb)}  cols: {list(wpe_bb.columns)}")
    if "vm_id" not in wpe_bb.columns:
        for alt in ["series_id", "id"]:
            if alt in wpe_bb.columns:
                print(f"    renaming '{alt}' -> 'vm_id'")
                wpe_bb = wpe_bb.rename(columns={alt: "vm_id"})
                break
        else:
            raise SystemExit(f"wpe_bitbrains.csv has no vm_id column; "
                             f"got {list(wpe_bb.columns)}")

    # ---- step 3: load per-VM ACF@24h -------------------------------
    print("\n[3] loading omega_bitbrains.csv ...")
    omega_bb = pd.read_csv(OMEGA_BB)
    print(f"    rows: {len(omega_bb)}  cols: {list(omega_bb.columns)}")
    if "vm_id" not in omega_bb.columns:
        raise SystemExit(f"omega_bitbrains.csv has no vm_id column; "
                         f"got {list(omega_bb.columns)}")

    # ---- step 4: join WPE + ACF at VM level ------------------------
    print("\n[4] joining per-VM predictors (inner join) ...")
    vm_predictors = wpe_bb[["vm_id", "wpe"]].merge(
        omega_bb[["vm_id", "acf_24h"]], on="vm_id", how="inner"
    )
    print(f"    VMs with both WPE and ACF: {len(vm_predictors)}")
    print(f"    (expected ~142 per D4 journal)")

    # ---- step 5: join with per-VM x horizon delta_pp ---------------
    print("\n[5] joining design matrix ...")
    panel = per_vm[["vm_id", "horizon", "horizon_min", "delta_pp"]].merge(
        vm_predictors, on="vm_id", how="inner"
    )
    panel = panel.sort_values(["vm_id", "horizon_min"]).reset_index(drop=True)
    print(f"    final panel rows: {len(panel)}  "
          f"(should be {len(vm_predictors)} VMs x 4 horizons)")
    print(f"    head:")
    print(panel.head(8).to_string(index=False))

    OUT_BB_JOIN.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_BB_JOIN, index=False)
    print(f"    saved: {OUT_BB_JOIN.relative_to(REPO_ROOT)}")

    # ---- step 6: regression ----------------------------------------
    print("\n[6] regression — full vs reduced ...")
    full_preds = ["wpe", "acf_24h", "horizon_min"]
    partial, r2_full, r2_red, coef_wpe, n = fit_partial_r2(
        panel, y_col="delta_pp", full_predictors=full_preds, drop_var="wpe"
    )
    print(f"    R^2 full:                           {r2_full:.4f}")
    print(f"    R^2 reduced:                        {r2_red:.4f}")
    print(f"    coef(WPE):                          {coef_wpe:+.4f}")
    print(f"    partial-R^2(WPE | ACF, horizon):    {partial:.4f}")
    print(f"    passes pre-reg (>= 0.30)?           "
          f"{'YES' if partial >= THRESHOLD else 'NO'}")

    # ---- step 6b: sensitivity --------------------------------------
    print("\n[6b] sensitivity — drop horizon covariate ...")
    partial_noh, _, _, coef_wpe_noh, _ = fit_partial_r2(
        panel, y_col="delta_pp",
        full_predictors=["wpe", "acf_24h"], drop_var="wpe"
    )
    print(f"    partial-R^2(WPE | ACF, no horizon): {partial_noh:.4f}")
    print(f"    coef(WPE) in this model:            {coef_wpe_noh:+.4f}")

    # ---- step 7: VIF -----------------------------------------------
    print("\n[7] VIF (per-VM design has real cross-VM variation) ...")
    vifs = compute_vif(panel, full_preds)
    for p, v in vifs.items():
        flag = "OK" if v < 5 else ("WARN" if v < 10 else "PROBLEM")
        print(f"    VIF({p}) = {v:.2f}  [{flag}]")
    print("    NOTE: D4 within-Bitbrains Spearman(WPE, ACF) = +0.7553,")
    print("    so high VIF on these two is expected and bounds partial-R^2.")

    # ---- step 8: cluster-bootstrap by vm_id ------------------------
    print(f"\n[8] cluster-bootstrap by vm_id (n_boot={N_BOOT}) ...")
    ci_low, ci_high, _, n_failed = cluster_bootstrap_partial_r2(
        panel, y_col="delta_pp", full_predictors=full_preds, drop_var="wpe",
        cluster_col="vm_id", n_boot=N_BOOT, seed=SEED + 1
    )
    print(f"    failed resamples: {n_failed} / {N_BOOT}")
    print(f"    95% percentile CI: [{ci_low:.4f}, {ci_high:.4f}]")

    return {
        "framing":              "robustness_bitbrains_pervm",
        "n":                    n,
        "partial_r2":           partial,
        "ci_low":               ci_low,
        "ci_high":              ci_high,
        "r2_full":              r2_full,
        "r2_reduced":           r2_red,
        "coef_wpe":             coef_wpe,
        "vif_wpe":              vifs.get("wpe", float("nan")),
        "vif_acf_24h":          vifs.get("acf_24h", float("nan")),
        "passes_threshold":     bool(partial >= THRESHOLD),
        "partial_r2_nohorizon": partial_noh,
    }


# ============================================================
# main
# ============================================================

def main():
    print("F2 partial-R^2(WPE | ACF@24h) test")
    print("Pre-reg threshold: >= 0.30  (Dr. Ho written acceptance 2026-05-22)")
    print("DECISION-010 (2026-05-27): per-cell headline + Bitbrains per-VM "
          "robustness")
    print(f"seed={SEED}  n_boot={N_BOOT}")
    print()

    head_result = run_headline()
    bb_result   = run_bitbrains_pervm()

    # ---- write combined results ------------------------------------
    print("\n" + "=" * 68)
    print("WRITING RESULTS")
    print("=" * 68)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame([head_result, bb_result])
    cols_order = ["framing", "n", "partial_r2", "ci_low", "ci_high",
                  "passes_threshold", "r2_full", "r2_reduced",
                  "coef_wpe", "partial_r2_nohorizon",
                  "vif_wpe", "vif_acf_24h"]
    df_out = df_out[cols_order]
    df_out.to_csv(OUT_RESULTS, index=False)
    print(f"\nsaved: {OUT_RESULTS.relative_to(REPO_ROOT)}")
    print(df_out.to_string(index=False))

    # ---- final headline --------------------------------------------
    print("\n" + "=" * 68)
    print("F2 VERDICT")
    print("=" * 68)
    head_passes = head_result["passes_threshold"]
    print(f"  Headline (per-cell, n=12):  "
          f"partial-R^2 = {head_result['partial_r2']:.4f}  "
          f"CI [{head_result['ci_low']:.4f}, {head_result['ci_high']:.4f}]  "
          f"=> {'PASSES' if head_passes else 'BELOW THRESHOLD'}")
    bb_passes = bb_result["passes_threshold"]
    print(f"  Bitbrains per-VM (n={bb_result['n']}):  "
          f"partial-R^2 = {bb_result['partial_r2']:.4f}  "
          f"CI [{bb_result['ci_low']:.4f}, {bb_result['ci_high']:.4f}]  "
          f"=> {'PASSES' if bb_passes else 'BELOW THRESHOLD'}")

    if not head_passes:
        print("")
        print("  Headline below 0.30 threshold. F2 reports a null result")
        print("  under the pre-registered design. Discussion section frames")
        print("  the negative result honestly per DECISION-005 ('If any")
        print("  phase misses threshold, result reported honestly as null /")
        print("  below-threshold (no post-hoc adjustment)').")

    print("\ndone.")


if __name__ == "__main__":
    sys.exit(main())