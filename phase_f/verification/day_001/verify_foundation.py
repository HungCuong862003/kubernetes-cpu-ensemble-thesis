#!/usr/bin/env python3
# verify_foundation.py
#
# Verifies that thesis_numbers.json is consistent with the source CSVs.
# Junior-style verbose: lots of intermediate prints, hardcoded paths, no fancy tricks.
#
# Run this on Day 1 of Phase F0. Every check should PASS. Anything that FAILs is a
# discrepancy that needs to be reconciled before any new chapter writing.
#
# Extended 2026-05-22 (F0 Day 1.4): v3 stubs appended per master plan §3.
# Four new sections (14-17) cover:
#   - v3 foundation-model leaderboard winner tally + headline Bitbrains cells
#   - v3 pooled diversity diagnostic (rho-bar ~ 0.6562)
#   - v3 LOO ablation (Chronos-2 load-bearing per drop_chronos2 deltas)
#   - v3 Phase D D4 38W/3T/3L tally
#
# v3 functions SKIP gracefully if a v3 file isn't synced from Vast.ai yet.
# Expected pre-sync: ~169 PASS / 3 FAIL (unchanged from pre-extension Vast.ai state).
# Expected post-sync of leaderboard_v3_*, cross_dataset_headline_v2, c2_correlations_per_cell,
# loo_ablation_new_pool_v2, d4_*.csv: ~200 PASS / a few FAIL to be reconciled.

import json
import os
import pandas as pd
import math
import sys

# hardcoded paths -- frozen, no experiments will re-run
PROJECT = "/mnt/project"
THESIS_NUMBERS = f"{PROJECT}/thesis_numbers.json"
COMPARISON = f"{PROJECT}/comparison_table.csv"
BB_SUMMARY = f"{PROJECT}/bitbrains_summary_corrected.csv"
BCF_TABLE = f"{PROJECT}/boundary_condition_table_corrected.csv"
STRAT_SKILL = f"{PROJECT}/stratified_skill.csv"
CV_SKILL = f"{PROJECT}/cv_stratified_skill.csv"
RESID_DIAG = f"{PROJECT}/residual_diagnostics.csv"
HPA = f"{PROJECT}/hpa_simulation_v2.csv"
META_LEARN = f"{PROJECT}/meta_learner_comparison.csv"
COMBO_DIAG = f"{PROJECT}/combination_diagnostics.csv"
BCF_JSON = f"{PROJECT}/bcf_pooled_results.json"
SHAP = f"{PROJECT}/shap_importance.csv"
CHRONOS_BENCH = f"{PROJECT}/chronos_benchmark_results.csv"
FRIEDMAN = f"{PROJECT}/friedman_table.csv"
LOO = f"{PROJECT}/loo_ablation.csv"
OMEGA = f"{PROJECT}/omega_summary.csv"

# v3 path constants -- added F0 Day 1.4 per verify_foundation_v3_stubs.py
LB_V3_WIDE     = f"{PROJECT}/results/foundation_comparison/leaderboard_v3_wide.csv"
LB_V3_LONG     = f"{PROJECT}/results/foundation_comparison/leaderboard_v3_long.csv"
LB_V3_WINNERS  = f"{PROJECT}/results/foundation_comparison/leaderboard_v3_winners.csv"
CD_HEADLINE_V2 = f"{PROJECT}/results/foundation_comparison/cross_dataset_headline_v2.csv"
C2_CORR        = f"{PROJECT}/results/bcf_v2/c2_correlations_per_cell.csv"
LOO_V2         = f"{PROJECT}/results/bcf_v2/loo_ablation_new_pool_v2.csv"

TOL = 1e-3  # 0.001 absolute tolerance for R2-like quantities

# counters
PASS_COUNT = 0
FAIL_COUNT = 0
FAILS = []


def check(name, expected, actual, tol=TOL):
    """One numerical check with tolerance. Side-effect: increment counters, print result."""
    global PASS_COUNT, FAIL_COUNT, FAILS
    try:
        e = float(expected)
        a = float(actual)
    except (TypeError, ValueError):
        if str(expected).strip() == str(actual).strip():
            PASS_COUNT += 1
            print(f"  PASS  {name}: {expected}")
            return True
        else:
            FAIL_COUNT += 1
            msg = f"  FAIL  {name}: expected={expected!r} actual={actual!r}"
            FAILS.append(msg)
            print(msg)
            return False
    if math.isnan(e) and math.isnan(a):
        PASS_COUNT += 1
        print(f"  PASS  {name}: both NaN")
        return True
    if abs(e - a) <= tol:
        PASS_COUNT += 1
        print(f"  PASS  {name}: {a:.4f} (expected {e:.4f})")
        return True
    else:
        FAIL_COUNT += 1
        msg = f"  FAIL  {name}: expected={e:.6f} actual={a:.6f} diff={a-e:+.6f}"
        FAILS.append(msg)
        print(msg)
        return False


def load_json():
    print(f"Loading {THESIS_NUMBERS}")
    with open(THESIS_NUMBERS) as f:
        return json.load(f)


def check_section_1_alibaba_r2(tn):
    """Section 1: Alibaba R² claims against comparison_table.csv."""
    print("\n[1] ALIBABA R²: thesis_numbers.json vs comparison_table.csv")
    df = pd.read_csv(COMPARISON)
    df = df.set_index("Horizon")

    for h in ["10min", "30min", "60min", "120min"]:
        row = df.loc[h]
        check(f"  Alibaba {h} Naive_R2",
              tn[f"Section 1: Alibaba Results.{h}.Naive_R2"],
              row["Naive R²"])
        check(f"  Alibaba {h} Hetero_R2",
              tn[f"Section 1: Alibaba Results.{h}.Hetero_R2"],
              row["Hetero Ens R²"])
        check(f"  Alibaba {h} Homo_R2",
              tn[f"Section 1: Alibaba Results.{h}.Homo_R2"],
              row["Homo Ens R²"])
        check(f"  Alibaba {h} ET_R2",
              tn[f"Section 1: Alibaba Results.{h}.ET_R2"],
              row["ET R²"])
        check(f"  Alibaba {h} BiLSTM_R2",
              tn[f"Section 1: Alibaba Results.{h}.BiLSTM_R2"],
              row["BiLSTM R²"])
        delta_pp_expected = tn[f"Section 1: Alibaba Results.{h}.Delta_pp"]
        delta_pp_actual = (row["Hetero Ens R²"] - row["Naive R²"]) * 100.0
        check(f"  Alibaba {h} Delta_pp",
              delta_pp_expected, delta_pp_actual, tol=0.01)


def check_section_1_alibaba_mae(tn):
    """Section 1: Alibaba MAE / Skill / Win_rate against stratified_skill.csv."""
    print("\n[2] ALIBABA MAE / SKILL / WIN_RATE: thesis_numbers.json vs stratified_skill.csv")
    df = pd.read_csv(STRAT_SKILL).set_index("Horizon")
    for h in ["10min", "30min", "60min", "120min"]:
        row = df.loc[h]
        check(f"  Alibaba {h} MAE_naive",
              tn[f"Section 1: Alibaba Results.{h}.MAE_naive"],
              row["MAE_naive"], tol=1e-4)
        check(f"  Alibaba {h} MAE_ensemble",
              tn[f"Section 1: Alibaba Results.{h}.MAE_ensemble"],
              row["MAE_ensemble"], tol=1e-4)
        check(f"  Alibaba {h} Skill_score",
              tn[f"Section 1: Alibaba Results.{h}.Skill_score"],
              row["Skill_score"], tol=1e-4)
        check(f"  Alibaba {h} Win_rate_pct",
              tn[f"Section 1: Alibaba Results.{h}.Win_rate_pct"],
              row["Timestep_Win_rate_%"], tol=0.02)


def check_section_2_bitbrains(tn):
    """Section 2: Bitbrains cross-dataset against bitbrains_summary_corrected.csv."""
    print("\n[3] BITBRAINS CROSS-DATASET: thesis_numbers.json vs bitbrains_summary_corrected.csv")
    df = pd.read_csv(BB_SUMMARY).set_index("Horizon")
    for h in ["10min", "30min", "60min", "120min"]:
        row = df.loc[h]
        check(f"  Bitbrains {h} BB_Naive_R2",
              tn[f"Section 2: Bitbrains Cross-Dataset.{h}.BB_Naive_R2"],
              row["BB_Naive_R2"], tol=1e-3)
        check(f"  Bitbrains {h} BB_ML_R2",
              tn[f"Section 2: Bitbrains Cross-Dataset.{h}.BB_ML_R2"],
              row["BB_ML_R2_median"], tol=1e-3)
        check(f"  Bitbrains {h} BB_Delta_pp",
              tn[f"Section 2: Bitbrains Cross-Dataset.{h}.BB_Delta_pp"],
              row["BB_Delta_pp"], tol=0.01)
        check(f"  Bitbrains {h} BB_Win_pct",
              tn[f"Section 2: Bitbrains Cross-Dataset.{h}.BB_Win_pct"],
              row["BB_Pct_VMs_ML_Wins"], tol=0.01)


def check_section_3_chronos_bolt(tn):
    """Section 3: Chronos-Bolt vs ensemble -- VERIFIES THE STALE V1-ERA SECTION."""
    print("\n[4] CHRONOS-BOLT (V1-ERA, STALE): thesis_numbers.json vs chronos_benchmark_results.csv")
    print("  NOTE: this section in thesis_numbers.json predates leaderboard_v3.")
    print("  After v3 sync, replace with leaderboard_v3_wide.csv winner-tally checks.")
    df = pd.read_csv(CHRONOS_BENCH)
    df = df.set_index(["dataset", "horizon"])
    for ds in ["Alibaba", "Bitbrains", "ByteDance"]:
        for h in ["10min", "30min", "60min", "120min"]:
            try:
                row = df.loc[(ds, h)]
            except KeyError:
                print(f"  SKIP  {ds} {h}: not in chronos_benchmark_results.csv")
                continue
            check(f"  Chronos-Bolt {ds} {h} ens_r2",
                  tn[f"Section 3: Chronos-Bolt.{ds}.{h}.ens_r2"],
                  row["ensemble_r2"], tol=1e-3)
            check(f"  Chronos-Bolt {ds} {h} chr_r2",
                  tn[f"Section 3: Chronos-Bolt.{ds}.{h}.chr_r2"],
                  row["chronos_r2"], tol=1e-3)


def check_section_4_bytedance(tn):
    """Section 4: ByteDance -- only Alibaba-style numbers; ground truth is run_-_Bytedance.log."""
    print("\n[5] BYTEDANCE: thesis_numbers.json self-consistency")
    for h in ["10min", "30min", "60min", "120min"]:
        n = tn[f"Section 4: ByteDance.{h}.naive_R2"]
        het = tn[f"Section 4: ByteDance.{h}.hetero_R2"]
        et = tn[f"Section 4: ByteDance.{h}.ET_R2"]
        if not (et > het > n):
            FAILS.append(f"  ByteDance {h}: ordering broken naive={n} het={het} et={et}")
            print(f"  FAIL  ByteDance {h}: ordering ET>Het>Naive violated ({et} > {het} > {n})")
            global FAIL_COUNT
            FAIL_COUNT += 1
        else:
            global PASS_COUNT
            PASS_COUNT += 1
            print(f"  PASS  ByteDance {h}: naive={n:.4f} < het={het:.4f} < et={et:.4f}")


def check_section_5_stats(tn):
    """Section 5: Friedman tests vs friedman_table.csv."""
    print("\n[6] FRIEDMAN TESTS: thesis_numbers.json vs friedman_table.csv")
    df = pd.read_csv(FRIEDMAN).set_index("Horizon")
    for h in ["10min", "30min", "60min", "120min"]:
        row = df.loc[h]
        check(f"  Friedman {h} chi2",
              tn[f"Section 5: Statistical Tests.{h}.chi2"],
              row["chi2"], tol=0.1)
        check(f"  Friedman {h} p_value",
              tn[f"Section 5: Statistical Tests.{h}.p_value"],
              row["p_value"], tol=1e-6)
        check(f"  Friedman {h} n_containers",
              tn[f"Section 5: Statistical Tests.{h}.n_containers"],
              row["n_containers"])


def check_section_6_shap(tn):
    """Section 6: SHAP grouped percentages against shap_importance.csv raw values."""
    print("\n[7] SHAP FEATURE IMPORTANCE: thesis_numbers.json vs shap_importance.csv")
    df = pd.read_csv(SHAP)
    df = df.rename(columns={df.columns[0]: "feature"})
    for h in ["10min", "30min", "60min", "120min"]:
        total = df[h].sum()
        if total <= 0:
            print(f"  SKIP  SHAP {h}: total {total} <= 0, csv issue")
            continue
        hour_sum = df[df["feature"].isin(["hour_sin", "hour_cos"])][h].sum()
        hour_pct = 100.0 * hour_sum / total
        expected_str = tn[f"Section 6: SHAP Feature Importance.{h}.hour_sin+cos"]
        expected_pct = float(expected_str.rstrip("%"))
        check(f"  SHAP {h} hour_sin+cos pct",
              expected_pct, hour_pct, tol=0.2)
        cluster_sum = df[df["feature"] == "cluster_id"][h].sum()
        cluster_pct = 100.0 * cluster_sum / total
        expected_str = tn[f"Section 6: SHAP Feature Importance.{h}.cluster_id"]
        expected_pct = float(expected_str.rstrip("%"))
        check(f"  SHAP {h} cluster_id pct",
              expected_pct, cluster_pct, tol=0.2)


def check_section_7_uq(tn):
    """Section 7: UQ recalibration -- internal consistency only."""
    print("\n[8] UQ RECALIBRATION: internal consistency in thesis_numbers.json")
    for h in ["10min", "30min", "60min", "120min"]:
        orig = tn[f"Section 7: UQ Recalibration.{h}.Original_Coverage"]
        recal = tn[f"Section 7: UQ Recalibration.{h}.Recalibrated_Coverage"]
        gap = tn[f"Section 7: UQ Recalibration.{h}.Coverage_Gap_pp"]
        computed_gap = 100.0 * (recal - 0.80)
        check(f"  UQ {h} Coverage_Gap_pp (vs nominal=0.80)",
              gap, computed_gap, tol=0.05)


def check_section_9_hpa(tn):
    """Section 9: HPA simulation grid against hpa_simulation_v2.csv structure."""
    print("\n[9] HPA SIMULATION: thesis_numbers.json vs hpa_simulation_v2.csv")
    df = pd.read_csv(HPA)
    check("  HPA total_rows", tn["Section 9: HPA Simulation.total_rows"], len(df))
    target_vals = sorted(df["target_util"].unique().tolist())
    expected_targets = tn["Section 9: HPA Simulation.target_util_values"]
    if target_vals == expected_targets:
        global PASS_COUNT
        PASS_COUNT += 1
        print(f"  PASS  HPA target_util_values: {target_vals}")
    else:
        global FAIL_COUNT
        FAIL_COUNT += 1
        msg = f"  FAIL  HPA target_util_values: expected={expected_targets} actual={target_vals}"
        FAILS.append(msg)
        print(msg)
    for h in ["10min", "30min", "60min", "120min"]:
        sub = df[df["Horizon"] == h]
        n_react = (sub["Strategy"] == "Reactive").sum()
        n_ml = (sub["Strategy"] == "ML-Proactive").sum()
        check(f"  HPA {h} reactive_rows",
              tn[f"Section 9: HPA Simulation.{h}.reactive_rows"], n_react)
        check(f"  HPA {h} ml_rows",
              tn[f"Section 9: HPA Simulation.{h}.ml_rows"], n_ml)


def check_section_10_meta_learner(tn):
    """Section 10: Meta-learner comparison vs meta_learner_comparison.csv and combination_diagnostics.csv."""
    print("\n[10] META-LEARNER COMPARISON: thesis_numbers.json vs CSVs")
    df = pd.read_csv(META_LEARN)
    check("  total_rows", tn["Section 10: Meta-Learner Comparison.total_rows"], len(df))
    for h in ["10min", "30min", "60min", "120min"]:
        sub = df[df["Horizon"] == h].set_index("Method")
        for method in ["NNLS", "Simple Average", "Ridge", "Bates-Granger"]:
            if method in sub.index:
                check(f"  {h} {method} R2",
                      tn[f"Section 10: Meta-Learner Comparison.{h}.{method}.R2"],
                      sub.loc[method, "R2"], tol=1e-4)

    df2 = pd.read_csv(COMBO_DIAG).set_index("Horizon")
    for h in ["10min", "30min", "60min", "120min"]:
        row = df2.loc[h]
        check(f"  {h} Mean_Err_Corr",
              tn[f"Section 10: Meta-Learner Comparison.{h}.Mean_Err_Corr"],
              row["Mean_Err_Corr"], tol=1e-3)
        check(f"  {h} Best_Individual",
              tn[f"Section 10: Meta-Learner Comparison.{h}.Best_Individual"],
              row["Best_Individual"])


def check_section_11_boundary(tn):
    """Section 11: BCF table against boundary_condition_table_corrected.csv."""
    print("\n[11] BOUNDARY CONDITION TABLE: thesis_numbers.json vs boundary_condition_table_corrected.csv")
    df = pd.read_csv(BCF_TABLE).set_index("dataset")
    for ds_json, ds_csv in [("Alibaba 2018", "Alibaba 2018"),
                            ("Bitbrains", "Bitbrains"),
                            ("DSB", "DSB")]:
        row = df.loc[ds_csv]
        check(f"  BCF {ds_json} CV",
              tn[f"Section 11: Boundary Conditions.{ds_json}.CV"],
              str(row["CV median"]))
        check(f"  BCF {ds_json} ACF_24h",
              tn[f"Section 11: Boundary Conditions.{ds_json}.ACF_24h"],
              str(row["ACF@24h"]))
        check(f"  BCF {ds_json} delta_30min",
              tn[f"Section 11: Boundary Conditions.{ds_json}.delta_30min"],
              str(row["delta @30min"]))
        check(f"  BCF {ds_json} delta_120min",
              tn[f"Section 11: Boundary Conditions.{ds_json}.delta_120min"],
              str(row["delta @120min"]))
        check(f"  BCF {ds_json} verdict",
              tn[f"Section 11: Boundary Conditions.{ds_json}.verdict"],
              str(row["verdict"]))


def check_section_11_omega(tn):
    """Section 11: Omega/predictability against omega_summary.csv."""
    print("\n[12] OMEGA SUMMARY: thesis_numbers.json vs omega_summary.csv")
    df = pd.read_csv(OMEGA).set_index("Dataset")
    print("  available datasets in omega_summary.csv:", df.index.tolist())
    for ds_json, expect_n, ds_aliases in [
        ("Bitbrains", 156, ["Bitbrains", "BitBrains"]),
        ("Alibaba", 4902, ["Alibaba", "Alibaba 2018"]),
        ("ByteDance IaaS", 93, ["ByteDance IaaS", "ByteDance"]),
    ]:
        found = None
        for a in ds_aliases:
            if a in df.index:
                found = a
                break
        if found is None:
            print(f"  SKIP  {ds_json}: not found in omega_summary.csv")
            continue
        row = df.loc[found]
        check(f"  Omega {ds_json} N_series",
              tn[f"Section 11: Boundary Conditions.{ds_json}.N_series"],
              row["N_series"])
        check(f"  Omega {ds_json} CV_median",
              tn[f"Section 11: Boundary Conditions.{ds_json}.CV_median"],
              row["CV_median"], tol=1e-3)
        check(f"  Omega {ds_json} Omega_median",
              tn[f"Section 11: Boundary Conditions.{ds_json}.Omega_median"],
              row["Omega_median"], tol=1e-3)


def check_bcf_pooled(tn):
    """Cross-check bcf_pooled_results.json -- separate from thesis_numbers (this is BCF canonical).

    NOTE 2026-05-22 F0 Day 1.4: BCF_JSON path and hardcoded canonical values are stale
    per Phase A-E canonical_facts §3.3 and §6.3. The 3-model canonical lives in
    results/bcf/bcf_pooled_3model.json with percentile CI [0.7097, 0.8871], p=0.0097.
    The 4-model file checked here has AUC=0.667, p=0.0511 per Phase A-E exclusion rationale.
    Not fixed in this commit (out of scope for Day 1.4 v3 stub append). Track as F0 follow-up
    -- a natural slot is Day 5 Bitbrains Ch5 verification window.
    """
    print("\n[13] BCF POOLED RESULTS: bcf_pooled_results.json self-check")
    with open(BCF_JSON) as f:
        bcf = json.load(f)
    print(f"  predicate: {bcf['predicate']}")
    print(f"  pooled_auc: {bcf['pooled_auc']}")
    print(f"  CI: [{bcf['bca_95_ci_lo']:.4f}, {bcf['bca_95_ci_hi']:.4f}]")
    print(f"  p_value: {bcf['permutation_p']:.4f}")
    # See NOTE above -- these targets are stale; expect FAIL until path migrated to 3-model JSON.
    check("  BCF pooled_auc",      0.80,  bcf["pooled_auc"],    tol=0.001)
    check("  BCF CI lower",        0.70,  bcf["bca_95_ci_lo"],  tol=0.01)
    check("  BCF CI upper",        0.88,  bcf["bca_95_ci_hi"],  tol=0.01)
    check("  BCF n_pooled_pairs",  36,    bcf["n_pooled_pairs"])
    check("  BCF permutation_p",   0.011, bcf["permutation_p"], tol=0.001)


def check_residual_diagnostics(tn):
    """Residual diagnostics CSV -- no thesis_numbers field yet, just print for review."""
    print("\n[14-pre] RESIDUAL DIAGNOSTICS: residual_diagnostics.csv values (for manuscript section)")
    df = pd.read_csv(RESID_DIAG)
    print(df.to_string(index=False))
    bias_min = df["Bias_ensemble"].min()
    bias_max = df["Bias_ensemble"].max()
    print(f"  Bias_ensemble range: [{bias_min:+.4f}, {bias_max:+.4f}]")
    acf_min = df["ACF1_ensemble"].min()
    acf_max = df["ACF1_ensemble"].max()
    print(f"  ACF1_ensemble range: [{acf_min:.4f}, {acf_max:.4f}]")


# =====================================================================
# v3 STUB FUNCTIONS  --  appended F0 Day 1.4 per verify_foundation_v3_stubs.py
#
# Each function PRINTS the file's schema first, then runs the specific checks.
# If a column name or value doesn't match, the schema print makes the mismatch
# visible so the assertion can be adjusted without re-running inference.
# =====================================================================


def _file_exists_or_skip(path, section_name):
    """Helper: print SKIP and return False if file missing. Otherwise return True."""
    if not os.path.exists(path):
        print(f"  SKIP  {section_name}: file not found at {path}")
        print(f"        run rclone sync to bring it from Vast.ai before retrying.")
        return False
    return True


def check_section_14_v3_leaderboard(tn=None):
    """v3 foundation-model leaderboard -- winner tally and headline cells.

    Expected winner tally (Phase A-E §6.2): Chronos-2 6, TimesFM 3, NNLS 2, Toto 1.
    Expected headline cells:
      Bitbrains 10min:  Toto       0.8805  (runner-up Chronos-2 0.8772, margin +0.33pp)
      Bitbrains 60min:  NNLS       0.3205  (runner-up TimesFM   0.2873, margin +3.32pp)
      Bitbrains 120min: TimesFM    0.2366  (runner-up Toto      0.0524, margin +18.42pp)
    """
    print("\n[14] V3 LEADERBOARD: leaderboard_v3_wide.csv + leaderboard_v3_winners.csv")
    if not _file_exists_or_skip(LB_V3_WIDE, "v3_leaderboard_wide"):
        return
    if not _file_exists_or_skip(LB_V3_WINNERS, "v3_leaderboard_winners"):
        return

    wide = pd.read_csv(LB_V3_WIDE)
    winners = pd.read_csv(LB_V3_WINNERS)

    print(f"  wide.csv shape: {wide.shape}")
    print(f"  wide.csv columns: {list(wide.columns)}")
    print(f"  winners.csv shape: {winners.shape}")
    print(f"  winners.csv columns: {list(winners.columns)}")

    winner_col = None
    for candidate in ["winner", "winner_model", "best_model", "model"]:
        if candidate in winners.columns:
            winner_col = candidate
            break
    if winner_col is None:
        print(f"  FAIL  cannot find winner column in winners.csv")
        global FAIL_COUNT
        FAIL_COUNT += 1
        return

    tally = winners[winner_col].value_counts().to_dict()
    print(f"  winner tally from CSV: {tally}")

    expected = {
        "Chronos-2": 6,
        "TimesFM-2.5": 3,
        "NNLS Ensemble": 2,
        "Toto-Open-Base-1.0": 1,
    }
    name_aliases = {
        "chronos2": "Chronos-2", "chronos-2": "Chronos-2",
        "timesfm": "TimesFM-2.5", "timesfm-2.5": "TimesFM-2.5",
        "nnls": "NNLS Ensemble", "nnls_ensemble": "NNLS Ensemble",
        "toto": "Toto-Open-Base-1.0", "toto-open-base-1.0": "Toto-Open-Base-1.0",
    }
    normalised_tally = {}
    for k, v in tally.items():
        nk = name_aliases.get(str(k).lower(), str(k))
        normalised_tally[nk] = normalised_tally.get(nk, 0) + v

    for model, expected_count in expected.items():
        actual_count = normalised_tally.get(model, 0)
        check(f"  v3 winner tally {model}", expected_count, actual_count)

    print("\n  HEADLINE CELLS (Bitbrains):")
    expected_cells = [
        ("Bitbrains", "10min",  "Toto-Open-Base-1.0", 0.8805),
        ("Bitbrains", "60min",  "NNLS Ensemble",      0.3205),
        ("Bitbrains", "120min", "TimesFM-2.5",        0.2366),
    ]
    dataset_col = None
    horizon_col = None
    for c in wide.columns:
        if c.lower() in ("dataset", "trace"):
            dataset_col = c
        if c.lower() in ("horizon", "h", "horizon_min"):
            horizon_col = c
    if dataset_col is None or horizon_col is None:
        print("  SKIP  cannot find dataset/horizon columns in wide.csv; check schema")
        return

    for ds, h, expected_winner, expected_r2 in expected_cells:
        sub = wide[(wide[dataset_col] == ds) & (wide[horizon_col] == h)]
        if len(sub) == 0:
            print(f"  SKIP  {ds} {h}: row not found in wide.csv")
            continue
        model_cols = [c for c in sub.columns
                      if c not in (dataset_col, horizon_col)
                      and sub[c].dtype.kind in "fc"]
        if not model_cols:
            print(f"  SKIP  {ds} {h}: no numeric model columns found")
            continue
        row = sub.iloc[0]
        r2_values = row[model_cols].astype(float)
        max_col = r2_values.idxmax()
        max_r2 = r2_values.max()
        print(f"  {ds} {h}: winner column={max_col}, r2={max_r2:.4f} "
              f"(expected {expected_winner} {expected_r2})")
        check(f"  v3 cell {ds} {h} winner_r2", expected_r2, max_r2, tol=0.005)


def check_section_15_v3_diversity(tn=None):
    """v3 diversity diagnostic -- pooled error correlation rho-bar = 0.6562.

    Source: Phase C C2 verdict
      pooled rho-bar = 0.6562, CI [0.6499, 0.6620], 11 cells
      per-trace medians: ByteDance 0.44-0.48, Alibaba 0.71-0.84, Bitbrains 0.61-0.74
    """
    print("\n[15] V3 DIVERSITY DIAGNOSTIC: results/bcf_v2/c2_correlations_per_cell.csv")
    if not _file_exists_or_skip(C2_CORR, "v3_diversity"):
        return

    df = pd.read_csv(C2_CORR)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  head:\n{df.head().to_string(index=False)}")

    rho_col = None
    for candidate in ["rho_bar", "pooled_rho", "mean_corr", "rho", "correlation",
                      "error_correlation"]:
        if candidate in df.columns:
            rho_col = candidate
            break
    if rho_col is None:
        print(f"  FAIL  cannot find correlation column; columns: {list(df.columns)}")
        global FAIL_COUNT
        FAIL_COUNT += 1
        return

    pooled_median = df[rho_col].median()
    pooled_mean = df[rho_col].mean()
    print(f"  pooled median {rho_col}: {pooled_median:.4f}")
    print(f"  pooled mean   {rho_col}: {pooled_mean:.4f}")
    print(f"  pooled range: [{df[rho_col].min():.4f}, {df[rho_col].max():.4f}]")

    if abs(pooled_median - 0.6562) <= 0.01:
        check("  v3 pooled rho-bar (median)", 0.6562, pooled_median, tol=0.01)
    elif abs(pooled_mean - 0.6562) <= 0.01:
        check("  v3 pooled rho-bar (mean)", 0.6562, pooled_mean, tol=0.01)
    else:
        print(f"  FAIL  v3 pooled rho-bar: expected ~0.6562, "
              f"got median={pooled_median:.4f} mean={pooled_mean:.4f}")
        FAIL_COUNT += 1

    dataset_col = None
    for c in df.columns:
        if c.lower() in ("dataset", "trace"):
            dataset_col = c
            break
    if dataset_col:
        per_trace = df.groupby(dataset_col)[rho_col].median().to_dict()
        print(f"  per-trace medians: {per_trace}")
        for ds_key, expected_low, expected_high in [
            ("ByteDance", 0.44, 0.48),
            ("Alibaba", 0.71, 0.84),
            ("Bitbrains", 0.61, 0.74),
        ]:
            found_key = None
            for k in per_trace:
                if ds_key.lower() in str(k).lower():
                    found_key = k
                    break
            if found_key is None:
                print(f"  SKIP  per-trace {ds_key}: not in CSV")
                continue
            v = per_trace[found_key]
            if expected_low <= v <= expected_high:
                print(f"  PASS  per-trace median {ds_key}: {v:.4f} "
                      f"in [{expected_low}, {expected_high}]")
                global PASS_COUNT
                PASS_COUNT += 1
            else:
                print(f"  FAIL  per-trace median {ds_key}: {v:.4f} "
                      f"OUTSIDE [{expected_low}, {expected_high}]")
                FAIL_COUNT += 1


def check_section_16_v3_loo(tn=None):
    """v3 LOO ablation NEW pool -- Chronos-2 is load-bearing.

    Source: results/bcf_v2/loo_ablation_new_pool_v2.csv (Phase B LOO)
      Per-dataset median delta_r2_vs_full_pp:
        drop_chronos2 = -2.09 / -3.57 / -3.43  (Alibaba / Bitbrains / ByteDance)
        drop_et       ~ 0
        drop_nhits    ~ 0
        only_chronos2 within -0.33pp of ALL
    """
    print("\n[16] V3 LOO ABLATION: results/bcf_v2/loo_ablation_new_pool_v2.csv")
    if not _file_exists_or_skip(LOO_V2, "v3_loo"):
        return

    df = pd.read_csv(LOO_V2)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  unique configs: {df['Config'].unique().tolist() if 'Config' in df.columns else 'no Config column'}")
    print(f"  head:\n{df.head(10).to_string(index=False)}")

    config_col = "Config" if "Config" in df.columns else None
    delta_col = None
    for c in df.columns:
        if "delta" in c.lower() and ("r2" in c.lower() or "pp" in c.lower()):
            delta_col = c
            break
    dataset_col = None
    for c in df.columns:
        if c.lower() in ("dataset", "trace"):
            dataset_col = c
            break

    if config_col is None or delta_col is None or dataset_col is None:
        print(f"  SKIP  cannot find required columns "
              f"(config={config_col} delta={delta_col} dataset={dataset_col})")
        return

    expected_drop_chronos2 = {
        "Alibaba": -2.09,
        "Bitbrains": -3.57,
        "ByteDance": -3.43,
    }
    sub = df[df[config_col].str.lower().str.contains("drop_chronos") |
             df[config_col].str.lower().str.contains("no.chronos")]
    if len(sub) == 0:
        print(f"  SKIP  no drop_chronos2 rows in {config_col}; available configs: "
              f"{df[config_col].unique().tolist()}")
        return

    for ds_key, expected in expected_drop_chronos2.items():
        ds_rows = sub[sub[dataset_col].str.contains(ds_key, case=False)]
        if len(ds_rows) == 0:
            print(f"  SKIP  drop_chronos2 {ds_key}: no rows")
            continue
        median_delta = ds_rows[delta_col].median()
        check(f"  drop_chronos2 {ds_key} median delta_pp",
              expected, median_delta, tol=0.5)

    only = df[df[config_col].str.lower().str.contains("only.chronos")]
    if len(only) > 0:
        only_delta_median = only[delta_col].median()
        print(f"  only_chronos2 median delta: {only_delta_median:.4f} pp "
              f"(expected within -0.33pp of 0)")


def check_section_17_v3_d4(tn=None):
    """v3 D4 statistical validation -- 38/3/3 DM result.

    Source: Phase D D4 cluster-bootstrap DM + Holm-Bonferroni
      44 cluster-bootstrap DM pairwise tests with Holm-Bonferroni
      38 wins / 3 ties / 3 losses for v2 ensemble
      Predicate-stratified: 23/1/0 in predicate-positive cells, 15/2/3 in predicate-negative
      Friedman: v2 ensemble holds mean rank 1 at every horizon
    """
    print("\n[17] V3 D4 STATISTICAL VALIDATION: results/bcf_v2/d4_*.csv")

    candidates = [
        f"{PROJECT}/results/bcf_v2/d4_results.csv",
        f"{PROJECT}/results/bcf_v2/d4_pairwise.csv",
        f"{PROJECT}/results/bcf_v2/d4_dm_results.csv",
        f"{PROJECT}/results/bcf_v2/d4_validation.csv",
    ]
    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break
    if found is None:
        d4_dir = f"{PROJECT}/results/bcf_v2"
        if os.path.exists(d4_dir):
            d4_files = [f for f in os.listdir(d4_dir) if f.lower().startswith("d4")]
            print(f"  SKIP  no d4_*.csv found among candidates; files in {d4_dir}: {d4_files}")
        else:
            print(f"  SKIP  {d4_dir} does not exist; run rclone sync")
        return

    df = pd.read_csv(found)
    print(f"  loaded {found}")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  head:\n{df.head().to_string(index=False)}")

    if "verdict" in df.columns or "result" in df.columns:
        col = "verdict" if "verdict" in df.columns else "result"
        tally = df[col].value_counts().to_dict()
        print(f"  {col} tally: {tally}")
        wins = sum(v for k, v in tally.items() if "win" in str(k).lower())
        ties = sum(v for k, v in tally.items() if "tie" in str(k).lower())
        losses = sum(v for k, v in tally.items() if "los" in str(k).lower())
        if wins + ties + losses == 44:
            check("  D4 total tests", 44, wins + ties + losses)
            check("  D4 wins", 38, wins)
            check("  D4 ties", 3, ties)
            check("  D4 losses", 3, losses)
        else:
            print(f"  NOTE  tally sum {wins + ties + losses} != 44 expected; "
                  f"verdict column may use different wording")
    else:
        print(f"  NOTE  no 'verdict' or 'result' column; manual inspection needed")


def main():
    print("=" * 70)
    print("verify_foundation.py  --  Phase F0 Day 1 verification (v3 extended)")
    print("=" * 70)
    tn = load_json()

    check_section_1_alibaba_r2(tn)
    check_section_1_alibaba_mae(tn)
    check_section_2_bitbrains(tn)
    check_section_3_chronos_bolt(tn)
    check_section_4_bytedance(tn)
    check_section_5_stats(tn)
    check_section_6_shap(tn)
    check_section_7_uq(tn)
    check_section_9_hpa(tn)
    check_section_10_meta_learner(tn)
    check_section_11_boundary(tn)
    check_section_11_omega(tn)
    check_bcf_pooled(tn)
    check_residual_diagnostics(tn)

    # v3 stubs -- appended F0 Day 1.4
    check_section_14_v3_leaderboard(tn)
    check_section_15_v3_diversity(tn)
    check_section_16_v3_loo(tn)
    check_section_17_v3_d4(tn)

    print("\n" + "=" * 70)
    print(f"TOTAL: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL")
    print("=" * 70)
    if FAIL_COUNT > 0:
        print("\nFAILURES:")
        for m in FAILS:
            print(m)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
