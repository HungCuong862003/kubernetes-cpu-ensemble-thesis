"""
task4_verify_tables.py — Regenerate every thesis number from source CSVs.

Single source of truth for defense. Reads all result CSVs (originals +
corrected) and prints a section-by-section verification report. Every
number that appears in the thesis should be traceable to this output.

Inputs:
    comparison_table.csv, stratified_skill.csv,
    bitbrains_summary_corrected.csv, bitbrains_cv_stratified.csv,
    chronos_benchmark_results.csv, results_summary__Bytedance.csv,
    friedman_table.csv, dm_pvalues_{hz}.csv,
    shap_importance.csv, uq_recalibration_table.csv,
    residual_diagnostics.csv, base_model_correlations.csv,
    loo_ablation.csv, omega_summary.csv,
    boundary_condition_table_corrected.csv,
    hpa_simulation_v2.csv, meta_learner_comparison.csv,
    combination_diagnostics.csv, bootstrap_weight_stats.csv
    (optional) retrain_results.csv

Outputs:
    thesis_numbers.json                        (OUTPUT_DIR)
    thesis_numbers.txt                         (OUTPUT_DIR)

Run on Colab:
    # adjust paths in CONFIG section, then:
    !python task4_verify_tables.py
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


# ── CONFIG (adjust paths for your environment) ──────────────────────

# directory with original result CSVs
DATA_DIR = "/content/drive/MyDrive/k8s-ensemble-forecast/results/thesis_figures"

# subdirectories where some CSVs live (thesis_analysis.py saves them here)
DATA_SUBDIRS = [
    os.path.join(DATA_DIR, "dm_heatmaps"),
    os.path.join(DATA_DIR, "cd_diagrams"),
    os.path.join(DATA_DIR, "residual_diag"),
]

# directory with corrected CSVs from script 1 + outputs from scripts 2/3
CORRECTED_DIR = "/content/drive/MyDrive/k8s-ensemble-forecast/corrected"

OUTPUT_DIR = "."

HORIZONS = ["10min", "30min", "60min", "120min"]


# ── HELPERS ─────────────────────────────────────────────────────────

def load(name, directory=None):
    """Try loading a CSV from CORRECTED_DIR first, then DATA_DIR and subdirs."""
    if directory:
        p = os.path.join(directory, name)
        if os.path.exists(p):
            return pd.read_csv(p)
        return None

    # try corrected first, then data dir, then subdirs
    for d in [CORRECTED_DIR, DATA_DIR] + DATA_SUBDIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


class Report:
    """Collects verification output for both console and file."""

    def __init__(self):
        self.lines = []
        self.numbers = {}   # keyed by section
        self._section = ""

    def section(self, title):
        self._section = title
        sep = "=" * 70
        self.lines.append("")
        self.lines.append(sep)
        self.lines.append(title)
        self.lines.append(sep)
        print(f"\n{sep}\n{title}\n{sep}")

    def put(self, key, value, text=""):
        """Record a number and print it."""
        # convert numpy types to native Python for clean JSON serialization
        if hasattr(value, 'item'):
            value = value.item()
        full_key = f"{self._section}.{key}" if self._section else key
        self.numbers[full_key] = value
        line = f"  {key}: {value}"
        if text:
            line += f"  ({text})"
        self.lines.append(line)
        print(line)

    def info(self, text):
        self.lines.append(f"  {text}")
        print(f"  {text}")

    def warn(self, text):
        self.lines.append(f"  WARNING: {text}")
        print(f"  WARNING: {text}")

    def save(self, json_path, txt_path):
        with open(json_path, "w") as f:
            json.dump(self.numbers, f, indent=2, default=str)
        with open(txt_path, "w") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\nSaved: {json_path}")
        print(f"Saved: {txt_path}")


# ── MAIN ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = Report()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.lines.append("THESIS NUMBER VERIFICATION REPORT")
    report.lines.append(f"Generated: {now}")
    report.lines.append(f"Data dir:     {DATA_DIR}")
    report.lines.append(f"Corrected dir: {CORRECTED_DIR}")
    print(f"THESIS NUMBER VERIFICATION REPORT")
    print(f"Generated: {now}")

    # ────────────────────────────────────────────────────────────────
    # Section 1: Alibaba Results
    # ────────────────────────────────────────────────────────────────
    report.section("Section 1: Alibaba Results")

    comp = load("comparison_table.csv")
    if comp is not None:
        report.info("--- R² comparison (from comparison_table.csv) ---")
        for _, row in comp.iterrows():
            hz = row["Horizon"]
            naive_r2 = float(row["Naive R²"])
            hetero_r2 = float(row["Hetero Ens R²"])
            delta_pp = (hetero_r2 - naive_r2) * 100
            resid_var_red = 1 - (1 - hetero_r2) / (1 - naive_r2)

            report.put(f"{hz}.Naive_R2", round(naive_r2, 4))
            report.put(f"{hz}.Hetero_R2", round(hetero_r2, 4))
            report.put(f"{hz}.Delta_pp", round(delta_pp, 2))
            report.put(f"{hz}.Resid_Var_Reduction", f"{resid_var_red*100:.1f}%")

        # also record ET and BiLSTM R² for reference
        report.info("")
        report.info("--- Individual model R² ---")
        for _, row in comp.iterrows():
            hz = row["Horizon"]
            report.put(f"{hz}.ET_R2", round(float(row["ET R²"]), 4))
            report.put(f"{hz}.BiLSTM_R2", round(float(row["BiLSTM R²"]), 4))
            report.put(f"{hz}.Homo_R2", round(float(row["Homo Ens R²"]), 4))
    else:
        report.warn("comparison_table.csv not found")

    strat = load("stratified_skill.csv")
    if strat is not None:
        report.info("")
        report.info("--- MAE skill scores (from stratified_skill.csv) ---")
        for _, row in strat.iterrows():
            hz = row["Horizon"]
            report.put(f"{hz}.MAE_naive", round(float(row["MAE_naive"]), 4))
            report.put(f"{hz}.MAE_ensemble", round(float(row["MAE_ensemble"]), 4))
            report.put(f"{hz}.Skill_score", round(float(row["Skill_score"]), 4))
            report.put(f"{hz}.Win_rate_pct", round(float(row["Timestep_Win_rate_%"]), 2))
    else:
        report.warn("stratified_skill.csv not found")

    cv_strat = load("cv_stratified_skill.csv")
    if cv_strat is not None:
        report.info("")
        report.info("--- CV-stratified skill (Alibaba, from cv_stratified_skill.csv) ---")
        for _, row in cv_strat.iterrows():
            hz = row["Horizon"]
            cv_bin = row["CV_bin"]
            report.put(f"{hz}.{cv_bin}.Skill", round(float(row["Skill_score"]), 4))
            report.put(f"{hz}.{cv_bin}.Win_pct", round(float(row["Container_win_%"]), 1))
            report.put(f"{hz}.{cv_bin}.N", int(row["N_containers"]))
    else:
        report.warn("cv_stratified_skill.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 2: Bitbrains Cross-Dataset
    # ────────────────────────────────────────────────────────────────
    report.section("Section 2: Bitbrains Cross-Dataset")

    bb = load("bitbrains_summary_corrected.csv")
    if bb is not None:
        report.info("--- Corrected summary (median aggregation) ---")
        for _, row in bb.iterrows():
            hz = row["Horizon"]
            report.put(f"{hz}.BB_Naive_R2", row["BB_Naive_R2"])
            report.put(f"{hz}.BB_ML_R2", row["BB_ML_R2_median"])
            report.put(f"{hz}.BB_Delta_pp", row["BB_Delta_pp"])
            report.put(f"{hz}.BB_Win_pct", row["BB_Pct_VMs_ML_Wins"])
    else:
        report.warn("bitbrains_summary_corrected.csv not found")

    bb_cv = load("bitbrains_cv_stratified.csv")
    if bb_cv is not None:
        report.info("")
        report.info("--- CV-stratified breakdown ---")
        for _, row in bb_cv.iterrows():
            hz = row["Horizon"]
            cv_bin = row["CV_bin"]
            report.put(f"{hz}.{cv_bin}.Win_pct", row["Win_pct"])
            report.put(f"{hz}.{cv_bin}.N", int(row["N"]))
    else:
        report.warn("bitbrains_cv_stratified.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 3: Chronos-Bolt
    # ────────────────────────────────────────────────────────────────
    report.section("Section 3: Chronos-Bolt")

    chronos = load("chronos_benchmark_results.csv")
    if chronos is not None:
        ens_wins = 0
        chr_wins = 0
        ens_details = []
        for _, row in chronos.iterrows():
            ds = row["dataset"]
            hz = row["horizon"]
            e = float(row["ensemble_r2"])
            c = float(row["chronos_r2"])
            delta = (e - c) * 100
            if e > c:
                ens_wins += 1
                ens_details.append(f"{ds} {hz}: +{delta:.2f}pp")
            else:
                chr_wins += 1
            report.put(f"{ds}.{hz}.ens_r2", round(e, 4))
            report.put(f"{ds}.{hz}.chr_r2", round(c, 4))
            report.put(f"{ds}.{hz}.delta_pp", round(delta, 2))

        report.info("")
        report.put("Ensemble_wins", f"{ens_wins}/12")
        report.put("Chronos_wins", f"{chr_wins}/12")
        if ens_details:
            report.info(f"Ensemble wins at: {'; '.join(ens_details)}")

        # fair comparison (exclude Bitbrains)
        fair_ens = sum(1 for _, r in chronos.iterrows()
                       if r["dataset"] != "Bitbrains"
                       and float(r["ensemble_r2"]) > float(r["chronos_r2"]))
        fair_total = sum(1 for _, r in chronos.iterrows()
                         if r["dataset"] != "Bitbrains")
        report.put("Fair_Ensemble_wins", f"{fair_ens}/{fair_total}",
                    "excluding Bitbrains global ensemble")
    else:
        report.warn("chronos_benchmark_results.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 4: ByteDance
    # ────────────────────────────────────────────────────────────────
    report.section("Section 4: ByteDance")

    bd = load("results_summary__Bytedance.csv")
    if bd is not None:
        report.info("--- Hetero ensemble vs naive and vs ExtraTrees ---")
        for _, row in bd.iterrows():
            hz = row["Horizon"]
            naive = float(row["naive_R2"])
            hetero = float(row["hetero_ensemble_R2"])
            et = float(row["extratrees_R2"])
            report.put(f"{hz}.naive_R2", round(naive, 4))
            report.put(f"{hz}.hetero_R2", round(hetero, 4))
            report.put(f"{hz}.ET_R2", round(et, 4))
            report.put(f"{hz}.vs_naive_pp", round((hetero - naive) * 100, 2))
            report.put(f"{hz}.vs_ET_pp", round((hetero - et) * 100, 2))
    else:
        report.warn("results_summary__Bytedance.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 5: Statistical Tests
    # ────────────────────────────────────────────────────────────────
    report.section("Section 5: Statistical Tests")

    friedman = load("friedman_table.csv")
    if friedman is not None:
        report.info("--- Friedman test ---")
        for _, row in friedman.iterrows():
            hz = row["Horizon"]
            report.put(f"{hz}.chi2", round(float(row["chi2"]), 1))
            report.put(f"{hz}.p_value", float(row["p_value"]))
            report.put(f"{hz}.n_containers", int(row["n_containers"]))
    else:
        report.warn("friedman_table.csv not found")

    # DM tests: count significant pairs for hetero_ensemble
    report.info("")
    report.info("--- Diebold-Mariano: hetero_ensemble vs others ---")
    for hz in HORIZONS:
        dm = load(f"dm_pvalues_{hz}.csv")
        if dm is None:
            report.warn(f"dm_pvalues_{hz}.csv not found")
            continue

        # the row index is the first column (unnamed)
        dm = dm.set_index(dm.columns[0])
        if "hetero_ensemble" in dm.columns:
            sig_pairs = []
            for model in dm.index:
                if model == "hetero_ensemble":
                    continue
                p = float(dm.loc[model, "hetero_ensemble"])
                if p < 0.05:
                    sig_pairs.append(model)
            report.put(f"{hz}.DM_sig_pairs", len(sig_pairs),
                        f"vs {', '.join(sig_pairs)}" if sig_pairs else "none")
        else:
            report.warn(f"  {hz}: hetero_ensemble column not found in DM table")

    # ────────────────────────────────────────────────────────────────
    # Section 6: SHAP Feature Importance
    # ────────────────────────────────────────────────────────────────
    report.section("Section 6: SHAP Feature Importance")

    shap = load("shap_importance.csv")
    if shap is not None:
        # first column is feature name (unnamed)
        shap = shap.rename(columns={shap.columns[0]: "feature"})

        for hz in HORIZONS:
            if hz not in shap.columns:
                continue
            total = shap[hz].sum()
            if total < 1e-10:
                continue

            # group features
            groups = {}
            for _, row in shap.iterrows():
                name = row["feature"]
                val = float(row[hz])
                pct = val / total * 100

                if name in ("hour_sin", "hour_cos"):
                    groups["hour_sin+cos"] = groups.get("hour_sin+cos", 0) + pct
                if name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "business_hours"):
                    groups["broad_temporal"] = groups.get("broad_temporal", 0) + pct
                if "cpu_roll" in name:
                    groups["cpu_roll"] = groups.get("cpu_roll", 0) + pct
                elif name.startswith("cpu_lag"):
                    groups["cpu_lag"] = groups.get("cpu_lag", 0) + pct
                elif name.startswith("mem_") or name.startswith("disk_"):
                    groups["cross_resource"] = groups.get("cross_resource", 0) + pct
                elif name == "cluster_id":
                    groups["cluster_id"] = groups.get("cluster_id", 0) + pct

            report.info(f"--- {hz} ---")
            for g in ["hour_sin+cos", "broad_temporal", "cpu_roll",
                       "cpu_lag", "cross_resource", "cluster_id"]:
                if g in groups:
                    report.put(f"{hz}.{g}", f"{groups[g]:.1f}%")
    else:
        report.warn("shap_importance.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 7: Uncertainty Quantification
    # ────────────────────────────────────────────────────────────────
    report.section("Section 7: UQ Recalibration")

    uq = load("uq_recalibration_table.csv")
    if uq is not None:
        for _, row in uq.iterrows():
            hz = row["Horizon"]
            report.put(f"{hz}.Original_Coverage", row["Original_Coverage"])
            report.put(f"{hz}.Recalibrated_Coverage", row["Recalibrated_Coverage"])
            report.put(f"{hz}.Coverage_Gap_pp", row["Coverage_Gap_pp"])
            report.put(f"{hz}.Original_Width", row["Original_Width"])
            report.put(f"{hz}.Recalibrated_Width", row["Recalibrated_Width"])
    else:
        report.warn("uq_recalibration_table.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 8: Residual Diagnostics
    # ────────────────────────────────────────────────────────────────
    report.section("Section 8: Residual Diagnostics")

    resid = load("residual_diagnostics.csv")
    if resid is not None:
        for _, row in resid.iterrows():
            hz = row["Horizon"]
            report.put(f"{hz}.Bias_naive", round(float(row["Bias_naive"]), 4))
            report.put(f"{hz}.Bias_ensemble", round(float(row["Bias_ensemble"]), 4))
            report.put(f"{hz}.ACF1_naive", round(float(row["ACF1_naive"]), 3))
            report.put(f"{hz}.ACF1_ensemble", round(float(row["ACF1_ensemble"]), 3))
            report.put(f"{hz}.LjungBox_p_ens", float(row["LjungBox_p_ens"]))
    else:
        report.warn("residual_diagnostics.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 9: HPA Simulation (corrected)
    # ────────────────────────────────────────────────────────────────
    report.section("Section 9: HPA Simulation")

    hpa = load("hpa_simulation_v2.csv")
    if hpa is not None:
        report.put("total_rows", len(hpa))
        for hz in HORIZONS:
            sub = hpa[hpa["Horizon"] == hz]
            react = sub[sub["Strategy"] == "Reactive"]
            ml = sub[sub["Strategy"] == "ML-Proactive"]
            n_unique_react = len(react.drop_duplicates(
                subset=["violation_rate", "waste_rate"]))
            report.put(f"{hz}.reactive_rows", len(react))
            report.put(f"{hz}.reactive_unique", n_unique_react,
                        "was 4 in buggy version")
            report.put(f"{hz}.ml_rows", len(ml))

        # check that target_util column exists
        if "target_util" in hpa.columns:
            report.put("has_target_util_col", True)
            report.put("target_util_values",
                        sorted(hpa["target_util"].unique().tolist()))
        else:
            report.warn("target_util column missing from hpa_simulation_v2.csv")
    else:
        hpa_old = load("hpa_simulation.csv")
        if hpa_old is not None:
            report.warn("Using OLD hpa_simulation.csv (buggy reactive)")
            report.put("total_rows", len(hpa_old))
        else:
            report.warn("No HPA simulation CSV found")

    # ────────────────────────────────────────────────────────────────
    # Section 10: Meta-Learner Comparison (new)
    # ────────────────────────────────────────────────────────────────
    report.section("Section 10: Meta-Learner Comparison")

    meta = load("meta_learner_comparison.csv")
    if meta is not None:
        report.put("total_rows", len(meta))
        report.info("")
        for hz in HORIZONS:
            sub = meta[meta["Horizon"] == hz]
            if sub.empty:
                continue
            report.info(f"--- {hz} ---")
            for _, row in sub.iterrows():
                method = row["Method"]
                r2 = float(row["R2"])
                mae = float(row["MAE"])
                report.put(f"{hz}.{method}.R2", round(r2, 6))
                report.put(f"{hz}.{method}.MAE", round(mae, 4))
    else:
        report.warn("meta_learner_comparison.csv not found")

    diag = load("combination_diagnostics.csv")
    if diag is not None:
        report.info("")
        report.info("--- Combination puzzle diagnostics ---")
        for _, row in diag.iterrows():
            hz = row["Horizon"]
            report.put(f"{hz}.Cond_Number", row["Cond_Number"])
            report.put(f"{hz}.Mean_Err_Corr", row["Mean_Err_Corr"])
            report.put(f"{hz}.R2_Spread_pp", row["R2_Spread_pp"])
            report.put(f"{hz}.Best_Individual", row["Best_Individual"])
    else:
        report.warn("combination_diagnostics.csv not found")

    boot = load("bootstrap_weight_stats.csv")
    if boot is not None:
        report.info("")
        report.info("--- Bootstrap weight stability ---")
        for _, row in boot.iterrows():
            hz = row["Horizon"]
            model = row["Model"]
            report.put(f"{hz}.{model}.Mean_W", row["Mean_W"])
            report.put(f"{hz}.{model}.Std_W", row["Std_W"])
            report.put(f"{hz}.{model}.Frac_Zero", row["Frac_Zero"])
    else:
        report.warn("bootstrap_weight_stats.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 10b: Retrain Results (optional)
    # ────────────────────────────────────────────────────────────────
    retrain = load("retrain_results.csv")
    if retrain is not None:
        report.section("Section 10b: Retrain ET Enhanced (optional)")
        for _, row in retrain.iterrows():
            hz = row["Horizon"]
            var = row["Variant"]
            report.put(f"{hz}.{var}.R2", row["R2"])
            report.put(f"{hz}.{var}.MAE", row["MAE"])
            report.put(f"{hz}.{var}.Delta_R2_pp", row["Delta_R2_pp"])

    # ────────────────────────────────────────────────────────────────
    # Section 11: Boundary Conditions
    # ────────────────────────────────────────────────────────────────
    report.section("Section 11: Boundary Conditions")

    boundary = load("boundary_condition_table_corrected.csv")
    if boundary is not None:
        report.info("--- Corrected boundary condition table ---")
        for _, row in boundary.iterrows():
            ds = row["dataset"]
            report.put(f"{ds}.CV", row["CV median"])
            report.put(f"{ds}.Hurst", row["Hurst"])
            report.put(f"{ds}.ACF_24h", row["ACF@24h"])
            report.put(f"{ds}.delta_30min", row["delta @30min"])
            report.put(f"{ds}.delta_120min", row["delta @120min"])
            report.put(f"{ds}.verdict", row["verdict"])
    else:
        report.warn("boundary_condition_table_corrected.csv not found")

    omega = load("omega_summary.csv")
    if omega is not None:
        report.info("")
        report.info("--- Omega summary (canonical descriptive stats) ---")
        for _, row in omega.iterrows():
            ds = row["Dataset"]
            report.put(f"{ds}.N_series", int(row["N_series"]))
            report.put(f"{ds}.CV_median", row["CV_median"])
            report.put(f"{ds}.Hurst_median", row["Hurst_median"])
            report.put(f"{ds}.ACF_1h", row["ACF@1h"])
            report.put(f"{ds}.ACF_24h", row["ACF@24h"])
            report.put(f"{ds}.Omega_median", row["Omega_median"])
    else:
        report.warn("omega_summary.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 12: Base Model Correlations
    # ────────────────────────────────────────────────────────────────
    report.section("Section 12: Base Model Correlations")

    corr = load("base_model_correlations.csv")
    if corr is not None:
        for hz in HORIZONS:
            sub = corr[corr["Horizon"] == hz]
            if sub.empty:
                continue
            err_corrs = sub["Error_Corr"].values
            report.put(f"{hz}.min_err_corr", round(float(err_corrs.min()), 4))
            report.put(f"{hz}.max_err_corr", round(float(err_corrs.max()), 4))
            report.put(f"{hz}.mean_err_corr", round(float(err_corrs.mean()), 4))
    else:
        report.warn("base_model_correlations.csv not found")

    # ────────────────────────────────────────────────────────────────
    # Section 13: LOO Ablation
    # ────────────────────────────────────────────────────────────────
    report.section("Section 13: LOO Ablation")

    loo = load("loo_ablation.csv")
    if loo is not None:
        report.put("total_rows", len(loo))
        for hz in HORIZONS:
            sub = loo[loo["Horizon"] == hz]
            full = sub[sub["Config"] == "Full ensemble"]
            if len(full) > 0:
                report.put(f"{hz}.Full_R2", float(full.iloc[0]["R2"]))
                report.put(f"{hz}.Full_MAE", float(full.iloc[0]["MAE"]))

            # ET alone vs full
            et_alone = sub[sub["Config"] == "ExtraTrees alone"]
            if len(et_alone) > 0:
                delta = float(et_alone.iloc[0]["Delta_R2_pp"])
                report.put(f"{hz}.ET_alone_delta_pp", round(delta, 4))
    else:
        report.warn("loo_ablation.csv not found")

    # ────────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────────
    report.section("SUMMARY")
    report.info(f"Total numbers recorded: {len(report.numbers)}")
    report.info(f"Sections covered: 13")
    report.info("")
    report.info("KEY NUMBERS FOR QUICK REFERENCE:")
    report.info("")

    # pull key numbers if they exist
    key_checks = [
        ("Chronos wins",           "Section 3: Chronos-Bolt.Chronos_wins",
         "10/12"),
        ("Win rate 10min",         "Section 1: Alibaba Results.10min.Win_rate_pct",
         "17.26"),
        ("Win rate 30min",         "Section 1: Alibaba Results.30min.Win_rate_pct",
         "31.53"),
        ("Win rate 60min",         "Section 1: Alibaba Results.60min.Win_rate_pct",
         "38.64"),
        ("Win rate 120min",        "Section 1: Alibaba Results.120min.Win_rate_pct",
         "43.45"),
        ("Alibaba 120min delta",   "Section 1: Alibaba Results.120min.Delta_pp",
         "4.64"),
        ("BB 120min ML wins",      "Section 2: Bitbrains Cross-Dataset.120min.BB_Win_pct",
         "51.9"),
    ]

    for label, key, expected in key_checks:
        actual = report.numbers.get(key, "NOT FOUND")
        match = str(actual) == expected
        status = "OK" if match else "MISMATCH"
        report.info(f"  [{status}] {label}: {actual} (expected {expected})")

    # ── save outputs ────────────────────────────────────────────────

    json_path = os.path.join(OUTPUT_DIR, "thesis_numbers.json")
    txt_path = os.path.join(OUTPUT_DIR, "thesis_numbers.txt")
    report.save(json_path, txt_path)

    print(f"\ndone.")
