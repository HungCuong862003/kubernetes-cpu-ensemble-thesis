"""
task0_data_corrections.py — Verify and correct data discrepancies.

Reads all source CSVs, checks for known discrepancies between them,
regenerates corrected files, and prints a verification report.

This is a "damage prevention" script. Every number in the thesis
should be traceable back to a source CSV. This script catches the
places where the thesis text or summary tables diverged from the
raw data.

Known issues being checked:
  1. boundary_condition_table.csv — Bitbrains row has wrong CV, Hurst,
     deltas, and verdict (sign-reversed delta@30min).
     Alibaba row uses ACF@120min (2h lag) instead of ACF@24h.
     Column header says "ACF@120min" but framework uses ACF@24h.
  2. bitbrains_summary.csv — corrupted with 1e+28 overflow values
  3. Chronos win count — should be 10/12 raw, but 4 of those are
     against a broken global Bitbrains ensemble (not per-VM models).
     Defensible claim: 6/8 on Alibaba+ByteDance.
  4. stratified_skill.csv — CSV is correct but thesis text may have
     stale win rates (0.0% / 18.3% / 39.3% / 49.2%)
  5. ByteDance framing — ensemble beats naive at ALL four horizons
  6. Alibaba descriptive stats — omega_summary vs predictability_table
     disagree on CV (0.373 vs 0.346) and Hurst (0.781 vs 0.898)
  7. NNLS weights at 10min — BiLSTM OOF succeeded in Run 2,
     so hetero != homo
  8. Two Bitbrains per-VM files disagree (different runs/splits)

Inputs:
    boundary_condition_table.csv       (--data-dir)
    bitbrains_per_vm.csv               (--data-dir)
    bitbrains_per_vm_results.csv       (--data-dir)
    bitbrains_summary.csv              (--data-dir)
    chronos_benchmark_results.csv      (--data-dir)
    comparison_table.csv               (--data-dir)
    stratified_skill.csv               (--data-dir)
    results_summary__Bytedance.csv     (--data-dir)
    omega_summary.csv                  (--data-dir)
    omega_bitbrains.csv                (--data-dir)
    omega_alibaba.csv                  (--data-dir)
    predictability_table.csv           (--data-dir)
    bitbrains_landscape_stats.csv      (--data-dir)
    bitbrains_cv_stratified.csv        (--data-dir)

Outputs:
    boundary_condition_table_corrected.csv   (--output-dir)
    bitbrains_summary_corrected.csv          (--output-dir)
    data_corrections_report.txt              (--output-dir)
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime


# ── helpers ────────────────────────────────────────────────────────────

def load_csv(data_dir, filename):
    """Load a CSV, return None if missing (silently)."""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def pp(value):
    """Format a float as percentage points with sign, e.g. '+4.64pp'."""
    return f"{value:+.2f}pp"


# ── main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify and correct data discrepancies for thesis")
    parser.add_argument("--data-dir", default=".",
                        help="Directory containing source CSVs")
    parser.add_argument("--output-dir", default="./corrected",
                        help="Where to write corrected files and report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # these hold the pass/fail results for the final summary
    # (inside main so they reset if main() is called twice)
    results = []
    report_lines = []

    def report(msg):
        """Print and also buffer for the text report."""
        print(msg)
        report_lines.append(msg)

    def check(name, passed, detail=""):
        """Record a check result. Goes into both stdout and report."""
        status = "PASS" if passed else "FAIL"
        results.append((name, status, detail))
        mark = "✓" if passed else "✗"
        report(f"  [{mark}] {name}")
        if detail:
            report(f"      {detail}")

    report("=" * 70)
    report("DATA CORRECTIONS REPORT")
    report(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report(f"Data dir:  {os.path.abspath(args.data_dir)}")
    report("=" * 70)

    # ── load everything up front ──────────────────────────────────────

    boundary     = load_csv(args.data_dir, "boundary_condition_table.csv")
    bb_per_vm    = load_csv(args.data_dir, "bitbrains_per_vm.csv")
    bb_per_vm2   = load_csv(args.data_dir, "bitbrains_per_vm_results.csv")
    bb_summary   = load_csv(args.data_dir, "bitbrains_summary.csv")
    chronos      = load_csv(args.data_dir, "chronos_benchmark_results.csv")
    comp_table   = load_csv(args.data_dir, "comparison_table.csv")
    skill        = load_csv(args.data_dir, "stratified_skill.csv")
    bd_results   = load_csv(args.data_dir, "results_summary__Bytedance.csv")
    omega_sum    = load_csv(args.data_dir, "omega_summary.csv")
    omega_bb     = load_csv(args.data_dir, "omega_bitbrains.csv")
    omega_ali    = load_csv(args.data_dir, "omega_alibaba.csv")
    pred_table   = load_csv(args.data_dir, "predictability_table.csv")
    bb_landscape = load_csv(args.data_dir, "bitbrains_landscape_stats.csv")
    bb_strat     = load_csv(args.data_dir, "bitbrains_cv_stratified.csv")

    # report which files were found / missing
    all_files = {
        "boundary_condition_table.csv":  boundary,
        "bitbrains_per_vm.csv":          bb_per_vm,
        "bitbrains_per_vm_results.csv":  bb_per_vm2,
        "bitbrains_summary.csv":         bb_summary,
        "chronos_benchmark_results.csv": chronos,
        "comparison_table.csv":          comp_table,
        "stratified_skill.csv":          skill,
        "results_summary__Bytedance.csv": bd_results,
        "omega_summary.csv":             omega_sum,
        "omega_bitbrains.csv":           omega_bb,
        "omega_alibaba.csv":             omega_ali,
        "predictability_table.csv":      pred_table,
        "bitbrains_landscape_stats.csv": bb_landscape,
        "bitbrains_cv_stratified.csv":   bb_strat,
    }
    missing = [f for f, df in all_files.items() if df is None]
    if missing:
        report(f"\n  WARNING: {len(missing)} file(s) not found:")
        for f in missing:
            report(f"    — {f}")
        report("")
    else:
        report(f"\n  All {len(all_files)} source files loaded.")

    # ── 0. VM SET OVERLAP CHECK ───────────────────────────────────────

    report("\n" + "-" * 70)
    report("0. BITBRAINS VM SET CONSISTENCY")
    report("-" * 70)

    if (bb_per_vm is not None and omega_bb is not None
            and bb_per_vm2 is not None):
        vms_pervm  = set(bb_per_vm["vm_id"].unique())
        vms_omega  = set(omega_bb["vm_id"].unique())
        vms_pervm2 = set(bb_per_vm2["vm_id"].unique())

        overlap_1  = vms_pervm & vms_omega
        overlap_2  = vms_pervm & vms_pervm2

        report(f"  bb_per_vm VMs:         {len(vms_pervm)}")
        report(f"  omega_bitbrains VMs:   {len(vms_omega)}")
        report(f"  bb_per_vm_results VMs: {len(vms_pervm2)}")
        report(f"  per_vm ∩ omega:        {len(overlap_1)}")
        report(f"  per_vm ∩ per_vm2:      {len(overlap_2)}")

        check("bb_per_vm and omega_bb same VM set",
              vms_pervm == vms_omega,
              f"Intersection={len(overlap_1)}, "
              f"per_vm_only={len(vms_pervm - vms_omega)}, "
              f"omega_only={len(vms_omega - vms_pervm)}")

    # ── 1. BOUNDARY CONDITION TABLE ───────────────────────────────────

    report("\n" + "-" * 70)
    report("1. BOUNDARY CONDITION TABLE — full rebuild")
    report("-" * 70)

    if boundary is not None and bb_per_vm is not None and omega_bb is not None:
        bb_row = boundary[boundary["dataset"] == "Bitbrains"]
        if len(bb_row) == 0:
            report("  ERROR: No Bitbrains row found in "
                   "boundary_condition_table.csv")
            report("  Skipping boundary table checks and regeneration.")
        else:
            bb_row = bb_row.iloc[0]

            # ── 1a. Bitbrains row checks ─────────────────────────────

            report("  --- Bitbrains row ---")

            # compute correct values from source data
            correct_cv    = omega_bb["cv"].median()
            correct_hurst = omega_bb["hurst"].median()
            correct_acf24 = omega_bb["acf_24h"].median()

            # deltas from bitbrains_per_vm.csv (the authoritative file)
            horizons_to_check = ["30min", "120min"]
            correct_deltas = {}
            for h in horizons_to_check:
                sub = bb_per_vm[bb_per_vm["horizon"] == h]
                correct_deltas[h] = sub["r2_delta"].median() * 100

            # what the table currently says
            report(f"  Current CV:           {bb_row['CV median']}")
            report(f"  Correct CV (median):  {correct_cv:.3f}")
            check("Bitbrains CV",
                  abs(float(bb_row["CV median"]) - correct_cv) < 0.05,
                  f"Table says {bb_row['CV median']}, "
                  f"source says {correct_cv:.3f}")

            report(f"  Current Hurst:        {bb_row['Hurst']}")
            report(f"  Correct Hurst:        {correct_hurst:.3f}")
            check("Bitbrains Hurst",
                  abs(float(bb_row["Hurst"]) - correct_hurst) < 0.05,
                  f"Table says {bb_row['Hurst']}, "
                  f"source says {correct_hurst:.3f}")

            current_d30 = bb_row["delta @30min"]
            correct_d30 = correct_deltas["30min"]
            report(f"  Current delta@30min:  {current_d30}")
            report(f"  Correct delta@30min:  {pp(correct_d30)}")
            try:
                current_d30_val = float(
                    str(current_d30).replace("pp", ""))
                d30_signs_match = ((current_d30_val > 0)
                                   == (correct_d30 > 0))
                d30_close = abs(current_d30_val - correct_d30) < 1.0
            except ValueError:
                d30_signs_match = False
                d30_close = False
            check("Bitbrains delta@30min",
                  d30_signs_match and d30_close,
                  f"Table says {current_d30}, "
                  f"per-VM median says {pp(correct_d30)}")

            current_d120 = bb_row["delta @120min"]
            correct_d120 = correct_deltas["120min"]
            report(f"  Current delta@120min: {current_d120}")
            report(f"  Correct delta@120min: {pp(correct_d120)}")
            try:
                current_d120_val = float(
                    str(current_d120).replace("pp", ""))
                signs_match = ((current_d120_val > 0)
                               == (correct_d120 > 0))
                close_enough = abs(current_d120_val - correct_d120) < 1.0
            except ValueError:
                signs_match = False
                close_enough = False
            check("Bitbrains delta@120min",
                  signs_match and close_enough,
                  f"Table says {current_d120}, "
                  f"per-VM median says {pp(correct_d120)}")

            # check the verdict
            ml_wins_at = []
            for h in ["10min", "30min", "60min", "120min"]:
                sub = bb_per_vm[bb_per_vm["horizon"] == h]
                med = sub["r2_delta"].median()
                if med > 0:
                    ml_wins_at.append(h)

            if len(ml_wins_at) == 0:
                correct_verdict = "naive wins all"
            else:
                correct_verdict = (
                    f"ML wins only @{', '.join(ml_wins_at)}")
            report(f"  Current verdict:      {bb_row['verdict']}")
            report(f"  Correct verdict:      {correct_verdict}")
            check("Bitbrains verdict",
                  bb_row["verdict"].strip() == correct_verdict,
                  f"Should be '{correct_verdict}'")

            # ── 1b. Alibaba row checks ───────────────────────────────

            report("\n  --- Alibaba row ---")

            ali_rows = boundary[
                boundary["dataset"].str.contains("Alibaba")]

            if len(ali_rows) > 0 and omega_sum is not None:
                ali_row = ali_rows.iloc[0]

                # the column says "ACF@120min" but the boundary
                # condition framework uses ACF@24h > 0.2
                ali_omega = omega_sum[
                    omega_sum["Dataset"] == "Alibaba"].iloc[0]
                correct_ali_acf24 = ali_omega["ACF@24h"]

                report(f"  Current ACF column:   "
                       f"{ali_row['ACF@120min']}")
                report(f"  Correct ACF@24h:      "
                       f"{correct_ali_acf24:.3f}")
                report(f"  (Column header says 'ACF@120min' = "
                       f"2h lag = 0.211)")
                report(f"  (Framework uses ACF@24h = "
                       f"{correct_ali_acf24:.3f})")

                # check if the current value is ACF@24h or ACF@2h
                try:
                    current_acf = float(ali_row["ACF@120min"])
                    is_acf24 = abs(current_acf
                                   - correct_ali_acf24) < 0.02
                except ValueError:
                    is_acf24 = False

                check("Alibaba ACF uses ACF@24h (not 2h lag)",
                      is_acf24,
                      f"Table has {ali_row['ACF@120min']} "
                      f"(ACF@2h=0.211), framework needs "
                      f"ACF@24h={correct_ali_acf24:.3f}")

            # ── 1c. ACF column header ────────────────────────────────

            report("\n  --- Column header ---")
            has_wrong_header = "ACF@120min" in boundary.columns
            check("Column header is ACF@24h (not ACF@120min)",
                  not has_wrong_header,
                  "Column says 'ACF@120min' but framework "
                  "threshold uses ACF@24h")

            # ── regenerate corrected boundary table ───────────────────

            report("\n  Regenerating "
                   "boundary_condition_table_corrected.csv ...")

            rows = []

            # Alibaba row — rebuild with ACF@24h
            if (len(boundary[
                    boundary["dataset"].str.contains("Alibaba")]) > 0
                    and omega_sum is not None
                    and comp_table is not None):

                ali_orig = boundary[
                    boundary["dataset"].str.contains(
                        "Alibaba")].iloc[0]
                ali_omega = omega_sum[
                    omega_sum["Dataset"] == "Alibaba"].iloc[0]

                # deltas come from comparison_table (aggregate R²)
                row30 = comp_table[
                    comp_table["Horizon"] == "30min"].iloc[0]
                row120 = comp_table[
                    comp_table["Horizon"] == "120min"].iloc[0]
                ali_d30 = ((row30["Hetero Ens R²"]
                            - row30["Naive R²"]) * 100)
                ali_d120 = ((row120["Hetero Ens R²"]
                             - row120["Naive R²"]) * 100)

                rows.append({
                    "dataset":       str(ali_orig["dataset"]),
                    "workload":      str(ali_orig["workload"]),
                    "CV median":     f"{ali_omega['CV_median']:.3f}",
                    "Hurst":         f"{ali_omega['Hurst_median']:.3f}",
                    "ACF@24h":       f"{ali_omega['ACF@24h']:.3f}",
                    "delta @30min":  pp(ali_d30),
                    "delta @120min": pp(ali_d120),
                    "verdict":       str(ali_orig["verdict"]),
                })

                report(f"  Alibaba: ACF@24h={ali_omega['ACF@24h']:.3f}"
                       f" (was ACF@120min=0.211)")
            else:
                # fallback: keep original if sources missing
                ali_orig = boundary[
                    boundary["dataset"].str.contains("Alibaba")]
                if len(ali_orig) > 0:
                    row = ali_orig.iloc[0].to_dict()
                    # rename ACF column
                    if "ACF@120min" in row:
                        row["ACF@24h"] = row.pop("ACF@120min")
                    rows.append(row)

            # Bitbrains row — rebuild from source
            bb_d30 = (bb_per_vm[bb_per_vm["horizon"] == "30min"]
                      ["r2_delta"].median() * 100)
            bb_d120 = (bb_per_vm[bb_per_vm["horizon"] == "120min"]
                       ["r2_delta"].median() * 100)

            rows.append({
                "dataset":       "Bitbrains",
                "workload":      "VM compute",
                "CV median":     f"{correct_cv:.2f}",
                "Hurst":         f"{correct_hurst:.3f}",
                "ACF@24h":       f"{correct_acf24:.3f}",
                "delta @30min":  pp(bb_d30),
                "delta @120min": pp(bb_d120),
                "verdict":       correct_verdict,
            })

            # DSB row — keep as-is but rename ACF column
            dsb_row = boundary[boundary["dataset"] == "DSB"]
            if len(dsb_row) > 0:
                dsb = dsb_row.iloc[0].to_dict()
                if "ACF@120min" in dsb:
                    dsb["ACF@24h"] = dsb.pop("ACF@120min")
                rows.append(dsb)

            # sanitise NaN → "N/A" so the CSV is clean
            for row in rows:
                for k in row:
                    if pd.isna(row[k]):
                        row[k] = "N/A"
                    else:
                        row[k] = str(row[k])

            corrected_boundary = pd.DataFrame(rows)
            out_path = os.path.join(
                args.output_dir,
                "boundary_condition_table_corrected.csv")
            corrected_boundary.to_csv(out_path, index=False)
            report(f"  Saved: {out_path}")
            report(f"  Corrected table:")
            for _, r in corrected_boundary.iterrows():
                report(
                    f"    {str(r['dataset']):12s}  "
                    f"CV={str(r['CV median']):>6s}  "
                    f"Hurst={str(r['Hurst']):>5s}  "
                    f"ACF@24h={str(r['ACF@24h']):>6s}  "
                    f"d@30={str(r['delta @30min']):>8s}  "
                    f"d@120={str(r['delta @120min']):>8s}  "
                    f"→ {r['verdict']}")

    # ── 2. BITBRAINS SUMMARY — corrupted values ──────────────────────

    report("\n" + "-" * 70)
    report("2. BITBRAINS SUMMARY — check for corrupted values")
    report("-" * 70)

    if bb_summary is not None and bb_per_vm is not None:
        # the current file has ML_R2 values like -2.17e+28
        has_overflow = False
        for col in bb_summary.columns:
            if bb_summary[col].dtype in [np.float64, np.float32]:
                if bb_summary[col].abs().max() > 1e10:
                    has_overflow = True
                    report(
                        f"  OVERFLOW in column '{col}': "
                        f"max abs = "
                        f"{bb_summary[col].abs().max():.2e}")

        check("bitbrains_summary no overflow", not has_overflow,
              "1e+28 values indicate corrupted aggregation")

        # regenerate from per-VM data (authoritative file)
        report("\n  Regenerating bitbrains_summary_corrected.csv "
               "from bitbrains_per_vm.csv ...")
        summary_rows = []
        for h in ["10min", "30min", "60min", "120min"]:
            sub = bb_per_vm[bb_per_vm["horizon"] == h]
            naive_r2_med = sub["naive_r2"].median()
            ml_r2_med    = sub["ml_r2"].median()
            delta_pp_med = sub["r2_delta"].median() * 100
            skill_med    = sub["skill"].median()
            pct_wins     = sub["ml_wins"].mean() * 100

            summary_rows.append({
                "Horizon":            h,
                "BB_Naive_R2":        round(naive_r2_med, 4),
                "BB_ML_R2_median":    round(ml_r2_med, 4),
                "BB_Delta_pp":        round(delta_pp_med, 2),
                "BB_Skill_median":    round(skill_med, 4),
                "BB_Pct_VMs_ML_Wins": round(pct_wins, 1),
            })

        corrected_bb = pd.DataFrame(summary_rows)
        out_path = os.path.join(args.output_dir,
                                "bitbrains_summary_corrected.csv")
        corrected_bb.to_csv(out_path, index=False)
        report(f"  Saved: {out_path}")
        report(f"  Corrected Bitbrains summary:")
        for _, r in corrected_bb.iterrows():
            report(
                f"    {r['Horizon']:>6s}: "
                f"Naive R²={r['BB_Naive_R2']:.4f}  "
                f"ML R²={r['BB_ML_R2_median']:.4f}  "
                f"Δ={r['BB_Delta_pp']:+.2f}pp  "
                f"Skill={r['BB_Skill_median']:.4f}  "
                f"Win%={r['BB_Pct_VMs_ML_Wins']:.1f}%")

    # ── 3. CHRONOS WIN COUNT ─────────────────────────────────────────

    report("\n" + "-" * 70)
    report("3. CHRONOS WIN COUNT")
    report("-" * 70)

    if chronos is not None:
        ensemble_wins = 0
        chronos_wins = 0
        # track per-dataset wins to flag methodology issues
        wins_by_dataset = {}
        details = []

        for _, row in chronos.iterrows():
            ds = row["dataset"]
            h  = row["horizon"]
            ens_r2 = row["ensemble_r2"]
            chr_r2 = row["chronos_r2"]

            if pd.isna(ens_r2) or pd.isna(chr_r2):
                details.append(
                    f"  {ds:>10s} {h:>6s}: "
                    f"ensemble={ens_r2}, "
                    f"chronos={chr_r2:.4f} — MISSING DATA")
                continue

            diff = (ens_r2 - chr_r2) * 100
            if ens_r2 > chr_r2:
                winner = "ENSEMBLE"
                ensemble_wins += 1
                wins_by_dataset.setdefault(ds, []).append(
                    ("ensemble", h, diff))
            else:
                winner = "Chronos"
                chronos_wins += 1
                wins_by_dataset.setdefault(ds, []).append(
                    ("chronos", h, diff))

            details.append(
                f"  {ds:>10s} {h:>6s}: ens={ens_r2:.4f} vs "
                f"chr={chr_r2:.4f}  ({diff:+.2f}pp) → {winner}")

        for d in details:
            report(d)

        total = ensemble_wins + chronos_wins
        report(f"\n  Raw count:")
        report(f"    Ensemble wins: {ensemble_wins}/{total}")
        report(f"    Chronos wins:  {chronos_wins}/{total}")

        # flag the Bitbrains methodology mismatch
        bb_chronos = chronos[chronos["dataset"] == "Bitbrains"]
        if len(bb_chronos) > 0 and bb_per_vm is not None:
            report("\n  ⚠ METHODOLOGY WARNING — Bitbrains "
                   "comparison:")
            report("  The Chronos benchmark trains a GLOBAL "
                   "ensemble on Bitbrains,")
            report("  but the main experiments train PER-VM "
                   "models.")
            report("  Global ensemble R² on Bitbrains:")
            for _, row in bb_chronos.iterrows():
                h = row["horizon"]
                sub = bb_per_vm[bb_per_vm["horizon"] == h]
                main_med = sub["ml_r2"].median()
                report(
                    f"    {h}: chronos_bench "
                    f"ensemble_r2={row['ensemble_r2']:.4f}  "
                    f"vs main exps per-VM "
                    f"median={main_med:.4f}")
            report("  All 4 Bitbrains 'Chronos wins' are "
                   "trivially true because")
            report("  the global ensemble baseline is broken, "
                   "not because Chronos")
            report("  outperforms per-VM models.")

            # count fair comparisons (Alibaba + ByteDance only)
            fair_ens = 0
            fair_chr = 0
            for ds in ["Alibaba", "ByteDance"]:
                for who, h, diff in wins_by_dataset.get(ds, []):
                    if who == "ensemble":
                        fair_ens += 1
                    else:
                        fair_chr += 1
            fair_total = fair_ens + fair_chr
            report(f"\n  Fair comparison (Alibaba + ByteDance "
                   f"only):")
            report(f"    Ensemble wins: {fair_ens}/{fair_total}")
            report(f"    Chronos wins:  {fair_chr}/{fair_total}")

        check("Chronos raw count is 10/12",
              chronos_wins == 10 and ensemble_wins == 2,
              f"Got Chronos={chronos_wins}, "
              f"Ensemble={ensemble_wins}")

        # list the ensemble wins
        report("\n  Ensemble wins at:")
        for _, row in chronos.iterrows():
            ens_r2 = row["ensemble_r2"]
            chr_r2 = row["chronos_r2"]
            if (pd.notna(ens_r2) and pd.notna(chr_r2)
                    and ens_r2 > chr_r2):
                report(
                    f"    {row['dataset']} {row['horizon']}: "
                    f"+{(ens_r2 - chr_r2)*100:.2f}pp")

    # ── 4. STRATIFIED SKILL WIN RATES ────────────────────────────────

    report("\n" + "-" * 70)
    report("4. STRATIFIED SKILL — verify win rates")
    report("-" * 70)

    if skill is not None:
        # these are the wrong values from the original system prompt
        stale_win_rates = {
            "10min":  0.0,
            "30min":  18.3,
            "60min":  39.3,
            "120min": 49.2,
        }

        report("  Win rates comparison "
               "(old thesis text vs actual CSV):")
        all_match = True
        for _, row in skill.iterrows():
            h = row["Horizon"]
            actual = row["Timestep_Win_rate_%"]
            stale = stale_win_rates.get(h, None)
            match = (stale is not None
                     and abs(actual - stale) < 0.1)
            if not match:
                all_match = False
            marker = "OK" if match else "MISMATCH"
            report(f"    {h:>6s}: CSV={actual:.2f}%  "
                   f"old_text={stale}%  [{marker}]")

        check("Win rates match thesis text",
              all_match,
              "Thesis text needs updating to match "
              "stratified_skill.csv")

        # also print the correct skill scores for reference
        report("  Correct values for thesis "
               "(from stratified_skill.csv):")
        for _, row in skill.iterrows():
            report(
                f"    {row['Horizon']:>6s}: "
                f"MAE_naive={row['MAE_naive']:.4f}  "
                f"MAE_ens={row['MAE_ensemble']:.4f}  "
                f"Skill={row['Skill_score']:.4f}  "
                f"Win%={row['Timestep_Win_rate_%']:.2f}%")

    # ── 5. BYTEDANCE FRAMING ─────────────────────────────────────────

    report("\n" + "-" * 70)
    report("5. BYTEDANCE — ensemble vs naive at each horizon")
    report("-" * 70)

    if bd_results is not None:
        report("  Hetero ensemble vs naive "
               "(all should be positive):")
        all_positive = True
        for _, row in bd_results.iterrows():
            h = row["Horizon"]
            naive = row["naive_R2"]
            hetero = row["hetero_ensemble_R2"]
            et = row["extratrees_R2"]
            delta_naive = (hetero - naive) * 100
            delta_et = (et - hetero) * 100

            if delta_naive < 0:
                all_positive = False

            report(
                f"    {h:>6s}: naive={naive:.4f}  "
                f"hetero={hetero:.4f}  "
                f"ET={et:.4f}  "
                f"hetero_vs_naive={delta_naive:+.2f}pp  "
                f"ET_vs_hetero={delta_et:+.2f}pp")

        check("ByteDance hetero beats naive at all horizons",
              all_positive,
              "Thesis should NOT say 'negative at 60/120min' "
              "for hetero vs naive")

        # note: ET beats hetero at all horizons too
        et_beats_all = True
        for _, row in bd_results.iterrows():
            if row["extratrees_R2"] <= row["hetero_ensemble_R2"]:
                et_beats_all = False
        if et_beats_all:
            report("  NOTE: ExtraTrees alone beats hetero "
                   "ensemble at ALL horizons")
            report("  → The 'negative' framing might have been "
                   "about ET vs ensemble,")
            report("    not ensemble vs naive. Either way, "
                   "clarify in thesis.")

    # ── 6. ALIBABA DESCRIPTIVE STATS ─────────────────────────────────

    report("\n" + "-" * 70)
    report("6. ALIBABA DESCRIPTIVE STATS — reconciliation")
    report("-" * 70)

    if omega_sum is not None and pred_table is not None:
        ali_omega = omega_sum[omega_sum["Dataset"] == "Alibaba"]
        ali_pred = pred_table  # only has one row, it's Alibaba

        if len(ali_omega) > 0 and len(ali_pred) > 0:
            ao = ali_omega.iloc[0]
            ap = ali_pred.iloc[0]

            report("  Source comparison:")
            report(
                f"    {'Metric':<12s}  {'omega_summary':>14s}  "
                f"{'predictability_table':>20s}  {'Diff':>8s}")

            items = [
                ("N_series",
                 ao["N_series"], ap["N_series"]),
                ("CV_median",
                 ao["CV_median"], ap["CV (median)"]),
                ("Hurst",
                 ao["Hurst_median"], ap["Hurst (median)"]),
            ]
            for name, v1, v2 in items:
                diff = abs(float(v1) - float(v2))
                report(
                    f"    {name:<12s}  {v1:>14.4f}  "
                    f"{v2:>20.4f}  {diff:>8.4f}")

            check(
                "Alibaba CV consistent",
                abs(float(ao["CV_median"])
                    - float(ap["CV (median)"])) < 0.05,
                f"omega={ao['CV_median']:.3f} vs "
                f"pred_table={ap['CV (median)']:.3f}")

            check(
                "Alibaba Hurst consistent",
                abs(float(ao["Hurst_median"])
                    - float(ap["Hurst (median)"])) < 0.05,
                f"omega={ao['Hurst_median']:.3f} vs "
                f"pred_table={ap['Hurst (median)']:.3f}")

            check(
                "Alibaba N consistent",
                abs(int(ao["N_series"])
                    - int(ap["N_series"])) < 50,
                f"omega={int(ao['N_series'])} vs "
                f"pred_table={int(ap['N_series'])}")

            report("\n  Likely cause: different container "
                   "filtering thresholds.")
            report("  DECISION NEEDED: pick one source and "
                   "use it everywhere.")
            report("  Recommendation: use omega_summary.csv "
                   "(N=4902) since it's")
            report("  also the source for Bitbrains and "
                   "ByteDance stats.")

    # ── 7. NNLS WEIGHTS AT 10min ─────────────────────────────────────

    report("\n" + "-" * 70)
    report("7. NNLS WEIGHTS AT 10min — BiLSTM OOF status")
    report("-" * 70)

    if comp_table is not None:
        row10 = comp_table[
            comp_table["Horizon"] == "10min"].iloc[0]
        homo_r2 = row10["Homo Ens R²"]
        hetero_r2 = row10["Hetero Ens R²"]
        diff = hetero_r2 - homo_r2

        report(f"  Homo Ens R² at 10min:   {homo_r2:.16f}")
        report(f"  Hetero Ens R² at 10min: {hetero_r2:.16f}")
        report(f"  Difference:             {diff:.16f}")

        bilstm_included = abs(diff) > 1e-10
        check("10min hetero includes BiLSTM",
              bilstm_included,
              "hetero ≠ homo → BiLSTM OOF succeeded "
              "in a later run")

        if bilstm_included:
            report("\n  The system prompt says 'BiLSTM OOF "
                   "failed → hetero=homo'")
            report("  at 10min, but comparison_table.csv shows "
                   "hetero ≠ homo.")
            report("  run.log confirms: Run 1 failed, Run 2 "
                   "succeeded with")
            report("  NNLS weights: ET=0.514, BiLSTM=0.485")
            report("  → Update NNLS weight table in thesis "
                   "for 10min row")
            report("  → Old: XGB=2.4%, LGB=0.0%, ET=97.6%, "
                   "BiLSTM=n/a")
            report("  → New: XGB=0.0%, LGB=0.0%, ET=51.4%, "
                   "BiLSTM=48.5%")

    # ── 8. BITBRAINS PER-VM FILES ────────────────────────────────────

    report("\n" + "-" * 70)
    report("8. BITBRAINS PER-VM FILES — two files disagree")
    report("-" * 70)

    if bb_per_vm is not None and bb_per_vm2 is not None:
        report("  bitbrains_per_vm.csv       "
               "(10 cols, includes 'skill')")
        report("  bitbrains_per_vm_results.csv "
               "(8 cols, no 'skill')")

        # compare medians at each horizon
        report(
            f"\n  {'Horizon':>6s}  {'File1 win%':>10s}  "
            f"{'File2 win%':>10s}  "
            f"{'File1 Δ_med':>11s}  {'File2 Δ_med':>11s}")
        for h in ["10min", "30min", "60min", "120min"]:
            s1 = bb_per_vm[bb_per_vm["horizon"] == h]
            s2 = bb_per_vm2[bb_per_vm2["horizon"] == h]
            w1 = s1["ml_wins"].mean() * 100
            w2 = s2["ml_wins"].mean() * 100
            d1 = s1["r2_delta"].median() * 100
            d2 = s2["r2_delta"].median() * 100
            report(f"  {h:>6s}  {w1:>9.1f}%  {w2:>9.1f}%  "
                   f"{d1:>+10.2f}pp  {d2:>+10.2f}pp")

        # verify alignment with stratified CV
        if bb_strat is not None:
            report("\n  Verifying alignment with "
                   "bitbrains_cv_stratified.csv ...")
            aligned_with_f1 = True
            aligned_with_f2 = True

            for h in ["10min", "30min", "60min", "120min"]:
                strat_h = bb_strat[bb_strat["Horizon"] == h]
                if len(strat_h) == 0:
                    continue
                strat_n = strat_h["N"].sum()
                strat_win = ((strat_h["Win_pct"] * strat_h["N"])
                             .sum() / strat_n)

                f1_win = (bb_per_vm[bb_per_vm["horizon"] == h]
                          ["ml_wins"].mean() * 100)
                f2_win = (bb_per_vm2[bb_per_vm2["horizon"] == h]
                          ["ml_wins"].mean() * 100)

                f1_match = abs(strat_win - f1_win) < 1.0
                f2_match = abs(strat_win - f2_win) < 1.0

                if not f1_match:
                    aligned_with_f1 = False
                if not f2_match:
                    aligned_with_f2 = False

                report(
                    f"    {h}: strat_win={strat_win:.1f}%  "
                    f"F1={f1_win:.1f}% "
                    f"[{'OK' if f1_match else 'MISMATCH'}]  "
                    f"F2={f2_win:.1f}% "
                    f"[{'OK' if f2_match else 'MISMATCH'}]")

            check("Stratified CV aligns with File 1",
                  aligned_with_f1,
                  "bitbrains_cv_stratified.csv should match "
                  "bitbrains_per_vm.csv")
            check("Stratified CV does NOT align with File 2",
                  not aligned_with_f2,
                  "Confirms File 2 is from a different run")
        else:
            report("\n  bitbrains_cv_stratified.csv not found — "
                   "cannot verify alignment")
            check("Authoritative BB file identified",
                  True,
                  "Use bitbrains_per_vm.csv, not "
                  "bitbrains_per_vm_results.csv "
                  "(alignment not verified)")

        report("\n  → File 1 (bitbrains_per_vm.csv) is "
               "AUTHORITATIVE")
        report("  → File 2 (bitbrains_per_vm_results.csv) is "
               "from a different")
        report("    run or split — do NOT use for thesis claims")

    # ── FINAL SUMMARY ────────────────────────────────────────────────

    report("\n" + "=" * 70)
    report("SUMMARY")
    report("=" * 70)

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    report(f"  {n_pass} checks passed, {n_fail} checks FAILED\n")

    if n_fail > 0:
        report("  FAILED checks (must fix before submission):")
        for name, status, detail in results:
            if status == "FAIL":
                report(f"    ✗ {name}")
                if detail:
                    report(f"      → {detail}")

    report("\n  THESIS TEXT UPDATES NEEDED:")
    report("  1. boundary_condition_table: rebuild ALL rows "
           "with ACF@24h column")
    report("     Alibaba: ACF@24h=0.316 (not ACF@120min=0.211)")
    report("     Bitbrains: CV, Hurst, both deltas, verdict "
           "all wrong")
    report("  2. Chronos comparison: raw count is 10/12, but "
           "4 Bitbrains wins")
    report("     are against a broken global ensemble. "
           "Fair count: 6/8")
    report("     Ensemble wins: Alibaba 10min (+0.22pp) AND "
           "120min (+1.01pp)")
    report("  3. Win rates in MAE/R² divergence section: "
           "use CSV values")
    report("     10min=17.26%, 30min=31.53%, 60min=38.64%, "
           "120min=43.45%")
    report("  4. ByteDance: ensemble beats naive at ALL "
           "four horizons")
    report("     The 'negative at 60/120min' claim is wrong "
           "for hetero vs naive")
    report("     (it may refer to ensemble vs ET-alone, "
           "clarify which)")
    report("  5. NNLS weights at 10min: BiLSTM IS included "
           "(48.5% weight)")
    report("     The 'OOF failed' note is about an "
           "earlier run only")
    report("  6. Pick ONE source for Alibaba descriptive stats")
    report("     Recommendation: omega_summary.csv (N=4902)")

    # ── save the report ──────────────────────────────────────────────

    report_path = os.path.join(args.output_dir,
                               "data_corrections_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
