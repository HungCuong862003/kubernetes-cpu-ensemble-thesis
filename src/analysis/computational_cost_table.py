"""
computational_cost_table.py — Format training cost data for thesis.

All values were manually extracted from the Vast.ai run.log.
This script just formats them into tables — no computation.

Inputs:
    None (timing data hardcoded from run.log)

Outputs:
    computational_cost_table.tex               (--output-dir)
    computational_cost_table.md                (--output-dir)
"""

import os
import argparse

# ── timing data from run.log ────────────────────────────────────────────
# all of these were manually verified from the Vast.ai run log
COST_DATA = {
    "gpu":             "NVIDIA GeForce RTX 5090 (33.7\\,GB VRAM)",
    "gpu_md":          "NVIDIA GeForce RTX 5090 (33.7 GB VRAM)",
    "cpu_cores":       384,
    "execution_mode":  "4 horizons in parallel",
    "wall_clock_hrs":  12.3,
    "horizons": {
        "10min":  738.7,   # minutes
        "30min":  536.8,
        "60min":  177.9,
        "120min":  97.7,
    },
    "bilstm_hpo":      "10 Optuna trials/horizon, approx.\\ 25--50\\,min each",
    "bilstm_hpo_md":   "10 Optuna trials/horizon, ~25-50 min each",
    "tree_caching":    "Cached after first fold",
    "inference":       "Sub-millisecond per sample (tree models, CPU)",
}


def make_latex_table():
    """Build a LaTeX table string."""
    hz = COST_DATA["horizons"]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Computational cost of the full training pipeline. "
                 r"The experiment was conducted on a rented cloud instance (Vast.ai).}")
    lines.append(r"  \label{tab:computational-cost}")
    lines.append(r"  \begin{tabular}{ll}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Item} & \textbf{Value} \\")
    lines.append(r"    \midrule")
    lines.append(f"    GPU & {COST_DATA['gpu']} \\\\")
    lines.append(f"    CPU cores & {COST_DATA['cpu_cores']} \\\\")
    lines.append(f"    Execution mode & {COST_DATA['execution_mode']} \\\\")
    lines.append(f"    Total wall clock & $\\sim${COST_DATA['wall_clock_hrs']:.1f} hours \\\\")
    lines.append(r"    \midrule")
    for hz_name, mins in hz.items():
        lines.append(f"    \\quad {hz_name} horizon & {mins:.1f}\\,min \\\\")
    lines.append(r"    \midrule")
    lines.append(f"    BiLSTM HPO & {COST_DATA['bilstm_hpo']} \\\\")
    lines.append(f"    Tree models & {COST_DATA['tree_caching']} \\\\")
    lines.append(f"    Inference & {COST_DATA['inference']} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_markdown_table():
    """Build a markdown table string."""
    hz = COST_DATA["horizons"]
    lines = []
    lines.append("| Item | Value |")
    lines.append("|------|-------|")
    lines.append(f"| GPU | {COST_DATA['gpu_md']} |")
    lines.append(f"| CPU cores | {COST_DATA['cpu_cores']} |")
    lines.append(f"| Execution mode | {COST_DATA['execution_mode']} |")
    lines.append(f"| Total wall clock | ~{COST_DATA['wall_clock_hrs']:.1f} hours |")
    for hz_name, mins in hz.items():
        lines.append(f"| {hz_name} horizon | {mins:.1f} min |")
    lines.append(f"| BiLSTM HPO | {COST_DATA['bilstm_hpo_md']} |")
    lines.append(f"| Tree models | {COST_DATA['tree_caching']} |")
    lines.append(f"| Inference | {COST_DATA['inference']} |")
    lines.append("")
    lines.append("*Note: Conducted on a rented Vast.ai cloud instance.*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate computational cost table for the thesis")
    parser.add_argument("--output-dir", default="./thesis_figures",
                        help="Where to save the output files")
    parser.add_argument("--format", choices=["latex", "markdown", "both"],
                        default="both", help="Output format (default: both)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.format in ("latex", "both"):
        latex_str = make_latex_table()
        out_path = os.path.join(args.output_dir, "computational_cost_table.tex")
        with open(out_path, "w") as f:
            f.write(latex_str + "\n")
        print(f"LaTeX table saved: {out_path}")
        print()
        print(latex_str)
        print()

    if args.format in ("markdown", "both"):
        md_str = make_markdown_table()
        out_path = os.path.join(args.output_dir, "computational_cost_table.md")
        with open(out_path, "w") as f:
            f.write(md_str + "\n")
        print(f"Markdown table saved: {out_path}")
        print()
        print(md_str)


if __name__ == "__main__":
    main()