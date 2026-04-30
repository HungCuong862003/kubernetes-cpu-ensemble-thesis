"""
fix_uq_figure.py — Regenerate UQ figure and CSV (AgACI, no rescaling).

Produces both figure and CSV from the SAME computation, matching the
`run_uq()` function in thesis_analysis.py. Reports original AgACI
prediction intervals directly — no post-hoc rescaling.

Evaluation protocol matches thesis_analysis.py exactly:
    * residual-space intervals converted to absolute CPU space via + naive
    * test set split in half: n_cal = n // 2
    * coverage and mean width computed on the SECOND HALF only
    * plot shows the first 500 samples of the evaluation segment

Inputs:
    <DATA_DIR>/test.parquet (or test.csv)
    <CKPT_DIR>/<hz>/cqr_intervals.npz
    <CKPT_DIR>/<hz>/pred_naive.npy
    <CKPT_DIR>/<hz>/pred_hetero_ensemble.npy

Outputs (written into OUTPUT_DIR):
    uq_recalibration.pdf
    uq_recalibration_table.csv

Run on Colab:
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/drive/MyDrive/k8s-ensemble-forecast/src
    !python fix_uq_figure.py

Run locally:
    python fix_uq_figure.py \
        --drive-root "E:\\K8S-ENSEMBLE-FORECAST" \
        --output-dir "E:\\K8S-ENSEMBLE-FORECAST\\results\\thesis_figures"
"""

import argparse
import importlib.util
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── CONFIG (defaults; override with CLI flags) ──────────────────────
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/k8s-ensemble-forecast"

HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}
CQR_ALPHA = 0.2            # 80% nominal coverage
PLOT_POINTS = 500          # number of timesteps drawn in each subplot

HZ_COLORS = {
    "10min":  "#4C72B0",
    "30min":  "#DD8452",
    "60min":  "#55A868",
    "120min": "#C44E52",
}

# Same 6-column dropna list sprint1_Main.py uses so row counts match.
REQUIRED_COLS = [
    "cpu_target", "cpu_residual", "naive_cpu",
    "mem_target", "mem_residual", "naive_mem",
]


# ── IMPORT create_features ──────────────────────────────────────────
# thesis_analysis.py defines create_features() as a thin wrapper that
# delegates to sprint1_v9.create_features — which requires calling
# load_sprint1() first, which in turn looks for a file literally named
# "sprint1_v9.py". In this project that file is called sprint1_Main.py.
#
# The robust path is therefore to import create_features DIRECTLY from
# sprint1_Main.py (or, as a fallback, whatever the thesis_analysis
# wrapper delegates to).
def import_create_features(drive_root):
    """Find and import create_features() from the authoritative source."""
    # When the script is pasted into a Jupyter/Colab cell rather than
    # executed as a file, __file__ is undefined. Fall back to cwd.
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()

    # Prefer sprint1_Main.py: it defines create_features as a standalone
    # function with no hidden state. The thesis_analysis.py wrapper is
    # a fallback only.
    candidates = [
        (os.path.join(here, "sprint1_Main.py"),              "direct"),
        (os.path.join(drive_root, "src", "sprint1_Main.py"), "direct"),
        (os.path.join(here, "thesis_analysis.py"),              "wrapper"),
        (os.path.join(drive_root, "src", "thesis_analysis.py"), "wrapper"),
    ]

    for fpath, kind in candidates:
        if not os.path.exists(fpath):
            continue
        try:
            spec = importlib.util.spec_from_file_location("_thesis_mod", fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "create_features"):
                continue

            if kind == "wrapper":
                # thesis_analysis.py's create_features is a wrapper around
                # sprint1_v9. We must call its load_sprint1() first, pointing
                # at whichever directory contains sprint1_Main.py (the real
                # implementation). Call it here so the wrapper is usable.
                sprint1_dirs = [here, os.path.join(drive_root, "src")]
                loaded = False
                for sd in sprint1_dirs:
                    sprint1_path = os.path.join(sd, "sprint1_Main.py")
                    if not os.path.exists(sprint1_path):
                        continue
                    # The wrapper's load_sprint1 expects a file named
                    # sprint1_v9.py. If the real file is named something
                    # else, bypass the wrapper entirely by poking its
                    # module-level _s1 attribute directly.
                    try:
                        sub_spec = importlib.util.spec_from_file_location(
                            "sprint1_v9", sprint1_path)
                        sub_mod = importlib.util.module_from_spec(sub_spec)
                        sub_spec.loader.exec_module(sub_mod)
                        mod._s1 = sub_mod
                        print(f"[info] preloaded sprint1_v9 from {sprint1_path}")
                        loaded = True
                        break
                    except Exception as exc:
                        print(f"[warn] preload of {sprint1_path} failed: {exc}")
                if not loaded:
                    print(f"[warn] could not preload sprint1 for the "
                          f"thesis_analysis wrapper at {fpath}")
                    continue

            print(f"[info] imported create_features from {fpath}")
            return mod.create_features
        except Exception as exc:
            print(f"[warn] failed to import from {fpath}: {exc}")

    print("[ERROR] cannot find a usable create_features implementation. "
          "Tried:")
    for fpath, _ in candidates:
        print(f"          {fpath}")
    sys.exit(1)


# ── HELPERS ─────────────────────────────────────────────────────────
def load_test_data(data_dir):
    """Load the test split (parquet or csv)."""
    for ext in (".parquet", ".csv"):
        p = os.path.join(data_dir, f"test{ext}")
        if os.path.exists(p):
            print(f"[info] loading {p}")
            if ext == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Cannot find test.parquet or test.csv in {data_dir}"
    )


def load_npy(ckpt_dir, hz, name):
    """Load a saved prediction array from <ckpt_dir>/<hz>/pred_<name>.npy."""
    p = os.path.join(ckpt_dir, hz, f"pred_{name}.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="c").astype(np.float32)
    return None


def load_cqr(ckpt_dir, hz):
    """Load AgACI/CQR intervals (residual space) from cqr_intervals.npz."""
    p = os.path.join(ckpt_dir, hz, "cqr_intervals.npz")
    if not os.path.exists(p):
        print(f"  [{hz}] WARNING: cqr_intervals.npz not found")
        return None, None

    data = np.load(p, mmap_mode="c")
    lo = data["lower"].astype(np.float32)
    hi = data["upper"].astype(np.float32)
    return lo, hi


def fe_for_horizon(raw_df, hz_steps, create_features):
    """Run feature engineering and drop rows with NaN targets."""
    fe = create_features(raw_df, hz_steps, split_id="test")
    fe = fe.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    return fe


# ── MAIN ────────────────────────────────────────────────────────────
def _clean_argv(argv):
    """Strip Jupyter's ``-f /path/to/kernel.json`` pair from argv.

    When this script is run inside a Jupyter/Colab cell (via %run or by
    pasting), the kernel injects its own -f kernel-file argument into
    sys.argv and argparse rejects it. Filter it out so the script works
    in both shell and notebook environments.
    """
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-f" and i + 1 < len(argv) and argv[i + 1].endswith(".json"):
            i += 2  # skip "-f" and the kernel-file path
            continue
        out.append(a)
        i += 1
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate UQ figure and CSV from AgACI checkpoints."
    )
    parser.add_argument(
        "--drive-root",
        default=DEFAULT_DRIVE_ROOT,
        help=f"Project root. Default: {DEFAULT_DRIVE_ROOT}",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Test-data dir. Default: <drive-root>/data/alibaba",
    )
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="Checkpoints dir. Default: <drive-root>/results/alibaba",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=("Where to write the figure and CSV. Default: "
              "<drive-root>/results/thesis_figures"),
    )
    args = parser.parse_args(_clean_argv(sys.argv[1:]))

    drive_root = args.drive_root
    data_dir   = args.data_dir   or os.path.join(drive_root, "data", "alibaba")
    ckpt_dir   = args.ckpt_dir   or os.path.join(drive_root, "results", "alibaba")
    output_dir = args.output_dir or os.path.join(drive_root, "results", "thesis_figures")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("UQ Figure + CSV Generator (AgACI, no rescaling)")
    print("=" * 60)
    print(f"drive_root: {drive_root}")
    print(f"data_dir:   {data_dir}")
    print(f"ckpt_dir:   {ckpt_dir}")
    print(f"output_dir: {output_dir}")
    print()

    create_features = import_create_features(drive_root)
    test_raw = load_test_data(data_dir)

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, (hz, hz_steps) in enumerate(HORIZONS.items()):
        print(f"\n[{hz}] processing...")

        # --- load predictions and AgACI intervals ---
        hetero_pred = load_npy(ckpt_dir, hz, "hetero_ensemble")
        naive_pred  = load_npy(ckpt_dir, hz, "naive")
        cqr_lo_res, cqr_hi_res = load_cqr(ckpt_dir, hz)

        if hetero_pred is None or naive_pred is None:
            print(f"  [{hz}] missing predictions, skipping")
            axes[i].set_visible(False)
            continue
        if cqr_lo_res is None:
            print(f"  [{hz}] missing CQR intervals, skipping")
            axes[i].set_visible(False)
            continue

        # --- build test targets via feature engineering ---
        test_fe = fe_for_horizon(test_raw, hz_steps, create_features)
        y_test  = test_fe["cpu_target"].values.astype(np.float32)

        # align array lengths (take the last n samples of each)
        n = min(len(y_test), len(hetero_pred), len(naive_pred),
                len(cqr_lo_res), len(cqr_hi_res))
        y_test = y_test[-n:]
        y_pred = hetero_pred[-n:]
        naive  = naive_pred[-n:]

        # convert CQR intervals from residual to absolute CPU space
        cqr_lo = cqr_lo_res[-n:] + naive
        cqr_hi = cqr_hi_res[-n:] + naive

        # --- evaluate on SECOND HALF only (matches run_uq) ---
        n_cal   = n // 2
        eval_y  = y_test[n_cal:]
        eval_lo = cqr_lo[n_cal:]
        eval_hi = cqr_hi[n_cal:]
        eval_p  = y_pred[n_cal:]

        coverage = float(np.mean((eval_y >= eval_lo) & (eval_y <= eval_hi)))
        width    = float(np.mean(eval_hi - eval_lo))

        print(f"  [{hz}] n_total={n}, n_eval={len(eval_y)}")
        print(f"  [{hz}] AgACI coverage: {coverage:.4f} ({coverage*100:.1f}%)")
        print(f"  [{hz}] mean width:     {width:.4f}")

        # --- store row for CSV ---
        rows.append({
            "Horizon":         hz,
            "Coverage":        round(coverage, 4),
            "Nominal":         1.0 - CQR_ALPHA,
            "Coverage_Gap_pp": round((coverage - (1.0 - CQR_ALPHA)) * 100, 2),
            "Mean_Width":      round(width, 4),
            "Method":          "AgACI (Zaffran et al., ICML 2022)",
        })

        # --- plot: show first PLOT_POINTS samples of the EVALUATION segment ---
        ax = axes[i]
        n_p = min(PLOT_POINTS, len(eval_y))
        xs = np.arange(n_p)

        ax.fill_between(xs, eval_lo[:n_p], eval_hi[:n_p],
                        alpha=0.25, color=HZ_COLORS[hz],
                        label="AgACI 80% PI")
        ax.plot(xs, eval_y[:n_p], "k-", lw=0.8, alpha=0.7, label="actual")
        ax.plot(xs, eval_p[:n_p], color=HZ_COLORS[hz], lw=1.0,
                label="ensemble")

        ax.set_title(f"{hz}  {coverage:.1%} coverage", fontsize=10)
        ax.legend(fontsize=8, frameon=False)

    # --- save figure ---
    if rows:
        fig.suptitle("UQ: AgACI prediction intervals (80% nominal)",
                     fontsize=12)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "uq_recalibration.pdf")
        fig.savefig(fig_path, dpi=300)
        print(f"\nsaved: {fig_path}")
    plt.close(fig)

    # --- save CSV ---
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "uq_recalibration_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"saved: {csv_path}")

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(df[["Horizon", "Coverage", "Coverage_Gap_pp",
                  "Mean_Width"]].to_string(index=False))
    else:
        print("\n[ERROR] no horizons succeeded")
        sys.exit(1)


if __name__ == "__main__":
    main()