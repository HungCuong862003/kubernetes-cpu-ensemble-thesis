"""
residual_diagnostics_figure.py — 4×4 residual diagnostic grid.

Generates a publication figure with four rows × four columns:
    Row 1: residuals over time (first ~2000 points)
    Row 2: ACF of residuals (~20 lags)
    Row 3: histogram of residuals
    Row 4: QQ plot against normal distribution

One column per horizon (10, 30, 60, 120 min). Both naive and
ensemble residuals are shown for comparison.

Inputs:
    test.parquet                               (--data-dir)
    pred_naive.npy, pred_hetero_ensemble.npy   (--ckpt-dir/<hz>/)
    sprint1_Main.py                            (--sprint1-dir, for create_features())

Outputs:
    residual_diagnostics_figure.pdf            (--output-dir)
"""

import os
import sys
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf

# ── config ──────────────────────────────────────────────────────────────
HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}
REQUIRED_COLS = [
    "cpu_target", "cpu_residual", "naive_cpu",
    "mem_target", "mem_residual", "naive_mem",
]
N_RESID_PLOT = 2000   # how many residual points to plot in row 1
ACF_NLAGS    = 20     # lags for the ACF plot
HIST_BINS    = 60


# ── sprint1 dynamic import ─────────────────────────────────────────────
_s1 = None

def load_sprint1(sprint1_dir):
    """Import sprint1_Main.py (or sprint1_v9.py) at runtime."""
    global _s1
    # try sprint1_Main first, fall back to sprint1_v9
    for name in ("sprint1_Main", "sprint1_v9"):
        path = os.path.join(sprint1_dir, f"{name}.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _s1 = mod
            print(f"Loaded {name}.py from {sprint1_dir}")
            return
    raise FileNotFoundError(
        f"Neither sprint1_Main.py nor sprint1_v9.py found in {sprint1_dir}")


def fe_for_horizon(raw_df, horizon_steps):
    """
    Run feature engineering for one horizon, then drop rows with NaN targets.
    This reproduces the exact row alignment that sprint1 used when it saved
    the prediction .npy files.
    """
    fe = _s1.create_features(raw_df, horizon_steps, split_id="test")
    fe = fe.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    return fe


# ── I/O helpers ─────────────────────────────────────────────────────────
def find_file(data_dir, stem):
    for ext in (".parquet", ".csv"):
        p = os.path.join(data_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Can't find {stem}.parquet or {stem}.csv in {data_dir}")


def load_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_pred(ckpt_dir, hz, name):
    p = os.path.join(ckpt_dir, hz, f"pred_{name}.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="c")
    return None


# ── main plotting logic ────────────────────────────────────────────────
def get_ensemble_pred(ckpt_dir, hz):
    """
    Load the best available ensemble prediction.
    At 10min, hetero == homo (BiLSTM failed), so either works.
    """
    arr = load_pred(ckpt_dir, hz, "hetero_ensemble")
    if arr is not None:
        return arr, "hetero_ensemble"
    arr = load_pred(ckpt_dir, hz, "homo_ensemble")
    if arr is not None:
        return arr, "homo_ensemble"
    return None, None


def compute_residuals(test_raw, ckpt_dir, hz, hz_steps):
    """
    Reconstruct y_true from test.parquet, load ensemble predictions,
    align lengths, and return the ensemble residuals.
    """
    test_fe = fe_for_horizon(test_raw, hz_steps)
    y_full  = test_fe["cpu_target"].values.astype(np.float32)

    ens_arr, ens_name = get_ensemble_pred(ckpt_dir, hz)
    if ens_arr is None:
        print(f"  WARNING: no ensemble prediction for {hz}, skipping")
        return None

    # align using the tail — same logic as thesis_analysis_v2_fixed.py
    n = min(len(y_full), len(ens_arr))
    y_true = y_full[-n:]
    ens_p  = ens_arr[-n:]

    residuals = (y_true - ens_p).astype(np.float64)
    print(f"  {hz}: loaded {ens_name}, n={n:,}, "
          f"mean_resid={residuals.mean():.6f}, std={residuals.std():.4f}")
    return residuals


def make_figure(all_residuals, output_path):
    """
    Create the 4x4 diagnostic grid.
    Columns = horizons, rows = [time series, ACF, histogram, QQ].
    """
    hz_names = list(HORIZONS.keys())
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    for col_idx, hz in enumerate(hz_names):
        resid = all_residuals[hz]
        if resid is None:
            # blank out the column
            for row in range(4):
                axes[row, col_idx].text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=axes[row, col_idx].transAxes)
                axes[row, col_idx].set_title(hz if row == 0 else "")
            continue

        # ── row 0: residuals over time ──
        ax = axes[0, col_idx]
        n_plot = min(N_RESID_PLOT, len(resid))
        ax.plot(range(n_plot), resid[:n_plot],
                linewidth=0.4, color="steelblue", alpha=0.7)
        ax.axhline(0, color="red", linewidth=1.0, linestyle="-")
        ax.set_title(hz, fontsize=13, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel("Residual")
        ax.set_xlabel("Test sample index")
        ax.tick_params(labelsize=8)

        # ── row 1: ACF plot ──
        ax = axes[1, col_idx]
        # use up to 50k samples for ACF (matches analysis script)
        samp = min(50_000, len(resid))
        acf_vals = acf(resid[:samp], nlags=ACF_NLAGS, fft=True)
        ax.bar(range(ACF_NLAGS + 1), acf_vals,
               color="steelblue", edgecolor="white", width=0.6)
        # 95% confidence band
        ci = 1.96 / np.sqrt(samp)
        ax.axhline(ci,  color="red", linewidth=0.8, linestyle="--")
        ax.axhline(-ci, color="red", linewidth=0.8, linestyle="--")
        ax.axhline(0,   color="black", linewidth=0.5)
        ax.set_ylim(-0.15, 1.05)
        if col_idx == 0:
            ax.set_ylabel("ACF")
        ax.set_xlabel("Lag")
        ax.tick_params(labelsize=8)
        # annotate ACF(1) value
        ax.text(0.95, 0.95, f"ACF(1)={acf_vals[1]:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        # ── row 2: histogram ──
        ax = axes[2, col_idx]
        ax.hist(resid, bins=HIST_BINS, density=True,
                color="steelblue", edgecolor="white", alpha=0.7)
        # overlay normal fit
        mu, sigma = resid.mean(), resid.std()
        x_range = np.linspace(resid.min(), resid.max(), 200)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
                color="red", linewidth=1.5, label="Normal fit")
        if col_idx == 0:
            ax.set_ylabel("Density")
        ax.set_xlabel("Residual")
        ax.tick_params(labelsize=8)
        # annotate bias and std
        ax.text(0.95, 0.95,
                f"bias={mu:.4f}\nstd={sigma:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        # ── row 3: QQ plot ──
        ax = axes[3, col_idx]
        osm, osr = stats.probplot(resid, dist="norm", fit=False)
        ax.scatter(osm, osr, s=1, color="steelblue", alpha=0.3)
        # reference line through Q1-Q3
        q1_t, q3_t = np.percentile(osm, [25, 75])
        q1_r, q3_r = np.percentile(osr, [25, 75])
        slope = (q3_r - q1_r) / (q3_t - q1_t + 1e-12)
        intercept = q1_r - slope * q1_t
        line_x = np.array([osm.min(), osm.max()])
        ax.plot(line_x, slope * line_x + intercept,
                color="red", linewidth=1.5)
        if col_idx == 0:
            ax.set_ylabel("Sample quantiles")
        ax.set_xlabel("Theoretical quantiles")
        ax.tick_params(labelsize=8)

    # row labels on the left edge
    row_labels = [
        "Residuals over time",
        "Autocorrelation (ACF)",
        "Histogram",
        "Q-Q plot (Normal)",
    ]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].annotate(
            label, xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", rotation=90,
            ha="center", va="center")

    fig.suptitle("Ensemble Residual Diagnostics by Forecast Horizon",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.04, 0.0, 1.0, 0.96])

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


# ── entry point ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate residual diagnostics figure (4x4 grid)")
    parser.add_argument("--sprint1-dir", required=True,
                        help="Directory containing sprint1_Main.py")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with train/val/test parquets")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Checkpoint dir (contains 10min/, 30min/, etc.)")
    parser.add_argument("--output-dir", default="./thesis_figures",
                        help="Where to save the PDF")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # step 1: import sprint1 pipeline module
    load_sprint1(args.sprint1_dir)

    # step 2: load test data once (shared across horizons)
    test_path = find_file(args.data_dir, "test")
    print(f"Loading test data from {test_path} ...")
    test_raw = load_df(test_path)
    test_raw.sort_values(["container_id", "time_stamp"], inplace=True)
    test_raw.reset_index(drop=True, inplace=True)
    print(f"  {len(test_raw):,} rows loaded")

    # step 3: compute residuals for each horizon
    all_residuals = {}
    for hz, hz_steps in HORIZONS.items():
        print(f"\nProcessing {hz} (shift={hz_steps}) ...")
        all_residuals[hz] = compute_residuals(
            test_raw, args.ckpt_dir, hz, hz_steps)

    # step 4: plot
    out_path = os.path.join(args.output_dir, "residual_diagnostics_figure.pdf")
    make_figure(all_residuals, out_path)
    print("Done.")


if __name__ == "__main__":
    main()