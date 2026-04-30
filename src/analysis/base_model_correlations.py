"""
base_model_correlations.py — Pairwise error correlation analysis.

Computes prediction correlation and error correlation between all
base model pairs to check whether the ensemble actually benefits
from combining them.

Theory (Ueda & Nakano 1996, Brown et al. 2005 JMLR):
    MSE(ensemble) = bias² + (1/M)·var + (1 − 1/M)·covar
    When error covariance is high, adding more models does nothing.
    We measure this directly from test predictions.

Inputs:
    test.parquet                               (--data-dir)
    pred_*.npy per model per horizon           (--ckpt-dir/<hz>/)

Outputs:
    base_model_correlations.csv                (--output-dir)
"""

import os
import argparse
import importlib.util
import numpy as np
import pandas as pd

# same as sprint1_Main.py
HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}

# the 6 columns that sprint1 drops NaN on -- must match exactly
# or row counts will be different from the saved pred_*.npy
REQUIRED_COLS = [
    "cpu_target", "cpu_residual", "naive_cpu",
    "mem_target", "mem_residual", "naive_mem",
]

# models we want in the correlation matrix
# at 10min bilstm failed so it won't be found, that's fine
MODEL_NAMES = ["xgboost", "lightgbm", "extratrees", "bilstm"]
SHORT_NAMES = {
    "xgboost": "XGB",
    "lightgbm": "LGB",
    "extratrees": "ET",
    "bilstm": "BiLSTM",
}


# ── load sprint1 module at runtime ──────────────────────────────────────
# we need create_features() to reconstruct y_true from test.parquet
# same approach as thesis_analysis_v2_fixed.py

_s1 = None

def load_sprint1(sprint1_dir):
    global _s1
    for name in ("sprint1_Main", "sprint1_v9"):
        path = os.path.join(sprint1_dir, f"{name}.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _s1 = mod
            print(f"Loaded {name}.py")
            return
    raise FileNotFoundError("sprint1_Main.py / sprint1_v9.py not found")


def build_test_features(raw_df, horizon_steps):
    """Run feature engineering for one horizon, drop NaN target rows.
    This gives us the exact same row alignment as the pipeline had
    when it saved the .npy prediction files."""
    fe = _s1.create_features(raw_df, horizon_steps, split_id="test")
    fe = fe.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    return fe


# ── file helpers ────────────────────────────────────────────────────────

def find_file(data_dir, stem):
    for ext in (".parquet", ".csv"):
        p = os.path.join(data_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{stem} not found in {data_dir}")


def load_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_pred(ckpt_dir, hz, name):
    p = os.path.join(ckpt_dir, hz, f"pred_{name}.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="c").astype(np.float64)
    return None


# ── the actual correlation math ─────────────────────────────────────────

def compute_correlation_matrices(preds_dict, y_true, model_names):
    """
    Takes a dict of model_name -> prediction_array, plus y_true.
    Returns two DataFrames: prediction correlation and error correlation.

    Handles BiLSTM's NaN offset by masking out any sample where
    any model has NaN. This drops ~11 samples out of ~1.6M so it
    doesn't affect anything.
    """
    # only keep models that were actually loaded
    names = [m for m in model_names if m in preds_dict]
    if len(names) < 2:
        return pd.DataFrame(), pd.DataFrame(), names

    # tail-align arrays (same as thesis_analysis_v2_fixed.py does)
    min_len = min(len(preds_dict[m]) for m in names)
    min_len = min(min_len, len(y_true))
    preds = {m: preds_dict[m][-min_len:] for m in names}
    yt = y_true[-min_len:]

    # NaN mask -- bilstm has NaN at the start due to lookback offset
    valid = np.ones(min_len, dtype=bool)
    for m in names:
        valid &= np.isfinite(preds[m])
    valid &= np.isfinite(yt)

    n_valid = int(valid.sum())
    n_dropped = min_len - n_valid
    if n_dropped > 0:
        print(f"    masked {n_dropped} NaN samples, {n_valid:,} remaining")

    # drop any model that ended up with no valid data
    # (shouldn't happen at 30/60/120min but just in case)
    keep = []
    for m in names:
        if np.std(preds[m][valid]) > 1e-12:
            keep.append(m)
        else:
            print(f"    WARNING: {m} has zero variance, excluding")
    names = keep

    if len(names) < 2:
        return pd.DataFrame(), pd.DataFrame(), names

    # apply the mask
    p = {m: preds[m][valid] for m in names}
    y = yt[valid]

    # --- prediction correlation ---
    pred_mat = np.column_stack([p[m] for m in names])
    pred_corr = np.corrcoef(pred_mat, rowvar=False)

    # --- error correlation (the important one) ---
    # error = y_true - prediction, i.e. the residual
    errs = {m: y - p[m] for m in names}
    err_mat = np.column_stack([errs[m] for m in names])
    err_corr = np.corrcoef(err_mat, rowvar=False)

    labels = [SHORT_NAMES.get(m, m) for m in names]
    pred_df = pd.DataFrame(pred_corr, index=labels, columns=labels)
    err_df = pd.DataFrame(err_corr, index=labels, columns=labels)

    return pred_df, err_df, names


def print_matrix(title, df):
    """Print a correlation matrix nicely."""
    if df.empty:
        print(f"\n  {title}: (empty)")
        return
    print(f"\n  {title}")
    print("  " + "-" * (10 + 10 * len(df.columns)))
    header = "          " + "  ".join(f"{c:>8s}" for c in df.columns)
    print(f"  {header}")
    for idx, row in df.iterrows():
        vals = "  ".join(f"{v:8.4f}" for v in row)
        print(f"  {idx:>8s}  {vals}")


# ── main logic ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprint1-dir", required=True,
                        help="dir containing sprint1_Main.py")
    parser.add_argument("--data-dir", required=True,
                        help="dir with test.parquet")
    parser.add_argument("--ckpt-dir", required=True,
                        help="checkpoint dir with 10min/ 30min/ etc subfolders")
    parser.add_argument("--output-dir", default="./thesis_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    load_sprint1(args.sprint1_dir)

    # load test data once -- shared across all horizons
    test_path = find_file(args.data_dir, "test")
    print(f"Loading {test_path} ...")
    test_raw = load_df(test_path)
    test_raw.sort_values(["container_id", "time_stamp"], inplace=True)
    test_raw.reset_index(drop=True, inplace=True)
    print(f"  {len(test_raw):,} rows\n")

    csv_rows = []  # collect all pairwise results for the output CSV

    for hz, hz_steps in HORIZONS.items():
        print("=" * 60)
        print(f"  {hz} (horizon_steps={hz_steps})")
        print("=" * 60)

        # reconstruct y_true
        test_fe = build_test_features(test_raw, hz_steps)
        y_true = test_fe["cpu_target"].values.astype(np.float64)
        print(f"  y_true length: {len(y_true):,}")

        # load saved predictions for each base model
        loaded = {}
        for model in MODEL_NAMES:
            arr = load_pred(args.ckpt_dir, hz, model)
            if arr is not None:
                loaded[model] = arr
                print(f"  loaded pred_{model}.npy ({len(arr):,} samples)")
            else:
                print(f"  pred_{model}.npy not found -- skipping")

        if len(loaded) < 2:
            print(f"  only {len(loaded)} model(s) available, need >= 2, skipping\n")
            continue

        # compute both matrices
        available = [m for m in MODEL_NAMES if m in loaded]
        pred_corr, err_corr, used_models = compute_correlation_matrices(
            loaded, y_true, available)

        print_matrix(f"PREDICTION correlation ({hz})", pred_corr)
        print_matrix(f"ERROR correlation ({hz})", err_corr)

        if err_corr.empty:
            print()
            continue

        # save pairwise values to CSV
        labels = [SHORT_NAMES.get(m, m) for m in used_models]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                csv_rows.append({
                    "Horizon": hz,
                    "Model_A": labels[i],
                    "Model_B": labels[j],
                    "Pred_Corr": round(float(pred_corr.iloc[i, j]), 6),
                    "Error_Corr": round(float(err_corr.iloc[i, j]), 6),
                })

        # --- summary diagnostics ---

        # tree-tree redundancy check
        tree_names = [m for m in ["xgboost", "lightgbm", "extratrees"]
                      if m in used_models]
        if len(tree_names) >= 2:
            tree_err_vals = []
            tree_pred_vals = []
            for i in range(len(tree_names)):
                for j in range(i + 1, len(tree_names)):
                    li = SHORT_NAMES[tree_names[i]]
                    lj = SHORT_NAMES[tree_names[j]]
                    tree_err_vals.append(float(err_corr.loc[li, lj]))
                    tree_pred_vals.append(float(pred_corr.loc[li, lj]))

            mean_tree_err = np.mean(tree_err_vals)
            mean_tree_pred = np.mean(tree_pred_vals)
            print(f"\n  Tree-tree avg prediction corr: {mean_tree_pred:.4f}")
            print(f"  Tree-tree avg error corr:      {mean_tree_err:.4f}")

            if mean_tree_pred > 0.95:
                print("  --> tree models produce near-identical outputs")
            if mean_tree_err > 0.90:
                print("  --> high error redundancy, minimal variance reduction")

        # tree-vs-bilstm diversity check
        if "bilstm" in used_models and len(tree_names) >= 1:
            bilstm_err_vals = []
            for tm in tree_names:
                li = SHORT_NAMES[tm]
                bilstm_err_vals.append(float(err_corr.loc[li, "BiLSTM"]))
            mean_bilstm_err = np.mean(bilstm_err_vals)
            print(f"  Tree-vs-BiLSTM avg error corr: {mean_bilstm_err:.4f}")

            if len(tree_names) >= 2 and mean_bilstm_err < mean_tree_err - 0.03:
                print("  --> BiLSTM brings more diversity than any tree pair")
            elif len(tree_names) >= 2 and mean_bilstm_err >= mean_tree_err:
                print("  --> BiLSTM error correlation is similar to tree-tree")
                print("       (weight increase likely due to relative accuracy,")
                print("        not diversity)")

        print()

    # write CSV
    if csv_rows:
        out_df = pd.DataFrame(csv_rows)
        csv_path = os.path.join(args.output_dir, "base_model_correlations.csv")
        out_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        print()
        print(out_df.to_string(index=False))
    else:
        print("No results to save.")

    print("\nDone.")


if __name__ == "__main__":
    main()