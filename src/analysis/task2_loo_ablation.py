"""
task2_loo_ablation.py
─────────────────────
Leave-one-out ablation of the NNLS ensemble.

For each horizon, measures:
  - Full ensemble R² and MAE
  - R²/MAE when each base model is removed (leave-one-out)
  - R²/MAE of each base model standalone

Uses saved OOF .npy files from the sprint1 pipeline checkpoint,
plus parquet data files to reconstruct ground truth targets.

Input:
  - train.parquet, test.parquet (from DATA_DIR)
  - pred_oof_xgb.npy, pred_oof_lgb.npy, pred_oof_et.npy,
    pred_oof_xgb_test.npy, pred_oof_lgb_test.npy, pred_oof_et_test.npy,
    pred_bilstm_oof.npy, pred_bilstm_test.npy, pred_naive.npy
    (from CKPT_DIR/{horizon}/)

Output:
  - loo_ablation.csv

Run on Colab:
    # adjust paths in the CONFIG section below, then:
    !python task2_loo_ablation.py
"""

import pandas as pd
import numpy as np
import os
from scipy.optimize import nnls as scipy_nnls
from sklearn.metrics import r2_score, mean_absolute_error


# ── CONFIG (adjust these paths for your Drive) ────────────────────────────

# where train.parquet and test.parquet live
DATA_DIR = "/content/drive/MyDrive/workspace_backup/thesis"

# where the sprint1 checkpoint folders live (10min/, 30min/, 60min/, 120min/)
# each subfolder should contain pred_oof_xgb.npy, pred_naive.npy, etc.
CKPT_DIR = "/content/drive/MyDrive/sprint1_results"

# output file
OUTPUT_CSV = "loo_ablation.csv"

# horizon name -> number of 5-minute timesteps to shift
HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}


# ── helper: NNLS fit + predict (matches sprint1_Main.py NNLSEnsemble) ─────

def nnls_fit(base_preds, y_true):
    """Fit NNLS weights. Returns normalized weight vector."""
    w, _ = scipy_nnls(base_preds, y_true)
    s = w.sum()
    if s > 1e-8:
        return w / s
    else:
        # all weights zero — fall back to uniform
        return np.ones(base_preds.shape[1]) / base_preds.shape[1]


def load_npy(ckpt_dir, hz, name):
    """Load a .npy file from the checkpoint directory."""
    p = os.path.join(ckpt_dir, hz, f"pred_{name}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


# ── step 1: verify all inputs exist ───────────────────────────────────────

print("checking input files ...")

missing = []
for f in ["train.parquet", "test.parquet"]:
    p = os.path.join(DATA_DIR, f)
    if not os.path.exists(p):
        missing.append(p)

# tree OOF files must exist for all 4 horizons
for hz in HORIZONS:
    for name in ["oof_xgb", "oof_lgb", "oof_et",
                  "oof_xgb_test", "oof_lgb_test", "oof_et_test",
                  "naive"]:
        p = os.path.join(CKPT_DIR, hz, f"pred_{name}.npy")
        if not os.path.exists(p):
            missing.append(p)

# bilstm OOF files — check but don't abort if missing
# (the dynamic detection in step 3 handles missing files gracefully)
bilstm_warnings = []
for hz in HORIZONS:
    for name in ["bilstm_oof", "bilstm_test"]:
        p = os.path.join(CKPT_DIR, hz, f"pred_{name}.npy")
        if not os.path.exists(p):
            bilstm_warnings.append(p)
if bilstm_warnings:
    print(f"  WARNING: {len(bilstm_warnings)} BiLSTM file(s) missing "
          f"(will use 3-model ensemble for those horizons):")
    for p in bilstm_warnings:
        print(f"    {p}")

if missing:
    print(f"\n*** {len(missing)} file(s) missing: ***")
    for p in missing:
        print(f"  {p}")
    print("\nfix paths in CONFIG section and re-run.")
    raise SystemExit(1)

print("  all files found.")


# ── step 2: reconstruct ground truth targets from parquet ──────────────────

print("\nloading parquet files ...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
print(f"  train: {len(train_df):,} rows, test: {len(test_df):,} rows")


def reconstruct_targets(df, horizon_steps):
    """
    Minimal target reconstruction matching sprint1_Main.py:
      cpu_target   = shift(-horizon_steps) of cpu_util_percent per container
      naive_cpu    = cpu_util_percent (current value)
      cpu_residual = cpu_target - naive_cpu
    Then dropna on the 6 required columns (same as pipeline).
    """
    out = df.copy()
    out = out.sort_values(["container_id", "time_stamp"]).reset_index(drop=True)
    g = out.groupby("container_id")

    out["cpu_target"]   = g["cpu_util_percent"].shift(-horizon_steps)
    out["naive_cpu"]    = out["cpu_util_percent"]
    out["cpu_residual"] = out["cpu_target"] - out["naive_cpu"]

    out["mem_target"]   = g["mem_util_percent"].shift(-horizon_steps)
    out["naive_mem"]    = out["mem_util_percent"]
    out["mem_residual"] = out["mem_target"] - out["naive_mem"]

    required = ["cpu_target", "cpu_residual", "naive_cpu",
                "mem_target", "mem_residual", "naive_mem"]
    out = out.dropna(subset=required).reset_index(drop=True)

    y_cres   = out["cpu_residual"].values.astype(np.float32)
    y_cpu    = out["cpu_target"].values.astype(np.float32)
    naive    = out["naive_cpu"].values.astype(np.float32)
    return y_cres, y_cpu, naive


# ── step 3: run LOO ablation per horizon ───────────────────────────────────

results = []

for hz, steps in HORIZONS.items():
    print(f"\n{'='*60}")
    print(f"  {hz} (horizon_steps={steps})")
    print(f"{'='*60}")

    # reconstruct ground truth
    y_cres_tr, _, _           = reconstruct_targets(train_df, steps)
    _,         y_cpu_te, _    = reconstruct_targets(test_df,  steps)

    # load saved naive predictions on test (absolute)
    naive_cpu = load_npy(CKPT_DIR, hz, "naive")

    # load tree OOF arrays (residual predictions on train)
    xgb_oof = load_npy(CKPT_DIR, hz, "oof_xgb")
    lgb_oof = load_npy(CKPT_DIR, hz, "oof_lgb")
    et_oof  = load_npy(CKPT_DIR, hz, "oof_et")

    # load tree test arrays (OOF-averaged residual predictions on test)
    xgb_te = load_npy(CKPT_DIR, hz, "oof_xgb_test")
    lgb_te = load_npy(CKPT_DIR, hz, "oof_lgb_test")
    et_te  = load_npy(CKPT_DIR, hz, "oof_et_test")

    # sanity check: OOF train array length must match reconstructed y_cres_tr
    if len(xgb_oof) != len(y_cres_tr):
        print(f"  *** LENGTH MISMATCH: xgb_oof={len(xgb_oof)}, "
              f"y_cres_tr={len(y_cres_tr)} ***")
        print(f"  skipping {hz} — check DATA_DIR parquet vs CKPT_DIR .npy files")
        continue
    print(f"  train length OK: {len(y_cres_tr):,}")

    # sanity check: test array lengths
    if len(xgb_te) != len(y_cpu_te):
        print(f"  *** TEST LENGTH MISMATCH: xgb_te={len(xgb_te)}, "
              f"y_cpu_te={len(y_cpu_te)} ***")
        print(f"  skipping {hz}")
        continue
    if len(naive_cpu) != len(y_cpu_te):
        print(f"  *** NAIVE LENGTH MISMATCH: naive={len(naive_cpu)}, "
              f"y_cpu_te={len(y_cpu_te)} ***")
        print(f"  skipping {hz}")
        continue
    print(f"  test  length OK: {len(y_cpu_te):,}")

    # --- set up model list for this horizon ---
    # try loading BiLSTM OOF at every horizon; decide dynamically
    # whether to include it based on alignment (same logic as pipeline)
    bilstm_oof = load_npy(CKPT_DIR, hz, "bilstm_oof")
    bilstm_te  = load_npy(CKPT_DIR, hz, "bilstm_test")

    # check how many samples have valid predictions from ALL 4 models
    MIN_ALIGNED = 100  # same threshold as pipeline (sprint1_Main.py ~line 2223)

    # at 10min, pipeline's BiLSTM OOF failed during the actual run and the
    # published ensemble uses 3 tree models only. force 3 models here so the
    # ablation table explains the thesis's reported numbers.
    if hz == "10min":
        has_bilstm = False
        print(f"  10min: using 3 models (matching thesis reported ensemble)")
    elif bilstm_oof is not None and bilstm_te is not None:
        if len(bilstm_oof) != len(xgb_oof):
            print(f"  BiLSTM OOF length mismatch: bilstm={len(bilstm_oof)}, "
                  f"xgb={len(xgb_oof)} — using 3 models")
            has_bilstm = False
        else:
            aligned_4 = (~np.isnan(xgb_oof) & ~np.isnan(lgb_oof) &
                          ~np.isnan(et_oof)  & ~np.isnan(bilstm_oof))
            n_aligned = int(aligned_4.sum())
            has_bilstm = n_aligned > MIN_ALIGNED
            print(f"  BiLSTM 4-model alignment: {n_aligned:,} samples "
                  f"({'OK' if has_bilstm else 'TOO FEW — using 3 models'})")
    else:
        has_bilstm = False
        print(f"  BiLSTM OOF files missing — using 3 models")

    if has_bilstm:
        model_names = ["XGBoost", "LightGBM", "ExtraTrees", "BiLSTM"]
        oof_list    = [xgb_oof, lgb_oof, et_oof, bilstm_oof]

        # bilstm_te may be shorter than n_test (lookback offset)
        n_test   = len(y_cpu_te)
        n_bilstm = len(bilstm_te)
        n_pad    = n_test - n_bilstm
        if n_pad < 0:
            print(f"  WARNING: bilstm_te ({n_bilstm}) > n_test ({n_test}), clamping")
            bilstm_te = bilstm_te[:n_test]
            n_bilstm = n_test
            n_pad = 0
        print(f"  BiLSTM covers {n_bilstm}/{n_test} test samples (offset={n_pad})")

        # for fair comparison, evaluate on the common subset where all models
        # have predictions (last n_bilstm test samples)
        eval_slice   = slice(n_pad, None)
        y_cpu_eval   = y_cpu_te[eval_slice]
        naive_eval   = naive_cpu[eval_slice]

        # build aligned test matrix (common subset only)
        test_cols_4  = np.column_stack([
            xgb_te[eval_slice], lgb_te[eval_slice],
            et_te[eval_slice],  bilstm_te
        ])
        test_cols_3_map = {
            "XGBoost":    np.column_stack([lgb_te[eval_slice], et_te[eval_slice], bilstm_te]),
            "LightGBM":   np.column_stack([xgb_te[eval_slice], et_te[eval_slice], bilstm_te]),
            "ExtraTrees":  np.column_stack([xgb_te[eval_slice], lgb_te[eval_slice], bilstm_te]),
            "BiLSTM":     np.column_stack([xgb_te[eval_slice], lgb_te[eval_slice], et_te[eval_slice]]),
        }

    else:
        model_names = ["XGBoost", "LightGBM", "ExtraTrees"]
        oof_list    = [xgb_oof, lgb_oof, et_oof]

        eval_slice  = slice(None)  # use full test set
        y_cpu_eval  = y_cpu_te
        naive_eval  = naive_cpu

        test_cols_4 = None  # not used (no BiLSTM)
        test_cols_3_map = None  # LOO handles inline for 3-model case
        test_matrix_full = np.column_stack([xgb_te, lgb_te, et_te])

    print(f"  evaluating on {len(y_cpu_eval):,} test samples")

    # --- build OOF train matrix with valid mask ---
    oof_stack = np.column_stack(oof_list)
    valid = np.all(~np.isnan(oof_stack), axis=1)
    n_valid = int(valid.sum())
    print(f"  OOF valid samples: {n_valid:,} / {len(y_cres_tr):,}")

    oof_valid   = oof_stack[valid]
    y_cres_valid = y_cres_tr[valid]

    # --- full ensemble ---
    w_full = nnls_fit(oof_valid, y_cres_valid)
    w_dict = {n: round(float(w), 4) for n, w in zip(model_names, w_full)}
    print(f"  NNLS weights: {w_dict}")

    if has_bilstm:
        pred_full = naive_eval + test_cols_4 @ w_full
    else:
        pred_full = naive_eval + test_matrix_full @ w_full

    r2_full  = r2_score(y_cpu_eval, pred_full)
    mae_full = mean_absolute_error(y_cpu_eval, pred_full)
    print(f"  full ensemble: R2={r2_full:.6f}  MAE={mae_full:.4f}")

    results.append({
        "Horizon": hz, "Config": "Full ensemble",
        "R2": round(r2_full, 6), "MAE": round(mae_full, 4),
        "Delta_R2_pp": 0.0, "Delta_MAE": 0.0,
    })

    # --- leave-one-out: remove each model ---
    n_models = len(model_names)

    for drop_idx, drop_name in enumerate(model_names):
        keep = [j for j in range(n_models) if j != drop_idx]

        # refit NNLS on training OOF without dropped model
        oof_loo   = oof_valid[:, keep]
        w_loo     = nnls_fit(oof_loo, y_cres_valid)

        # predict on test without dropped model
        if has_bilstm:
            te_loo = test_cols_3_map[drop_name]
        else:
            te_keep = [xgb_te, lgb_te, et_te]
            te_loo  = np.column_stack([te_keep[j][eval_slice] for j in keep])

        pred_loo = naive_eval + te_loo @ w_loo
        r2_loo   = r2_score(y_cpu_eval, pred_loo)
        mae_loo  = mean_absolute_error(y_cpu_eval, pred_loo)

        delta_r2  = (r2_loo - r2_full) * 100  # in percentage points
        delta_mae = mae_loo - mae_full

        kept_names = [model_names[j] for j in keep]
        w_loo_dict = {n: round(float(w), 4) for n, w in zip(kept_names, w_loo)}
        print(f"  without {drop_name:12s}: R2={r2_loo:.6f} "
              f"(Δ={delta_r2:+.4f}pp)  MAE={mae_loo:.4f}  "
              f"w={w_loo_dict}")

        results.append({
            "Horizon": hz, "Config": f"Without {drop_name}",
            "R2": round(r2_loo, 6), "MAE": round(mae_loo, 4),
            "Delta_R2_pp": round(delta_r2, 4),
            "Delta_MAE": round(delta_mae, 4),
        })

    # --- each model standalone ---
    for solo_idx, solo_name in enumerate(model_names):
        # single-model "NNLS" is just that model's prediction directly
        if has_bilstm:
            if solo_name == "BiLSTM":
                solo_te = bilstm_te
            else:
                solo_te = [xgb_te, lgb_te, et_te][solo_idx][eval_slice]
        else:
            solo_te = [xgb_te, lgb_te, et_te][solo_idx][eval_slice]

        pred_solo = naive_eval + solo_te
        r2_solo   = r2_score(y_cpu_eval, pred_solo)
        mae_solo  = mean_absolute_error(y_cpu_eval, pred_solo)

        delta_r2  = (r2_solo - r2_full) * 100
        delta_mae = mae_solo - mae_full

        print(f"  {solo_name:12s} alone: R2={r2_solo:.6f} "
              f"(Δ={delta_r2:+.4f}pp)  MAE={mae_solo:.4f}")

        results.append({
            "Horizon": hz, "Config": f"{solo_name} alone",
            "R2": round(r2_solo, 6), "MAE": round(mae_solo, 4),
            "Delta_R2_pp": round(delta_r2, 4),
            "Delta_MAE": round(delta_mae, 4),
        })


# ── step 4: save results ──────────────────────────────────────────────────

df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n{'='*60}")
print(f"saved {OUTPUT_CSV} ({len(df_out)} rows)")
print(f"{'='*60}")
print(df_out.to_string(index=False))
print("\ndone.")
