"""
task1_meta_learner_bench.py — Compare meta-learner strategies on saved OOF predictions.

Empirically tests the forecast combination puzzle (Claeskens et al., 2016):
with error correlations > 0.92, does the choice of meta-learner matter?
Compares NNLS (current), simple average, Ridge, Bates-Granger inverse-MSE,
and best-individual selection. Also runs NNLS bootstrap stability analysis
and computes the OOF covariance matrix condition number.

Inputs:
    train.parquet, test.parquet                (DATA_DIR)
    pred_oof_xgb.npy, pred_oof_lgb.npy, ...   (CKPT_DIR/{horizon}/)
    pred_oof_xgb_test.npy, ...                 (CKPT_DIR/{horizon}/)
    pred_bilstm_oof.npy, pred_bilstm_test.npy  (CKPT_DIR/{horizon}/, may be absent)
    pred_naive.npy                             (CKPT_DIR/{horizon}/)

Outputs:
    meta_learner_comparison.csv                (OUTPUT_DIR)
    bootstrap_weight_stats.csv                 (OUTPUT_DIR)
    combination_diagnostics.csv                (OUTPUT_DIR)

Run on Colab:
    # adjust paths in CONFIG section, then:
    !python task1_meta_learner_bench.py
"""

import pandas as pd
import numpy as np
import os
from scipy.optimize import nnls as scipy_nnls
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import RidgeCV


# ── CONFIG (adjust these paths for your Drive) ──────────────────────

DATA_DIR = "/content/drive/MyDrive/workspace_backup/thesis"
CKPT_DIR = "/content/drive/MyDrive/sprint1_results"
OUTPUT_DIR = "."

HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}

# bootstrap settings
N_BOOTSTRAP = 200
BOOTSTRAP_SEED = 42

# minimum aligned OOF samples to include BiLSTM (same as sprint1_Main.py)
MIN_ALIGNED = 100


# ── HELPERS ─────────────────────────────────────────────────────────

def load_npy(ckpt_dir, hz, name):
    """Load a .npy file from the checkpoint directory."""
    p = os.path.join(ckpt_dir, hz, f"pred_{name}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


def nnls_fit(base_preds, y_true):
    """Fit NNLS weights on residual-space predictions.
    Returns normalized weight vector (sums to 1)."""
    w, _ = scipy_nnls(base_preds, y_true)
    s = w.sum()
    if s > 1e-8:
        return w / s
    else:
        return np.ones(base_preds.shape[1]) / base_preds.shape[1]


def reconstruct_targets(df, horizon_steps):
    """Minimal target reconstruction matching sprint1_Main.py.
    Returns (y_cpu_residual, y_cpu_target, naive_cpu)."""
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

    y_cres = out["cpu_residual"].values.astype(np.float32)
    y_cpu  = out["cpu_target"].values.astype(np.float32)
    naive  = out["naive_cpu"].values.astype(np.float32)
    return y_cres, y_cpu, naive


def fmt_weights(names, weights):
    """Format weight dict as compact string for CSV."""
    parts = []
    for n, w in zip(names, weights):
        parts.append(f"{n}={w:.3f}")
    return " ".join(parts)


# ── MAIN ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── step 1: verify inputs ───────────────────────────────────────

    print("checking input files ...")

    missing = []
    for f in ["train.parquet", "test.parquet"]:
        p = os.path.join(DATA_DIR, f)
        if not os.path.exists(p):
            missing.append(p)

    for hz in HORIZONS:
        for name in ["oof_xgb", "oof_lgb", "oof_et",
                      "oof_xgb_test", "oof_lgb_test", "oof_et_test",
                      "naive"]:
            p = os.path.join(CKPT_DIR, hz, f"pred_{name}.npy")
            if not os.path.exists(p):
                missing.append(p)

    if missing:
        print(f"\n*** {len(missing)} required file(s) missing: ***")
        for p in missing:
            print(f"  {p}")
        print("\nfix paths in CONFIG section and re-run.")
        raise SystemExit(1)

    # check bilstm files (optional)
    for hz in HORIZONS:
        for name in ["bilstm_oof", "bilstm_test"]:
            p = os.path.join(CKPT_DIR, hz, f"pred_{name}.npy")
            if not os.path.exists(p):
                print(f"  note: {hz}/{name} missing (will use 3-model ensemble)")

    print("  all required files found.\n")

    # ── step 2: load parquets ───────────────────────────────────────

    print("loading parquet files ...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    print(f"  train: {len(train_df):,} rows, test: {len(test_df):,} rows")

    # ── step 3: run meta-learner comparison per horizon ─────────────

    results = []
    diag_rows = []
    boot_rows = []

    for hz, steps in HORIZONS.items():
        print(f"\n{'='*60}")
        print(f"  {hz} (horizon_steps={steps})")
        print(f"{'='*60}")

        # reconstruct ground truth
        y_cres_tr, _, _         = reconstruct_targets(train_df, steps)
        _,         y_cpu_te, _  = reconstruct_targets(test_df, steps)

        # load saved arrays
        naive_cpu = load_npy(CKPT_DIR, hz, "naive")
        xgb_oof   = load_npy(CKPT_DIR, hz, "oof_xgb")
        lgb_oof   = load_npy(CKPT_DIR, hz, "oof_lgb")
        et_oof    = load_npy(CKPT_DIR, hz, "oof_et")
        xgb_te    = load_npy(CKPT_DIR, hz, "oof_xgb_test")
        lgb_te    = load_npy(CKPT_DIR, hz, "oof_lgb_test")
        et_te     = load_npy(CKPT_DIR, hz, "oof_et_test")

        # sanity checks
        if len(xgb_oof) != len(y_cres_tr):
            print(f"  *** LENGTH MISMATCH: xgb_oof={len(xgb_oof)}, "
                  f"y_cres_tr={len(y_cres_tr)} ***")
            print(f"  skipping {hz}")
            continue
        if len(xgb_te) != len(y_cpu_te):
            print(f"  *** TEST LENGTH MISMATCH ***")
            print(f"  skipping {hz}")
            continue
        print(f"  train: {len(y_cres_tr):,}  test: {len(y_cpu_te):,}")

        # ── detect BiLSTM dynamically (NOT hardcoded) ───────────────
        bilstm_oof = load_npy(CKPT_DIR, hz, "bilstm_oof")
        bilstm_te  = load_npy(CKPT_DIR, hz, "bilstm_test")

        has_bilstm = False
        if bilstm_oof is not None and bilstm_te is not None:
            if len(bilstm_oof) == len(xgb_oof):
                aligned_4 = (~np.isnan(xgb_oof) & ~np.isnan(lgb_oof) &
                              ~np.isnan(et_oof) & ~np.isnan(bilstm_oof))
                n_aligned = int(aligned_4.sum())
                has_bilstm = n_aligned > MIN_ALIGNED
                print(f"  BiLSTM alignment: {n_aligned:,} samples "
                      f"({'OK — using 4 models' if has_bilstm else 'TOO FEW — 3 models'})")
            else:
                print(f"  BiLSTM OOF length mismatch — using 3 models")
        else:
            print(f"  BiLSTM files missing — using 3 models")

        # ── build model arrays ──────────────────────────────────────
        if has_bilstm:
            model_names = ["XGB", "LGB", "ET", "BiLSTM"]
            oof_list = [xgb_oof, lgb_oof, et_oof, bilstm_oof]

            # BiLSTM test may be shorter (lookback offset)
            n_test   = len(y_cpu_te)
            n_bilstm = len(bilstm_te)
            n_pad    = n_test - n_bilstm
            if n_pad < 0:
                bilstm_te = bilstm_te[:n_test]
                n_pad = 0
            print(f"  BiLSTM covers {n_bilstm}/{n_test} test samples (offset={n_pad})")

            # evaluate on common subset
            eval_slice  = slice(n_pad, None)
            y_cpu_eval  = y_cpu_te[eval_slice]
            naive_eval  = naive_cpu[eval_slice]
            te_matrix   = np.column_stack([
                xgb_te[eval_slice], lgb_te[eval_slice],
                et_te[eval_slice], bilstm_te
            ])
        else:
            model_names = ["XGB", "LGB", "ET"]
            oof_list = [xgb_oof, lgb_oof, et_oof]

            eval_slice  = slice(None)
            y_cpu_eval  = y_cpu_te
            naive_eval  = naive_cpu
            te_matrix   = np.column_stack([xgb_te, lgb_te, et_te])

        K = len(model_names)
        print(f"  K={K} models, evaluating on {len(y_cpu_eval):,} test samples")

        # ── build OOF train matrix with valid mask ──────────────────
        oof_stack = np.column_stack(oof_list)
        valid = np.all(~np.isnan(oof_stack), axis=1)
        n_valid = int(valid.sum())
        print(f"  OOF valid: {n_valid:,} / {len(y_cres_tr):,}")

        oof_valid   = oof_stack[valid]
        y_cres_valid = y_cres_tr[valid]

        # ── diagnostics: condition number and error correlations ────
        cov_mat = np.cov(oof_valid.T)
        cond_num = np.linalg.cond(cov_mat)

        # pairwise error correlations (residual-space errors)
        err_corrs = []
        for i in range(K):
            err_i = y_cres_valid - oof_valid[:, i]
            for j in range(i+1, K):
                err_j = y_cres_valid - oof_valid[:, j]
                r = np.corrcoef(err_i, err_j)[0, 1]
                err_corrs.append(r)
        mean_err_corr = np.mean(err_corrs) if err_corrs else 0.0

        print(f"  cond(Σ) = {cond_num:.1f}")
        print(f"  mean error correlation = {mean_err_corr:.4f}")

        # ── Method 1: NNLS (current pipeline) ──────────────────────
        w_nnls = nnls_fit(oof_valid, y_cres_valid)
        pred_nnls = naive_eval + te_matrix @ w_nnls
        r2_nnls   = r2_score(y_cpu_eval, pred_nnls)
        mae_nnls  = mean_absolute_error(y_cpu_eval, pred_nnls)

        print(f"\n  Method 1 — NNLS:")
        print(f"    weights: {fmt_weights(model_names, w_nnls)}")
        print(f"    R2={r2_nnls:.6f}  MAE={mae_nnls:.4f}")

        results.append({
            "Horizon": hz, "Method": "NNLS", "N_models": K,
            "R2": round(r2_nnls, 6), "MAE": round(mae_nnls, 4),
            "Weights": fmt_weights(model_names, w_nnls),
        })

        # ── Method 2: Simple Average (1/K) ─────────────────────────
        w_avg = np.ones(K) / K
        pred_avg = naive_eval + te_matrix @ w_avg
        r2_avg   = r2_score(y_cpu_eval, pred_avg)
        mae_avg  = mean_absolute_error(y_cpu_eval, pred_avg)

        print(f"\n  Method 2 — Simple Average:")
        print(f"    weights: {fmt_weights(model_names, w_avg)}")
        print(f"    R2={r2_avg:.6f}  MAE={mae_avg:.4f}")

        results.append({
            "Horizon": hz, "Method": "Simple Average", "N_models": K,
            "R2": round(r2_avg, 6), "MAE": round(mae_avg, 4),
            "Weights": fmt_weights(model_names, w_avg),
        })

        # ── Method 3: Ridge (sklearn RidgeCV, no intercept) ────────
        # fit_intercept=False for fair comparison with NNLS (which has
        # no intercept). Both operate in residual space where mean ≈ 0.
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=False)
        ridge.fit(oof_valid, y_cres_valid)
        w_ridge = ridge.coef_
        # normalize for interpretability (Ridge doesn't constrain to sum=1)
        w_ridge_norm = w_ridge / w_ridge.sum() if abs(w_ridge.sum()) > 1e-8 else w_ridge

        pred_ridge = naive_eval + ridge.predict(te_matrix)
        r2_ridge   = r2_score(y_cpu_eval, pred_ridge)
        mae_ridge  = mean_absolute_error(y_cpu_eval, pred_ridge)

        has_neg = any(w < -0.01 for w in w_ridge)
        print(f"\n  Method 3 — Ridge (alpha={ridge.alpha_}, no intercept):")
        print(f"    raw coefs:  {fmt_weights(model_names, w_ridge)}")
        print(f"    normalized: {fmt_weights(model_names, w_ridge_norm)}")
        if has_neg:
            print(f"    WARNING: negative coefficient(s) — Ridge 'bets against' a model")
        print(f"    R2={r2_ridge:.6f}  MAE={mae_ridge:.4f}")

        results.append({
            "Horizon": hz, "Method": "Ridge", "N_models": K,
            "R2": round(r2_ridge, 6), "MAE": round(mae_ridge, 4),
            "Weights": fmt_weights(model_names, w_ridge_norm),
        })

        # ── Method 4: Bates-Granger (inverse MSE) ──────────────────
        mse_per_model = np.mean((oof_valid - y_cres_valid[:, None]) ** 2, axis=0)
        inv_mse = 1.0 / np.maximum(mse_per_model, 1e-12)
        w_bg = inv_mse / inv_mse.sum()

        pred_bg = naive_eval + te_matrix @ w_bg
        r2_bg   = r2_score(y_cpu_eval, pred_bg)
        mae_bg  = mean_absolute_error(y_cpu_eval, pred_bg)

        print(f"\n  Method 4 — Bates-Granger (inverse MSE):")
        print(f"    per-model MSE: {['%.4f' % m for m in mse_per_model]}")
        print(f"    weights: {fmt_weights(model_names, w_bg)}")
        print(f"    R2={r2_bg:.6f}  MAE={mae_bg:.4f}")

        results.append({
            "Horizon": hz, "Method": "Bates-Granger", "N_models": K,
            "R2": round(r2_bg, 6), "MAE": round(mae_bg, 4),
            "Weights": fmt_weights(model_names, w_bg),
        })

        # ── Method 5: Best Individual (oracle from OOF) ─────────────
        oof_maes = []
        for i in range(K):
            pred_i_val = oof_valid[:, i]
            mae_i = np.mean(np.abs(y_cres_valid - pred_i_val))
            oof_maes.append(mae_i)

        best_idx = np.argmin(oof_maes)
        best_name = model_names[best_idx]

        # use only that model on test
        pred_best = naive_eval + te_matrix[:, best_idx]
        r2_best   = r2_score(y_cpu_eval, pred_best)
        mae_best  = mean_absolute_error(y_cpu_eval, pred_best)

        w_best = np.zeros(K)
        w_best[best_idx] = 1.0

        print(f"\n  Method 5 — Best Individual ({best_name}, selected by OOF MAE):")
        print(f"    OOF MAEs: {[f'{model_names[i]}={oof_maes[i]:.4f}' for i in range(K)]}")
        print(f"    R2={r2_best:.6f}  MAE={mae_best:.4f}")

        results.append({
            "Horizon": hz, "Method": f"Best Individual ({best_name})", "N_models": 1,
            "R2": round(r2_best, 6), "MAE": round(mae_best, 4),
            "Weights": fmt_weights(model_names, w_best),
        })

        # ── diagnostics: R² spread across methods ──────────────────
        all_r2s = [r2_nnls, r2_avg, r2_ridge, r2_bg, r2_best]
        r2_spread = (max(all_r2s) - min(all_r2s)) * 100  # in pp

        all_maes = [mae_nnls, mae_avg, mae_ridge, mae_bg, mae_best]
        mae_spread = max(all_maes) - min(all_maes)

        print(f"\n  --- Diagnostics ---")
        print(f"  R² spread across 5 methods: {r2_spread:.4f} pp")
        print(f"  MAE spread: {mae_spread:.4f}")
        print(f"  Condition number: {cond_num:.1f}")
        print(f"  Mean error correlation: {mean_err_corr:.4f}")

        diag_rows.append({
            "Horizon": hz,
            "N_models": K,
            "Cond_Number": round(cond_num, 1),
            "Mean_Err_Corr": round(mean_err_corr, 4),
            "R2_Spread_pp": round(r2_spread, 4),
            "MAE_Spread": round(mae_spread, 4),
            "Best_Individual": best_name,
        })

        # ── bootstrap weight stability ──────────────────────────────
        print(f"\n  Bootstrap ({N_BOOTSTRAP} resamples) ...")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot_weights = np.zeros((N_BOOTSTRAP, K))

        for b in range(N_BOOTSTRAP):
            idx = rng.choice(n_valid, size=n_valid, replace=True)
            w_b = nnls_fit(oof_valid[idx], y_cres_valid[idx])
            boot_weights[b] = w_b

        for i, name in enumerate(model_names):
            w_col = boot_weights[:, i]
            frac_zero = np.mean(w_col < 1e-6)
            boot_rows.append({
                "Horizon": hz,
                "Model": name,
                "Mean_W": round(float(np.mean(w_col)), 4),
                "Std_W": round(float(np.std(w_col)), 4),
                "Frac_Zero": round(float(frac_zero), 3),
                "Min_W": round(float(np.min(w_col)), 4),
                "Max_W": round(float(np.max(w_col)), 4),
            })
            print(f"    {name:8s}: mean={np.mean(w_col):.3f} ±{np.std(w_col):.3f}  "
                  f"zero in {frac_zero*100:.0f}% of resamples  "
                  f"range=[{np.min(w_col):.3f}, {np.max(w_col):.3f}]")

    # ── step 4: save results ────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_results = pd.DataFrame(results)
    out1 = os.path.join(OUTPUT_DIR, "meta_learner_comparison.csv")
    df_results.to_csv(out1, index=False)
    print(f"\n{'='*60}")
    print(f"saved {out1} ({len(df_results)} rows)")
    print(df_results.to_string(index=False))

    df_diag = pd.DataFrame(diag_rows)
    out2 = os.path.join(OUTPUT_DIR, "combination_diagnostics.csv")
    df_diag.to_csv(out2, index=False)
    print(f"\nsaved {out2} ({len(df_diag)} rows)")
    print(df_diag.to_string(index=False))

    df_boot = pd.DataFrame(boot_rows)
    out3 = os.path.join(OUTPUT_DIR, "bootstrap_weight_stats.csv")
    df_boot.to_csv(out3, index=False)
    print(f"\nsaved {out3} ({len(df_boot)} rows)")
    print(df_boot.to_string(index=False))

    # ── step 5: summary ────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("COMBINATION PUZZLE SUMMARY")
    print(f"{'='*60}")

    for _, row in df_diag.iterrows():
        hz = row["Horizon"]
        spread = row["R2_Spread_pp"]
        cond = row["Cond_Number"]
        corr = row["Mean_Err_Corr"]
        verdict = "CONFIRMED" if spread < 0.5 else "NOT confirmed"
        print(f"  {hz}: spread={spread:.4f}pp  cond={cond:.0f}  "
              f"ρ_err={corr:.3f}  → puzzle {verdict}")

    print(f"\n  If all spreads < 0.5pp: meta-learner choice is irrelevant.")
    print(f"  Cite: Claeskens et al. (2016), Int. J. Forecasting 32(3)")
    print(f"\ndone.")
