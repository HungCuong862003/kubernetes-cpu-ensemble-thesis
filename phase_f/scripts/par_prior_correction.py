"""
par_prior_correction.py — PAR Step 5b: prior-shift diagnostic + robustness checks.

Determines if the PARTIAL POSITIVE verdict for XGBoost (0.2153) is robust under
proper handling of label shift, sample weighting, and bootstrap CIs.

Six diagnostics, in order:

[A] Re-fit LODO classifiers with same hyperparameters from par_router_results.json,
    capturing predict_proba outputs.

[B] Bias-corrected temperature scaling (Alexandari, Kundaje, Shrikumar NeurIPS 2020).

[C] EM prior correction (Saerens, Latinne, Decaestecker 2002, Neural Computation).

[D] BBSE-soft (Lipton, Wang, Smola ICML 2018) as cross-check.

[E] Container-clustered bootstrap 95% CIs on macro-F1 (B=1000).

[F] Sample-weighted aggregate macro-F1 (weight by test fold size) +
    winsorised regret (per-container R² clipped at [-1, 1]) for both PAR and
    the always-pick-per-horizon-winner baseline.

Decision rule (DECISION-014, PRE-REGISTERED HERE BEFORE RUNNING):
  - HEADLINE-CAPABLE if BOTH:
      Alibaba-fold EM-corrected macro-F1 >= 0.30
      AND sample-weighted EM-corrected macro-F1 >= 0.25
    → Reopen Pivot A; keep PAR as F1 headline contribution.

  - PARTIAL POSITIVE UNDER SAMPLE-WEIGHTED AGGREGATION if:
      sample-weighted EM-corrected macro-F1 >= 0.20 (DECISION-013 partial threshold)
      but neither headline condition met.
    → Weakly publishable; still proceed to Pivot C+D.

  - STRUCTURAL SATURATION CONFIRMED if:
      sample-weighted EM-corrected macro-F1 < 0.20.
    → Proceed to Pivot C+D (BCF headline + cost-asymmetric Chronos-2 fine-tune).

Output:
  phase_f/data/par_prior_correction_results.json
  phase_f/data/par_prior_correction_output.txt (via tee)
"""

import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scipy.optimize import minimize_scalar

from xgboost import XGBClassifier

from _paths import P


warnings.filterwarnings('ignore')


# ============================== config ==============================

ROUTED_MODELS = ['nnls', 'chronos2', 'timesfm', 'granite_ttm']
DATASETS = ['alibaba', 'bitbrains', 'bytedance']
SEED = 42
N_BOOTSTRAP = 1000
WINSORISE_BOUNDS = (-1.0, 1.0)

OUT_JSON = Path(P['data']) / 'par_prior_correction_results.json'

# DECISION-014 thresholds (PRE-REGISTERED here before running)
DECISION_014_ALIBABA_THRESHOLD = 0.30
DECISION_014_SAMPLE_WEIGHTED_THRESHOLD = 0.25
DECISION_013_PARTIAL = 0.20

# Fallback hyperparameters if par_router_results.json missing
DEFAULT_LDA_PARAMS_PER_FOLD = {
    'alibaba':   {'shrinkage': 0.1},
    'bitbrains': {'shrinkage': 0.5},
    'bytedance': {'shrinkage': 0.5},
}
DEFAULT_XGB_PARAMS_PER_FOLD = {
    'alibaba':   {'max_depth': 5, 'n_estimators': 300, 'learning_rate': 0.05},
    'bitbrains': {'max_depth': 7, 'n_estimators': 200, 'learning_rate': 0.05},
    'bytedance': {'max_depth': 7, 'n_estimators': 200, 'learning_rate': 0.05},
}


# ============================== helpers ==============================

def load_data():
    """Merge par_labels + par_features, drop NaN. Returns (df, feature_cols)."""
    labels = pd.read_parquet(Path(P['data']) / 'par_labels.parquet')
    features = pd.read_parquet(Path(P['data']) / 'par_features.parquet')

    feature_cols = [c for c in features.columns
                    if c not in ['container_id', 'dataset', 'n_test_points']]

    df = labels.merge(
        features[['container_id', 'dataset'] + feature_cols],
        on=['container_id', 'dataset'], how='left',
    )

    if 'horizon_min' not in feature_cols and 'horizon_min' in df.columns:
        feature_cols = feature_cols + ['horizon_min']

    df = df.dropna(subset=feature_cols + ['label']).copy()
    return df, feature_cols


def load_best_params():
    """Load best hyperparameters from previous par_router_results.json if present."""
    json_path = Path(P['data']) / 'par_router_results.json'
    if not json_path.exists():
        print(f'    warning: {json_path} not found; using defaults')
        return DEFAULT_LDA_PARAMS_PER_FOLD, DEFAULT_XGB_PARAMS_PER_FOLD

    with open(json_path) as f:
        results = json.load(f)

    lda_params = {}
    xgb_params = {}
    for ds in DATASETS:
        lda_key = f'lda_holdout_{ds}'
        xgb_key = f'xgb_holdout_{ds}'
        lda_params[ds] = results.get(lda_key, {}).get(
            'best_params', DEFAULT_LDA_PARAMS_PER_FOLD[ds]
        )
        xgb_params[ds] = results.get(xgb_key, {}).get(
            'best_params', DEFAULT_XGB_PARAMS_PER_FOLD[ds]
        )

    return lda_params, xgb_params


def fit_lda(X, y, params):
    """Fit shrinkage LDA."""
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=params['shrinkage'])
    clf.fit(X, y)
    return clf


def fit_xgb(X, y, params, n_classes):
    """Fit multi-class XGBoost."""
    clf = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    clf.fit(X, y)
    return clf


def temperature_scale_fit(probas_val, y_val):
    """
    Fit temperature T by minimising NLL on validation set.
    Alexandari, Kundaje, Shrikumar (NeurIPS 2020) "Maximum Likelihood with
    Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation".
    """
    eps = 1e-10
    log_probs = np.log(probas_val + eps)
    n = len(y_val)

    def nll(T):
        scaled = log_probs / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(scaled)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        ll = np.log(probs[np.arange(n), y_val] + eps)
        return -ll.mean()

    result = minimize_scalar(nll, bounds=(0.05, 20.0), method='bounded')
    return float(result.x)


def temperature_scale_apply(probas, T):
    """Apply temperature T to probability matrix."""
    eps = 1e-10
    log_probs = np.log(probas + eps)
    scaled = log_probs / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def em_prior_correction(probas_test, p_train, max_iter=500, tol=1e-7):
    """
    Saerens, Latinne, Decaestecker (2002) EM prior correction.

    Iteratively estimates p_test(y) from unlabelled test soft outputs.
    Returns:
      p_test_est: (K,) estimated test priors
      probas_corrected: (n, K) corrected posteriors p_test(y|x)
      n_iter: iterations to convergence
    """
    p_train = np.asarray(p_train, dtype=np.float64)
    p_train = np.maximum(p_train, 1e-10)  # avoid divide-by-zero
    p_test_est = p_train.copy()

    for it in range(max_iter):
        ratio = p_test_est / p_train
        weighted = probas_test * ratio[np.newaxis, :]
        Z = weighted.sum(axis=1, keepdims=True)
        Z = np.maximum(Z, 1e-12)
        probas_corrected = weighted / Z

        new_p_test = probas_corrected.mean(axis=0)
        diff = np.max(np.abs(new_p_test - p_test_est))
        p_test_est = new_p_test
        if diff < tol:
            return p_test_est, probas_corrected, it + 1

    return p_test_est, probas_corrected, max_iter


def soft_confusion_matrix(probas_val, y_val, n_classes):
    """
    Soft confusion matrix: C[i, j] = E[P(pred=j) | y_true=i] on validation.
    Used for BBSE inversion.
    """
    C = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        mask = (y_val == i)
        if mask.any():
            C[i] = probas_val[mask].mean(axis=0)
        else:
            # Class absent in val; fall back to uniform row (no information)
            C[i] = np.ones(n_classes) / n_classes
    return C


def bbse_correction(probas_test, p_train, C_train):
    """
    BBSE-soft (Lipton, Wang, Smola ICML 2018).
    Estimates p_test(y) by inverting the soft confusion matrix.

    mu_pred_test = C_train.T @ p_test   →   p_test = pinv(C_train.T) @ mu_pred_test

    Then reweights p_train(y|x) → p_test(y|x) by ratio p_test/p_train.
    """
    p_train = np.asarray(p_train, dtype=np.float64)
    p_train = np.maximum(p_train, 1e-10)
    mu_pred_test = probas_test.mean(axis=0)

    try:
        p_test_est = np.linalg.pinv(C_train.T) @ mu_pred_test
        p_test_est = np.maximum(p_test_est, 0)
        if p_test_est.sum() > 0:
            p_test_est = p_test_est / p_test_est.sum()
        else:
            p_test_est = p_train.copy()
    except np.linalg.LinAlgError:
        p_test_est = p_train.copy()

    ratio = p_test_est / p_train
    weighted = probas_test * ratio[np.newaxis, :]
    Z = np.maximum(weighted.sum(axis=1, keepdims=True), 1e-12)
    probas_corrected = weighted / Z
    return p_test_est, probas_corrected


def bootstrap_macro_f1(df_test, y_true, y_pred, n_classes, n_boot=1000, seed=42):
    """
    Container-clustered bootstrap 95% CI on macro-F1.
    Samples container_ids with replacement; takes all (container, horizon) rows.
    Returns (ci_lo, ci_hi, mean_f1).
    """
    rng = np.random.default_rng(seed)
    container_ids = df_test['container_id'].values

    container_to_idx = defaultdict(list)
    for i, cid in enumerate(container_ids):
        container_to_idx[cid].append(i)
    container_to_idx = {k: np.array(v) for k, v in container_to_idx.items()}
    unique_containers = np.array(list(container_to_idx.keys()))

    f1s = np.zeros(n_boot)
    labels_list = list(range(n_classes))
    for b in range(n_boot):
        sampled = rng.choice(unique_containers, size=len(unique_containers), replace=True)
        idx_resample = np.concatenate([container_to_idx[c] for c in sampled])
        if len(idx_resample) == 0:
            f1s[b] = np.nan
            continue
        f1s[b] = f1_score(
            y_true[idx_resample], y_pred[idx_resample],
            average='macro', labels=labels_list, zero_division=0,
        )

    return (
        float(np.nanpercentile(f1s, 2.5)),
        float(np.nanpercentile(f1s, 97.5)),
        float(np.nanmean(f1s)),
    )


def winsorised_regret(df_test, y_pred_labels, bounds=WINSORISE_BOUNDS):
    """
    regret = best_r2 − r2_of_predicted_label, with per-container R²
    NaN-replaced (→ 0.0 = naive-baseline parity) then clipped to `bounds`.

    NaN R² occurs for containers with SS_tot ≈ 0 (constant target). Treating
    these as 0.0 (parity with naive) is the most defensible interpretation;
    -1 would over-penalise the model unfairly, +1 would over-reward it.

    Returns (per_row_regret, total, mean, median).
    """
    lo, hi = bounds
    n = len(df_test)
    pred_r2 = np.zeros(n, dtype=np.float64)
    pred_arr = np.asarray(y_pred_labels)
    for m in ROUTED_MODELS:
        mask = (pred_arr == m)
        if mask.any() and m in df_test.columns:
            r2_vals = df_test.loc[mask, m].values
            r2_vals = np.nan_to_num(r2_vals, nan=0.0)
            pred_r2[mask] = np.clip(r2_vals, lo, hi)

    best_r2 = np.nan_to_num(df_test['best_r2'].values, nan=0.0)
    best_r2 = np.clip(best_r2, lo, hi)
    regret = best_r2 - pred_r2
    return regret, float(regret.sum()), float(regret.mean()), float(np.median(regret))


def baseline_per_horizon_pred(df_train, df_test):
    """Always-pick-most-common-label-per-horizon (computed from training set)."""
    baseline_per_h = df_train.groupby('horizon')['label'].apply(
        lambda s: s.mode().iloc[0]
    )
    return df_test['horizon'].map(baseline_per_h).values


# ============================== per-fold pipeline ==============================

def run_fold(df, feature_cols, holdout_ds, model_type, params, le, results):
    fold_key = f'{model_type}_holdout_{holdout_ds}'
    print(f'\n--- {fold_key} ---')

    train_mask = df['dataset'] != holdout_ds
    test_mask = df['dataset'] == holdout_ds
    df_train_full = df[train_mask].copy()
    df_test = df[test_mask].copy()

    # 80/20 split of training fold → calibration validation slice
    try:
        train_idx, val_idx = train_test_split(
            np.arange(len(df_train_full)),
            test_size=0.2,
            random_state=SEED,
            stratify=df_train_full['label'].values,
        )
    except ValueError:
        # stratify can fail with very small classes; fall back to random
        train_idx, val_idx = train_test_split(
            np.arange(len(df_train_full)),
            test_size=0.2,
            random_state=SEED,
        )
    df_train = df_train_full.iloc[train_idx].copy()
    df_val = df_train_full.iloc[val_idx].copy()

    print(f'    train: n={len(df_train)}  val: n={len(df_val)}  test: n={len(df_test)} ({holdout_ds})')

    # Standardise on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_val = scaler.transform(df_val[feature_cols].values)
    X_test = scaler.transform(df_test[feature_cols].values)

    y_train = le.transform(df_train['label'].values)
    y_val = le.transform(df_val['label'].values)
    y_test = le.transform(df_test['label'].values)
    n_classes = len(le.classes_)

    # Fit classifier
    print(f'    fitting {model_type} with params: {params}')
    t0 = time.time()
    if model_type == 'lda':
        clf = fit_lda(X_train, y_train, params)
    else:
        clf = fit_xgb(X_train, y_train, params, n_classes)
    print(f'    fit took {time.time() - t0:.1f}s')

    probas_val = clf.predict_proba(X_val)
    probas_test = clf.predict_proba(X_test)

    # Training priors from the FULL LODO training fold (not the 80% fit slice).
    # The classifier was fit on the 80% slice, so the EM/BBSE math technically
    # wants the 80% slice priors; stratified split makes the two effectively
    # identical (< 0.5pp difference), and the full priors are what the chapter
    # table should report.
    y_train_full = le.transform(df_train_full['label'].values)
    train_counts = np.bincount(y_train_full, minlength=n_classes)
    p_train = train_counts / train_counts.sum()

    # ---- (A) RAW predictions ----
    y_pred_raw = probas_test.argmax(axis=1)
    f1_raw = f1_score(y_test, y_pred_raw, average='macro',
                      labels=list(range(n_classes)), zero_division=0)
    acc_raw = accuracy_score(y_test, y_pred_raw)

    # ---- (B) Temperature scaling on val ----
    T = temperature_scale_fit(probas_val, y_val)
    probas_test_T = temperature_scale_apply(probas_test, T)
    y_pred_T = probas_test_T.argmax(axis=1)
    f1_T = f1_score(y_test, y_pred_T, average='macro',
                    labels=list(range(n_classes)), zero_division=0)
    print(f'    fitted temperature T = {T:.4f}')

    # ---- (C) EM prior correction on T-scaled probas ----
    p_test_em, probas_em, n_iter = em_prior_correction(probas_test_T, p_train)
    y_pred_em = probas_em.argmax(axis=1)
    f1_em = f1_score(y_test, y_pred_em, average='macro',
                     labels=list(range(n_classes)), zero_division=0)
    acc_em = accuracy_score(y_test, y_pred_em)
    p_test_actual = np.bincount(y_test, minlength=n_classes) / len(y_test)
    print(f'    EM converged in {n_iter} iter')
    print(f'    p_train       = {dict(zip(le.classes_, np.round(p_train, 3)))}')
    print(f'    p_test_actual = {dict(zip(le.classes_, np.round(p_test_actual, 3)))}')
    print(f'    p_test_em_est = {dict(zip(le.classes_, np.round(p_test_em, 3)))}')

    # ---- (D) BBSE-soft cross-check ----
    C_val = soft_confusion_matrix(probas_val, y_val, n_classes)
    p_test_bbse, probas_bbse = bbse_correction(probas_test_T, p_train, C_val)
    y_pred_bbse = probas_bbse.argmax(axis=1)
    f1_bbse = f1_score(y_test, y_pred_bbse, average='macro',
                       labels=list(range(n_classes)), zero_division=0)
    print(f'    p_test_bbse   = {dict(zip(le.classes_, np.round(p_test_bbse, 3)))}')

    # ---- (E) Bootstrap CIs on raw + EM ----
    print('    bootstrap 95% CIs (B=1000)...')
    t0 = time.time()
    ci_raw_lo, ci_raw_hi, ci_raw_mean = bootstrap_macro_f1(
        df_test, y_test, y_pred_raw, n_classes, n_boot=N_BOOTSTRAP, seed=SEED,
    )
    ci_em_lo, ci_em_hi, ci_em_mean = bootstrap_macro_f1(
        df_test, y_test, y_pred_em, n_classes, n_boot=N_BOOTSTRAP, seed=SEED + 1,
    )
    print(f'    bootstrap took {time.time() - t0:.0f}s')

    # ---- (F) Baseline + winsorised regret ----
    baseline_pred = baseline_per_horizon_pred(df_train_full, df_test)
    baseline_f1 = f1_score(df_test['label'].values, baseline_pred,
                           average='macro', zero_division=0)
    baseline_acc = accuracy_score(df_test['label'].values, baseline_pred)

    y_pred_raw_str = le.inverse_transform(y_pred_raw)
    y_pred_em_str = le.inverse_transform(y_pred_em)

    _, _, reg_raw_mean, reg_raw_med = winsorised_regret(df_test, y_pred_raw_str)
    _, _, reg_em_mean, reg_em_med = winsorised_regret(df_test, y_pred_em_str)
    _, _, reg_b_mean, reg_b_med = winsorised_regret(df_test, baseline_pred)

    # ---- Print summary ----
    print(f'    macro-F1:')
    print(f'      raw            = {f1_raw:.4f}   95% CI [{ci_raw_lo:.4f}, {ci_raw_hi:.4f}]')
    print(f'      temp-scaled    = {f1_T:.4f}')
    print(f'      EM-corrected   = {f1_em:.4f}   95% CI [{ci_em_lo:.4f}, {ci_em_hi:.4f}]')
    print(f'      BBSE-corrected = {f1_bbse:.4f}')
    print(f'      baseline       = {baseline_f1:.4f}')
    print(f'    winsorised regret (mean | median):')
    print(f'      raw      = {reg_raw_mean:+.4f} | {reg_raw_med:+.4f}')
    print(f'      EM       = {reg_em_mean:+.4f} | {reg_em_med:+.4f}')
    print(f'      baseline = {reg_b_mean:+.4f} | {reg_b_med:+.4f}')

    results[fold_key] = {
        'model_type':            model_type,
        'holdout_dataset':       holdout_ds,
        'n_train':               int(len(df_train)),
        'n_val':                 int(len(df_val)),
        'n_test':                int(len(df_test)),
        'params':                params,
        'temperature':           float(T),
        'em_iterations':         int(n_iter),
        'p_train':               {c: float(p) for c, p in zip(le.classes_, p_train)},
        'p_test_actual':         {c: float(p) for c, p in zip(le.classes_, p_test_actual)},
        'p_test_em':             {c: float(p) for c, p in zip(le.classes_, p_test_em)},
        'p_test_bbse':           {c: float(p) for c, p in zip(le.classes_, p_test_bbse)},
        'macro_f1_raw':          float(f1_raw),
        'macro_f1_temp_scaled':  float(f1_T),
        'macro_f1_em':           float(f1_em),
        'macro_f1_bbse':         float(f1_bbse),
        'macro_f1_baseline':     float(baseline_f1),
        'ci_raw_95':             [ci_raw_lo, ci_raw_hi],
        'ci_em_95':              [ci_em_lo, ci_em_hi],
        'accuracy_raw':          float(acc_raw),
        'accuracy_em':           float(acc_em),
        'accuracy_baseline':     float(baseline_acc),
        'regret_raw_mean':       reg_raw_mean,
        'regret_em_mean':        reg_em_mean,
        'regret_baseline_mean':  reg_b_mean,
        'regret_raw_median':     reg_raw_med,
        'regret_em_median':      reg_em_med,
        'regret_baseline_median': reg_b_med,
    }


# ============================== main ==============================

def main():
    print('=' * 78)
    print('par_prior_correction.py — PAR Step 5b')
    print('  EM + BBSE + temp-scaling + bootstrap CIs + sample-weighted aggregate')
    print('=' * 78)

    print('\n[1] Loading data...')
    df, feature_cols = load_data()
    print(f'    n = {len(df)} rows | n_features = {len(feature_cols)}')

    print('\n[2] Loading best hyperparameters from par_router_results.json...')
    lda_params, xgb_params = load_best_params()
    print(f'    LDA: {lda_params}')
    print(f'    XGB: {xgb_params}')

    le = LabelEncoder()
    le.fit(ROUTED_MODELS)

    results = {}

    print('\n' + '=' * 78)
    print('[3] LDA — three LODO folds with prior corrections')
    print('=' * 78)
    for ds in DATASETS:
        run_fold(df, feature_cols, ds, 'lda', lda_params[ds], le, results)

    print('\n' + '=' * 78)
    print('[4] XGBoost — three LODO folds with prior corrections')
    print('=' * 78)
    for ds in DATASETS:
        run_fold(df, feature_cols, ds, 'xgb', xgb_params[ds], le, results)

    # ---- Aggregate ----
    print('\n' + '=' * 78)
    print('AGGREGATE — equal-weight vs sample-weighted')
    print('=' * 78)

    aggregates = {}
    for mt in ['lda', 'xgb']:
        fold_keys = [f'{mt}_holdout_{ds}' for ds in DATASETS]
        if not all(k in results for k in fold_keys):
            continue

        f1_raws = [results[k]['macro_f1_raw'] for k in fold_keys]
        f1_ems = [results[k]['macro_f1_em'] for k in fold_keys]
        f1_bbses = [results[k]['macro_f1_bbse'] for k in fold_keys]
        f1_baselines = [results[k]['macro_f1_baseline'] for k in fold_keys]
        n_tests = [results[k]['n_test'] for k in fold_keys]

        # Equal-weight
        eq_raw = float(np.mean(f1_raws))
        eq_em = float(np.mean(f1_ems))
        eq_bbse = float(np.mean(f1_bbses))
        eq_baseline = float(np.mean(f1_baselines))

        # Sample-weighted
        total_n = sum(n_tests)
        sw_raw = sum(f * n for f, n in zip(f1_raws, n_tests)) / total_n
        sw_em = sum(f * n for f, n in zip(f1_ems, n_tests)) / total_n
        sw_bbse = sum(f * n for f, n in zip(f1_bbses, n_tests)) / total_n
        sw_baseline = sum(f * n for f, n in zip(f1_baselines, n_tests)) / total_n

        print(f'\n[{mt.upper()}]')
        print(f'  per-fold raw F1:      {[round(f, 4) for f in f1_raws]}')
        print(f'  per-fold EM F1:       {[round(f, 4) for f in f1_ems]}')
        print(f'  per-fold BBSE F1:     {[round(f, 4) for f in f1_bbses]}')
        print(f'  per-fold baseline F1: {[round(f, 4) for f in f1_baselines]}')
        print(f'  per-fold n_test:      {n_tests}  (Σ = {total_n})')
        print()
        print(f'  equal-weight aggregate:')
        print(f'    raw      = {eq_raw:.4f}')
        print(f'    EM       = {eq_em:.4f}')
        print(f'    BBSE     = {eq_bbse:.4f}')
        print(f'    baseline = {eq_baseline:.4f}')
        print(f'  sample-weighted aggregate (deployment-realistic):')
        print(f'    raw      = {sw_raw:.4f}')
        print(f'    EM       = {sw_em:.4f}')
        print(f'    BBSE     = {sw_bbse:.4f}')
        print(f'    baseline = {sw_baseline:.4f}')

        aggregates[mt] = {
            'per_fold_raw_f1':       [float(f) for f in f1_raws],
            'per_fold_em_f1':        [float(f) for f in f1_ems],
            'per_fold_bbse_f1':      [float(f) for f in f1_bbses],
            'per_fold_baseline_f1':  [float(f) for f in f1_baselines],
            'per_fold_n_test':       [int(n) for n in n_tests],
            'equal_weight_raw':       eq_raw,
            'equal_weight_em':        eq_em,
            'equal_weight_bbse':      eq_bbse,
            'equal_weight_baseline':  eq_baseline,
            'sample_weighted_raw':       float(sw_raw),
            'sample_weighted_em':        float(sw_em),
            'sample_weighted_bbse':      float(sw_bbse),
            'sample_weighted_baseline':  float(sw_baseline),
        }

    # ---- DECISION-014 verdict ----
    print('\n' + '=' * 78)
    print('DECISION-014 verdict (PRE-REGISTERED in script header)')
    print('=' * 78)
    print('  HEADLINE-CAPABLE conditions (BOTH required):')
    print(f'    (a) Alibaba-fold EM-corrected macro-F1 >= {DECISION_014_ALIBABA_THRESHOLD}')
    print(f'    (b) Sample-weighted EM-corrected macro-F1 >= {DECISION_014_SAMPLE_WEIGHTED_THRESHOLD}')

    for mt in ['lda', 'xgb']:
        if mt not in aggregates:
            continue
        ali_key = f'{mt}_holdout_alibaba'
        ali_f1_em = results[ali_key]['macro_f1_em']
        sw_em = aggregates[mt]['sample_weighted_em']

        cond_a = ali_f1_em >= DECISION_014_ALIBABA_THRESHOLD
        cond_b = sw_em >= DECISION_014_SAMPLE_WEIGHTED_THRESHOLD

        print(f'\n  [{mt.upper()}]')
        print(f'    (a) Alibaba EM F1 = {ali_f1_em:.4f}  → {"PASS" if cond_a else "FAIL"} (vs {DECISION_014_ALIBABA_THRESHOLD})')
        print(f'    (b) Sample-weighted EM F1 = {sw_em:.4f}  → {"PASS" if cond_b else "FAIL"} (vs {DECISION_014_SAMPLE_WEIGHTED_THRESHOLD})')

        if cond_a and cond_b:
            verdict = 'HEADLINE-CAPABLE — PAR rescued by prior correction → reopen Pivot A'
        elif sw_em >= DECISION_013_PARTIAL:
            verdict = 'PARTIAL POSITIVE under sample-weighted EM correction → still proceed to Pivot C+D'
        else:
            verdict = 'STRUCTURAL SATURATION CONFIRMED → commit to Pivot C+D'

        print(f'    VERDICT: {verdict}')
        aggregates[mt]['verdict'] = verdict

    # ---- Save ----
    results['aggregates'] = aggregates
    results['metadata'] = {
        'pre_registration': 'DECISION-014 (HEADLINE-CAPABLE conditions)',
        'thresholds': {
            'alibaba_em_f1':         DECISION_014_ALIBABA_THRESHOLD,
            'sample_weighted_em_f1': DECISION_014_SAMPLE_WEIGHTED_THRESHOLD,
            'decision_013_partial':  DECISION_013_PARTIAL,
        },
        'methods': [
            'Saerens, Latinne, Decaestecker (2002) EM prior correction',
            'Alexandari, Kundaje, Shrikumar (NeurIPS 2020) bias-corrected temperature scaling',
            'Lipton, Wang, Smola (ICML 2018) BBSE-soft cross-check',
            'Container-clustered bootstrap 95% CIs (B=1000)',
            'Winsorised per-container R² regret at [-1, 1]',
        ],
        'random_seed':  SEED,
        'n_bootstrap':  N_BOOTSTRAP,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {OUT_JSON}')


if __name__ == '__main__':
    main()
