"""
par_router.py — PAR Step 5. The PAR classifier test.

LODO 3-fold CV (Leave-One-Dataset-Out):
  Fold 1: train on {alibaba, bytedance}, test on bitbrains
  Fold 2: train on {alibaba, bitbrains},  test on bytedance
  Fold 3: train on {bitbrains, bytedance}, test on alibaba

Inner 3-fold StratifiedKFold (by label) for hyperparameter selection
inside each outer fold (nested CV).

Two classifiers:
  - Shrinkage LDA  (linear baseline, regularised LDA)
  - XGBoost        (non-linear primary)

Features (30 total):
  - catch22_1 .. catch22_22  (22 canonical TSC features)
  - WPE, SampEn, LZC, DFA    (predictability/complexity)
  - cv, std, mean            (basic stats)
  - horizon_min              (routing context)

Primary metric:    macro-F1 (DECISION-013 thresholds: ≥0.55 HEADLINE,
                   0.20–0.55 PARTIAL, <0.20 STRUCTURAL SATURATION DEEPENS)
Secondary metric:  regret = best_r2 − r2_of_predicted_label per test row
                   compared vs always-pick-per-horizon-winner baseline

Output:
  phase_f/data/par_router_results.json       (per-fold + aggregate metrics)
  phase_f/data/par_router_predictions.csv    (per-row predictions for inspection)

Run from phase_f/scripts/.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from xgboost import XGBClassifier

from _paths import P


warnings.filterwarnings('ignore')


# -------------------------------- config --------------------------------

ROUTED_MODELS = ['nnls', 'chronos2', 'timesfm', 'granite_ttm']
DATASETS      = ['alibaba', 'bitbrains', 'bytedance']
SEED          = 42

OUT_JSON     = Path(P['data']) / 'par_router_results.json'
OUT_PRED_CSV = Path(P['data']) / 'par_router_predictions.csv'

DECISION_THRESHOLD_HEADLINE = 0.55
DECISION_THRESHOLD_PARTIAL  = 0.20

# Hyperparameter grids — kept compact to keep nested-CV runtime tractable
LDA_GRID = [
    {'shrinkage': 'auto'},
    {'shrinkage': 0.1},
    {'shrinkage': 0.3},
    {'shrinkage': 0.5},
]

XGB_GRID = [
    {'max_depth': 3, 'n_estimators': 150, 'learning_rate': 0.10},
    {'max_depth': 5, 'n_estimators': 150, 'learning_rate': 0.10},
    {'max_depth': 5, 'n_estimators': 300, 'learning_rate': 0.05},
    {'max_depth': 7, 'n_estimators': 200, 'learning_rate': 0.05},
]


# -------------------------------- helpers --------------------------------

def load_data():
    """Merge labels + features, drop NaN. Returns (df, feature_cols)."""
    labels = pd.read_parquet(Path(P['data']) / 'par_labels.parquet')
    features = pd.read_parquet(Path(P['data']) / 'par_features.parquet')

    # feature cols = everything in features except identifiers and n_test_points
    feature_cols = [c for c in features.columns
                    if c not in ['container_id', 'dataset', 'n_test_points']]

    df = labels.merge(
        features[['container_id', 'dataset'] + feature_cols],
        on=['container_id', 'dataset'], how='left',
    )

    # add horizon_min as a feature (it's in labels, not features)
    if 'horizon_min' not in feature_cols and 'horizon_min' in df.columns:
        feature_cols = feature_cols + ['horizon_min']

    n_before = len(df)
    df = df.dropna(subset=feature_cols + ['label']).copy()
    print(f'    loaded: {n_before} rows, after dropna: {len(df)}')
    return df, feature_cols


def fit_lda(X, y, params):
    """Fit LDA with shrinkage."""
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=params['shrinkage'])
    clf.fit(X, y)
    return clf


def fit_xgb(X, y, params, n_classes):
    """Fit XGBoost multi-class classifier."""
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


def inner_cv_select(X_train, y_train, model_type, grid, n_classes, n_folds=3):
    """Nested-CV inner loop: pick hyperparameters maximising mean macro-F1."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    best_params = None
    best_f1 = -np.inf
    all_results = []

    for params in grid:
        fold_f1s = []
        for inner_tr, inner_val in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[inner_tr], y_train[inner_tr]
            X_val, y_val = X_train[inner_val], y_train[inner_val]
            try:
                if model_type == 'lda':
                    clf = fit_lda(X_tr, y_tr, params)
                else:
                    clf = fit_xgb(X_tr, y_tr, params, n_classes)
                y_pred = clf.predict(X_val)
                fold_f1s.append(f1_score(y_val, y_pred, average='macro', zero_division=0))
            except Exception:
                fold_f1s.append(0.0)

        mean_f1 = float(np.mean(fold_f1s))
        all_results.append({'params': params, 'inner_f1': mean_f1})
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params

    return best_params, best_f1, all_results


def baseline_per_horizon_pred(df_train, df_test):
    """Always-pick-most-common-label-per-horizon (from training set)."""
    baseline_per_h = df_train.groupby('horizon')['label'].apply(
        lambda s: s.mode().iloc[0]
    )
    return df_test['horizon'].map(baseline_per_h).values


def compute_regret(df_test, y_pred_labels):
    """
    Vectorised regret per test row = best_r2 − r2_of_predicted_label.
    Returns (per_row_regret_array, total, mean).
    """
    n = len(df_test)
    pred_r2 = np.zeros(n)
    pred_arr = np.asarray(y_pred_labels)
    for m in ROUTED_MODELS:
        mask = (pred_arr == m)
        if mask.any() and m in df_test.columns:
            pred_r2[mask] = df_test.loc[mask, m].values
    per_row = df_test['best_r2'].values - pred_r2
    return per_row, float(per_row.sum()), float(per_row.mean())


def decide_verdict(mean_f1):
    """Apply DECISION-013 thresholds."""
    if mean_f1 >= DECISION_THRESHOLD_HEADLINE:
        return f'HEADLINE: per-series routing works (mean macro-F1 = {mean_f1:.4f} ≥ {DECISION_THRESHOLD_HEADLINE})'
    elif mean_f1 >= DECISION_THRESHOLD_PARTIAL:
        return f'PARTIAL POSITIVE: per-series routing adds value ({DECISION_THRESHOLD_PARTIAL} ≤ {mean_f1:.4f} < {DECISION_THRESHOLD_HEADLINE})'
    else:
        return f'STRUCTURAL SATURATION DEEPENS: per-series routing fails (mean macro-F1 = {mean_f1:.4f} < {DECISION_THRESHOLD_PARTIAL})'


# -------------------------------- LODO fold --------------------------------

def run_lodo_fold(df, feature_cols, holdout_ds, model_type, grid, le, results):
    """Run one LODO outer fold with inner-CV hyperparameter selection."""
    label_key = f'{model_type}_holdout_{holdout_ds}'
    print(f'\n  --- {label_key} ---')

    train_mask = df['dataset'] != holdout_ds
    test_mask  = df['dataset'] == holdout_ds
    df_train = df[train_mask].copy()
    df_test  = df[test_mask].copy()
    print(f'    train: n={len(df_train)} ({df_train["dataset"].value_counts().to_dict()})')
    print(f'    test:  n={len(df_test)}  ({holdout_ds})')

    if len(df_train) < 100 or len(df_test) < 50:
        print('    too few rows; skipping')
        return None

    # Standardise on training only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_test  = scaler.transform(df_test[feature_cols].values)
    y_train = le.transform(df_train['label'].values)
    y_test  = le.transform(df_test['label'].values)
    n_classes = len(le.classes_)

    # Training label distribution
    train_dist = dict(zip(*np.unique(df_train['label'].values, return_counts=True)))
    test_dist  = dict(zip(*np.unique(df_test['label'].values, return_counts=True)))
    print(f'    train label dist: {train_dist}')
    print(f'    test  label dist: {test_dist}')

    # Inner CV
    print(f'    inner 3-fold CV ({len(grid)} hyperparameter candidates)...')
    t0 = time.time()
    best_params, inner_f1, inner_results = inner_cv_select(
        X_train, y_train, model_type, grid, n_classes, n_folds=3,
    )
    print(f'    best inner params (inner macro-F1={inner_f1:.4f}): {best_params}')
    print(f'    inner CV took {time.time() - t0:.0f}s')

    # Final fit on full training set
    print(f'    training final {model_type} on full training set...')
    t0 = time.time()
    if model_type == 'lda':
        clf = fit_lda(X_train, y_train, best_params)
    else:
        clf = fit_xgb(X_train, y_train, best_params, n_classes)
    print(f'    final fit took {time.time() - t0:.0f}s')

    # Predict and score
    y_pred = clf.predict(X_test)
    y_pred_str = le.inverse_transform(y_pred)

    macro_f1   = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    accuracy   = float(accuracy_score(y_test, y_pred))
    per_cls_f1 = f1_score(y_test, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    cm         = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

    # Regret
    _, total_regret, mean_regret = compute_regret(df_test, y_pred_str)

    # Baseline
    baseline_pred = baseline_per_horizon_pred(df_train, df_test)
    baseline_macro_f1 = float(f1_score(df_test['label'].values, baseline_pred,
                                       average='macro', zero_division=0))
    baseline_accuracy = float(accuracy_score(df_test['label'].values, baseline_pred))
    _, baseline_total_regret, baseline_mean_regret = compute_regret(df_test, baseline_pred)

    print(f'    macro-F1:    {macro_f1:.4f}   (baseline {baseline_macro_f1:.4f})')
    print(f'    accuracy:    {accuracy:.4f}   (baseline {baseline_accuracy:.4f})')
    print(f'    mean regret: {mean_regret:.4f}   (baseline {baseline_mean_regret:.4f})')
    print(f'    per-class F1: ' + ', '.join(
        [f'{c}={f:.3f}' for c, f in zip(le.classes_, per_cls_f1)]
    ))
    print(f'    confusion matrix (rows=true, cols=pred), classes={list(le.classes_)}:')
    for i, row in enumerate(cm):
        print(f'      {le.classes_[i]:<12}: {row.tolist()}')

    results[label_key] = {
        'model_type':           model_type,
        'holdout_dataset':      holdout_ds,
        'n_train':              int(len(df_train)),
        'n_test':               int(len(df_test)),
        'best_params':          best_params,
        'inner_cv_macro_f1':    float(inner_f1),
        'inner_cv_all_results': inner_results,
        'macro_f1':             macro_f1,
        'accuracy':             accuracy,
        'mean_regret':          mean_regret,
        'total_regret':         total_regret,
        'per_class_f1':         {c: float(f) for c, f in zip(le.classes_, per_cls_f1)},
        'baseline_macro_f1':    baseline_macro_f1,
        'baseline_accuracy':    baseline_accuracy,
        'baseline_mean_regret': baseline_mean_regret,
        'confusion_matrix':     cm.tolist(),
        'class_order':          list(le.classes_),
    }

    return df_test.assign(
        y_pred=y_pred_str,
        baseline_pred=baseline_pred,
        model_type=model_type,
        fold=f'holdout_{holdout_ds}',
    )


# -------------------------------- main --------------------------------

def main():
    print('=' * 78)
    print('par_router.py — PAR Step 5: classifier test (LODO 3-fold CV)')
    print('=' * 78)

    print('\n[1] Loading + merging labels and features...')
    df, feature_cols = load_data()
    print(f'    n features: {len(feature_cols)}')
    print(f'    sample features: {feature_cols[:8]}...')

    print('\n[2] Per-dataset label distribution:')
    ds_lbl = df.groupby(['dataset', 'label']).size().unstack(fill_value=0)
    for m in ROUTED_MODELS:
        if m not in ds_lbl.columns:
            ds_lbl[m] = 0
    ds_lbl = ds_lbl[ROUTED_MODELS]
    print(ds_lbl.to_string())
    print(f'\n    overall n = {len(df)}')

    le = LabelEncoder()
    le.fit(ROUTED_MODELS)

    results = {}
    all_preds = []

    # LDA folds
    print()
    print('=' * 78)
    print('[3] LODO 3-fold CV — Shrinkage LDA')
    print('=' * 78)
    for ds in DATASETS:
        preds = run_lodo_fold(df, feature_cols, ds, 'lda', LDA_GRID, le, results)
        if preds is not None:
            all_preds.append(preds)

    # XGBoost folds
    print()
    print('=' * 78)
    print('[4] LODO 3-fold CV — XGBoost')
    print('=' * 78)
    for ds in DATASETS:
        preds = run_lodo_fold(df, feature_cols, ds, 'xgb', XGB_GRID, le, results)
        if preds is not None:
            all_preds.append(preds)

    # Aggregate
    print()
    print('=' * 78)
    print('AGGREGATE — per model type')
    print('=' * 78)

    for mt in ['lda', 'xgb']:
        fold_keys = [f'{mt}_holdout_{ds}' for ds in DATASETS]
        if not all(k in results for k in fold_keys):
            continue
        f1s          = [results[k]['macro_f1'] for k in fold_keys]
        regrets      = [results[k]['mean_regret'] for k in fold_keys]
        baselines_f1 = [results[k]['baseline_macro_f1'] for k in fold_keys]
        baselines_r  = [results[k]['baseline_mean_regret'] for k in fold_keys]

        mean_f1 = float(np.mean(f1s))
        verdict = decide_verdict(mean_f1)

        print(f'\n[{mt.upper()}]')
        print(f'  per-fold macro-F1:    {[round(f, 4) for f in f1s]}  mean={mean_f1:.4f}')
        print(f'  per-fold baseline F1: {[round(f, 4) for f in baselines_f1]}  mean={np.mean(baselines_f1):.4f}')
        print(f'  per-fold mean regret: {[round(r, 4) for r in regrets]}  mean={np.mean(regrets):.4f}')
        print(f'  per-fold baseline reg: {[round(r, 4) for r in baselines_r]}  mean={np.mean(baselines_r):.4f}')
        print(f'  VERDICT: {verdict}')

        results[f'{mt}_aggregate'] = {
            'mean_macro_f1':           mean_f1,
            'mean_regret':             float(np.mean(regrets)),
            'mean_baseline_macro_f1':  float(np.mean(baselines_f1)),
            'mean_baseline_regret':    float(np.mean(baselines_r)),
            'per_fold_macro_f1':       [float(f) for f in f1s],
            'per_fold_regret':         [float(r) for r in regrets],
            'verdict':                 verdict,
        }

    # Metadata
    results['metadata'] = {
        'pre_registration':            'DECISION-013',
        'thresholds':                  {
            'headline':                DECISION_THRESHOLD_HEADLINE,
            'partial':                 DECISION_THRESHOLD_PARTIAL,
        },
        'cv':                          'LODO 3-fold outer + StratifiedKFold 3-fold inner',
        'classifiers':                 ['shrinkage_lda', 'xgboost'],
        'lda_grid':                    LDA_GRID,
        'xgb_grid':                    XGB_GRID,
        'n_features':                  len(feature_cols),
        'feature_cols':                feature_cols,
        'n_obs_total':                 int(len(df)),
        'n_classes':                   len(le.classes_),
        'class_order':                 list(le.classes_),
        'random_seed':                 SEED,
        'baseline':                    'always-pick-most-common-label-per-horizon (from training set)',
        'primary_metric':              'macro-F1',
        'secondary_metric':            'regret = best_r2 − r2_of_predicted_label per test row',
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: {OUT_JSON}')

    if all_preds:
        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_df.to_csv(OUT_PRED_CSV, index=False)
        print(f'Saved: {OUT_PRED_CSV}')

    # Final summary
    print()
    print('=' * 78)
    print('FINAL SUMMARY')
    print('=' * 78)
    for mt in ['lda', 'xgb']:
        key = f'{mt}_aggregate'
        if key in results:
            r = results[key]
            print(f'\n[{mt.upper()}] mean macro-F1 = {r["mean_macro_f1"]:.4f}')
            print(f'  per fold: {r["per_fold_macro_f1"]}')
            print(f'  VERDICT: {r["verdict"]}')


if __name__ == '__main__':
    main()
