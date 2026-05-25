"""
par_f2_series.py — per-series F2 partial-R² test.

PRE-REGISTRATION (DECISION-005, D5 close):
  Primary endpoint:  partial-R²(WPE | ACF@24h, horizon) at series scope
  Decision rule:
    ≥ 0.30  → F2 RESCUED at series scope (cell-level NULL revises)
    0.10 ≤ partial-R² < 0.30  → BORDERLINE
    < 0.10  → F2 NULL DEEPENS to series scope

Outcome variable:
  ml_benefit = NNLS_R² - naive_R² at (series, horizon) granularity
  (NNLS = NEW pool ensemble for Alibaba/ByteDance; global XGBoost for Bitbrains,
   matching par_per_series_r2.parquet v2 conventions)

Method:
  Reduced:  ml_benefit ~ ACF@24h + horizon_min          (controls only)
  Full:     ml_benefit ~ ACF@24h + horizon_min + WPE    (+ predictability feature)
  partial-R² = (R²_full − R²_reduced) / (1 − R²_reduced)

Cluster bootstrap CI:
  1000 resamples, clustered by container_id
  95% percentile CI
  (BCa not used; partial-R² distribution generally well-behaved at n>>12 unlike the
   cell-level F2 case where BCa was degenerate)

Sensitivities (not pre-registered):
  (a) WITH dataset dummies (controls for dataset-level confounding)
  (b) Per-dataset stratified
  (c) SampEn substituted for WPE
  (d) LZC substituted for WPE

Output:
  phase_f/data/par_f2_series_results.json
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from _paths import P


# -------------------------------- config --------------------------------

N_BOOT = 1000
SEED   = 42

DECISION_THRESHOLD_HIGH = 0.30   # ≥ → RESCUE per DECISION-005
DECISION_THRESHOLD_LOW  = 0.10   # < → NULL DEEPENS

HORIZONS_MIN = {'h010': 10, 'h030': 30, 'h060': 60, 'h120': 120}

OUT_JSON = Path(P['data']) / 'par_f2_series_results.json'


# -------------------------------- helpers --------------------------------

def find_first_existing(*candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def load_omega_file(filename, kind):
    """Try a few likely locations for omega CSVs."""
    candidates = [
        Path(P['results']) / 'omega' / filename,
        Path(P['results']) / filename,
        Path(P['thesis']) / 'results' / 'omega' / filename,
        Path(P['thesis']) / 'omega' / filename,
        Path(P['thesis']) / filename,
        Path('/workspace/kubernetes-cpu-ensemble-thesis/results/omega') / filename,
        Path('/workspace/kubernetes-cpu-ensemble-thesis') / filename,
    ]
    p = find_first_existing(*candidates)
    if p is None:
        print(f'  ERROR: could not find {filename} ({kind}). Tried:')
        for c in candidates:
            print(f'    {c}')
        return None
    print(f'  found {kind}: {p}')
    return pd.read_csv(p)


def standardize_bitbrains_ids(omega_bb):
    """vm_id in omega_bitbrains is plain integer; chronos2 spine uses 'bb_X'."""
    if 'vm_id' in omega_bb.columns:
        omega_bb = omega_bb.copy()
        omega_bb['container_id'] = 'bb_' + omega_bb['vm_id'].astype(str)
    elif 'container_id' not in omega_bb.columns:
        raise RuntimeError(f"omega_bitbrains has neither vm_id nor container_id: {list(omega_bb.columns)}")
    return omega_bb


def standardize_bytedance_ids(bd_stats):
    """instance_id; chronos2 spine uses 'bd_instance_X'."""
    bd_stats = bd_stats.copy()
    if 'instance_id' in bd_stats.columns:
        sample = str(bd_stats['instance_id'].iloc[0])
        if sample.startswith('bd_'):
            bd_stats['container_id'] = bd_stats['instance_id'].astype(str)
        else:
            bd_stats['container_id'] = 'bd_instance_' + bd_stats['instance_id'].astype(str)
    elif 'container_id' not in bd_stats.columns:
        raise RuntimeError(f"bytedance stats has neither instance_id nor container_id: {list(bd_stats.columns)}")
    return bd_stats


def load_and_merge():
    """Build the per-(series, horizon) analysis dataframe."""
    print('\n[1] Loading par_features.parquet...')
    feat = pd.read_parquet(Path(P['data']) / 'par_features.parquet')
    print(f'    rows={len(feat)}, cols={feat.shape[1]}')

    print('\n[2] Loading par_per_series_r2.parquet...')
    perr2 = pd.read_parquet(Path(P['data']) / 'par_per_series_r2.parquet')
    print(f'    rows={len(perr2)}')

    # extract NNLS rows → ml_benefit per (series, horizon)
    nnls = perr2[perr2['model'] == 'nnls'].copy()
    nnls['ml_benefit'] = nnls['r2'] - nnls['naive_r2']
    nnls = nnls[['container_id', 'dataset', 'horizon', 'ml_benefit']]
    n_before = len(nnls)
    nnls = nnls.dropna(subset=['ml_benefit'])
    print(f'    nnls rows: {n_before}, after drop NaN ml_benefit: {len(nnls)}')

    nnls['horizon_min'] = nnls['horizon'].map(HORIZONS_MIN)

    print('\n[3] Loading per-series ACF@24h from omega files...')
    omega_ali = load_omega_file('omega_alibaba.csv', 'alibaba')
    omega_bb  = load_omega_file('omega_bitbrains.csv', 'bitbrains')

    # bytedance: well-known path
    bd_path = find_first_existing(
        Path(P['results']) / 'bytedance' / 'diagnostics' / 'bytedance_per_instance_stats.csv',
        Path(P['results']) / 'bytedance_per_instance_stats.csv',
        Path('/workspace/kubernetes-cpu-ensemble-thesis/results/bytedance/diagnostics/bytedance_per_instance_stats.csv'),
    )
    if bd_path is None:
        print('  ERROR: bytedance_per_instance_stats.csv not found')
        bd_stats = None
    else:
        print(f'  found bytedance: {bd_path}')
        bd_stats = pd.read_csv(bd_path)

    if any(x is None for x in [omega_ali, omega_bb, bd_stats]):
        print('\nFATAL: missing omega/stats files. Aborting.')
        sys.exit(1)

    omega_bb = standardize_bitbrains_ids(omega_bb)
    bd_stats = standardize_bytedance_ids(bd_stats)

    print('  ID format checks:')
    print(f'    alibaba omega container_id sample: {list(omega_ali["container_id"].astype(str).head(3))}')
    print(f'    bitbrains omega container_id sample: {list(omega_bb["container_id"].astype(str).head(3))}')
    print(f'    bytedance container_id sample: {list(bd_stats["container_id"].astype(str).head(3))}')

    # stack ACF
    ali_acf = omega_ali[['container_id', 'acf_24h', 'acf_1h']].copy()
    ali_acf['dataset'] = 'alibaba'
    bb_acf = omega_bb[['container_id', 'acf_24h', 'acf_1h']].copy()
    bb_acf['dataset'] = 'bitbrains'

    bd_cols = ['container_id', 'acf_24h']
    if 'acf_1h' in bd_stats.columns:
        bd_cols.append('acf_1h')
    bd_acf = bd_stats[bd_cols].copy()
    bd_acf['dataset'] = 'bytedance'
    if 'acf_1h' not in bd_acf.columns:
        bd_acf['acf_1h'] = np.nan

    acf_df = pd.concat([ali_acf, bb_acf, bd_acf], ignore_index=True)
    acf_df['acf_24h'] = pd.to_numeric(acf_df['acf_24h'], errors='coerce')
    acf_df['acf_1h']  = pd.to_numeric(acf_df['acf_1h'],  errors='coerce')

    print(f'    acf_df rows: {len(acf_df)} '
          f'(alibaba={len(ali_acf)}, bitbrains={len(bb_acf)}, bytedance={len(bd_acf)})')

    print('\n[4] Merging...')
    df = nnls.merge(acf_df, on=['container_id', 'dataset'], how='left')
    df = df.merge(
        feat[['container_id', 'dataset', 'WPE', 'SampEn', 'LZC', 'DFA', 'cv', 'std', 'mean']],
        on=['container_id', 'dataset'], how='left',
    )
    print(f'    merged rows: {len(df)}')
    print(f'    per-dataset:\n{df.groupby("dataset").size().to_string()}')

    return df


def partial_r2(y, X_red, X_full):
    """partial R² of full model vs reduced; both fit OLS on same y."""
    lr_r = LinearRegression().fit(X_red, y)
    r2_r = lr_r.score(X_red, y)
    lr_f = LinearRegression().fit(X_full, y)
    r2_f = lr_f.score(X_full, y)
    if (1.0 - r2_r) < 1e-10:
        return np.nan
    return (r2_f - r2_r) / (1.0 - r2_r)


def cluster_bootstrap(df, y_col, x_test_col, x_control_cols, cluster_col, n_boot=1000, seed=42):
    """Bootstrap partial-R² by sampling clusters with replacement."""
    y_all      = df[y_col].values
    X_red_all  = df[x_control_cols].values
    X_full_all = df[x_control_cols + [x_test_col]].values

    # cluster row indices
    cluster_idx = df.groupby(cluster_col).indices  # {cluster_id: array_of_row_idx}
    clusters    = list(cluster_idx.keys())
    n_clusters  = len(clusters)

    rng = np.random.default_rng(seed)
    out = []
    n_fail = 0
    for b in range(n_boot):
        # sample n_clusters clusters with replacement
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        boot_idx = np.concatenate([cluster_idx[clusters[i]] for i in sampled])
        try:
            y_boot      = y_all[boot_idx]
            X_red_boot  = X_red_all[boot_idx]
            X_full_boot = X_full_all[boot_idx]
            pr2 = partial_r2(y_boot, X_red_boot, X_full_boot)
            if not np.isnan(pr2):
                out.append(pr2)
            else:
                n_fail += 1
        except Exception:
            n_fail += 1
        if (b + 1) % 200 == 0:
            print(f'      bootstrap {b+1}/{n_boot}  (n_fail={n_fail})')

    return np.array(out)


def verdict_from_partial_r2(pr2_point, ci_lower, ci_upper):
    """Apply DECISION-005 thresholds."""
    if pr2_point >= DECISION_THRESHOLD_HIGH:
        if ci_lower >= DECISION_THRESHOLD_HIGH:
            return 'F2 RESCUED at series scope (point estimate ≥ 0.30 and 95% CI lower ≥ 0.30)'
        return 'F2 RESCUE BORDERLINE (point estimate ≥ 0.30 but 95% CI overlaps threshold)'
    if pr2_point < DECISION_THRESHOLD_LOW:
        if ci_upper < DECISION_THRESHOLD_LOW:
            return 'F2 NULL DEEPENS at series scope (point estimate < 0.10 and 95% CI upper < 0.10)'
        return 'F2 BORDERLINE (point estimate < 0.10 but 95% CI overlaps threshold)'
    return f'F2 BORDERLINE (0.10 ≤ partial-R² = {pr2_point:.4f} < 0.30)'


def run_test(df, feature_name, controls, label, results_dict):
    """Run one partial-R² test config + bootstrap, store in results_dict."""
    print(f'\n[{label}]')
    print(f'  feature: {feature_name}, controls: {controls}')
    cols_needed = ['ml_benefit', feature_name] + controls
    cols_needed = [c for c in cols_needed if c in df.columns]
    df_clean = df.dropna(subset=cols_needed).copy()
    n_obs = len(df_clean)
    n_clusters = int(df_clean['container_id'].nunique())
    print(f'  n_obs: {n_obs}, n_clusters: {n_clusters}')

    if n_obs < 100 or n_clusters < 10:
        print(f'  too few observations or clusters; skipping')
        results_dict[label] = {'skipped': True, 'reason': 'insufficient data',
                               'n_obs': n_obs, 'n_clusters': n_clusters}
        return

    # Point estimate
    y      = df_clean['ml_benefit'].values
    X_red  = df_clean[controls].values
    X_full = df_clean[controls + [feature_name]].values
    pr2_point = partial_r2(y, X_red, X_full)
    r2_red  = LinearRegression().fit(X_red, y).score(X_red, y)
    r2_full = LinearRegression().fit(X_full, y).score(X_full, y)
    print(f'  R²(reduced)={r2_red:.4f}, R²(full)={r2_full:.4f}, partial-R²={pr2_point:.4f}')

    # Cluster bootstrap
    print(f'  cluster bootstrap ({N_BOOT} resamples)...')
    t0 = time.time()
    boot_vals = cluster_bootstrap(df_clean, 'ml_benefit', feature_name, controls,
                                  'container_id', n_boot=N_BOOT, seed=SEED)
    elapsed = time.time() - t0
    print(f'  bootstrap: {len(boot_vals)} valid samples in {elapsed:.0f}s')

    if len(boot_vals) < 50:
        verdict = 'INCONCLUSIVE: too few valid bootstrap samples'
        ci_low, ci_high = np.nan, np.nan
    else:
        ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
        verdict = verdict_from_partial_r2(pr2_point, ci_low, ci_high)

    print(f'  95% CI: [{ci_low:.4f}, {ci_high:.4f}]')
    print(f'  VERDICT: {verdict}')

    results_dict[label] = {
        'feature': feature_name,
        'controls': controls,
        'n_obs': n_obs,
        'n_clusters': n_clusters,
        'r2_reduced': float(r2_red),
        'r2_full': float(r2_full),
        'partial_r2_point': float(pr2_point) if not np.isnan(pr2_point) else None,
        'ci_lower_95': float(ci_low) if not np.isnan(ci_low) else None,
        'ci_upper_95': float(ci_high) if not np.isnan(ci_high) else None,
        'n_bootstrap_valid': int(len(boot_vals)),
        'verdict': verdict,
    }


def main():
    print('=' * 78)
    print('par_f2_series.py — per-series F2 partial-R² test')
    print('=' * 78)

    df = load_and_merge()

    print('\n[5] Distribution checks for analysis variables:')
    for col in ['ml_benefit', 'acf_24h', 'WPE', 'SampEn', 'LZC', 'DFA']:
        if col not in df.columns:
            continue
        n_nan = int(df[col].isna().sum())
        d = df[col].dropna()
        if len(d) == 0:
            print(f'   {col}: all NaN')
            continue
        print(f'   {col:<12} n_nan={n_nan:5d}  '
              f'mean={d.mean():.4f}  '
              f'std={d.std():.4f}  '
              f'min={d.min():.4f}  '
              f'p25={d.quantile(0.25):.4f}  '
              f'p50={d.median():.4f}  '
              f'p75={d.quantile(0.75):.4f}  '
              f'max={d.max():.4f}')

    print()
    print('=' * 78)
    print('PRE-REGISTERED PRIMARY TEST')
    print('=' * 78)

    results = {}

    # ---- PRIMARY: partial-R²(WPE | ACF@24h, horizon) pooled across all datasets ----
    run_test(df, 'WPE', ['acf_24h', 'horizon_min'],
             'primary_WPE_pooled', results)

    print()
    print('=' * 78)
    print('SENSITIVITY ANALYSES (not pre-registered)')
    print('=' * 78)

    # ---- (a) WITH dataset dummies ----
    df_dum = pd.get_dummies(df, columns=['dataset'], drop_first=True)
    ds_dummy_cols = [c for c in df_dum.columns if c.startswith('dataset_')]
    # statsmodels-style: dummies must be float for OLS
    for c in ds_dummy_cols:
        df_dum[c] = df_dum[c].astype(float)
    run_test(df_dum, 'WPE', ['acf_24h', 'horizon_min'] + ds_dummy_cols,
             'sens_WPE_with_dataset_dummies', results)

    # ---- (b) per-dataset stratified ----
    for ds in ['alibaba', 'bitbrains', 'bytedance']:
        sub = df[df['dataset'] == ds].copy()
        if len(sub) > 100:
            run_test(sub, 'WPE', ['acf_24h', 'horizon_min'],
                     f'sens_WPE_{ds}', results)

    # ---- (c) SampEn substituted for WPE ----
    run_test(df, 'SampEn', ['acf_24h', 'horizon_min'],
             'sens_SampEn_pooled', results)

    # ---- (d) LZC substituted for WPE ----
    run_test(df, 'LZC', ['acf_24h', 'horizon_min'],
             'sens_LZC_pooled', results)

    # ---- (e) DFA substituted for WPE ----
    run_test(df, 'DFA', ['acf_24h', 'horizon_min'],
             'sens_DFA_pooled', results)

    # ---- combined: all four predictability features ----
    print('\n=== combined partial-R² (WPE + SampEn + LZC + DFA together) ===')
    df_clean = df.dropna(subset=['ml_benefit', 'acf_24h', 'horizon_min',
                                  'WPE', 'SampEn', 'LZC', 'DFA']).copy()
    if len(df_clean) > 100:
        y = df_clean['ml_benefit'].values
        X_red = df_clean[['acf_24h', 'horizon_min']].values
        X_full = df_clean[['acf_24h', 'horizon_min', 'WPE', 'SampEn', 'LZC', 'DFA']].values
        pr2 = partial_r2(y, X_red, X_full)
        r2_red = LinearRegression().fit(X_red, y).score(X_red, y)
        r2_full = LinearRegression().fit(X_full, y).score(X_full, y)
        print(f'  n_obs={len(df_clean)}, R²(red)={r2_red:.4f}, R²(full)={r2_full:.4f}, '
              f'partial-R²={pr2:.4f}')
        results['sens_all_4_features_combined'] = {
            'feature': 'WPE+SampEn+LZC+DFA',
            'controls': ['acf_24h', 'horizon_min'],
            'n_obs': len(df_clean),
            'r2_reduced': float(r2_red),
            'r2_full': float(r2_full),
            'partial_r2_point': float(pr2),
            'note': 'point estimate only; no bootstrap (descriptive)',
        }

    # metadata
    results['metadata'] = {
        'pre_registration': 'DECISION-005, D5 close',
        'primary_feature': 'WPE',
        'primary_controls': ['acf_24h', 'horizon_min'],
        'outcome': 'ml_benefit = NNLS_R² - naive_R² per (series, horizon)',
        'decision_threshold_high': DECISION_THRESHOLD_HIGH,
        'decision_threshold_low': DECISION_THRESHOLD_LOW,
        'n_bootstrap_iterations': N_BOOT,
        'bootstrap_clustering': 'container_id',
        'bootstrap_seed': SEED,
        'cell_level_F2_result': {
            'partial_r2_pooled_n12': 0.0790,
            'verdict': 'NULL at cell scope (DECISION-009/010)',
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print('=' * 78)
    print(f'Saved: {OUT_JSON}')
    print('=' * 78)

    # final summary
    print()
    print('FINAL SUMMARY')
    print('-' * 78)
    if 'primary_WPE_pooled' in results and 'verdict' in results['primary_WPE_pooled']:
        r = results['primary_WPE_pooled']
        print(f'PRIMARY (pre-registered):')
        print(f'  partial-R²(WPE | ACF@24h, horizon) = {r["partial_r2_point"]:.4f} '
              f'[{r["ci_lower_95"]:.4f}, {r["ci_upper_95"]:.4f}]')
        print(f'  VERDICT: {r["verdict"]}')
    print()
    print('Sensitivity point estimates (no bootstrap printed for brevity):')
    for k in ['sens_WPE_with_dataset_dummies', 'sens_WPE_alibaba',
              'sens_WPE_bitbrains', 'sens_WPE_bytedance',
              'sens_SampEn_pooled', 'sens_LZC_pooled', 'sens_DFA_pooled',
              'sens_all_4_features_combined']:
        if k in results and 'partial_r2_point' in results[k] and results[k].get('partial_r2_point') is not None:
            pr2 = results[k]['partial_r2_point']
            n = results[k].get('n_obs', '?')
            print(f'  {k:<42} partial-R² = {pr2:.4f}  (n={n})')


if __name__ == '__main__':
    main()
