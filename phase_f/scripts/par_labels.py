"""
par_labels.py — PAR Step 4. Per-series routing labels from per-series R².

Input:  phase_f/data/par_per_series_r2.parquet
Output: phase_f/data/par_labels.parquet

For each (container_id, dataset, horizon), pivots the 4 routed-model R² values
into wide format and computes:
  label           = argmax R² across {nnls, chronos2, timesfm, granite_ttm}
  best_r2         = max R²
  second_best_r2  = second-best R²
  margin          = best_r2 - second_best_r2   (how decisive the label is)
  n_candidates    = number of models with non-NaN R² for this (series, horizon)

Drop rows where n_candidates < 2 (argmax not meaningful).

Per DECISION-013, the routing pool is 4 models: NNLS, Chronos-2, TimesFM, Granite-TTM.
Toto is excluded (Phase A Appendix C robustness check, not a routing-pool member).

Output schema:
  container_id, dataset, horizon, horizon_min, label,
  best_r2, second_best_r2, margin, n_candidates,
  nnls, chronos2, timesfm, granite_ttm   (per-model R² for downstream join with features)

Run from phase_f/scripts/. Junior style — sequential, verbose.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import P


ROUTED_MODELS = ['nnls', 'chronos2', 'timesfm', 'granite_ttm']
HORIZONS_MIN  = {'h010': 10, 'h030': 30, 'h060': 60, 'h120': 120}

OUT_PARQUET = Path(P['data']) / 'par_labels.parquet'
OUT_CSV     = Path(P['data']) / 'par_labels.csv'


def main():
    print('=' * 78)
    print('par_labels.py — PAR Step 4: per-series routing labels')
    print('=' * 78)

    # ---- load per-series R² ----
    pq_path = Path(P['data']) / 'par_per_series_r2.parquet'
    print(f'\n[1] Loading {pq_path}...')
    df = pd.read_parquet(pq_path)
    print(f'    rows: {len(df)}')
    print(f'    models present: {sorted(df["model"].unique())}')

    # filter to the 4 routed models
    df_routed = df[df['model'].isin(ROUTED_MODELS)].copy()
    print(f'    routed-model rows: {len(df_routed)}')

    # Pivot: (container_id, dataset, horizon) × model → r2
    print('\n[2] Pivoting to wide format...')
    pivot = df_routed.pivot_table(
        index=['container_id', 'dataset', 'horizon'],
        columns='model',
        values='r2',
    ).reset_index()

    # ensure all 4 columns exist (defensive)
    for m in ROUTED_MODELS:
        if m not in pivot.columns:
            pivot[m] = np.nan

    pivot.columns.name = None  # remove the columns axis name from pivot

    print(f'    pivot rows: {len(pivot)}')
    print(f'    columns: {[c for c in pivot.columns if c not in ROUTED_MODELS]} + 4 model cols')

    # ---- n_candidates per row ----
    pivot['n_candidates'] = pivot[ROUTED_MODELS].notna().sum(axis=1)
    print('\n[3] n_candidates distribution:')
    print(pivot['n_candidates'].value_counts().sort_index().to_string())

    # drop rows with <2 candidate models (argmax not meaningful)
    n_before = len(pivot)
    pivot = pivot[pivot['n_candidates'] >= 2].copy()
    print(f'    dropped {n_before - len(pivot)} rows with <2 candidates')
    print(f'    rows with all 4 candidates: {(pivot["n_candidates"] == 4).sum()}')
    print(f'    rows kept: {len(pivot)}')

    # ---- argmax → label ----
    print('\n[4] Computing argmax label, best/second-best R², margin...')
    r2_only = pivot[ROUTED_MODELS]
    pivot['label']    = r2_only.idxmax(axis=1)
    pivot['best_r2']  = r2_only.max(axis=1)

    # second-best: sort descending, take index [1]
    # use fillna(-inf) so NaNs don't interfere with sort ordering
    r2_filled = pivot[ROUTED_MODELS].fillna(-np.inf).values
    sorted_desc = np.sort(r2_filled, axis=1)[:, ::-1]
    pivot['second_best_r2'] = sorted_desc[:, 1]
    pivot['second_best_r2'] = pivot['second_best_r2'].replace(-np.inf, np.nan)
    pivot['margin'] = pivot['best_r2'] - pivot['second_best_r2']

    pivot['horizon_min'] = pivot['horizon'].map(HORIZONS_MIN)

    # reorder columns
    out_cols = ['container_id', 'dataset', 'horizon', 'horizon_min', 'label',
                'best_r2', 'second_best_r2', 'margin', 'n_candidates'] + ROUTED_MODELS
    pivot = pivot[out_cols]

    # ---- summaries ----
    print()
    print('=' * 78)
    print('LABEL DISTRIBUTION')
    print('=' * 78)

    print(f'\n[a] Pooled across all (series, horizon) — n={len(pivot)}:')
    cnt = pivot['label'].value_counts()
    for m in ROUTED_MODELS:
        c = int(cnt.get(m, 0))
        p = 100.0 * c / len(pivot)
        print(f'    {m:<15} {c:7d}  ({p:5.1f}%)')

    print(f'\n[b] Per (dataset, horizon) — % of containers per cell choosing each label:')
    cell_dist = pivot.groupby(['dataset', 'horizon'])['label'].value_counts(normalize=True).unstack('label', fill_value=0)
    cell_dist = (cell_dist * 100).round(1)
    # ensure all model columns present
    for m in ROUTED_MODELS:
        if m not in cell_dist.columns:
            cell_dist[m] = 0.0
    cell_dist = cell_dist[ROUTED_MODELS]
    print(cell_dist.to_string())

    print(f'\n[c] Per dataset (across horizons):')
    ds_dist = pivot.groupby('dataset')['label'].value_counts(normalize=True).unstack('label', fill_value=0)
    ds_dist = (ds_dist * 100).round(1)
    for m in ROUTED_MODELS:
        if m not in ds_dist.columns:
            ds_dist[m] = 0.0
    ds_dist = ds_dist[ROUTED_MODELS]
    print(ds_dist.to_string())

    print()
    print('=' * 78)
    print('MARGIN DISTRIBUTION  (how decisive each label is)')
    print('=' * 78)
    print(pivot['margin'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4).to_string())

    conf = pivot[pivot['margin'] > 0.10]
    mod  = pivot[(pivot['margin'] >= 0.01) & (pivot['margin'] <= 0.10)]
    amb  = pivot[pivot['margin'] < 0.01]
    print('\nLabel decisiveness buckets:')
    print(f'  margin > 0.10  (clear-cut):  {len(conf):7d}  ({100*len(conf)/len(pivot):5.1f}%)')
    print(f'  0.01 ≤ margin ≤ 0.10:        {len(mod):7d}  ({100*len(mod)/len(pivot):5.1f}%)')
    print(f'  margin < 0.01  (ambiguous):  {len(amb):7d}  ({100*len(amb)/len(pivot):5.1f}%)')

    print()
    print('=' * 78)
    print('STEP 5 BASELINES (what PAR must beat)')
    print('=' * 78)

    # Always-pick-most-common (global) baseline
    most_common      = cnt.idxmax()
    baseline_global  = float(cnt[most_common]) / len(pivot)
    print(f'\n[baseline 1] Always-pick-globally-most-common ("{most_common}"):')
    print(f'    accuracy = {baseline_global:.4f}')
    print(f'    macro-F1 ≤ {1.0/len(ROUTED_MODELS):.4f}  (only 1 class predicted → other 3 classes get F1=0)')

    # Always-pick-cell-winner baseline
    cell_top = pivot.groupby(['dataset', 'horizon']).apply(
        lambda g: g['label'].value_counts().iloc[0], include_groups=False,
    )
    overall_cell_correct = int(cell_top.sum())
    baseline_cell = overall_cell_correct / len(pivot)
    print(f'\n[baseline 2] Always-pick-cell-winner (the per-cell most common label):')
    print(f'    correct: {overall_cell_correct}/{len(pivot)} = {baseline_cell:.4f}')

    # Per-cell concentration of the top label
    cell_top_pct = pivot.groupby(['dataset', 'horizon']).apply(
        lambda g: g['label'].value_counts().iloc[0] / len(g), include_groups=False,
    )
    cell_top_pct.name = 'top_label_pct'
    print(f'\n[baseline 2 detail] Per-cell concentration of the cell-winner label:')
    print(cell_top_pct.round(3).to_string())
    print(f'  median across cells: {cell_top_pct.median():.4f}')
    print(f'  range: [{cell_top_pct.min():.4f}, {cell_top_pct.max():.4f}]')

    # Pre-registered DECISION-013 thresholds
    print()
    print('=' * 78)
    print('PRE-REGISTERED DECISION-013 THRESHOLDS')
    print('=' * 78)
    print('  macro-F1 ≥ 0.55              → HEADLINE (per-series routing works)')
    print('  0.20 ≤ macro-F1 < 0.55       → PARTIAL POSITIVE')
    print('  macro-F1 < 0.20              → STRUCTURAL SATURATION DEEPENS')
    print()
    print(f'  Baseline 1 macro-F1 ceiling: {1.0/len(ROUTED_MODELS):.4f}  '
          f'(always-pick-{most_common})')
    print(f'  Baseline 2 (cell-winner): we will compute its actual macro-F1 in Step 5')

    # Save
    print()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        pivot.to_parquet(OUT_PARQUET, index=False)
        print(f'Saved: {OUT_PARQUET}')
    except Exception as e:
        print(f'Parquet save failed: {e}')
    pivot.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}')


if __name__ == '__main__':
    main()
