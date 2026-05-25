"""
par_per_series_r2_v2.py — per-series R² for PAR, with corrected NNLS sources.

Changes from v1:

  1. Alibaba and ByteDance NNLS now use NEW pool (y_naive + test_ensemble_v3.npy)
     instead of OLD pool (test_ensemble_hetero.npy). Matches leaderboard_v1.csv
     for these datasets.

  2. Bitbrains NNLS now uses test_xgb.npy directly. The hetero file at the Bitbrains
     path was found to contain Alibaba's data (shape and value distribution match
     Alibaba exactly, not Bitbrains; confirmed in par_diagnose_issues.py Section A).
     test_xgb.npy is the closest single-model surrogate matching leaderboard_v1.csv's
     "per_vm_median" / "OLD (XGB+LGB+ET+BiLSTM)" convention for the Bitbrains row.
     The substitution is documented in the chapter's methods section.

  3. Two pre-flight sanity checks abort the script if alignment is wrong:
       (i)  Alibaba h120 NEW pool pooled R² == 0.8093 ± 0.02 (leaderboard_v1.csv)
       (ii) Bitbrains h30 test_xgb pooled R²  == 0.4973 ± 0.005 (diagnostic verified)

NNLS source per (dataset, horizon):
  Alibaba   × all 4         → y_naive.npy + test_ensemble_v3.npy   (NEW pool ensemble)
  Bitbrains × all 4         → test_xgb.npy                          (substitution; documented)
  ByteDance × h30/60/120    → y_naive.npy + test_ensemble_v3.npy   (NEW pool ensemble)

Within each cell, all 5 models (NNLS surrogate + Chronos-2 + TimesFM + Granite-TTM
+ Toto) are evaluated on the same chronos2 K=20/K=50 spine (apples-to-apples
per-series argmax). 142 Bitbrains VMs of 156 are covered by the spine (intersection).

Output: phase_f/data/par_per_series_r2.parquet (overwrites v1)
        phase_f/data/par_per_series_r2.csv

Run from phase_f/scripts/. Junior style — verbose, sequential.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import P


# -------------------------------- config --------------------------------

DATASETS = {
    'alibaba':   ['h010', 'h030', 'h060', 'h120'],
    'bitbrains': ['h010', 'h030', 'h060', 'h120'],
    'bytedance': ['h030', 'h060', 'h120'],
}

FOUNDATION_MODELS = ['chronos2', 'timesfm', 'granite_ttm', 'toto']

TOTO_PRED_KEY    = 'y_pred_p50'   # toto y_pred_mean has extreme outliers
DEFAULT_PRED_KEY = 'y_pred_mean'

# NNLS source per dataset
NNLS_METHOD_PER_DATASET = {
    'alibaba':   'v3_plus_naive',   # NEW pool
    'bytedance': 'v3_plus_naive',   # NEW pool
    'bitbrains': 'test_xgb',        # substitution (hetero file corrupted)
}

NNLS_LABELS = {
    'v3_plus_naive': {
        'pred_key': 'y_naive.npy + test_ensemble_v3.npy',
        'source':   'aligned_new_pool',
    },
    'test_xgb': {
        'pred_key': 'test_xgb.npy (hetero substitution; chapter-documented)',
        'source':   'aligned_xgb_substitution',
    },
}

CADENCE_MIN = {'alibaba': 5, 'bitbrains': 5, 'bytedance': 10}

# Pre-flight sanity checks
SANITY_CHECKS = [
    {
        'name':         'Alibaba h120 NEW pool (y_naive + v3) pooled R²',
        'dataset':      'alibaba',
        'horizon':      'h120',
        'method':       'v3_plus_naive',
        'expected_r2':  0.8093,
        'tolerance':    0.02,
        'source_note':  'leaderboard_v1.csv Alibaba h120 nnls_ensemble_r2',
    },
    {
        'name':         'Bitbrains h30 test_xgb pooled R²',
        'dataset':      'bitbrains',
        'horizon':      'h030',
        'method':       'test_xgb',
        'expected_r2':  0.4973,
        'tolerance':    0.005,
        'source_note':  'par_diagnose_issues.py Section F (verified)',
    },
]

OUT_PARQUET = Path(P['data']) / 'par_per_series_r2.parquet'
OUT_CSV     = Path(P['data']) / 'par_per_series_r2.csv'


# -------------------------------- helpers --------------------------------

def detect_id_col(df):
    for c in ['container_id', 'vm_id', 'instance_id', 'machine_id']:
        if c in df.columns:
            return c
    raise RuntimeError(f"No ID column found among {list(df.columns)[:8]}")


def build_container_starts(sorted_df, id_col, h_steps):
    total = sorted_df.groupby(id_col, sort=False).size()
    valid = total - h_steps
    valid = valid[valid > 0]
    starts = valid.cumsum().shift(fill_value=0).astype(np.int64)
    return starts, valid


def per_container_r2(y_true, y_pred, container_ids):
    yp = np.asarray(y_pred, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64)
    cids = np.asarray(container_ids)
    valid = ~np.isnan(yp) & ~np.isnan(yt)
    if valid.sum() == 0:
        return pd.DataFrame(columns=['container_id', 'r2', 'n_points'])
    df = pd.DataFrame({'cid': cids[valid], 'yt': yt[valid], 'yp': yp[valid]})
    df['_mean']  = df.groupby('cid', sort=False)['yt'].transform('mean')
    df['_sst_t'] = (df['yt'] - df['_mean']) ** 2
    df['_ssr_t'] = (df['yt'] - df['yp']) ** 2
    agg = df.groupby('cid', sort=False).agg(
        sst=('_sst_t', 'sum'),
        ssr=('_ssr_t', 'sum'),
        n_points=('yt', 'size'),
    )
    agg['r2'] = np.where(
        (agg['n_points'] >= 2) & (agg['sst'] > 0),
        1.0 - agg['ssr'] / agg['sst'],
        np.nan,
    )
    return agg.reset_index().rename(columns={'cid': 'container_id'})[
        ['container_id', 'r2', 'n_points']
    ]


def pooled_r2(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = ~np.isnan(yp) & ~np.isnan(yt)
    yt = yt[mask]; yp = yp[mask]
    sst = ((yt - yt.mean()) ** 2).sum()
    if sst == 0:
        return np.nan
    return float(1.0 - ((yt - yp) ** 2).sum() / sst)


def load_nnls_prediction_aligned(ds, h, method):
    """
    Align NNLS prediction to chronos2 spine.
    Returns (yt_aligned, y_nnls_aligned, y_naive_aligned, cids_aligned, n_aligned)
    or None on failure.
    """
    # spine
    npz_path = Path(P['results']) / ds / h / 'foundation_models' / 'chronos2_perpoint.npz'
    if not npz_path.exists():
        print(f'      spine .npz MISSING')
        return None
    npz = np.load(npz_path, allow_pickle=True)
    cids    = np.asarray(npz['container_ids']).astype(str)
    origins = np.asarray(npz['origins']).astype(np.int64)
    yt_c2   = np.asarray(npz['y_true']).astype(np.float64)
    h_steps = int(npz['h_steps'])

    # test.parquet → sorted → ranges
    test_pq = Path(P['thesis']) / 'data' / 'processed' / ds / 'test.parquet'
    if not test_pq.exists():
        print(f'      test.parquet MISSING')
        return None
    test_df = pd.read_parquet(test_pq)
    id_col = detect_id_col(test_df)
    sorted_df = test_df[[id_col, 'time_stamp']].sort_values(
        [id_col, 'time_stamp']
    ).reset_index(drop=True)
    starts, _ = build_container_starts(sorted_df, id_col, h_steps)

    # NNLS index per spine row
    start_arr = starts.reindex(cids).values
    valid_mask = ~pd.isna(start_arr.astype(np.float64))
    nnls_idx = np.zeros(len(cids), dtype=np.int64)
    nnls_idx[valid_mask] = start_arr[valid_mask].astype(np.int64) + origins[valid_mask]

    # bounds
    pred_dir = Path(P['results']) / ds / h / 'predictions'
    y_true_full = np.load(pred_dir / 'y_true.npy')
    bounds = (nnls_idx >= 0) & (nnls_idx < len(y_true_full))
    valid_mask = valid_mask & bounds

    # y_true sanity
    yt_npy_aligned = y_true_full[nnls_idx[valid_mask]]
    yt_c2_aligned  = yt_c2[valid_mask]
    n_mismatch = int((np.abs(yt_npy_aligned - yt_c2_aligned) > 1e-3).sum())
    n_aligned = int(valid_mask.sum())
    if n_mismatch > 0.005 * max(n_aligned, 1):
        print(f'      ALIGNMENT MISMATCH: {n_mismatch}/{n_aligned} y_true diffs')
        return None

    # load NNLS prediction (full-test scope)
    y_naive_full = np.load(pred_dir / 'y_naive.npy')
    if method == 'v3_plus_naive':
        v3_path = pred_dir / 'test_ensemble_v3.npy'
        if not v3_path.exists():
            print(f'      v3 file MISSING')
            return None
        v3_full = np.load(v3_path)
        y_pred_full = y_naive_full + v3_full
    elif method == 'test_xgb':
        xgb_path = pred_dir / 'test_xgb.npy'
        if not xgb_path.exists():
            print(f'      test_xgb file MISSING')
            return None
        y_pred_full = np.load(xgb_path)
    else:
        raise ValueError(f'unknown method: {method}')

    y_nnls_aligned  = y_pred_full[nnls_idx[valid_mask]]
    y_naive_aligned = y_naive_full[nnls_idx[valid_mask]]
    cids_aligned    = cids[valid_mask]
    return yt_npy_aligned, y_nnls_aligned, y_naive_aligned, cids_aligned, n_aligned


# -------------------------------- pre-flight --------------------------------

def run_sanity_checks():
    print('=' * 72)
    print('PRE-FLIGHT SANITY CHECKS')
    print('=' * 72)
    all_pass = True
    for chk in SANITY_CHECKS:
        print(f'\n[{chk["name"]}]')
        print(f'   expected source: {chk["source_note"]}')
        result = load_nnls_prediction_aligned(chk['dataset'], chk['horizon'], chk['method'])
        if result is None:
            print(f'   FAILED — could not compute.')
            all_pass = False
            continue
        yt, yp, _, _, n = result
        actual = pooled_r2(yt, yp)
        delta = abs(actual - chk['expected_r2'])
        status = 'PASS' if delta <= chk['tolerance'] else 'FAIL'
        print(f'   pooled R² actual   = {actual:.4f}')
        print(f'   pooled R² expected = {chk["expected_r2"]:.4f}')
        print(f'   delta = {delta:.4f}, tolerance = {chk["tolerance"]}')
        print(f'   n_aligned = {n}')
        print(f'   {status}')
        if status == 'FAIL':
            all_pass = False
    return all_pass


# -------------------------------- main --------------------------------

def main():
    print('=' * 72)
    print('par_per_series_r2_v2.py')
    print('=' * 72)

    if not run_sanity_checks():
        print('\nSANITY CHECK FAILURE — aborting before writing labels.')
        sys.exit(1)
    print('\nAll sanity checks PASSED. Proceeding with full regen.\n')

    rows_all = []
    t_global = time.time()

    for ds, horizons in DATASETS.items():
        print()
        print('=' * 72)
        print(f'DATASET: {ds}  (NNLS method = {NNLS_METHOD_PER_DATASET[ds]})')
        print('=' * 72)

        # Per-dataset test.parquet load
        test_pq = Path(P['thesis']) / 'data' / 'processed' / ds / 'test.parquet'
        if not test_pq.exists():
            print(f'  test.parquet MISSING; NNLS for {ds} skipped')
            sorted_df = None
            id_col = None
        else:
            print(f'  loading + sorting test.parquet...')
            raw = pd.read_parquet(test_pq)
            id_col = detect_id_col(raw)
            sorted_df = raw[[id_col, 'time_stamp']].sort_values(
                [id_col, 'time_stamp']
            ).reset_index(drop=True)
            del raw
            print(f'    sorted, id_col={id_col}, n_rows={len(sorted_df)}')

        for h in horizons:
            print(f'\n-- {ds} {h} --')
            chronos2_data = None

            # Foundation models — direct from .npz
            for model in FOUNDATION_MODELS:
                npz_path = Path(P['results']) / ds / h / 'foundation_models' / f'{model}_perpoint.npz'
                if not npz_path.exists():
                    print(f'    {model:<12}: .npz MISSING')
                    continue
                try:
                    npz = np.load(npz_path, allow_pickle=True)
                    cids    = np.asarray(npz['container_ids']).astype(str)
                    origins = np.asarray(npz['origins']).astype(np.int64)
                    yt      = np.asarray(npz['y_true']).astype(np.float64)
                    yn      = np.asarray(npz['y_naive']).astype(np.float64)
                    h_steps = int(npz['h_steps'])
                    pred_key = TOTO_PRED_KEY if model == 'toto' else DEFAULT_PRED_KEY
                    if pred_key not in npz.files:
                        pred_key = DEFAULT_PRED_KEY
                    yp = np.asarray(npz[pred_key]).astype(np.float64)
                except Exception as e:
                    print(f'    {model:<12}: READ ERROR {e}')
                    continue

                df_model = per_container_r2(yt, yp, cids)
                df_naive = per_container_r2(yt, yn, cids).rename(
                    columns={'r2': 'naive_r2'}
                )[['container_id', 'naive_r2']]
                df_model = df_model.merge(df_naive, on='container_id', how='left')
                df_model['model']       = model
                df_model['dataset']     = ds
                df_model['horizon']     = h
                df_model['horizon_min'] = h_steps * CADENCE_MIN[ds]
                df_model['pred_key']    = pred_key
                df_model['source']      = 'foundation_npz'
                rows_all.append(df_model)

                med = float(df_model['r2'].median())
                print(f'    {model:<12}: n={len(df_model)}, median R²={med:.4f}, pred_key={pred_key}')

                if model == 'chronos2':
                    chronos2_data = True  # only used for guard below

            # NNLS slot
            if chronos2_data is None:
                print(f'    {"nnls":<12}: skipped (no chronos2 spine)')
                continue
            if sorted_df is None:
                print(f'    {"nnls":<12}: skipped (no test.parquet)')
                continue

            method = NNLS_METHOD_PER_DATASET[ds]
            labels = NNLS_LABELS[method]

            result = load_nnls_prediction_aligned(ds, h, method)
            if result is None:
                print(f'    {"nnls":<12}: alignment / load failed; skipping')
                continue
            yt_aligned, y_nnls_aligned, y_naive_aligned, cids_aligned, n_aligned = result

            df_nnls = per_container_r2(yt_aligned, y_nnls_aligned, cids_aligned)
            df_naive = per_container_r2(yt_aligned, y_naive_aligned, cids_aligned).rename(
                columns={'r2': 'naive_r2'}
            )[['container_id', 'naive_r2']]
            df_nnls = df_nnls.merge(df_naive, on='container_id', how='left')

            # horizon_min: derive from chronos2 npz's h_steps
            npz_for_hsteps = np.load(
                Path(P['results']) / ds / h / 'foundation_models' / 'chronos2_perpoint.npz',
                allow_pickle=True,
            )
            h_steps = int(npz_for_hsteps['h_steps'])

            df_nnls['model']       = 'nnls'
            df_nnls['dataset']     = ds
            df_nnls['horizon']     = h
            df_nnls['horizon_min'] = h_steps * CADENCE_MIN[ds]
            df_nnls['pred_key']    = labels['pred_key']
            df_nnls['source']      = labels['source']
            rows_all.append(df_nnls)

            med = float(df_nnls['r2'].median())
            med_n = float(df_nnls['naive_r2'].median())
            print(f'    {"nnls":<12}: n={len(df_nnls)}, median R²={med:.4f}, '
                  f'median naive={med_n:.4f}, method={method}')

        if sorted_df is not None:
            del sorted_df

    # ----------------------- assemble + save -----------------------
    if not rows_all:
        print('\nNo rows accumulated. Aborting.')
        sys.exit(1)

    all_df = pd.concat(rows_all, ignore_index=True)
    all_df = all_df[['container_id', 'dataset', 'horizon', 'horizon_min',
                     'model', 'r2', 'n_points', 'naive_r2',
                     'pred_key', 'source']]

    print()
    print('=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'Total rows: {len(all_df)}, total elapsed: {time.time() - t_global:.1f}s')

    print('\nContainer count per (dataset, horizon, model):')
    cnt = all_df.groupby(['dataset', 'horizon', 'model']).size().unstack('model', fill_value=0)
    print(cnt.to_string())

    print('\nMedian R² per (dataset, horizon, model):')
    med = all_df.groupby(['dataset', 'horizon', 'model'])['r2'].median().unstack('model')
    print(med.round(4).to_string())

    print('\nMedian Δpp (model − naive) per (dataset, horizon, model):')
    all_df['delta_pp'] = (all_df['r2'] - all_df['naive_r2']) * 100
    delta = all_df.groupby(['dataset', 'horizon', 'model'])['delta_pp'].median().unstack('model')
    print(delta.round(2).to_string())
    all_df = all_df.drop(columns='delta_pp')

    # per-cell winner (per-container argmax → mode per cell)
    print('\nPer-cell winner (per-container argmax across {nnls, chronos2, timesfm, granite_ttm}):')
    routed = ['nnls', 'chronos2', 'timesfm', 'granite_ttm']
    rdf = all_df[all_df['model'].isin(routed)].copy()
    pivot = rdf.pivot_table(
        index=['dataset', 'horizon', 'container_id'],
        columns='model', values='r2',
    ).dropna(how='all')
    pivot['_winner'] = pivot[routed].idxmax(axis=1)
    winners = pivot.reset_index().groupby(['dataset', 'horizon'])['_winner'].apply(
        lambda s: s.mode().iloc[0] if not s.empty else None
    )
    print(winners.to_string())
    print('\nTally:')
    print(winners.value_counts().to_string())

    # also: per-cell vote distribution
    print('\nPer-cell vote distribution (% of containers per cell that pick each model):')
    cell_votes = pivot.reset_index().groupby(['dataset', 'horizon'])['_winner'].value_counts(
        normalize=True
    ).unstack('_winner').fillna(0.0)
    print((cell_votes * 100).round(1).to_string())

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        all_df.to_parquet(OUT_PARQUET, index=False)
        print(f'\nSaved: {OUT_PARQUET}')
    except Exception as e:
        print(f'\nParquet save failed: {e}')
    all_df.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}')


if __name__ == '__main__':
    main()
