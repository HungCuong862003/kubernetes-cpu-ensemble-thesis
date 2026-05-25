"""
par_per_series_r2.py — compute per-series R² for {NNLS, Chronos-2, TimesFM,
Granite-TTM, Toto} on the chronos2 K=20 spine, per (dataset, horizon, container).

Foundation models: per-container R² computed directly from each .npz
                   (each .npz has y_true, y_pred_mean, y_pred_p50, y_naive,
                    container_ids, origins, horizon_label, h_steps).

NNLS:               aligned to chronos2's K=20 spine via the formula
                       nnls_idx = container_start_in_sorted_dropped(cid) + origin
                    where sorted_dropped =
                       test.parquet
                         .sort_values([id_col, 'time_stamp'])
                         .pipe(drop last h_steps rows per container)
                    Alignment verified D17 alignment test (1000/1000 on Alibaba h120).

Output:
  phase_f/data/par_per_series_r2.parquet
  phase_f/data/par_per_series_r2.csv

Schema: container_id, dataset, horizon, horizon_min, model,
        r2, n_points, naive_r2, pred_key, source

Run from phase_f/scripts/. Junior style — verbose, sequential, no abstractions.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import P


# ----------------------------- config -----------------------------

DATASETS = {
    'alibaba':   ['h010', 'h030', 'h060', 'h120'],
    'bitbrains': ['h010', 'h030', 'h060', 'h120'],
    'bytedance': ['h030', 'h060', 'h120'],
}

FOUNDATION_MODELS = ['chronos2', 'timesfm', 'granite_ttm', 'toto']

TOTO_PRED_KEY    = 'y_pred_p50'   # toto y_pred_mean has extreme outliers (min -3087)
DEFAULT_PRED_KEY = 'y_pred_mean'

NNLS_FILE_FALLBACK = ['test_ensemble_hetero.npy', 'test_ensemble_homo.npy']

# cadence in minutes; bytedance is 10-min, others 5-min
CADENCE_MIN = {'alibaba': 5, 'bitbrains': 5, 'bytedance': 10}

OUT_PARQUET = Path(P['data']) / 'par_per_series_r2.parquet'
OUT_CSV     = Path(P['data']) / 'par_per_series_r2.csv'


# --------------------------- helpers ---------------------------

def detect_id_col(df):
    """Find the container ID column in test.parquet."""
    for c in ['container_id', 'vm_id', 'instance_id', 'machine_id']:
        if c in df.columns:
            return c
    raise RuntimeError(f"No ID column found among {list(df.columns)[:8]}")


def per_container_r2(y_true, y_pred, container_ids):
    """
    Vectorised per-container R² computation.
    Returns DataFrame: container_id, r2, n_points.
    Drops rows with NaN y_pred or y_true.
    """
    yp = np.asarray(y_pred, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64)
    cids = np.asarray(container_ids)

    valid = ~np.isnan(yp) & ~np.isnan(yt)
    if valid.sum() == 0:
        return pd.DataFrame(columns=['container_id', 'r2', 'n_points'])

    df = pd.DataFrame({
        'cid': cids[valid],
        'yt':  yt[valid],
        'yp':  yp[valid],
    })

    # per-container yt mean (broadcast back to row level)
    df['_mean'] = df.groupby('cid', sort=False)['yt'].transform('mean')
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
    out = agg.reset_index().rename(columns={'cid': 'container_id'})
    return out[['container_id', 'r2', 'n_points']]


def build_container_starts(sorted_df, id_col, h_steps):
    """
    Compute per-container start index in sorted_dropped.
    Returns:
      starts: Series, container_id → start index (int64)
      sizes:  Series, container_id → number of rows in sorted_dropped
    Containers with total_size <= h_steps are excluded (no valid evaluation rows).
    """
    total = sorted_df.groupby(id_col, sort=False).size()
    valid = total - h_steps
    valid = valid[valid > 0]
    starts = valid.cumsum().shift(fill_value=0).astype(np.int64)
    starts.name = 'start'
    valid.name = 'size'
    return starts, valid


# --------------------------- main ---------------------------

def main():
    rows_all = []
    t_global = time.time()

    for ds, horizons in DATASETS.items():
        print()
        print('=' * 70)
        print(f'DATASET: {ds}')
        print('=' * 70)

        # ---- load + sort test.parquet once per dataset ----
        test_pq = Path(P['thesis']) / 'data' / 'processed' / ds / 'test.parquet'
        if not test_pq.exists():
            print(f'  test.parquet MISSING ({test_pq}); NNLS for {ds} will be skipped')
            sorted_df = None
            id_col = None
        else:
            print(f'  loading test.parquet ({test_pq})...')
            t = time.time()
            raw = pd.read_parquet(test_pq)
            print(f'    rows={len(raw)}, columns={list(raw.columns)[:6]}..., '
                  f'elapsed {time.time() - t:.1f}s')

            id_col = detect_id_col(raw)
            print(f'    detected id_col={id_col}')

            print(f'  sorting by [{id_col}, time_stamp]...')
            t = time.time()
            sorted_df = raw[[id_col, 'time_stamp']].sort_values(
                [id_col, 'time_stamp']
            ).reset_index(drop=True)
            print(f'    sorted; elapsed {time.time() - t:.1f}s')
            del raw

        for h in horizons:
            print()
            print(f'-- {ds} {h} --')

            chronos2_data = None

            # ---- foundation models from .npz ----
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

                    # toto: use p50 (mean has outliers). others: use mean. Granite p50 all NaN.
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

                med   = float(df_model['r2'].median())
                med_n = float(df_model['naive_r2'].median())
                print(f'    {model:<12}: n_containers={len(df_model)}, '
                      f'median R²={med:.4f}, median naive={med_n:.4f} '
                      f'(pred_key={pred_key}, n_points_total={len(yt)})')

                if model == 'chronos2':
                    chronos2_data = dict(cids=cids, origins=origins, yt=yt, h_steps=h_steps)

            # ---- NNLS aligned to chronos2 spine ----
            if chronos2_data is None:
                print(f'    {"nnls":<12}: skipped (no chronos2 spine)')
                continue
            if sorted_df is None or id_col is None:
                print(f'    {"nnls":<12}: skipped (no test.parquet)')
                continue

            h_steps = chronos2_data['h_steps']
            print(f'    {"nnls":<12}: building container starts for h_steps={h_steps}...')
            starts, sizes = build_container_starts(sorted_df, id_col, h_steps)
            print(f'                  {len(starts)} containers in test.parquet')

            # ID match check
            npz_cids = chronos2_data['cids']
            unique_npz_cids = pd.unique(npz_cids)
            in_starts = pd.Series(unique_npz_cids).isin(starts.index).sum()
            print(f'                  ID match: {in_starts}/{len(unique_npz_cids)} '
                  f'chronos2 containers present in test.parquet')
            if in_starts < 0.5 * len(unique_npz_cids):
                samp_npz  = list(unique_npz_cids[:3])
                samp_test = list(starts.index[:3])
                print(f'                  POOR MATCH. Sample npz IDs: {samp_npz}, '
                      f'test IDs: {samp_test}. Skipping NNLS for {ds} {h}.')
                continue

            # NNLS ensemble file (hetero preferred, homo fallback)
            pred_dir = Path(P['results']) / ds / h / 'predictions'
            ens_path = None
            for fname in NNLS_FILE_FALLBACK:
                if (pred_dir / fname).exists():
                    ens_path = pred_dir / fname
                    break
            if ens_path is None:
                print(f'    {"nnls":<12}: no ensemble file in {pred_dir}, skipping')
                continue
            print(f'                  ensemble file: {ens_path.name}')

            try:
                y_true_full  = np.load(pred_dir / 'y_true.npy')
                y_nnls_full  = np.load(ens_path)
                y_naive_path = pred_dir / 'y_naive.npy'
                y_naive_full = np.load(y_naive_path) if y_naive_path.exists() else None
            except Exception as e:
                print(f'    {"nnls":<12}: load error {e}')
                continue

            # Map each chronos2 row to its NNLS index
            start_arr = starts.reindex(npz_cids).values
            valid_mask = ~np.isnan(start_arr.astype(np.float64))

            nnls_idx = np.zeros(len(npz_cids), dtype=np.int64)
            nnls_idx[valid_mask] = start_arr[valid_mask].astype(np.int64) + chronos2_data['origins'][valid_mask]

            # bounds check
            within = (nnls_idx >= 0) & (nnls_idx < len(y_nnls_full))
            valid_mask = valid_mask & within
            n_aligned = int(valid_mask.sum())

            # sanity: y_true_npy at aligned index must match chronos2's y_true
            yt_npy_aligned = y_true_full[nnls_idx[valid_mask]]
            yt_c2_aligned  = chronos2_data['yt'][valid_mask]
            n_mismatch = int((np.abs(yt_npy_aligned - yt_c2_aligned) > 1e-3).sum())
            print(f'                  aligned {n_aligned}/{len(npz_cids)}, '
                  f'y_true mismatches: {n_mismatch}')

            if n_mismatch > 0.005 * max(n_aligned, 1):
                print(f'                  TOO MANY MISMATCHES ({n_mismatch}); skipping NNLS for {ds} {h}')
                continue

            # extract aligned NNLS predictions + compute per-container R²
            y_nnls_aligned = y_nnls_full[nnls_idx[valid_mask]]
            cids_aligned   = npz_cids[valid_mask]

            df_nnls = per_container_r2(yt_npy_aligned, y_nnls_aligned, cids_aligned)

            # naive on the same spine
            if y_naive_full is not None:
                y_naive_aligned = y_naive_full[nnls_idx[valid_mask]]
            else:
                # fall back to chronos2's y_naive
                y_naive_aligned = npz['y_naive'][valid_mask].astype(np.float64)
            df_nnls_naive = per_container_r2(yt_npy_aligned, y_naive_aligned, cids_aligned).rename(
                columns={'r2': 'naive_r2'}
            )[['container_id', 'naive_r2']]
            df_nnls = df_nnls.merge(df_nnls_naive, on='container_id', how='left')

            df_nnls['model']       = 'nnls'
            df_nnls['dataset']     = ds
            df_nnls['horizon']     = h
            df_nnls['horizon_min'] = h_steps * CADENCE_MIN[ds]
            df_nnls['pred_key']    = ens_path.name
            df_nnls['source']      = 'aligned_to_chronos2_spine'
            rows_all.append(df_nnls)

            med   = float(df_nnls['r2'].median())
            med_n = float(df_nnls['naive_r2'].median())
            print(f'    {"nnls":<12}: n_containers={len(df_nnls)}, '
                  f'median R²={med:.4f}, median naive={med_n:.4f}')

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
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Total rows: {len(all_df)}, total elapsed: {time.time() - t_global:.1f}s')

    print('\nContainer count per (dataset, horizon, model):')
    cnt = all_df.groupby(['dataset', 'horizon', 'model']).size().unstack('model', fill_value=0)
    print(cnt.to_string())

    print('\nMedian R² per (dataset, horizon, model):')
    med = all_df.groupby(['dataset', 'horizon', 'model'])['r2'].median().unstack('model')
    print(med.round(4).to_string())

    print('\nMedian (R² − naive_r2) in pp per (dataset, horizon, model):')
    all_df['delta_pp'] = (all_df['r2'] - all_df['naive_r2']) * 100
    delta = all_df.groupby(['dataset', 'horizon', 'model'])['delta_pp'].median().unstack('model')
    print(delta.round(2).to_string())
    all_df = all_df.drop(columns='delta_pp')

    # Per-cell winner tally — sanity check against leaderboard_v1.csv
    print('\nPer-cell winner tally (argmax R² across {nnls, c2, tfm, granite}, per container, then majority per cell):')
    routed_models = ['nnls', 'chronos2', 'timesfm', 'granite_ttm']
    rdf = all_df[all_df['model'].isin(routed_models)].copy()
    if not rdf.empty:
        pivot = rdf.pivot_table(index=['dataset', 'horizon', 'container_id'],
                                columns='model', values='r2')
        pivot = pivot.dropna(how='all')
        pivot['_winner'] = pivot[routed_models].idxmax(axis=1)
        cell_winners = pivot.reset_index().groupby(['dataset', 'horizon'])['_winner'].apply(
            lambda s: s.mode().iloc[0] if not s.empty else None
        )
        print(cell_winners.to_string())

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
