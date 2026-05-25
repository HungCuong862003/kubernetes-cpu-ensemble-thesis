"""
par_diagnose_issues.py — investigate the two failure modes surfaced by
par_per_series_r2.py:

  1. Bitbrains NNLS R² catastrophic at h30/h60/h120 (median ~-0.40)
  2. ByteDance Granite-TTM R² catastrophic at all horizons (median ~-2 to -4.5)

Plus prep for the fix:

  3. Bitbrains vm_id format in bitbrains_per_vm_results.csv vs chronos2 container_ids
  4. leaderboard_v1.csv nnls_pool / nnls_agg_method for bitbrains rows
  5. Pooled R² sanity check (Alibaba NNLS h120 vs canonical 0.7642)

Sections A–F, read-only. Run from phase_f/scripts/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from _paths import P


def print_block(title):
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


# =====================================================================
# SECTION A — Bitbrains ensemble files: what's actually in them?
# =====================================================================

print_block('[A] Bitbrains ensemble files at h30 / h60 / h120')

for h in ['h010', 'h030', 'h060', 'h120']:
    pred_dir = Path(P['results']) / 'bitbrains' / h / 'predictions'
    print()
    print(f'-- bitbrains {h} -- ({pred_dir})')
    if not pred_dir.exists():
        print('   DIR MISSING')
        continue

    # list all .npy files in predictions/
    files = sorted(pred_dir.glob('*.npy'))
    print(f'   .npy files ({len(files)}):')
    for f in files:
        print(f'     {f.name}')

    # inspect key files
    key_files = ['y_true.npy', 'y_naive.npy', 'test_ensemble_hetero.npy',
                 'test_ensemble_homo.npy',  'test_ensemble_v2.npy',
                 'test_ensemble_v3.npy',    'test_xgb.npy',
                 'test_et.npy']

    print(f'   inspection (mean / min / max / n_nan):')
    for name in key_files:
        p = pred_dir / name
        if not p.exists():
            print(f'     {name:<28} MISSING')
            continue
        arr = np.load(p, allow_pickle=True)
        if arr.dtype.kind != 'f':
            print(f'     {name:<28} non-float dtype {arr.dtype}')
            continue
        n_nan = int(np.isnan(arr).sum())
        if arr.size == 0 or n_nan == arr.size:
            print(f'     {name:<28} shape={arr.shape} all-NaN/empty')
            continue
        mean = float(np.nanmean(arr))
        mn   = float(np.nanmin(arr))
        mx   = float(np.nanmax(arr))
        first5 = arr.flatten()[:5].tolist()
        print(f'     {name:<28} shape={arr.shape} mean={mean:.3f} '
              f'min={mn:.3f} max={mx:.3f} n_nan={n_nan} first5={[f"{x:.2f}" for x in first5]}')


# =====================================================================
# SECTION B — ByteDance Granite-TTM .npz outlier inspection
# =====================================================================

print_block('[B] ByteDance Granite-TTM .npz — where are the outliers?')

for h in ['h030', 'h060', 'h120']:
    npz_path = Path(P['results']) / 'bytedance' / h / 'foundation_models' / 'granite_ttm_perpoint.npz'
    print()
    print(f'-- bytedance {h} granite_ttm --')
    if not npz_path.exists():
        print('   .npz MISSING')
        continue
    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    print(f'   keys: {keys}')

    for key in ['y_true', 'y_pred_mean', 'y_pred_p50', 'y_naive']:
        if key not in npz.files:
            print(f'     {key:<14} (not present)')
            continue
        arr = np.asarray(npz[key], dtype=np.float64)
        n_nan = int(np.isnan(arr).sum())
        if arr.size == 0 or n_nan == arr.size:
            print(f'     {key:<14} shape={arr.shape} all-NaN')
            continue
        # robust stats
        finite = arr[np.isfinite(arr)]
        print(f'     {key:<14} shape={arr.shape} '
              f'mean={finite.mean():.2f} min={finite.min():.3f} '
              f'max={finite.max():.3f} '
              f'p1={np.percentile(finite, 1):.2f} p99={np.percentile(finite, 99):.2f} '
              f'n_nan={n_nan}')

    # for h060, find the worst predictions (cid, origin, y_true, y_pred_mean)
    y_true = np.asarray(npz['y_true'])
    y_pred = np.asarray(npz['y_pred_mean'])
    cids   = np.asarray(npz['container_ids']).astype(str)
    sq_err = (y_true - y_pred) ** 2
    finite_mask = np.isfinite(sq_err)
    if finite_mask.sum() > 0:
        worst_idx = np.argsort(sq_err[finite_mask])[-5:]
        finite_indices = np.where(finite_mask)[0]
        worst_actual_idx = finite_indices[worst_idx]
        print(f'   worst 5 (cid, y_true, y_pred_mean, sq_err):')
        for i in worst_actual_idx:
            print(f'     {cids[i]:<14} y_true={y_true[i]:8.2f} '
                  f'y_pred_mean={y_pred[i]:12.2f} sq_err={sq_err[i]:.2e}')


# =====================================================================
# SECTION C — Bitbrains per-VM XGBoost source CSV
# =====================================================================

print_block('[C] bitbrains_per_vm_results.csv schema + ID format')

per_vm_path = Path(P['results']) / 'bitbrains' / 'diagnostics' / 'bitbrains_per_vm_results.csv'
print(f'path: {per_vm_path}')
if per_vm_path.exists():
    df_vm = pd.read_csv(per_vm_path)
    print(f'   rows: {len(df_vm)}')
    print(f'   columns: {list(df_vm.columns)}')
    print(f'   head:')
    print(df_vm.head(5).to_string())
    print()
    print(f'   horizon values: {sorted(df_vm["horizon"].unique()) if "horizon" in df_vm.columns else "(no horizon col)"}')

    # ID format check
    id_col = 'vm_id' if 'vm_id' in df_vm.columns else None
    if id_col:
        ids = df_vm[id_col].astype(str).unique()
        print(f'   {id_col} format: first 5 = {list(ids[:5])}')
        print(f'   {id_col} unique count: {len(ids)}')
else:
    print('   MISSING')

# Now load chronos2 for bitbrains h030, get its container_id format
npz_bb_path = Path(P['results']) / 'bitbrains' / 'h030' / 'foundation_models' / 'chronos2_perpoint.npz'
print()
print(f'bitbrains h030 chronos2 container_ids format:')
print(f'   path: {npz_bb_path}')
if npz_bb_path.exists():
    npz_bb = np.load(npz_bb_path, allow_pickle=True)
    bb_cids = np.asarray(npz_bb['container_ids']).astype(str)
    uniq = np.unique(bb_cids)
    print(f'   unique cids: {len(uniq)}')
    print(f'   first 5: {list(uniq[:5])}')
    print(f'   last 5:  {list(uniq[-5:])}')
else:
    print('   MISSING')

# Also check bitbrains test.parquet container_id format
bb_test_pq = Path(P['thesis']) / 'data' / 'processed' / 'bitbrains' / 'test.parquet'
print()
print(f'bitbrains test.parquet container_id format:')
print(f'   path: {bb_test_pq}')
if bb_test_pq.exists():
    bb_test = pd.read_parquet(bb_test_pq, columns=['container_id'])
    bb_test_ids = bb_test['container_id'].astype(str).unique()
    print(f'   unique ids: {len(bb_test_ids)}')
    print(f'   first 5: {list(sorted(bb_test_ids)[:5])}')
    print(f'   last 5:  {list(sorted(bb_test_ids)[-5:])}')


# =====================================================================
# SECTION D — leaderboard_v1.csv: what does nnls_pool say per cell?
# =====================================================================

print_block('[D] leaderboard_v1.csv — nnls_pool / nnls_agg_method per cell')

lb_path = Path(P['foundation']) / 'leaderboard_v1.csv'
print(f'path: {lb_path}')
if lb_path.exists():
    lb = pd.read_csv(lb_path)
    print(f'   rows: {len(lb)}')
    print(f'   columns: {list(lb.columns)}')

    print('\n   full table (selected cols):')
    show = ['dataset', 'horizon', 'naive_r2', 'nnls_ensemble_r2',
            'chronos2_r2', 'timesfm_r2', 'granite_ttm_r2',
            'nnls_agg_method', 'nnls_pool']
    cols = [c for c in show if c in lb.columns]
    print(lb[cols].to_string())

    # Per-cell winner — what does the canonical leaderboard say?
    print('\n   per-cell winner (argmax across nnls / chronos2 / timesfm / granite_ttm):')
    rcols = ['nnls_ensemble_r2', 'chronos2_r2', 'timesfm_r2', 'granite_ttm_r2']
    rcols = [c for c in rcols if c in lb.columns]
    if rcols:
        lb['_winner'] = lb[rcols].idxmax(axis=1).str.replace('_r2', '').str.replace('_ensemble', '')
        print(lb[['dataset', 'horizon', '_winner'] + rcols].to_string())
        print('\n   tally:')
        print(lb['_winner'].value_counts().to_string())
else:
    print('   MISSING')


# =====================================================================
# SECTION E — Pooled R² sanity check (Alibaba NNLS h120)
# =====================================================================

print_block('[E] Pooled R² sanity check — Alibaba NNLS h120')

print('Canonical: comparison_table.csv reports Alibaba NNLS h120 R² = 0.7642 (CPU scale, pooled)')

# load chronos2 spine
ali_c2 = np.load(Path(P['results']) / 'alibaba' / 'h120' / 'foundation_models' / 'chronos2_perpoint.npz',
                 allow_pickle=True)
c2_cids    = np.asarray(ali_c2['container_ids']).astype(str)
c2_origins = np.asarray(ali_c2['origins']).astype(np.int64)
c2_ytrue   = np.asarray(ali_c2['y_true']).astype(np.float64)
c2_h_steps = int(ali_c2['h_steps'])

# build alignment to NNLS
ali_test = pd.read_parquet(Path(P['thesis']) / 'data' / 'processed' / 'alibaba' / 'test.parquet',
                            columns=['container_id', 'time_stamp'])
ali_test = ali_test.sort_values(['container_id', 'time_stamp']).reset_index(drop=True)
total = ali_test.groupby('container_id', sort=False).size()
valid = total - c2_h_steps
valid = valid[valid > 0]
starts = valid.cumsum().shift(fill_value=0).astype(np.int64)

start_arr = starts.reindex(c2_cids).values
nnls_idx  = start_arr.astype(np.int64) + c2_origins

y_true_full   = np.load(Path(P['results']) / 'alibaba' / 'h120' / 'predictions' / 'y_true.npy')
y_nnls_hetero = np.load(Path(P['results']) / 'alibaba' / 'h120' / 'predictions' / 'test_ensemble_hetero.npy')
y_nnls_v3     = np.load(Path(P['results']) / 'alibaba' / 'h120' / 'predictions' / 'test_ensemble_v3.npy')

yt   = y_true_full[nnls_idx]
yp_h = y_nnls_hetero[nnls_idx]
yp_v = y_nnls_v3[nnls_idx]

def pooled_r2(y_true, y_pred):
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return 1.0 - ss_res / ss_tot

print(f'\n   pooled R² on chronos2 K=20 spine ({len(yt)} points):')
print(f'     y_true mean / range: {yt.mean():.3f} / [{yt.min():.2f}, {yt.max():.2f}]')
print(f'     ')
print(f'     hetero (test_ensemble_hetero.npy):')
print(f'       y_pred mean / range:   {yp_h.mean():.3f} / [{yp_h.min():.2f}, {yp_h.max():.2f}]')
print(f'       pooled R²:             {pooled_r2(yt, yp_h):.4f}')
print(f'       expected canonical:    0.7642 (full-test scope; may differ on K=20 subset)')
print(f'     ')
print(f'     v3 (test_ensemble_v3.npy, residual scale):')
print(f'       y_pred mean / range:   {yp_v.mean():.3f} / [{yp_v.min():.2f}, {yp_v.max():.2f}]')
print(f'       pooled R²:             {pooled_r2(yt, yp_v):.4f}  (negative = residual scale mismatch)')


# =====================================================================
# SECTION F — Pooled R² for Bitbrains hetero (find the scale bug)
# =====================================================================

print_block('[F] Pooled R² test — Bitbrains hetero h030 (the broken cell)')

bb_c2 = np.load(Path(P['results']) / 'bitbrains' / 'h030' / 'foundation_models' / 'chronos2_perpoint.npz',
                allow_pickle=True)
bb_cids    = np.asarray(bb_c2['container_ids']).astype(str)
bb_origins = np.asarray(bb_c2['origins']).astype(np.int64)
bb_ytrue   = np.asarray(bb_c2['y_true']).astype(np.float64)
bb_h_steps = int(bb_c2['h_steps'])

bb_test = pd.read_parquet(Path(P['thesis']) / 'data' / 'processed' / 'bitbrains' / 'test.parquet',
                          columns=['container_id', 'time_stamp'])
bb_test = bb_test.sort_values(['container_id', 'time_stamp']).reset_index(drop=True)
bb_total = bb_test.groupby('container_id', sort=False).size()
bb_valid = bb_total - bb_h_steps
bb_valid = bb_valid[bb_valid > 0]
bb_starts = bb_valid.cumsum().shift(fill_value=0).astype(np.int64)

# Try matching IDs three ways
print(f'   chronos2 c_ids first 5: {list(np.unique(bb_cids)[:5])}')
print(f'   test.parquet ids first 5: {list(sorted(bb_starts.index)[:5])}')

# direct match
start_arr = bb_starts.reindex(bb_cids).values
direct_match = (~pd.isna(start_arr.astype(np.float64))).sum()
print(f'   direct match: {direct_match}/{len(bb_cids)}')

if direct_match < 0.5 * len(bb_cids):
    print('   need ID transformation')
else:
    nnls_idx_bb = start_arr.astype(np.int64) + bb_origins

    y_true_full   = np.load(Path(P['results']) / 'bitbrains' / 'h030' / 'predictions' / 'y_true.npy')
    bb_hetero     = np.load(Path(P['results']) / 'bitbrains' / 'h030' / 'predictions' / 'test_ensemble_hetero.npy')

    yt = y_true_full[nnls_idx_bb]
    yp = bb_hetero[nnls_idx_bb]
    print(f'\n   y_true (chronos2 spine, full test):')
    print(f'     mean={yt.mean():.3f} range=[{yt.min():.2f}, {yt.max():.2f}]')
    print(f'\n   test_ensemble_hetero.npy (aligned):')
    print(f'     mean={yp.mean():.3f} range=[{yp.min():.2f}, {yp.max():.2f}]')
    print(f'\n   first 10 (y_true, y_pred_hetero, y_naive):')
    bb_naive = np.load(Path(P['results']) / 'bitbrains' / 'h030' / 'predictions' / 'y_naive.npy')
    yn = bb_naive[nnls_idx_bb]
    for i in range(10):
        print(f'     [{i:3d}] cid={bb_cids[i]:<10} origin={bb_origins[i]:<4} '
              f'y_true={yt[i]:7.2f}  y_pred={yp[i]:7.2f}  y_naive={yn[i]:7.2f}')

    print(f'\n   pooled R² (hetero):  {pooled_r2(yt, yp):.4f}')
    print(f'   pooled R² (naive):   {pooled_r2(yt, yn):.4f}')

    # Also check other ensemble candidates
    for fname in ['test_ensemble_homo.npy', 'test_ensemble_v2.npy', 'test_ensemble_v3.npy',
                  'test_xgb.npy', 'test_et.npy']:
        p = Path(P['results']) / 'bitbrains' / 'h030' / 'predictions' / fname
        if not p.exists():
            print(f'   {fname:<28} MISSING')
            continue
        try:
            arr = np.load(p, allow_pickle=True)
            if arr.dtype.kind != 'f':
                continue
            yp2 = arr[nnls_idx_bb]
            print(f'   {fname:<28} mean={yp2.mean():.3f}  range=[{yp2.min():.2f}, {yp2.max():.2f}]  '
                  f'pooled R² vs y_true = {pooled_r2(yt, yp2):.4f}')
        except Exception as e:
            print(f'   {fname:<28} read error: {e}')


print()
print('=' * 72)
print('END OF DIAGNOSTIC')
print('=' * 72)
