"""
par_inventory_predictions.py — inspect per-point prediction file structure.

V2: allow_pickle=True for .npz files because they contain object-dtype
arrays (likely container_id strings).

Goal: figure out (1) what arrays/keys are inside the .npz foundation model files,
(2) shapes and dtypes, (3) how container_id alignment is preserved, (4) whether
y_true.npy aligns with foundation model arrays.

Read-only. No mutation. Junior style — verbose prints, sequential, no abstractions.
Run from phase_f/scripts/.
"""

import sys
import json
from pathlib import Path
import numpy as np

# bring in the path resolver
from _paths import P


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DATASETS = {
    'alibaba':   ['h010', 'h030', 'h060', 'h120'],
    'bitbrains': ['h010', 'h030', 'h060', 'h120'],
    'bytedance': ['h030', 'h060', 'h120'],  # no h10 by design
}

FOUNDATION_MODELS = ['chronos2', 'timesfm', 'granite_ttm', 'toto']

PREDICTION_FILES = {
    'y_true':           'predictions/y_true.npy',
    'y_naive':          'predictions/y_naive.npy',
    'test_naive':       'predictions/test_naive.npy',
    'ensemble_v3':      'predictions/test_ensemble_v3.npy',
    'ensemble_hetero':  'predictions/test_ensemble_hetero.npy',
    'ensemble_v2':      'predictions/test_ensemble_v2.npy',
}

NPZ_FILES_TPL = 'foundation_models/{model}_perpoint.npz'


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _safe_first_few(arr, n=5):
    """Get first n entries safely regardless of dtype (object → str)."""
    flat = arr.flatten()
    if flat.dtype == object:
        return [str(x) for x in flat[:n]]
    if flat.dtype.kind in 'fui':
        return flat[:n].tolist()
    # bytes / void / other — cast to str
    return [str(x) for x in flat[:n]]


def inspect_npy(path: Path) -> dict:
    """Read .npy and return shape, dtype, simple stats."""
    try:
        # allow_pickle=True is safe — these are our own files
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        return {'error': str(e)}
    info = {
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'first5': _safe_first_few(arr, 5),
        'last5':  _safe_first_few(arr.flatten()[-5:][::-1], 5)[::-1],
    }
    if arr.dtype.kind == 'f':
        info['n_nan']  = int(np.isnan(arr).sum())
        info['n_inf']  = int(np.isinf(arr).sum())
        info['min']    = float(np.nanmin(arr)) if arr.size else None
        info['max']    = float(np.nanmax(arr)) if arr.size else None
        info['mean']   = float(np.nanmean(arr)) if arr.size else None
    return info


def inspect_npz(path: Path) -> dict:
    """Read .npz and return keys + per-key shape/dtype."""
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        return {'error': str(e)}
    info = {'keys': list(npz.keys()), 'per_key': {}}
    for k in npz.keys():
        try:
            arr = npz[k]
        except Exception as e:
            info['per_key'][k] = {'error': str(e)}
            continue
        kinfo = {
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'first5': _safe_first_few(arr, 5),
        }
        if arr.dtype.kind == 'f' and arr.size > 0:
            kinfo['n_nan'] = int(np.isnan(arr).sum())
            kinfo['min']   = float(np.nanmin(arr))
            kinfo['max']   = float(np.nanmax(arr))
        info['per_key'][k] = kinfo
    return info


def print_block(title: str):
    print()
    print('=' * 70)
    print(title)
    print('=' * 70)


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------

print_block('PAR PATH 2 — PREDICTION FILE INVENTORY (v2)')

results_root = Path(P['results'])
print(f"results root: {results_root}")

# 1) Foundation model .npz inspection — full sample on Alibaba h120
print_block('[A] Foundation model .npz structure (Alibaba h120 sample)')

for model in FOUNDATION_MODELS:
    npz_path = results_root / 'alibaba' / 'h120' / 'foundation_models' / f'{model}_perpoint.npz'
    print()
    print(f'-- {model} --')
    print(f'   path: {npz_path}')
    if not npz_path.exists():
        print('   MISSING')
        continue
    info = inspect_npz(npz_path)
    if 'error' in info:
        print(f'   ERROR: {info["error"]}')
        continue
    print(f'   keys: {info["keys"]}')
    for k, kinfo in info['per_key'].items():
        if 'error' in kinfo:
            print(f'     [{k}]  ERROR: {kinfo["error"]}')
            continue
        print(f'     [{k}]  shape={kinfo["shape"]}  dtype={kinfo["dtype"]}')
        print(f'           first5={kinfo["first5"]}')
        if 'n_nan' in kinfo:
            print(f'           min={kinfo["min"]:.4f}  max={kinfo["max"]:.4f}  n_nan={kinfo["n_nan"]}')

# 2) Ground truth + ensemble .npy inspection (Alibaba h120 sample)
print_block('[B] Ground truth + ensemble .npy structure (Alibaba h120 sample)')

for label, relpath in PREDICTION_FILES.items():
    p = results_root / 'alibaba' / 'h120' / relpath
    print()
    print(f'-- {label} --')
    print(f'   path: {p}')
    if not p.exists():
        print('   MISSING')
        continue
    info = inspect_npy(p)
    if 'error' in info:
        print(f'   ERROR: {info["error"]}')
        continue
    print(f'   shape={info["shape"]}  dtype={info["dtype"]}')
    print(f'   first5={info["first5"]}')
    print(f'   last5={info["last5"]}')
    if 'n_nan' in info:
        print(f'   min={info["min"]:.4f}  max={info["max"]:.4f}  mean={info["mean"]:.4f}  n_nan={info["n_nan"]}  n_inf={info["n_inf"]}')

# 3) Container-id alignment metadata search
print_block('[C] Container-id alignment search (foundation_comparison/)')

fc_root = Path(P['foundation'])
candidates = [
    'oof_all_new.parquet',
    'leaderboard_v1.csv',
    'chronos2_leaderboard.csv',
]
print()
for c in candidates:
    p = fc_root / c
    if p.exists():
        print(f'   FOUND  {p}')
        if p.suffix == '.csv':
            try:
                with open(p) as fh:
                    header = fh.readline().strip()
                print(f'          header: {header[:200]}')
            except Exception as e:
                print(f'          could not read header: {e}')
        elif p.suffix == '.parquet':
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                print(f'          shape: {df.shape}')
                print(f'          columns: {list(df.columns)[:30]}')
                print(f'          head:')
                print(df.head(2).to_string())
            except Exception as e:
                print(f'          could not read parquet: {e}')
    else:
        print(f'   MISSING  {p}')

# 4) Existence matrix — dataset × horizon × foundation model
print_block('[D] Existence matrix — foundation model .npz')

print()
print(f'  {"dataset":<10} {"horizon":<7}  ' + '  '.join(f'{m:<12}' for m in FOUNDATION_MODELS))
print(f'  {"-"*10} {"-"*7}  ' + '  '.join('-' * 12 for _ in FOUNDATION_MODELS))

missing_count = 0
expected_count = 0
for ds, horizons in DATASETS.items():
    for h in horizons:
        row = f'  {ds:<10} {h:<7}  '
        for model in FOUNDATION_MODELS:
            npz_path = results_root / ds / h / 'foundation_models' / f'{model}_perpoint.npz'
            expected_count += 1
            if npz_path.exists():
                cell = 'PRESENT'
            else:
                cell = 'MISSING'
                missing_count += 1
            row += f'{cell:<12}  '
        print(row)

print()
print(f'  expected files: {expected_count}')
print(f'  missing:        {missing_count}')

# 5) Existence matrix — predictions/ siblings
print_block('[E] Existence matrix — predictions/ files')

key_pred_files = ['y_true.npy', 'test_naive.npy', 'test_ensemble_v3.npy',
                  'test_ensemble_hetero.npy']

print()
print(f'  {"dataset":<10} {"horizon":<7}  ' + '  '.join(f'{f:<22}' for f in key_pred_files))
print(f'  {"-"*10} {"-"*7}  ' + '  '.join('-' * 22 for _ in key_pred_files))

for ds, horizons in DATASETS.items():
    for h in horizons:
        row = f'  {ds:<10} {h:<7}  '
        for fname in key_pred_files:
            p = results_root / ds / h / 'predictions' / fname
            cell = 'PRESENT' if p.exists() else 'MISSING'
            row += f'{cell:<22}  '
        print(row)

# 6) BCF-pairs canonical-version check
print_block('[F] BCF pairs canonical version check')

bcf_v1 = Path(P['bcf']) / 'bcf_pairs.csv'
bcf_v2 = Path(P['bcf_v2']) / 'bcf_pairs_v2.csv'
print()
print(f'bcf_pairs.csv (v1):  {"EXISTS" if bcf_v1.exists() else "MISSING"}  {bcf_v1}')
print(f'bcf_pairs_v2.csv:    {"EXISTS" if bcf_v2.exists() else "MISSING"}  {bcf_v2}')

for label, p in [('v1', bcf_v1), ('v2', bcf_v2)]:
    if not p.exists():
        continue
    try:
        with open(p) as fh:
            header = fh.readline().strip()
        n_lines = sum(1 for _ in open(p)) - 1
        print(f'   {label} header: {header}')
        print(f'   {label} rows:   {n_lines}')
    except Exception as e:
        print(f'   {label} read error: {e}')

print()
print('=' * 70)
print('END OF INVENTORY')
print('=' * 70)