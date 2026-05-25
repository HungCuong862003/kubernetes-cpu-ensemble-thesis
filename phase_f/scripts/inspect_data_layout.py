"""
inspect_data_layout.py — discover what data files exist and their schemas.

Run from phase_f/scripts/. Prints everything we need to patch f3_baseline_chronos2.py.
"""

from pathlib import Path
import pandas as pd

ROOT = Path('/workspace/kubernetes-cpu-ensemble-thesis')

print('=' * 70)
print('Data layout inspector')
print('=' * 70)


# ---- [1] Explore data/processed/ ----
print('\n[1] data/processed/ — looking for raw time-series files')
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    print(f'\n  --- {ds} ---')
    proc_dir = ROOT / 'data' / 'processed' / ds
    if not proc_dir.exists():
        # Also try data/<dataset>
        proc_dir = ROOT / 'data' / ds
        if not proc_dir.exists():
            print(f'    no dir at data/processed/{ds} or data/{ds}')
            continue
    for f in sorted(proc_dir.iterdir()):
        try:
            sz = f.stat().st_size / 1e6
        except FileNotFoundError:
            continue
        print(f'    {f.name}: {sz:.1f} MB')


# ---- [2] Inspect test_spine.parquet schema ----
print('\n[2] test_spine.parquet schemas')
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    h_dirs = {'alibaba': ['h010', 'h030', 'h060', 'h120'],
              'bitbrains': ['h010', 'h030', 'h060', 'h120'],
              'bytedance': ['h030', 'h060', 'h120']}[ds]
    # Just inspect one horizon per dataset
    h = h_dirs[0]
    path = ROOT / 'results' / ds / h / 'predictions' / 'test_spine.parquet'
    print(f'\n  --- {ds} {h} ({path.name}) ---')
    if not path.exists():
        print(f'    MISSING: {path}')
        continue
    df = pd.read_parquet(path)
    print(f'    shape:   {df.shape}')
    print(f'    columns: {df.columns.tolist()}')
    print(f'    dtypes:')
    for c, dt in df.dtypes.items():
        print(f'      {c}: {dt}')
    print(f'    first 3 rows:')
    print(df.head(3).to_string())
    if len(df) > 5:
        print(f'    last 2 rows:')
        print(df.tail(2).to_string())


# ---- [3] Inspect spine metadata ----
print('\n[3] spine metadata files (one per dataset)')
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    path = ROOT / 'results' / ds / 'spines' / f'spine_{ds}_60min.parquet'
    if ds == 'bytedance':
        # ByteDance might have different available horizons
        for cand in ['60min', '120min', '30min']:
            p = ROOT / 'results' / ds / 'spines' / f'spine_{ds}_{cand}.parquet'
            if p.exists():
                path = p
                break
    print(f'\n  --- {ds} ({path.name}) ---')
    if not path.exists():
        print(f'    MISSING: {path}')
        continue
    df = pd.read_parquet(path)
    print(f'    shape:   {df.shape}')
    print(f'    columns: {df.columns.tolist()}')
    print(f'    first 3 rows:')
    print(df.head(3).to_string())


# ---- [4] Look for raw CSVs (legacy) ----
print('\n[4] Raw CSV files in root and common locations')
for pattern in ['test_*.csv', 'train_*.csv', 'val_*.csv', '*alibaba*.csv', '*bitbrains*.csv', '*bytedance*.csv', 'iaas*.csv']:
    matches = list(ROOT.glob(pattern))
    if matches:
        print(f'\n  pattern "{pattern}":')
        for m in matches[:5]:
            print(f'    {m.relative_to(ROOT)} ({m.stat().st_size / 1e6:.1f} MB)')


# ---- [5] List all top-level dirs in data/ ----
print('\n[5] Top-level layout of data/')
data_dir = ROOT / 'data'
if data_dir.exists():
    for f in sorted(data_dir.iterdir()):
        kind = 'DIR' if f.is_dir() else 'file'
        print(f'  [{kind}] {f.name}')


# ---- [6] List _paths.py expected keys ----
print('\n[6] _paths.py P[] dictionary keys (from prior PAR scripts):')
print("  Expected: ['scripts', 'phase_f', 'thesis', 'data', 'results', 'src', 'reports',")
print("            'bcf', 'bcf_v2', 'foundation']")
print('  P["data"] = phase_f/data/   (NOT raw data; F-task outputs)')
print('  P["results"] = results/    (per-horizon test_spine.parquet lives here)')
print('  No P["raw_data"] key for raw time series.')
