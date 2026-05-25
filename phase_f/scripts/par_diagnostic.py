"""
par_diagnostic.py — PAR Step 0 diagnostic.

Checks:
  1. Path resolution via _paths.P
  2. Library availability + versions
  3. Raw time-series data locations
  4. Per-series performance file existence
  5. Summary table

Run from phase_f/scripts/. Junior programmer style — verbose, sequential prints.
"""

import sys
from pathlib import Path
import importlib

print("=" * 70)
print("PAR STEP 0 DIAGNOSTIC")
print("=" * 70)
print()

# -----------------------------------------------------------------------
# 1. Path resolution
# -----------------------------------------------------------------------
print("[1/5] Path resolution via _paths.P")
print("-" * 70)

try:
    from _paths import P
    print("  imported _paths.P OK")
except Exception as e:
    print(f"  FATAL — could not import _paths: {e}")
    print("  Check that phase_f/scripts/_paths.py exists and the working")
    print("  directory is phase_f/scripts/.")
    sys.exit(1)

# Path keys we expect — exact set from F1 work
expected_keys = ['thesis', 'phase_f', 'data', 'results', 'bcf',
                 'bcf_v2', 'foundation', 'src', 'reports']

print()
print(f"  {'KEY':<14} {'STATUS':<10} PATH")
print(f"  {'-'*14} {'-'*10} {'-'*40}")
for key in expected_keys:
    if key in P:
        path = Path(P[key])
        status = "EXISTS" if path.exists() else "MISSING"
        print(f"  {key:<14} {status:<10} {path}")
    else:
        print(f"  {key:<14} {'NO_KEY':<10} (not in P)")

# Also check phase_f/data and phase_f/scripts directly
extra_paths = {
    'phase_f/data':    Path(P['phase_f']) / 'data',
    'phase_f/scripts': Path(P['phase_f']) / 'scripts',
}
print()
print("  Extra phase_f subdirs:")
for label, path in extra_paths.items():
    status = "EXISTS" if path.exists() else "MISSING"
    print(f"  {label:<20} {status:<10} {path}")

print()

# -----------------------------------------------------------------------
# 2. Library availability + versions
# -----------------------------------------------------------------------
print("[2/5] Library availability")
print("-" * 70)

# (name, attribute_for_version_or_None_to_try_default)
libs_to_check = [
    'antropy',
    'pycatch22',
    'ordpy',
    'nolds',
    'statsmodels',
    'xgboost',
    'sklearn',
    'numpy',
    'pandas',
    'scipy',
]

lib_status = {}
print()
print(f"  {'LIBRARY':<14} {'STATUS':<10} VERSION")
print(f"  {'-'*14} {'-'*10} {'-'*20}")
for lib in libs_to_check:
    try:
        mod = importlib.import_module(lib)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  {lib:<14} {'OK':<10} {version}")
        lib_status[lib] = version
    except Exception:
        print(f"  {lib:<14} {'MISSING':<10} —")
        lib_status[lib] = None

# sklearn must be >= 1.5
print()
sk_ver = lib_status.get('sklearn')
if sk_ver:
    # crude major.minor parse
    try:
        major, minor = sk_ver.split('.')[:2]
        ok = (int(major), int(minor)) >= (1, 5)
        print(f"  sklearn version check (>= 1.5): {'OK' if ok else 'TOO_OLD'} ({sk_ver})")
    except Exception:
        print(f"  sklearn version check: could not parse ({sk_ver})")
else:
    print("  sklearn version check: MISSING")

print()

# -----------------------------------------------------------------------
# 3. Raw time-series data locations
# -----------------------------------------------------------------------
print("[3/5] Raw time-series data locations")
print("-" * 70)

# Candidate directories to look in
thesis_root = Path(P['thesis'])
results_root = Path(P['results'])

# Per-dataset search
dataset_search = {
    'Alibaba': [
        thesis_root / 'data' / 'alibaba',
        results_root / 'alibaba',
        thesis_root / 'data',
    ],
    'Bitbrains': [
        thesis_root / 'data' / 'bitbrains',
        results_root / 'bitbrains',
        thesis_root / 'data',
    ],
    'ByteDance': [
        thesis_root / 'data' / 'bytedance',
        results_root / 'bytedance',
        thesis_root / 'data',
    ],
}

raw_data_summary = {}
for ds, dirs in dataset_search.items():
    print()
    print(f"  Dataset: {ds}")
    found_any = False
    for d in dirs:
        if not d.exists():
            print(f"    {d}  MISSING")
            continue
        # count parquets and csvs in this dir (non-recursive shallow)
        parquets = list(d.glob('*.parquet'))
        csvs = list(d.glob('*.csv'))
        # also one level down
        parquets_sub = list(d.glob('*/*.parquet'))
        csvs_sub = list(d.glob('*/*.csv'))
        total_pq = len(parquets) + len(parquets_sub)
        total_csv = len(csvs) + len(csvs_sub)
        print(f"    {d}  EXISTS  parquet={total_pq}  csv={total_csv}")
        # show a few example filenames if any
        if total_pq > 0:
            examples = [p.name for p in (parquets + parquets_sub)[:3]]
            print(f"      example parquets: {examples}")
        if total_csv > 0:
            examples = [p.name for p in (csvs + csvs_sub)[:3]]
            print(f"      example csvs: {examples}")
        if total_pq + total_csv > 0:
            found_any = True
            raw_data_summary[ds] = (d, total_pq, total_csv)
            break  # first hit wins for this dataset
    if not found_any:
        raw_data_summary[ds] = None

# Also do a project-wide glob for *<dataset>* parquets/csvs as a backstop
print()
print("  Project-wide backstop search (top-level + 1 level deep):")
for ds_keyword in ['alibaba', 'bitbrains', 'bytedance']:
    matches = []
    for ext in ['parquet', 'csv']:
        matches.extend(thesis_root.glob(f'*{ds_keyword}*.{ext}'))
        matches.extend(thesis_root.glob(f'*/*{ds_keyword}*.{ext}'))
    matches = sorted({m for m in matches})
    print(f"    *{ds_keyword}*  matches={len(matches)}")
    for m in matches[:5]:
        try:
            rel = m.relative_to(thesis_root)
        except Exception:
            rel = m
        print(f"      {rel}")
    if len(matches) > 5:
        print(f"      ... +{len(matches) - 5} more")

print()

# -----------------------------------------------------------------------
# 4. Per-series performance files
# -----------------------------------------------------------------------
print("[4/5] Per-series performance files (for labels)")
print("-" * 70)

# Files known/expected to exist
per_series_candidates = {
    'Bitbrains per-VM R² (known)': [
        thesis_root / 'bitbrains_per_vm_results.csv',
        thesis_root / 'bitbrains_per_vm.csv',
        results_root / 'bitbrains_per_vm_results.csv',
        results_root / 'bitbrains_per_vm.csv',
    ],
    'Alibaba per-container R²': [
        thesis_root / 'per_container_r2.csv',
        thesis_root / 'alibaba_per_container_r2.csv',
        results_root / 'per_container_r2.csv',
        results_root / 'alibaba' / 'per_container_r2.csv',
    ],
    'ByteDance per-instance R²': [
        thesis_root / 'bytedance_per_instance_results.csv',
        thesis_root / 'bytedance_per_instance.csv',
        results_root / 'bytedance_per_instance.csv',
    ],
}

# Also do a wildcard glob
print()
for label, paths in per_series_candidates.items():
    print(f"  {label}:")
    found = False
    for p in paths:
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_csv(p, nrows=0)
                cols = list(df.columns)
                n_rows = sum(1 for _ in open(p)) - 1  # cheap rowcount
                print(f"    FOUND  {p}  rows={n_rows}  cols={cols[:10]}")
                found = True
                break
            except Exception as e:
                print(f"    FOUND but unreadable  {p}  ({e})")
                found = True
                break
    if not found:
        print(f"    MISSING (none of the candidate paths exist)")
    print()

# Wildcard glob backstop
print("  Wildcard glob backstop (*per_container*, *per_vm*, *per_instance*, *per_series*):")
patterns = ['*per_container*', '*per_vm*', '*per_instance*', '*per_series*']
for pat in patterns:
    matches = []
    matches.extend(thesis_root.glob(pat))
    matches.extend(thesis_root.glob(f'*/{pat}'))
    matches.extend(results_root.glob(pat))
    matches = sorted({m for m in matches if m.is_file()})
    print(f"    {pat:<22}  matches={len(matches)}")
    for m in matches[:5]:
        try:
            rel = m.relative_to(thesis_root)
        except Exception:
            rel = m
        print(f"      {rel}")

print()

# -----------------------------------------------------------------------
# 5. Summary table
# -----------------------------------------------------------------------
print("[5/5] SUMMARY TABLE")
print("-" * 70)
print()
print(f"  {'ITEM':<32} {'STATUS':<24}")
print(f"  {'-'*32} {'-'*24}")

# Libraries
for lib in ['antropy', 'pycatch22', 'ordpy', 'nolds', 'statsmodels',
            'xgboost', 'sklearn']:
    v = lib_status.get(lib)
    label = f"library: {lib}"
    if v:
        print(f"  {label:<32} v{v}")
    else:
        print(f"  {label:<32} MISSING")

# Raw data
for ds, info in raw_data_summary.items():
    label = f"raw data: {ds}"
    if info is None:
        print(f"  {label:<32} MISSING")
    else:
        d, npq, ncsv = info
        print(f"  {label:<32} FOUND (pq={npq} csv={ncsv})")

# Per-series perf files — re-check
ps_status = {}
for label, paths in per_series_candidates.items():
    ps_status[label] = any(p.exists() for p in paths)

for label, ok in ps_status.items():
    short = label.split(' ')[0] + ' ' + label.split(' ')[1] if len(label.split(' ')) > 1 else label
    print(f"  per-series: {short[:20]:<20} {'FOUND' if ok else 'MISSING'}")

print()
print("=" * 70)
print("END OF DIAGNOSTIC")
print("=" * 70)
