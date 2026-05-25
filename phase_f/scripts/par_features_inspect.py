"""
par_features_inspect.py — hand inspection of par_features.parquet.

Read-only. Run from phase_f/scripts/. Produces 8 sections for visual review.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from _paths import P


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda v: f'{v:8.4f}')


def block(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


# ----- load -----

feat_path = Path(P['data']) / 'par_features.parquet'
print(f'loading: {feat_path}')
df = pd.read_parquet(feat_path)
print(f'   rows = {len(df)}, cols = {df.shape[1]}')


# ----- [A] shape + columns -----

block('[A] Schema')
print('Columns:')
for c in df.columns:
    n_nan = int(df[c].isna().sum())
    dtype = df[c].dtype
    print(f'  {c:<22} {str(dtype):<10} n_nan = {n_nan}')


# ----- [B] per-dataset percentile table -----

block('[B] Per-dataset percentiles for key features')
features_to_describe = ['n_test_points', 'mean', 'std', 'cv',
                        'DFA', 'LZC', 'SampEn', 'WPE']
for f in features_to_describe:
    if f not in df.columns:
        continue
    print(f'\n  {f}:')
    desc = df.groupby('dataset')[f].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).round(4)
    print(desc.to_string())


# ----- [C] NaN audit -----

block('[C] NaN pattern analysis')

# Mark which rows have how many NaN
feature_cols = [c for c in df.columns if c not in ['container_id', 'dataset', 'n_test_points']]
df['_n_nan'] = df[feature_cols].isna().sum(axis=1)

print('Distribution of n_nan per series (across all 28 feature cols):')
print(df['_n_nan'].value_counts().sort_index().to_string())

print('\nPer-dataset breakdown of n_nan:')
nan_by_ds = df.groupby('dataset')['_n_nan'].agg(['mean', 'median', 'max', 'count'])
print(nan_by_ds.round(2).to_string())

print('\nSeries with > 5 NaN features — what do they look like?')
high_nan = df[df['_n_nan'] > 5].copy()
print(f'  count: {len(high_nan)}')
if len(high_nan) > 0:
    print(f'  per dataset: {high_nan["dataset"].value_counts().to_dict()}')
    print(f'\n  n_test_points percentiles (these are the suspect series):')
    print(high_nan['n_test_points'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(2).to_string())
    print(f'\n  mean percentiles (are they zero-mean / idle?):')
    print(high_nan['mean'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4).to_string())
    print(f'\n  std percentiles (are they constant?):')
    print(high_nan['std'].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(4).to_string())
    print(f'\n  Sample of 10 such series:')
    print(high_nan[['container_id', 'dataset', 'n_test_points', 'mean', 'std',
                    'WPE', 'LZC', '_n_nan']].head(10).to_string(index=False))


# ----- [D] Sample rows -----

block('[D] Sample rows — head per dataset (basic features only)')
view_cols = ['container_id', 'dataset', 'n_test_points', 'mean', 'std', 'cv',
             'DFA', 'LZC', 'SampEn', 'WPE']
view_cols = [c for c in view_cols if c in df.columns]
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    print(f'\n  --- {ds} head ---')
    print(df[df['dataset'] == ds][view_cols].head(5).to_string(index=False))


# ----- [E] WPE extremes -----

block('[E] WPE extremes — does WPE pass the eye test?')
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    ds_df = df[df['dataset'] == ds].dropna(subset=['WPE'])
    print(f'\n  --- {ds} ---')
    lowest = ds_df.nsmallest(3, 'WPE')[['container_id', 'WPE', 'mean', 'std', 'cv', 'DFA', 'LZC']]
    highest = ds_df.nlargest(3, 'WPE')[['container_id', 'WPE', 'mean', 'std', 'cv', 'DFA', 'LZC']]
    print(f'  lowest WPE (most structured ordinally):')
    print(lowest.to_string(index=False))
    print(f'  highest WPE (most random ordinally):')
    print(highest.to_string(index=False))


# ----- [F] Inter-feature correlations within each dataset -----

block('[F] Inter-feature correlations (per dataset) — orthogonality check')
key_features = ['cv', 'DFA', 'LZC', 'SampEn', 'WPE']
key_features = [c for c in key_features if c in df.columns]
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    print(f'\n  --- {ds} ---')
    sub = df[df['dataset'] == ds][key_features].dropna()
    if len(sub) < 10:
        print(f'    too few rows: {len(sub)}')
        continue
    corr = sub.corr().round(3)
    print(corr.to_string())


# ----- [G] Cross-dataset WPE vs ACF@24h sanity -----

block('[G] Cross-dataset WPE story — does it match macro-ACF inversion claim?')
print("""
Memory says:
  ACF@24h:   bytedance(0.489) > alibaba(0.316) > bitbrains(0.116)
  ML benefit-by-ACF@24h holds at h60/h120 but inverts at h30 (bitbrains > alibaba)

WPE story from this run:
  WPE median: bytedance(0.94) > bitbrains(0.70) > alibaba(0.54)
  (inverted from ACF@24h, except bitbrains/alibaba swap)
""")
wpe_med = df.groupby('dataset')['WPE'].median()
sampen_med = df.groupby('dataset')['SampEn'].median()
lzc_med = df.groupby('dataset')['LZC'].median()
acf_canon = {'alibaba': 0.316, 'bitbrains': 0.116, 'bytedance': 0.489}

print('   Dataset       n_series   WPE_med   SampEn_med   LZC_med   ACF@24h_canon')
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    n = (df['dataset'] == ds).sum()
    print(f'   {ds:<13} {n:8d}   {wpe_med[ds]:.4f}   {sampen_med[ds]:.4f}      {lzc_med[ds]:.4f}    {acf_canon[ds]:.3f}')

# Rank correlation across the 3 dataset medians
print('\n  rank correlation (n=3 datasets — descriptive only):')
ds_order = ['alibaba', 'bitbrains', 'bytedance']
wpe_ranks = pd.Series(wpe_med[ds_order]).rank().values
acf_ranks = pd.Series([acf_canon[d] for d in ds_order]).rank().values
sampen_ranks = pd.Series(sampen_med[ds_order]).rank().values
lzc_ranks = pd.Series(lzc_med[ds_order]).rank().values
print(f'    WPE ranks      : {wpe_ranks.tolist()}')
print(f'    SampEn ranks   : {sampen_ranks.tolist()}')
print(f'    LZC ranks      : {lzc_ranks.tolist()}')
print(f'    ACF@24h ranks  : {acf_ranks.tolist()}')


# ----- [H] catch22 feature variance — which features actually vary? -----

block('[H] catch22 feature std per dataset — which features are informative?')
c22_cols = sorted([c for c in df.columns if c.startswith('catch22_')],
                  key=lambda x: int(x.split('_')[1]))
print(f'   ({len(c22_cols)} catch22 features)')
c22_stds = df.groupby('dataset')[c22_cols].std().T
c22_stds.columns = [f'std_{c}' for c in c22_stds.columns]
print(c22_stds.round(3).to_string())

print('\n  catch22 features with constant value (std=0) per dataset:')
for ds in ['alibaba', 'bitbrains', 'bytedance']:
    constant_cols = [c for c in c22_cols if df[df['dataset'] == ds][c].dropna().std() == 0]
    print(f'    {ds:<12}: {len(constant_cols)} constant features '
          f'{constant_cols[:5] if constant_cols else "(none)"}')

print()
print('=' * 78)
print('END OF INSPECTION')
print('=' * 78)
