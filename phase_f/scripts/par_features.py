"""
par_features.py — per-series predictability/complexity features for PAR.

For each (dataset, container_id) pair, compute 26 features on the full
test-parquet time series:

  - catch22                : 22 canonical TSC features (pycatch22)
  - DFA                    : Detrended Fluctuation Analysis exponent (nolds)
  - LZC                    : Lempel-Ziv complexity, normalised (antropy)
  - SampEn                 : Sample entropy (antropy, m=2, r=0.2*std)
  - WPE                    : Weighted Permutation Entropy, Fadlallah variant (manual)

Plus 3 basic stats:
  - n_test_points, mean, std, cv

Scope: test parquet only (apples-to-apples with PAR labels at chronos2 spine).
Only containers present in any chronos2 K=20/K=50 spine are processed.

Output: phase_f/data/par_features.parquet

Run from phase_f/scripts/.
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycatch22
import nolds
import antropy

from _paths import P


# -------------------------------- config --------------------------------

DATASETS = ['alibaba', 'bitbrains', 'bytedance']
HORIZONS = ['h010', 'h030', 'h060', 'h120']

WPE_M   = 4    # embedding dimension for Fadlallah WPE
WPE_TAU = 1    # time delay

# Minimum series length for full feature computation
MIN_LEN = 50

# Checkpoint
CHECKPOINT_EVERY = 500

OUT_PARQUET = Path(P['data']) / 'par_features.parquet'
OUT_CSV     = Path(P['data']) / 'par_features.csv'


# -------------------------------- helpers --------------------------------

def detect_id_col(df):
    for c in ['container_id', 'vm_id', 'instance_id', 'machine_id']:
        if c in df.columns:
            return c
    raise RuntimeError(f"No ID column among {list(df.columns)[:8]}")


def compute_wpe_fadlallah(x, m=4, tau=1):
    """
    Weighted Permutation Entropy, Fadlallah variant.

    Reference: Fadlallah et al. (2013), "Weighted-permutation entropy: a complexity
    measure for time series incorporating amplitude information", Phys Rev E.

    Each ordinal pattern is weighted by the variance of the underlying segment,
    so high-amplitude regions contribute more than low-amplitude regions.

    Returns normalised WPE in [0, 1] = H / log(m!).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    eff_len = n - (m - 1) * tau
    if eff_len < 2:
        return np.nan

    # build (eff_len, m) segment matrix
    segs = np.zeros((eff_len, m), dtype=np.float64)
    for j in range(m):
        segs[:, j] = x[j * tau : j * tau + eff_len]

    # weights = variance of each segment
    weights = segs.var(axis=1)
    total_w = weights.sum()
    if total_w == 0:
        return 0.0

    # ordinal pattern of each segment: rank-of-rank → tuple
    # use argsort twice trick
    ranks = np.argsort(np.argsort(segs, axis=1), axis=1)

    # encode each pattern as a single integer (positional encoding)
    # since m is small (4), m! = 24 unique patterns at most
    # use a base-m polynomial hash
    codes = np.zeros(eff_len, dtype=np.int64)
    for j in range(m):
        codes = codes * m + ranks[:, j]

    # group weights by code
    df = pd.DataFrame({'code': codes, 'w': weights})
    pw = df.groupby('code', sort=False)['w'].sum()
    probs = pw.values / total_w
    probs = probs[probs > 0]

    H = -np.sum(probs * np.log(probs))
    H_max = math.log(math.factorial(m))
    return float(H / H_max)


def compute_features_one_series(x):
    """
    Compute all features for one time series. Returns dict.

    x : 1D numpy array of float
    """
    out = {}
    n = len(x)
    out['n_test_points'] = n

    if n == 0:
        out['mean'] = np.nan
        out['std']  = np.nan
        out['cv']   = np.nan
    else:
        m = float(x.mean())
        s = float(x.std(ddof=0))
        out['mean'] = m
        out['std']  = s
        out['cv']   = (s / m) if (m != 0 and not np.isnan(m)) else np.nan

    # too short → all NaN for complex features
    if n < MIN_LEN:
        out['DFA']    = np.nan
        out['LZC']    = np.nan
        out['SampEn'] = np.nan
        out['WPE']    = np.nan
        for i in range(22):
            out[f'catch22_{i+1}'] = np.nan
        return out

    # DFA (Hurst-like long-range exponent)
    try:
        out['DFA'] = float(nolds.dfa(x))
    except Exception:
        out['DFA'] = np.nan

    # LZC — antropy expects integer/bool sequence; binarise by median
    try:
        binarised = (x > np.median(x)).astype(np.int8).tolist()
        out['LZC'] = float(antropy.lziv_complexity(binarised, normalize=True))
    except Exception:
        out['LZC'] = np.nan

    # SampEn (m=2, r=0.2*std default)
    try:
        out['SampEn'] = float(antropy.sample_entropy(x, order=2))
    except Exception:
        out['SampEn'] = np.nan

    # WPE (Fadlallah)
    try:
        out['WPE'] = compute_wpe_fadlallah(x, m=WPE_M, tau=WPE_TAU)
    except Exception:
        out['WPE'] = np.nan

    # catch22 (22 features)
    try:
        c22 = pycatch22.catch22_all(x.tolist())
        # c22 is a dict with 'names' (22 strings) and 'values' (22 floats)
        for i, val in enumerate(c22['values']):
            out[f'catch22_{i+1}'] = float(val) if val is not None else np.nan
    except Exception:
        for i in range(22):
            out[f'catch22_{i+1}'] = np.nan

    return out


def collect_spine_cids(ds):
    """Union of chronos2 container_ids across all horizons for this dataset."""
    spine_cids = set()
    for h in HORIZONS:
        npz_path = Path(P['results']) / ds / h / 'foundation_models' / 'chronos2_perpoint.npz'
        if not npz_path.exists():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        spine_cids.update(np.asarray(npz['container_ids']).astype(str).tolist())
    return spine_cids


def save_partial(rows, suffix='partial'):
    """Write a partial parquet so we don't lose work if the script crashes mid-run."""
    if not rows:
        return
    p = OUT_PARQUET.with_suffix(f'.{suffix}.parquet')
    try:
        pd.DataFrame(rows).to_parquet(p, index=False)
    except Exception:
        pass


# -------------------------------- main --------------------------------

def main():
    print('=' * 72)
    print('par_features.py')
    print('=' * 72)
    print(f'pycatch22 ver: {pycatch22.__version__ if hasattr(pycatch22, "__version__") else "?"}')
    print(f'nolds     ver: {nolds.__version__   if hasattr(nolds, "__version__") else "?"}')
    print(f'antropy   ver: {antropy.__version__ if hasattr(antropy, "__version__") else "?"}')
    print(f'WPE m={WPE_M}, tau={WPE_TAU}')
    print()

    all_rows = []
    t_global = time.time()

    for ds in DATASETS:
        print('=' * 72)
        print(f'DATASET: {ds}')
        print('=' * 72)

        test_pq = Path(P['thesis']) / 'data' / 'processed' / ds / 'test.parquet'
        if not test_pq.exists():
            print(f'  test.parquet MISSING ({test_pq})')
            continue

        print(f'  loading test.parquet...')
        test_df = pd.read_parquet(test_pq)
        id_col = detect_id_col(test_df)
        if 'cpu_util_percent' not in test_df.columns:
            print(f'  cpu_util_percent column MISSING; skipping')
            continue
        print(f'  rows={len(test_df)}, id_col={id_col}')

        print(f'  collecting chronos2 spine container_ids across {len(HORIZONS)} horizons...')
        spine_cids = collect_spine_cids(ds)
        print(f'    {len(spine_cids)} unique containers in any spine')

        # filter + sort
        test_df = test_df[test_df[id_col].astype(str).isin(spine_cids)]
        test_df = test_df.sort_values([id_col, 'time_stamp']).reset_index(drop=True)
        print(f'  filtered rows: {len(test_df)}')

        grouped = test_df.groupby(id_col, sort=False)
        n_total = grouped.ngroups
        print(f'  computing features for {n_total} containers...')

        t_ds = time.time()
        for i, (cid, group) in enumerate(grouped):
            x = group['cpu_util_percent'].values.astype(np.float64)

            try:
                feats = compute_features_one_series(x)
            except Exception as e:
                # log and keep going; record a stub row
                print(f'    WARN cid={cid} feature error: {e}')
                feats = {'n_test_points': len(x), 'mean': np.nan, 'std': np.nan, 'cv': np.nan,
                         'DFA': np.nan, 'LZC': np.nan, 'SampEn': np.nan, 'WPE': np.nan}
                for k in range(22):
                    feats[f'catch22_{k+1}'] = np.nan

            feats['container_id'] = str(cid)
            feats['dataset'] = ds
            all_rows.append(feats)

            done = i + 1
            if done % 50 == 0 or done == n_total:
                elapsed = time.time() - t_ds
                rate = done / max(elapsed, 1e-6)
                remain = (n_total - done) / max(rate, 1e-6)
                print(f'    {done}/{n_total}  elapsed={elapsed:.0f}s  '
                      f'rate={rate:.2f} series/s  ETA(ds)={remain:.0f}s')

            if done % CHECKPOINT_EVERY == 0:
                save_partial(all_rows, suffix=f'partial_{ds}_{done}')

        print(f'  {ds} done in {time.time() - t_ds:.0f}s\n')

    # --------------------------- assemble + save ---------------------------
    if not all_rows:
        print('No rows. Aborting.')
        sys.exit(1)

    feat_df = pd.DataFrame(all_rows)

    # column order
    base_cols = ['container_id', 'dataset', 'n_test_points', 'mean', 'std', 'cv',
                 'DFA', 'LZC', 'SampEn', 'WPE']
    catch22_cols = sorted(
        [c for c in feat_df.columns if c.startswith('catch22_')],
        key=lambda x: int(x.split('_')[1]),
    )
    ordered = base_cols + catch22_cols
    feat_df = feat_df[[c for c in ordered if c in feat_df.columns]]

    print('=' * 72)
    print('SUMMARY')
    print('=' * 72)
    print(f'Total series:  {len(feat_df)}')
    print(f'Total elapsed: {time.time() - t_global:.0f}s')

    print('\nPer-dataset counts:')
    print(feat_df.groupby('dataset').size().to_string())

    print('\nNaN counts per feature (sorted; only features with any NaN):')
    nan_counts = feat_df.isna().sum().sort_values(ascending=False)
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts) > 0:
        print(nan_counts.to_string())
    else:
        print('  (no NaN values)')

    print('\nMedian of key features per dataset:')
    keys = ['n_test_points', 'mean', 'std', 'cv',
            'DFA', 'LZC', 'SampEn', 'WPE']
    keys = [k for k in keys if k in feat_df.columns]
    stats = feat_df.groupby('dataset')[keys].median().round(4)
    print(stats.to_string())

    print('\nRange of WPE per dataset (predictability proxy):')
    if 'WPE' in feat_df.columns:
        wpe_stats = feat_df.groupby('dataset')['WPE'].describe().round(4)
        print(wpe_stats.to_string())

    # save
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    try:
        feat_df.to_parquet(OUT_PARQUET, index=False)
        print(f'\nSaved: {OUT_PARQUET}')
    except Exception as e:
        print(f'\nParquet save failed: {e}')
    feat_df.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}')

    # clean up partials
    for p in OUT_PARQUET.parent.glob('par_features.partial_*.parquet'):
        try:
            p.unlink()
        except Exception:
            pass


if __name__ == '__main__':
    main()
