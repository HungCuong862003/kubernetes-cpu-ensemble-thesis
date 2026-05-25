"""
par_alignment_test.py — v2.

V1 found test.parquet is interleaved by time_stamp, not contiguous by container.
y_true.npy is 4921 × 24 rows shorter than test.parquet, suggesting:
  y_true.npy = sort(test.parquet, by=[container_id, time_stamp]) → drop last h_steps per container → cpu_util_percent

This v2 sorts first, then tests alignment. Same Alibaba h120, first 1000 chronos2 points.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from _paths import P


DS = 'alibaba'
H = 'h120'
N_CHECK = 1000

RESULTS = Path(P['results'])
DATA_RAW = Path(P['thesis']) / 'data' / 'processed'


print('=' * 70)
print(f'PAR ALIGNMENT TEST V2  —  {DS} {H}  —  {N_CHECK} sample points')
print('=' * 70)


# -----------------------------------------------------------------------
# 1. Load chronos2 .npz
# -----------------------------------------------------------------------

print('\n[1] Loading chronos2 .npz...')
npz_path = RESULTS / DS / H / 'foundation_models' / 'chronos2_perpoint.npz'
npz = np.load(npz_path, allow_pickle=True)
c2_cids    = npz['container_ids']
c2_origins = npz['origins']
c2_ytrue   = npz['y_true']
c2_ynaive  = npz['y_naive']
c2_h_steps = int(npz['h_steps'])
print(f'  n_points: {len(c2_cids)}  h_steps: {c2_h_steps}')


# -----------------------------------------------------------------------
# 2. Load + sort test.parquet
# -----------------------------------------------------------------------

print('\n[2] Loading + sorting test.parquet by (container_id, time_stamp)...')
test_pq = DATA_RAW / DS / 'test.parquet'
test_df = pd.read_parquet(test_pq, columns=['container_id', 'time_stamp', 'cpu_util_percent'])
print(f'  raw rows:        {len(test_df)}')
print(f'  before sort, first 3 rows:')
print(test_df.head(3).to_string())

test_df = test_df.sort_values(['container_id', 'time_stamp']).reset_index(drop=True)
print(f'  after sort, first 3 rows:')
print(test_df.head(3).to_string())
print(f'  unique containers: {test_df["container_id"].nunique()}')


# -----------------------------------------------------------------------
# 3. Drop last h_steps rows per container
# -----------------------------------------------------------------------

print(f'\n[3] Dropping last h_steps={c2_h_steps} rows per container...')
# We keep all rows per container EXCEPT the last h_steps ones.
# Use groupby + cumcount from the tail.
test_df['_rev_position'] = test_df.groupby('container_id').cumcount(ascending=False)
sorted_dropped = test_df[test_df['_rev_position'] >= c2_h_steps].drop(columns='_rev_position').reset_index(drop=True)
print(f'  rows after drop: {len(sorted_dropped)}')


# -----------------------------------------------------------------------
# 4. Verify against y_true.npy length
# -----------------------------------------------------------------------

print(f'\n[4] Verifying against y_true.npy length...')
y_true_npy_path = RESULTS / DS / H / 'predictions' / 'y_true.npy'
y_true_npy = np.load(y_true_npy_path)
print(f'  sorted_dropped rows: {len(sorted_dropped)}')
print(f'  y_true.npy length:   {len(y_true_npy)}')
print(f'  match: {len(sorted_dropped) == len(y_true_npy)}')


# -----------------------------------------------------------------------
# 5. Build container → start in sorted_dropped
# -----------------------------------------------------------------------

print(f'\n[5] Building container → start index in sorted_dropped...')
sorted_dropped['_row'] = np.arange(len(sorted_dropped))
ranges = sorted_dropped.groupby('container_id', sort=False)['_row'].agg(['min', 'max'])
ranges['size'] = ranges['max'] - ranges['min'] + 1
print(f'  containers: {len(ranges)}')
print(f'  sample (c_10032):')
if 'c_10032' in ranges.index:
    print(f'    start={ranges.loc["c_10032", "min"]}, size={ranges.loc["c_10032", "size"]}')
print(f'  size stats: min={ranges["size"].min()}, max={ranges["size"].max()}, median={int(ranges["size"].median())}')


# -----------------------------------------------------------------------
# 6. Alignment hypotheses
# -----------------------------------------------------------------------

print(f'\n[6] Alignment check across first {N_CHECK} chronos2 points...')

# For h120 with h_steps=24 and origin=23:
#   - context = positions 0..23 within container
#   - target = position 23 + 24 = 47 within container
#   - In y_true.npy: each row j corresponds to "evaluation row j within container",
#     and y_true.npy[start_c + j] = cpu_util at sorted-position (j + h_steps) within c.
#   - So for chronos2 origin=O, we want y_true.npy[start_c + (O + h_steps - h_steps)] = y_true.npy[start_c + O]
#   - But there's ambiguity in whether the "evaluation row j" is the *origin* (j=O) or the *target row* (j=O+h_steps).

hypotheses = {
    'H1: start + origin':             lambda s, o: s + o,
    'H2: start + (origin - h_steps + 1)': lambda s, o: s + (o - c2_h_steps + 1),
    'H3: start + (origin - h_steps)': lambda s, o: s + (o - c2_h_steps),
}

range_dict = ranges.to_dict(orient='index')
n_check = min(N_CHECK, len(c2_cids))

results = {}
for hname, hfunc in hypotheses.items():
    n_match_ytrue = 0
    n_match_ynaive = 0
    n_missing = 0
    n_mismatch = 0
    mismatch_examples = []
    
    for i in range(n_check):
        cid = c2_cids[i]
        origin = int(c2_origins[i])
        c2_yt = float(c2_ytrue[i])
        c2_yn = float(c2_ynaive[i])
        
        if cid not in range_dict:
            n_missing += 1
            continue
        
        start = int(range_dict[cid]['min'])
        size  = int(range_dict[cid]['size'])
        idx_offset = hfunc(start, origin)
        
        if idx_offset < 0 or idx_offset >= len(y_true_npy):
            n_missing += 1
            continue
        if idx_offset - start < 0 or idx_offset - start >= size:
            n_missing += 1
            continue
        
        npy_yt = float(y_true_npy[idx_offset])
        if abs(c2_yt - npy_yt) < 1e-3:
            n_match_ytrue += 1
        else:
            n_mismatch += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append((cid, origin, idx_offset, c2_yt, npy_yt, c2_yn))
    
    results[hname] = {
        'match': n_match_ytrue, 'missing': n_missing, 'mismatch': n_mismatch,
        'examples': mismatch_examples,
    }

print()
print(f'  {"hypothesis":<40} {"match":<8} {"miss":<8} {"mismatch":<10}')
print(f'  {"-"*40} {"-"*8} {"-"*8} {"-"*10}')
for hname, r in results.items():
    print(f'  {hname:<40} {r["match"]:<8} {r["missing"]:<8} {r["mismatch"]:<10}')


# -----------------------------------------------------------------------
# 7. Direct lookup: what IS y_true.npy[start_c_10032 + 23]?
# -----------------------------------------------------------------------

print(f'\n[7] Diagnostic dump for c_10032 origin=23:')
cid = 'c_10032'
if cid in range_dict:
    start = int(range_dict[cid]['min'])
    print(f'  start of c_10032 in sorted_dropped: {start}')
    print(f'  chronos2 says y_true=3.0, y_naive={c2_ynaive[0]}')
    print(f'  y_true.npy at various offsets:')
    for off in range(20, 50):
        print(f'    [start + {off:3d}] = {y_true_npy[start + off]:.3f}')
    print(f'  sorted_dropped cpu_util_percent for c_10032 at sorted positions 20..49:')
    c_rows = sorted_dropped[sorted_dropped['container_id'] == cid].head(50).tail(30)
    print(c_rows[['time_stamp', 'cpu_util_percent']].to_string())


# -----------------------------------------------------------------------
# 8. Find correct offset by direct search
# -----------------------------------------------------------------------

print(f'\n[8] Brute-force search for correct offset (c_10032, expected y_true=3.0)...')
if 'c_10032' in range_dict:
    start = int(range_dict[cid]['min'])
    size = int(range_dict[cid]['size'])
    target = 3.0
    matches = []
    for off in range(min(size, 200)):
        if abs(y_true_npy[start + off] - target) < 1e-3:
            matches.append(off)
    print(f'  positions where y_true.npy[start_c_10032 + off] == 3.0 (off ≤ 200):')
    print(f'    {matches[:20]}')


print()
print('=' * 70)
print('END OF ALIGNMENT TEST V2')
print('=' * 70)