# phase_f/scripts/compute_wpe.py
# Phase F D4 — Weighted Permutation Entropy (Fadlallah et al. 2013)
# for Alibaba, Bitbrains, ByteDance. Reads processed parquets.

import numpy as np
import pandas as pd
from itertools import permutations
import os
import math
import time

DATA_ROOT = "data/processed"
DATASETS = ["bitbrains", "bytedance", "alibaba"]
ID_COL = "container_id"
TIME_COL = "time_stamp"
VAL_COL = "cpu_util_percent"

OUTPUT_DIR = "phase_f/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 4
EMBED_LAG = 1
MIN_SERIES_LEN = 100


def weighted_permutation_entropy(x, m=4, tau=1):
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_motifs = n - (m - 1) * tau
    if n_motifs < 2:
        return np.nan

    patterns = list(permutations(range(m)))
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    pattern_weights = np.zeros(len(patterns))

    for i in range(n_motifs):
        motif = x[i : i + m * tau : tau]
        ranks = tuple(np.argsort(np.argsort(motif)))
        mu = motif.mean()
        weight = np.mean((motif - mu) ** 2)
        if ranks in pattern_to_idx:
            pattern_weights[pattern_to_idx[ranks]] += weight

    total = pattern_weights.sum()
    if total == 0:
        return np.nan

    p = pattern_weights / total
    p_nonzero = p[p > 0]
    h = -np.sum(p_nonzero * np.log(p_nonzero))
    return h / np.log(math.factorial(m))


all_results = {}

for ds_name in DATASETS:
    print("=" * 60)
    print(ds_name.upper())
    print("=" * 60)
    t0 = time.time()

    print(f"  reading parquets from {DATA_ROOT}/{ds_name}/...")
    train = pd.read_parquet(f"{DATA_ROOT}/{ds_name}/train.parquet")
    val = pd.read_parquet(f"{DATA_ROOT}/{ds_name}/val.parquet")
    test = pd.read_parquet(f"{DATA_ROOT}/{ds_name}/test.parquet")
    full = pd.concat([train, val, test], ignore_index=True)
    full = full[[ID_COL, TIME_COL, VAL_COL]].copy()
    full = full.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    n_series = full[ID_COL].nunique()
    print(f"  combined rows: {len(full):,}  series: {n_series}")

    results = []
    for i, (sid, group) in enumerate(full.groupby(ID_COL)):
        series = group[VAL_COL].to_numpy()
        if len(series) < MIN_SERIES_LEN:
            print(f"  SKIP {sid}: only {len(series)} points")
            continue

        wpe = weighted_permutation_entropy(series, m=EMBED_DIM, tau=EMBED_LAG)
        results.append({ID_COL: sid, "n_points": len(series), "wpe_m4_t1": wpe})

        if ds_name == "alibaba" and (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/{n_series} containers in {elapsed:.0f}s "
                  f"({elapsed/(i+1)*1000:.1f}ms each)")

    df = pd.DataFrame(results)
    elapsed = time.time() - t0
    print(f"  computed: {len(df)} series in {elapsed:.1f}s")
    print(f"  WPE  min={df['wpe_m4_t1'].min():.4f}  "
          f"median={df['wpe_m4_t1'].median():.4f}  "
          f"max={df['wpe_m4_t1'].max():.4f}")

    out_path = os.path.join(OUTPUT_DIR, f"wpe_{ds_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"  saved: {out_path}\n")

    all_results[ds_name] = df


print("=" * 60)
print("Sanity: WPE vs ACF@24h correlation (expect negative)")
print("=" * 60)

bb_omega = pd.read_csv("results/bitbrains/omega_bitbrains.csv")
omega_id = "vm_id" if "vm_id" in bb_omega.columns else "container_id"
bb_merged = all_results["bitbrains"].merge(
    bb_omega.rename(columns={omega_id: ID_COL})[[ID_COL, "acf_24h", "cv", "hurst"]],
    on=ID_COL, how="inner"
)
print(f"\nBitbrains (n={len(bb_merged)}):")
print(f"  WPE vs ACF@24h: r = {bb_merged['wpe_m4_t1'].corr(bb_merged['acf_24h']):.4f}")
print(f"  WPE vs CV:      r = {bb_merged['wpe_m4_t1'].corr(bb_merged['cv']):.4f}")
print(f"  WPE vs Hurst:   r = {bb_merged['wpe_m4_t1'].corr(bb_merged['hurst']):.4f}")

byt_stats = pd.read_csv("results/bytedance/bytedance_per_instance_stats.csv")
stats_id = "instance_id" if "instance_id" in byt_stats.columns else "container_id"
byt_merged = all_results["bytedance"].merge(
    byt_stats.rename(columns={stats_id: ID_COL})[[ID_COL, "acf_24h", "hurst"]],
    on=ID_COL, how="inner"
)
print(f"\nByteDance (n={len(byt_merged)}):")
print(f"  WPE vs ACF@24h: r = {byt_merged['wpe_m4_t1'].corr(byt_merged['acf_24h']):.4f}")
print(f"  WPE vs Hurst:   r = {byt_merged['wpe_m4_t1'].corr(byt_merged['hurst']):.4f}")

ali_omega = pd.read_csv("results/alibaba/omega_alibaba.csv")
ali_merged = all_results["alibaba"].merge(
    ali_omega[["container_id", "acf_24h", "cv", "hurst"]],
    on="container_id", how="inner"
)
print(f"\nAlibaba (n={len(ali_merged)}):")
print(f"  WPE vs ACF@24h: r = {ali_merged['wpe_m4_t1'].corr(ali_merged['acf_24h']):.4f}")
print(f"  WPE vs CV:      r = {ali_merged['wpe_m4_t1'].corr(ali_merged['cv']):.4f}")
print(f"  WPE vs Hurst:   r = {ali_merged['wpe_m4_t1'].corr(ali_merged['hurst']):.4f}")

print("\nDone.")
