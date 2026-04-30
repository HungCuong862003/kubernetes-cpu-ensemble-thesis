"""
task1_bytedance_omega.py
────────────────────────
Fill the two blank cells in omega_summary.csv for ByteDance IaaS:
  - Hurst_median
  - ACF@1h

Input:  iaas.csv           (321k rows, 93 instances, 10-min intervals)
Output: omega_summary.csv  (updated ByteDance row, other rows untouched)

Run on Colab:
    !pip install nolds --quiet
    # upload iaas.csv and omega_summary.csv to working dir, then:
    !python task1_bytedance_omega.py

Matching the exact same methods used for Alibaba/Bitbrains omega
values (from BITBRAINS_23_03_26.ipynb):
  - Hurst: nolds.hurst_rs with fit='RANSAC', corrected=True
  - ACF:   np.corrcoef (plain sample autocorrelation at fixed lag)
"""

import pandas as pd
import numpy as np
import nolds

# ── config ─────────────────────────────────────────────────────────────────
INPUT_CSV   = "iaas.csv"
OMEGA_CSV   = "omega_summary.csv"
MIN_POINTS  = 100   # skip instances shorter than this

# iaas.csv has 10-min intervals
# ACF@1h  = lag 6   (6  * 10min = 60min)
# ACF@24h = lag 144 (144 * 10min = 1440min = 24h)
LAG_1H  = 6
LAG_24H = 144


# ── helper functions (same as BITBRAINS_23_03_26.ipynb) ────────────────────

def compute_hurst(s):
    """R/S Hurst exponent. Matches notebook: fit='RANSAC', corrected=True."""
    s = s[~np.isnan(s)]
    if len(s) < MIN_POINTS:
        return np.nan
    try:
        return float(nolds.hurst_rs(s, fit="RANSAC", corrected=True))
    except Exception:
        return np.nan


def compute_acf(s, lag):
    """Plain sample autocorrelation at a given lag. Matches notebook."""
    s = s[~np.isnan(s)]
    if len(s) <= lag:
        return np.nan
    return float(np.corrcoef(s[:-lag], s[lag:])[0, 1])


# ── load data ──────────────────────────────────────────────────────────────

print(f"loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
print(f"  rows: {len(df)}")
print(f"  columns: {list(df.columns)}")

instances = sorted(df["cols"].unique())
n_inst = len(instances)
print(f"  instances: {n_inst}")


# ── compute per-instance stats ─────────────────────────────────────────────

print(f"\ncomputing Hurst + ACF for {n_inst} instances ...")
rows = []

# RANSAC inside nolds.hurst_rs uses random sampling, so results vary
# between runs. fix the seed so the thesis numbers are reproducible.
np.random.seed(42)

for i, inst_id in enumerate(instances):
    grp = df[df["cols"] == inst_id].sort_values("date")
    ts = grp["data"].values

    if len(ts) < MIN_POINTS:
        print(f"  skip {inst_id}: only {len(ts)} points")
        continue

    h     = compute_hurst(ts)
    a_1h  = compute_acf(ts, LAG_1H)
    a_24h = compute_acf(ts, LAG_24H)

    rows.append({
        "instance_id": inst_id,
        "n_points":    len(ts),
        "hurst":       h,
        "acf_1h":      a_1h,
        "acf_24h":     a_24h,
    })

    # progress every 20 instances
    if (i + 1) % 20 == 0 or (i + 1) == n_inst:
        print(f"  done {i + 1}/{n_inst}")

per_inst = pd.DataFrame(rows)
print(f"\nvalid instances: {len(per_inst)}")


# ── compute medians ────────────────────────────────────────────────────────

hurst_median = per_inst["hurst"].median()
acf_1h_median = per_inst["acf_1h"].median()
acf_24h_median = per_inst["acf_24h"].median()

print(f"\n{'='*50}")
print(f"ByteDance IaaS results:")
print(f"  Hurst_median = {hurst_median:.3f}")
print(f"  ACF@1h       = {acf_1h_median:.3f}")
print(f"  ACF@24h      = {acf_24h_median:.3f}  (sanity check, should be ~0.489)")
print(f"{'='*50}")

# sanity check: ACF@24h should be close to the existing value (0.489)
EXPECTED_ACF24H = 0.489
diff = abs(acf_24h_median - EXPECTED_ACF24H)
sanity_ok = diff <= 0.02

if not sanity_ok:
    print(f"\n*** WARNING: ACF@24h = {acf_24h_median:.3f} differs from "
          f"existing value {EXPECTED_ACF24H} by {diff:.3f} ***")
    print("    check whether a different ACF method was used originally")
    print("    NOT updating omega_summary.csv -- investigate first")
else:
    print(f"  ACF@24h sanity check passed (diff = {diff:.4f})")


# ── save per-instance detail (optional, useful for debugging) ──────────────

detail_path = "bytedance_per_instance_stats.csv"
per_inst.to_csv(detail_path, index=False)
print(f"\nsaved per-instance details to {detail_path}")


# ── update omega_summary.csv ──────────────────────────────────────────────

if not sanity_ok:
    print(f"\nskipping {OMEGA_CSV} update due to failed sanity check.")
else:
    print(f"\nupdating {OMEGA_CSV} ...")
    omega = pd.read_csv(OMEGA_CSV)

    print("  before:")
    print(omega.to_string(index=False))

    # find the ByteDance row
    mask = omega["Dataset"].str.contains("ByteDance", case=False)
    n_matches = mask.sum()

    if n_matches != 1:
        print(f"\n*** ERROR: expected 1 ByteDance row, found {n_matches} ***")
        print("    NOT updating. Fix omega_summary.csv manually.")
    else:
        omega.loc[mask, "Hurst_median"] = round(hurst_median, 3)
        omega.loc[mask, "ACF@1h"]       = round(acf_1h_median, 3)

        print("\n  after:")
        print(omega.to_string(index=False))

        omega.to_csv(OMEGA_CSV, index=False)
        print(f"\n  saved to {OMEGA_CSV}")

print("\ndone.")
