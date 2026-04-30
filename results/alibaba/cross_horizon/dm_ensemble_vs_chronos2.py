"""
Diebold-Mariano test: NNLS hetero ensemble vs Chronos-2 zero-shot on Alibaba.

Reads the per-point arrays saved by the patched day1_chronos2_sanity.py and
aligns them against the ensemble's saved pred_hetero_ensemble.npy.

sprint1 does not persist y_true.npy as a separate file. We reconstruct
y_true from test.parquet using the validated rule (sort by cid, time_stamp,
shift target by -h_steps per container, drop NaN).

For a Chronos point at (cid, origin):
    - target = raw_series[cid][origin + h_steps]
    - ensemble row = start_idx[cid] + origin
where start_idx[cid] = cumulative (series_length - h_steps) for all
containers iterated before cid in groupby(sort=False) order.

Runs DM with both power=2 (MSE loss) and power=1 (MAE loss).

Usage:
    python dm_ensemble_vs_chronos2.py \
        --test_parquet /workspace/data/alibaba/test.parquet \
        --ensemble_dir /workspace/ensemble_out \
        --chronos_dir  /workspace \
        --out dm_ensemble_vs_chronos2.csv

ensemble_dir layout expected: <horizon>/pred_hetero_ensemble.npy
chronos_dir layout expected:  chronos2_perpoint_<horizon>.npz
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as scipy_t


HORIZONS = [
    ("10min",  2),
    ("30min",  6),
    ("60min", 12),
    ("120min", 24),
]

ID_COL     = os.environ.get("ID_COL",     "container_id")
TIME_COL   = os.environ.get("TIME_COL",   "time_stamp")
TARGET_COL = os.environ.get("TARGET_COL", "cpu_util_percent")


# =============================================================================
# DM with Harvey-Leybourne-Newbold small-sample correction.
# Copy of sprint1_Main.py:928 so this script has zero dependency on the rest
# of the project.
# =============================================================================

def diebold_mariano(y_true, pred1, pred2, h=1, power=2):
    """Positive dm_stat means pred1 has larger loss than pred2 (pred2 wins)."""
    e1 = np.asarray(y_true - pred1, np.float64)
    e2 = np.asarray(y_true - pred2, np.float64)
    d  = np.abs(e1) ** power - np.abs(e2) ** power
    n  = len(d)

    if n < 3:
        return {"dm_stat": 0.0, "dm_stat_hln": 0.0,
                "dm_p_two": 1.0, "n": n, "dm_bandwidth": 0}

    d_bar    = d.mean()
    d_demean = d - d_bar
    gamma_0  = np.mean(d_demean ** 2)
    max_lags = min(max(h - 1, 1), n // 4)

    gamma_sum = 0.0
    for k in range(1, max_lags + 1):
        if k >= n:
            break
        gamma_sum += 2 * np.mean(d_demean[k:] * d_demean[:-k])
    var_d = (gamma_0 + gamma_sum) / n

    dm_stat   = d_bar / np.sqrt(max(var_d, 1e-12))
    hln_inner = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
    hln_factor = np.sqrt(max(hln_inner, 0.0))
    dm_hln = hln_factor * dm_stat

    p_two = float(2.0 * scipy_t.sf(abs(dm_hln), df=n - 1))

    return {
        "dm_stat":     float(dm_stat),
        "dm_stat_hln": float(dm_hln),
        "dm_p_two":    p_two,
        "dm_bandwidth": max_lags,
        "n":            n,
    }


# =============================================================================
# Reconstruct sprint1's flat array layout from test.parquet.
# Every container contributes (raw_length - h_steps) rows in order.
# =============================================================================

def build_series_and_startidx(test_parquet_path, h_steps):
    """Returns (series_by_id, start_idx, usable_len, flat_len).

    series_by_id: {cid: np.array of raw target values, pre-shift}
    start_idx:    {cid: global row index in the flat ensemble array}
    usable_len:   {cid: number of rows this container contributes}
    flat_len:     total row count in the flat ensemble array
    """
    df = pd.read_parquet(test_parquet_path)
    needed = [ID_COL, TIME_COL, TARGET_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"test.parquet missing cols {missing}")
    df = df[needed].sort_values([ID_COL, TIME_COL]).reset_index(drop=True)

    series_by_id = {}
    start_idx = {}
    usable_len = {}
    cursor = 0
    for cid, sub in df.groupby(ID_COL, sort=False):
        raw = sub[TARGET_COL].to_numpy(dtype=np.float64)
        series_by_id[cid] = raw
        raw_len = len(raw)
        ulen = raw_len - h_steps
        if ulen < 0:
            ulen = 0
        start_idx[cid] = cursor
        usable_len[cid] = ulen
        cursor += ulen

    return series_by_id, start_idx, usable_len, cursor


# =============================================================================
# Align chronos points to the flat ensemble array
# =============================================================================

def align(chronos_npz_path, series_by_id, start_idx, usable_len,
          pred_ens, h_steps):
    d = np.load(chronos_npz_path, allow_pickle=True)
    cids    = d["container_ids"]
    origins = d["origins"]
    y_c_npz = d["y_true"].astype(np.float64)
    p_c_npz = d["y_pred_mean"].astype(np.float64)

    y_t_list   = []
    y_ens_list = []
    y_c_list   = []
    drift_list = []
    misses = 0
    ens_oob = 0

    for i in range(len(cids)):
        cid    = str(cids[i])
        origin = int(origins[i])

        if cid not in start_idx:
            misses += 1
            continue
        if origin >= usable_len[cid]:
            ens_oob += 1
            continue

        # Ground truth from raw series at target position
        series = series_by_id[cid]
        target_pos = origin + h_steps
        if target_pos >= len(series):
            ens_oob += 1
            continue
        y_from_raw = float(series[target_pos])

        # Sanity drift: chronos's own y_true at this point vs our recomputation
        drift = abs(y_from_raw - y_c_npz[i])

        # Ensemble prediction at the aligned row
        row = start_idx[cid] + origin
        if row >= len(pred_ens):
            ens_oob += 1
            continue

        y_t_list.append(y_from_raw)
        y_ens_list.append(float(pred_ens[row]))
        y_c_list.append(float(p_c_npz[i]))
        drift_list.append(drift)

    y_t   = np.array(y_t_list,   dtype=np.float64)
    y_ens = np.array(y_ens_list, dtype=np.float64)
    y_c   = np.array(y_c_list,   dtype=np.float64)
    drift = np.array(drift_list, dtype=np.float64)

    print(f"  chronos points: {len(cids):,}  paired: {len(y_t):,}  "
          f"missed: {misses}  oob: {ens_oob}")
    print(f"  y_true drift (raw vs chronos npz): "
          f"max={drift.max():.2e}  mean={drift.mean():.2e}")

    # float32 roundtrip in the npz → drift up to ~1e-5.
    if drift.max() > 1e-2:
        raise RuntimeError(
            f"y_true drift {drift.max():.4f} too large - alignment broken"
        )

    return y_t, y_ens, y_c


# =============================================================================
# Main
# =============================================================================

def r2_and_mae(y, p):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(r2), float(np.mean(np.abs(y - p)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_parquet", required=True)
    parser.add_argument("--ensemble_dir", required=True,
                        help="Dir containing <horizon>/pred_hetero_ensemble.npy")
    parser.add_argument("--chronos_dir", required=True,
                        help="Dir containing chronos2_perpoint_<label>.npz")
    parser.add_argument("--out", default="dm_ensemble_vs_chronos2.csv")
    args = parser.parse_args()

    rows = []
    for horizon_label, h_steps in HORIZONS:
        print(f"\n=== {horizon_label} (h_steps={h_steps}) ===")

        pred_ens_path = os.path.join(
            args.ensemble_dir, horizon_label, "pred_hetero_ensemble.npy"
        )
        if not os.path.exists(pred_ens_path):
            print(f"  SKIP - missing {pred_ens_path}")
            continue
        pred_ens = np.load(pred_ens_path, mmap_mode="r")
        print(f"  ensemble pred shape={pred_ens.shape}")

        chronos_path = os.path.join(
            args.chronos_dir, f"chronos2_perpoint_{horizon_label}.npz"
        )
        if not os.path.exists(chronos_path):
            print(f"  SKIP - missing {chronos_path}")
            continue

        series_by_id, start_idx, usable_len, flat_len = \
            build_series_and_startidx(args.test_parquet, h_steps)
        print(f"  reconstructed flat array length: {flat_len:,}  "
              f"(pred_hetero len: {len(pred_ens):,})")
        if flat_len != len(pred_ens):
            print(f"  WARNING: length mismatch - drift check will confirm "
                  f"whether alignment still works")

        y_t, y_ens, y_c = align(
            chronos_path, series_by_id, start_idx, usable_len,
            pred_ens, h_steps,
        )

        # Recompute metrics on paired sample for sanity
        r2_ens, mae_ens = r2_and_mae(y_t, y_ens)
        r2_c,   mae_c   = r2_and_mae(y_t, y_c)
        print(f"  paired sample (N={len(y_t):,}):")
        print(f"    ensemble  R2={r2_ens:.4f}  MAE={mae_ens:.4f}")
        print(f"    chronos-2 R2={r2_c:.4f}  MAE={mae_c:.4f}")
        print(f"    gap (ens-c2): R2={(r2_ens-r2_c)*100:+.2f}pp  "
              f"MAE={mae_ens-mae_c:+.4f}")

        dm_mse = diebold_mariano(y_t, y_ens, y_c, h=h_steps, power=2)
        dm_mae = diebold_mariano(y_t, y_ens, y_c, h=h_steps, power=1)
        print(f"    DM-MSE stat={dm_mse['dm_stat_hln']:+.3f}  "
              f"p_two={dm_mse['dm_p_two']:.4f}")
        print(f"    DM-MAE stat={dm_mae['dm_stat_hln']:+.3f}  "
              f"p_two={dm_mae['dm_p_two']:.4f}")

        rows.append({
            "Horizon":       horizon_label,
            "N":             len(y_t),
            "R2_ens":        round(r2_ens, 6),
            "R2_chronos":    round(r2_c,   6),
            "R2_gap_pp":     round((r2_ens - r2_c) * 100, 4),
            "MAE_ens":       round(mae_ens, 6),
            "MAE_chronos":   round(mae_c,   6),
            "MAE_gap":       round(mae_ens - mae_c, 6),
            "DM_MSE_stat":   round(dm_mse["dm_stat_hln"], 4),
            "DM_MSE_p":      round(dm_mse["dm_p_two"],    6),
            "DM_MSE_sig05":  dm_mse["dm_p_two"] < 0.05,
            "DM_MAE_stat":   round(dm_mae["dm_stat_hln"], 4),
            "DM_MAE_p":      round(dm_mae["dm_p_two"],    6),
            "DM_MAE_sig05":  dm_mae["dm_p_two"] < 0.05,
            "DM_bandwidth":  dm_mse["dm_bandwidth"],
        })

    if not rows:
        print("No horizons produced a result.")
        sys.exit(1)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
