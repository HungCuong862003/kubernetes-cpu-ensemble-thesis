"""
merge_timesfm_into_leaderboard.py — Extend same_sample_leaderboard_full.csv
with TimesFM columns.

Reads the three TimesFM JSONs plus the existing 12-row same-sample leaderboard,
writes same_sample_leaderboard_v2.csv with extra columns:
    timesfm_r2, delta_timesfm_naive_pp, delta_timesfm_ens_pp, delta_timesfm_c2_pp

For Bitbrains (ensemble is N/A), delta_timesfm_ens_pp is NaN.

Run on Colab:
    !python merge_timesfm_into_leaderboard.py
"""

import json
import os

import pandas as pd

BASE = "/content/drive/MyDrive/k8s-ensemble-forecast"

LEADERBOARD_IN  = os.path.join(BASE, "same_sample_leaderboard_full.csv")
LEADERBOARD_OUT = os.path.join(BASE, "same_sample_leaderboard_v2.csv")

TIMESFM_JSONS = {
    "alibaba":   os.path.join(BASE, "src/day2_timesfm/timesfm_k20_alibaba.json"),
    "bitbrains": os.path.join(BASE, "src/day2_timesfm/timesfm_k50_bitbrains.json"),
    "bytedance": os.path.join(BASE, "src/day2_timesfm/timesfm_k50_bytedance.json"),
}


def main():
    print("=" * 64)
    print("MERGE TimesFM into same-sample leaderboard")
    print("=" * 64)

    df = pd.read_csv(LEADERBOARD_IN)
    print(f"  loaded {len(df)} rows from {LEADERBOARD_IN}")

    df["timesfm_r2"] = float("nan")
    df["timesfm_n_points"] = float("nan")
    df["timesfm_coverage"] = float("nan")

    for dataset_name, json_path in TIMESFM_JSONS.items():
        if not os.path.exists(json_path):
            print(f"  MISSING: {dataset_name} -> {json_path}")
            continue
        with open(json_path) as f:
            payload = json.load(f)
        print(f"  {dataset_name}: loaded {json_path}")
        print(f"    K={payload.get('num_origins')}  "
              f"horizons={list(payload['timesfm_zero_shot'].keys())}")

        for hz, r in payload["timesfm_zero_shot"].items():
            if "error" in r:
                print(f"    {hz}: ERROR {r['error']}")
                continue
            mask = (df["dataset"] == dataset_name) & (df["horizon"] == hz)
            if mask.sum() == 0:
                print(f"    {hz}: no matching row, skipping")
                continue
            df.loc[mask, "timesfm_r2"] = r["r2_mean"]
            df.loc[mask, "timesfm_naive_sub"] = r["r2_naive_subsample"]
            df.loc[mask, "timesfm_n_points"] = r["n_points"]
            df.loc[mask, "timesfm_coverage"] = r["pct_inside_p10_p90"]

            existing_n = int(df.loc[mask, "n_points"].values[0])
            tf_n = int(r["n_points"])
            if existing_n != tf_n:
                print(f"    {hz}: WARN n_points mismatch "
                      f"(leaderboard={existing_n}, TimesFM={tf_n})")

    df["delta_timesfm_naive_pp"] = (df["timesfm_r2"] - df["timesfm_naive_sub"]) * 100
    df["delta_timesfm_ens_pp"]   = (df["timesfm_r2"] - df["ensemble_sub_r2"]) * 100
    df["delta_timesfm_c2_pp"]    = (df["timesfm_r2"] - df["chronos2_r2"]) * 100

    df.to_csv(LEADERBOARD_OUT, index=False)
    print(f"\nwrote {LEADERBOARD_OUT}")

    cols = ["dataset", "horizon",
            "naive_sub_r2", "ensemble_sub_r2", "chronos2_r2", "timesfm_r2",
            "delta_timesfm_naive_pp", "delta_timesfm_ens_pp", "delta_timesfm_c2_pp"]
    with pd.option_context("display.float_format", "{:.4f}".format,
                            "display.width", 200):
        print()
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()