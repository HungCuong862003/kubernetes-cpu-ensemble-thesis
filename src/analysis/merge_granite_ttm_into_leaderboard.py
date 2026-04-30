"""
merge_granite_ttm_into_leaderboard.py — Extend same_sample_leaderboard_v2.csv
with Granite TTM columns.

Sibling to merge_timesfm_into_leaderboard.py. Reads three TTM JSONs plus the
v2 leaderboard (which already has Chronos-2 and TimesFM columns), writes
same_sample_leaderboard_v3.csv with extra columns:
    ttm_r2, ttm_n_points, ttm_coverage,
    delta_ttm_naive_pp, delta_ttm_ens_pp,
    delta_ttm_c2_pp, delta_ttm_tfm_pp

For Bitbrains (ensemble is N/A) delta_ttm_ens_pp is NaN, same as the TimesFM
merger. Bitbrains uses K=50, the other two K=20/K=50 matches whatever was
saved during the day3 sanity run.

Run on Colab:
    !python merge_granite_ttm_into_leaderboard.py
"""

import json
import os

import pandas as pd

BASE = "/content/drive/MyDrive/k8s-ensemble-forecast"

LEADERBOARD_IN  = os.path.join(BASE, "same_sample_leaderboard_v2.csv")
LEADERBOARD_OUT = os.path.join(BASE, "same_sample_leaderboard_v3.csv")

TTM_JSONS = {
    "alibaba":   os.path.join(BASE, "src/day3_ttm/ttm_k20_alibaba.json"),
    "bitbrains": os.path.join(BASE, "src/day3_ttm/ttm_k50_bitbrains.json"),
    "bytedance": os.path.join(BASE, "src/day3_ttm/ttm_k50_bytedance.json"),
}

# top-level key inside each TTM JSON that holds the per-horizon dict.
# matches the timesfm_zero_shot / chronos2_zero_shot convention.
TTM_RESULTS_KEY = "ttm_zero_shot"


def main():
    print("=" * 64)
    print("MERGE Granite TTM into same-sample leaderboard")
    print("=" * 64)

    df = pd.read_csv(LEADERBOARD_IN)
    print(f"  loaded {len(df)} rows from {LEADERBOARD_IN}")

    # quick sanity check that v2 actually has the timesfm columns we need
    # for the delta_ttm_tfm_pp computation. if not, warn but keep going.
    if "timesfm_r2" not in df.columns:
        print("  WARN: v2 leaderboard has no timesfm_r2 column. "
              "delta_ttm_tfm_pp will be all NaN.")

    df["ttm_r2"] = float("nan")
    df["ttm_n_points"] = float("nan")
    df["ttm_coverage"] = float("nan")

    for dataset_name, json_path in TTM_JSONS.items():
        if not os.path.exists(json_path):
            print(f"  MISSING: {dataset_name} -> {json_path}")
            continue
        with open(json_path) as f:
            payload = json.load(f)

        if TTM_RESULTS_KEY not in payload:
            print(f"  {dataset_name}: key '{TTM_RESULTS_KEY}' not in JSON, "
                  f"skipping. (top-level keys: {list(payload.keys())})")
            continue

        results = payload[TTM_RESULTS_KEY]
        print(f"  {dataset_name}: loaded {json_path}")
        print(f"    K={payload.get('num_origins')}  "
              f"horizons={list(results.keys())}")

        for hz, r in results.items():
            if "error" in r:
                print(f"    {hz}: ERROR {r['error']}")
                continue
            mask = (df["dataset"] == dataset_name) & (df["horizon"] == hz)
            if mask.sum() == 0:
                print(f"    {hz}: no matching row, skipping")
                continue

            df.loc[mask, "ttm_r2"] = r["r2_mean"]
            df.loc[mask, "ttm_n_points"] = r["n_points"]
            # TTM may or may not produce p10/p90 depending on the variant.
            # keep this defensive — fall back to NaN rather than crash.
            df.loc[mask, "ttm_coverage"] = r.get("pct_inside_p10_p90",
                                                 float("nan"))

            # cross-check that TTM ran on the same number of points the
            # leaderboard expects. n_points should always be set in v2 but
            # guard against NaN anyway — int(NaN) crashes.
            existing_n_raw = df.loc[mask, "n_points"].values[0]
            ttm_n = int(r["n_points"])
            if pd.isna(existing_n_raw):
                print(f"    {hz}: WARN existing n_points is NaN, "
                      f"can't cross-check (TTM={ttm_n})")
            else:
                existing_n = int(existing_n_raw)
                if existing_n != ttm_n:
                    print(f"    {hz}: WARN n_points mismatch "
                          f"(leaderboard={existing_n}, TTM={ttm_n})")

    # deltas in percentage points. NaN propagates naturally for Bitbrains
    # ensemble (which is N/A in the leaderboard).
    df["delta_ttm_naive_pp"] = (df["ttm_r2"] - df["naive_sub_r2"]) * 100
    df["delta_ttm_ens_pp"]   = (df["ttm_r2"] - df["ensemble_sub_r2"]) * 100
    df["delta_ttm_c2_pp"]    = (df["ttm_r2"] - df["chronos2_r2"]) * 100
    if "timesfm_r2" in df.columns:
        df["delta_ttm_tfm_pp"] = (df["ttm_r2"] - df["timesfm_r2"]) * 100
    else:
        df["delta_ttm_tfm_pp"] = float("nan")

    df.to_csv(LEADERBOARD_OUT, index=False)
    print(f"\nwrote {LEADERBOARD_OUT}")

    # preview — only show the columns directly relevant to TTM.
    cols = ["dataset", "horizon",
            "naive_sub_r2", "ensemble_sub_r2",
            "chronos2_r2", "timesfm_r2", "ttm_r2",
            "delta_ttm_naive_pp", "delta_ttm_ens_pp",
            "delta_ttm_c2_pp", "delta_ttm_tfm_pp"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.float_format", "{:.4f}".format,
                            "display.width", 200):
        print()
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
