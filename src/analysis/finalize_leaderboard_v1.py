"""
finalize_leaderboard_v1.py — Trim same_sample_leaderboard_v3.csv down to the
public 12-row 5-model leaderboard.

Reads same_sample_leaderboard_v3.csv (which has the kitchen-sink set of
columns added by the chronos2/timesfm/ttm mergers) and writes leaderboard_v1.csv
with just the columns we want to show in the thesis: the five model R²s plus
deltas vs naive.

Five models:
    - Naive (subsample baseline, same K points as the foundation models)
    - NNLS Ensemble (own model, full-test on Alibaba/ByteDance, NaN on Bitbrains)
    - Chronos-2 zero-shot
    - TimesFM zero-shot
    - Granite TTM zero-shot

Run on Colab AFTER the three mergers have produced v3:
    !python finalize_leaderboard_v1.py
"""

import os

import pandas as pd

BASE = "/content/drive/MyDrive/k8s-ensemble-forecast"

LEADERBOARD_IN  = os.path.join(BASE, "same_sample_leaderboard_v3.csv")
LEADERBOARD_OUT = os.path.join(BASE, "leaderboard_v1.csv")

# horizon ordering for the final table. CSVs sort horizons alphabetically by
# default which puts 120min first — fix that here.
HORIZON_ORDER = ["10min", "30min", "60min", "120min"]
DATASET_ORDER = ["alibaba", "bitbrains", "bytedance"]


def main():
    print("=" * 64)
    print("FINALIZE leaderboard_v1.csv")
    print("=" * 64)

    df = pd.read_csv(LEADERBOARD_IN)
    print(f"  loaded {len(df)} rows from {LEADERBOARD_IN}")

    # bail loudly if any of the 5 model columns are missing — the merger
    # chain didn't finish properly and we'd silently drop NaN columns later.
    required = ["dataset", "horizon", "n_points",
                "naive_sub_r2", "ensemble_sub_r2",
                "chronos2_r2", "timesfm_r2", "ttm_r2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"  ERROR: missing columns in v3: {missing}\n"
                         f"  did you run the chronos2/timesfm/ttm mergers?")

    # rename to the friendlier names we'll use in the thesis table.
    out = df.rename(columns={
        "naive_sub_r2":    "naive_r2",
        "ensemble_sub_r2": "nnls_ensemble_r2",
        "chronos2_r2":     "chronos2_r2",
        "timesfm_r2":      "timesfm_r2",
        "ttm_r2":          "granite_ttm_r2",
    })[[
        "dataset", "horizon", "n_points",
        "naive_r2", "nnls_ensemble_r2",
        "chronos2_r2", "timesfm_r2", "granite_ttm_r2",
    ]].copy()

    # deltas vs naive in pp. NaN naturally propagates where ensemble is N/A.
    out["delta_nnls_naive_pp"]    = (out["nnls_ensemble_r2"] - out["naive_r2"]) * 100
    out["delta_chronos2_naive_pp"] = (out["chronos2_r2"] - out["naive_r2"]) * 100
    out["delta_timesfm_naive_pp"]  = (out["timesfm_r2"]  - out["naive_r2"]) * 100
    out["delta_ttm_naive_pp"]      = (out["granite_ttm_r2"] - out["naive_r2"]) * 100

    # round R² to 4dp, deltas to 2dp. matches the precision used elsewhere
    # in the thesis tables.
    r2_cols = ["naive_r2", "nnls_ensemble_r2",
               "chronos2_r2", "timesfm_r2", "granite_ttm_r2"]
    delta_cols = [c for c in out.columns if c.startswith("delta_")]
    out[r2_cols] = out[r2_cols].round(4)
    out[delta_cols] = out[delta_cols].round(2)

    # ordered sort: dataset first (alibaba/bitbrains/bytedance), then horizon
    # (10/30/60/120). use a rank-map approach so unknown values land at the
    # end with their original string preserved (Categorical silently turns
    # them into NaN which is too easy to miss).
    #
    # filter NaN/None before sorting — sorted() crashes with TypeError when
    # a set contains both strings and NaN floats (str < float is undefined).
    # report NaN counts separately. count over rows, not over the
    # deduplicated set, because np.nan is a singleton in pandas — set()
    # collapses all blank cells into a single entry.
    unknown_ds_set = set(out["dataset"]) - set(DATASET_ORDER)
    unknown_hz_set = set(out["horizon"]) - set(HORIZON_ORDER)
    unknown_ds = sorted(v for v in unknown_ds_set if isinstance(v, str))
    unknown_hz = sorted(v for v in unknown_hz_set if isinstance(v, str))
    n_nan_ds = (~out["dataset"].apply(lambda v: isinstance(v, str))).sum()
    n_nan_hz = (~out["horizon"].apply(lambda v: isinstance(v, str))).sum()
    if unknown_ds:
        print(f"  WARN: unexpected dataset values (kept, sorted to end): "
              f"{unknown_ds}")
    if unknown_hz:
        print(f"  WARN: unexpected horizon values (kept, sorted to end): "
              f"{unknown_hz}")
    if n_nan_ds:
        print(f"  WARN: {n_nan_ds} row(s) have non-string dataset "
              f"(blank cell or NaN, kept, sorted to end)")
    if n_nan_hz:
        print(f"  WARN: {n_nan_hz} row(s) have non-string horizon "
              f"(blank cell or NaN, kept, sorted to end)")

    ds_rank = {d: i for i, d in enumerate(DATASET_ORDER)}
    hz_rank = {h: i for i, h in enumerate(HORIZON_ORDER)}
    out["_ds_sort"] = out["dataset"].map(lambda d: ds_rank.get(d, 999))
    out["_hz_sort"] = out["horizon"].map(lambda h: hz_rank.get(h, 999))
    out = (out.sort_values(["_ds_sort", "_hz_sort"])
              .drop(columns=["_ds_sort", "_hz_sort"])
              .reset_index(drop=True))

    if len(out) != 12:
        print(f"  WARN: expected 12 rows but got {len(out)}. "
              f"check that all (dataset, horizon) cells are populated.")

    out.to_csv(LEADERBOARD_OUT, index=False)
    print(f"\nwrote {LEADERBOARD_OUT}  ({len(out)} rows, {len(out.columns)} cols)")

    # preview the whole thing — it's only 12 rows.
    with pd.option_context("display.float_format", "{:.4f}".format,
                            "display.width", 220,
                            "display.max_columns", None):
        print()
        print(out.to_string(index=False))

    # quick per-horizon "winner" summary — handy for the discussion section.
    print("\n--- best model per (dataset, horizon) ---")
    model_cols = ["naive_r2", "nnls_ensemble_r2",
                  "chronos2_r2", "timesfm_r2", "granite_ttm_r2"]
    for _, row in out.iterrows():
        scores = {m: row[m] for m in model_cols if pd.notna(row[m])}
        ds = str(row["dataset"])
        hz = str(row["horizon"])
        if not scores:
            print(f"  {ds:10s} {hz:>6s}  winner: <all models NaN>")
            continue
        winner = max(scores, key=scores.get)
        print(f"  {ds:10s} {hz:>6s}  "
              f"winner: {winner:20s}  R²={scores[winner]:.4f}")


if __name__ == "__main__":
    main()
