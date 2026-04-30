"""
build_leaderboard.py

Combines the three Chronos-2 zero-shot JSON result files into a single
flat CSV that Chapter 5 can cite directly.

Inputs (expected under src/day2_chronos2/):
    k20_alibaba_v2.json      (K=20 origins)
    k50_bitbrains_check.json (K=50 origins)
    k50_bytedance_check.json (K=50 origins - preferred over K=20 for bytedance)

Output:
    chronos2_leaderboard.csv

Run:
    python build_leaderboard.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Config: which files to read, dataset labels, and which K to report.
# Keep this as a list of tuples so it's easy to eyeball and edit.
# ---------------------------------------------------------------------------
INPUT_DIR = Path("src/day2_chronos2")

CONFIGS = [
    # (dataset_name, filename, K_origins)
    ("alibaba",   "k20_alibaba_v2.json",     20),
    ("bitbrains", "k50_bitbrains_check.json", 50),
    ("bytedance", "k50_bytedance_check.json", 50),  # prefer K=50 for bytedance
]

# Horizons we expect to find in every file. If one is missing we want to know.
EXPECTED_HORIZONS = ["10min", "30min", "60min", "120min"]

OUTPUT_CSV = "chronos2_leaderboard.csv"


def load_one_json(path):
    """Load a single JSON and do minimal sanity checks before returning it.
    Returns the parsed dict. Raises with a clear message on any issue."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Quick structural check - better to blow up here with a clear message
    # than to KeyError deep inside the row loop.
    required_top_keys = ["chronos2_zero_shot", "baselines", "cadence_min", "ens_label"]
    for k in required_top_keys:
        if k not in data:
            raise KeyError(f"{path.name} is missing top-level key '{k}'")

    return data


def build_rows_for_dataset(dataset_name, data, k_origins):
    """Turn one JSON's contents into a list of row-dicts, one row per horizon.
    Doing this one dataset at a time keeps the loop readable."""

    rows_for_this_dataset = []

    cadence_min = data["cadence_min"]
    ens_label   = data["ens_label"]
    chronos_block  = data["chronos2_zero_shot"]
    baselines_block = data["baselines"]

    # Loop in the expected horizon order, not dict iteration order,
    # so the CSV comes out sorted 10 -> 30 -> 60 -> 120 every time.
    for hz in EXPECTED_HORIZONS:

        if hz not in chronos_block:
            print(f"  WARNING: horizon {hz} missing from chronos block in {dataset_name}, skipping")
            continue
        if hz not in baselines_block:
            print(f"  WARNING: horizon {hz} missing from baselines in {dataset_name}, skipping")
            continue

        chronos_row = chronos_block[hz]
        baseline_row = baselines_block[hz]

        # Pull the fields we care about. Use .get for things we do not strictly
        # need so the script does not die if the JSON schema drifts slightly.
        naive_full       = baseline_row["naive"]
        project_baseline = baseline_row["hetero_ens"]

        chronos_r2   = chronos_row["r2_mean"]
        chronos_mae  = chronos_row["mae_mean"]
        naive_sub    = chronos_row["r2_naive_subsample"]
        n_points     = chronos_row["n_points"]
        coverage     = chronos_row.get("pct_inside_p10_p90", None)

        # Deltas in percentage-points. Keeping them here means Chapter 5
        # does not have to recompute them.
        delta_vs_baseline_pp  = (chronos_r2 - project_baseline) * 100.0
        delta_vs_naive_sub_pp = (chronos_r2 - naive_sub) * 100.0

        row = {
            "dataset": dataset_name,
            "horizon": hz,
            "cadence_min": cadence_min,
            "k_origins": k_origins,
            "n_points": n_points,
            "naive_full": naive_full,
            "naive_sub": naive_sub,
            "project_baseline": project_baseline,
            "project_baseline_label": ens_label,
            "chronos2_r2": chronos_r2,
            "chronos2_mae": chronos_mae,
            "p10_p90_coverage": coverage,
            "delta_vs_baseline_pp": delta_vs_baseline_pp,
            "delta_vs_naive_sub_pp": delta_vs_naive_sub_pp,
        }
        rows_for_this_dataset.append(row)

    return rows_for_this_dataset


def main():
    print("=" * 70)
    print("build_leaderboard.py  -- combining Chronos-2 zero-shot JSONs")
    print("=" * 70)
    print(f"Input dir : {INPUT_DIR}")
    print(f"Output    : {OUTPUT_CSV}")
    print()

    # If the default input dir does not exist but the files are sitting next to
    # the script (common on laptops), fall back to '.' so the script still runs.
    input_dir = INPUT_DIR
    if not input_dir.exists():
        print(f"NOTE: {input_dir} does not exist. Falling back to current directory.")
        input_dir = Path(".")

    all_rows = []

    for dataset_name, fname, k in CONFIGS:
        path = input_dir / fname
        print(f"Loading {path} ...")

        try:
            data = load_one_json(path)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            # Print a clear message and bail out - don't write a half-built CSV.
            print(f"ERROR while loading {path}: {e}")
            sys.exit(1)

        # Quick sanity print so we can eyeball a few numbers during the run.
        print(f"  dataset       : {data.get('dataset')}")
        print(f"  cadence_min   : {data.get('cadence_min')}")
        print(f"  num_origins   : {data.get('num_origins')}  (config expects K={k})")
        print(f"  horizons seen : {list(data['chronos2_zero_shot'].keys())}")

        # Soft check: warn if the K in the JSON doesn't match what we expected.
        json_k = data.get("num_origins")
        if json_k is not None and json_k != k:
            print(f"  WARNING: num_origins in JSON ({json_k}) != configured K ({k}).")

        rows = build_rows_for_dataset(dataset_name, data, k)
        print(f"  -> built {len(rows)} rows")
        all_rows.extend(rows)
        print()

    # ----- Build the dataframe and save -------------------------------------
    df = pd.DataFrame(all_rows)

    # Put columns in a stable, human-friendly order (pandas does not guarantee
    # insertion order across versions strictly, so spell it out).
    column_order = [
        "dataset",
        "horizon",
        "cadence_min",
        "k_origins",
        "n_points",
        "naive_full",
        "naive_sub",
        "project_baseline",
        "project_baseline_label",
        "chronos2_r2",
        "chronos2_mae",
        "p10_p90_coverage",
        "delta_vs_baseline_pp",
        "delta_vs_naive_sub_pp",
    ]
    df = df[column_order]

    # Sort so the CSV is deterministic regardless of CONFIGS ordering.
    horizon_order = {h: i for i, h in enumerate(EXPECTED_HORIZONS)}
    df["_hz_sort"] = df["horizon"].map(horizon_order)
    df = df.sort_values(by=["dataset", "_hz_sort"]).drop(columns="_hz_sort").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}  ({len(df)} rows, {len(df.columns)} columns)")
    print()

    # ----- Print it to stdout so it lands in the run log --------------------
    # Round the float-heavy columns for readability in the console. The CSV
    # on disk still has full precision.
    print("Leaderboard preview (rounded for display only):")
    preview = df.copy()
    for col in ["naive_full", "naive_sub", "project_baseline", "chronos2_r2",
                "chronos2_mae", "p10_p90_coverage",
                "delta_vs_baseline_pp", "delta_vs_naive_sub_pp"]:
        preview[col] = preview[col].round(4)
    print(preview.to_string(index=False))
    print()
    print("Done.")


if __name__ == "__main__":
    main()
