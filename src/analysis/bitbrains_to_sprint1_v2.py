from __future__ import annotations

"""
bitbrains_to_sprint1.py
=======================
Converts active Bitbrains VMs into the parquet format expected by sprint1_v9.py.

Prerequisite
------------
Run  bitbrains_xval.py --stage scan_vms  first.
That produces active_vms.csv (157 VMs after mean>15 % and std>3 % filter).

Usage
-----
python bitbrains_to_sprint1.py \
    --data_dir   /path/to/fastStorage/2013-8 \
    --active_csv /path/to/results/active_vms.csv \
    --out_dir    ./bitbrains_formatted

Output
------
./bitbrains_formatted/
    train.parquet   (60 % of time range)
    val.parquet     (20 %)
    test.parquet    (20 %)
    manifest.json   (row counts, VM counts, time range, sha256)

Required columns (sprint1_v9 load_data contract):
    container_id      str
    time_stamp        int64  (Unix seconds)
    cpu_util_percent  float32
    mem_util_percent  float32
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time as _time

import numpy as np
import pandas as pd


# ── column name constants ────────────────────────────────────────────────────
CPU_COL  = "CPU usage [%]"
MEM_COL  = "Memory usage [KB]"
CAP_COL  = "Memory capacity provisioned [KB]"
TIME_COL = "Timestamp [ms]"

REQUIRED_COLS = [TIME_COL, CPU_COL, MEM_COL, CAP_COL]

SPLIT = (0.60, 0.20, 0.20)          # train / val / test (temporal)
MIN_ROWS_PER_VM = 300                # ~25 h at 5-min resolution


# ── helpers ──────────────────────────────────────────────────────────────────

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading / trailing whitespace from all column names."""
    df.columns = df.columns.str.strip()
    return df


def _to_unix_seconds(ts: pd.Series) -> pd.Series:
    """
    Auto-detect whether a timestamp column is in seconds or milliseconds
    and return Unix seconds.

    Heuristic: Unix seconds for any date between 2000-01-01 and 2040-01-01
    fall in [946684800, 2208988800].  If the median is above 1e12, assume ms.

    Returns float64 (not int64) so that NaN rows survive until the
    downstream dropna() call.  The final int64 cast happens in save_parquet.
    """
    median = ts.median()
    if median > 1e12:                     # milliseconds
        return (ts // 1000).astype("float64")
    if 9e8 < median < 2.3e9:             # seconds (plausible Unix range)
        return ts.astype("float64")
    raise ValueError(
        f"Timestamp median {median:.0f} outside plausible Unix range — "
        f"cannot auto-detect seconds vs milliseconds."
    )


def load_vm(fpath: str, vm_id: str) -> pd.DataFrame | None:
    """
    Read one Bitbrains CSV and return a normalised DataFrame, or None on error.

    * Strips whitespace from column names before matching.
    * Auto-detects seconds vs milliseconds for the timestamp column.
    * Drops rows with NaN CPU or memory.
    * Clips mem_util to [0, 100].
    * Deduplicates on (container_id, time_stamp).
    """
    try:
        df = pd.read_csv(fpath, sep=";", header=0)
    except Exception as exc:
        print(f"  SKIP {vm_id}: read error — {exc}")
        return None

    df = _strip_columns(df)

    # Validate required columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            print(f"  SKIP {vm_id}: missing column '{col}'  "
                  f"(have: {list(df.columns)[:6]}…)")
            return None

    # Guard against zero / NaN capacity before division
    cap = df[CAP_COL].replace(0, np.nan)

    out = pd.DataFrame({
        "container_id":     f"bb_{vm_id}",
        "time_stamp":       _to_unix_seconds(df[TIME_COL]),
        "cpu_util_percent":  df[CPU_COL].astype("float32"),
        "mem_util_percent": (df[MEM_COL] / cap * 100)
                             .clip(0, 100).astype("float32"),
    })

    out = out.dropna(subset=["time_stamp", "cpu_util_percent", "mem_util_percent"])
    out["time_stamp"] = out["time_stamp"].astype("int64")

    # Deduplicate rows that share the same (container, timestamp)
    out = out.drop_duplicates(subset=["container_id", "time_stamp"], keep="first")

    out = out.sort_values("time_stamp").reset_index(drop=True)

    if len(out) < MIN_ROWS_PER_VM:
        print(f"  SKIP {vm_id}: only {len(out)} rows after cleaning "
              f"(need ≥{MIN_ROWS_PER_VM})")
        return None

    return out


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    60 / 20 / 20 split on the **global** time axis.

    sprint1_v9 asserts  train_max < val_min < test_min,
    so we cut on sorted time, not on container-level index.

    Raises ValueError if any split is empty (degenerate time range).
    """
    t_min = int(df["time_stamp"].min())
    t_max = int(df["time_stamp"].max())
    span  = t_max - t_min

    if span == 0:
        raise ValueError("All timestamps are identical — cannot split.")

    cut1 = t_min + int(span * SPLIT[0])
    cut2 = t_min + int(span * (SPLIT[0] + SPLIT[1]))

    train = df.loc[df["time_stamp"] <  cut1].copy()
    val   = df.loc[(df["time_stamp"] >= cut1) & (df["time_stamp"] < cut2)].copy()
    test  = df.loc[df["time_stamp"] >= cut2].copy()

    for name, part in [("train", train), ("val", val), ("test", test)]:
        if part.empty:
            raise ValueError(
                f"'{name}' split is empty — time range too narrow or skewed.  "
                f"t_min={t_min}  t_max={t_max}  cut1={cut1}  cut2={cut2}"
            )

    return train, val, test


def save_parquet(df: pd.DataFrame, path: str) -> int:
    """
    Save to zstd-compressed parquet.  Returns file size in bytes.

    Operates on a **copy** so the caller's DataFrame is not mutated.
    Keeps time_stamp as int64 to avoid int32 overflow on future data.
    """
    out = df.copy()
    out["time_stamp"]       = out["time_stamp"].astype("int64")
    out["cpu_util_percent"] = out["cpu_util_percent"].astype("float32")
    out["mem_util_percent"] = out["mem_util_percent"].astype("float32")

    out.to_parquet(path, index=False, compression="zstd")
    size = os.path.getsize(path)
    print(f"  saved {path}  ({size / 1e6:.1f} MB, {len(out):,} rows)")
    return size


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: str, n_vms: int, n_rows: int,
                   split_sizes: dict, t_min: int, t_max: int) -> None:
    """Write a manifest.json for downstream traceability."""
    manifest = {
        "generator":    "bitbrains_to_sprint1.py",
        "created_utc":  _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "vm_count":     n_vms,
        "total_rows":   n_rows,
        "time_range":   {"min_unix_s": t_min, "max_unix_s": t_max},
        "splits":       {},
    }
    for name in ("train", "val", "test"):
        fpath = os.path.join(out_dir, f"{name}.parquet")
        manifest["splits"][name] = {
            "rows":   split_sizes[name],
            "bytes":  os.path.getsize(fpath),
            "sha256": _sha256(fpath),
        }

    mpath = os.path.join(out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  saved {mpath}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bitbrains → sprint1_v9 parquet adapter")
    ap.add_argument("--data_dir",
                    default="/content/drive/MyDrive/Thesis/bitbrains/"
                            "fastStorage/2013-8",
                    help="Directory with Bitbrains VM CSVs")
    ap.add_argument("--active_csv",
                    default="/content/drive/MyDrive/Thesis/bitbrains/"
                            "results/active_vms.csv",
                    help="active_vms.csv from bitbrains_xval.py --stage scan_vms")
    ap.add_argument("--out_dir",
                    default="./bitbrains_formatted",
                    help="Output directory for train/val/test parquets")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load active VM list ──────────────────────────────────────────────────
    if not os.path.exists(args.active_csv):
        sys.exit(
            f"ERROR: active_vms.csv not found at {args.active_csv}\n"
            f"Run:  python bitbrains_xval.py --stage scan_vms  first."
        )

    active = pd.read_csv(args.active_csv)
    # Guard against NaN / float promotion in the vm_id column
    vm_ids = active["vm_id"].dropna().astype(str).tolist()
    # Strip ".0" suffix that appears if pandas read the column as float
    vm_ids = [v[:-2] if v.endswith(".0") else v for v in vm_ids]
    print(f"Active VMs to process: {len(vm_ids)}")

    # ── Load + convert all active VMs ────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    skipped = 0
    for i, vm_id in enumerate(vm_ids, 1):
        fpath = os.path.join(args.data_dir, f"{vm_id}.csv")
        if not os.path.exists(fpath):
            print(f"  SKIP {vm_id}: file not found")
            skipped += 1
            continue
        df_vm = load_vm(fpath, vm_id)
        if df_vm is not None:
            frames.append(df_vm)
        else:
            skipped += 1
        if i % 50 == 0 or i == len(vm_ids):
            print(f"  processed {i}/{len(vm_ids)}  "
                  f"(loaded {len(frames)}, skipped {skipped})")

    if not frames:
        sys.exit("ERROR: no valid VMs loaded — check paths and column names.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["container_id", "time_stamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    n_vms  = combined["container_id"].nunique()
    n_rows = len(combined)
    t_min  = int(combined["time_stamp"].min())
    t_max  = int(combined["time_stamp"].max())
    span_days = (t_max - t_min) / 86400

    print(f"\nCombined: {n_vms} VMs, {n_rows:,} rows")
    print(f"  time range: {t_min}–{t_max}  "
          f"({_dt.datetime.utcfromtimestamp(t_min).date()} to "
          f"{_dt.datetime.utcfromtimestamp(t_max).date()}, "
          f"{span_days:.1f} days)")

    # Sanity gate: timestamps must span at least 1 day for meaningful
    # temporal features.  Catches the "column labelled ms but is s" bug.
    if span_days < 1.0:
        sys.exit(
            f"ERROR: combined time span is only {span_days:.4f} days "
            f"({t_max - t_min} seconds).\n"
            f"This almost certainly means timestamps were divided "
            f"incorrectly.  Check _to_unix_seconds()."
        )

    # ── Temporal split (global time axis) ────────────────────────────────────
    train, val, test = temporal_split(combined)

    print(f"\nSplit sizes:  train={len(train):,}  "
          f"val={len(val):,}  test={len(test):,}")
    print(f"Time cuts (Unix s):  "
          f"train < {train['time_stamp'].max()}  "
          f"val < {val['time_stamp'].max()}")

    # sprint1_v9 assertion guard
    assert train["time_stamp"].max() < val["time_stamp"].min(), \
        "train/val overlap!"
    assert val["time_stamp"].max() < test["time_stamp"].min(), \
        "val/test overlap!"
    print("Temporal ordering check: OK")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\nSaving parquets …")
    save_parquet(train, os.path.join(args.out_dir, "train.parquet"))
    save_parquet(val,   os.path.join(args.out_dir, "val.parquet"))
    save_parquet(test,  os.path.join(args.out_dir, "test.parquet"))

    write_manifest(
        args.out_dir, n_vms, n_rows,
        {"train": len(train), "val": len(val), "test": len(test)},
        t_min, t_max,
    )

    print(f"\nDone.  Run sprint1_v9 with:")
    print(f"  python sprint1_v9.py --data_dir {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()