"""
f4_v1b_inspect_f3_eval.py — Schema-agnostic inspection of f3_eval_lora_rank8.json.

Just shows what's actually in the file. No assumptions about layout.

Run on Vast:
    cd /workspace/kubernetes-cpu-ensemble-thesis
    python phase_f/scripts/f4_v1b_inspect_f3_eval.py
"""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))

CANDIDATE_FILES = [
    PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.json",
    PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.csv",
    PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.parquet",
    PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.npz",
]


def describe(obj, depth=0, max_depth=4, key_path=""):
    """Recursively describe a nested JSON structure. Show types, sizes, and
    a small sample of values at each level. Stop at max_depth."""
    indent = "  " * depth
    if depth > max_depth:
        print(f"{indent}... (truncated at depth {max_depth})")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}dict with {len(keys)} keys: {keys[:10]}"
              f"{' ...' if len(keys) > 10 else ''}")
        for k in keys[:5]:                   # first 5 keys only
            print(f"{indent}  [{k!r}] ->")
            describe(obj[k], depth + 2, max_depth, key_path + f".{k}")
        if len(keys) > 5:
            print(f"{indent}  ... ({len(keys) - 5} more keys not shown)")
    elif isinstance(obj, list):
        n = len(obj)
        print(f"{indent}list of len {n}")
        if n > 0:
            # check if it's a list of numbers, list of dicts, etc.
            sample = obj[0]
            print(f"{indent}  element[0] type: {type(sample).__name__}")
            if isinstance(sample, (int, float, str, bool)):
                # show first few values
                preview = obj[:5]
                print(f"{indent}  first values: {preview}")
                if n > 5:
                    print(f"{indent}  ... (length {n})")
            elif isinstance(sample, list):
                # nested list - describe one level
                sub_n = len(sample)
                print(f"{indent}  element[0] is list of len {sub_n}")
                if sub_n > 0 and isinstance(sample[0], (int, float)):
                    print(f"{indent}  flat 2D array? shape ~ ({n}, {sub_n})")
                    print(f"{indent}  element[0] first values: {sample[:5]}")
            elif isinstance(sample, dict):
                print(f"{indent}  element[0]:")
                describe(sample, depth + 2, max_depth, key_path + "[0]")
    elif isinstance(obj, (int, float, bool)):
        print(f"{indent}{type(obj).__name__}: {obj}")
    elif isinstance(obj, str):
        if len(obj) > 80:
            print(f"{indent}str (len {len(obj)}): {obj[:80]}...")
        else:
            print(f"{indent}str: {obj!r}")
    elif obj is None:
        print(f"{indent}None")
    else:
        print(f"{indent}{type(obj).__name__}: {obj}")


def main():
    print("=" * 70)
    print("F3 EVAL FILE INSPECTOR")
    print("=" * 70)

    print("\nLooking for F3-FT eval files in phase_f/data/ ...")
    found = []
    for p in CANDIDATE_FILES:
        if p.exists():
            sz_kb = p.stat().st_size / 1024
            found.append((p, sz_kb))
            print(f"  found: {p}  ({sz_kb:.1f} KB)")
    if not found:
        # broader sweep
        print("  no canonical candidates found; sweeping for f3_* files ...")
        for p in (PROJECT_ROOT / "phase_f" / "data").iterdir():
            if "f3" in p.name.lower():
                sz_kb = p.stat().st_size / 1024
                found.append((p, sz_kb))
                print(f"  found: {p.name}  ({sz_kb:.1f} KB)")

    if not found:
        print("\nFATAL: no F3 files found in phase_f/data/")
        return

    for path, sz_kb in found:
        if "lora_rank8" not in str(path):
            continue
        print(f"\n{'='*70}")
        print(f"INSPECTING: {path}")
        print(f"size: {sz_kb:.1f} KB")
        print("=" * 70)

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            print("\n--- top-level structure ---")
            describe(data, depth=0, max_depth=4)

            # also dump raw first 2000 chars so we see the literal format
            print("\n--- raw first 2000 chars ---")
            with open(path) as f:
                raw = f.read(2000)
            print(raw)
            if sz_kb * 1024 > 2000:
                print("... (truncated)")

        elif path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            print(f"\nshape: {df.shape}")
            print(f"columns: {list(df.columns)}")
            print(f"dtypes:\n{df.dtypes}")
            print(f"head:\n{df.head()}")

        elif path.suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            print(f"\nshape: {df.shape}")
            print(f"columns: {list(df.columns)}")
            print(f"dtypes:\n{df.dtypes}")
            print(f"head:\n{df.head()}")

        elif path.suffix == ".npz":
            import numpy as np
            data = np.load(path, allow_pickle=True)
            print(f"\nkeys: {list(data.keys())}")
            for k in list(data.keys())[:5]:
                arr = data[k]
                print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
                if arr.size <= 20:
                    print(f"     values: {arr}")
                else:
                    print(f"     first 5: {arr.flat[:5]}")


if __name__ == "__main__":
    main()
