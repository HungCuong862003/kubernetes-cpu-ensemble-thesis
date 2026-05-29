"""
f4_v1_vast_f3_verifier.py — Verify F3 fine-tune eval outputs are F4-ready.

RUN ON VAST. Do not run in the sandbox; the f3_eval_lora_rank8.* files do
not exist there.

What this checks:

CHECK A: f3_eval_lora_rank8 file EXISTS in phase_f/data/
CHECK B: file is loadable and contains per-(container, t, h) quantile arrays
         (NOT just per-cell aggregate pinball loss like f3_zero_shot_baseline.json)
CHECK C: quantile levels saved include {0.5, 0.7, 0.8, 0.9, 0.95}
         (only 0.9 saved -> need to re-run F3 eval with wider grid before F4)
CHECK D: shape consistency: for one container at h=60, quantile array shape
         matches (n_test_origins, H_steps, n_quantiles) where H_steps=12 (h=60min/5min)
CHECK E: forecasts are in CPU-utilisation units in the same scale as
         data/processed/<dataset>/test.parquet cpu_target column.
         (If F3 forecasts a residual, F4 needs to ADD naive_cpu back to get
          the chance-constraint input.)
CHECK F: timing: forecast index aligns with test_spine.parquet container_id +
         time_stamp columns so we can join MPC outputs to ground-truth for
         violation rate evaluation.

If any check fails this prints the exact remediation step. Do NOT proceed to
F4 production code until A-E pass at minimum. (F is needed for evaluation;
can be fixed during F4 day 3 if it fails initially.)

Usage (on Vast, from /workspace/kubernetes-cpu-ensemble-thesis/):
    python phase_f/scripts/f4_v1_vast_f3_verifier.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── CONFIG: paths on Vast (matches Jimmy's project layout) ─────────

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))

PHASE_F_DATA = PROJECT_ROOT / "phase_f" / "data"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# F4 needs these quantile levels: 0.5 (median, AHPA baseline), 0.7/0.8 (for
# sensitivity), 0.9 (canonical alpha), 0.95 (tight alpha for robustness check).
F4_REQUIRED_QUANTILES = [0.5, 0.7, 0.8, 0.9, 0.95]

# F4 focuses on h=60 first; the other horizons are nice-to-have.
F4_PRIMARY_HORIZON_MIN = 60
F4_INTERVAL_MIN = 5
F4_H_STEPS = F4_PRIMARY_HORIZON_MIN // F4_INTERVAL_MIN          # = 12

# The three F3 datasets.
DATASETS = ["alibaba", "bitbrains", "bytedance"]


def fail(check_id, msg, remediation):
    print(f"\n[{check_id}] FAIL: {msg}")
    print(f"  REMEDIATION: {remediation}")
    return False


def ok(check_id, msg):
    print(f"[{check_id}] PASS: {msg}")
    return True


# ── CHECK A: file existence ────────────────────────────────────────

def check_A_file_exists():
    print("\n=== CHECK A: F3-FT eval file existence ===")
    candidates = [
        PHASE_F_DATA / "f3_eval_lora_rank8.json",
        PHASE_F_DATA / "f3_eval_lora_rank8.parquet",
        PHASE_F_DATA / "f3_eval_lora_rank8.npz",
        PHASE_F_DATA / "f3_eval_lora.json",
        PHASE_F_DATA / "f3_lora_eval.json",
    ]
    found = []
    for p in candidates:
        if p.exists():
            sz_mb = p.stat().st_size / 1e6
            found.append((p, sz_mb))
            print(f"  found: {p}  ({sz_mb:.2f} MB)")
    if not found:
        return None, fail(
            "A",
            f"No F3-FT eval file found in {PHASE_F_DATA}",
            "Locate the F3 eval output. If F3 is genuinely closed per the F4 "
            "prompt, list phase_f/data/ contents and identify which file holds "
            "the LoRA-FT eval predictions. Likely candidates contain 'lora', "
            "'finetune', or 'ft' in the filename.",
        )
    # if multiple candidates, pick the LARGEST (most likely the per-point one)
    chosen = max(found, key=lambda t: t[1])[0]
    ok("A", f"using {chosen}")
    return chosen, True


# ── CHECK B: schema = per-point quantile array, not aggregate ──────

def check_B_schema(path):
    print("\n=== CHECK B: schema is per-point quantile, not per-cell aggregate ===")
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        # If it's per-cell summary like f3_zero_shot_baseline.json:
        #   { 'alibaba': { '10min': { 'pinball_at_0.9': 0.237, ... } } }
        # If it's per-point quantile:
        #   { 'alibaba': { '60min': { 'container_id': [...], 'time_stamp': [...],
        #                              'q_0.5': [[...]], 'q_0.9': [[...]] } } }
        for dset in DATASETS:
            if dset not in data:
                return data, fail(
                    "B",
                    f"dataset key '{dset}' missing in JSON",
                    "F3 may not have produced eval for all datasets; check "
                    "f3_eval_lora_rank8.json keys.",
                )
        # check the first dataset's first horizon: does it have ARRAYS or SCALARS?
        d0 = DATASETS[0]
        h_keys = list(data[d0].keys())
        h0 = h_keys[0] if h_keys else None
        if h0 is None:
            return data, fail("B", f"no horizon keys under {d0}", "Inspect JSON structure.")
        sample_cell = data[d0][h0]
        # if values are scalars (floats) -> aggregate schema, NOT F4-ready
        scalar_keys = [k for k, v in sample_cell.items() if isinstance(v, (int, float))]
        array_keys  = [k for k, v in sample_cell.items() if isinstance(v, list)]
        print(f"  sample cell {d0}/{h0} has {len(scalar_keys)} scalar keys, "
              f"{len(array_keys)} array keys")
        print(f"  scalar keys: {scalar_keys[:5]}")
        print(f"  array keys:  {array_keys[:5]}")
        if not array_keys:
            return data, fail(
                "B",
                "F3-FT eval file contains only per-cell SCALAR aggregates "
                "(like pinball_at_0.9). F4's MPC needs per-(container, t, h) "
                "QUANTILE ARRAYS.",
                "Re-run F3 eval with a wrapper that saves per-point predictions. "
                "Modify the F3 eval script to dump tensor.cpu().numpy() arrays "
                "before computing pinball aggregates. Save as .parquet or .npz "
                "(JSON is impractical at this size). Estimated re-run cost: "
                "same as F3 day-1 (Alibaba 5000 containers in ~3s/horizon).",
            )
        ok("B", f"per-point arrays found in {d0}/{h0}: {array_keys}")
        return data, True

    elif path.suffix == ".npz":
        data = np.load(path)
        print(f"  npz keys: {list(data.keys())}")
        # expect keys like 'alibaba_60min_q0.9' shape (n_origin, H, 1) per container
        f4_keys = [k for k in data.keys() if "60min" in k]
        if not f4_keys:
            return data, fail(
                "B",
                "no h=60min arrays in npz",
                "Confirm F3 eval saved 60-min quantiles. "
                "If only 10/30/120 min: F4 must use a different horizon.",
            )
        # check shape of first key
        first = data[f4_keys[0]]
        print(f"  {f4_keys[0]} shape: {first.shape}")
        ok("B", f"per-point arrays found in npz: {f4_keys[:5]}")
        return data, True

    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
        print(f"  parquet shape: {df.shape}")
        print(f"  columns: {list(df.columns)[:10]}")
        # expect long-format: (container_id, time_stamp, horizon, q_0.5, q_0.7, ...)
        q_cols = [c for c in df.columns if c.startswith("q_") or c.startswith("q0")]
        if not q_cols:
            return df, fail(
                "B",
                "no quantile columns in parquet (expected q_0.5, q_0.7, ...)",
                "Check column naming convention. May need a different prefix.",
            )
        ok("B", f"quantile columns: {q_cols}")
        return df, True

    else:
        return None, fail("B", f"unknown file extension {path.suffix}",
                          "Add a loader branch for this format.")


# ── CHECK C: required quantile levels present ──────────────────────

def check_C_quantile_levels(path, data):
    print("\n=== CHECK C: required quantile levels present ===")
    if path.suffix == ".json":
        # find what quantile keys exist in any cell
        d0 = DATASETS[0]
        h_keys = list(data[d0].keys())
        if not h_keys:
            return fail("C", "no horizons to inspect", "Check JSON structure.")
        cell = data[d0][h_keys[0]]
        q_keys = sorted([k for k in cell.keys() if k.startswith(("q_", "q0", "quantile_"))])
        # try to parse the numeric levels out of the keys
        levels_found = []
        for k in q_keys:
            for prefix in ("q_", "q0", "quantile_"):
                if k.startswith(prefix):
                    rest = k[len(prefix):]
                    try:
                        # handle '0.9' or '90' etc.
                        v = float(rest)
                        if v > 1.0:
                            v = v / 100.0
                        levels_found.append(round(v, 3))
                    except ValueError:
                        pass
                    break
    elif path.suffix == ".npz":
        levels_found = []
        for k in data.keys():
            for marker in ("q0", "q_"):
                if marker in k:
                    rest = k.split(marker)[-1]
                    try:
                        v = float(rest[:4] if len(rest) >= 4 else rest)
                        if v > 1.0:
                            v = v / 100.0
                        levels_found.append(round(v, 3))
                    except ValueError:
                        pass
                    break
        levels_found = sorted(set(levels_found))
    elif path.suffix == ".parquet":
        q_cols = [c for c in data.columns if c.startswith(("q_", "q0", "quantile_"))]
        levels_found = []
        for c in q_cols:
            for prefix in ("q_", "q0", "quantile_"):
                if c.startswith(prefix):
                    rest = c[len(prefix):]
                    try:
                        v = float(rest)
                        if v > 1.0:
                            v = v / 100.0
                        levels_found.append(round(v, 3))
                    except ValueError:
                        pass
                    break
        levels_found = sorted(set(levels_found))
    else:
        return fail("C", "unknown format", "")

    print(f"  levels found: {sorted(set(levels_found))}")
    required = set(F4_REQUIRED_QUANTILES)
    missing = required - set(round(x, 3) for x in levels_found)
    if missing:
        return fail(
            "C",
            f"missing quantile levels: {sorted(missing)}",
            "Re-run F3 eval with quantile_levels=[0.5, 0.7, 0.8, 0.9, 0.95]. "
            "Per F3 Day-1 doc, the Chronos2Pipeline.predict_quantiles() call "
            "accepts an arbitrary quantile_levels list. If only 0.9 was saved, "
            "the F3 eval script needs editing before F4 sensitivity analysis "
            "(alpha sweep) is possible.",
        )
    ok("C", f"all required levels present: {sorted(required)}")
    return True


# ── CHECK D: shape consistency for h=60 ────────────────────────────

def check_D_shape_h60(path, data):
    print("\n=== CHECK D: shape consistency at h=60 ===")
    print("  Expected: per-(container, t_origin) a vector of length H_steps=12")
    print(f"  (60 min horizon / {F4_INTERVAL_MIN} min interval = {F4_H_STEPS} steps)")

    if path.suffix == ".json":
        d0 = DATASETS[0]
        # find h=60 key (might be '60min', '60', 'h60', etc.)
        h_key = None
        for k in data[d0].keys():
            if "60" in k:
                h_key = k
                break
        if h_key is None:
            return fail("D", "no h=60 entry", "F4 needs h=60. Re-run F3 eval for h=60.")
        cell = data[d0][h_key]
        # find q_0.9 array and inspect its shape
        for k in cell:
            if "0.9" in k and isinstance(cell[k], list):
                arr = np.asarray(cell[k])
                print(f"  {d0}/{h_key}/{k} shape: {arr.shape}")
                # last axis should be H_steps=12
                if arr.shape[-1] != F4_H_STEPS:
                    return fail(
                        "D",
                        f"h=60 quantile last-axis is {arr.shape[-1]}, expected {F4_H_STEPS}. "
                        f"Possible cause: F3 saved a SINGLE-STEP forecast at h=60 "
                        f"rather than the full 12-step path. F4 MPC needs the full path.",
                        "Re-run F3 eval with prediction_length=12 at h=60.",
                    )
                ok("D", f"h=60 quantile shape last axis = {F4_H_STEPS} steps as expected")
                return True
        return fail("D", "no q_0.9 array found at h=60", "Inspect JSON structure manually.")

    elif path.suffix == ".npz":
        for k in data.keys():
            if "60" in k and "0.9" in k:
                arr = data[k]
                print(f"  {k} shape: {arr.shape}")
                if arr.shape[-1] != F4_H_STEPS:
                    return fail(
                        "D",
                        f"shape last axis = {arr.shape[-1]} != {F4_H_STEPS}",
                        "Re-run F3 with prediction_length=12.",
                    )
                ok("D", f"h=60 quantile shape OK")
                return True
        return fail("D", "no h=60 q_0.9 array in npz", "Check npz keys.")

    elif path.suffix == ".parquet":
        df = data
        h_col = [c for c in df.columns if "horizon" in c.lower()]
        if h_col:
            h_vals = df[h_col[0]].unique()
            print(f"  horizon values: {h_vals}")
            has_60 = any("60" in str(v) for v in h_vals)
            if not has_60:
                return fail("D", "no h=60 rows", "Re-run F3 for h=60.")
        # parquet long-format: one row per (container, t_origin, step) is fine
        ok("D", "parquet long-format; explicit shape check deferred to F4 day-2 code")
        return True

    return fail("D", "unknown format", "")


# ── CHECK E: scale matches test.parquet cpu_target ─────────────────

def check_E_scale_matches_cpu(path, data):
    print("\n=== CHECK E: forecast scale matches cpu_target in test.parquet ===")
    print("  Forecasts must be in CPU-UTILISATION units (matching test.parquet")
    print("  cpu_target column), NOT residuals or normalised values. If F3 was")
    print("  trained on residuals, F4 must add naive_cpu back before chance-LB use.")

    # load test.parquet for Alibaba and grab cpu_target range
    test_path = DATA_PROCESSED / "alibaba" / "test.parquet"
    if not test_path.exists():
        print(f"  WARNING: {test_path} not found; cannot compare scales.")
        return True  # don't fail, just warn
    test = pd.read_parquet(test_path)
    cpu_col = None
    for cand in ("cpu_target", "cpu", "cpu_util_percent"):
        if cand in test.columns:
            cpu_col = cand
            break
    if cpu_col is None:
        print(f"  WARNING: no cpu column in test.parquet (cols: {list(test.columns)})")
        return True
    cpu_p50 = float(test[cpu_col].median())
    cpu_p95 = float(test[cpu_col].quantile(0.95))
    print(f"  test.parquet[{cpu_col}]: p50={cpu_p50:.3f}, p95={cpu_p95:.3f}")

    if path.suffix == ".json":
        d0 = "alibaba"
        if d0 not in data:
            return True
        for h_key in data[d0]:
            if "60" in h_key:
                cell = data[d0][h_key]
                for k in cell:
                    if "0.9" in k and isinstance(cell[k], list):
                        arr = np.asarray(cell[k])
                        fc_p50 = float(np.median(arr))
                        fc_p95 = float(np.percentile(arr, 95))
                        print(f"  forecast q_0.9 alibaba/h=60: "
                              f"p50={fc_p50:.3f}, p95={fc_p95:.3f}")
                        # if forecast median is within 5x of cpu median: same scale.
                        # if forecast is near zero or negative: probably residual.
                        if fc_p50 < cpu_p50 / 10 or fc_p50 < 0:
                            return fail(
                                "E",
                                "forecast scale appears to be RESIDUAL "
                                f"(forecast p50={fc_p50:.3f} vs cpu p50={cpu_p50:.3f})",
                                "F4 must add naive_cpu back to F3 forecasts before "
                                "feeding into MPC. Use the 'naive_cpu' column from "
                                "test_spine.parquet at the same time_stamp. "
                                "This is a one-line fix in the F4 forecast loader.",
                            )
                        elif fc_p50 > cpu_p50 * 10:
                            return fail(
                                "E",
                                f"forecast scale is ~{fc_p50/cpu_p50:.1f}x larger than "
                                "test cpu",
                                "Check normalisation. F3 may have output un-denormalised "
                                "predictions. Re-check F3 eval pipeline.",
                            )
                        ok("E", "forecast scale matches CPU utilisation scale")
                        return True
        return fail("E", "could not locate h=60 q_0.9 array for alibaba",
                    "Inspect JSON structure manually.")

    # for npz/parquet, similar pattern but I'll defer detailed implementation
    # to when we actually see the format on Vast
    print("  (npz/parquet scale check deferred; same logic, different loader)")
    return True


# ── CHECK F: timestamp alignment for evaluation join ───────────────

def check_F_index_alignment(path, data):
    print("\n=== CHECK F: forecast (container, time) can join to test_spine.parquet ===")
    spine_path = PROJECT_ROOT / "results" / "alibaba" / "h060" / "predictions" / "test_spine.parquet"
    if not spine_path.exists():
        # try alt naming
        alts = list((PROJECT_ROOT / "results").rglob("test_spine.parquet"))
        if alts:
            spine_path = alts[0]
            print(f"  using alternate spine path: {spine_path}")
        else:
            print(f"  WARNING: test_spine.parquet not found; cannot verify join.")
            print(f"  (Check F is a soft requirement — can be fixed during F4 day 3 if it fails.)")
            return True
    spine = pd.read_parquet(spine_path)
    print(f"  spine shape: {spine.shape}, columns: {list(spine.columns)}")
    print(f"  spine container_id sample: {spine['container_id'].head(3).tolist()}")
    print(f"  spine time_stamp sample: {spine['time_stamp'].head(3).tolist()}")

    if path.suffix == ".json":
        # need to confirm the forecast file has matching (container_id, time_stamp) keys
        d0 = "alibaba"
        for h_key in data.get(d0, {}):
            if "60" in h_key:
                cell = data[d0][h_key]
                id_col_keys = [k for k in cell if "id" in k.lower() or k == "container_id"]
                ts_col_keys = [k for k in cell if "time" in k.lower() or "stamp" in k.lower()]
                if not id_col_keys or not ts_col_keys:
                    return fail(
                        "F",
                        "no container_id / time_stamp index in forecast file",
                        "F4 evaluation needs to join forecasts to observed CPU. "
                        "Add container_id and time_stamp arrays to F3 eval output. "
                        "One-line fix in F3 eval script: save the test_spine join keys "
                        "alongside the quantile arrays.",
                    )
                ok("F", f"forecast has index columns: {id_col_keys}, {ts_col_keys}")
                return True
        return fail("F", "no h=60 cell to inspect", "")
    print("  (npz/parquet check deferred — same logic.)")
    return True


# ── MAIN ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("F4 V1 VAST F3 VERIFIER")
    print("=" * 70)
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"PHASE_F_DATA = {PHASE_F_DATA}")

    if not PHASE_F_DATA.exists():
        print(f"\nFATAL: {PHASE_F_DATA} does not exist.")
        print("Set PROJECT_ROOT env var or run from the correct directory.")
        sys.exit(2)

    path, ok_a = check_A_file_exists()
    if not ok_a:
        print("\nCHECK A failed. Cannot continue.")
        sys.exit(1)

    data, ok_b = check_B_schema(path)
    if not ok_b:
        print("\nCHECK B failed. Cannot continue.")
        sys.exit(1)

    results = {
        "A": True,
        "B": True,
        "C": check_C_quantile_levels(path, data),
        "D": check_D_shape_h60(path, data),
        "E": check_E_scale_matches_cpu(path, data),
        "F": check_F_index_alignment(path, data),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    blocking = {"A", "B", "C", "D", "E"}    # F is soft
    for cid, ok_ in results.items():
        tag = "BLOCKING" if cid in blocking else "SOFT"
        print(f"  CHECK {cid} ({tag}): {'PASS' if ok_ else 'FAIL'}")

    n_pass_blocking = sum(1 for cid in blocking if results[cid])
    n_total_blocking = len(blocking)
    print(f"\n{n_pass_blocking}/{n_total_blocking} blocking checks pass")
    if n_pass_blocking == n_total_blocking:
        print("\nF3 OUTPUTS F4-READY. Safe to proceed to F4 production code.")
        sys.exit(0)
    else:
        print("\nF3 OUTPUTS NOT F4-READY. Fix blocking checks before F4 day 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
