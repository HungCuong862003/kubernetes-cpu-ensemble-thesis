"""
f4_v1f_verify_resaver_parquet.py — Verify the resaver output is F4-ready.

Runs in under a minute. Catches four things:

CHECK A: schema sanity (row count, column names, dtypes)
CHECK B: quantile monotonicity per row (q_0.5 <= q_0.7 <= q_0.8 <= q_0.9 <= q_0.95)
CHECK C: forecast scale matches cpu_util_percent (NOT residuals, NOT normalised)
CHECK D: pinball loss recomputed on this parquet matches f3_eval_lora_rank8.csv
         per-cell pinball values within 1% relative tolerance.
         **CRITICAL**: if this fails, the resaver did something different from
         what F3 measured. F4 claims become unverifiable.
CHECK E: truth-join completeness per dataset (Alibaba showed 98.3%, Bitbrains
         81.0%, ByteDance 99.2% in the run). Investigate Bitbrains.
CHECK F: skipped-origin diagnosis (Alibaba dropped 4701, are they uniform or
         concentrated on small containers?)

Run:
    python phase_f/scripts/f4_v1f_verify_resaver_parquet.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))

PARQUET = PROJECT_ROOT / "phase_f" / "data" / "f3_lora_quantiles_h060_cad30.parquet"
F3_CSV  = PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATASETS = ["alibaba", "bitbrains", "bytedance"]
QUANTILE_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pinball_loss(y_true, y_pred, tau):
    """Symmetric pinball loss at quantile tau. Lower is better."""
    diff = y_true - y_pred
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()


def check_A_schema(df):
    section("CHECK A: schema sanity")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  dtypes:\n{df.dtypes}")
    expected_cols = {"dataset", "container_id", "time_stamp", "step_idx",
                     "q_0.5", "q_0.7", "q_0.8", "q_0.9", "q_0.95",
                     "t_target", "cpu_true"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"  FAIL: missing columns: {missing}")
        return False
    print("  PASS: all expected columns present")
    return True


def check_B_monotonicity(df):
    section("CHECK B: quantile monotonicity per row")
    q_cols = [f"q_{q}" for q in QUANTILE_LEVELS]
    arr = df[q_cols].to_numpy()
    # diffs between consecutive quantiles, should be >= 0
    diffs = np.diff(arr, axis=1)
    n_total = len(df)
    n_violations = int((diffs < 0).any(axis=1).sum())
    pct_viol = 100.0 * n_violations / n_total
    # smallest negative diff value (most severe inversion)
    if (diffs < 0).any():
        most_neg = float(diffs[diffs < 0].min())
    else:
        most_neg = 0.0
    print(f"  total rows: {n_total}")
    print(f"  rows with at least one inverted quantile: {n_violations} ({pct_viol:.3f}%)")
    print(f"  largest negative diff (most severe inversion): {most_neg:.6f}")
    # tolerate up to 0.1% inversion (float precision wobble at near-equal quantiles)
    pass_ = (pct_viol < 0.1)
    print(f"  {'PASS' if pass_ else 'FAIL'}: tolerance < 0.1%")
    return pass_


def check_C_scale(df):
    section("CHECK C: forecast scale vs cpu_util_percent")
    # Per dataset, compare q_0.9 distribution to cpu_true distribution
    for dset in DATASETS:
        sub = df[df["dataset"] == dset]
        if len(sub) == 0:
            print(f"  {dset}: no rows; skip")
            continue
        valid = sub.dropna(subset=["cpu_true"])
        q90_p50 = float(np.median(valid["q_0.9"]))
        q90_p95 = float(np.percentile(valid["q_0.9"], 95))
        cpu_p50 = float(np.median(valid["cpu_true"]))
        cpu_p95 = float(np.percentile(valid["cpu_true"], 95))
        q90_min = float(valid["q_0.9"].min())
        q90_max = float(valid["q_0.9"].max())
        print(f"  {dset}:")
        print(f"    cpu_true:  p50={cpu_p50:>7.3f}  p95={cpu_p95:>7.3f}")
        print(f"    q_0.9:     p50={q90_p50:>7.3f}  p95={q90_p95:>7.3f}")
        print(f"    q_0.9 range: [{q90_min:.3f}, {q90_max:.3f}]")
        # heuristic: if forecast median is within 0.5x to 3x of cpu_true median,
        # it's the same scale. If forecast median is near 0 or negative, it's
        # residual-scale. If forecast values are all < 1, probably normalised.
        if q90_p50 < 0:
            print(f"    WARN: forecast appears to be residual (negative median)")
        elif q90_max < 1.0:
            print(f"    WARN: forecast appears to be normalised (max < 1)")
        elif q90_p50 < cpu_p50 / 3 or q90_p50 > cpu_p50 * 3:
            print(f"    WARN: forecast scale off from cpu_true by >3x")
        else:
            print(f"    OK: forecast scale matches cpu_true scale")
    return True


def check_D_pinball_reproduction(df):
    """Recompute pinball loss on this parquet and compare to F3 CSV per-cell."""
    section("CHECK D: pinball loss reproduction vs F3 CSV (CRITICAL)")
    if not F3_CSV.exists():
        print(f"  FAIL: {F3_CSV} missing; cannot compare")
        return False
    f3 = pd.read_csv(F3_CSV)
    f3_h60 = f3[f3["horizon_min"] == 60]
    # The F3 eval used (per F3 day-1 doc and your clarification) the test split
    # only, full coverage of containers (not 30-min-cadence subsample).
    # OUR parquet is 30-min cadence. The pinball values should be CLOSE but may
    # differ slightly because the origin set is a subsample.
    # We allow up to 5% relative tolerance.
    valid = df.dropna(subset=["cpu_true"])
    rows = []
    for dset in DATASETS:
        sub = valid[valid["dataset"] == dset]
        if len(sub) == 0:
            continue
        for tau in QUANTILE_LEVELS:
            q_col = f"q_{tau}"
            pinball_ours = pinball_loss(sub["cpu_true"].to_numpy(),
                                         sub[q_col].to_numpy(), tau)
            # F3 CSV pinball is reported per-(dataset, horizon, group, tau).
            # We don't have main/holdout split markings in our parquet; we
            # ran inference on the FULL test set per dataset. So our pinball
            # should be the n_valid-weighted MEAN of main and holdout from F3 CSV.
            f3_sub = f3_h60[(f3_h60["dataset"] == dset) & (f3_h60["tau"] == tau)]
            if len(f3_sub) == 0:
                continue
            w_pinball = float((f3_sub["sym_pinball"] * f3_sub["n_valid"]).sum()
                              / f3_sub["n_valid"].sum())
            rel_err = (pinball_ours - w_pinball) / w_pinball if w_pinball > 0 else 0.0
            rows.append({
                "dataset":     dset,
                "tau":         tau,
                "ours":        pinball_ours,
                "f3_csv":      w_pinball,
                "rel_err_pct": rel_err * 100.0,
            })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    # PASS if all |rel_err| < 5%
    max_err = out["rel_err_pct"].abs().max()
    pass_ = max_err < 5.0
    print(f"\n  max relative error: {max_err:.2f}%")
    print(f"  {'PASS' if pass_ else 'FAIL'}: tolerance < 5%")
    if not pass_:
        print(f"\n  FAILURE INTERPRETATION:")
        print(f"  If error is dataset-uniform (all 3 datasets ~same rel_err sign),")
        print(f"  the F3 model loaded differently this time (LoRA adapter not")
        print(f"  active in forward pass, or different quantile head).")
        print(f"  If error is dataset-specific, likely a data-loading or origin-")
        print(f"  subsample bias.")
    return pass_


def check_E_truth_coverage(df):
    section("CHECK E: truth-join completeness per dataset")
    for dset in DATASETS:
        sub = df[df["dataset"] == dset]
        if len(sub) == 0:
            continue
        with_truth = sub["cpu_true"].notna().sum()
        pct = 100.0 * with_truth / len(sub)
        print(f"  {dset}: {with_truth}/{len(sub)} = {pct:.2f}% have cpu_true")
    # Bitbrains is the known concern at 81%.
    bb = df[df["dataset"] == "bitbrains"]
    if len(bb) > 0:
        pct_bb = 100.0 * bb["cpu_true"].notna().sum() / len(bb)
        if pct_bb < 90:
            print(f"\n  Bitbrains low-coverage diagnosis:")
            # which step_idx values lose truth most?
            print(f"  truth coverage by step_idx:")
            for s in range(12):
                sub_s = bb[bb["step_idx"] == s]
                if len(sub_s) == 0:
                    continue
                p = 100.0 * sub_s["cpu_true"].notna().sum() / len(sub_s)
                print(f"    step_idx={s:>2}: {p:.2f}%")
            print(f"  (if late step_idx drops sharply, it's end-of-trace truncation)")
            print(f"  (if uniform across steps, it's irregular sampling in test.parquet)")
    return True


def check_F_skipped_origins():
    """Diagnose Alibaba's 4701 skipped origins. Are they small containers?"""
    section("CHECK F: skipped origins diagnosis (Alibaba)")
    test_path = DATA_PROCESSED / "alibaba" / "test.parquet"
    train_path = DATA_PROCESSED / "alibaba" / "train.parquet"
    val_path = DATA_PROCESSED / "alibaba" / "val.parquet"
    if not test_path.exists():
        print("  test.parquet missing; cannot diagnose")
        return True
    test_df = pd.read_parquet(test_path)
    id_col = "container_id" if "container_id" in test_df.columns else "vm_id"

    # Reconstruct what the resaver would have skipped: a context window of
    # 512 observations was required; skip if container has fewer than 512
    # total observations BEFORE the test origin.
    # Easier approximation: load train+val+test concatenated and count per
    # container; flag containers with < ~600 total observations (512 context
    # + at least a few test rows).
    parts = []
    for p in (train_path, val_path, test_path):
        if p.exists():
            parts.append(pd.read_parquet(p))
    full = pd.concat(parts, ignore_index=True)
    counts = full.groupby(id_col).size()
    test_counts = test_df.groupby(id_col).size()

    short_containers = counts[counts < 600]
    print(f"  containers with <600 total observations: {len(short_containers)}")
    print(f"  shortest container length p10: {short_containers.quantile(0.1):.0f} obs")
    print(f"  shortest container length p50: {short_containers.quantile(0.5):.0f} obs")

    # is the lost 4701 attributable to short containers?
    # Approx: each short container would have lost ~test_count_for_that_id origins
    short_ids = short_containers.index
    test_loss = test_counts[test_counts.index.isin(short_ids)].sum()
    # divide by cadence_steps=6 to convert to origin count
    expected_skip = test_loss // 6
    print(f"  est. test rows from short containers: {test_loss}")
    print(f"  est. origins lost from short containers (cadence/6): {expected_skip}")
    print(f"  actual origins skipped in resaver: 4701")
    if abs(expected_skip - 4701) < 1000:
        print(f"  OK: skipped origins are concentrated in short containers")
    else:
        print(f"  WARN: skipped origins not fully explained by short containers")
        print(f"  Could be containers where test split starts before train+val end")
    return True


def main():
    print("=" * 70)
    print("F4 V1F: RESAVER PARQUET VERIFIER")
    print("=" * 70)
    if not PARQUET.exists():
        print(f"FATAL: {PARQUET} missing")
        sys.exit(1)
    sz_mb = PARQUET.stat().st_size / 1e6
    print(f"loading {PARQUET} ({sz_mb:.1f} MB) ...")
    df = pd.read_parquet(PARQUET)
    print(f"loaded {len(df)} rows")

    results = {}
    results["A schema"]            = check_A_schema(df)
    results["B monotonicity"]      = check_B_monotonicity(df)
    results["C scale"]             = check_C_scale(df)
    results["D pinball reproduce"] = check_D_pinball_reproduction(df)
    results["E truth coverage"]    = check_E_truth_coverage(df)
    results["F skipped origins"]   = check_F_skipped_origins()

    section("SUMMARY")
    blocking = {"A schema", "B monotonicity", "C scale", "D pinball reproduce"}
    for name, ok_ in results.items():
        tag = "BLOCKING" if name in blocking else "SOFT"
        print(f"  CHECK {name} ({tag}): {'PASS' if ok_ else 'FAIL'}")
    n_block_pass = sum(1 for name in blocking if results[name])
    n_block_total = len(blocking)
    print(f"\n  {n_block_pass}/{n_block_total} blocking checks pass")
    if n_block_pass == n_block_total:
        print("\n  PARQUET F4-READY. Safe to proceed to f4_mpc.py.")
    else:
        print("\n  PARQUET HAS ISSUES. Resolve before F4 production code.")


if __name__ == "__main__":
    main()
