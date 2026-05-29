"""
f4_pareto_analysis.py — paired stationary-bootstrap Pareto analysis.

Reads phase_f/data/f4_trajectories.parquet (per-container, per-pipeline, per-seed
summaries written by f4_evaluation_harness.py) and computes:

    1. Per-dataset (Δ-violation, Δ-cost) point estimates: v7 mean - v6 mean
    2. Per-dataset 95% paired stationary-bootstrap CIs (Politis-Romano 1994)
    3. Per-dataset p-values via bootstrap: 2 * min(P(Δ>=0), P(Δ<=0))
    4. Holm-Bonferroni correction across the 6 tests (3 datasets * 2 metrics)
    5. Per-dataset Pareto verdict (v7 dominates / indistinguishable / v6 dominates)
    6. Final routing recommendation per the pre-registered decision rule
    7. Three per-dataset scatter plots + one combined PDF

Outputs:
    phase_f/data/f4_pareto_verdicts.csv
    phase_f/data/f4_pareto_alibaba.pdf
    phase_f/data/f4_pareto_bitbrains.pdf
    phase_f/data/f4_pareto_bytedance.pdf
    phase_f/data/f4_pareto_combined.pdf

Usage:
    python phase_f/scripts/f4_pareto_analysis.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))
TRAJ_PATH   = PROJECT_ROOT / "phase_f" / "data" / "f4_trajectories.parquet"
OUT_DIR     = PROJECT_ROOT / "phase_f" / "data"

R_BOOTSTRAP = 1000
ALPHA_LEVEL = 0.05
N_TESTS     = 6      # 3 datasets * 2 metrics


# ── stationary bootstrap (Politis-Romano 1994) ──────────────────────

def stationary_bootstrap_indices(n, expected_block_len, rng):
    """
    Generate one stationary-bootstrap index sample of length n.

    The probability of starting a new block at each step is p = 1/expected_block_len.
    Within a block, indices increment circularly.
    """
    indices = np.empty(n, dtype=np.int64)
    p = 1.0 / expected_block_len
    # First index: uniform random
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i-1] + 1) % n
    return indices


def paired_bootstrap_diff(v6_values, v7_values, R=R_BOOTSTRAP, expected_block_len=None,
                          seed=42):
    """
    Compute paired stationary-bootstrap CI for mean(v7 - v6).

    Args:
        v6_values: 1D np.ndarray, length n
        v7_values: 1D np.ndarray, length n (same n; paired)
        R: number of bootstrap resamples
        expected_block_len: block length b; default ~ n^(1/3)
        seed: rng seed

    Returns:
        dict with point, ci_lo, ci_hi, p_value
    """
    n = len(v6_values)
    if n != len(v7_values):
        raise ValueError(f"length mismatch {n} vs {len(v7_values)}")
    if n == 0:
        return {"point": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_value": np.nan, "n": 0}

    if expected_block_len is None:
        expected_block_len = max(2, int(round(n ** (1/3))))

    point = float(np.mean(v7_values - v6_values))
    rng = np.random.default_rng(seed)

    boot_means = np.empty(R)
    for r in range(R):
        idx = stationary_bootstrap_indices(n, expected_block_len, rng)
        d = v7_values[idx] - v6_values[idx]
        boot_means[r] = float(np.mean(d))

    ci_lo = float(np.percentile(boot_means, 100 * ALPHA_LEVEL / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - ALPHA_LEVEL / 2)))

    # Two-sided p-value
    p_right = float((boot_means >= 0).sum() / R)
    p_left  = float((boot_means <= 0).sum() / R)
    p = 2.0 * min(p_right, p_left)

    return {
        "point": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p,
        "n": n, "block_len": expected_block_len,
    }


def holm_bonferroni(p_values, alpha=ALPHA_LEVEL):
    """
    Apply Holm-Bonferroni correction. Returns array of booleans (significant).
    """
    p_arr = np.asarray(p_values, dtype=float)
    order = np.argsort(p_arr)
    sig = np.zeros(len(p_arr), dtype=bool)
    for rank, idx in enumerate(order):
        threshold = alpha / (len(p_arr) - rank)
        if p_arr[idx] <= threshold:
            sig[idx] = True
        else:
            # Stop at first non-rejection
            break
    return sig


def per_dataset_verdict(v_lo, v_hi, c_lo, c_hi, v_sig, c_sig):
    """
    Given CI on (Δ-violation) and (Δ-cost) and significance flags, return verdict.

    Strict Pareto dominance for v7: both Δ <= 0 (CI lies in non-positive orthant)
    AND at least one is strictly negative at significance level after correction.

    Strict Pareto dominance for v6: symmetric reverse.

    Otherwise: indistinguishable.
    """
    v7_dominant = (v_hi <= 0) and (c_hi <= 0) and (v_sig or c_sig)
    v6_dominant = (v_lo >= 0) and (c_lo >= 0) and (v_sig or c_sig)

    if v7_dominant:
        return "v7_dominates"
    elif v6_dominant:
        return "v6_dominates"
    else:
        return "indistinguishable"


# ── plotting ────────────────────────────────────────────────────────

def plot_dataset_pareto(df, dataset, ax=None, title=None):
    """Per-dataset Pareto scatter. v6 markers blue, v7 markers orange."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    sub = df[df.dataset == dataset]

    for pipeline, color, marker in [("v6", "#1f77b4", "o"), ("v7", "#ff7f0e", "s")]:
        sub_p = sub[sub.pipeline == pipeline]
        ax.scatter(
            sub_p.replica_mean, sub_p.violation_rate,
            c=color, marker=marker, alpha=0.3, s=20,
            label=f"{pipeline} (n={len(sub_p)})",
        )

    ax.set_xlabel("Average replicas (cost proxy)")
    ax.set_ylabel("Violation rate")
    ax.set_title(title or dataset)
    ax.legend()
    ax.grid(True, alpha=0.3)


def make_combined_pareto(df, output_path):
    """3-panel side-by-side Pareto plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, ds in enumerate(["alibaba", "bitbrains", "bytedance"]):
        plot_dataset_pareto(df, ds, axes[i], title=ds)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=str, default=str(TRAJ_PATH))
    ap.add_argument("--R", type=int, default=R_BOOTSTRAP)
    args = ap.parse_args()

    if not Path(args.trajectories).exists():
        print(f"FAIL: trajectory file {args.trajectories} not found")
        return 1

    print(f"Loading: {args.trajectories}")
    df = pd.read_parquet(args.trajectories)
    print(f"  shape: {df.shape}")
    print(f"  datasets: {df.dataset.unique().tolist()}")
    print(f"  pipelines: {df.pipeline.unique().tolist()}")
    print(f"  seeds: {df.seed.unique().tolist()}")

    # The container-level evaluation gives per-container summaries; aggregate to
    # (dataset, pipeline, seed) means before computing the paired difference,
    # so that we are comparing the same containers under v6 vs v7.
    # Strategy: pivot by container_id, then take the inner-join across pipelines.

    print(f"\nComputing per-(dataset, container, seed) pivots ...")
    pivots = {}
    for ds in df.dataset.unique():
        pivot_v = df[df.dataset == ds].pivot_table(
            index=["container_id", "seed"],
            columns="pipeline",
            values="violation_rate",
            aggfunc="first",
        ).dropna()
        pivot_c = df[df.dataset == ds].pivot_table(
            index=["container_id", "seed"],
            columns="pipeline",
            values="replica_mean",
            aggfunc="first",
        ).dropna()
        # Inner-join on the index so that we only keep containers present in BOTH pipelines
        common_idx = pivot_v.index.intersection(pivot_c.index)
        pivot_v = pivot_v.loc[common_idx]
        pivot_c = pivot_c.loc[common_idx]
        pivots[ds] = {
            "violation": pivot_v,
            "cost":      pivot_c,
        }
        print(f"  {ds}: {len(pivot_v)} (container, seed) pairs in BOTH pipelines")

    # Compute bootstrap CIs
    print(f"\nRunning paired stationary bootstrap (R={args.R}) ...")
    results = []
    for ds in pivots:
        for metric, key in [("violation_rate", "violation"), ("replica_mean", "cost")]:
            pivot = pivots[ds][key]
            if "v6" not in pivot.columns or "v7" not in pivot.columns:
                print(f"  {ds}/{metric}: missing one pipeline; skipping")
                continue
            v6 = pivot["v6"].to_numpy()
            v7 = pivot["v7"].to_numpy()
            res = paired_bootstrap_diff(v6, v7, R=args.R)
            print(f"  {ds:10s}/{metric:14s}: n={res['n']}, b={res['block_len']}, "
                  f"point={res['point']:+.5f}, "
                  f"CI=[{res['ci_lo']:+.5f}, {res['ci_hi']:+.5f}], "
                  f"p={res['p_value']:.4f}")
            results.append({
                "dataset": ds, "metric": metric,
                "point": res["point"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                "p_value": res["p_value"], "n": res["n"], "block_len": res["block_len"],
            })

    res_df = pd.DataFrame(results)

    # Holm-Bonferroni
    if len(res_df) > 0:
        sig = holm_bonferroni(res_df["p_value"].values, alpha=ALPHA_LEVEL)
        res_df["sig_holm"] = sig
        print(f"\nHolm-Bonferroni at family alpha={ALPHA_LEVEL}:")
        for _, r in res_df.iterrows():
            print(f"  {r.dataset:10s}/{r.metric:14s}: "
                  f"p={r.p_value:.4f}, sig={r.sig_holm}")

    # Per-dataset Pareto verdict
    print(f"\nPER-DATASET VERDICTS")
    print("=" * 70)
    verdicts = {}
    for ds in pivots:
        v_row = res_df[(res_df.dataset == ds) & (res_df.metric == "violation_rate")]
        c_row = res_df[(res_df.dataset == ds) & (res_df.metric == "replica_mean")]
        if v_row.empty or c_row.empty:
            verdicts[ds] = "incomplete"
            continue
        v = v_row.iloc[0]
        c = c_row.iloc[0]
        verdict = per_dataset_verdict(
            v.ci_lo, v.ci_hi, c.ci_lo, c.ci_hi, v.sig_holm, c.sig_holm
        )
        verdicts[ds] = verdict
        print(f"  {ds:10s}: {verdict}")
        print(f"     viol Δ = {v.point:+.5f} CI [{v.ci_lo:+.5f}, {v.ci_hi:+.5f}] sig={v.sig_holm}")
        print(f"     cost Δ = {c.point:+.5f} CI [{c.ci_lo:+.5f}, {c.ci_hi:+.5f}] sig={c.sig_holm}")

    # Apply pre-registered decision rule
    print(f"\nFINAL ROUTING DECISION")
    print("=" * 70)
    if (verdicts.get("alibaba") == "v7_dominates"
        and verdicts.get("bitbrains") in ("v7_dominates", "indistinguishable")
        and verdicts.get("bytedance") in ("v7_dominates", "indistinguishable")):
        decision = "ADOPT v7"
        rationale = "v7 dominates Alibaba (the hypothesis target); ties or wins elsewhere."
    elif any(v == "v6_dominates" for v in verdicts.values()):
        decision = "ADOPT v6"
        rationale = "v7 lost on at least one dataset; v6 remains canonical."
    else:
        decision = "ADOPT v6 (default)"
        rationale = "v7 did not dominate Alibaba; the routing hypothesis is not vindicated."
    print(f"  Decision:  {decision}")
    print(f"  Rationale: {rationale}")

    # Save verdicts
    verdict_df = pd.DataFrame([
        {"dataset": ds, "verdict": v} for ds, v in verdicts.items()
    ])
    verdict_df.to_csv(OUT_DIR / "f4_pareto_verdicts.csv", index=False)
    res_df.to_csv(OUT_DIR / "f4_bootstrap_results.csv", index=False)

    with open(OUT_DIR / "f4_pareto_decision.json", "w") as f:
        json.dump({
            "verdicts": verdicts,
            "decision": decision,
            "rationale": rationale,
            "bootstrap_R": args.R,
            "alpha_level": ALPHA_LEVEL,
            "n_tests": N_TESTS,
            "correction": "Holm-Bonferroni",
            "decision_rule_source": "phase_f/f4_evaluation_plan.md (git-committed pre-registration)",
        }, f, indent=2)
    print(f"\nWROTE: {OUT_DIR / 'f4_pareto_verdicts.csv'}")
    print(f"       {OUT_DIR / 'f4_bootstrap_results.csv'}")
    print(f"       {OUT_DIR / 'f4_pareto_decision.json'}")

    # Plots
    print(f"\nGenerating plots ...")
    for ds in pivots:
        out_path = OUT_DIR / f"f4_pareto_{ds}.pdf"
        plot_dataset_pareto(df, ds, ax=None, title=ds)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  wrote: {out_path}")

    combined_path = OUT_DIR / "f4_pareto_combined.pdf"
    make_combined_pareto(df, combined_path)
    print(f"  wrote: {combined_path}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
