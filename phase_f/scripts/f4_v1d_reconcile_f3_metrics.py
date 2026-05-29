"""
f4_v1d_reconcile_f3_metrics.py — Reconcile the F3-FT pre-prompt-claimed numbers
                                  against what's actually in f3_eval_lora_rank8.json.

F4 prompt claimed:
  - +11.10% pooled improvement
  - Alibaba +5.19%, Bitbrains +12.49%, ByteDance +5.07%

Inspector showed JSON contains:
  - main_group_improvement_pct: +36.24%
  - holdout_group_improvement_pct: -46.76%
  - replication_delta_pct: +82.99%
  - per_dataset_main: alibaba 0.408, bitbrains 2.838, bytedance 0.430 (RAW pinball values)

These do not look like the same numbers. Before any F4 production code, we
need to know which set is correct so the F4 chapter cites the right F3 result.

This script prints:
  1. The full primary_metric block from JSON
  2. Per-dataset main vs holdout pinball values
  3. Per-dataset percent improvement vs the F3 day-1 baseline (1.922)
     and vs the per-cell zero-shot baseline if available
  4. The discrepancy with the F4-prompt numbers, line by line

Run on Vast:
    python phase_f/scripts/f4_v1d_reconcile_f3_metrics.py
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/workspace/kubernetes-cpu-ensemble-thesis")
FT_JSON = PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.json"
ZS_JSON = PROJECT_ROOT / "phase_f" / "data" / "f3_zero_shot_baseline.json"
FT_CSV  = PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.csv"

# F4 prompt's claims (to compare against)
F4_PROMPT_CLAIMS = {
    "pooled_improvement_pct":     11.10,
    "alibaba_improvement_pct":    5.19,
    "bitbrains_improvement_pct":  12.49,
    "bytedance_improvement_pct":  5.07,
}


def main():
    print("=" * 70)
    print("F4 V1D: RECONCILE F3-FT METRICS")
    print("=" * 70)

    if not FT_JSON.exists():
        print(f"FATAL: {FT_JSON} missing")
        return
    with open(FT_JSON) as f:
        ft = json.load(f)

    print(f"\nFT JSON tag: {ft.get('tag')!r}")
    print(f"adapter_dir: {ft.get('adapter_dir')!r}")

    print("\n--- primary_metric block (verbatim) ---")
    pm = ft.get("primary_metric", {})
    for k, v in pm.items():
        print(f"  {k}: {v}")

    print("\n--- per_dataset_main (raw pinball, lower=better) ---")
    pdm = ft.get("per_dataset_main", {})
    for k, v in pdm.items():
        print(f"  {k}: {v:.6f}")

    print("\n--- per_dataset_holdout ---")
    pdh = ft.get("per_dataset_holdout", {})
    for k, v in pdh.items():
        print(f"  {k}: {v:.6f}")

    # Zero-shot baseline (per-dataset h=60 tau=0.9 from F3 day-1)
    # The F3 Day-1 doc gave:
    #   Alibaba h=60 tau=0.9:   0.447
    #   Bitbrains h=60 tau=0.9: 4.859
    #   ByteDance h=60 tau=0.9: 0.459
    # But primary_metric uses MEAN at h=60 tau=0.9 across 3 datasets:
    #   (0.447 + 4.859 + 0.459) / 3 = 1.922 -- matches baseline field above.

    print("\n--- F3 day-1 baseline reference (zero-shot h=60 tau=0.9) ---")
    zs_ref = {"alibaba": 0.447, "bitbrains": 4.859, "bytedance": 0.459}
    for k, v in zs_ref.items():
        print(f"  {k}: {v}")

    print("\n--- per-dataset improvement vs zero-shot (h=60 tau=0.9 only) ---")
    print("  HOWEVER per_dataset_main may NOT be h=60 tau=0.9 specifically;")
    print("  could be a different aggregation. Verify by reading the F3 FT script.")
    for k in ("alibaba", "bitbrains", "bytedance"):
        if k in pdm and k in zs_ref:
            ft_val = pdm[k]
            zs_val = zs_ref[k]
            improvement_pct = (zs_val - ft_val) / zs_val * 100.0
            print(f"  {k}: zs={zs_val:.3f}, ft={ft_val:.3f}, "
                  f"improvement={improvement_pct:+.2f}%")

    # Compare against F4 prompt
    print("\n--- F4 PROMPT CLAIMS vs JSON RECONCILIATION ---")
    print(f"  F4 prompt: pooled improvement = +{F4_PROMPT_CLAIMS['pooled_improvement_pct']}%")
    print(f"  JSON:      main_group_improvement_pct = "
          f"{pm.get('main_group_improvement_pct', 'MISSING')}%")
    print()
    print(f"  F4 prompt: Alibaba   = +{F4_PROMPT_CLAIMS['alibaba_improvement_pct']}%")
    print(f"  F4 prompt: Bitbrains = +{F4_PROMPT_CLAIMS['bitbrains_improvement_pct']}%")
    print(f"  F4 prompt: ByteDance = +{F4_PROMPT_CLAIMS['bytedance_improvement_pct']}%")
    print(f"  (JSON does not appear to expose per-dataset improvement %; would need")
    print(f"   to compute from raw pinball values plus a per-dataset baseline.)")

    # ── now inspect the CSV: this is where the per-cell detail lives ────
    print("\n=== CSV per-cell detail ===")
    if not FT_CSV.exists():
        print(f"  {FT_CSV} missing; can't proceed")
        return
    df = pd.read_csv(FT_CSV)
    print(f"  shape: {df.shape}")
    print(f"  unique tags:    {df['tag'].unique()}")
    print(f"  unique datasets: {df['dataset'].unique()}")
    print(f"  unique horizons: {sorted(df['horizon_min'].unique())}")
    print(f"  unique groups:   {df['group'].unique()}")
    print(f"  unique taus:     {sorted(df['tau'].unique())}")

    # Pivot the canonical primary metric (h=60, tau=0.9) per dataset, main only
    print("\n  h=60 tau=0.9 main group only (canonical F3 primary):")
    sub = df[(df["horizon_min"] == 60) & (df["tau"] == 0.9) & (df["group"] == "main")]
    print(sub[["dataset", "n_valid", "sym_pinball"]].to_string(index=False))

    print("\n  h=60 tau=0.9 HOLDOUT group:")
    sub_h = df[(df["horizon_min"] == 60) & (df["tau"] == 0.9) & (df["group"] == "holdout")]
    print(sub_h[["dataset", "n_valid", "sym_pinball"]].to_string(index=False))

    # ── also load zero-shot baseline JSON if it exists ──────────────────
    if ZS_JSON.exists():
        print("\n=== ZERO-SHOT BASELINE JSON (for direct comparison) ===")
        with open(ZS_JSON) as f:
            zs = json.load(f)
        print(f"  top-level keys: {list(zs.keys())}")
        # F3 day-1 doc said zero-shot baseline JSON contains per-cell pinball
        # Try to extract h=60 tau=0.9 per dataset
        # Schema unknown; print first level
        for k, v in zs.items():
            if isinstance(v, dict):
                print(f"  {k}: dict with keys {list(v.keys())[:8]}")
            elif isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("Before F4 production code, Jimmy must confirm:")
    print()
    print("  1. WHICH numbers are correct -- F4 prompt or this JSON?")
    print("     If JSON: F4 prompt was wrong; F4 chapter must use JSON values.")
    print("     If F4 prompt: f3_eval_lora_rank8.json is stale; need to identify")
    print("                   the correct F3-FT eval output file (different tag?)")
    print()
    print("  2. WHAT does 'main_group' vs 'holdout_group' mean in F3 eval?")
    print("     If holdout = generalisation test, then holdout_group_improvement_pct")
    print("     = -46.76% is a SERIOUS finding for the F3 chapter, and F4 chapter")
    print("     must report both groups separately (not pool them).")
    print()
    print("  3. WHICH tau (asymmetric cost ratio) should F4 use?")
    print("     JSON tested cr in {1, 3, 5, 10}. Higher cr = more asymmetric pinball.")
    print("     F4 MPC's alpha = 0.9 maps roughly to cr=9 (1/(1-alpha)). Pick the")
    print("     cost ratio that matches the F4 alpha you'll use.")


if __name__ == "__main__":
    main()
