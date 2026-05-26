"""
f3_compare_variants.py — Aggregate F3 ablation results into single comparison table

Reads f3_eval_*.json for each variant and produces:
  - phase_f/data/f3_variant_comparison.csv  (one row per variant)
  - phase_f/data/f3_variant_comparison.md   (markdown table for chapter writing)

Usage:
    python phase_f/scripts/f3_compare_variants.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path("/workspace/kubernetes-cpu-ensemble-thesis")
PHASE_F   = WORKSPACE / "phase_f"
DATA_DIR  = PHASE_F / "data"

OUT_CSV = DATA_DIR / "f3_variant_comparison.csv"
OUT_MD  = DATA_DIR / "f3_variant_comparison.md"

VARIANTS = [
    "lora_rank4", "lora_rank8", "lora_rank16",
    "dora_rank8", "curriculum_rank8",
]

# -------------------------------------------------------------------------
# Helper to format values that might be None or NaN
# -------------------------------------------------------------------------

def fmt_signed(x):
    if x is None:
        return "N/A"
    try:
        if pd.isna(x):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{x:+.2f}"

def fmt_unsigned(x):
    if x is None:
        return "N/A"
    try:
        if pd.isna(x):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{x:.2f}"

# -------------------------------------------------------------------------

print("=" * 70)
print("f3_compare_variants - aggregate all F3 ablation results")
print("=" * 70)

orig_path = DATA_DIR / "f3_finetune_results.json"
orig_csv  = DATA_DIR / "f3_finetune_results.csv"

per_ds_baseline_h60_t90 = {}
if orig_csv.exists():
    df_orig = pd.read_csv(orig_csv)
    for ds in ["alibaba", "bitbrains", "bytedance"]:
        r = df_orig[(df_orig["dataset"] == ds) &
                    (df_orig["horizon_min"] == 60) &
                    (df_orig["tau"] == 0.9)]
        if len(r) > 0 and "baseline_pinball" in r.columns:
            bl = r["baseline_pinball"].iloc[0]
            if not pd.isna(bl):
                per_ds_baseline_h60_t90[ds] = float(bl)

print(f"  Per-dataset baselines (h=60, tau=0.9): {per_ds_baseline_h60_t90}")

rows = []

if orig_path.exists():
    with open(orig_path) as f:
        d = json.load(f)
    rows.append({
        "variant": "lora_rank8 (original v2)",
        "primary_improvement_pct": d.get("primary_metric", {}).get("improvement_pct"),
        "secondary_geom_mean_pct": d.get("secondary_metric", {}).get("geom_mean_pct"),
        "alibaba_h60_t90_pct": d.get("secondary_metric", {}).get("per_dataset_pct", {}).get("alibaba"),
        "bitbrains_h60_t90_pct": d.get("secondary_metric", {}).get("per_dataset_pct", {}).get("bitbrains"),
        "bytedance_h60_t90_pct": d.get("secondary_metric", {}).get("per_dataset_pct", {}).get("bytedance"),
        "main_group_improvement_pct": None,
        "holdout_improvement_pct": None,
        "replication_delta_pp": None,
        "verdict": "SUCCESS" if (d.get("primary_metric", {}).get("improvement_pct") or 0) >= 5 else "PARTIAL",
    })
    print(f"  Loaded original: {orig_path.name}")

for variant_tag in VARIANTS:
    path = DATA_DIR / f"f3_eval_{variant_tag}.json"
    if not path.exists():
        print(f"  Skipping (not found): {path.name}")
        continue

    with open(path) as f:
        d = json.load(f)

    pm = d.get("primary_metric", {})
    main_imp    = pm.get("main_group_improvement_pct")
    holdout_imp = pm.get("holdout_group_improvement_pct")
    repl_delta  = pm.get("replication_delta_pct")

    per_ds_main = d.get("per_dataset_main", {})
    per_ds_pct = {}
    for ds, ft_val in per_ds_main.items():
        bl_val = per_ds_baseline_h60_t90.get(ds)
        if bl_val is not None and bl_val > 0:
            per_ds_pct[ds] = (bl_val - ft_val) / bl_val * 100

    geom_mean = None
    if per_ds_pct:
        vals = [v for v in per_ds_pct.values() if v is not None]
        if vals and all(v > 0 for v in vals):
            geom_mean = float(np.exp(np.mean(np.log(vals))))

    if main_imp is None:
        verdict = "UNKNOWN"
    elif main_imp >= 5:
        verdict = "SUCCESS"
    elif main_imp >= 2:
        verdict = "PARTIAL"
    else:
        verdict = "FAILURE"

    rows.append({
        "variant": variant_tag,
        "primary_improvement_pct": main_imp,
        "secondary_geom_mean_pct": geom_mean,
        "alibaba_h60_t90_pct": per_ds_pct.get("alibaba"),
        "bitbrains_h60_t90_pct": per_ds_pct.get("bitbrains"),
        "bytedance_h60_t90_pct": per_ds_pct.get("bytedance"),
        "main_group_improvement_pct": main_imp,
        "holdout_improvement_pct": holdout_imp,
        "replication_delta_pp": repl_delta,
        "verdict": verdict,
    })
    print(f"  Loaded: {path.name}")

if not rows:
    print("No variants found.")
    raise SystemExit(1)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

md_lines = [
    "# F3 LoRA / DoRA / Curriculum variant comparison",
    "",
    "## Primary metric (mean pinball h=60min tau=0.9 vs baseline 1.921658)",
    "",
    "| Variant | Main Delta% | Geom mean Delta% | Alibaba | Bitbrains | ByteDance | Hold-out Delta% | Repl. Delta_pp | Verdict |",
    "|---|---|---|---|---|---|---|---|---|",
]

for r in rows:
    md_lines.append(
        f"| {r['variant']} | "
        f"{fmt_signed(r['primary_improvement_pct'])} | "
        f"{fmt_signed(r['secondary_geom_mean_pct'])} | "
        f"{fmt_signed(r['alibaba_h60_t90_pct'])} | "
        f"{fmt_signed(r['bitbrains_h60_t90_pct'])} | "
        f"{fmt_signed(r['bytedance_h60_t90_pct'])} | "
        f"{fmt_signed(r['holdout_improvement_pct'])} | "
        f"{fmt_unsigned(r['replication_delta_pp'])} | "
        f"{r['verdict']} |"
    )

md_lines.extend([
    "",
    "## Notes on metrics",
    "",
    "- Main Delta%: improvement on first 70% of test series (alphabetically), where",
    "  +5% = SUCCESS, +2-5% = PARTIAL, <+2% = FAILURE per DECISION-005.",
    "- Hold-out Delta%: improvement on last 30% of test series, never seen by val",
    "  checkpoint selection. Large gap to main indicates checkpoint-selection bias.",
    "- Replication Delta_pp: |main - holdout|. Smaller = better generalisation.",
    "- Geom mean Delta%: geometric mean of per-dataset percentage improvements at",
    "  the primary cell (h=60, tau=0.9). N/A when any dataset has negative Delta.",
    "",
    "## Methodological caveats",
    "",
    "1. Training objective for all variants: native Chronos-2 ForCausalLMLoss",
    "   (Chronos-2 API does not expose differentiable quantile head).",
    "2. Asymmetric pinball applied only at evaluation (post-hoc), not at training.",
    "3. Curriculum variant mixes horizons within each batch (vs rotating per epoch).",
    "4. DoRA variant uses weight-decomposed LoRA (Liu et al., ICML 2024, arXiv:2402.09353).",
    "5. LoRA rank sweep verifies rank=8 from DECISION-016.",
    "6. Original v2 evaluation pooled all series; main-vs-holdout split exposes",
    "   cross-series heterogeneity that the pooled +8.17% headline obscures.",
])

with open(OUT_MD, "w") as f:
    f.write("\n".join(md_lines))
print(f"Saved: {OUT_MD}")

print()
print("Comparison table:")
print(df.to_string(index=False))