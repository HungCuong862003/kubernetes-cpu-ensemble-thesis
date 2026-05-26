"""
f3_investigate_per_dataset.py — Per-dataset hold-out investigation (corrected for actual baseline JSON structure)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path("/workspace/kubernetes-cpu-ensemble-thesis")
PHASE_F   = WORKSPACE / "phase_f"
DATA_DIR  = PHASE_F / "data"

OUT_CSV = DATA_DIR / "f3_per_dataset_diagnosis.csv"
OUT_MD  = DATA_DIR / "f3_per_dataset_diagnosis.md"

VARIANTS = [
    "lora_rank4", "lora_rank8", "lora_rank16",
    "dora_rank8", "curriculum_rank8",
]

PRIMARY_H_MIN = 60
PRIMARY_TAU   = 0.9
PRIMARY_TAU_KEY = "0.90"  # The string key used in baseline JSON

print("=" * 70)
print("f3_investigate_per_dataset — per-dataset hold-out diagnosis")
print("=" * 70)
print()

# -------------------------------------------------------------------------
# Step 1: Load canonical per-dataset baselines (correct path)
# -------------------------------------------------------------------------

baseline_path = DATA_DIR / "f3_zero_shot_baseline.json"
print(f"Step 1: Loading baselines from {baseline_path.name}")

with open(baseline_path) as f:
    baseline_data = json.load(f)

per_ds_baseline = {}
for entry in baseline_data.get('results', []):
    if entry.get('horizon_min') == PRIMARY_H_MIN:
        ds = entry.get('dataset')
        pinball_dict = entry.get('pinball_loss', {})
        bl = pinball_dict.get(PRIMARY_TAU_KEY)
        if bl is not None:
            per_ds_baseline[ds] = float(bl)

print(f"  Per-dataset baselines (h={PRIMARY_H_MIN}, tau={PRIMARY_TAU}): {per_ds_baseline}")
print()

if not per_ds_baseline:
    print("ERROR: Could not extract baselines.")
    raise SystemExit(1)

# -------------------------------------------------------------------------
# Step 2: Per-variant per-dataset improvement
# -------------------------------------------------------------------------

print("Step 2: Computing per-dataset improvements")

rows = []
for variant in VARIANTS:
    path = DATA_DIR / f"f3_eval_{variant}.json"
    if not path.exists():
        print(f"  Skipping (not found): {variant}")
        continue

    with open(path) as f:
        d = json.load(f)

    per_ds_main = d.get("per_dataset_main", {})
    per_ds_holdout = d.get("per_dataset_holdout", {})

    for ds in ['alibaba', 'bitbrains', 'bytedance']:
        bl = per_ds_baseline.get(ds)
        if bl is None or bl <= 0:
            continue

        main_pin = per_ds_main.get(ds)
        ho_pin   = per_ds_holdout.get(ds)

        main_imp = ((bl - main_pin) / bl * 100) if main_pin is not None else None
        ho_imp   = ((bl - ho_pin)   / bl * 100) if ho_pin   is not None else None
        repl_delta = (abs(main_imp - ho_imp)) if (main_imp is not None and ho_imp is not None) else None

        rows.append({
            "variant": variant,
            "dataset": ds,
            "baseline_pinball": bl,
            "main_pinball": main_pin,
            "holdout_pinball": ho_pin,
            "main_imp_pct": main_imp,
            "holdout_imp_pct": ho_imp,
            "replication_delta_pp": repl_delta,
        })

df = pd.DataFrame(rows)
if df.empty:
    print("No data assembled.")
    raise SystemExit(1)

df.to_csv(OUT_CSV, index=False)
print(f"  Saved: {OUT_CSV}")
print()

# -------------------------------------------------------------------------
# Step 3: Pivots and summary
# -------------------------------------------------------------------------

pivot_main = df.pivot(index='variant', columns='dataset', values='main_imp_pct').round(2)
pivot_ho   = df.pivot(index='variant', columns='dataset', values='holdout_imp_pct').round(2)
pivot_del  = df.pivot(index='variant', columns='dataset', values='replication_delta_pp').round(2)

# Re-order to consistent variant ordering
variant_order = [v for v in VARIANTS if v in pivot_main.index]
pivot_main = pivot_main.reindex(variant_order)
pivot_ho   = pivot_ho.reindex(variant_order)
pivot_del  = pivot_del.reindex(variant_order)

# Re-order columns
ds_order = [c for c in ['alibaba', 'bitbrains', 'bytedance'] if c in pivot_main.columns]
pivot_main = pivot_main[ds_order]
pivot_ho   = pivot_ho[ds_order]
pivot_del  = pivot_del[ds_order]

print("=" * 70)
print("MAIN GROUP IMPROVEMENT % (per dataset)")
print("=" * 70)
print(pivot_main.to_string())
print()
print("=" * 70)
print("HOLD-OUT GROUP IMPROVEMENT % (per dataset)")
print("=" * 70)
print(pivot_ho.to_string())
print()
print("=" * 70)
print("REPLICATION DELTA (per dataset)  |main - holdout|")
print("=" * 70)
print(pivot_del.to_string())
print()

# -------------------------------------------------------------------------
# Step 4: Diagnostic interpretation
# -------------------------------------------------------------------------

print("=" * 70)
print("DIAGNOSTIC")
print("=" * 70)

# Compare replication delta variance across datasets per variant
print()
print("Replication delta range per variant (max - min across datasets):")
for variant in variant_order:
    deltas = pivot_del.loc[variant].dropna().values
    if len(deltas) >= 2:
        spread = float(deltas.max() - deltas.min())
        mean_d = float(deltas.mean())
        print(f"  {variant:25s} mean={mean_d:6.1f}pp  spread={spread:6.1f}pp")

print()
print("Replication delta per dataset (mean across variants):")
for ds in ds_order:
    vals = pivot_del[ds].dropna().values
    if len(vals):
        print(f"  {ds:12s} mean={vals.mean():6.1f}pp  std={vals.std():5.1f}pp")

# -------------------------------------------------------------------------
# Step 5: Markdown report
# -------------------------------------------------------------------------

def fmt_signed(x):
    if x is None or pd.isna(x): return "N/A"
    return f"{x:+.2f}"

def fmt_unsigned(x):
    if x is None or pd.isna(x): return "N/A"
    return f"{x:.2f}"

md = []
md.append("# F3 per-dataset hold-out diagnosis")
md.append("")
md.append("**Goal:** Determine whether the F3 hold-out failure is uniform (checkpoint-")
md.append("selection bias) or concentrated on specific datasets (transfer issue).")
md.append("")
md.append(f"**Metric:** Pinball loss at h={PRIMARY_H_MIN}min, tau={PRIMARY_TAU}, per dataset.")
md.append("")

md.append("## Per-dataset baselines (zero-shot)")
md.append("")
md.append("| Dataset | Baseline pinball |")
md.append("|---|---|")
for ds in ds_order:
    md.append(f"| {ds} | {fmt_unsigned(per_ds_baseline.get(ds))} |")
md.append("")

md.append("## Main group improvement %")
md.append("")
md.append("First 70% of series alphabetically — partially overlaps val cohort.")
md.append("")
md.append("| Variant | " + " | ".join(ds_order) + " |")
md.append("|---|" + "---|" * len(ds_order))
for variant in variant_order:
    cells = [fmt_signed(pivot_main.loc[variant, ds]) for ds in ds_order]
    md.append(f"| {variant} | " + " | ".join(cells) + " |")
md.append("")

md.append("## Hold-out group improvement %")
md.append("")
md.append("Last 30% of series alphabetically — never seen by val checkpoint selection.")
md.append("")
md.append("| Variant | " + " | ".join(ds_order) + " |")
md.append("|---|" + "---|" * len(ds_order))
for variant in variant_order:
    cells = [fmt_signed(pivot_ho.loc[variant, ds]) for ds in ds_order]
    md.append(f"| {variant} | " + " | ".join(cells) + " |")
md.append("")

md.append("## Replication delta |main - holdout|")
md.append("")
md.append("Larger = worse generalisation across the within-series alphabetical split.")
md.append("")
md.append("| Variant | " + " | ".join(ds_order) + " |")
md.append("|---|" + "---|" * len(ds_order))
for variant in variant_order:
    cells = [fmt_unsigned(pivot_del.loc[variant, ds]) for ds in ds_order]
    md.append(f"| {variant} | " + " | ".join(cells) + " |")
md.append("")

md.append("## Summary statistics")
md.append("")
md.append("**Per dataset (across variants):**")
md.append("")
md.append("| Dataset | Mean replication delta | Std |")
md.append("|---|---|---|")
for ds in ds_order:
    vals = pivot_del[ds].dropna().values
    if len(vals):
        md.append(f"| {ds} | {vals.mean():.2f}pp | {vals.std():.2f}pp |")
md.append("")

md.append("**Per variant (across datasets):**")
md.append("")
md.append("| Variant | Mean replication delta | Spread (max - min) |")
md.append("|---|---|---|")
for variant in variant_order:
    deltas = pivot_del.loc[variant].dropna().values
    if len(deltas) >= 2:
        md.append(f"| {variant} | {deltas.mean():.2f}pp | {(deltas.max() - deltas.min()):.2f}pp |")
md.append("")

md.append("## Interpretation rubric")
md.append("")
md.append("- **Uniform high delta (60-90pp on all 3 datasets):** Checkpoint-selection bias.")
md.append("  F3 fine-tuning overfits the val cohort (first ~500 series alphabetically).")
md.append("  PAR per-series approach is the right remedy.")
md.append("- **Bitbrains delta much higher than others:** Scale-artefact dominance.")
md.append("  Consistent with memory: Bitbrains contributes 84% of mean pinball.")
md.append("- **One dataset isolated catastrophe (e.g. only ByteDance fails):** Cross-dataset")
md.append("  transfer issue. Per-dataset adapters would help; not in current scope.")
md.append("")

with open(OUT_MD, "w") as f:
    f.write("\n".join(md))
print(f"Saved: {OUT_MD}")
print()
print("Done.")
