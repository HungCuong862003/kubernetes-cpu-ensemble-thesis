# D13 Journal — LOO-dataset CV + Dual-baseline + ERRATA-012 Instructions
**Date:** 2026-06-04 (D13)

---

## Task A — LOO-dataset CV (post-hoc robustness, 3 folds)

| Fold (held-out) | Correct / 4 | Notes |
|---|---|---|
| Alibaba | 0/4 | All predicted Chronos-2 |
| Bitbrains | 1/4 | Only h120 (TimesFM) correct |
| ByteDance | 0/4 | All predicted minority-class noise |

**LOO-dataset macro-F1: 0.1000**
LOO-cell macro-F1: 0.2532
Gap: +0.1532

LOO-dataset (0.10) drops below the trivial baseline (0.167). When trained on
two datasets and tested on the third, the router performs near-chance. The
predictability features it uses are dataset identifiers at cell level — the
mapping does not transfer across datasets.

This asymmetry (LOO-cell 0.25 vs LOO-dataset 0.10) is the most defensible
finding of the F1 chapter. It exposes the structural limit of cell-level
routing with only three production datasets.

## Task B — Dual-baseline comparison

| Baseline | Macro-F1 | Uplift from trivial |
|---|---|---|
| Trivial (always Chronos-2) | 0.167 | — |
| Structural (dataset + horizon only) | 0.225 | +0.058 |
| Full F1 router (4 features) | 0.253 | +0.086 |

**Structural → Full delta: +0.028**

ACF@24h, CV, and ACF@1h are perfectly rank-correlated with dataset identity
(Spearman ρ = ±1.0 from D11). Adding them as explicit features over a
(dataset_dummy + horizon) structural baseline adds only +0.028 macro-F1 —
numerical noise in tree split selection, not genuine predictability signal.

**Finding: predictability features do not add meaningful signal beyond
dataset and horizon knowledge at cell-level granularity with n=12.**

CSVs saved: `f1_loo_dataset_results.csv`, `f1_dual_baseline_comparison.csv`

## Task C — ERRATA-012 Overleaf application instructions

Target: Ch4 BCF Table 4.10 Bitbrains row + surrounding §4.X prose.

### Grep to locate loci

```bash
grep -n "5\.46\|3\.30\|ML wins only" chapters/04*.tex
```

### Table 4.10 Bitbrains row replacement

Original values: delta@30min = −5.46pp, delta@120min = +3.30pp,
verdict "ML wins only @120min"

Replace with:
- delta@30min = **+1.53pp**
- delta@120min = **−2.81pp**
- verdict: "Mixed: +1.53pp @h30 (61% VMs), −2.81pp @h120 (35% VMs)"

Source: `reports/tables/boundary_condition_table_corrected.csv` (NEW pool per-VM medians)

### §4.X prose (one sentence → three sentences)

Replace the single sentence claiming "ML wins only at h=120 for Bitbrains" with:

> Under the NEW pool aggregation (per-VM median across 142 VMs), Bitbrains shows
> a modest positive delta at h30 (+1.53 pp, 61\% of VMs beating naive) but
> negative deltas at h10 (−16.66 pp), h60 (−1.99 pp), and h120 (−2.81 pp,
> 35\% of VMs). The sign inversion from the OLD pool (which reported +3.30 pp
> at h120 and −5.46 pp at h30) reflects the aggregation scope shift: OLD pool
> used pooled full-test R² dominated by high-variance VMs, while NEW pool uses
> per-VM median. The BCF predicate (ACF@24h $> 0.2$ AND $h \geq 30$ min)
> correctly flags Bitbrains (ACF@24h = 0.116) as a dataset where ML benefit
> is expected to be small and unstable.

### Unchanged loci — do not touch

- §3.8: AUC 0.80, p=0.0097, CI [0.71, 0.89]
- §5.1: BCa footnote
- §6.1 + Table 4.13: HPA v4 dual-metric paragraph

### Overleaf commit message

```
Apply ERRATA-012 — Ch4 Table 4.10 Bitbrains OLD->NEW pool per-VM medians;
§4.X prose 1->3 sentences. Source: boundary_condition_table_corrected.csv.
Phase F D13 (2026-06-04).
```

After rebuild: confirm delta@30min changed −5.46→+1.53 and delta@120min
changed +3.30→−2.81 in PDF. No other loci changed.

## EOD status

- LOO-dataset: 0.1000 (below trivial baseline — structural limit confirmed)
- Dual-baseline: full vs structural = +0.028 (no meaningful signal from predictability features)
- ERRATA-012: substitution text fully specified above; Jimmy applies in Overleaf
