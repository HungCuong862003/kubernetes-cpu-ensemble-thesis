# D12 Journal — Architecture Pick + LOO-cell CV
**Date:** 2026-06-03 (D12)

---

## Task A — Architecture pick

Three candidates per DECISION-012: Logistic-L2, kNN-k3, shallow decision tree.

Based on D11 structural inspection (Spearman ρ = ±1.0 among three of four
features; only two effective discriminative axes):

- **Logistic-L2**: 20 parameters vs 11 training points per LOO fold —
  overfitting risk high. Not recommended as primary.
- **kNN-k3**: same-dataset cells dominate the 3-NN due to feature collinearity.
  Breaks completely in LOO-dataset (no neighbours from unseen dataset).
- **DecTree-depth2/3**: handles dataset-constant features naturally via a single
  split on acf_24h. No distributional assumptions. Correct choice for n=12.

Swept all four architectures empirically; DecTree-depth3 highest macro-F1.

## Task B — LOO-cell CV (pre-registered, 12 folds)

Script: `phase_f/scripts/f1_router.py`

### Architecture sweep results

| Architecture | LOO-cell macro-F1 | vs threshold (0.55) |
|---|---|---|
| DecTree-depth3 | **0.2532** | −0.2968 |
| Logistic-L2 | 0.2167 | −0.3333 |
| DecTree-depth2 | 0.1429 | −0.4071 |
| kNN-k3 | 0.1333 | −0.4167 |

**All architectures below threshold. Null is not architecture-specific.**

### Canonical LOO-cell result (DecTree-depth3)

| Metric | Value |
|---|---|
| Macro-F1 | 0.2532 |
| Pre-reg threshold | 0.55 |
| Baseline (always-C2) | 0.167 |
| Uplift over baseline | +0.086 |
| **Pre-reg outcome** | **NULL (BELOW THRESHOLD)** |

Per-class F1: NNLS=0.00, Chronos-2=0.73, TimesFM=0.29, Granite-TTM=0.00

Confusion matrix (rows=actual, cols=predicted):

|  | NNLS | Chronos-2 | TimesFM | Granite-TTM |
|---|---|---|---|---|
| **NNLS** | 0 | 0 | 0 | 1 |
| **Chronos-2** | 0 | 4 | 2 | 0 |
| **TimesFM** | 0 | 1 | 1 | 1 |
| **Granite-TTM** | 1 | 0 | 1 | 0 |

Router correctly identifies 4/6 Chronos-2 cells but F1=0 on both minority
classes (NNLS n=1, Granite-TTM n=2). Expected failure mode at n=12 with 4:1
class imbalance.

## EOD status

- Architecture picked: DecTree-depth3
- Pre-reg result: NULL — 0.2532 < 0.55
- CSVs saved: `f1_router_predictions.csv`, `f1_loo_cell_results.csv`
