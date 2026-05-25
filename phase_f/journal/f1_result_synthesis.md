# F1 Router — Result Synthesis
**Date:** 2026-06-05 (D14)
**Sources:** f1_setup.py, f1_router.py output confirmed on Vast.ai 2026-06-03/04

---

## Pre-registration outcome

**Threshold: macro-F1 ≥ 0.55 (DECISION-005, Dr. Ho written acceptance 2026-05-22)**
**Result: NULL. Best LOO-cell macro-F1 = 0.2532 across all four candidate architectures.**

Per DECISION-005, null results are reported without post-hoc adjustment.

---

## Architecture sweep (LOO-cell)

| Architecture | Macro-F1 | vs 0.55 | vs baseline 0.167 |
|---|---|---|---|
| DecTree-depth3 | 0.2532 | −0.297 | +0.086 |
| Logistic-L2 | 0.2167 | −0.333 | +0.050 |
| DecTree-depth2 | 0.1429 | −0.407 | −0.024 |
| kNN-k3 | 0.1333 | −0.417 | −0.034 |

Canonical architecture: DecTree-depth3. Null is not architecture-specific.

### Canonical confusion matrix (DecTree-depth3, LOO-cell, n=12)

|  | NNLS | Chronos-2 | TimesFM | Granite-TTM |
|---|---|---|---|---|
| **NNLS** | 0 | 0 | 0 | 1 |
| **Chronos-2** | 0 | 4 | 2 | 0 |
| **TimesFM** | 0 | 1 | 1 | 1 |
| **Granite-TTM** | 1 | 0 | 1 | 0 |

Per-class F1: NNLS=0.00, Chronos-2=0.73, TimesFM=0.29, Granite-TTM=0.00.
Router identifies the majority class (Chronos-2) with moderate accuracy but
cannot distinguish any minority class.

---

## LOO-cell vs LOO-dataset asymmetry

| Evaluation | Macro-F1 | Interpretation |
|---|---|---|
| Trivial baseline | 0.167 | Always predict Chronos-2 |
| LOO-cell (pre-reg) | 0.253 | Within-dataset horizon interpolation |
| LOO-dataset (robustness) | 0.100 | Cross-dataset generalisation |
| Structural (ds+horizon) | 0.225 | No predictability features |
| Full router (4 features) | 0.253 | ACF@24h + CV + ACF@1h + horizon |

LOO-dataset (0.10) falls below the trivial baseline. Cross-dataset
generalisation is essentially zero.

---

## Dual-baseline

Full vs structural delta: +0.028. ACF@24h, CV, ACF@1h are Spearman ρ = ±1.0
correlated with dataset identity. Adding them as explicit features over a
(dataset_dummy + horizon) baseline provides noise-level uplift, not signal.

**Finding: predictability features add no meaningful signal beyond dataset
and horizon knowledge at cell-level granularity with n=12.**

---

## Structural diagnosis (chapter contribution)

The F1 null is a structural consequence of three facts:

1. **Dataset-constant features.** ACF@24h, CV, ACF@1h take one value per
   dataset — they bijectively encode dataset identity, not smooth continuous
   predictors. Spearman ρ = ±1.0 among all three.

2. **n=12 insufficient for 4-class classification with complex label
   structure.** Labels vary non-monotonically with horizon within Alibaba
   (NNLS→Granite→Granite→TimesFM) and Bitbrains (C2→C2→TFM→TFM). A
   depth-3 tree trained on 11 points cannot reliably learn this.

3. **Cross-dataset generalisation is zero.** LOO-dataset macro-F1 = 0.10
   confirms that the patterns learned per dataset do not transfer. A production
   router needs far more than 3 datasets.

These three structural facts are the chapter contribution. They constrain
future work: a viable router requires (a) per-series features rather than
dataset-level medians, (b) more than 3 training datasets, and (c) a metric
that does not collapse to dataset identity.

The BCF framework (binary ML-wins/loses, AUC=0.80, p=0.0097) achieves
reliable discrimination because binary classification is structurally simpler
than 4-class winner prediction. This contrast strengthens BCF as the thesis's
primary contribution.

---

## Defence Q&A framing

**Q: Why did F1 fail the pre-registered threshold?**
Three structural reasons: (1) cell-level predictability features collapse to
dataset identifiers at n=12; (2) four-class imbalanced classification with
minority counts of 1 and 2 is reliably unsolvable at this sample size; (3)
LOO-dataset = 0.10 confirms the learned mapping does not generalise. These are
sample-size and granularity constraints, not implementation failures.

**Q: What does the positive LOO-cell score of 0.25 mean?**
It reflects a dataset-identity classifier with noisy horizon splits, not
genuine predictability-aware routing. The +0.086 uplift over trivial evaporates
under cross-dataset LOO. The result is interpretable but not useful in production.
