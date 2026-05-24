# F1 Prep Scope — D8 Design (2026-05-30)

casual notes. design-only document, no implementation. d8 task per the
D6-D8 sequence plan: lock the F1 router design before implementation
starts ~D11+.

## scope

F1 router task: predict which foundation model wins per cell given
predictability features. Goal: macro-F1 ≥ 0.55 per DECISION-005
(Dr. Ho written acceptance 2026-05-22). Cell = (dataset, horizon)
pair. Twelve cells total.

D7 F2 null actively informs F1 design — drop WPE as a candidate
router feature, slimmer search space.

## feature set

### IN — three uncontroversial

**ACF@24h.** Load-bearing F1 feature. The BCF predicate uses ACF@24h
(predicate: ACF@24h > 0.2 AND h ≥ 30 min, with AUC 0.80, p=0.0097).
Demonstrated cell-level ordering of ML benefit at h ≥ 60
(ByteDance > Alibaba > Bitbrains). Per-cell value = dataset-median
from omega_summary.csv: Alibaba 0.316, Bitbrains 0.116, ByteDance
0.489. Three distinct values across the 12 cells.

**horizon_min.** Free input to F1 — the router knows what horizon
it is routing for. Horizon dominates the F2 R²_reduced regression
(0.903 with ACF + horizon, no WPE). Four distinct values: 10, 30,
60, 120. Use as continuous; alternative one-hot encoding for 4
levels is symmetric with similar parameter count.

**CV (coefficient of variation).** cv_stratified_skill.csv shows
container-level ML win rate varies by CV bin (best at the CV 0.3–0.5
bin, 62.3% win rate at h120). At cell level, CV collapses to
dataset-median (3 distinct values). Keep IN but acknowledge the
low effective rank.

### OUT — one F2-driven

**WPE (weighted permutation entropy).** F2 null implies WPE adds
no marginal signal beyond ACF@24h + horizon at the cell level. Per
F2 outline §7, drop WPE as a router feature candidate. Saves a
hyperparameter dimension and reduces collinearity risk.

### Borderline — two argued

**Hurst.** Recommend OUT.
- *For IN:* Hurst is a long-memory metric different in construction
  from ACF; could capture residual signal. Part of omega_summary.csv
  predictability metrics; canonical in the BCF framework.
- *For OUT:* Low cross-dataset variance — Alibaba 0.775, Bitbrains
  0.997, ByteDance 0.956 all sit above 0.7, total range 0.22.
  Within-dataset variance is also low (workloads are all long-memory
  on these traces). Collinearity with ACF@24h (both measure
  long-range dependence).
- *Verdict:* OUT. The 0.22 cross-dataset range gives F1 essentially
  no discriminative signal that ACF@24h doesn't already carry.

**ACF@1h.** Recommend IN.
- *For OUT:* Per-cell granularity collapses to dataset-median (3
  values) like CV/Hurst. Not part of the BCF predicate.
- *For IN:* Different time-scale predictability axis from ACF@24h.
  Critically — Bitbrains has HIGH ACF@1h (0.748) and LOW ACF@24h
  (0.116). This is precisely the cross-dataset pattern F1 needs
  to detect: Bitbrains is the dataset where ML benefit is small
  and unstable, and ACF@24h alone does not distinguish it from
  Alibaba (0.316 vs 0.116 is a small absolute gap). ACF@1h gives
  F1 a second axis to discriminate Bitbrains from Alibaba+ByteDance.
- *Verdict:* IN. The "Bitbrains has unusual short-vs-long-lag
  asymmetry" signal is exactly what a router should pick up.

### Final feature set

Four features, all at cell level:
1. ACF@24h (continuous, dataset-median, 3 distinct values)
2. horizon_min (continuous or one-hot, 4 distinct values)
3. CV (continuous, dataset-median, 3 distinct values)
4. ACF@1h (continuous, dataset-median, 3 distinct values)

Total feature matrix: 12 rows × 4 columns. Three of four features
are dataset-constant (only horizon_min varies within a dataset).
This is a real structural constraint — see honest concerns §2.

## training data

12 cells from `results/bcf/bcf_pairs.csv` (NNLS rows for cell
identification; per D5 close handoff lesson #4, 12 NNLS cells exist
including ByteDance h10 at delta_pp +6.97).

Cells:
- Alibaba: h10, h30, h60, h120 (4 cells)
- Bitbrains: h10, h30, h60, h120 (4 cells)
- ByteDance: h10, h30, h60, h120 (4 cells)

Per-cell label = winner foundation model from
`results/foundation_comparison/leaderboard_v1.csv`. Class distribution
per the canonical leaderboard winner tally:

| Model | Wins | Cells |
|---|---|---|
| Chronos-2 | 6 | (majority class) |
| TimesFM | 3 | |
| Granite-TTM | 2 | |
| NNLS | 1 | |

Class imbalance 6 / 3 / 2 / 1. Macro-averaging chosen for the F1
metric specifically to penalise majority-class baselines (the
discrimination relevant at defence).

## evaluation

**Protocol: LOO-cell cross-validation.** 12 folds, each leaves one
cell out, trains on the other 11, predicts on the held-out cell.
Aggregate predictions into a single 4×4 confusion matrix; compute
macro-averaged F1.

**Threshold:** macro-F1 ≥ 0.55 per DECISION-005 pre-registration.

**Why LOO-cell not k-fold:** k=12 is small. LOO is the standard
default for small-n classification. k=3 or 4 folds risk training-test
contamination if splits go by horizon (router sees same dataset in
train + test) or by dataset (same horizon in both).

**Why not LOO-dataset:** LOO-dataset leaves 4 cells per fold (3 folds
total), training on 8. Tests the harder generalisation question
"can F1 predict winners on a totally unseen dataset?" — important
but more demanding than the pre-registered LOO-cell. Reserve as
robustness check IF LOO-cell clears 0.55. NOT part of the pre-reg
threshold.

## baseline

**Always-predict-Chronos-2.** Per leaderboard winner tally
(Chronos-2:6, TimesFM:3, Granite-TTM:2, NNLS:1), always predicting
the majority class yields:

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| NNLS | 0 | 0 | 1 | 0 | 0 | 0 |
| Chronos-2 | 6 | 6 | 0 | 0.500 | 1.000 | 0.667 |
| TimesFM | 0 | 0 | 3 | 0 | 0 | 0 |
| Granite-TTM | 0 | 0 | 2 | 0 | 0 | 0 |

**Macro-F1 = (0 + 0.667 + 0 + 0) / 4 = 0.167.**

**Honest note on the D6-D8 prompt's "baseline may exceed 0.55"
warning.** The prompt flagged this as a possibility. With macro-F1
(not accuracy), the baseline is 0.167, comfortably below 0.55.
The test IS informative as designed. If accuracy had been the
metric, always-predict-Chronos-2 would score 6/12 = 0.500, marginal
against any plausible threshold. Macro-averaging makes the test
discriminative on this imbalanced 4-class problem.

## honest concerns

1. **n=12 is small for 4-class classification with 4 features.**
   Each LOO fold trains on 11 cells. Even with regularisation,
   parameter count for multinomial logistic regression is 4
   features × 4 classes + 4 intercepts = 20 parameters, over-fitting
   risk against 11 training points. F1 could plausibly report null
   for sample-size reasons rather than feature-set reasons.

2. **Three of four features are dataset-constant at cell level.**
   ACF@24h, CV, ACF@1h all collapse to dataset-median. Only
   horizon_min varies within a dataset. The classifier essentially
   learns a function of (dataset_fixed_effect, horizon) with the
   three predictability features providing dataset-fixed-effect
   modulation. LOO-cell with held-out cells from the same dataset
   tests whether (horizon | dataset) predicts the winner; that's
   easier than true cross-dataset generalisation.

3. **Cross-dataset generalisation is the harder question.**
   LOO-cell tests within-dataset interpolation. LOO-dataset (3
   folds, 4 cells held out per fold) tests cross-dataset
   generalisation. Pre-reg is LOO-cell only. If LOO-cell clears
   0.55 but LOO-dataset doesn't, defence honesty requires
   disclosing both numbers and explaining the distinction.

4. **Classifier architecture is intentionally unlocked at D8.**
   Default candidates for n=12 + 4 features + 4 classes:
   multinomial logistic regression with L2 regularisation;
   k-NN with k=3; shallow decision tree (depth ≤ 3). NOT deep
   neural networks or gradient boosting (over-parameterised).
   Architecture choice locks at F1 implementation, D11+.

5. **F1 baseline is 0.167 macro-F1; threshold is 0.55. Gap is
   substantial but not insurmountable** for a router that
   exploits the (dataset, horizon) interaction. Honest expectation
   range: macro-F1 ∈ [0.30, 0.65]. If F1 lands in [0.30, 0.55],
   it reports null per DECISION-005; the structural finding
   ("dataset-constant features dominate; router signal limited by
   n=12") would be the chapter contribution analogous to F2's
   "ACF saturates the predictability axis".

## DECISION-012 candidate

If F1 scope is firm by D8 EOD, lock as DECISION-012:

> **F1 router design lock (D8, 2026-05-30).** 4-feature classifier
> at cell-level granularity: ACF@24h, horizon_min, CV, ACF@1h.
> WPE excluded per F2 outline §7 implication. Hurst excluded for
> low cross-dataset variance and collinearity with ACF@24h.
> Training: 12 NNLS cells from `results/bcf/bcf_pairs.csv` with
> labels from `results/foundation_comparison/leaderboard_v1.csv`
> winner tally (Chronos-2:6, TimesFM:3, Granite-TTM:2, NNLS:1).
> Evaluation: LOO-cell cross-validation (12 folds), macro-F1 ≥
> 0.55 per DECISION-005 pre-registration. Baseline:
> always-predict-Chronos-2 yields macro-F1 = 0.167. Classifier
> architecture deliberately unlocked at D8; locked at F1
> implementation (D11+) among defaults: multinomial logistic
> regression with L2, k-NN with k=3, shallow decision tree
> (depth ≤ 3). LOO-dataset CV reserved as post-hoc robustness
> check.

## next steps

- D9: memory snapshot (weekly cadence, due Sunday 2026-05-31).
  Capture the F2 null + F1 scope lock if DECISION-012 lands today.
- D10: F0 close batch — apply ERRATA-012 + biblio audit + any
  ERRATA-013+ rows opened. F0 lockdown ends.
- D11+: F1 implementation begins per this scope.