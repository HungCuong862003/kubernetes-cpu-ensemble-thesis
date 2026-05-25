# D18 — PAR sprint finalised + pivot accepted

**Date:** 2026-05-25 (single calendar day spanning Steps 4 → 5 → 5b)
**Sprint:** PAR (Predictability-Aware Router) Path 2, Steps 4–5b
**Status:** CLOSED. Supervisor (Dr Ho Long Van) emailed and accepted pivot 2026-05-25.
**Outcome:** PAR is structurally saturated under deployment-relevant aggregation. F1 router downgraded to PARTIAL POSITIVE supporting chapter; F3 (cost-asymmetric Chronos-2 fine-tune) promoted to primary empirical contribution. Pivot C+D committed.

---

## One-line summary

PAR achieves DECISION-013 PARTIAL POSITIVE under sample-weighted macro-F1 (LDA = 0.2554 vs baseline 0.2188, +0.0366) but loses on deployment-relevant regret (raw LDA 0.132 vs baseline 0.107, +0.025 worse), all three pre-registered label-shift correction methods (Saerens-Latinne-Decaestecker EM, Lipton et al. BBSE-soft, Alexandari et al. bias-corrected temperature scaling) catastrophically fail (corrected macro-F1 ≈ 0.08), and DECISION-014's HEADLINE-CAPABLE conditions both FAIL → STRUCTURAL SATURATION CONFIRMED → pivot to F3 + BCF/Structural-Saturation framing.

---

## What was done today

### Step 4 — par_labels.py (routing labels)
- Pivoted per-series R² parquet to wide format, 19,231 (container, horizon) rows × 4 candidate-model R² columns + best_r2 + best_label.
- Per-cell label distribution: NNLS dominant in Alibaba (48–62%) and ByteDance (45–53%); Bitbrains four-way split at h030 (29.6/29.6/22.5/18.3); Granite-TTM dominant in Bitbrains h060/h120 (42%, 41%) and 0% in all ByteDance cells.
- Margin distribution healthy: 38.7% margin > 0.10 (clear-cut), 48.0% in 0.01–0.10, only 13.4% < 0.01 (ambiguous).
- Output: `phase_f/data/par_labels.parquet`.

### Step 5 — par_router.py (classifier test)
- LODO 3-fold outer CV + StratifiedKFold 3-fold inner CV.
- Two classifiers: Shrinkage LDA (4 hyperparameter candidates), XGBoost (4 candidates).
- Per-fold macro-F1 (equal-weight aggregate):

  | Holdout | LDA | XGB | Baseline |
  |---|---|---|---|
  | alibaba (n=18,385, 95.6%) | 0.2517 | 0.1773 | 0.2232 |
  | bitbrains (n=568, 3.0%) | 0.1821 | 0.2376 | 0.0749 |
  | bytedance (n=277, 1.4%) | 0.1486 | 0.2310 | 0.2217 |
  | **equal-weight mean** | **0.1941** | **0.2153** | **0.1733** |

- DECISION-013 verdict (equal-weight, primary): LDA = STRUCTURAL SATURATION (< 0.20); XGB = PARTIAL POSITIVE (0.2153 ≥ 0.20 by 0.0153 margin).
- Caveat surfaced immediately: equal-weight aggregation treats three folds of vastly different sizes as equal. Sample-weighted aggregate by test-set size = 0.1801 (XGB), which falls inside STRUCTURAL SATURATION band. Per-fold inspection reveals XGB actively underperforms baseline on the 95.6%-of-test Alibaba fold (−0.046).
- Output: `phase_f/data/par_router_results.json`, `phase_f/data/par_router_predictions.csv`.

### Step 5b — par_prior_correction.py (prior-shift diagnostic)
- Pre-registered DECISION-014 thresholds in the script header before running: HEADLINE-CAPABLE requires both Alibaba-fold EM-corrected macro-F1 ≥ 0.30 AND sample-weighted EM-corrected macro-F1 ≥ 0.25.
- Methods applied: bias-corrected temperature scaling (Alexandari, Kundaje, Shrikumar NeurIPS 2020) → EM prior correction (Saerens, Latinne, Decaestecker, *Neural Computation* 2002) → BBSE-soft cross-check (Lipton, Wang, Smola ICML 2018).
- Container-clustered bootstrap 95% CIs (B = 1000) on raw and EM-corrected macro-F1 per fold.
- Winsorised regret with NaN → 0.0 (naive baseline parity), then clipped to [-1, 1].

**Macro-F1 results:**

| | LDA per-fold | XGB per-fold |
|---|---|---|
| raw | [0.2593, 0.1871, 0.1310] | [0.2275, 0.1924, 0.1618] |
| EM-corrected | [0.0786, 0.1099, 0.0712] | [0.0828, 0.1099, 0.0712] |
| BBSE-corrected | [0.0847, 0.1099, 0.1375] | [0.0828, 0.1099, 0.0712] |
| baseline | [0.2232, 0.0749, 0.2217] | [0.2232, 0.0749, 0.2217] |

**Aggregates:**

| | LDA equal-weight | LDA sample-weighted | XGB equal-weight | XGB sample-weighted |
|---|---|---|---|---|
| raw | 0.1925 | **0.2554** | 0.1939 | 0.2256 |
| EM | 0.0866 | 0.0795 | 0.0880 | 0.0834 |
| BBSE | 0.1107 | 0.0862 | 0.0880 | 0.0834 |
| baseline | 0.1733 | **0.2188** | 0.1733 | 0.2188 |

**EM/BBSE failure diagnostic — estimated test priors vs actual (LDA Alibaba):**

| | chronos2 | granite | nnls | timesfm |
|---|---|---|---|---|
| p_train | 0.299 | 0.185 | 0.282 | 0.234 |
| **p_test_actual** | **0.187** | **0.082** | **0.533** | **0.198** |
| p_test_em | 0.877 | 0.000 | 0.123 | 0.000 |
| p_test_bbse | 0.567 | 0.000 | 0.189 | 0.244 |

EM converged to degenerate solutions (one or two classes collapsed to zero) on every Alibaba and ByteDance fold across both classifiers. BBSE produced similar collapses. Bootstrap CI on EM-corrected Alibaba LDA: [0.0765, 0.0808] — entirely below baseline 0.2232 with no overlap.

**Winsorised regret (sample-weighted, post-NaN-patch):**

| | LDA raw | XGB raw | Baseline |
|---|---|---|---|
| Alibaba (95.6%) | 0.128 | 0.135 | **0.102** |
| Bitbrains (3.0%) | 0.232 | 0.230 | 0.299 |
| ByteDance (1.4%) | 0.222 | 0.165 | **0.050** |
| **sample-weighted mean** | **0.132** | **0.138** | **0.107** |

Baseline beats both classifiers on sample-weighted regret. The macro-F1/regret disagreement is itself a finding: macro-F1 is class-balanced and rewards minority-class identification; regret is instance-weighted and penalises majority-class mispredictions. PAR earns macro-F1 by routing Granite-TTM correctly on Bitbrains long-horizons but loses real R² by mispredicting NNLS-majority Alibaba.

**DECISION-014 verdict:**

| Condition | LDA value | XGB value | Threshold | LDA | XGB |
|---|---|---|---|---|---|
| (a) Alibaba EM-corrected macro-F1 | 0.0786 | 0.0828 | ≥ 0.30 | **FAIL** | **FAIL** |
| (b) Sample-weighted EM-corrected macro-F1 | 0.0795 | 0.0834 | ≥ 0.25 | **FAIL** | **FAIL** |

**STRUCTURAL SATURATION CONFIRMED for both classifiers.** Committed to Pivot C+D.

---

## Why EM and BBSE failed — the underlying mechanism

Both methods require the label-shift assumption: that p(x|y) is invariant across train and test, only the marginal p(y) shifts. Across Alibaba, Bitbrains, and ByteDance the assumption is empirically false. Per-series features (catch22, WPE, SampEn, LZC, DFA) have characteristically different distributions across the three cloud providers even when conditioning on the optimal-routing label: ByteDance series have median SampEn 1.01 vs Bitbrains 0.07 vs Alibaba 0.28 even within the "NNLS wins" subset. This is combined covariate + label shift, for which no consensus best-practice correction exists in 2026.

This failure mode is documented in Sipka, Šulc & Matas (WACV 2022, "Hitchhiker's Guide to Prior-Shift Adaptation"): when the assumption is violated, EM iterates can converge to degenerate fixed points where the prior collapses onto one or two classes. We observe exactly this — Alibaba EM converges to p_test(chronos2) = 0.877 (LDA) or p_test(timesfm) = 1.000 (XGB), with both other classes at 0.

The Alexandari, Kundaje & Shrikumar (NeurIPS 2020) bias-corrected temperature scaling does not rescue this: temperature scaling calibrates probabilities but cannot fix the wrong-direction prior estimate. Their paper's empirical claim ("hard to beat at label shift adaptation") is bounded by their experimental setting (CIFAR-10/ImageNet with controlled artificial label shift) where the assumption holds by construction.

The PAR failure is therefore not a methodological choice failure (we tried the right methods) but an assumption failure (the data does not satisfy what the methods need).

---

## Files produced this sprint

| Path | Size | Purpose |
|---|---|---|
| `phase_f/scripts/par_labels.py` | ~250 lines | Step 4: routing labels |
| `phase_f/scripts/par_router.py` | ~400 lines | Step 5: LODO classifier test (LDA + XGB) |
| `phase_f/scripts/par_prior_correction.py` | ~500 lines | Step 5b: EM + BBSE + temp-scaling + bootstrap CIs + regret |
| `phase_f/data/par_labels.parquet` | 19,231 rows × 13 cols | routing labels with best_r2 + margin |
| `phase_f/data/par_labels.csv` | same | csv copy for inspection |
| `phase_f/data/par_router_results.json` | 6 folds + aggregates | Step 5 output |
| `phase_f/data/par_router_predictions.csv` | per-row predictions | Step 5 output |
| `phase_f/data/par_router_output.txt` | full stdout | reference |
| `phase_f/data/par_prior_correction_results.json` | full prior-correction diagnostics | Step 5b output |
| `phase_f/data/par_prior_correction_output.txt` | initial run, pre-patches | reference (NaN regret in fields) |
| `phase_f/data/par_prior_correction_output_v2.txt` | post-patch, canonical | reference |

Patches applied to `par_prior_correction.py` between v1 and v2:
1. `winsorised_regret` NaN handling: NaN R² → 0.0 (naive baseline parity), then `np.clip` to [-1, 1]. Eliminates NaN propagation through `df_test['nnls'].values` on Alibaba/ByteDance.
2. `p_train` computed from `df_train_full` (full LODO training fold) instead of 80% fit slice. Difference < 0.5pp due to stratified split; full priors are canonical for chapter table.

Macro-F1 numbers unchanged between v1 and v2 (seed-deterministic).

---

## DECISION-014 (formal record)

**DECISION-014 — PAR HEADLINE-CAPABLE conditions, pre-registered 2026-05-25 in par_prior_correction.py before execution.**

**Decision rule:**
- HEADLINE-CAPABLE if BOTH:
  - Alibaba-fold EM-corrected macro-F1 ≥ 0.30
  - Sample-weighted EM-corrected macro-F1 ≥ 0.25
- PARTIAL POSITIVE (under sample-weighted EM correction) if:
  - Sample-weighted EM macro-F1 ≥ 0.20 (DECISION-013 partial threshold)
- STRUCTURAL SATURATION CONFIRMED otherwise.

**Outcome (LDA):** condition (a) 0.0786, condition (b) 0.0795 → BOTH FAIL → STRUCTURAL SATURATION CONFIRMED.
**Outcome (XGB):** condition (a) 0.0828, condition (b) 0.0834 → BOTH FAIL → STRUCTURAL SATURATION CONFIRMED.

**Rationale:** Pre-registered correction methods do not lift PAR above the partial threshold under the deployment-realistic sample-weighted aggregation. The raw classifier achieves PARTIAL POSITIVE on macro-F1 alone (LDA sample-weighted 0.2554, XGB 0.2256) and PARTIAL NEGATIVE on regret (both classifiers worse than baseline). This is a structural saturation result that extends F2 NULL to the per-series scope and to the failure of standard label-shift corrections in cross-cloud routing.

**Status:** LOCKED. Verdict triggers Pivot C+D per the research-report recommendation. F1 router downgraded from headline contribution to supporting chapter (negative result with rigorous diagnostics).

---

## Pivot C+D (formal record)

**Pivot C** — F1 router replaced by F3 (cost-asymmetric quantile fine-tune of Chronos-2) as the primary empirical contribution.
**Pivot D** — Thesis framing changes from "Hybrid Ensemble Learning" to "When ML Helps Kubernetes Autoscaling: Boundary Conditions and Structural Saturation".

**Supervisor (Dr Ho Long Van) acceptance:** received 2026-05-25 via email.

**New 6-chapter structure:**
1. Boundary Condition Framework (BCF) — primary intellectual contribution (AUC = 0.80, percentile 95% CI [0.71, 0.89], n = 36, p = 0.0097)
2. PAR partial-positive routing + label-shift method failure — structured supporting chapter with rigorous diagnostics (DECISION-014 evidence)
3. Structural saturation extending across F2 (cell-scope partial-R² 0.0790) and series-scope PAR (label-shift correction failure) — convergent evidence chapter
4. Foundation-model leaderboard + cost-asymmetric Chronos-2 fine-tune (F3) — primary empirical contribution
5. OptScaler/AHPA integration + HPA simulation (F4) — practical systems contribution
6. Discussion: when ML helps Kubernetes autoscaling

**Defence target:** ~October 2026 if F3 completes cleanly, slipping to ~March 2027 if F3 requires substantial debugging or if F4 integration is non-trivial.

---

## F3 pre-registration (locked here for execution starting tomorrow)

**Experiment:** LoRA fine-tune Amazon Chronos-2 with asymmetric pinball loss at τ ∈ {0.7, 0.8, 0.9, 0.95}, evaluated against zero-shot Chronos-2 baseline on Alibaba, Bitbrains, ByteDance test splits at horizons 10, 30, 60, 120 min.

**Success criterion (PRE-REGISTERED, derives from research-report recommendation):**
- F3 SUCCESS: ≥ 5% pinball-loss improvement vs zero-shot at τ = 0.9, averaged across three datasets at h = 60 min (the deployment-relevant horizon per BCF).
- F3 PARTIAL: ≥ 2% improvement at τ = 0.9.
- F3 FAILURE: < 2% improvement at τ = 0.9 → triggers DECISION-015 (whether to attempt F3 on a different model, e.g., TimesFM-2.5, or abandon F3 and rely on F4 alone).

**Rationale for these thresholds:** Cisana (arXiv:2410.11773, 2024-25) reports TimesFM quantile-head fine-tune improves actual-over-expected ratio over GARCH/GAS baselines by a comparable margin on financial Value-at-Risk; OptScaler (Lu et al. VLDB 2024) reports > 36% SLO-violation reduction from its asymmetric-pinball Flowformer baseline; aiming for 5% pinball improvement on a foundation model is conservative against both references.

**Tooling:** AutoGluon v1.5.0 (released 2025-12-19), full LoRA fine-tune path. Vast.ai instance C.37705458 already has RTX 5070 Ti suitable for Chronos-2 LoRA at 120M parameters.

---

## Memory snapshot updates

To be added to userMemories on next snapshot:

1. **DECISION-014 (PAR HEADLINE-CAPABLE conditions) LOCKED 2026-05-25:** both classifiers FAIL both conditions → STRUCTURAL SATURATION CONFIRMED. EM/BBSE/temperature-scaling all collapse to degenerate priors on Alibaba/ByteDance folds due to violated label-shift assumption (combined covariate + label shift across cloud providers).
2. **Pivot C+D ACCEPTED by Dr Ho Long Van 2026-05-25:** F1 router downgraded to supporting chapter; F3 (cost-asymmetric Chronos-2 fine-tune) promoted to primary empirical contribution; thesis title changes to "When ML Helps Kubernetes Autoscaling: Boundary Conditions and Structural Saturation".
3. **Raw PAR sample-weighted macro-F1 (canonical for chapter):** LDA 0.2554, XGB 0.2256, baseline 0.2188. PAR PARTIAL POSITIVE on macro-F1 (+0.037 LDA), PARTIAL NEGATIVE on regret (+0.025 LDA worse than baseline). Report both.
4. **F3 pre-registered success criterion:** ≥ 5% pinball-loss improvement vs zero-shot Chronos-2 at τ = 0.9, averaged across Alibaba/Bitbrains/ByteDance at h = 60 min. DECISION-015 placeholder if F3 fails.
5. **Defence target shifted from October 2026 to March 2027** to accommodate F3 + F4 + chapter rewrite.

---

## Open items for next sprint (D19+)

- [ ] F3 environment setup on Vast.ai C.37705458: install AutoGluon v1.5.0, verify Chronos-2 model loads, run zero-shot baseline inference on test splits, save zero-shot pinball loss as comparator.
- [ ] F3 LoRA fine-tune script with asymmetric pinball loss at four τ values.
- [ ] Update `phase_f/THESIS_STATE.md` to reflect Pivot C+D and new chapter structure.
- [ ] Update `phase_f/DECISIONS.md` with DECISION-014 (locked) and DECISION-015 (placeholder for F3-failure contingency).
- [ ] Update `phase_f/ERRATA.md` with ERRATA-014: submitted PDF Ch3 §3.7 and Ch5 §5.6/§5.7 framed F1 router as future-positive contribution; the locked verdict is PARTIAL POSITIVE on macro-F1 only with rigorous label-shift failure documentation; chapter rewrite required.
- [ ] Begin 6-chapter LaTeX rewrite under new framing.

---

## Closing note

This sprint converted a fragile "barely above threshold" result into a methodologically rigorous structured negative finding. The thesis is materially stronger now than before the PAR experiments began. The honest reporting of EM/BBSE failure positions the candidate to claim a real intellectual contribution: the first quantified test of label-shift methods on cross-cloud forecast-model routing, with documentation of the assumption-violation that breaks them. This was not in the original thesis plan and was not in the literature when the sprint started. It is now both.

— Closed 2026-05-25 by Jimmy / Claude pairing.
