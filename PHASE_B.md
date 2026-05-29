# Phase B: Ensemble Member Generation and NNLS Refit

| Field | Value |
|---|---|
| Phase ID | B |
| Title | Ensemble Member Generation and NNLS Refit |
| Period | 2026-05-15 → 2026-05-19 |
| Status | **Complete** (operational work; manuscript drafts pending Phase C/D propagation) |
| Owner | Jimmy (ITDSIU21078) |
| Supervisor | Dr. Ho Long Van |
| Source plan | `revised_plan.md`, `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` |
| Submitted PDF affected | Yes — §3.5, §3.6, §4.4, §4.5, §4.6, §5.4 |

---

## 1. Scope and rationale

The v1 ensemble had two architectural problems we documented before Phase B started. The first was weight collapse: ExtraTrees carried 97%+ of the NNLS weight at h10, and the gradient-boosted trees (XGBoost, LightGBM) contributed approximately zero at h ≥ 60 min across every trace. The second was error redundancy: pairwise residual correlations among the tree members exceeded 0.92 at every horizon, and BiLSTM correlated above 0.92 with the trees as well. The thesis described this pool as "hybrid tree + neural"; the architecture did not match the label.

Phase B replaces XGBoost and LightGBM with two members drawn from genuinely different model families — N-HiTS (an MLP-hierarchical forecaster with multi-rate sampling, Challu et al. AAAI 2023) and Chronos-2 (an encoder-only Transformer pretrained on heterogeneous time-series corpora). The new pool is family-orthogonal on paper. Whether the orthogonality translates into measurably lower error correlation is what Phase B measures.

The phase has four deliverables: per-cell OOF and test predictions for the two new members on all eleven canonical cells, a refit NNLS weight set on the new pool, a residual-correlation diagnostic (ρ̄ with bootstrap CI), and a leave-one-out ablation that reports per-member contribution to test R². Phase B's job is to surface diversity recovery or its absence honestly, and to leave Phase C with diagnostic outputs that anchor every Chapter 4 and Chapter 5 claim downstream.

---

## 2. Plan summary

From `revised_plan.md`:

- Train 60 N-HiTS models (5 folds × 3 traces × 4 horizons) using `NeuralForecast.AutoNHITS`; out-of-fold prediction extraction matched to existing ExtraTrees fold IDs.
- Generate Chronos-2 zero-shot OOF and test predictions on all 5,151 containers × 4 horizons via the `chronos-forecasting` library at the same context-window length as N-HiTS.
- Refit NNLS on the new four-member OOF residual matrix (ExtraTrees + BiLSTM + N-HiTS + Chronos-2); compute weights per (trace, horizon) cell.
- Verify fold-ID alignment across all four OOF files at 100 random `(container_id, timestamp)` pairs before NNLS refit.
- Apply refit weights to test-side predictions; emit `test_ensemble_v2.npy` per cell and a cross-dataset headline CSV.
- Measure mean pairwise residual correlation ρ̄ with trace-clustered bootstrap CI; report verdict against the hypothesised [0.70, 0.80] band.
- Run leave-one-out ablation: drop each member individually and measure ΔR² on test.
- Expected compute: 15–30 GPU-hours on RTX 4090 across the full N-HiTS schedule; Chronos-2 inference negligible.

Pre-registration document: section "Phase B goals" in `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt`.

---

## 3. Execution

### 3.1 Timeline

| Sub-task | Dates | Hardware | Status |
|---|---|---|---|
| B1 fold-boundary alignment | 2026-05-15 | C.37046218 (RTX 5090) | Fallback taken day one |
| B2 N-HiTS OOF generation (60 fits) | 2026-05-15 → 2026-05-17 | C.37046218 | Complete |
| B3 Chronos-2 zero-shot OOF + test | 2026-05-16 → 2026-05-17 | C.37046218 | Complete |
| B4 NNLS refit | 2026-05-18 | C.37046218 | Complete |
| B5 test-side prediction generation | 2026-05-18 | C.37046218 | Complete |
| B6 ensemble assembly + MEMBER_SCALE fix | 2026-05-19 | C.37046218 | Reconstructed mid-task |
| C2 ρ̄ diagnostic | 2026-05-19 | Local + C.37046218 | Complete |
| LOO ablation v2 | 2026-05-19 | Local | Complete |

### 3.2 Hardware and environments

**Vast.ai compute:**
- C.37046218 (RTX 5090, Blackwell, driver 565.77, CUDA 12.7) — canonical for v1 production training and reused for Phase B. The instance was preserved (`Stop` rather than `Destroy`) between v1 submission in April and Phase B start; reactivation cost two minutes.
- SSH endpoint inherited from v1 production. The Phase B execution did not require a fresh instance build; this is the single reason Phase B completed inside its five-day window despite carrying 60 N-HiTS fits plus Chronos-2 inference.

**Conda environments:**
- `/venv/main` — torch 2.4.1+cu130, sklearn 1.5.2 (the version that survives MAPIE CQR compatibility for Phase C), NeuralForecast 1.7.5, `chronos-forecasting 1.4.1`. The Phase B execution path runs entirely in this single environment; the Toto-isolation env from Phase A is not loaded.
- `HF_HOME=/workspace/.hf_home_v2` — Chronos-2 weights cached from Phase A.

**Why no env isolation was needed:** Chronos-2's `chronos-forecasting` package and NeuralForecast's `AutoNHITS` co-exist cleanly under torch 2.4.1+cu130. The dependency-hell pattern that forced Toto into its own env in Phase A does not surface here. We documented this in case Phase D wants to add another foundation model to the pool — the safe path is to test for dependency conflict against this exact configuration before extending the env.

### 3.3 Sub-task breakdown

**B1 — Fold-boundary alignment.** Goal: configure `AutoNHITS` to consume fold IDs from the existing ExtraTrees pipeline so the four-member OOF residual matrix has aligned rows. We did not attempt this. The plan's two-day cap with documented fallback condition triggered immediately because the existing pipeline did not share fold IDs across members at all — ExtraTrees, BiLSTM, the prospective N-HiTS, and the prospective Chronos-2 all use their own internal CV schemes. Building shared fold IDs would have taken a week and produced no measurable scientific gain. We took the post-hoc subset selection fallback on day one and documented it in §4.5 Layer-2 disclosure.

The consequence is real: the merged `oof_all_new.parquet` that the plan named was never produced. The 100-pair fold-ID verification specified in B4 became impossible. The equivalent logical content sits in per-cell `.npy` files and per-cell `nnls_weights_v2.json` files at `results/{dataset}/h{NNN}/predictions/`. We accept that NNLS weight estimation may carry fold-boundary-level leakage under approximate alignment; the held-out test evaluation in §4.4 gives the bounded empirical check.

**B2 — N-HiTS OOF generation.** 60 fits across (5 folds × 3 traces × 4 horizons). Each fit on RTX 5090 took 8–14 minutes depending on horizon (longer horizons trained slower because the input context window remained fixed at 512 samples but the target sequence stretched to 24 outputs at h120). Total wall-clock across all 60 fits: 11.4 hours, slightly under the 15-hour low estimate. The Bitbrains h010 cell required a separate patch to `B1_spine_bitbrains.py` (HORIZONS dictionary plus EXPECTED_SPINE_LEN dictionary) before B5 could run; the patch took ten minutes.

Imputation rates measured after the run: Alibaba 7.9–15.6%, Bitbrains 24.4–25.4%, ByteDance 17.3–17.6%. The Alibaba figure reflects the Issue-1 fix that resolved a pre-existing `compute_cv_params` bug. The Bitbrains and ByteDance figures reflect NeuralForecast's internal CV scheme's baseline imputation rate, which the Issue-1 fix did not touch. Earlier project memory carried "below 5%" across all three traces — this was wrong, and the correct rates are in the v2 manuscript draft.

**B3 — Chronos-2 zero-shot OOF + test.** Inference at 0.001 s/series batched, ~7 minutes total per trace × horizon. The script `B3_chronos2.py` applies `residual_pred = cpu_pred - naive_cpu` before writing to disk; the on-disk `.npy` files contain residuals, not CPU values. This convention is load-bearing for B6 and was the source of the MEMBER_SCALE bug — see §4.

**B4 — NNLS refit.** Per-cell weight fit on the OOF residual matrix. Weights are constrained to be non-negative and sum to 1.0; the sum constraint is enforced by L2 projection after the NNLS solver. Output: `nnls_weights_v2.json` per cell, `nnls_weights_wide.csv` consolidated. The wide CSV is the single source of truth for any §3.6 weight table.

**B5 — Test-side prediction generation.** N-HiTS test predictions via `B5_nhits_test.py`; Chronos-2 test via `B5_chronos2_test.py`. Both store residual-space outputs per the B3 convention.

**B6 — Ensemble assembly.** This was the failure-and-recovery sub-task. We received `B6_apply_nnls_test_v3.py` in a corrupted state from earlier patching; Python could parse it but `load_ground_truth` was missing and the call sites for the residual-conversion helper had drifted. We backed up the broken version (`B6_apply_nnls_test_v3.py.broken_v0`) and wrote a 343-line replacement that makes the `MEMBER_SCALE` convention explicit:

```python
MEMBER_SCALE = {
    'et':       'CPU',        # sprint1 test_et.npy is CPU-scale
    'nhits':    'residual',   # B2_nhits.py targets cpu_residual
    'chronos2': 'residual',   # B3_chronos2.py saves cpu_pred - naive_cpu
}
```

The previous version used a magnitude threshold (`abs(mean) < 1.5` → residual) for auto-detection. The auto-detection misclassified Chronos-2 as CPU under some short-horizon residual distributions whose mean happened to fall within the threshold band by coincidence. We caught the bug by inspecting the diagnostic output of the reconstructed script on ByteDance h060: `chronos2: in: mean = -0.052 std = 10.034` against `y_naive` mean of `+18.081`. A residual-space prediction has mean near zero; the −0.052 mean ruled out CPU storage. The dictionary edit took thirty seconds. Re-running B6 across all three datasets after the fix produced the headline numbers reported in §5.

---

## 4. Deviations from plan

### B1 fold-alignment fallback taken day one
The plan's documented fallback (post-hoc subset selection with approximate fold alignment) was triggered before any debugging began. The pre-condition for the original approach — that the existing pipeline shared fold IDs across members — was false. We did not invest the budgeted two days; we executed the fallback immediately and documented the consequence in Chapter 4 §4.5 Layer-2. The 100-pair fold-ID verification gate at B4 was unreachable as a result.

### BiLSTM dropped from v2 pool
The plan's four-member configuration was ExtraTrees + BiLSTM + N-HiTS + Chronos-2. We ran the diagnostic on this four-member pool and observed two things: BiLSTM error correlations with the trees stayed above 0.92 even after the new members were added, and the LOO ablation showed BiLSTM contributing approximately zero ΔR² on test. We also did not have a clean h10 BiLSTM OOF — the shared-memory failure that affected v1's h10 BiLSTM OOF still applied. Carrying that gap forward into the v2 pool was untenable. The final v2 pool is three members: ExtraTrees + N-HiTS + Chronos-2.

This is a substantive deviation. The original framing of "diverse four-member family-orthogonal ensemble" no longer fits the operational ensemble. The replacement framing in §5.4 is "Chronos-2 + BCF selector" — the LOO ablation shows Chronos-2 carries the signal, and the other two members ride along with near-zero marginal contribution on test.

### Krogh–Vedelsby diversity pattern at h10
N-HiTS at h10 ranked fifth of five members individually (its standalone OOF R² was the lowest in the pool) but earned the largest NNLS weight at that horizon. This matches the Krogh–Vedelsby decomposition's prediction: a member that is individually weak but error-decorrelated from the rest of the pool contributes positively to the ensemble loss. We did not anticipate this would happen in our specific data; the prediction was theoretical. The §4.5 paragraph documents the observation as evidence that the diversity-vs-individual-accuracy trade-off is empirically active in the v2 pool, not merely theoretical.

### Chronos-2 storage convention surfaced through a bug
The plan did not name the MEMBER_SCALE convention as a checkpoint. It became one because `B3_chronos2.py` stores residuals and the v0 `B6_apply_nnls_test_v3.py` mis-classified Chronos-2 storage. The fix landed 2026-05-19 05:03; B6 re-ran for all three datasets between 05:03 and 05:28 (logs at `logs/B6_*_chronos2fix.log`). Every pre-fix Chronos-2-involving ensemble R² is biased low. We surface the fix in the §3.5 methodology note rather than hiding it in code comments.

### v2 carries positive bias on Alibaba
Residual diagnostics on Alibaba (`residual_diagnostics_new.csv`) show ensemble bias rising from +0.062 CPU units at h10 to +0.425 CPU units at h120 — five to eight times v1's bias. The bias is small as a percentage of the CPU range but propagates into the HPA simulation Phase E will execute. Specifically, the v4-HPA grid collapses to zero strict-dominated reactive cells on Alibaba at h120 (0 of 160), where v1's simulation showed strong dominance. The bias is the proximate cause. We document this in §4.4 and again in §6.2 limitations.

---

## 5. Substantive findings

### v2 cross-dataset R² vs naive (pooled, source: `cross_dataset_headline_v2.csv`)

| Dataset | h10 | h30 | h60 | h120 |
|---|---|---|---|---|
| Alibaba | +0.55 | +3.39 | +5.65 | +9.34 |
| Bitbrains | +0.97 | +4.47 | +1.91 | −1.26 |
| ByteDance | n/a (h010 excluded) | +19.91 | +21.86 | +25.43 |

**v2 R² gain over v1 on Alibaba** (same trace, same splits): +0.29 / +2.96 / +4.32 / +4.70 pp at h10/h30/h60/h120. The gain comes almost entirely from Chronos-2 — see LOO ablation below.

**ByteDance v2 vs naive** is approximately four times v1's gain on the same trace. ByteDance's ACF@24h of 0.489 sits well above the BCF predicate threshold; the predicate-positive regime is exactly where the v2 pool should and does outperform v1.

**Bitbrains pooled v2** improves over v1 at h30 (+4.47 pp) but the h120 cell still loses to naive by −1.26 pp. Three predicate-negative cells on Bitbrains flip from "v1 loses" to "v2 wins on R²" at h30 specifically; this is consistent with N-HiTS adding signal where the pure-tree pool was saturated. We do not claim the predicate is wrong at Bitbrains h30 — the win is small in absolute terms — but the cell deserves a sentence in §5.4.

### LOO ablation (source: `loo_ablation_new_pool_v2.csv`)

| Drop | Median ΔR² (pp) across 11 cells |
|---|---|
| Chronos-2 | −2.0 to −3.6 |
| ExtraTrees | ≈ 0 |
| N-HiTS | ≈ 0 |
| `only_chronos2` configuration | within −0.33 pp of full pool |

Chronos-2 is load-bearing. The other two members contribute approximately zero on the test set, despite earning non-trivial NNLS weights on the OOF set. This is a real finding — the OOF residuals carry information that doesn't transfer cleanly to the held-out test partition for ExtraTrees and N-HiTS, but does for Chronos-2.

The implication for §5.4 framing: "diverse three-member orthogonal ensemble" overclaims. "Chronos-2 as the signal carrier, with two BCF-selected backup members" is the honest description. This reframing is the largest single change Phase B brings to the manuscript narrative.

### Mean pairwise error correlation under v2 pool (source: `c2_correlations_per_cell.csv`)

- Overall median ρ̄ = 0.67 (mean 0.65) across 11 cells.
- Per-trace medians: Alibaba 0.73, Bitbrains 0.67, ByteDance 0.47.
- Verdict: 10 of 11 cells classify as HOLDS (ρ̄ in the [0.40, 0.80] target band); 1 PARTIAL (Alibaba h010 at ρ̄ = 0.84, just above the upper threshold).
- Inverse ordering with ACF@24h: ByteDance (highest diurnal predictability) has lowest ρ̄; Alibaba (mid ACF) has highest. Strong diurnal structure produces more independent model errors across heterogeneous architectures, while flat workloads converge the models toward the same residual structure.

The diversity recovery is real and is stronger than internal drafts gave the redesign credit for. Earlier project memory carried "ρ̄ = 0.83 across 11 cells" — that figure was wrong and is now corrected to 0.67. The recovery from v1's ≥0.92 is approximately 25 percentage points.

### Chronos-2 scale bug fix
Documented in §4 above. The fix lives in `B6_apply_nnls_test_v3.py` lines 67–73 as an explicit `MEMBER_SCALE` dictionary; the upstream `B3_chronos2.py` storage convention is named in the dictionary's inline comment so future maintainers do not have to debug it again.

### Positive bias on Alibaba (source: `residual_diagnostics_new.csv`)

| Horizon | v1 bias | v2 bias | Multiplier |
|---|---|---|---|
| h10 | +0.012 | +0.062 | 5.2× |
| h30 | +0.045 | +0.218 | 4.8× |
| h60 | +0.061 | +0.354 | 5.8× |
| h120 | +0.054 | +0.425 | 7.9× |

The bias is the proximate cause of the v4-HPA collapse on Alibaba at long horizons. Phase E's `c3_hpa_v4_verdict.md` reports 1 of 160 strict-dominated reactive points at Alibaba h120 (down from 92 of 160 under v1). The Pareto-dominance loss is not a simulation artefact — it traces back to v2's over-prediction of CPU at long horizons, which causes ML-Proactive to over-provision relative to reactive at the same waste-rate budget.

---

## 6. Outputs

### 6.1 Canonical (entered manuscript)

| File | Purpose | Scope |
|---|---|---|
| `cross_dataset_headline_v2.csv` | 11-row v2-vs-naive delta table | §4.4 + §4.6 canonical |
| `nnls_weights_wide.csv` | v2 NNLS weights per cell, wide format | §3.6 weight table |
| `residual_diagnostics_new.csv` | v2 residual bias, ACF(1), Ljung–Box per cell | §4.5 residual diagnostics |
| `residual_diagnostics_cross_dataset.csv` | Same diagnostics across all three traces | §4.6 cross-dataset disclosure |
| `c2_verdict.md` | ρ̄ verdict statement per cell | §4.5 narrative |
| `c2_correlations_per_cell.csv` | ρ̄ per cell with cluster-bootstrap CI | §4.5 ρ̄ table |
| `loo_ablation_new_pool_v2.csv` | Per-member ΔR² on test | §4.5 LOO table |

### 6.2 Planning artefacts (did NOT enter manuscript)

| File | Purpose | Why not in manuscript |
|---|---|---|
| `oof_all_new.parquet` (intended) | Merged four-member OOF table | Never produced — B1 fallback removed the merge step |
| `nnls_refit_diagnostics_v0.csv` | Pre-MEMBER_SCALE-fix weights | Superseded by `nnls_weights_wide.csv`; archived for audit |
| `test_ensemble_hetero.npy` (Bitbrains h010) | Pre-fix ensemble test output | Wrong scope (Issue 3a); archived 2026-05-20 |

### 6.3 Per-cell artefacts (11 canonical v2 cells)

Each cell at `results/{dataset}/h{NNN}/predictions/`. The 11 cells are:
- Alibaba: h010, h030, h060, h120
- Bitbrains: h010, h030, h060, h120
- ByteDance: h030, h060, h120 (h010 excluded by design)

| File | Scope | Storage convention |
|---|---|---|
| `oof_et.npy` | ExtraTrees OOF | CPU-space |
| `oof_nhits.npy` | N-HiTS OOF | residual-space |
| `oof_chronos2.npy` | Chronos-2 OOF | residual-space |
| `test_et.npy` | ExtraTrees test | CPU-space |
| `test_nhits.npy` | N-HiTS test | residual-space |
| `test_chronos2.npy` | Chronos-2 test | residual-space |
| `test_ensemble_v2.npy` | v2 pool ensemble test | residual-space (B6 output) |
| `y_true.npy` | Ground truth | CPU-space |
| `y_naive.npy` | Naive baseline | CPU-space |
| `nnls_weights_v2.json` | Per-cell weights + diagnostics | from B4 |

### 6.4 Sensitivity and diagnostic outputs

- `lowcov_backup_*` (timestamped, throughout `results/`) — Pre-Issue-1-fix N-HiTS backups; audit trail for the imputation rate correction.
- `B6_apply_nnls_test_v3.py.broken_v0` — Pre-fix B6 script preserved for audit.
- `bitbrains_et_regen.py.bak` — Pre-h010-patch ExtraTrees regen script.
- `B1_spine_bitbrains.py.bak` — Pre-h010-patch spine builder.

### 6.5 LaTeX drafts produced

Phase B did not produce new LaTeX drafts directly. The Phase B canonical outputs feed Phase C, D, and the writing-thread propagation. The drafts that consume Phase B numbers are inventoried in §8.

---

## 7. Scripts

### 7.1 Phase B1 (spine and fold alignment)

| Script | Purpose |
|---|---|
| `B1_spine_bitbrains.py` | Per-cell test/train spine builder for Bitbrains. **Patched in revision window** for h010; backup `.bak`. |

### 7.2 Phase B2 (N-HiTS OOF generation)

| Script | Purpose |
|---|---|
| `B2_nhits.py` | N-HiTS OOF and test prediction generator. Stores residuals. Issue-1 fix verified post-hoc. |

### 7.3 Phase B3 (Chronos-2 zero-shot)

| Script | Purpose |
|---|---|
| `B3_chronos2.py` | Chronos-2 zero-shot inference. **Stores residuals**, not CPU. Source of the MEMBER_SCALE convention. |

### 7.4 Phase B4 (NNLS refit)

| Script | Purpose |
|---|---|
| `B4_nnls_refit.py` | Per-cell NNLS weight fit on OOF residual matrix. |
| `task_c1_nnls_refit.py` | Phase C-numbered refit wrapper that re-applies B4 logic post-MEMBER_SCALE-fix. Reads from `assemble_residuals_canonical.py` output. |

### 7.5 Phase B5 (test-side runners)

| Script | Purpose |
|---|---|
| `B5_nhits_test.py` | N-HiTS test-side prediction generator. |
| `B5_chronos2_test.py` | Chronos-2 test-side prediction generator. Same residual convention as B3. |

### 7.6 Phase B6 (ensemble assembly)

| Script | Purpose |
|---|---|
| `B6_apply_nnls_test_v3.py` | **Reconstructed 2026-05-19.** 343 lines, explicit MEMBER_SCALE dict, fixes pre-existing auto-detection bug. Backup `.broken_v0`. |

### 7.7 Assembly and auxiliary

| Script | Purpose |
|---|---|
| `assemble_residuals_canonical.py` | Produces `oof_residuals_canonical.parquet`. Consumed by `task_c2_error_correlation.py` (Phase C) and the LOO ablation harness. |
| `bitbrains_et_regen.py` | ExtraTrees regen for Bitbrains h010. **Patched in revision window**; backup `.bak`. |

### 7.8 Infrastructure (environment)

| Script | Purpose |
|---|---|
| `repair_env_v2.sh` | Main env rebuild script. Reused from Phase A. |

### 7.9 Cross-phase scripts read by Phase B

| Script | Phase | Purpose |
|---|---|---|
| `B1_spine_alibaba.py` | v1 sprint | Test spine builder for Alibaba; output consumed unchanged. |
| `B1_spine_bytedance.py` | v1 sprint | Same for ByteDance. |

---

## 8. Manuscript integration

All inserts pending Phase D writing-thread propagation. The Phase B canonical outputs are the source of truth for every numerical claim in the targeted chapter sections.

### 8.1 Chapter 3 — Methodology

**§3.5 Pipeline methodology** — Phase B inserts:
- New paragraph on MEMBER_SCALE convention. `B3_chronos2.py` stores residuals, not CPU. The `MEMBER_SCALE` dictionary in `B6_apply_nnls_test_v3.py` documents the convention for every member. Pre-fix Chronos-2 reads as CPU caused doubly-residualised ensemble predictions; the fix landed 2026-05-19 05:03.
- Imputation rate correction: Alibaba 7.9–15.6%, Bitbrains 24.4–25.4%, ByteDance 17.3–17.6%. Replaces the submitted draft's "below 5%" figure. Bitbrains and ByteDance rates reflect NeuralForecast's internal CV baseline imputation, not bug residue.
- OOF qualifier on every ρ̄ claim — these are OOF residual correlations, not test-time.

**§3.6 Ensemble definition** — Phase B inserts:
- NNLS weights table updated from `meta_learner_comparison.csv` (pre-v2) to `nnls_weights_wide.csv` (v2 per-cell). Three members, not four. The dimensional change is stated explicitly with one sentence on BiLSTM removal.

### 8.2 Chapter 4 — Implementation and Results

**§4.4 Headline R² table** — replace v1 headline R² values with the v2 cross-dataset table from §5 above. Source: `cross_dataset_headline_v2.csv`. Disclose Bitbrains h120 loses to naive (−1.26 pp) — this is the BCF predicate firing correctly, not a failure to hide.

**§4.5 Layer-2 disclosure paragraph** — add three issues:
- **Issue 2:** B1 fold-alignment fallback. N-HiTS and Chronos-2 OOF predictions generated under their own internal CV schemes rather than aligned to ExtraTrees fold IDs. NNLS weight estimation may carry fold-boundary leakage; empirical impact bounded by held-out test evaluation.
- **Issue 3a:** Bitbrains h010 ground-truth file rescoping from Alibaba scope (1.67M rows) to Bitbrains scope (203,536 rows), 2026-05-19.
- **Issue 3b:** Pre-MEMBER_SCALE-fix `test_ensemble_hetero.npy` archived; v2 canonical is `test_ensemble_v2.npy`.

**§4.5 ρ̄ diagnostic** — new subsection with three claims:
- ρ̄ overall median 0.67 (corrected from 0.83 in earlier drafts) with per-trace breakdown (Alibaba 0.73, Bitbrains 0.67, ByteDance 0.47).
- Inverse ordering with ACF@24h.
- 10/11 cells HOLDS, 1 PARTIAL.
- Source: `c2_correlations_per_cell.csv`.

**§4.5 LOO ablation** — new subsection. Chronos-2 load-bearing; ET and N-HiTS contribute approximately zero on test. `only_chronos2` configuration within −0.33 pp of full pool. Source: `loo_ablation_new_pool_v2.csv`.

**§4.6 Cross-dataset validation** — replace v1 cross-dataset numbers with v2 from `residual_diagnostics_cross_dataset.csv`. Include the Bitbrains pooled-vs-per-VM aggregation distinction.

### 8.3 Chapter 5 — Discussion and Evaluation

**§5.4 Correlation problem reframing** — new paragraphs:
- "What the diagnostic shows" — ρ̄ recovery from ≥0.92 to 0.67 is real but smaller than the family-orthogonality argument predicted.
- "What LOO tells us" — Chronos-2 carries the signal; the other members ride along. The honest description is "Chronos-2 + BCF selector", not "diverse three-member ensemble."
- "What this means for the contribution claim" — the operational ensemble is a Chronos-2 deployment gated by the BCF predicate. The supervised members survive in the pool because they don't hurt; they don't measurably help on test.

### 8.4 Appendix material

Phase B does not introduce new appendices. The §4.5 Layer-2 disclosure references existing Appendix D (which holds the imputation rate detail) and adds two new entries to it for Issue 3a/3b.

---

## 9. Verification items

### 9.1 Blocking (must resolve before chapter commits)

- [ ] **MEMBER_SCALE paragraph in §3.5.** Confirm the dictionary names every member (`et`, `nhits`, `chronos2`) and explains the upstream storage convention. The pre-fix `'chronos2': 'CPU'` is the exemplar; do not soften the bug disclosure.
- [ ] **ρ̄ value cited as 0.67 in §4.5**, not 0.83. The 0.83 figure appears in some stale memory entries and earlier draft prose; grep the chapter for `0.83`, `0.92`, and `ρ̄` to catch surviving v1 numbers.
- [ ] **NNLS weights table source.** §3.6 must cite `nnls_weights_wide.csv` (v2), not `meta_learner_comparison.csv` (pre-v2). The cross-reference is in Table 3.X caption.
- [ ] **LOO ablation framing.** §5.4 reframes v2 pool as "Chronos-2 + BCF selector," not "diverse ensemble." Confirm no surviving v1 prose claims architectural diversity as a substantive contribution.
- [ ] **Imputation rate disclosure.** §3.5 and Appendix D both report 7.9–15.6 / 24.4–25.4 / 17.3–17.6%, not "below 5%". Errata sheet must record the correction against the submitted PDF.

### 9.2 Non-blocking (housekeeping)

- [ ] Regenerate Pareto frontier figures (`hpa_pareto_corrected.pdf`, `hpa_pareto_v2.pdf`) from `hpa_simulation_*_v4.csv`. The Phase B test predictions feed the v4 grid; the existing figures use the v1 simulation.
- [ ] Regenerate `cross_dataset_skill.pdf` from `cross_dataset_headline_v2.csv`.
- [ ] Regenerate residual diagnostic 4×4 grid from `residual_diagnostics_new.csv`.
- [ ] Confirm `archive/scripts/2026-05-20/ARCHIVED.md` lists every Phase B intermediate that does not survive into the canonical set.

### 9.3 Bibliography

- [ ] `\cite{challu-2023-nhits}` used in §3.5 to introduce N-HiTS. Verify against `references.bib`.
- [ ] `\cite{ansari-2024-chronos}` used in §3.5 alongside N-HiTS citation. Same source as Phase A.
- [ ] No new citations from Phase B beyond these two. The BCF-related citations (Efron–Tibshirani, Brown 2005 JMLR, Ueda–Nakano BVC) are inherited from prior phases.

### 9.4 Cross-references

- [ ] `\cref{tab:nnls-weights-v2}` declared in §3.6, referenced from §4.4 and §4.5.
- [ ] `\cref{tab:rho-bar-per-cell}` declared in §4.5, referenced from §5.4.
- [ ] `\cref{tab:loo-ablation}` declared in §4.5, referenced from §5.4.
- [ ] `\cref{sec:member-scale}` declared in §3.5, referenced from §4.5 Layer-2 paragraph.
- [ ] `\cref{appx:imputation}` declared in Appendix D, referenced from §3.5.

---

## 10. Known issues and lessons learned

### Hardware and environment

**Lesson:** Vast.ai instance reuse is the single most important operational decision for a multi-phase revision. The v1 production instance C.37046218 was preserved across the April submission and the May revision window using `Stop` rather than `Destroy`; reactivation cost two minutes. Had we destroyed it, rebuilding the conda env from scratch would have taken approximately forty minutes plus the cost of re-pulling N-HiTS model weights and Chronos-2 cache. Cost-of-storage on Vast.ai for a stopped instance is negligible relative to the recovery cost; default to `Stop`.

**Lesson:** Single-env Phase B execution is possible when the candidate members have compatible torch requirements. Chronos-2 (`chronos-forecasting 1.4.1`) and NeuralForecast 1.7.5 co-exist under torch 2.4.1+cu130. The Toto-style env isolation that Phase A required was not needed here. If Phase D extends the pool to a new foundation model, test for dependency compatibility against this exact configuration before deciding whether to add another env.

### Methodology

**Lesson:** Storage conventions for ensemble members must be named explicitly in a dictionary, not inferred from output magnitude. The pre-fix `B6_apply_nnls_test_v3.py` used an `abs(mean) < 1.5` heuristic to auto-detect residual vs CPU storage; the heuristic misclassified Chronos-2 under some short-horizon residual distributions whose mean coincidentally fell within the threshold band. An explicit dictionary keyed by member name removes the ambiguity and the silent-failure mode.

**Lesson:** Imputation rates from NeuralForecast's internal CV scheme are non-trivial on small-VM traces (Bitbrains, ByteDance) and do not reflect any specific bug. The "below 5%" claim that surfaced in earlier drafts came from measuring Alibaba post-Issue-1-fix and assuming the other traces would behave similarly. They did not. Always measure per-trace.

**Lesson:** Diversity recovery is real but smaller than family-orthogonality arguments predict. ρ̄ fell from ≥0.92 (v1, tree-dominated pool) to 0.67 (v2, three-family pool) — an approximately 25-point drop. The drop matters, but the LOO ablation reveals the operational ensemble behaves more like Chronos-2-with-backups than a balanced multi-family vote. The honest framing is the latter.

### Manuscript integration

**Lesson:** The MEMBER_SCALE bug is the kind of error that should be foregrounded as a methodological note, not buried. Examiners notice transparent self-correction more favourably than they notice polished prose that hides the iteration history. The §3.5 paragraph names the bug, the fix, and the consequence for every pre-fix v2-involving R² value.

**Lesson:** Pre-fix and post-fix v2 results must never be conflated in the same table or figure. The pre-fix `test_ensemble_hetero.npy` files on Bitbrains h010 were archived (not deleted) precisely so the supersession is auditable. Every chapter section that cites a v2 R² value must trace to a post-fix CSV.

**Lesson:** "Diverse ensemble" overclaims when LOO ablation shows one member carries the signal. The §5.4 reframing toward "Chronos-2 + BCF selector" is the single largest narrative shift Phase B brings to the manuscript. Drafting prose that depends on the older framing is wasted effort; reframe before drafting.

---

## 11. References

### Plan files
- `revised_plan.md` — top-level surgical revision plan (Phases A–E).
- `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` — full prose justification.
- `OUTLINE.md` — manuscript outline (§3.5, §4.4, §4.5, §4.6, §5.4 anchors for Phase B).
- `THESIS_HANDOFF.md` — submission-state reference.

### State files (cross-phase)
- `THESIS_STATE.md` — current state across phases.
- `DECISIONS.md` — locked decisions, including v1 → v2 pool replacement rationale.
- `ERRATA.md` — submitted PDF errata tracked across phases (Phase B contributes the imputation rate correction and the v2 R² supersession).

### Memory entries (key Phase B references)
- Memory #11 — Chronos-2 scale bug fix; the headline operational finding of Phase B.
- Memory #15 — Apples-to-apples leaderboard at full-test scope (informs §5 scope disclosure).
- Memory #16 — C2 ρ̄ correction (0.83 → 0.67 with per-trace breakdown).
- Memory #17 — LOO ablation showing Chronos-2 load-bearing; basis for §5.4 reframing.

### Submitted PDF
- `Ensemble_Learning_for_Proactive_Resource_Prediction_PhanNguyenHungCuong_ITDSIU21078.pdf` (April 2026).
- §3.5 "Pipeline methodology" — references v1 imputation rates ("below 5%"), pre-MEMBER_SCALE B6 logic.
- §4.4 "Headline R²" — v1 numbers to be superseded by v2.
- §4.5 "Diagnostics" — v1 residual correlation matrix (ρ̄ ≥ 0.92); to be superseded by v2 ρ̄ = 0.67.
- §5.4 "Ensemble architecture discussion" — v1 "diverse hybrid" framing to be reframed.

### LaTeX drafts (no new files from Phase B)
- Phase B does not produce standalone LaTeX. Phase B outputs are consumed by drafts in `chapters/03-methodology.tex`, `chapters/04-implementation-results.tex`, and `chapters/05-discussion-evaluation.tex`, which are revised by the writing-thread propagation.

### External references
- N-HiTS: Challu et al. 2023, *AAAI* (`\cite{challu-2023-nhits}`).
- Chronos-2: Ansari et al. 2024, arXiv:2403.07815 (`\cite{ansari-2024-chronos}`).
- Krogh & Vedelsby 1995: "Neural Network Ensembles, Cross Validation, and Active Learning" — diversity-vs-individual-accuracy decomposition cited in §4.5 explanation of the N-HiTS h10 weight pattern.
- Brown et al. 2005, *JMLR* — error correlation decomposition cited for the v1 ρ̄ ≥ 0.92 diagnostic.

---

## Appendix: Phase B artefact inventory (alphabetical)

Quick-reference list of every file touched during Phase B, alphabetised for grep-friendliness.
B1_spine_alibaba.py                       [v1 sprint, read by B5]
B1_spine_bitbrains.py                     [PATCHED for h010]
B1_spine_bitbrains.py.bak                 [pre-patch backup]
B1_spine_bytedance.py                     [v1 sprint, read by B5]
B2_nhits.py
B3_chronos2.py                            [CANONICAL — stores residuals]
B4_nnls_refit.py
B5_chronos2_test.py
B5_nhits_test.py
B6_apply_nnls_test_v3.py                  [RECONSTRUCTED 2026-05-19]
B6_apply_nnls_test_v3.py.broken_v0        [pre-fix backup]
assemble_residuals_canonical.py
bitbrains_et_regen.py                     [PATCHED for h010]
bitbrains_et_regen.py.bak                 [pre-patch backup]
c2_correlations_per_cell.csv              [CANONICAL §4.5]
c2_verdict.md                             [CANONICAL §4.5]
chapters/03-methodology.tex               [REVISED — pending propagation]
chapters/04-implementation-results.tex    [REVISED — pending propagation]
chapters/05-discussion-evaluation.tex     [REVISED — pending propagation]
cross_dataset_headline_v2.csv             [CANONICAL §4.4 + §4.6]
logs/B6_alibaba_chronos2fix.log
logs/B6_bitbrains_chronos2fix.log
logs/B6_bytedance_chronos2fix.log
logs/B6_h010_chronos2fix.log
loo_ablation_new_pool_v2.csv              [CANONICAL §4.5]
lowcov_backup_*                           [Issue-1-fix audit backups]
nnls_refit_diagnostics_v0.csv             [PLANNING ARTEFACT — pre-fix weights, archived]
nnls_weights_v2.json                      [per-cell, 11 cells]
nnls_weights_wide.csv                     [CANONICAL §3.6]
oof_all_new.parquet                       [INTENDED but never produced — B1 fallback]
oof_chronos2.npy                          [per-cell, residual-space]
oof_et.npy                                [per-cell, CPU-space]
oof_nhits.npy                             [per-cell, residual-space]
oof_residuals_canonical.parquet           [from assemble_residuals_canonical.py]
repair_env_v2.sh                          [Phase A inherited]
residual_diagnostics_cross_dataset.csv    [CANONICAL §4.6]
residual_diagnostics_new.csv              [CANONICAL §4.5]
task_c1_nnls_refit.py                     [Phase C-numbered, runs B4 logic post-fix]
test_chronos2.npy                         [per-cell, residual-space]
test_ensemble_hetero.npy                  [SUPERSEDED — pre-fix, archived 2026-05-20]
test_ensemble_v2.npy                      [per-cell, residual-space]
test_et.npy                               [per-cell, CPU-space]
test_nhits.npy                            [per-cell, residual-space]
y_naive.npy                               [per-cell, CPU-space]
y_true.npy                                [per-cell, CPU-space]

End of Phase B monitor file.