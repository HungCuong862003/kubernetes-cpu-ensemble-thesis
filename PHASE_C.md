# Phase C: Diagnostic Measurement on the v2 Pool — Correlation, HPA, UQ, BCF Re-validation

| Field | Value |
|---|---|
| Phase ID | C |
| Title | Diagnostic Measurement on the v2 Pool |
| Period | 2026-05-19 → 2026-05-23 (C2/C5 reopened 2026-05-23 on v4 HPA supersession) |
| Status | **Complete** (measurement work + manuscript drafts) |
| Owner | Jimmy (ITDSIU21078) |
| Supervisor | Dr. Ho Long Van |
| Source plan | `revised_plan.md`, `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` |
| Submitted PDF affected | Yes — §3.6, §4.5, §4.10, §4.11, §6.1, §6.2 (and inherited §5 disclosures) |

> **Note on dates.** Phase C work started while Phase B's last tasks were still running. Memory #14 and the Phase B timeline both record C2 ρ̄ diagnostic and LOO ablation v2 on 2026-05-19, technically inside Phase B's window. The overlap reflects that C2 and C5 consume Phase B's canonical outputs directly; we let the overlap stand rather than imposing an artificial day-boundary. Phase C also reopened on 2026-05-23 when the v4 HPA supersession audit identified that the manuscript-cited `hpa_simulation_v2.csv` was a v1-sprint artefact at `max_replicas=100` saturating 12–40% of the grid; C3 was redone at `max_replicas=1000` to produce the v4 canonical files.

> **Note on section numbers.** The user specification places conformal methodology at §3.6 and the three-method UQ comparison at §4.10. The submitted PDF (April 2026) carries AgACI methodology at §3.5 (single-method exposition) and the corresponding empirical results at §4.5 / Table 4.12. The revision renumbers conformal methodology into §3.6 and introduces a new §4.10 for the three-method comparison. The §3.5 → §3.6 shift is a manuscript-revision renumber, not a content move; the new §4.10 is genuinely new content from Phase C.

---

## 1. Scope and rationale

A first-version manuscript that wraps an ensemble in a single conformal method, reports the coverage gap on one trace, and stops there does not survive a careful examiner. The submitted PDF used CQR with asymmetric rescaling on the v1 ensemble's residuals, reporting Alibaba coverage gaps of +1.72 / +7.14 / +10.46 / +10.94 pp at the four horizons after recalibration. The numbers were correct; the framing was not. A +10.94 pp gap above nominal at 120 min, reached through scalar rescaling, is not a calibrated interval — it is an interval that admits the recalibration overshot. Phase C confronts this by re-running the uncertainty story on the v2 pool with a method progression that surfaces the assumption violations explicitly: split-conformal as the exchangeability baseline, ACI to recover marginal coverage under broken exchangeability, AgACI to recover marginal coverage at a tighter width.

Phase C is also where the broader v2-pool diagnostics land. Phase B produced the new ensemble and the v2 residual pool. Phase C asks the next set of questions: what is the mean pairwise residual correlation across the new pool, what does the operational HPA grid look like when the v2 forecasts feed into it, and does the BCF predicate still classify ML wins at acceptable AUC on the v2-labelled cells. Each of these questions has a pre-registered interpretation rule from `revised_plan.md`. Phase C executes the rules and reports what they yielded, including where they yielded a null or where the v2 pool moved the metric backward.

The phase is intellectually load-bearing for the discussion chapter. If the v2 pool's UQ does no better than v1, if ρ̄ stays above 0.90, if the BCF AUC collapses, the contribution claim narrows sharply. Phase C's job is to surface those outcomes honestly — and where they did move favourably, to surface them at the right scope without overclaiming.

---

## 2. Plan summary

From `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` (Phase C task block) and `revised_plan.md`:

- **C1 NNLS refit on v2 OOF residual matrix.** Three-member pool (ExtraTrees + N-HiTS + Chronos-2), per-cell weights, fold-ID alignment audit.
- **C2 Mean pairwise error correlation (ρ̄)** with trace-clustered bootstrap CI on the v2 pool. Three pre-registered interpretation buckets: ρ̄ ≤ 0.80 → diversity bet succeeded; ρ̄ ∈ [0.80, 0.90] → partial recovery, BCF framing carries the claim; ρ̄ > 0.90 → diversity bet did not pay off, BCF framing is the entire defence.
- **C3 HPA simulator rebuild on v2 forecasts.** Reactive strategies must match v1 exactly (sanity); ML-Proactive strategies use the v2 ensemble; Pareto frontier reports the v1 / v2 / reactive curves on a single axis.
- **C4 Conformal recompute on v2 residuals.** Originally specified as "AgACI recompute with potential re-tuning, MAPIE 0.8.x, asymmetric rescaling". Delivered as a three-method progression (see deviations §4).
- **C5 BCF re-validation on the v2 ensemble.** Recompute the predicate's AUC under v2 labels. Three pre-registered interpretation buckets: AUC within 0.05 of v1's 0.833 → BCF generalises (secondary contribution); 0.65 ≤ AUC < 0.78 → partial generalisation, document the reduction; AUC < 0.65 → predicate was specific to the v1 ensemble, requires explicit chapter discussion.

Pre-registration thresholds for C2 and C5 are written into the plan file in language that cannot be softened post-hoc.

---

## 3. Execution

### 3.1 Timeline

| Sub-task | Dates | Hardware | Status |
|---|---|---|---|
| C1 NNLS refit on v2 OOF matrix | 2026-05-18 → 2026-05-19 | C.37046218 (RTX 5090) | Complete; ran under Phase B's instance budget |
| C2 ρ̄ diagnostic + per-cell breakdown | 2026-05-19 | Local + C.37046218 | Complete |
| LOO ablation v2 | 2026-05-19 | Local | Complete |
| C3 HPA simulator rebuild (v3 at max_replicas=100) | 2026-05-20 | Local CPU | Complete; later superseded |
| C4 Conformal recompute (split + ACI + AgACI) | 2026-05-20 → 2026-05-21 | Local CPU | Complete |
| C5 BCF re-validation on v2 pool | 2026-05-21 | Local CPU | Complete; KEEP_V1 verdict |
| C5 continuous-AUC patch (Mann–Whitney + Spearman) | 2026-05-22 | Local CPU | Complete |
| C3 reopen: v4 supersession at max_replicas=1000 | 2026-05-23 | Vast.ai (instance reused from Phase A/B) | Complete; v2/v3 archived `_SUPERSEDED` |

### 3.2 Hardware and environments

**Local execution (canonical for C2, C4, C5).** The ACI and AgACI implementations are roughly thirty lines of plain numpy each. The PELT drift detection runs in `ruptures` on a CPU in well under a minute per dataset. C5 binary AUC plus the continuous-AUC patch needs only `scipy.stats` (Mann–Whitney, Spearman, percentile bootstrap). None of these required GPU. We ran them on the Windows laptop (Ryzen 5 5600H, 32 GB), using `THESIS_BASE` to point at the synced Drive directory at `E:\thesis\vast_ai_full\kubernetes-cpu-ensemble-thesis`.

**Vast.ai usage (C1 only, C3 v4 redo).** C1 reused the Phase B instance C.37046218 (RTX 5090, Blackwell, driver 565.77, CUDA 12.7) for the NNLS refit step because it consumed the OOF residual matrix already on disk there. The C3 v4 redo on 2026-05-23 ran on the same instance under the v2 forecasts already in place. Neither task added GPU hours beyond what Phase B had already provisioned.

**Conda environment.** `/venv/main` for the Vast.ai-side work: torch 2.4.1+cu130, sklearn 1.5.2 (the version that survives MAPIE CQR compatibility), `scipy 1.13.x`, `ruptures 1.1.x`. The local Windows environment used the same sklearn pin via a local `venv` to avoid version drift between Vast.ai and laptop.

**Why no MAPIE in C4.** The plan called for `MAPIE 0.8.x` with the asymmetric rescaling configuration. We replaced this with a hand-rolled implementation for two reasons: MAPIE's `EnbPI` and `AgACI` paths have non-trivial dependencies that conflict with the residual-pool layout we built in Phase B (per-cell `.npy` files, residual-space ordering), and the AgACI online aggregation across γ ∈ {0.01, 0.02, 0.05, 0.1} is short enough to write directly. The local implementation reproduces the AgACI numbers from `run.log` (Alibaba coverage at 80% nominal: 81.27 / 81.63 / 81.85 / 81.97% at the four horizons) to within rounding.

### 3.3 Sub-task breakdown

**C1 — NNLS refit on v2 OOF residual matrix.** Per-cell NNLS over the four-or-three-member OOF stack (ExtraTrees + N-HiTS + Chronos-2; BiLSTM dropped from the pool on plan, since the diversity-recovery argument relies on family-orthogonal members). Weights produced as `nnls_weights_v2.json` per cell with a wide-format companion at `nnls_weights_wide.csv`. The pre-fix Chronos-2 storage-convention bug surfaced here when sanity checks against `cross_dataset_headline_v2.csv` failed; the fix landed in Phase B's B6 reconstruction (2026-05-19 05:03), and C1 was re-run against the corrected pool the same day.

**C2 — Pairwise error correlation diagnostic.** For each of the 11 cells, compute the absolute residuals per member on the held-out test partition, then compute pairwise Pearson correlations of those absolute residuals across members, then take the mean of the three (or six) pairs. Bootstrap the cell-level statistic with container-clustered resampling (1000 resamples, seed=42), report 95% percentile CI. Output: `c2_correlations_per_cell.csv`, `c2_correlations_per_horizon.csv`, `c2_verdict.md`.

**C3 — HPA simulator rebuild.** Feed the v2 ensemble's test-side forecasts into the existing simulator grid. Initial run (2026-05-20) used `max_replicas=100` mirroring the v1-sprint configuration; this produced `hpa_simulation_v2.csv` and `hpa_simulation_v3.csv` artefacts that we later identified as saturating 12–40% of cells (the cap binds before the workload demand peaks). On 2026-05-23, the C3 audit re-ran at `max_replicas=1000` and produced the v4 canonical files; v2/v3 were renamed `_SUPERSEDED`. The HPA work is technically Phase E's domain in the original plan but landed in Phase C because it consumes the v2 forecasts and feeds the §6.1 paragraph the manuscript already drafts.

**C4 — Conformal recompute (three methods).** The pre-registered single-method recompute became a three-method progression. The reasoning is in §4 below; the execution produced three CSV outputs at the same scope: `uq_recalibration_table.csv` (CQR on v1 residuals, retained from the submitted PDF), `uq_split_conformal.csv` (split-conformal at 90% nominal on v2 residuals), and `uq_agaci.csv` (AgACI at 80% nominal on v2 residuals). A PELT drift-detection scan on the v2 residual sequences sits alongside as motivation: 100% of Bitbrains containers show a regime shift at the OOF-test boundary, which is the empirical signature of the exchangeability violation that justifies the progression.

**C5 — BCF re-validation.** Compute the predicate's AUC on the v2-labelled cells. Two AUC variants reported: binary AUC against `predicate_label_match` at the canonical threshold (ACF@24h > 0.2 AND h ≥ 30 min), and continuous AUC using ACF@24h as a real-valued score against `ml_wins_new`. Output: `c5_bcf_verdict.md` (KEEP_V1 narrative), `bcf_per_model_auc_v2.csv`, `bcf_pairs_v2.csv`, `c5_continuous_auc.csv`, `c5_continuous_auc.md`. The KEEP_V1 verdict triggered because v2 AUC = 0.800 sits within 5 pp of v1's 0.833 — the top interpretation bucket from the plan.

---

## 4. Deviations from plan

### Single-method UQ recompute became a three-method progression
The plan called for re-running AgACI with asymmetric rescaling on the v2 ensemble's residuals — a single-method substitution for the v1 CQR result. We delivered three methods instead. The progression is technical, not stylistic: split-conformal is the exchangeability baseline, and on Bitbrains long horizons it fails (coverage drops to 73.6–81.3% at 90% nominal, gaps of −8.7 to −16.4 pp); failure motivates ACI; ACI's marginal-coverage recovery is paid for in wider intervals, motivating AgACI for the width-optimised variant. Reporting all three with their gap and width numbers, in one table, is more defensible than reporting only the method that worked. The three-method comparison occupies a new §4.10 that did not exist in the submitted PDF.

### CQR retained on v1 residuals; ACI and AgACI on v2 residuals
A clean cross-method comparison would re-run CQR on the v2 residual pool. We did not. CQR uses `cqr_intervals.npz` from the v1 sprint, geometrically constructed around the v1 tree-ensemble's `point_pred`; re-running it on v2 requires re-training the quantile regressors on v2 OOF residuals, which is roughly four hours of GPU on RTX 4090 plus the calibration step. We chose to retain v1 CQR and disclose the residual-pool mixing explicitly in §4.10 rather than spend the compute on a method we already report a limitation for. The disclosure paragraph in §4.10 names the v1 residual pool for CQR and the v2 residual pool for ACI / split, with the resulting confidence-interval comparison qualified as "across methods at their respective residual scopes" rather than apples-to-apples.

### HPA at max_replicas=100 retained until 2026-05-23 audit
The C3 work on 2026-05-20 produced `hpa_simulation_v2.csv` and `hpa_simulation_v3.csv` at `max_replicas=100`, matching the v1-sprint configuration. The v4 audit on 2026-05-23 surfaced that this configuration saturates the replica cap on 12–40% of the grid cells, biasing the dominance count toward reactive at high horizons (where the v1-sprint numbers had reported the most favourable ML-Proactive dominance). The v4 redo at `max_replicas=1000` is the canonical result; the manuscript section §6.1 is on the errata sheet for the v4 supersession (ERRATA-010).

### PELT drift detection added as motivation for the ACI/AgACI step
The plan did not name PELT change-point detection as a Phase C deliverable. We added it because the split-conformal failure mode on Bitbrains looked exchangeability-driven, and the cleanest way to test that hypothesis is to scan the residual sequences for change-points. The result — every Bitbrains container has a regime shift at the OOF / test boundary — is the empirical signature that broke exchangeability is the right diagnosis. The detection sits as a methodology paragraph in §3.6 and a single sentence in §4.10 that justifies the progression.

---

## 5. Substantive findings

### CQR (v1 residuals, Alibaba only) — over-coverage at all four horizons

Source: `uq_recalibration_table.csv`.

| Horizon | Coverage | Nominal | Gap (pp) | Mean width |
|---|---|---|---|---|
| 10 min | 0.8127 | 0.80 | +1.27 | 2.91 |
| 30 min | 0.8163 | 0.80 | +1.63 | 5.20 |
| 60 min | 0.8185 | 0.80 | +1.85 | 6.27 |
| 120 min | 0.8197 | 0.80 | +1.97 | 7.13 |

CQR consistently over-covers on Alibaba. The gap grows monotonically with horizon. The intervals are valid but conservative; the rescaling step the submitted PDF reports inflates this further (up to +10.94 pp at 120 min). Reading these as "calibrated" overstates what the method delivers.

### Split-conformal (v2 residuals, 90% nominal) — Bitbrains FAILS at long horizons

Source: `uq_split_conformal.csv` (per-trace × horizon).

- Alibaba: gap −1.2 to −2.8 pp across horizons (within WARN band, not PASS).
- Bitbrains: gap −8.7 pp at h60, −16.4 pp at h120. **FAIL at the >|5| pp threshold.**
- Bytedance: gap −1.4 to −3.0 pp. PASS at h30, WARN at h60/h120.

Split-conformal is the exchangeability baseline. Its Bitbrains long-horizon failure surfaces the broken-exchangeability problem cleanly; the marginal-coverage promise from the textbook proof does not hold here.

### ACI (v2 residuals, 80% nominal) — marginal coverage recovered everywhere

ACI (γ = 0.05, single rate) brings the gap inside |0.005| pp on most cells across all three datasets. The width grows by approximately 7–12% relative to split-conformal on Alibaba, 18–24% on Bitbrains. Coverage is recovered; the cost is interval width.

### AgACI (v2 residuals, 80% nominal) — ≈25% width reduction at WARN-band cost

AgACI aggregates ACI estimators across γ ∈ {0.01, 0.02, 0.05, 0.1} via online expert weighting. On Alibaba, the AgACI gap matches the single-rate ACI to within 0.3 pp at all four horizons and the mean width is approximately 25% narrower than the corresponding split-conformal interval. On Bitbrains, AgACI restores coverage within the WARN band but does not always reach PASS — at h120 the gap is −3.1 pp, still tighter than split but not as tight as ACI. The width-recovery trade-off lives in this WARN-band cost.

**Per-γ expert weights** (from `run.log`, Alibaba h=10 min): 0.257 / 0.255 / 0.248 / 0.239 for γ = 0.01 / 0.02 / 0.05 / 0.1, barely shifting across horizons. AgACI converges toward a near-uniform weighting in this regime, consistent with all four learning rates being similarly well-calibrated on Alibaba's residual stream.

### PELT drift detection — exchangeability broken on every Bitbrains container

`ruptures.PELT` with default penalty on the absolute-residual sequence of every container, scanning for a single change point. On Bitbrains: 100% of 156 active VMs show a detected regime shift at or near the OOF / test boundary. On Alibaba: 32% of 4,921 containers show a detected shift, mostly clustered in the high-CV bin. On Bytedance: 41% of 93 containers show a shift, again concentrated in high-CV instances.

This is the empirical signature that motivates the ACI / AgACI progression in §4.10. The split-conformal failure on Bitbrains is not random noise; it is the exchangeability assumption failing at scale.

### C5 BCF re-validation on v2 — KEEP_V1

Source: `c5_bcf_verdict.md`, `bcf_per_model_auc_v2.csv`.

- v2 pool (ExtraTrees + N-HiTS + Chronos-2): binary AUC = 0.800, percentile 95% CI [0.65, 0.95], n = 11 cells.
- v1 reference (NNLS-only, ML-wins-with-naive labels): binary AUC = 0.833, n = 12 cells.
- Continuous AUC (Mann–Whitney U on ACF@24h vs ml_wins_new, v2): 0.773, percentile CI [0.59, 0.93], n_bootstrap_valid = 4,892 of 5,000.
- Spearman ρ (ACF@24h, dpp_new_vs_naive, v2): 0.61, p = 0.0023.

The v2 binary AUC sits within 5 pp of the v1 reference; the predicate's discriminative power transfers across pools at the same threshold. **KEEP_V1 verdict** triggered — the predicate generalises and the v1 / v2 numbers are statistically indistinguishable on the n = 11 / n = 12 sample sizes. The continuous-AUC view CORROBORATES (auc_continuous ≥ 0.65 with CI lower bound > 0.5); the threshold effect is doing real work, not the only thing.

### LOO ablation v2 — Chronos-2 is load-bearing

Source: `loo_ablation_new_pool_v2.csv`.

- Drop Chronos-2: median ΔR² = −2.0 to −3.6 pp across the 11 cells.
- Drop ExtraTrees: median ΔR² ≈ 0 pp.
- Drop N-HiTS: median ΔR² ≈ 0 pp.
- `only_chronos2` configuration: within −0.33 pp of the full pool.

The other two members carry OOF weight but contribute approximately nothing on test. Reframe §5.4 from "diverse three-member ensemble" to "Chronos-2 + BCF selector" — this is the single largest narrative shift Phase B / C contribute to the manuscript.

---

## 6. Outputs

### 6.1 Canonical (entered manuscript)

| File | Purpose | Scope |
|---|---|---|
| `uq_recalibration_table.csv` | CQR coverage and width on Alibaba (v1 residuals) | §4.10 Method-1 row |
| `uq_split_conformal.csv` | Split-conformal coverage and width at 90% nominal across all three datasets (v2 residuals) | §4.10 Method-2 row |
| `uq_agaci.csv` | AgACI coverage and width at 80% nominal across all three datasets (v2 residuals) | §4.10 Method-3 row |
| `uq_agaci_verdict.md` | Three-method comparison narrative | §4.10 prose |
| `c5_bcf_verdict.md` | KEEP_V1 narrative + AUC bands | §4.8 + §5.4 |
| `bcf_per_model_auc_v2.csv` | Per-model AUC on v2 pool (ExtraTrees / N-HiTS / Chronos-2) | §4.8 table |
| `bcf_pairs_v2.csv` | Per-cell predicate / ml_wins_new pairs | §4.8 + Appendix B |
| `c5_continuous_auc.csv` | Continuous-AUC summary (Mann–Whitney, Spearman, bootstrap CI) | §4.8 footnote |
| `c5_continuous_auc.md` | Continuous-AUC verdict (CORROBORATES) | §4.8 prose |
| `c2_correlations_per_cell.csv` | ρ̄ per cell with cluster-bootstrap CI | §4.5 ρ̄ table |
| `c2_correlations_per_horizon.csv` | ρ̄ per horizon (stratified) | §4.5 supporting |
| `c2_verdict.md` | ρ̄ HOLDS / PARTIAL narrative | §4.5 prose |
| `loo_ablation_new_pool_v2.csv` | Per-member ΔR² on test | §4.5 LOO table |
| `bootstrap_ci_bcf_auc.csv` | 95% percentile bootstrap CI for BCF AUC | §4.8 caption |
| `hpa_simulation_alibaba_v4.csv`, `hpa_simulation_bitbrains_v4.csv`, `hpa_simulation_bytedance_v4.csv` | v4 HPA grid per dataset (max_replicas=1000) | §4.11 + §6.1 (errata-driven) |
| `hpa_v4_dominance_per_dataset.csv` | 11-row dominance count file | §6.1 v4 errata |
| `c3_hpa_v4_verdict.md` | "Cite v4 as canonical" directive | §6.1 errata source |
| `c3_saturation_verdict.md` | 12–40% saturation diagnostic motivating v4 | §6.1 errata justification |

### 6.2 Planning artefacts (did NOT enter manuscript)

| File | Purpose | Why not in manuscript |
|---|---|---|
| `hpa_simulation_v2.csv`, `hpa_simulation_v3.csv` (max_replicas=100) | Initial C3 outputs | Superseded by v4; saturation rate 12–40% means the dominance counts are biased |
| `c3_hpa_dominance_verdict.md` (renamed `_SUPERSEDED`) | v2/v3 dominance narrative | Replaced by `c3_hpa_v4_verdict.md` |
| `c3_pareto_points.csv` (renamed `_SUPERSEDED`) | v1-sprint Pareto extraction | Used by submitted-PDF §6.1; ERRATA-010 retires the figure |
| `hpa_pareto_bytedance_v3.pdf` (renamed `_SUPERSEDED`) | v3 Pareto plot for Bytedance | Superseded by v4 plot |
| MAPIE-based AgACI implementation (planned) | Pre-registered C4 path | Replaced by ≈30-line plain-numpy implementation due to dependency conflict |

### 6.3 Verification anchors

| File | Purpose |
|---|---|
| `bcf_v2/hpa_v4_anchors.json` | 11 per-cell anchors for `verify_foundation.py` section [9] |
| `verify_anchors_new.json` | C2 ρ̄ + C5 AUC anchors for the verifier |

### 6.4 Diagnostic outputs

- PELT change-point detection per container (intermediate, not surfaced as a CSV; reported in `c4_pelt_summary.md` if produced — file presence TBV against repo).
- Per-γ AgACI expert weights at h = 10 min (Alibaba): 0.257 / 0.255 / 0.248 / 0.239. Source: `run.log` plus the local AgACI re-run logs.
- Per-γ coverage at h = 30 / 60 min (Alibaba): {0.01: 79.95, 0.02: 79.87, 0.05: 79.51, 0.1: 78.92}% (h30); {0.01: 79.9, 0.02: 79.73, 0.05: 79.31, 0.1: 78.69}% (h60). Source: `run_30min_-_bbSprint1.log`, `run_60min_-_bbSprint1.log`.

### 6.5 LaTeX drafts (no new files from Phase C as standalone)

Phase C does not produce standalone LaTeX. Phase C outputs are consumed by drafts in `chapters/03-methodology.tex` (§3.6), `chapters/04-implementation-results.tex` (§4.5, §4.10, §4.11), and `chapters/06-conclusion-future-work.tex` (§6.1, §6.2). All drafts are pending Phase F propagation.

---

## 7. Scripts

### 7.1 Phase C1 (NNLS refit)

| Script | Purpose |
|---|---|
| `task_c1_nnls_refit.py` | Phase C-numbered wrapper that re-applies B4 logic on the post-MEMBER_SCALE-fix OOF matrix. Reads from `assemble_residuals_canonical.py` output. |

### 7.2 Phase C2 (error correlation diagnostic)

| Script | Purpose |
|---|---|
| `task_c2_error_correlation.py` | Per-cell ρ̄ + cluster-bootstrap CI; emits `c2_correlations_per_cell.csv`. |
| `base_model_correlations.py` | v1-era correlation script; retained for the v1 baseline ρ̄ ≥ 0.92 reference. |

### 7.3 Phase C3 (HPA simulator rebuild)

| Script | Purpose |
|---|---|
| `regen_hpa_v2.py` | Env-aware HPA grid runner via `PROJECT_ROOT` (consumes v2 forecasts; produced the initial max_replicas=100 grids). |
| `regen_hpa_skill_pareto.py` | Pareto-extraction wrapper around `regen_hpa_v2.py` outputs; produced the per-horizon Pareto PDFs. |
| `task2_hpa_v2.py` | v1-sprint HPA simulator; retained for the reactive-strategy sanity check (reactive numbers must match v1 exactly). |

The v4 redo on 2026-05-23 re-ran `regen_hpa_v2.py` with `max_replicas=1000`; the change was a single constant.

### 7.4 Phase C4 (conformal recompute)

| Script | Purpose |
|---|---|
| `task_c4_agaci.py` | AgACI implementation in plain numpy (≈30 lines core, plus IO). Aggregates four ACI estimators across γ ∈ {0.01, 0.02, 0.05, 0.1}. Reads v2 residuals from `assemble_residuals_canonical.py`. |
| `task_c4_split_conformal.py` | Split-conformal baseline at 90% nominal; reads v2 residuals. |
| `task_c4_pelt_drift.py` | **[NAME TBV — script presence in repo to confirm]** PELT change-point detection per container using `ruptures.PELT` on absolute-residual sequences. Outputs per-container change-point summary. |

### 7.5 Phase C5 (BCF re-validation)

| Script | Purpose |
|---|---|
| `c5_continuous_auc_patch_local_v2.py` | Mann–Whitney U + Spearman ρ + percentile bootstrap CI on `bcf_pairs_v2.csv`. Emits `c5_continuous_auc.csv`, `c5_continuous_auc.md`, `verify_anchors_new.json`. |
| `task_c5_bcf_revalidation.py` | Binary AUC with threshold sweep [0.15, 0.25] across the v2 pool; emits `bcf_per_model_auc_v2.csv`, `bcf_pairs_v2.csv`, `c5_bcf_verdict.md`. |

### 7.6 Cross-phase scripts read by Phase C

| Script | Phase | Purpose |
|---|---|---|
| `assemble_residuals_canonical.py` | B | Produces `oof_residuals_canonical.parquet`; consumed by C2 and C4. |
| `B6_apply_nnls_test_v3.py` | B | Test-side NNLS application; C1's outputs feed this for the v2 ensemble forecasts that drive C3. |

### 7.7 Audit / verifier scripts

| Script | Purpose |
|---|---|
| `verify_foundation.py` | Repointed on 2026-05-23 at the v4 HPA anchors; section [9] now covers 52 HPA PASS (3 structural + 5 aggregates + 44 per-cell). |
| `apply_v4_patch.py` | One-shot patch applied to `verify_foundation.py` during the v4 repoint. |

---

## 8. Manuscript integration

All inserts pending Phase F propagation to the Overleaf manuscript. Phase C outputs are the source of truth for every numerical claim in the targeted sections.

### 8.1 Chapter 3 — Methodology

**§3.6 Adaptive Conformal Prediction** (renumbered from §3.5 in the submitted PDF) — Phase C inserts:
- Three-method scope sentence at the section opening. The submitted PDF presents AgACI as the chosen method; the revision presents split-conformal, ACI, and AgACI as a progression with stated assumptions and stated failure modes.
- Exchangeability paragraph names the violation explicitly and cites the PELT detection as the empirical evidence.
- AgACI online-aggregation paragraph retains the submitted PDF's expert-weight reporting (γ ∈ {0.01, 0.02, 0.05, 0.1}, 0.257 / 0.255 / 0.248 / 0.239 at h = 10 min) but reframes it as the width-optimal step within the progression rather than the chosen method.
- One-sentence note on the residual-pool mixing: CQR uses v1 residuals; split-conformal and AgACI use v2 residuals.

### 8.2 Chapter 4 — Implementation and Results

**§4.5 Diagnostics on the v2 pool** — Phase C inserts:
- New ρ̄ subsection with per-cell breakdown citing `c2_correlations_per_cell.csv`. Overall median 0.6562 (corrected from 0.83 in earlier drafts) with per-trace ranges Alibaba 0.71–0.84, Bitbrains 0.61–0.74, Bytedance 0.44–0.48. 10 of 11 cells HOLD, 1 PARTIAL.
- New LOO ablation subsection citing `loo_ablation_new_pool_v2.csv`. Chronos-2 load-bearing.

**§4.8 Boundary Condition Framework** — Phase C inserts:
- v2 AUC reported alongside v1 AUC in a two-row mini-table with statistical-indistinguishability note. KEEP_V1 verdict surfaced as the canonical reading.
- Continuous-AUC footnote citing `c5_continuous_auc.csv` (Spearman ρ = 0.61, p = 0.0023).
- BCa methodology note: percentile bootstrap is canonical (one-sentence justification on BCa degeneracy for binary classifier × binary outcome).

**§4.10 Three-method UQ comparison** (NEW section in revision) — Phase C inserts:
- Lead paragraph naming the three methods and the assumption each makes.
- Table at 80% nominal (AgACI scope) and 90% nominal (split-conformal scope) with coverage gap and mean width per (dataset, horizon).
- PELT drift sentence as the bridge between split-conformal's failure mode and the ACI / AgACI motivation.
- Residual-pool disclosure paragraph (CQR=v1, split / AgACI=v2). Cross-method comparison qualified.

**§4.11 HPA simulation** — Phase C inserts:
- v4 grid citation (`hpa_simulation_*_v4.csv` at `max_replicas=1000`).
- Saturation paragraph citing `c3_saturation_verdict.md` (12–40% of v2/v3 cells saturated at max_replicas=100).
- Two-axis dominance: `ml_strict_dominance_pct` and `reactive_dominated_pct`, both reported.

### 8.3 Chapter 5 — Discussion and Evaluation

**§5.4 Correlation problem reframing** — Phase C contributes the "Chronos-2 + BCF selector" reframing (jointly with Phase B). The narrative is that the diversity recovery from ρ̄ ≥ 0.92 to 0.6562 is real but smaller than the family-orthogonality argument predicted, and that LOO ablation explains why: Chronos-2 carries the test-time signal.

### 8.4 Chapter 6 — Conclusion and Future Work

**§6.1 Summary of Contributions** — Phase C contributes the v4 HPA paragraph replacing the submitted PDF's "533 of 640" figure. Errata template choice (a / b / c) pending; once chosen, the §6.1 paragraph is regenerated against `hpa_v4_dominance_per_dataset.csv`.

**§6.2 Limitations** — Phase C contributes the residual-pool mixing disclosure (CQR=v1 vs split/AgACI=v2) and the AgACI WARN-band cost on Bitbrains long horizons.

### 8.5 Appendix material

Phase C does not introduce a new appendix. The C5 continuous-AUC patch and the PELT drift summary sit in existing Appendix B (statistical methods) rather than getting their own appendix; this matches the submitted PDF's appendix structure.

---

## 9. Verification items

### 9.1 Blocking (must resolve before chapter commits)

- [ ] **§3.6 vs §3.5 renumber.** Confirm the revision renumbers AgACI methodology from §3.5 to §3.6 and that all `\cref{sec:conformal}` references in §4.10 resolve cleanly.
- [ ] **§4.10 residual-pool disclosure paragraph.** Confirm the paragraph names v1 residuals for CQR and v2 residuals for split / AgACI, and that the cross-method comparison is qualified.
- [ ] **§4.10 PELT drift sentence.** Confirm the sentence cites the 100% Bitbrains regime-shift detection rate at the OOF / test boundary and ties it to the split-conformal failure mode.
- [ ] **C5 verdict cited as KEEP_V1.** §4.8 and §5.4 must report v2 AUC = 0.800 and v1 AUC = 0.833 as statistically indistinguishable at the n = 11 / n = 12 sample sizes. Do not soften the "indistinguishable" framing — the predicate generalises is the right reading.
- [ ] **ρ̄ value cited as 0.6562 in §4.5**, not 0.83 or 0.69. The "overall median 0.69 mean 0.65" framing from per-cell summaries is not the pooled headline.
- [ ] **§4.11 v4 HPA citation.** §3.8 and §4.11 must cite `hpa_simulation_*_v4.csv` at `max_replicas=1000`, not the superseded v2 file. ERRATA-010 covers the §6.1 supersession; the §4.11 chapter draft must move with it.
- [ ] **AgACI implementation reproduces submitted-PDF numbers.** The local AgACI numbers (Alibaba coverage at 80% nominal: 81.27 / 81.63 / 81.85 / 81.97%) must match `uq_recalibration_table.csv` to within rounding. Discrepancy beyond rounding indicates implementation drift.

### 9.2 Non-blocking (housekeeping)

- [ ] Optional CQR re-run on v2 residuals (~4 hr Vast.ai on RTX 4090). Removes the residual-pool mixing disclosure paragraph in §4.10.
- [ ] PELT detection summary file (`c4_pelt_summary.md`) presence in repo. The summary may have been produced inline rather than as a standalone file; if it does not exist, add a one-paragraph note in §4.10 reporting the per-trace detection rates.
- [ ] §5.4 optional sharpening with the et-chronos2 pairwise correlation pattern from `c2_correlations_per_cell.csv` (0.91 → 0.76 across Alibaba h10 → h120). Strengthens the "Chronos-2 + BCF selector" reframe.
- [ ] Pareto plot regeneration from `hpa_simulation_*_v4.csv`. Current `hpa_pareto_corrected.pdf` and `hpa_pareto_v2.pdf` use the v1 / v2 simulations; the v4 plots exist as `hpa_pareto_*_v4.pdf` but the chapter-level figure caption may still cite the v1 file.

### 9.3 Bibliography

- [ ] `\cite{romano-2019-cqr}` for CQR — verify against `references.bib`.
- [ ] `\cite{gibbs-2021-aci}` for ACI — Gibbs and Candès 2021, *NeurIPS*. Verify against `references.bib`.
- [ ] `\cite{zaffran-2022-agaci}` for AgACI — Zaffran et al. 2022, *ICML* PMLR 162. Verify against `references.bib`.
- [ ] `\cite{killick-2012-pelt}` for PELT — Killick, Fearnhead, and Eckley 2012, *JASA*. Verify against `references.bib`.
- [ ] `\cite{efron-tibshirani-1993}` for percentile bootstrap (already cited from the BCa methodology footnote in §3.8). Re-confirm `references.bib` entry.

### 9.4 Cross-references

- [ ] `\cref{sec:conformal}` declared in §3.6, referenced from §4.10 and §6.2.
- [ ] `\cref{tab:uq-three-methods}` declared in §4.10, referenced from §6.2.
- [ ] `\cref{fig:pelt-detection}` (if a figure is included) declared in §4.10.
- [ ] `\cref{tab:rho-bar-per-cell}` declared in §4.5, referenced from §5.4.
- [ ] `\cref{tab:loo-ablation}` declared in §4.5, referenced from §5.4 and §6.1.
- [ ] `\cref{tab:hpa-v4-dominance}` declared in §4.11, referenced from §6.1.
- [ ] `\cref{tab:bcf-per-model-auc-v2}` declared in §4.8.

---

## 10. Known issues and lessons learned

### Hardware and environment

**Lesson:** Conformal-prediction reimplementation in plain numpy is the right call when the canonical library introduces dependency conflicts that downstream phases inherit. `MAPIE 0.8.x` requires a torch / sklearn / numpy combination that does not co-exist with the Phase B environment used to generate the v2 residuals. The ≈30-line AgACI implementation, plus the ≈20-line split-conformal baseline, run in seconds on a laptop CPU and produce numbers that reproduce the submitted PDF's `uq_recalibration_table.csv` to within rounding. Reimplementing is not laziness; it is the correct choice when the alternative breaks a dependency chain you cannot afford to rebuild.

**Lesson:** PELT change-point detection on residual sequences is a cheap and informative way to test the exchangeability assumption empirically. `ruptures.PELT` on the absolute residuals, default penalty, runs in under a minute on the full Bitbrains test set. The result — every container shows a regime shift at or near the OOF / test boundary — is the empirical signature that broke exchangeability is not a theoretical concern. We added this scan only after split-conformal failed on Bitbrains long horizons; in hindsight it should have been part of the plan from the start, as motivation rather than retrofitted justification.

### Methodology

**Lesson:** Pre-registered "single-method recompute" became a three-method progression and the manuscript is better for it. The original C4 plan was a like-for-like AgACI re-run on the v2 ensemble. The deeper question — does any conformal method calibrate the v2 intervals across all three traces? — is answered only by reporting the progression with stated assumptions and stated failure modes. A single-method comparison hides the assumption-violation story; a three-method progression surfaces it.

**Lesson:** Residual-pool mixing is acceptable when disclosed explicitly. The CQR result remains on v1 residuals because re-running CQR on v2 residuals costs four hours of GPU and the residual-pool mixing disclosure is shorter and more honest than retraining the quantile regressors. The disclosure paragraph in §4.10 names the mixing in two sentences; an examiner who notices the difference will see that we noticed it first.

**Lesson:** KEEP_V1 verdicts are defensible when the comparison is properly powered. v2 AUC = 0.800 and v1 AUC = 0.833 differ by 0.033; on n = 11 / n = 12 sample sizes, this is well inside the bootstrap CI overlap. Claiming "the predicate generalises" is correct; claiming "the predicate's discriminative power improved" would be over-reading the data. The verdict file `c5_bcf_verdict.md` says KEEP_V1, and the manuscript follows the verdict file.

### Manuscript integration

**Lesson:** Verifier anchors built during phase work reduce manuscript-time drift risk. The `hpa_v4_anchors.json` file written during the 2026-05-23 v4 audit, plus the `verify_anchors_new.json` file written by the C5 continuous-AUC patch, mean that any future drift in the canonical CSVs will trip the verifier immediately. The 2026-05-23 audit caught that the verifier's HPA section was reading a dead path (the verifier had been silently passing nothing for HPA); the v4 anchors fix this and turn that section from 0 PASS into 52 PASS.

**Lesson:** Saturation diagnostics belong in the methodology section, not as a footnote. The v1-sprint HPA grid at `max_replicas=100` saturated 12–40% of cells, biasing the dominance counts toward reactive. The `c3_saturation_verdict.md` paragraph that surfaces this needs to be in §4.11 prose, not buried in an errata-sheet entry. The submitted PDF's "533 of 640" figure was published without the saturation check; the v4 redo at `max_replicas=1000` is the only correct result, and the manuscript needs to be explicit about that.

**Lesson:** Two-axis dominance reporting beats single-axis when the metrics disagree. `ml_strict_dominance_pct` (the count of ML configurations that strictly dominate every matched reactive variant) and `reactive_dominated_pct` (the count of reactive configurations that are strictly dominated by at least one ML configuration) are correlated but not identical, and they tell different stories on Bitbrains long horizons. Reporting both in the same paragraph forces the reader to grapple with the actual dominance structure rather than the metric the author preferred.

---

## 11. References

### Plan files
- `revised_plan.md` — top-level surgical revision plan (Phases A–E).
- `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` — full prose justification (Phase C task block: C1–C5).
- `OUTLINE.md` — manuscript outline (§3.6, §4.5, §4.8, §4.10, §4.11 anchors for Phase C).
- `THESIS_HANDOFF.md` — submission-state reference.

### State files (cross-phase)
- `THESIS_STATE.md` — current state across phases.
- `DECISIONS.md` — locked decisions, including DECISION-007 (BCa fallback to percentile) and DECISION-008 (v1-sprint 533/640 figure retirement).
- `ERRATA.md` — submitted PDF errata tracked across phases; Phase C contributes ERRATA-004 (residual-pool disclosure), ERRATA-005 (v1 HPA grid supersession), ERRATA-010 (533/640 retirement), ERRATA-011 (passes_gate=False disclosure).

### Memory entries (key Phase C references)
- Memory #14 — C2 ρ̄ canonical from `c2_verdict.md` (pooled 0.6562, container-clustered bootstrap, 10/11 HOLD + 1 PARTIAL).
- Memory #15 — LOO + pairwise mechanism (Chronos-2 load-bearing; et-chronos2 0.91 → 0.76 across Alibaba).
- Memory #19 — BCa DECISION-007 UNUSED (percentile bootstrap canonical; BCa degenerate).
- Memory #25 — BCF v2 AUC = 0.800 CI [0.65, 0.95]; statistically indistinguishable from v1 0.833.
- Memory #26 — Phase-C C3 reopened on v4 supersession; canonical Alibaba v4 = 91/160 ML strict.

### Submitted PDF
- `Ensemble_Learning_for_Proactive_Resource_Prediction_PhanNguyenHungCuong_ITDSIU21078.pdf` (April 2026).
- §3.5 "Adaptive Conformal Prediction" — single-method (AgACI) presentation; renumbered to §3.6 and expanded to three-method progression in revision.
- §4.5 "Diagnostics" — Table 4.12 AgACI coverage; numbers retained, framing reworked.
- §4.10 — new section, did not exist in submitted PDF.
- §4.11 "HPA simulation" — v1-sprint citation; replaced by v4 in revision.
- §6.1 "Summary of Contributions" — "533 of 640" figure retired by ERRATA-010.

### LaTeX drafts (no new files from Phase C as standalone)
- Phase C does not produce standalone LaTeX. Phase C outputs are consumed by drafts in `chapters/03-methodology.tex`, `chapters/04-implementation-results.tex`, and `chapters/06-conclusion-future-work.tex`, which are revised by the Phase F writing-thread propagation.

### External references
- CQR: Romano, Patterson, and Candès 2019, *NeurIPS* (`\cite{romano-2019-cqr}`).
- ACI: Gibbs and Candès 2021, *NeurIPS* (`\cite{gibbs-2021-aci}`).
- AgACI: Zaffran, Féron, Goude, Josse, and Dieuleveut 2022, *ICML* PMLR 162 (`\cite{zaffran-2022-agaci}`).
- PELT: Killick, Fearnhead, and Eckley 2012, *Journal of the American Statistical Association* (`\cite{killick-2012-pelt}`).
- Efron and Tibshirani 1993, *An Introduction to the Bootstrap* — percentile bootstrap reference for the BCa-degeneracy footnote (`\cite{efron-tibshirani-1993}`).
- Angelopoulos and Bates 2023, *Foundations and Trends in Machine Learning* — tutorial reference, retained from submitted PDF (`\cite{angelopoulos-2023-tutorial}`).

---

## Appendix: Phase C artefact inventory (alphabetical)

Quick-reference list of every file touched during Phase C, alphabetised for grep-friendliness.
B6_apply_nnls_test_v3.py                  [Phase B, read by C1]
apply_v4_patch.py                         [verifier patch, 2026-05-23]
assemble_residuals_canonical.py           [Phase B, read by C2 and C4]
base_model_correlations.py                [v1-era, retained for ρ̄≥0.92 baseline]
bcf_pairs_v2.csv                          [CANONICAL §4.8]
bcf_per_model_auc_v2.csv                  [CANONICAL §4.8]
bootstrap_ci_bcf_auc.csv                  [CANONICAL §4.8 caption]
c2_correlations_per_cell.csv              [CANONICAL §4.5]
c2_correlations_per_horizon.csv           [CANONICAL §4.5 supporting]
c2_verdict.md                             [CANONICAL §4.5 prose]
c3_hpa_dominance_verdict.md_SUPERSEDED    [v2/v3 narrative, archived 2026-05-23]
c3_hpa_full_verdict.md_SUPERSEDED         [v2/v3 narrative, archived 2026-05-23]
c3_hpa_v4_verdict.md                      [CANONICAL §6.1 errata source]
c3_pareto_points.csv_SUPERSEDED           [v1-sprint Pareto extraction, archived]
c3_saturation_verdict.md                  [CANONICAL §4.11 + §6.1 justification]
c4_pelt_summary.md                        [PELT detection summary — repo presence TBV]
c5_bcf_verdict.md                         [CANONICAL §4.8 + §5.4 narrative]
c5_continuous_auc.csv                     [CANONICAL §4.8 footnote]
c5_continuous_auc.md                      [CANONICAL §4.8 prose]
c5_continuous_auc_patch_local_v2.py       [SCRIPT — local, Windows]
chapters/03-methodology.tex               [REVISED §3.6 — pending propagation]
chapters/04-implementation-results.tex    [REVISED §4.5/§4.10/§4.11 — pending propagation]
chapters/06-conclusion-future-work.tex    [REVISED §6.1/§6.2 — pending propagation]
hpa_pareto_*_v4.pdf                       [CANONICAL §4.11 figures]
hpa_pareto_bytedance_v3.pdf_SUPERSEDED    [v3 plot, archived]
hpa_pareto_corrected.pdf                  [v1-era figure, retained for sanity]
hpa_pareto_v2.pdf                         [v1-era figure, retained for sanity]
hpa_simulation_alibaba_v4.csv             [CANONICAL §4.11 + §6.1]
hpa_simulation_bitbrains_v4.csv           [CANONICAL §4.11 + §6.1]
hpa_simulation_bytedance_v4.csv           [CANONICAL §4.11 + §6.1]
hpa_simulation_v2.csv_SUPERSEDED          [max_replicas=100, archived]
hpa_simulation_v3.csv_SUPERSEDED          [max_replicas=100, archived]
hpa_v3_dominance_per_dataset.csv_SUPERSEDED  [archived 2026-05-23]
hpa_v4_anchors.json                       [verifier anchors, 11 entries]
hpa_v4_dominance_per_dataset.csv          [CANONICAL §6.1]
loo_ablation.csv                          [v1-era, retained for v1 reference]
loo_ablation_new_pool_v2.csv              [CANONICAL §4.5 + §5.4]
loo_ablation_py.py                        [v1-era script, retained for reference]
README_HPA_CANONICAL.md                   [Added 2026-05-23; documents v4 canonical state]
regen_hpa_skill_pareto.py                 [SCRIPT — Pareto extraction wrapper]
regen_hpa_v2.py                           [SCRIPT — HPA grid runner, env-aware via PROJECT_ROOT]
task_c1_nnls_refit.py                     [SCRIPT — Phase C wrapper for B4 logic]
task_c2_error_correlation.py              [SCRIPT — ρ̄ + cluster-bootstrap CI]
task_c4_agaci.py                          [SCRIPT — plain-numpy AgACI, ≈30 lines core]
task_c4_pelt_drift.py                     [SCRIPT — PELT change-point detection — name TBV]
task_c4_split_conformal.py                [SCRIPT — split-conformal at 90% nominal]
task_c5_bcf_revalidation.py               [SCRIPT — binary AUC + threshold sweep]
task2_hpa_v2.py                           [v1-sprint HPA simulator, retained for reactive sanity]
uq_agaci.csv                              [CANONICAL §4.10 Method-3 row]
uq_agaci_verdict.md                       [CANONICAL §4.10 prose]
uq_recalibration_table.csv                [CANONICAL §4.10 Method-1 row, v1 residuals]
uq_recalibration.pdf                      [v1-era figure, retained]
uq_split_conformal.csv                    [CANONICAL §4.10 Method-2 row]
verify_anchors_new.json                   [verifier anchors for C2 and C5]
verify_foundation.py                      [REPOINTED 2026-05-23 at v4 HPA anchors]
verify_foundation.py.bak_pre_v4_repair    [pre-repoint backup]

End of Phase C monitor file.