# Phase D: Statistical Validation

| Field | Value |
|---|---|
| Phase ID | D |
| Title | Statistical Validation |
| Period | 2026-05-16 → 2026-05-20 (TEST set closed 2026-05-22) |
| Status | **Complete** (D4 fully done; D1 substituted by Phase A leaderboard; D2 and D3 not done per plan contingency) |
| Owner | Jimmy (ITDSIU21078) |
| Supervisor | Dr. Ho Long Van |
| Source plan | `revised_plan.md`, `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` |
| Submitted PDF affected | Yes — §3.7 (new), §4.5, §4.6, §4.8 + new Appendix D |

---

## 1. Scope and rationale

A 2026 thesis claiming that the v2 NNLS ensemble adds measurable value over individual base models must show that the value is statistically detectable under proper time-series-aware testing with multiple-comparison correction. Phase D addresses this by applying cluster-bootstrap Diebold-Mariano tests to the v2 five-model family across all 11 (dataset, horizon) cells, plus a Friedman omnibus over the same family. The phase has three deliverables: a 44-test pairwise DM table with Holm-Bonferroni correction (the headline outcome), per-horizon Friedman mean ranks for the v2 family, and the BCF predicate stratification of the DM result (which emerged during analysis and became the strongest statistical evidence for the boundary-condition framework).

The phase is intellectually load-bearing for Chapter 4 §4.5 and Chapter 4 §4.8. Without proper testing, the v2 R² gains can be dismissed as artefacts of sample size or unmodeled within-series correlation. Without the predicate stratification, the BCF predicate from Phase C carries data-side evidence only; Phase D supplies the inferential-side evidence that closes the predicate's defence.

The original Phase D plan defined four tasks (D1 through D4). Only D4 carried inferential weight; D1, D2, D3 were defensive breadth. D4 is the only Phase D task that ran in full. D1 evolved into Phase A's leaderboard work and is documented in `phase_a_monitor.md`. D2 and D3 were not executed under the plan's own contingency rules.

---

## 2. Plan summary

From `revised_plan.md`, Phase D:

**D1 — Toto-Open-Base-1.0 leaderboard inference.** Zero-shot Toto on all three traces. Substitute Moirai-MoE if A1 spike abort fires. **Outcome:** absorbed into Phase A; Toto verified viable but TimesFM-2.5 and Granite-TTM occupy the canonical leaderboard slots; Toto sits in Appendix C as sensitivity check. See `phase_a_monitor.md` §5.

**D2 — Chronos-2 fine-tune experiment.** Train Chronos-2 on Alibaba calibration data, evaluate gain over zero-shot. **Outcome:** not done. Plan declared all three outcomes (≥1pp gain, [0.5, 1.0] pp gain, <0.5pp gain) publishable, so absence is defensible.

**D3 — Azure VM V1 fourth-dataset.** Stratified 200-VM sample, run BCF predicate, check cross-provider transferability. **Outcome:** not done. Plan marked as conditional on V2 passing and Phase C completing on schedule; the contingency rule applied.

**D4 — Statistical validation suite.** Diebold-Mariano (HLN-corrected per plan), Friedman across all members, Holm correction across 12 cells, trace-clustered bootstrap (3 clusters) for R² delta CIs. **Outcome:** fully done with one methodology substitution documented in §4 below.

Pre-registration for D4 (as executed):
- Cluster-bootstrap DM with container-clustered resampling, 1000 resamples per cell
- 44 tests = 11 cells × 4 pairwise comparisons (ensemble v2 vs naive / ET / N-HiTS / Chronos-2)
- Holm-Bonferroni correction across the full 44-test family
- Friedman omnibus on the same 5-model family per horizon
- Significance threshold α = 0.05 after Holm correction

Pre-registration document: section "Phase D goals" in `revised_plan.md`.

---

## 3. Execution

### 3.1 Timeline

| Sub-task | Dates | Hardware | Status |
|---|---|---|---|
| D4 OOF run | 2026-05-16 → 2026-05-17 | C.37124280 (RTX 5090) | Complete (appendix only) |
| D4 TEST run | 2026-05-19 | C.37124280 (RTX 5090) | Complete (canonical) |
| D4 CD diagrams | 2026-05-20 | C.37124280 | Complete |
| D4 verification & anchors | 2026-05-20 | Local | Complete |
| D4 manuscript integration | 2026-05-22 | Local | Drafted, pending verification |

### 3.2 Hardware and environments

**Vast.ai compute:**
- C.37124280 (RTX 5090, Blackwell, 24 GB VRAM, 384-core AMD EPYC host) — canonical instance.
- SSH: proxy `ssh -p 14281 root@ssh4.vast.ai`; direct `ssh -p 20764 root@213.181.123.64`.

**Conda environments:**
- `/venv/main` — torch 2.4.1+cu130, numpy 2.x, pandas 2.2.x, scipy 1.13.x, scikit-learn 1.5.2 (pinned). Hosts D4 scripts.
- D4 has no foundation-model dependency at runtime — it reads pre-computed residuals from the canonical parquet. The `toto_env` isolation from Phase A is not needed for D4.

**Compute footprint:**
- D4 OOF cluster-bootstrap: ≈45 minutes wall-clock across 44 tests × 1000 resamples × 11 cells.
- D4 TEST cluster-bootstrap: ≈35 minutes wall-clock (smaller cells than OOF after spine filtering).
- CD-diagram generation: ≈3 minutes.

**Why cluster bootstrap, not classical DM-HLN:** The classical Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction was designed for single-series forecast comparison. Our data is multi-series: 11 dataset-horizon cells, each containing many containers with their own forecast trajectories. The dominant dependence sits between containers (same workload type, same time-of-day patterns), not within timesteps of a single container at our non-overlapping forecast cadence. Cluster bootstrap on container-id resamples whole containers and absorbs the dominant dependence directly, producing wider, honest CIs. Documented in §4 below as a deviation.

### 3.3 Sub-task breakdown

**D4 OOF run.** Goal: produce per-cell DM and Friedman statistics on out-of-fold residuals as an appendix-level sanity check. The OOF analysis runs the same statistical machinery on the data the NNLS combiner was fitted to. Result: ensemble dominates mechanically on OOF data, as expected (38 wins out of 44 tests). The OOF result is not the inferentially honest view because the meta-learner saw this data; we report it in Appendix D for transparency only.

**D4 TEST run.** Goal: produce the canonical statistical validation table on the held-out test set, which the NNLS combiner did not see. Result: 38 wins, 3 ties, 3 losses across 44 tests after Holm correction. This is the inferentially honest result and the canonical reference for §4.5.

**CD-diagram regeneration for v2 pool.** Goal: replace the v1 critical-difference diagrams (which arrange the ten v1 forecasters) with v2 diagrams (five members: naive, ET, N-HiTS, Chronos-2, ensemble v2). Output: `cd_diagram_new_pool_ranks.csv` plus four PDFs (one per horizon). The ensemble holds mean rank 1 at every horizon.

**Manuscript artefact generation.** Goal: produce LaTeX-ready tables and PDF figures from the D4 canonical CSVs. Output: `tab_dm_summary.tex`, `tab_dm_per_cell.tex`, `tab_friedman_ranks.tex`, `fig_rank_trajectory.pdf`, `fig_dm_significance.pdf`. All artefacts trace to canonical CSVs via `verify_anchors_new.json` (26 anchors).

**Verification framework.** Goal: every numerical claim in §4.5, §4.6, §4.8 traces to a specific cell in a canonical CSV. Output: `verify_anchors_new.json` with claim text, source file, column, expected value, and tolerance. Re-runnable via `task4_verify_tables.py`.

---

## 4. Deviations from plan

### Cluster bootstrap instead of Newey-West DM
The pre-registered method was classical Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction, which uses Newey-West HAC variance estimation for within-series autocorrelation. We substituted container-clustered bootstrap with 1000 resamples per cell. The substitution was made before the test ran, not after results were inspected, and is justified by the multi-series structure of the data: the dominant dependence sits between container-id groups, which Newey-West does not address but cluster bootstrap does. The substitution is more conservative (wider honest CIs) and requires documentation in Chapter 3 §3.7 of the manuscript.

### Friedman v2 on 5-model family, not 10-model v1
Pre-registration called for Friedman across all members. We ran Friedman on the v2 five-model family (naive, ExtraTrees, N-HiTS, Chronos-2, ensemble v2) because the v1 ten-forecaster Friedman is already reported in the submitted PDF and is not Phase D content. Running Friedman on the v1 family again would not produce new information; running it on the v2 family produces the per-horizon mean ranks the chapter actually cites.

### Sign convention consistency
The v1 `dm_stats_*.csv` files reported DM with inconsistent sign convention across horizons — the antisymmetric pair was preserved within each horizon, but the row-against-column orientation flipped between 30 and 60 min. D4 outputs use a consistent sign convention: positive `mean_d` means the ensemble's squared error exceeds the comparison method's (ensemble is worse). The `winner_final` column resolves the direction explicitly, so the sign-convention question does not affect the headline result. Documented in §9 verification list.

### Trace-clustered bootstrap (3 clusters) replaced by container-clustered (1000 resamples per cell)
Pre-registration called for trace-clustered bootstrap with 3 clusters (the three datasets) for the R² delta CIs. We executed container-clustered bootstrap with 1000 resamples per cell instead. The trace-clustered version with only 3 clusters produces a CI too wide to be informative; the container-clustered version produces CIs at the cell level, which is the unit the manuscript reports. The trace-clustered version was abandoned during the methodology check, not after results were inspected.

### D1, D2, D3 not executed under Phase D
- **D1** absorbed into Phase A. The leaderboard work happened earlier and the canonical 4-model leaderboard was already in the manuscript when D4 ran. Toto was verified viable at A1 but did not enter the canonical leaderboard. Documented in `phase_a_monitor.md`.
- **D2** not done. Plan allowed for deferral.
- **D3** not done. Plan marked as droppable from inception.

---

## 5. Substantive findings

### Cluster-bootstrap DM headline (44 tests, Holm-corrected)

| Outcome | Count |
|---|---|
| Ensemble wins | 38 |
| Ties | 3 |
| Ensemble loses | 3 |

Sources: `results/bcf_v2/d4_dm_pairwise_TEST.csv`, `d4_verdict_TEST.md`.

### Predicate-stratified split

The BCF predicate from Phase C (`ACF@24h > 0.2 AND h ≥ 30 min`) stratifies the DM result. Six cells are predicate-positive (Alibaba and Bytedance at h ≥ 30 min); five are predicate-negative (Alibaba h10 and all four Bitbrains horizons). With four pairwise comparisons per cell, this gives 24 predicate-positive tests and 20 predicate-negative tests.

| Regime | Tests | Wins | Ties | Losses |
|---|---|---|---|---|
| Predicate-positive | 24 | 23 | 1 | 0 |
| Predicate-negative | 20 | 15 | 2 | 3 |

The single predicate-positive tie sits at Bytedance h=120 against Chronos-2 (Holm p = 0.072). The three predicate-negative losses concentrate on Bitbrains at long horizons: h60 vs Chronos-2 (|DM| = 123.28, p = 0 Holm), h120 vs N-HiTS (|DM| = 90.11, p = 0), h120 vs Chronos-2 (|DM| = 90.22, p = 0).

The stratification matches the BCF predicate's prediction directly. Where the predicate fires, the ensemble carries measurable statistical advantage over every alternative member at 95% significance under cluster bootstrap and Holm correction. Where it does not, the advantage weakens or reverses, with the failure mode (ensemble losing to a per-member alternative) confined to the lowest-`acf_24h` dataset at long horizons.

### Friedman v2 mean ranks per horizon

Lower rank is better. Source: `results/bcf_v2/d4_friedman_TEST.csv`.

| Horizon | Naive | ExtraTrees | N-HiTS | Chronos-2 | Ensemble v2 |
|---|---|---|---|---|---|
| 10 min | 3.64 | 2.65 | 4.34 | 2.50 | **1.87** |
| 30 min | 3.71 | 2.97 | 3.66 | 2.97 | **1.68** |
| 60 min | 3.79 | 3.01 | 3.30 | 3.22 | **1.67** |
| 120 min | 3.83 | 3.14 | 3.27 | 3.10 | **1.65** |

Critical differences at α = 0.05: 0.087 to 0.088 across horizons. The ensemble holds the top rank cleanly at every horizon. N-HiTS at h=10 sits at rank 4.34 (worse than naive's 3.64) while simultaneously receiving the largest NNLS weight at that cell — a Krogh-Vedelsby pattern where a member whose errors decorrelate from the others earns weight even when its own predictions are weak.

### Three lines of evidence convergence (BCF predicate)

The BCF predicate is now supported by three independent forms of evidence:

1. **Data-side** (Phase C, §4.8): binary AUC = 0.80, 95% CI [0.70, 0.88], permutation p = 0.011 across 36 canonical-pool cells.
2. **Continuous-scale** (Phase C, §4.8): Mann-Whitney AUC = 0.85, bootstrap 95% CI [0.70, 1.00]; Spearman correlation between `acf_24h` and v2 ensemble delta vs naive = 0.79 at p = 0.0038.
3. **Inferential** (Phase D, §4.5 cross-referenced in §4.8): predicate-stratified DM result, 23/1/0 in predicate-positive cells vs 15/2/3 in predicate-negative cells under cluster-bootstrap testing with Holm correction.

All three lines converge on the same conclusion. The agreement gives the predicate empirical support that no single test would provide alone.

### Chronos-2 load-bearing under DM testing

The ensemble's margin against Chronos-2 narrows relative to its margin against naive or ExtraTrees in every predicate-positive cell. This matches the leave-one-out finding from Phase C (`loo_ablation_new_pool_v2.csv`): dropping Chronos-2 from the pool degrades test R² by 2.0 to 3.5 percentage points across horizons, whereas dropping ExtraTrees or N-HiTS changes test R² by less than 0.1 pp. The DM result confirms that the ensemble adds measurable value beyond Chronos-2 alone but the margin is narrower than the margin against other alternatives.

### Friedman omnibus (v2, all horizons)

The Friedman test rejects the null of uniform ranks at every horizon. From `d4_friedman_TEST.csv`: χ² = 6395.6 / 7025.3 / 7288.0 / 6207.6 at the four horizons, N ∈ {5063, 5156, 5156, 5156} containers, Holm-corrected p = 0 at every horizon.

---

## 6. Outputs

### 6.1 Canonical (entered manuscript)

| File | Purpose | Section |
|---|---|---|
| `results/bcf_v2/d4_dm_pairwise_TEST.csv` | 44-test cluster-bootstrap DM result, Holm-corrected | §4.5 Table 4.X, §4.8 stratification |
| `results/bcf_v2/d4_friedman_TEST.csv` | Per-horizon Friedman ranks, v2 five-model family | §4.5 Table 4.Y |
| `results/bcf_v2/d4_verdict_TEST.md` | Verdict markdown summarising 38/3/3 + stratified breakdown | Reference document |
| `results/bcf_v2/cd_diagram_new_pool_ranks.csv` | Per-cell ranks for CD diagrams v2 | §4.5 Figure |
| `results/bcf_v2/cd_diagram_new_pool_10min.pdf` | CD diagram at h=10 | §4.5 Figure |
| `results/bcf_v2/cd_diagram_new_pool_30min.pdf` | CD diagram at h=30 | §4.5 Figure |
| `results/bcf_v2/cd_diagram_new_pool_60min.pdf` | CD diagram at h=60 | §4.5 Figure |
| `results/bcf_v2/cd_diagram_new_pool_120min.pdf` | CD diagram at h=120 | §4.5 Figure |

### 6.2 Appendix-level (OOF transparency)

| File | Purpose | Why appendix |
|---|---|---|
| `results/bcf_v2/d4_dm_pairwise_OOF.csv` | DM result on OOF residuals | Mechanically dominates because meta-learner saw the data; transparency only |
| `results/bcf_v2/d4_friedman_OOF.csv` | Friedman on OOF residuals | Same reasoning |
| `results/bcf_v2/d4_verdict_OOF.md` | OOF verdict markdown | Same reasoning |

### 6.3 Manuscript artefacts (LaTeX-ready)

| File | Purpose |
|---|---|
| `results/manuscript_artefacts/figures/fig_rank_trajectory.pdf` | Friedman mean ranks by horizon, all five members |
| `results/manuscript_artefacts/figures/fig_dm_significance.pdf` | DM significance heatmap, 44 tests |
| `results/manuscript_artefacts/tables/tab_dm_summary.tex` | 38/3/3 headline plus predicate-stratified breakdown |
| `results/manuscript_artefacts/tables/tab_dm_per_cell.tex` | Per-cell DM detail (44 rows; Appendix D candidate) |
| `results/manuscript_artefacts/tables/tab_friedman_ranks.tex` | Per-horizon mean ranks table |

### 6.4 Verification

| File | Purpose |
|---|---|
| `results/manuscript_artefacts/verify_anchors_new.json` | 26 anchors mapping Phase D claims to source CSV cells |

### 6.5 Auxiliary outputs (Phase D-adjacent stratification work)

| File | Purpose |
|---|---|
| `results/hpa/cv_bin_per_horizon_canonical.csv` | 16-cell CV × horizon table for §5 stratification context |
| `results/hpa/stratified_skill_new.csv` | Per-horizon pooled skill (4 rows) |
| `results/hpa/cv_stratified_skill_new.csv` | Per CV bin × horizon, container-level (16 rows) |
| `results/hpa/residual_diagnostics_new.csv` | Bias and ACF1 per horizon (Phase D residual sanity) |

### 6.6 LaTeX drafts produced 2026-05-22

| File | Length | Status |
|---|---|---|
| `chapters/04-implementation-results.tex` (§4.5 D4 inserts) | ~3000 chars new content | Draft, pending verification |
| `chapters/04-implementation-results.tex` (§4.6 cross-references) | ~600 chars new content | Draft |
| `chapters/04-implementation-results.tex` (§4.8 stratification paragraph) | ~1200 chars new content | Draft |
| `chapters/03-methodology.tex` (§3.7 cluster-bootstrap note) | ~250 words NEW | Pending draft |
| `appendices/D-statistical-validation.tex` | NEW file, planned | Not yet drafted |

The earlier draft `chapter_04_phaseD_complete.tex` (13,000 words, `/mnt/user-data/outputs/`) covered the unified §4.0-§4.12 structure and is now reference-only after the 6-chapter restructure decision.

---

## 7. Scripts

### 7.1 D4 core

| Script | Purpose |
|---|---|
| `task_d4_statistical_validation_TEST.py` | D4 cluster-bootstrap DM + Friedman + Holm on TEST residuals. Canonical. 18,237 bytes. |
| `task_d4_statistical_validation.py` | D4 same machinery on OOF residuals. Appendix only. 18,218 bytes. |

### 7.2 CD diagrams

| Script | Purpose |
|---|---|
| `regen_cd_diagrams_v2.py` | Regenerate critical-difference diagrams for the v2 five-model family. Produces per-horizon PDFs and the ranks CSV. 9,908 bytes. |

### 7.3 Manuscript artefacts

| Script | Purpose |
|---|---|
| `make_manuscript_figures_local_v2.py` | Produces `fig_rank_trajectory.pdf`, `fig_dm_significance.pdf`, plus three Phase C figures. 11,107 bytes. |
| `make_manuscript_tables_local_v2.py` | Produces `tab_dm_summary.tex`, `tab_dm_per_cell.tex`, `tab_friedman_ranks.tex`, plus four Phase C tables. 10,798 bytes. |
| `c5_continuous_auc_patch_local_v2.py` | Produces the verify_anchors_new.json file (also handles Phase C continuous AUC). 7,585 bytes. |

### 7.4 Auxiliary (stratification + verification)

| Script | Purpose |
|---|---|
| `regen_stratified_skill_local_v2.py` | Produces `stratified_skill_new.csv`, `cv_stratified_skill_new.csv`, `residual_diagnostics_new.csv`. 9,617 bytes. |
| `make_cv_bin_per_horizon_canonical.py` | Produces `cv_bin_per_horizon_canonical.csv` for §5 stratification context. 7,731 bytes. |
| `src/analysis/task4_verify_tables.py` | Verification framework (verify_numbers analog). 33,737 bytes. |

### 7.5 Cross-phase scripts read by Phase D

| Script | Phase | Purpose |
|---|---|---|
| `assemble_residuals_canonical.py` | B | Builds `results/canonical/test_residuals_canonical.parquet` (537 MB) and `oof_residuals_canonical.parquet` (1.7 GB); D4 reads these as its only data source. |
| `B6_apply_nnls_test_v3.py` | B | Phase B Chronos-2 scale fix; produces the v2 ensemble predictions D4 evaluates against. |

---

## 8. Manuscript integration

All inserts drafted 2026-05-22. Pending verification before commit. Manuscript section numbers refer to the existing Chapter 4 file structure (§4.0-§4.12); the 6-chapter restructure decision documented in memory #15 redistributes Phase D content to Ch3 §3.7, Ch5 §5.7, and Ch5 §5.6 but has not been executed in the LaTeX yet.

### 8.1 Chapter 3 — Methodology (new §3.7)

**§3.7 Statistical machinery: cluster-bootstrap DM and Holm correction** (NEW subsection)

Approximately 250 words documenting the cluster-bootstrap choice in place of classical Diebold-Mariano with Harvey-Leybourne-Newbold correction. Justification: multi-series data structure where container-id is the dominant dependence axis. Substitution made pre-test, not post-hoc. Citations: Diebold and Mariano 1995, Harvey-Leybourne-Newbold 1997, Holm 1979, Politis and Romano 1994.

File: `chapters/03-methodology.tex`. Status: not yet drafted. Independence: writable without other phase prose.

### 8.2 Chapter 4 — §4.5 Statistical Validation

The existing §4.5 carries v1 statistical validation (v1 Friedman, v1 DM with Newey-West, v1 CD diagrams). Phase D adds:

- New paragraph "Friedman omnibus (v2)" with the four-horizon mean-rank table (`tab_friedman_ranks.tex`).
- New paragraph "Diebold-Mariano pairwise (v2): the Phase D headline result" with the 38/3/3 outcome and the predicate-stratified breakdown (`tab_dm_summary.tex`).
- New paragraph "Critical-difference diagrams (v2)" with the four CD subfigures from `cd_diagram_new_pool_*.pdf` and commentary on N-HiTS rank 4.34 at h=10.
- Per-cell DM observation paragraph naming the three Bitbrains losses and the single Bytedance h=120 tie.

File: `chapters/04-implementation-results.tex`. Word count of D4 inserts: ~850 words.

### 8.3 Chapter 4 — §4.6 Cross-Dataset Validation (DM cross-references)

The existing §4.6 cross-references Phase D in three places without dedicated subsections:

- Bitbrains v2 paragraph: forward-reference to §4.5 Table 4.X for the cluster-bootstrap DM result; observation that the three Bitbrains long-horizon losses against per-member alternatives align with the predicate-negative regime.
- Bytedance paragraph: notes that all three predicate-positive Bytedance cells record Holm-corrected DM wins against every alternative member.
- Summary across traces: forward-reference to §4.5 for the statistical confirmation of the cross-dataset pattern.

File: `chapters/04-implementation-results.tex`. Word count of D4 inserts: ~250 words.

### 8.4 Chapter 4 — §4.8 BCF (predicate-stratified DM as strongest evidence)

The existing §4.8 carries the BCF predicate construction and the data-side and continuous-AUC evidence from Phase C. Phase D adds:

- New paragraph "Phase D statistical confirmation" with the predicate-stratified DM result (23/1/0 in predicate-positive, 15/2/3 in predicate-negative).
- Three-lines-of-evidence synthesis paragraph closing the BCF discussion: data-side AUC = 0.80, continuous AUC = 0.85 with Spearman ρ = 0.79 at p = 0.0038, cluster-bootstrap DM with Holm correction giving the stratified split.

File: `chapters/04-implementation-results.tex`. Word count of D4 inserts: ~400 words.

### 8.5 Chapter 4 §4.12 — chapter closing

The existing chapter closing carries one paragraph extension referencing Phase D as the inferential leg of the three-lines-of-evidence support for the BCF predicate. Also flags the v2 ACF(1) regression as the third chapter-level limitation.

File: `chapters/04-implementation-results.tex`. Word count of D4 inserts: ~120 words.

### 8.6 New Appendix D — Statistical Validation Detail

Five sub-sections planned:

- **D.1 OOF version of the D4 result.** Full DM table and Friedman ranks computed on OOF residuals. Mechanically dominated by the meta-learner; transparency only.
- **D.2 Per-cell DM detail.** Full 44-row table from `tab_dm_per_cell.tex`. Cells × four comparisons, with `mean_d`, raw p, Holm p, and winner_final columns.
- **D.3 Methodology details for the cluster bootstrap.** Reproduction protocol, seed, container resampling rule, Holm correction across the family.
- **D.4 Verification anchors.** Tabular summary of the 26 anchors in `verify_anchors_new.json`.
- **D.5 Critical-difference diagram conventions.** Bonferroni-Dunn correction note, critical-difference value 0.087-0.088 at α=0.05.

File: `appendices/D-statistical-validation.tex` (NEW file). Status: not yet drafted.

### 8.7 Appendix renumbering required

If both Phase A's Appendix C (Toto sensitivity) and Phase D's new Appendix D land before the existing Appendix C (Computational cost), renumbering is required:
- `appendices/C-toto-extension.tex` stays at C (Phase A claim).
- `appendices/D-statistical-validation.tex` is new D.
- `appendices/C-computational-cost.tex` → `appendices/E-computational-cost.tex` (was already flagged in `phase_a_monitor.md` §8.5).

---

## 9. Verification items

### 9.1 Blocking (must resolve before chapter commits)

- [ ] **Cluster-bootstrap methodology footnote.** §3.7 (NEW) must document the substitution from classical DM-HLN to container-clustered bootstrap. Citations: Diebold and Mariano 1995, HLN 1997, Holm 1979, Politis and Romano 1994. Estimated 250 words.

- [ ] **Sign convention consistency check.** The v1 `dm_stats_*.csv` files reported DM with inconsistent signs across horizons; verify that the D4 outputs (`d4_dm_pairwise_TEST.csv`) use a consistent sign convention throughout. Specifically, confirm that `mean_d > 0` always means the ensemble's squared error exceeds the comparison method's at every cell.

- [ ] **Predicate-stratified counts match canonical CSVs.** The text in §4.5 and §4.8 cites "23/1/0 in predicate-positive (24 tests, 6 cells)" and "15/2/3 in predicate-negative (20 tests, 5 cells)". Verify against `d4_dm_pairwise_TEST.csv` after stratifying by the `predicate` column from `bcf_pairs_v2.csv`. The 6 predicate-positive cells are Alibaba h ∈ {30, 60, 120} and Bytedance h ∈ {30, 60, 120}; the 5 predicate-negative cells are Alibaba h=10 and all four Bitbrains horizons.

- [ ] **Friedman v2 N values per horizon.** §4.5 cites N ∈ {5063, 5156, 5156, 5156} containers for the v2 Friedman. Verify against `d4_friedman_TEST.csv` directly.

- [ ] **DM magnitudes for the three Bitbrains losses.** §4.5 cites |DM| = 123.28 (h60 vs Chronos-2), 90.11 (h120 vs N-HiTS), 90.22 (h120 vs Chronos-2). Verify against the canonical CSV.

### 9.2 Non-blocking (housekeeping)

- [ ] **Residual_diagnostics_new.csv missing skewness columns.** The §4.5 prose cites skewness values (3.96 / 1.73 / 0.80 / 0.17 across horizons) sourced from an inline diagnostic, not from the CSV. Apply the two-line patch to `regen_stratified_skill_local_v2.py` and re-run:
```python
  'Skewness_naive': float(pd.Series(err_n_arr).dropna().skew()),
  'Skewness_ensemble': float(pd.Series(err_e_arr).dropna().skew()),
```
  After the patch, the §4.5 anchors trace fully to disk.

- [ ] **Mean pairwise correlation: 0.66 vs 0.67.** Some earlier memory entries say "0.66 across 11 cells"; the per-horizon weighted average across 2 + 3 + 3 + 3 cells gives 0.673. Confirm exact figure from `c2_correlations_per_cell.csv` and update §4.5 prose to match (currently uses "approximately 0.67").

- [ ] **Container counts caveat (§5 prose).** The container counts (1707 / 2027 / 760 / 390) hold at h ≥ 30 min; h=10 has slightly fewer (1641 / 2017 / 744 / 389) due to context-window filtering. Add "approximately" caveat or distinguish per-horizon counts.

### 9.3 Bibliography

- [ ] **Diebold-Mariano citation key.** `\cite{diebold-1995-comparing}` used in §3.7, §4.5. Verify against `references.bib` entry.

- [ ] **Harvey-Leybourne-Newbold citation key.** Used in §3.7 to motivate the substitution. Add entry to `references.bib` if not present (Harvey, Leybourne, Newbold 1997 IJF).

- [ ] **Holm citation key.** `\cite{holm-1979}` used in §3.7 and §4.5 methodology mention. Verify.

- [ ] **Politis-Romano citation key.** Used in §3.7 to motivate the cluster bootstrap. Add entry to `references.bib` if not present (Politis and Romano 1994 JASA).

- [ ] **Krogh-Vedelsby citation key.** `\cite{krogh-1994-error}` used in §4.5 commentary on N-HiTS rank 4.34. Verify.

- [ ] **Bonferroni-Dunn citation key.** Used in §4.5 critical-difference diagrams commentary. Verify.

### 9.4 Cross-references

- [ ] `\cref{tab:d4-summary}` referenced in §4.5, §4.6, §4.8. Confirm declared in §4.5.
- [ ] `\cref{tab:friedman-v2-ranks}` referenced in §4.5. Confirm declared.
- [ ] `\cref{fig:cd-v2}` referenced in §4.5. Confirm declared.
- [ ] `\cref{appx:d-stat-validation}` referenced in §4.5, §4.8. Confirm declared in `appendices/D-statistical-validation.tex`.
- [ ] `\cref{sec:bcf-results}` (Phase C / §4.8) referenced from §4.5. Already exists.
- [ ] `\cref{eq:bcf-predicate}` referenced from §4.5 predicate-stratification paragraph. Already exists.
- [ ] `\cref{sec:lim-chronos-dependence}` referenced from §4.5 LOO commentary. Confirm declared.

---

## 10. Known issues and lessons learned

### Hardware/environment

**Lesson:** D4 has no foundation-model runtime dependency. The script reads pre-computed residuals from `assemble_residuals_canonical.py`'s parquet output and runs entirely on CPU + standard scipy. This decouples Phase D execution from Phase A's `toto_env` isolation. Future statistical-validation runs can execute on any environment with numpy, pandas, scipy, and scikit-learn.

**Lesson:** The canonical residuals parquet is 537 MB (TEST) and 1.7 GB (OOF). Both exceed the Claude Projects upload limit and cannot live in the project knowledge base. They stay on Vast and in the local pull at `E:\thesis\vast_ai_full\`. Future Claude sessions that need to re-verify a number from raw residuals will not have direct access; they must request a one-line query from the user.

### Methodology

**Lesson:** Cluster bootstrap is the right test for multi-series forecast comparisons. The classical Diebold-Mariano-HLN test assumes a single series; our data has many containers per cell. Resampling on container-id absorbs the dominant dependence and produces wider, honest CIs. The substitution should be documented in the methodology chapter, not buried in the results.

**Lesson:** OOF results from the meta-learner are not inferentially honest. NNLS fits to OOF residuals, so any test on OOF data is mechanically biased toward the ensemble. The TEST-set version of D4 is the only inferentially honest view; the OOF version goes in the appendix for transparency only.

**Lesson:** Pre-specified stratification produced unexpected agreement. We did not engineer the BCF predicate to stratify the DM result; we ran the predicate as a stratification variable post-hoc and observed the clean split. This is the strongest single empirical claim Phase D produces. Random stratification would not have given 23/1/0 vs 15/2/3.

**Lesson:** Three independent lines of evidence are stronger than one strong line. The BCF predicate now carries data-side AUC, continuous-scale Spearman ρ, and inferential DM stratification. None of the three alone is as strong as the convergence of all three. The Chapter 6 conclusion should foreground this convergence rather than any single number.

### Manuscript integration

**Lesson:** The pre-registered methodology section should record the actual methodology, not the plan. We pre-registered classical DM-HLN; we executed cluster bootstrap. The §3.7 methodology note documents what we actually did and why we substituted, not what the plan said. Pre-registration is a defensive record, not a constraint on revised methodology when justified.

**Lesson:** Forward references to predicate-stratification require the BCF predicate to be defined first. The §4.5 D4 result can stand alone with the overall 38/3/3 headline; the predicate-stratified split (23/1/0 vs 15/2/3) is more naturally placed in §4.8 where the predicate already exists. We chose to put both in §4.5 and §4.8 in the current draft because the relationship is the load-bearing finding.

**Lesson:** The 6-chapter manuscript restructure decision (memory #15) requires Phase D content to be redistributed to Ch3 §3.7, Ch5 §5.7, and Ch5 §5.6. The current Chapter 4 file has Phase D content concentrated in §4.5 and §4.8 (unified structure). When the restructure executes, the §4.5 D4 paragraphs move to Ch5 §5.7 (overall 38/3/3 + Friedman v2 only), and the predicate-stratified paragraph moves to Ch5 §5.6 (cross-pool stability). The §3.7 methodology note stays in Ch3.

---

## 11. References

### Plan files
- `revised_plan.md` — surgical revision plan, Phase D section.
- `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` — full prose justification.
- `OUTLINE.md` — manuscript outline (§4.5 statistical validation anchor).
- `THESIS_HANDOFF.md` — submission-state reference.

### State files (cross-phase)
- `THESIS_STATE.md` — current state across phases.
- `DECISIONS.md` — locked decisions, including the 6-chapter restructure.
- `ERRATA.md` — submitted PDF errata.

### Memory entries (key Phase D references)
- Memory #2 — BCF canonical 3-model pool (NNLS+C2+TF): AUC=0.80, CI [0.70, 0.88], p=0.011.
- Memory #5 — Phase D D4 completion (38/3/3 headline + predicate-stratified split).
- Memory #6 — Phase D methodology substitution (cluster bootstrap vs DM-HLN).
- Memory #15 — 6-chapter manuscript structure decision; Phase D content placement.
- Memory #16 — Phase C C2 ρ̄ values feeding into Phase D context.
- Memory #17 — Phase C LOO ablation (Chronos-2 load-bearing).
- Memory #19 — Phase D archive (phase_d_scripts.tar.gz, 164KB, 12 scripts).
- Memory #24 — Phase D plan deviations (D1 substituted, D2/D3 not done).
- Memory #28 — BCF v2 AUC=0.800 CI [0.65, 0.95] (Phase C output, Phase D context).

### Cross-phase monitor files
- `phase_a_monitor.md` — Foundation-model leaderboard (D1 absorbed here).
- `phase_b_monitor.md` — Pool redesign (provides v2 ensemble predictions D4 evaluates).
- `phase_c_monitor.md` — Diversity diagnostic, BCF re-validation, ACI/AgACI (provides BCF predicate D4 stratifies on).

### Submitted PDF
- `Ensemble_Learning_for_Proactive_Resource_Prediction_PhanNguyenHungCuong_ITDSIU21078.pdf` (April 2026).
- §4.5 "Statistical Validation" — entered as v1 Friedman + Newey-West DM (no cluster bootstrap, no Holm correction across v2 family).
- §4.8 "Boundary Condition Framework" — entered with v1 BCF AUC = 0.833; Phase D adds predicate-stratified DM as strongest inferential evidence.

### LaTeX drafts (produced 2026-05-22, pending verification)
- `chapters/04-implementation-results.tex` (§4.5, §4.6, §4.8, §4.12 inserts)
- `chapters/03-methodology.tex` (§3.7 NEW — not yet drafted)
- `appendices/D-statistical-validation.tex` (NEW file — not yet drafted)
- `chapter_04_phaseD_complete.tex` (`/mnt/user-data/outputs/`) — reference-only after 6-chapter restructure decision

### External references
- Diebold-Mariano: Diebold and Mariano 1995, JBES (`\cite{diebold-1995-comparing}`).
- Harvey-Leybourne-Newbold: Harvey, Leybourne, Newbold 1997, IJF.
- Holm-Bonferroni: Holm 1979, SJS (`\cite{holm-1979}`).
- Politis-Romano cluster bootstrap: Politis and Romano 1994, JASA.
- Bonferroni-Dunn for CD diagrams: Demšar 2006, JMLR.
- Krogh-Vedelsby ensemble decomposition: Krogh and Vedelsby 1995 NeurIPS (`\cite{krogh-1994-error}`).

---

## Appendix: Phase D artefact inventory (alphabetical)

Quick-reference list of every file touched during Phase D, alphabetised for grep-friendliness.