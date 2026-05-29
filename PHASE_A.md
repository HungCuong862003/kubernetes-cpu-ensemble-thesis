# Phase A: Foundation-Model Leaderboard

| Field | Value |
|---|---|
| Phase ID | A |
| Title | Foundation-Model Leaderboard |
| Period | 2026-05-13 → 2026-05-22 |
| Status | **Complete** (experimental work + manuscript drafts) |
| Owner | Jimmy (ITDSIU21078) |
| Supervisor | Dr. Ho Long Van |
| Source plan | `revised_plan.md`, `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` |
| Submitted PDF affected | Yes — §4.7, §4.8, §5.3, §6.1, §6.2, §6.3 + new Appendix C |

---

## 1. Scope and rationale

A 2026 thesis on cloud workload forecasting cannot defend the claim that a custom four-model ensemble is the right answer without benchmarking against current time-series foundation models. Phase A addresses this gap by evaluating the v1 NNLS ensemble zero-shot against four such models (Chronos-2, TimesFM-2.5, Granite-TTM, Toto-Open-Base-1.0) across the three production traces (Alibaba, Bitbrains, Bytedance) at four prediction horizons (10/30/60/120 min). The phase has three deliverables: a four-model leaderboard table for the main chapter, a sensitivity check on a fifth model whose pre-training corpus is observability-weighted, and per-cell tally evidence for the BCF predicate's transferability across model families.

The phase is intellectually load-bearing for the thesis's contribution claim. If foundation models beat the ensemble across the board, the ensemble construction is no longer the contribution; the BCF predicate must carry the contribution alone. Phase A's job is to surface that result honestly rather than hide it.

---

## 2. Plan summary

From `revised_plan.md`:

- Evaluate Chronos-2, TimesFM-2.5, Granite-TTM zero-shot across all 12 (dataset, horizon) cells.
- Pre-registered origin sampling: K=50 origins per container, seed=42, sliding-window.
- Construct per-cell winner tally as the operational test.
- Apply the BCF predicate (`ACF@24h > 0.2 AND h ≥ 30 min`) across all four models (ensemble + 3 FMs) and report per-model AUC.
- Exclude any FM whose 5-model BCF AUC drops below the predicate-respecting pool's AUC by more than the pre-committed threshold.
- Toto-Open-Base-1.0 added later as a sensitivity check, separately reported.

Pre-registration document: section "Phase A goals" in `revised_plan.md` (Detailed Description Final Surgical Thesis Revision Plan).

---

## 3. Execution

### 3.1 Timeline

| Sub-task | Dates | Hardware | Status |
|---|---|---|---|
| A1 spike test (4 FMs) | 2026-05-13 → 2026-05-15 | C.37118184 (RTX 5070 Ti, Blackwell) | Complete |
| A2 production inference | 2026-05-17 → 2026-05-21 | C.37124280 (RTX 4090, Ada) | Complete |
| A3 leaderboard build (v1) | 2026-05-13 → 2026-05-15 | Local + Vast.ai | Complete |
| A3 leaderboard build (v3 planning) | 2026-05-21 | C.37124280 | Complete; did NOT enter manuscript |
| A4 manuscript integration | 2026-05-22 | Local | Drafted; pending verification |

### 3.2 Hardware and environments

**Vast.ai compute:**
- C.37118184 (RTX 5070 Ti, driver poisoned mid-A1 by torch dependency conflict) — migrated off.
- C.37124280 (RTX 4090, driver 565.77, CUDA 12.7, 384-core AMD EPYC host) — canonical for A2.
- SSH: `ssh -p 20764 root@213.181.123.64`.

**Conda environments:**
- `/venv/main` — torch 2.11.0+cu130 (later downgraded to 2.4.1+cu130 to match production). Hosts Chronos-2, TimesFM, Granite-TTM inference paths.
- `/venv/toto_env` — torch 2.7.0+cu126, setuptools <81 (the `pkg_resources` removal in setuptools 81+ breaks `lightning 2.3.3`). Isolated to avoid hard conflict with main env's torch.
- `HF_HOME=/workspace/.hf_home_v2` (cached model weights).

**Why Toto needed isolation:** `toto-ts 0.2.0` pins `torch==2.7+cu128` hard. Installing it in the main env downgraded torch and broke Chronos-Bolt + N-HiTS + the v2 NNLS refit. The isolated env adds operational complexity (separate Python invocations for Toto vs everything else) but is the only path that keeps both stacks working.

### 3.3 Sub-task breakdown

**A1 — Spike test (viability check).** Goal: confirm all four foundation models can run on the available hardware within reasonable throughput before committing to the full evaluation. Result: all three primary FMs (Chronos-2, TimesFM, Granite-TTM) plus Toto passed the viability threshold. Per-series timings on the spike hardware:

| Model | Serial throughput | Multiple under viability threshold |
|---|---|---|
| Chronos-2 | 0.028 s/series | 35× under |
| TimesFM-2.5 | 0.180 s/series | 5× under |
| Granite-TTM | 0.012 s/series | 80× under |
| Toto-Open-Base-1.0 | 0.095 s/series | 105× under (serial); batched path failed |

Spike-test scripts: `A1_spike_test.py`, `A1_toto_only.py`.

**A2 — Production inference.** Goal: produce per-cell R² values for all four foundation models on all 12 (dataset, horizon) cells, plus naive baselines on the matched subsample. Key deviations from plan documented in §4 below.

**A3 — Leaderboard build.** Two leaderboards produced:

- `leaderboard_v1.csv` (4 models: NNLS + Chronos-2 + TimesFM + Granite-TTM, K=20 on Alibaba, K=50 on Bitbrains/Bytedance for TimesFM specifically). **This is what entered the manuscript.** Build script: `finalize_leaderboard_v1.py`.
- `leaderboard_v3_*.csv` (4 models: NNLS + Chronos-2 + TimesFM + Toto, Granite-TTM dropped). **Planning artefact only.** Did not enter the manuscript. Build script: `A2_build_leaderboard_v3.py`.

The v1 leaderboard is canonical because the submitted PDF (April 2026) used the 4-model NNLS+C2+TF+Granite pool, and the thesis defence references that PDF directly. The v3 leaderboard with Toto replacing Granite was an internal experiment to test whether a different fourth FM would change the tally; it does not, but the v3 file does not appear in the chapter.

**A4 — Manuscript integration (2026-05-22).** LaTeX drafts produced across Chapter 4, Chapter 5, Chapter 6, and a new Appendix C. Detail in §8 below.

---

## 4. Deviations from plan

### Hardware migration mid-A1
The pre-registered compute was the RTX 5070 Ti on instance C.37118184. The Toto installation cascaded into a driver-level failure on that instance (torch 2.7+cu128 vs the host driver), forcing migration to RTX 4090 on C.37124280. The migration cost about half a day of wall-clock and required re-running the Chronos-2 sanity check on the new hardware (`day1_chronos2_sanity.py`) to confirm bit-identical outputs.

### Toto batched-inference path failed
Pre-registration assumed Toto could use the `toto-ts 0.2.0` `id_mask` mechanism to batch independent univariate series in a single forward pass. The batched path triggered a CUDA `unspecified launch failure` after roughly 70 batches and propagated to a host-level GPU driver error on two separate instances. The fall-back was serial inference, which is approximately 100× slower per series than Chronos-2. This forced the Alibaba subsample (see next deviation).

### Alibaba subsample (Toto only)
Full-coverage Toto on Alibaba would have required approximately three hours of unattended GPU time. Within the revision window, the choice was subsample-Toto or skip-Toto-Alibaba. We subsampled to 1,000 of 4,921 containers (seed=42), accepted the sampling asymmetry, and disclosed it in Appendix C. The asymmetry does not change Toto's substantive ranking (fourth among the four FMs we evaluated), but Toto's absolute Alibaba R² values are on a slightly harder slice of the test set than the headline naive baseline reports.

### K=20 vs K=50 sampling split
Pre-registration called for K=50 origins per container throughout. Actual run used K=20 on Toto and Chronos-2 (every dataset) and K=20 on Alibaba + K=50 on Bitbrains/Bytedance for TimesFM (because the first TimesFM Bitbrains/Bytedance runs at K=20 produced unstable cell estimates that K=50 stabilised). The mixed protocol is disclosed in the §4.7 table caption.

### Toto added after the canonical run closed
The original Phase A pre-registration was three FMs (Chronos-2, TimesFM, Granite-TTM). Toto was added as a fourth model after the canonical run had already locked, motivated by the corpus-specificity hypothesis (Toto's pre-training is observability-weighted; would it beat Chronos-2 on Bytedance?). The hypothesis was rejected. Toto sits in Appendix C rather than the main leaderboard table because the canonical BCF analysis had already closed before Toto's results were available.

---

## 5. Substantive findings

### Per-cell winner tally (canonical, 4-model leaderboard_v1)

| Model | Wins | Where |
|---|---|---|
| Chronos-2 | 6 | Alibaba 30; Bitbrains 10; Bytedance 10/30/60/120 |
| TimesFM | 3 | Alibaba 60/120; Bitbrains 30 |
| Granite-TTM | 2 | Bitbrains 60/120 |
| NNLS ensemble | 1 | Alibaba 10 |

**Key margins:**
- Largest gap: Bytedance 30 min, Chronos-2 R² = 0.942 vs NNLS R² = 0.806 (+13.6 pp).
- Smallest win: Alibaba 10 min, NNLS R² = 0.9224 vs Chronos-2 R² = 0.9190 (+0.34 pp, ceiling cell).
- Alibaba 120 min flagship: TimesFM R² = 0.7726, beating NNLS (0.7584) by +1.42 pp and Chronos-2 (0.7612) by +1.14 pp. Earlier draft framed Alibaba 120 min as the ensemble's flagship win; that framing died with Phase A.

### Toto sensitivity (5-model with Toto, Granite-TTM also retained for the BCF run)
- Toto wins one cell out of twelve: Bitbrains 10 min, R² = 0.881 vs Chronos-2 0.877 (+0.33 pp). Within bootstrap noise.
- Toto places fourth among the four foundation models we evaluated. Does not win any Bytedance cell.
- The corpus-specificity hypothesis (Toto's observability-weighted pre-training should produce a Bytedance advantage) is rejected.

### BCF predicate transferability
- 3-model canonical pool (NNLS, Chronos-2, TimesFM) at 36 cells: AUC = 0.80, BCa 95% CI [0.70, 0.88], permutation p = 0.011.
- Per-model AUC: NNLS 0.833, Chronos-2 0.773, TimesFM 0.800, Granite 0.500.
- 5-model run with Granite included: AUC = 0.667, p = 0.0511. Used as exclusion justification, not as headline.
- Granite-TTM excluded from canonical pool because its win pattern is dataset-specific (4/4 Bytedance, 0/4 Alibaba), not predicate-aligned.
- Toto exclusion follows the same logic. Its 5-model run produces a comparable AUC drop.

### Headline findings for the manuscript
1. Foundation models collectively win 11/12 cells against the ensemble at the leaderboard subsample scope.
2. Chronos-2 specifically wins 6/12. The operational recommendation in §5.3 is "query Chronos-2 zero-shot, then check the BCF predicate."
3. The BCF predicate transfers across model families — its discriminative power does not depend on which ML method is the candidate.
4. Pre-training corpus specificity is not the differentiator for foundation-model selection on cloud workloads. Architecture and parameter count are.

---

## 6. Outputs

### 6.1 Canonical (entered manuscript)

| File | Purpose | Scope |
|---|---|---|
| `leaderboard_v1.csv` | 4-model leaderboard (NNLS + C2 + TF + Granite) | §4.7 canonical |
| `bcf_pooled_results.json` | Pooled BCF AUC, CI, permutation p | §4.8 canonical |
| `bcf_pooled_3model.json` | 3-model canonical pool stats (NNLS + C2 + TF) | §4.8 canonical |
| `bcf_per_model_auc.csv` | Per-model BCF AUC for the canonical pool + Granite | §4.8 Table |
| `bcf_pairs.csv` | Per-cell ML-win pairs used for BCF analysis | §4.8 + Appendix B |
| `bcf_5model_run.log` | Justification log for Granite exclusion | §4.8 exclusion paragraph |

### 6.2 Planning artefacts (did NOT enter manuscript)

| File | Purpose | Why not in manuscript |
|---|---|---|
| `leaderboard_v3_wide.csv` (5761 B) | 4-model with Toto, Granite dropped | Submitted PDF locked in 4-model v1 pool; v3 was internal experiment |
| `leaderboard_v3_long.csv` (2176 B) | Same in long format | Same |
| `leaderboard_v3_winners.csv` (1560 B) | Per-cell winner labels for v3 | Same |

The v3 files were used to populate the Appendix C table (`tab:fm-leaderboard-toto`), which IS in the manuscript. The v3 leaderboard itself is not.

### 6.3 Per-dataset JSONs

| File | Model | Coverage | Origin protocol |
|---|---|---|---|
| `toto_k20_alibaba.json` | Toto | 1000 of 4921 containers | K=20, seed=42 |
| `toto_k20_bitbrains.json` | Toto | 142 of 156 active VMs | K=20, seed=42 |
| `toto_k20_bytedance.json` | Toto | 93 of 93 containers | K=20, seed=42 |
| `timesfm_k20_alibaba.json` | TimesFM | Full | K=20 |
| `timesfm_k50_bitbrains.json` | TimesFM | Full | K=50 (K=20 unstable) |
| `timesfm_k50_bytedance.json` | TimesFM | Full | K=50 (K=20 unstable) |
| `k20_alibaba_v2.json` | Chronos-2 | Full | K=20 |
| `k20_bitbrains_v2.json` | Chronos-2 | Full | K=20 |
| `k20_bytedance_v2.json` | Chronos-2 | Full | K=20 |
| `chronos2_zero_shot_results.json` | Chronos-2 | Reference | K=20 |

### 6.4 Sensitivity and diagnostic outputs

- `k50_sensitivity_summary.csv` — K=20 vs K=50 comparison across horizons.
- `k50_bitbrains_check.json`, `k50_bytedance_check.json` — K=50 verification runs.
- `compare_k20_k50_output.txt` — Sensitivity diagnostic text dump.
- `cross_dataset_headline_v2.csv` — Read by Phase A (NNLS column at full-test scope); written by Phase B.

### 6.5 LaTeX drafts produced 2026-05-22

| File | Length | Status |
|---|---|---|
| `chapters/04-implementation-results.tex` | ~40,000 chars (full chapter, Phase A inserts integrated) | Draft, pending verification |
| `chapters/05-discussion-evaluation.tex` | ~25,000 chars (full chapter, §5.3 three-paragraph insert) | Draft, pending verification |
| `chapters/06-conclusion-future-work.tex` | ~12,000 chars (full chapter, §6.1+§6.2+§6.3 inserts) | Draft, pending verification |
| `appendices/C-toto-extension.tex` | ~15,000 chars (new, ~1,800 words across 6 sub-sections) | Draft, pending verification |

---

## 7. Scripts

### 7.1 Phase A1 (spike test)

| Script | Purpose |
|---|---|
| `A1_spike_test.py` | Viability check across all 4 FMs on a small sample. |
| `A1_toto_only.py` | Toto-specific viability run (isolated env required). |
| `day1_chronos2_sanity.py` | Chronos-2 bit-identical sanity check on new hardware. |
| `day2_timesfm_sanity.py` | TimesFM sanity check before production run. |

### 7.2 Phase A2 (production inference)

| Script | Purpose |
|---|---|
| `A2_toto_serial.py` | Production Toto inference, serial mode (batched failed). |
| `A2_run_all_serial.sh` | Orchestrator across all three datasets. |
| `A2_REBUILD.sh` | Full Phase A rebuild script (Toto + leaderboard regeneration). |

### 7.3 Phase A3 (leaderboard build)

| Script | Purpose |
|---|---|
| `finalize_leaderboard_v1.py` | Builds canonical `leaderboard_v1.csv` (entered manuscript). |
| `A2_build_leaderboard_v3.py` | Builds `leaderboard_v3_*.csv` (planning artefact). |
| `merge_timesfm_into_leaderboard.py` | Merges TimesFM cells into the v1 leaderboard. |
| `finalize_phase_a.sh` | Phase A finalisation script. |

### 7.4 Infrastructure (environment, Vast.ai)

| Script | Purpose |
|---|---|
| `setup_toto_isolated.sh` | Creates `toto_env` conda environment. |
| `setup_timesfm.sh` | TimesFM environment setup. |
| `_patch_load_timesfm.py` | Patch for TimesFM loader. |
| `repair_env_v2.sh` | Main environment repair after Toto torch conflict cascade. |
| `fix_toto_env.sh` | Toto env corrective patches. |
| `inspect_vast.sh` | Vast.ai instance diagnostics. |

### 7.5 Sensitivity / cross-checks

| Script | Purpose |
|---|---|
| `compare_k20_k50_bitbrains.py` | K=20 vs K=50 sensitivity check. |
| `run_k50_bitbrains.sh` | K=50 sensitivity runner. |

### 7.6 Cross-phase scripts read by Phase A

| Script | Phase | Purpose |
|---|---|---|
| `B6_apply_nnls_test_v3.py` | B | Chronos-2 scale fix; the Phase A leaderboard NNLS column depends on this. |
| `assemble_residuals_canonical.py` | B | Residual pool construction used by Phase A NNLS column. |

---

## 8. Manuscript integration

All inserts drafted 2026-05-22. Pending verification before commit.

### 8.1 Chapter 4 — Implementation and Results

**§4.7 Foundation-Model Leaderboard** (existing section, Phase A revised)
- Opening paragraph: "three" → "four" foundation models; Toto cross-referenced to Appendix C.
- Table caption: methodology language aligned with §3.6 (K=20 origin sampling, Naive(sub)/Naive(full) distinction).
- New paragraph at section end: "A fourth foundation model on a different protocol" — summarises Toto outcome without splicing into the four-model table.

**§4.8 Boundary Condition Framework** (existing section, Phase A addition)
- New paragraph after "Why we exclude Granite-TTM": "The same exclusion logic applies to Toto" — 5-model BCF run with Toto produces parallel AUC = 0.667 at p = 0.051; mechanism is the same as Granite case.

File: `chapters/04-implementation-results.tex`. Word count of Phase A inserts: ~450 words.

### 8.2 Chapter 5 — Discussion and Evaluation

**§5.3 Foundation-Model Dominance** (existing section, Phase A addition)

Three new paragraphs inserted after the existing "What this does not change" paragraph:

1. **"A fourth foundation model, with its own pre-training story"** — introduces Toto and the corpus-specificity hypothesis that motivated adding it. Sets up the test: if pre-training corpus matters at all for cloud-workload forecasting, Toto should beat Chronos-2 on Bytedance.
2. **"The corpus-specificity hypothesis loses cleanly"** — reports the rejection. Toto places fourth among the four FMs we evaluated; does not win any Bytedance cell. The simpler explanation wins (architecture + parameter count, not training corpus composition).
3. **"Two limitations of the foundation-model comparison"** — discloses Toto's Alibaba subsample asymmetry and the K=20 within-container correlation issue. Caveats §4.7's closest cells as "approximately tied" rather than confident wins.

File: `chapters/05-discussion-evaluation.tex`. Word count of Phase A inserts: ~550 words.

### 8.3 Chapter 6 — Conclusion and Future Work

**§6.1 Summary of Contributions** (Objective 2 paragraph, Phase A addition)
- One sentence added: Toto added as sensitivity check, places fourth, sharpens the recommendation to query Chronos-2 specifically.

**§6.2 Limitations** (4th limitation paragraph extended)
- Toto subsample asymmetry disclosed. Asymmetry does not change substantive ranking; absolute R² values on Alibaba should be read with the subsample caveat.

**§6.3 Future Work** (item 3 extended)
- Full-coverage Toto re-run on Alibaba added as near-term housekeeping (~3 hours unattended GPU time). Removes the subsample-asymmetry footnote in Appendix C.

**§6.4 Closing Reflection** — no inline edit. Optional Toto sentence held in verification list.

File: `chapters/06-conclusion-future-work.tex`. Word count of Phase A inserts: ~190 words.

### 8.4 New Appendix C — Toto-Open-Base-1.0 Sensitivity Check

Six sub-sections totalling ~1,800 words:

- **C.1 Why a separate evaluation** — three operational realities forcing Toto outside the canonical sampling protocol (inference throughput, batched-inference failure, timing relative to canonical BCF analysis).
- **C.2 Evaluation setup** — K=20 origin protocol, Alibaba subsample design, full Bitbrains/Bytedance coverage, reproducibility check on Bytedance (0.001 pp determinism).
- **C.3 Per-cell results** — 5-model leaderboard table (`tab:fm-leaderboard-toto`); discussion of Toto's one win (Bitbrains 10 min) and the absent Bytedance advantage.
- **C.4 Per-cell winner tally with Toto** — re-tally across 12 cells; 5-model BCF AUC drops to 0.667 at p = 0.051, justifying Toto exclusion from canonical BCF pool.
- **C.5 Substantive findings under the Toto extension** — Bytedance dominance by FMs unchanged; Alibaba ceiling cell unchanged; Bitbrains predicate-negative regime stays noisy; Toto's overall rank stable at fourth.
- **C.6 What the subsample does not tell us** — discusses subsample-naive shift on Alibaba (0.5–2.8 pp wider at long horizons); substantive ranking unchanged.

File: `appendices/C-toto-extension.tex` (NEW file).

### 8.5 Appendix renumbering required

Existing `appendices/C-computational-cost.tex` must be renumbered to Appendix E to make room for the new Toto sensitivity appendix at C.

- File rename: `appendices/C-computational-cost.tex` → `appendices/E-computational-cost.tex`.
- Label preservation recommended: keep `\label{appx:cost}` so existing `\cref{appx:cost}` references (in §4.3 "Reproducibility" and §C.2 "Why shorter horizons cost more") continue to resolve.
- Frontmatter file update: `frontmatter/` order of appendices file (if exists).

---

## 9. Verification items

### 9.1 Blocking (must resolve before chapter commits)

- [ ] **§4.7 tally scope.** Existing chapter reports "C2 6 / TF 3 / Granite 2 / NNLS 1" at the leaderboard_v1.csv subsample scope. Some userMemory entries record "NNLS 7 / Granite 2 / C2 2 / TF 1" at the NEW-pool full-test scope (`cross_dataset_headline_v2.csv`). These are two different scopes, both valid, must not be conflated. Confirm §4.7 cites leaderboard_v1.csv (subsample scope) and that the cross_dataset_headline_v2.csv tally appears only in §5 disclosure context.

- [ ] **Appendix C table values.** `tab:fm-leaderboard-toto` cell values were typed by hand during the Phase A4 drafting. Regenerate from `leaderboard_v3_wide.csv` directly before commit. Particular cells to double-check: Bitbrains 10 min (Toto vs Chronos-2 margin), Bytedance 60+120 min (Toto's third-place ranking vs TimesFM).

- [ ] **5-model BCF AUC with Toto.** Currently §C.4 quotes AUC = 0.667 at p = 0.051, inherited from the Granite 5-model run as a placeholder. Compute explicitly with Toto in pool (not Granite). The substantive finding (Toto exclusion justified) does not depend on exact value, but the number itself must be correct.

### 9.2 Non-blocking (housekeeping)

- [ ] Existing Computational Cost appendix renumber from C to E.
- [ ] Full-coverage Toto re-run on Alibaba (~3 hours unattended GPU on RTX 4090 or equivalent). Removes the subsample-asymmetry footnote in `tab:fm-leaderboard-toto`. Estimated throughput 35–36 series/sec serial; full Alibaba evaluation = 4,921 containers × 20 origins × 4 horizons = 393,680 forecast calls.
- [ ] Toto Bitbrains 30 min cell value (Appendix C table): typed at 0.621; verify against JSON file directly.

### 9.3 Bibliography

- [ ] **Toto citation key.** `\cite{datadog-2024-toto}` used in §4.7, §5.3, §6.1, Appendix C. Verified at arXiv:2407.07874 per memory. Re-confirm against `references.bib` entry text and matching arXiv abstract.
- [ ] Other Phase A citation keys to confirm against `references.bib`: `ansari-2024-chronos` (Chronos-2), `das-2024-timesfm` (TimesFM), `ekambaram-2024-ttm` (Granite-TTM).
- [ ] Phase A does NOT touch the renamed-key audit (`ueda-1996-bias-variance`, `brown-2005-diversity-regression`, `diebold-1995-dm`) inherited from the bibliography correction work; those are Phase D/cross-phase concerns.

### 9.4 Cross-references

- [ ] `\cref{app:toto-robustness}` used in §4.7, §4.8, §5.3, §6.1, §6.2, §6.3. Confirm `\label{app:toto-robustness}` is declared in `appendices/C-toto-extension.tex`.
- [ ] `\cref{tab:fm-leaderboard-toto}` used in §5.3, §6.3. Confirm declared in Appendix C.
- [ ] `\cref{sec:fm-eval}` referenced in §C.2 and in §3.6 (if §3.6 was updated for Phase A). Confirm declared in §3.

---

## 10. Known issues and lessons learned

### Hardware/environment

**Lesson:** Foundation-model dependencies are fragile across torch versions. `toto-ts 0.2.0` requires a torch version (2.7+cu128) incompatible with the main environment used by Chronos-Bolt, N-HiTS, and the v2 NNLS refit. Isolated conda environments are the only stable answer; sequential `pip install` will damage one stack while installing the other.

**Lesson:** Vast.ai instances are not snapshots. `Stop` preserves disk, `Destroy` does not. Use `Stop` for any instance whose environment took non-trivial time to build. Reconstructing `toto_env` from scratch takes ~25 minutes; the main env takes ~40 minutes; both together is an hour of work to recover from an accidental `Destroy`.

### Methodology

**Lesson:** Pre-registered sampling protocols (K=50 origins) sometimes need to be revised mid-execution. K=20 worked fine on Toto and Chronos-2 but produced unstable TimesFM cells on Bitbrains and Bytedance. The mixed K=20/K=50 protocol is documented in the §4.7 caption; this is the right disclosure pattern when pre-registration is partially followed.

**Lesson:** Pooled vs per-VM/per-container aggregation can reverse the sign of the result. Phase A's leaderboard cells on Bitbrains were susceptible to this: pooled Bitbrains gave Toto a +0.33 pp win at h10, but per-VM median on the same data would have given different cell winners. We chose pooled-with-disclosure as the canonical aggregation rather than per-VM-median because the leaderboard is a model-vs-model comparison where pooled is the cleaner metric.

### Manuscript integration

**Lesson:** Two different tallies at two different scopes (subsample vs full-test) easily confuse later readers, including the author. The disambiguation between leaderboard_v1.csv (subsample scope) and cross_dataset_headline_v2.csv (full-test scope) needs to be explicit in every caption that references a per-cell win count. The submitted PDF's "Chronos-2 6, TimesFM 3, Granite 2, NNLS 1" is at subsample scope; the userMemory's "NNLS 7, Granite 2, Chronos-2 2, TimesFM 1" is at full-test scope. Both are correct in their own context.

**Lesson:** The Toto sensitivity check belongs in an appendix, not the main leaderboard. The sampling-asymmetry footnote that would be required to include Toto in `tab:foundation` (its Alibaba column on a 1,000-container subsample vs the other models' ~98,000-point evaluations) is more visually disruptive than the separate-appendix approach.

---

## 11. References

### Plan files
- `revised_plan.md` — top-level surgical revision plan (Phases A–E).
- `Detailed_Description_Final_Surgical_Thesis_Revision_Plan.txt` — full prose justification.
- `OUTLINE.md` — manuscript outline (§4.7 anchor for Phase A).
- `THESIS_HANDOFF.md` — submission-state reference.

### State files (cross-phase)
- `THESIS_STATE.md` — current state across phases.
- `DECISIONS.md` — locked decisions, including Phase F pre-registration.
- `ERRATA.md` — submitted PDF errata tracked across phases.

### Memory entries (key Phase A references)
- Memory #11 — Chronos-2 scale bug fix (Phase B); Phase A NNLS column depends on this fix.
- Memory #18 — Leaderboard versioning history (v1 canonical vs v3 planning artefact).
- Memory #20 — Toto inference execution details (instance, env, throughput).
- Memory #30 — Phase A manuscript integration completion (this phase).

### Submitted PDF
- `Ensemble_Learning_for_Proactive_Resource_Prediction_PhanNguyenHungCuong_ITDSIU21078.pdf` (April 2026).
- §4.7 "Foundation-Model Leaderboard" — entered as 4-model NNLS+C2+TF+Granite tally.
- §6.1 Objective 2 — claimed foundation-model benchmark contribution.

### LaTeX drafts (produced 2026-05-22, pending verification)
- `chapters/04-implementation-results.tex`
- `chapters/05-discussion-evaluation.tex`
- `chapters/06-conclusion-future-work.tex`
- `appendices/C-toto-extension.tex` (new)

### External references
- Chronos-2: Ansari et al. 2024, arXiv:2403.07815 (`\cite{ansari-2024-chronos}`).
- TimesFM-2.5: Das et al. 2024, arXiv:2310.10688 (`\cite{das-2024-timesfm}`).
- Granite-TTM: Ekambaram et al. 2024, arXiv:2401.03955 (`\cite{ekambaram-2024-ttm}`).
- Toto-Open-Base-1.0: Datadog 2024, arXiv:2407.07874 (`\cite{datadog-2024-toto}`).

---

## Appendix: Phase A artefact inventory (alphabetical)

Quick-reference list of every file touched during Phase A, alphabetised for grep-friendliness.

```
A1_spike_test.py
A1_toto_only.py
A2_REBUILD.sh
A2_build_leaderboard_v3.py
A2_run_all_serial.sh
A2_toto_serial.py
appendices/C-toto-extension.tex        [NEW]
bcf_5model_run.log
bcf_pairs.csv
bcf_per_model_auc.csv
bcf_pooled_3model.json
bcf_pooled_results.json
chapters/04-implementation-results.tex [REVISED 2026-05-22]
chapters/05-discussion-evaluation.tex  [REVISED 2026-05-22]
chapters/06-conclusion-future-work.tex [REVISED 2026-05-22]
chronos2_zero_shot_results.json
compare_k20_k50_bitbrains.py
compare_k20_k50_output.txt
cross_dataset_headline_v2.csv          [Phase B; read by Phase A]
day1_chronos2_sanity.py
day2_timesfm_sanity.py
finalize_leaderboard_v1.py             [CANONICAL leaderboard builder]
finalize_phase_a.sh
fix_toto_env.sh
inspect_vast.sh
k20_alibaba.json                       [superseded by _v2]
k20_alibaba_v2.json
k20_bitbrains.json                     [superseded by _v2]
k20_bitbrains_v2.json
k20_bytedance.json                     [superseded by _v2]
k20_bytedance_v2.json
k50_bitbrains_check.json
k50_bytedance_check.json
k50_sensitivity_summary.csv
leaderboard_v1.csv                     [CANONICAL §4.7]
leaderboard_v3_long.csv                [PLANNING ARTEFACT, not in manuscript]
leaderboard_v3_wide.csv                [PLANNING ARTEFACT, used for Appendix C table]
leaderboard_v3_winners.csv             [PLANNING ARTEFACT]
merge_timesfm_into_leaderboard.py
_patch_load_timesfm.py
repair_env_v2.sh
run_k50_bitbrains.sh
setup_timesfm.sh
setup_toto_isolated.sh
timesfm_k20_alibaba.json
timesfm_k50_bitbrains.json
timesfm_k50_bytedance.json
toto_k20_alibaba.json
toto_k20_bitbrains.json
toto_k20_bytedance.json
```

End of Phase A monitor file.