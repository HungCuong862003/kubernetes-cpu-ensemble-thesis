# THESIS_STATE.md

**Last updated:** 2026-05-30 (F0-D8 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 8 of 10
- **Phase F overall:** day 8 of ~135
- **Days until defence:** ~128

## Today's headline

**Today completed (Day 8):**
- D8 Task A — F2 chapter outline promoted from D7 sketch v3.1 to D8 defensible-structure v4 at `phase_f/journal/f2_chapter_outline.md`. Two TBV items cleared at D8 (§5 h10 ordering via `bcf_pairs.csv` NNLS rows showing monotonic ByteDance > Alibaba > Bitbrains at all four horizons; §5 M4-literature citation resolved to Pennekamp et al. 2019 Ecological Monographs + Ponce-Flores et al. 2020 Entropy 22(1):89). Five TBV items remain, all method-detail or supporting-numeric, deferred to D11+ source reads. Critical scope correction in §5: h30 monotonicity exception (Bitbrains > Alibaba) is at NEW pool full-test aggregation (`cross_dataset_headline_v2.csv`), NOT F2's input scope (`bcf_pairs.csv`); v3.1 conflated these. v4 discloses both aggregations explicitly. Single strongest claim named: §5 WPE-ACF partial substitution finding. Single weakest claim named: §6 limitation 1, CI degeneracy at k=3 (uncertainty problem, not magnitude problem; bounds the headline statistical claim).
- D8 Task B — F1 router design lock at `phase_f/journal/f1_prep_scope.md`. Four-feature classifier (ACF@24h, horizon_min, CV, ACF@1h) at cell-level granularity. Training: 12 NNLS cells from `bcf_pairs.csv` with labels from `leaderboard_v1.csv` winner tally (Chronos-2:6, TimesFM:3, Granite-TTM:2, NNLS:1). Evaluation: LOO-cell CV with macro-F1 ≥ 0.55 per DECISION-005. Baseline: always-predict-Chronos-2 yields macro-F1 = 0.167 (hand-computed; comfortably below the 0.55 threshold; the D6-D8 prompt's "baseline may exceed 0.55" concern not realised under macro-F1). Classifier architecture intentionally deferred to F1 implementation (D11+) among multinomial logistic regression with L2, k-NN with k=3, shallow decision tree (depth ≤ 3). Locked as DECISION-012.
- D6 paperwork (ERRATA-012 Overleaf + biblio audit) and D7 biblio audit continuation remain deferred to D10 F0-close batch per `d6_deferred.md`.

**Tomorrow planned (Day 9, F0-D9, Sunday 2026-05-31):**
- Memory snapshot per weekly cadence. Capture F2 null + DECISION-012 F1 design lock + the §5 scope distinction.
- Light day otherwise; no compute, no manuscript edits.

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|------|-------------|-----------|-------|--------|
| Q-002 | Vast.ai C.37124280 fate | No | Jimmy | Check Vast.ai web UI when convenient |
| Q-003 | Public + MIT repo — supervisor approval | No (cheap to reverse) | Jimmy | Raise with Dr. Ho at next meeting |

Q-001 / Q-004 / Q-006 closed D2. Q-008 / Q-009 closed D3. Q-007 closed D5.

No new questions opened today.

## Canonical files (current versions)

| Topic | Current canonical file | Superseded |
|---|---|---|
| BCF 3-model statistics | `results/bcf/bcf_pooled_3model.json` | (none) |
| BCF 4-model justification | `results/bcf/bcf_pooled_results.json` | (none) |
| HPA dominance per dataset | `results/bcf_v2/hpa_v4_dominance_per_dataset.csv` | hpa_v3_dominance, hpa_simulation_v2.csv |
| HPA Alibaba grid | `results/bcf_v2/hpa_simulation_alibaba_v4.csv` | hpa_simulation_v2.csv |
| HPA Bitbrains grid | `results/bcf_v2/hpa_simulation_bitbrains_v4.csv` | (none) |
| HPA Bytedance grid | `results/bcf_v2/hpa_simulation_bytedance_v4.csv` | hpa_simulation_bytedance_v3.csv |
| Foundation leaderboard (canonical §4.7) | `results/foundation_comparison/leaderboard_v1.csv` | leaderboard_v3_wide.csv |
| NEW pool full test scope | `results/foundation_comparison/cross_dataset_headline_v2.csv` | (none) |
| R² comparison table | `comparison_table.csv` | (none) |
| Win rates timestep | `stratified_skill.csv` | (none) |
| Win rates container | `cv_stratified_skill.csv` | (none) |
| Boundary condition table | `reports/tables/boundary_condition_table_corrected.csv` (Bitbrains row NEW pool since prior session) | boundary_condition_table.csv |
| Bitbrains summary | `bitbrains_summary_corrected.csv` (OLD pool, used by Section 2 verifier) | bitbrains_summary.csv (×2) |
| NNLS production weights | `run.log` (lines tagged NNLS:) | (none) |
| Toto Alibaba results | `results/foundation_comparison/toto_k20_alibaba.json` | (none) |
| Error correlations | `results/bcf_v2/c2_*.csv` | (none) |
| LOO ablation | `results/bcf_v2/loo_ablation_new_pool_v2.csv` | (none) |
| F2 WPE per series | `phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv` (D4 outputs) | (none) |
| F2 partial-R² results | `phase_f/data/f2_partial_r2_results.csv` (D5) | (none) |
| F2 Bitbrains per-VM panel | `phase_f/data/per_series_deltas_bitbrains.csv` (D5 intermediate) | (none) |
| F2 chapter outline (D8 finalised) | `phase_f/journal/f2_chapter_outline.md` (v4 D8, 5 D11+ TBV items) | v3.1 D7 sketch |
| **F1 design lock** | `phase_f/journal/f1_prep_scope.md` (D8, DECISION-012) | (none) |
| Verifier expected anchors | `reports/tables/thesis_numbers.json` (NEW pool Bitbrains BCF as of D5) | prior OLD pool version (Q-007) |

## Pending Overleaf edits (full detail in ERRATA.md)

**1 of 12 edits pending: ERRATA-012** (Ch4 BCF Bitbrains row from OLD pool to NEW pool semantics, per DECISION-011 Q-007 Anchor A). Deferred to D10 F0-close Overleaf batch per `phase_f/journal/d6_deferred.md`. Any ERRATA-013+ rows from D10 biblio audit will also batch into the same D10 Overleaf commit. ERRATA-001 through ERRATA-011 applied 2026-05-25.

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold | Status |
|---|---|---|---|
| F1 | macro-F1 | ≥ 0.55 | design locked D8 per DECISION-012; baseline 0.167; implementation starts D11+ |
| F2 | partial-R²(WPE \| ACF@24h) | ≥ 0.30 | **BELOW (headline 0.079, Bitbrains per-VM 0.056). Null reported per DECISION-005. Chapter outline at D8 defensible-structure quality, 5 method-detail TBVs deferred to D11+.** |
| F3 | Spearman ρ | ≥ 0.6 | not yet tested |
| F3 | \|DFL−Pinball−τ\| | ≤ 5% | not yet tested |

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai instance C.37423026 | Stopped (untouched D6–D8) | Three consecutive no-compute days. statsmodels 0.14.6 + scipy 1.17.1 still installed from D4/D5. Disk preserved. |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Active | D8-close commit (planned) covers combined D6-light + D7 + D8 batch: 7 files in `phase_f/`. No off-scope edits; SYNC_PROTOCOL Revision 4 staging pattern not invoked. Branch `feature/live-demo`. |
| Dashboard repo | Separate, not synced | Unchanged |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ except data_snapshots/. |
| Overleaf | No commit D6, D7, or D8 | ERRATA-012 deferred to D10 F0-close batch per `phase_f/journal/d6_deferred.md`. No D6-D8 manuscript work. |

## Memory state

- **Slots used:** 30/30
- **Last memory snapshot:** `memory_snapshots/memory_snapshot_2026-05-23.md`
- **Next snapshot due:** Sunday 2026-05-31 (D9, weekly cadence, **tomorrow**). Snapshot should capture F2 null + DECISION-012 F1 design lock + §5 scope distinction. No `memory_user_edits` calls today.

## Update protocol

This file is refreshed at every day-close, not appended to. Replace stale sections with current state. Historical record lives in `handoffs/` and `DECISIONS.md`.