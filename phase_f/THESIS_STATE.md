# THESIS_STATE.md

**Last updated:** 2026-06-01 (F0-D10 transition → F1 implementation start)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F1 (router implementation start D11+). F0 work product complete; F0 manuscript application carries to F1+ paperwork slot.
- **Phase F overall:** day 10 of ~135
- **Days until defence:** ~126

## F0 transition — what F0 produced

F0 lockdown (D1-D10) work product is complete. Manuscript propagation is partially complete (11 of 12 ERRATA applied; ERRATA-012 + biblio audit deferred to F1+ batched paperwork slot).

**Done at F0:**
- 12 DECISIONS logged (DECISION-001 through DECISION-012)
- 12 ERRATA opened. 11 APPLIED at D3 batch. 1 PENDING (ERRATA-012, deferred to F1+ Overleaf slot)
- F2 pre-registered test executed → NULL reported per DECISION-005 (headline partial-R² 0.0790, Bitbrains per-VM 0.0558; both below 0.30; R²_reduced = 0.903)
- F1 router design locked at DECISION-012 (4 features, LOO-cell CV, macro-F1 ≥ 0.55, baseline 0.167)
- F2 chapter outline at v4 with 5 method-detail TBVs deferred to D11+
- Verifier audit cleared at 214/0 (D5)
- Two memory snapshots written (2026-05-23, 2026-05-31)

**Deferred to F1+:**
- ERRATA-012 Overleaf application (Ch4 §4.X prose + Table 4.10 Bitbrains row OLD→NEW pool per-VM medians; substitution text fully specified)
- Bibliography audit per-entry verification (schema set up D3; bonus scoping hint via _citation_keys_used.txt / _unused_bib_keys.txt at repo root)

## Today's headline

**Today completed (Day 10):**
- F0 transition state file refresh (this file).
- Biblio audit formally deferred to F1+ via deferral section appended to `phase_f/journal/biblio_audit_d3_notes.md`. F1+ work order specified, scope bounded by repo-root citation key files.
- ERRATA-012 Overleaf application deferred to F1+ paperwork slot, batched with biblio audit findings.
- D6-D10 combined handoff written at `phase_f/handoffs/2026-06-01_d10-close.md` for audit consistency.

**Tomorrow planned (Day 11, F1-D1, Tuesday 2026-06-02) — F1 implementation start:**
- Inspect feature matrix properties: 12 cells × 4 features (ACF@24h, horizon_min, CV, ACF@1h). Variance per feature, correlation matrix, rank check.
- Pick classifier architecture among DECISION-012 defaults: multinomial logistic regression with L2 / k-NN k=3 / shallow decision tree depth ≤ 3.
- Build feature matrix from `omega_summary.csv` dataset-medians + horizon-from-cell.
- Build label vector: per-cell winner from `leaderboard_v1.csv` (Chronos-2:6, TimesFM:3, Granite-TTM:2, NNLS:1).
- Implement LOO-cell CV loop; compute per-class F1 and macro-F1 against pre-reg threshold 0.55; report uplift over baseline 0.167.
- Output expectations: `phase_f/scripts/f1_router.py`, `phase_f/data/f1_router_predictions.csv`, `phase_f/data/f1_router_results.csv`, `phase_f/journal/d11_f1_implementation.md`.

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|------|-------------|-----------|-------|--------|
| Q-002 | Vast.ai C.37124280 fate | No | Jimmy | Check Vast.ai web UI when convenient |
| Q-003 | Public + MIT repo — supervisor approval | No (cheap to reverse) | Jimmy | Raise with Dr. Ho at next meeting |

Q-001 / Q-004 / Q-006 closed D2. Q-008 / Q-009 closed D3. Q-007 closed D5.

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold | Status |
|---|---|---|---|
| F1 | macro-F1 | ≥ 0.55 | implementation starts tomorrow D11; design locked D8 per DECISION-012 |
| F2 | partial-R²(WPE \| ACF@24h) | ≥ 0.30 | **BELOW (headline 0.079, Bitbrains per-VM 0.056). Null reported per DECISION-005. Chapter outline at v4, 5 method-detail TBVs deferred to D11+.** |
| F3 | Spearman ρ | ≥ 0.6 | not yet tested |
| F3 | \|DFL−Pinball−τ\| | ≤ 5% | not yet tested |

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai instance C.37423026 | Stopped (untouched D6–D10) | Five consecutive no-compute days. May restart D11+ depending on F1 classifier architecture choice; for the named small-n defaults local pandas/scikit-learn suffices. statsmodels 0.14.6 + scipy 1.17.1 still installed. Disk preserved. |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Active | D10-close commit covers `phase_f/journal/d10_f0_transition.md` + `biblio_audit_d3_notes.md` update + THESIS_STATE.md refresh + handoff. Branch `feature/live-demo`. Working tree carries substantial off-scope demo-branch WIP; surgical adds keep our commit clean. |
| Dashboard repo | Separate, not synced | Unchanged |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ |
| Overleaf | No commit today | ERRATA-012 deferred to F1+ paperwork slot. No D6-D10 Overleaf commits. |

## Memory state

- **Slots used:** 30/30 at D9 start → cycles freed after D9 snapshot landed. D10 produced no new substantive memory-worthy facts (Overleaf deferral is process, not content).
- **Last memory snapshot:** `memory_snapshots/memory_snapshot_2026-05-31.md` (D9).
- **Next snapshot due:** Sunday 2026-06-07 (D16, weekly cadence; first F1-implementation snapshot).

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
| Boundary condition table | `reports/tables/boundary_condition_table_corrected.csv` (Bitbrains row NEW pool) | boundary_condition_table.csv |
| Bitbrains summary | `bitbrains_summary_corrected.csv` (OLD pool, used by Section 2 verifier) | bitbrains_summary.csv (×2) |
| NNLS production weights | `run.log` (lines tagged NNLS:) | (none) |
| Toto Alibaba results | `results/foundation_comparison/toto_k20_alibaba.json` | (none) |
| Error correlations | `results/bcf_v2/c2_*.csv` | (none) |
| LOO ablation | `results/bcf_v2/loo_ablation_new_pool_v2.csv` | (none) |
| F2 WPE per series | `phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv` (D4 outputs) | (none) |
| F2 partial-R² results | `phase_f/data/f2_partial_r2_results.csv` (D5) | (none) |
| F2 Bitbrains per-VM panel | `phase_f/data/per_series_deltas_bitbrains.csv` (D5 intermediate) | (none) |
| F2 chapter outline (D8 finalised) | `phase_f/journal/f2_chapter_outline.md` (v4 D8, 5 D11+ TBV items) | v3.1 D7 sketch |
| F1 design lock | `phase_f/journal/f1_prep_scope.md` (D8, DECISION-012) | (none) |
| Verifier expected anchors | `reports/tables/thesis_numbers.json` (NEW pool Bitbrains BCF as of D5) | prior OLD pool version (Q-007) |
| Biblio audit schema | `phase_f/journal/biblio_audit_d3_notes.md` (schema + D10 deferral; F1+ work order specified) | (none) |
| Memory snapshots | `phase_f/memory_snapshots/memory_snapshot_2026-05-31.md` (D9, latest) | 2026-05-23 (historical reference) |

## Pending Overleaf edits (full detail in ERRATA.md)

**1 of 12 edits pending: ERRATA-012** (Ch4 BCF Bitbrains row OLD→NEW pool semantics per DECISION-011 Q-007 Anchor A). Deferred to F1+ paperwork slot, batched with biblio audit findings.

## Update protocol

This file is refreshed at every day-close, not appended to. Replace stale sections with current state. Historical record lives in `handoffs/` and `DECISIONS.md`.