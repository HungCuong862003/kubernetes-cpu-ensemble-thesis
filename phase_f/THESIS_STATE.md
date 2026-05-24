# THESIS_STATE.md

**Last updated:** 2026-05-29 (F0-D7 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 7 of 10
- **Phase F overall:** day 7 of ~135
- **Days until defence:** ~129

## Today's headline

**Today completed (Day 7):**
- D7 produced the F2 chapter outline sketch at `phase_f/journal/f2_chapter_outline.md` (v3.1, eight sections, one-line claim + data/file anchor per section, no prose). Refined through four self-examination passes producing a 5 / 3 / 1 / 0 substantive-issue diminishing-returns curve. Final v3.1 carries seven flagged D8-open-item TBV verifications: §2 resample count, §2 Bitbrains clustering scheme, §3 CI method (percentile assertion may not transfer from BCF context), §5 cross-dataset ML-benefit ordering at h10, §5 WPE median by dataset, §5 h30 double-monotonicity question, §5 M4-literature prior citation. Each is a 5–10 min script read or web search at D8.
- D6 paperwork (ERRATA-012 Overleaf application + biblio audit resume) deferred to D10 F0-close batch per `phase_f/journal/d6_deferred.md`. Rationale: both items are paperwork (manuscript .tex edit + .bib field verification) and batch well with D10's existing Overleaf batch slot. ERRATA-012 substitution text already fully specified; D10 applies in one commit alongside any ERRATA-013+ rows accumulated.
- D7 Task A (biblio audit continuation) also deferred to D10 by symmetry — same paperwork-batches-well rationale. No Overleaf commit D6 or D7.

**Tomorrow planned (Day 8, F0-D8):**
- D8 Task A — promote F2 outline from D7 sketch to defensible structure. Clear the seven TBV open items by reading `f2_partial_r2.py` source and the `wpe_*.csv` files. Name single strongest + single weakest claim. No prose yet.
- D8 Task B — F1 prep scope, design only, no implementation. Lock feature set (drop WPE per F2 implication; argue both sides for ACF@24h, horizon_min, CV, Hurst, ACF@1h). Identify training data scope (12 cells from `bcf_pairs.csv`, per-cell label = winner foundation model from `leaderboard_v1.csv`). Identify evaluation (LOO-cell CV, macro-F1 ≥ 0.55 per DECISION-005). Baseline (always-predict-Chronos-2). Output: `phase_f/journal/f1_prep_scope.md`. Lock as DECISION-012 if scope firm.

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
| **F2 chapter outline (sketch)** | `phase_f/journal/f2_chapter_outline.md` (D7, v3.1 sketch, 7 TBV items) | (none) |
| Verifier expected anchors | `reports/tables/thesis_numbers.json` (NEW pool Bitbrains BCF as of D5) | prior OLD pool version (Q-007) |

## Pending Overleaf edits (full detail in ERRATA.md)

**1 of 12 edits pending: ERRATA-012** (Ch4 BCF Bitbrains row from OLD pool to NEW pool semantics, per DECISION-011 Q-007 Anchor A). Deferred to D10 F0-close Overleaf batch per `phase_f/journal/d6_deferred.md`. Any ERRATA-013+ rows from D10 biblio audit will also batch into the same D10 Overleaf commit. ERRATA-001 through ERRATA-011 applied 2026-05-25.

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold | Status |
|---|---|---|---|
| F1 | macro-F1 | ≥ 0.55 | not yet tested (D8 prep scope; full test starts ~D11) |
| F2 | partial-R²(WPE \| ACF@24h) | ≥ 0.30 | **BELOW (headline 0.079, Bitbrains per-VM 0.056). Null reported per DECISION-005. Chapter outline at sketch quality D7.** |
| F3 | Spearman ρ | ≥ 0.6 | not yet tested |
| F3 | \|DFL−Pinball−τ\| | ≤ 5% | not yet tested |

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai instance C.37423026 | Stopped (untouched D6, D7) | No D6 or D7 compute. statsmodels 0.14.6 + scipy 1.17.1 still installed from D4/D5. Disk preserved. |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Active | D7 commit (planned) covers D6-light + D7 journals + F2 outline + this state file. No off-scope edits; SYNC_PROTOCOL Revision 4 staging pattern not invoked. Branch `feature/live-demo`. |
| Dashboard repo | Separate, not synced | Unchanged |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ except data_snapshots/. |
| Overleaf | No commit D6 or D7 | ERRATA-012 deferred to D10 F0-close batch per `phase_f/journal/d6_deferred.md`. No D7 manuscript work (D7 produced phase_f/ artefacts only). |

## Memory state

- **Slots used:** 30/30
- **Last memory snapshot:** `memory_snapshots/memory_snapshot_2026-05-23.md`
- **Next snapshot due:** Sunday 2026-05-31 (D9, weekly cadence). No `memory_user_edits` calls today.

## Update protocol

This file is refreshed at every day-close, not appended to. Replace stale sections with current state. Historical record lives in `handoffs/` and `DECISIONS.md`.