# THESIS_STATE.md

**Last updated:** 2026-05-25 (F0-D3 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 3 of 10
- **Phase F overall:** day 3 of ~135
- **Days until defence:** ~133

## Today's headline

**Today completed (Day 3):**
- Q-008 closed: extracted dominance + gate definitions from `task_c3_hpa_rebuild_v4.py` and `README_HPA_CANONICAL.md`. Two distinct dominance metrics — `reactive_dominated_pct` (out of 160 reactive points per cell) and `ml_strict_dominance_pct` (out of 40 ML configurations per cell). Gate at line 279: `passes_gate = (reactive_dominated_pct >= 80%)`. All eleven cells fail the gate; highest single-cell rate is ByteDance h30 at 78.12%.
- Q-009 closed: ERRATA-011 opened (Branch 1 — gate is a pre-registered production-readiness criterion, not exploratory diagnostic).
- DECISION-008 logged: full retirement of the 533/640 headline figure (not just per-horizon swap) with dual-metric (reactive_dominated + ml_strict_dominance) framing in the substitution.
- ERRATA-001 through 010 applied to Overleaf in a single batched commit; ERRATA-011 disclosure embedded in the ERRATA-010 substitution paragraph (no separate footnote).
- Ch4 Table 4.13 substituted with dual-metric per-cell format plus gate column.
- ERRATA.md updated: D2 update applied (ERRATA-010 BLOCKED → PENDING then APPLIED), Day 2 audit note section added at bottom, ERRATA-011 row added, schema extended with `Applied` column, all 11 rows transitioned PENDING → APPLIED with date `2026-05-25`.
- Bibliography audit begun per DECISION-004 reinvestment: ~40 entries inventoried in `phase_f/journal/biblio_audit_d3_notes.md`. *[Adjust count to actual before commit.]*

**Tomorrow planned (Day 4, F0-D4):**
- Continue bibliography audit; close out remaining .bib entries.
- Apply `.bib` corrections for the FABRICATED_REPLACE_ENTRY cluster.
- Queue prose-level errata (ERRATA-012+) for any citation whose correction changes meaning in-chapter.
- Optional / time-permitting: Q-007 verifier anchor maintenance (would clear audit to 214/0).

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|---|---|---|---|---|
| Q-002 | Vast.ai C.37124280 fate | No | Jimmy | Check Vast.ai web UI when convenient |
| Q-003 | Public + MIT repo — supervisor approval | No (cheap to reverse) | Jimmy | Raise with Dr. Ho at next meeting |
| Q-007 | verify_foundation 3 stale BCF Bitbrains anchors | No | Jimmy | Day 4 afternoon or later |

Q-001 / Q-004 / Q-006 closed D2. Q-008 / Q-009 closed D3.

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
| Boundary condition table | `boundary_condition_table_corrected.csv` | boundary_condition_table.csv |
| Bitbrains summary | `bitbrains_summary_corrected.csv` | bitbrains_summary.csv (×2) |
| NNLS production weights | `run.log` (lines tagged NNLS:) | (none) |
| Toto Alibaba results | `results/foundation_comparison/toto_k20_alibaba.json` | (none) |
| Error correlations | `results/bcf_v2/c2_*.csv` | (none) |
| LOO ablation | `results/bcf_v2/loo_ablation_new_pool_v2.csv` | (missing — Q deferred) |

## Pending Overleaf edits (full detail in ERRATA.md)

11 of 11 edits applied. No edits pending. Next manuscript edits will come from the bibliography audit; queued as ERRATA-012+ if biblio findings require in-prose changes rather than `.bib`-only fixes.

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold |
|---|---|---|
| F1 | macro-F1 | ≥ 0.55 |
| F2 | partial-R²(WPE \| ACF@24h) | ≥ 0.3 |
| F3 | Spearman ρ | ≥ 0.6 |
| F3 | \|DFL−Pinball−τ\| | ≤ 5% |

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai instance C.37423026 | Stopped (EOD) | Started for Q-008 SSH read; stopped after `hpa_v4_dominance_per_dataset.csv` paste returned. `/mnt/project` symlink tree intact. |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Active | Today's commits: ERRATA.md, DECISIONS.md, handoffs/2026-05-25_d3-close.md, phase_f/journal/biblio_audit_d3_notes.md, THESIS_STATE.md refresh |
| Dashboard repo | Separate, not synced | Unchanged |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ except data_snapshots/ |
| Overleaf | 1 commit today | "Apply ERRATA-001..011 — see ERRATA.md for source-of-truth substitutions, Phase F D3 (2026-05-25)" |

## Memory state

- **Slots used:** 30/30
- **Last memory snapshot:** memory_snapshots/memory_snapshot_2026-05-23.md
- **Next snapshot due:** Sunday 2026-05-31 (weekly cadence). No `memory_user_edits` calls today.

## Update protocol

This file is refreshed at every day-close, not appended to. Replace stale sections with current state. Historical record lives in handoffs/ and DECISIONS.md.