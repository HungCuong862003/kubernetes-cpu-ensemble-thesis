# THESIS_STATE.md

**Last updated:** 2026-05-27 (F0-D5 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 5 of 10
- **Phase F overall:** day 5 of ~135
- **Days until defence:** ~131

## Today's headline

**Today completed (Day 5):**
- F2 pre-reg test executed: headline per-cell partial-R²(WPE | ACF, horizon) = 0.0790, CI [0.0000, 0.0790] BELOW threshold; Bitbrains per-VM robustness partial-R² = 0.0558, CI [0.0007, 0.2126] BELOW threshold. F2 reports null per DECISION-005 (honest below-threshold reporting, no post-hoc adjustment). Structural diagnosis: R²_reduced (ACF + horizon, no WPE) = 0.903 — ACF@24h + horizon already explain 90% of cell-level delta_pp variance; WPE has nothing left to predict on this corpus. WPE coefficient positive cross-dataset (+3.45 headline), inverting the M4-literature prior — WPE and ACF@24h are partial substitutes on this corpus, not complements.
- Q-007 closed via DECISION-011 (Anchor A). 3-key update to `reports/tables/thesis_numbers.json` aligned the verifier's expected values to NEW pool per-VM medians. Verifier audit cleared 211/3 → 214/0. CSV side (`boundary_condition_table_corrected.csv`) had already been migrated to NEW pool by a prior session; only the JSON side was stale. D4 journal's "computed live by verifier" diagnosis was wrong — the BCF check is a static CSV-vs-JSON comparison in `check_section_11_boundary`.
- ByteDance stats duplicate diff resolved: byte-identical (md5 `a5e25823a28e10c6970c46e022e32ad4`). D4 TBD closed.
- DECISION-010 (F2 framing) and DECISION-011 (Q-007 Anchor A) appended to `DECISIONS.md`.
- ERRATA-012 added for Ch4 BCF Bitbrains row reframing (NEW pool semantics in manuscript); PENDING D6+ Overleaf application.

**Tomorrow planned (Day 6, F0-D6):**
- Apply ERRATA-012 in Overleaf early in the day. Manuscript Ch4 BCF prose + Table 4.10 Bitbrains row update from OLD pool to NEW pool semantics. Verdict prose grows from one sentence to ~3 sentences disclosing the per-VM win rates (61%/35% at h30/h120).
- Resume bibliography audit (deferred from D4, deferred again at D5). Cover remaining `.bib` entries per `phase_f/journal/biblio_audit_d3_notes.md` schema.
- Optional / time-permitting: draft a `phase_f/scripts/phase_f_session_start.sh` wrapper that pulls state files from Drive at session-start (prevents recurrence of today's session-start step-3 miss).
- Optional / time-permitting: begin F2 chapter outline sketch — narrative anchored on "ACF@24h saturates cell-level predictability" + "WPE-ACF as partial substitutes on cloud traces, not complements."

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|------|-------------|-----------|-------|--------|
| Q-002 | Vast.ai C.37124280 fate | No | Jimmy | Check Vast.ai web UI when convenient |
| Q-003 | Public + MIT repo — supervisor approval | No (cheap to reverse) | Jimmy | Raise with Dr. Ho at next meeting |

Q-001 / Q-004 / Q-006 closed D2. Q-008 / Q-009 closed D3. **Q-007 closed D5**.

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
| **F2 WPE per series** | `phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv` (D4 outputs) | (none) |
| **F2 partial-R² results** | `phase_f/data/f2_partial_r2_results.csv` (D5) | (none) |
| **F2 Bitbrains per-VM panel** | `phase_f/data/per_series_deltas_bitbrains.csv` (D5 intermediate) | (none) |
| Verifier expected anchors | `reports/tables/thesis_numbers.json` (NEW pool Bitbrains BCF as of D5) | prior OLD pool version (Q-007) |

## Pending Overleaf edits (full detail in ERRATA.md)

**1 of 12 edits pending: ERRATA-012** (Ch4 BCF Bitbrains row from OLD pool to NEW pool semantics, per DECISION-011 Q-007 Anchor A). ERRATA-001 through ERRATA-011 applied 2026-05-25.

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold | Status |
|---|---|---|---|
| F1 | macro-F1 | ≥ 0.55 | not yet tested (starts ~D11) |
| F2 | partial-R²(WPE \| ACF@24h) | ≥ 0.3 | **BELOW (headline 0.079, Bitbrains per-VM 0.056). Null reported per DECISION-005.** |
| F3 | Spearman ρ | ≥ 0.6 | not yet tested |
| F3 | \|DFL−Pinball−τ\| | ≤ 5% | not yet tested |

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai instance C.37423026 | Stopped (EOD) | Started today for F2 + Q-007. statsmodels 0.14.6 newly installed in `/venv/main/`. `/mnt/project` symlink tree intact. |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Active | Today's commit: D5 close, includes off-scope `reports/tables/thesis_numbers.json` (Q-007 fix) per SYNC_PROTOCOL Revision 4. Branch `feature/live-demo`. |
| Dashboard repo | Separate, not synced | Unchanged |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ except data_snapshots/. |
| Overleaf | No commit today | Casual mode; ERRATA-012 queued for D6 batch |

## Memory state

- **Slots used:** 30/30
- **Last memory snapshot:** `memory_snapshots/memory_snapshot_2026-05-23.md`
- **Next snapshot due:** Sunday 2026-05-31 (weekly cadence). No `memory_user_edits` calls today.

## Update protocol

This file is refreshed at every day-close, not appended to. Replace stale sections with current state. Historical record lives in `handoffs/` and `DECISIONS.md`.
