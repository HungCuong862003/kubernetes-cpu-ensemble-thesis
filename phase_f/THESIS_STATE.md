# THESIS_STATE.md

**Last updated:** 2026-05-26 (F0-D4 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 4 of 10
- **Phase F overall:** day 4 of ~135
- **Days until defence:** ~132

## Today's headline

**Today completed (Day 4):**
- F2 prep: WPE (Bandt-Pompe weighted PE per Fadlallah et al. 2013, m=4, tau=1) computed for all 3 datasets — Alibaba (5000 containers), Bitbrains (142 VMs), ByteDance (93 instances). Reads `data/processed/<ds>/{train,val,test}.parquet` and concatenates train+val+test per series. Outputs at `phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv`.
- WPE distribution: Alibaba lowest (median 0.567), Bitbrains middle (0.728), ByteDance highest (0.955). Note ByteDance is at 10-min cadence while the others are at 5-min, so motif covers different time scales cross-dataset.
- Pearson + Spearman sanity check vs ACF@24h, CV, Hurst. Headline: WPE vs ACF@24h is POSITIVE in all 3 datasets (not negative as initially predicted). WPE measures short-scale (20–40 min) ordinal regularity; ACF@24h measures long-scale periodicity. Orthogonality is what F2's partial-R² target needs.
- Within-Bitbrains WPE-ACF Spearman rho=+0.7553 — genuinely strong, not a Pearson artefact (Pearson +0.87 was only modestly inflated from rank-based +0.76). WPE adds little beyond ACF@24h on Bitbrains specifically. Alibaba (+0.12) and ByteDance (+0.28) show near-orthogonality, encouraging for F2.
- 3 stranded F0 D1 verify logs (`F0_D1_verify_{post-optionC,post-optionC-v2,pre-sync}.log`) caught up to Drive — they never made it during D1 close.
- Q-007 reopened — earlier "stale literal" diagnosis was wrong. Actual situation is a metric-scope mismatch: verifier expects OLD pool per-VM medians (-5.46, +3.30); data is NEW pool per-VM medians (-16.66, +1.53, -1.99, -2.81). Proper rewrite is 4 literal updates + verdict text rewrite, ~45–60 min, deferred to D5.
- Bibliography audit deferred — user pivoted away from manuscript work today.
- Casual journal entry at `phase_f/journal/d4_wpe_prep.md`. No formal D4 handoff written (casual mode; journal serves the purpose).
- Commit `843c16f` on `feature/live-demo` — 11 files, 6646 insertions.

**Tomorrow planned (Day 5, F0-D5):**
- F2 partial-R²(WPE | ACF@24h) regression against `delta_pp` from `bcf_pairs.csv`. Pre-reg threshold ≥0.3. Decision needed at execution time: report (a) pooled across all 11 cells, (b) per-dataset, or (c) pooled with dataset fixed effects. Bitbrains' high within-dataset WPE-ACF correlation will pull pooled estimate down; honest disclosure preferred if pooled fails while per-dataset succeeds.
- Q-007 proper rewrite — 4 literal updates (h10/h30/h60/h120) + verdict text change in `verify_foundation.py`. Possibly also regenerate `bitbrains_summary_corrected.csv` under NEW pool semantics.
- Optional / time-permitting: resume biblio audit, or start F1 router preliminary work (router needs WPE + ACF@24h + delta_pp features all in place; D5 F2 work generates the feature matrix F1 will use).

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|---|---|---|---|---|
| Q-002 | Vast.ai C.37124280 fate | No | Jimmy | Check Vast.ai web UI when convenient |
| Q-003 | Public + MIT repo — supervisor approval | No (cheap to reverse) | Jimmy | Raise with Dr. Ho at next meeting |
| Q-007 | verify_foundation Bitbrains BCF anchors — REOPENED with broader scope (metric migration, not literal swap) | No | Jimmy | D5 — 4-literal + verdict rewrite |

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
| **WPE per-series (F2 prep)** | **`phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv`** | **(new D4)** |

## Pending Overleaf edits (full detail in ERRATA.md)

11 of 11 edits applied. No edits pending. Next manuscript edits will come from the bibliography audit (deferred) or D5+ Q-007 work if the metric-scope rewrite changes any in-prose claim about Bitbrains per-VM behaviour.

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
| Vast.ai instance C.37423026 | Stopped (EOD) | Used today for WPE compute on all 3 datasets (~3 min total). `scipy 1.17.1` newly installed in `/venv/main/`. `/mnt/project` symlink tree intact. |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Active | Today's commit: `843c16f` (Phase F Day 4 — F2 WPE prep). 11 files / 6646 insertions. |
| Dashboard repo | Separate, not synced | Unchanged |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ except data_snapshots/. `__pycache__/` purged from scripts/ at EOD. |
| Overleaf | No commit today | Casual mode, no manuscript edits |

## Memory state

- **Slots used:** 30/30
- **Last memory snapshot:** `memory_snapshots/memory_snapshot_2026-05-23.md`
- **Next snapshot due:** Sunday 2026-05-31 (weekly cadence). No `memory_user_edits` calls today.

## Update protocol

This file is refreshed at every day-close, not appended to. Replace stale sections with current state. Historical record lives in `handoffs/` and `DECISIONS.md`.