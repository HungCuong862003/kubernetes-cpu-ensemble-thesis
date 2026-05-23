# THESIS_STATE.md

**Last updated:** 2026-05-23 (F0-D1 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 1 of 10
- **Phase F overall:** day 1 of ~135
- **Days until defence:** ~135

## Today's headline

**Today completed (Day 1):**
- verify_foundation audit on Vast.ai (184 PASS / 3 FAIL reported; conflicts with memory #24's 211/3 — Q-001)
- Cross-chapter audit Ch3/Ch4/Ch5/Ch6 — 10 corrections data-verified
- DECISION-007 closed UNUSED (submitted PDF BCa grep empty)
- Phase F Drive structure live at gdrive:.../phase_f/
- Discovered dormant thesis git repo at github.com/HungCuong862003/kubernetes-cpu-ensemble-thesis

**Tomorrow planned (Day 2, F0-D2):**
- Re-run verify_foundation to settle Q-001 (~5 min)
- Re-derive v4 per-horizon HPA dominance counts to settle Q-004 (~30 min)
- Draft and run convert_toto_json_to_npz.py (main task)

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|---|---|---|---|---|
| Q-001 | verify_foundation 184/3 vs memory #24's 211/3 — which is current? | No | Jimmy | Re-run on Vast.ai C.37423026 tomorrow morning |
| Q-002 | Vast.ai C.37124280 fate (destroyed/stopped/unknown) | No | Jimmy | Check Vast.ai web UI history when convenient |
| Q-003 | Public + MIT repo — supervisor approval status | No (cheap to reverse) | Jimmy | Make private tonight; raise with Dr. Ho at next meeting |
| Q-004 | V4 per-horizon HPA dominance counts re-derivation | Yes (blocks ERRATA-010) | Claude + Jimmy | Re-derive from hpa_simulation_alibaba_v4.csv tomorrow |
| Q-005 | F0 calendar reinvestment path | RESOLVED 2026-05-23 → path (a) | — | — |
| Q-006 | Memory #17 update with v4 per-horizon (depends on Q-004) | No | — | Update after Q-004 resolves |

## Canonical files (current versions)

These are the file paths the manuscript cites. Use these names in every prompt; ignore older versions.

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

10 edits queued from Day 1 audit, 0 applied. Status: blocked on Q-004 for ERRATA-010 (Ch6 per-horizon counts); others can apply independently. Batch plan: Day 3 morning.

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
| Vast.ai instance C.37423026 | Stopped (overnight) | Restart via web UI before Day 2 work; /mnt/project symlink tree intact |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Cloned to E:\thesis (tonight) | github.com/HungCuong862003/kubernetes-cpu-ensemble-thesis, public MIT (pending Q-003) |
| Dashboard repo | Separate, not synced | github.com/HungCuong862003/hpa-thesis-dashboard, deployed at hpa-thesis-hcuong.streamlit.app |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ except data_snapshots/ |

## Memory state

- **Slots used:** 30/30
- **Last memory snapshot:** memory_snapshots/memory_snapshot_2026-05-23.md
- **Next snapshot due:** Sunday 2026-05-31 (weekly cadence) or sooner if memory is modified

## Update protocol

This file is **refreshed at every day-close**, not appended to. Replace stale sections with current state. Historical record lives in handoffs/ and DECISIONS.md.
