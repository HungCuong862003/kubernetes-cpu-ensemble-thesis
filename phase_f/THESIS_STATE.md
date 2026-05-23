# THESIS_STATE.md

**Last updated:** 2026-05-24 (F0-D2 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 2 of 10
- **Phase F overall:** day 2 of ~135
- **Days until defence:** ~134

## Today's headline

**Today completed (Day 2):**
- Q-001 closed: verify_foundation re-run on Vast.ai C.37423026 → 211 PASS / 3 FAIL confirms memory #24; D1-close handoff's 184/3 was a stale intermediate. The 3 FAILs are stale verifier anchors (OLD pool expected, NEW pool actual data) — opens Q-007 as maintenance task.
- Q-004 closed: canonical `hpa_v4_dominance_per_dataset.csv` read directly via Path A; per-horizon counts confirm memory exactly (Alibaba 40/35/15/1, Bitbrains 40/0/0/0, Bytedance 33/32/29). Grand totals 225 ML-strict cells / 440 cells, 675 reactive-dominated / 1760 pairs.
- Q-006 closed by dependency on Q-004: no v1 HPA count survived in memory that needed updating.
- convert_toto_json_to_npz.py drafted, ran clean on all 3 datasets; Alibaba R² anchors match memory #23 exactly to 4 decimal places (0.9173/0.8513/0.8205/0.7586, n≈98k/horizon); 3 npz files synced to Drive.
- 3 incidental findings: Bytedance Toto h10 exists (R²=0.9234, n=1860 — contradicts d1-close planning note); `passes_gate=False` for ALL 11 cells in v4 dominance file; my Pareto dominance definition disagrees with the canonical script's definition (v1+v2 rederive scripts archived in phase_f/scripts/ as wrong-but-reference).
- 3 new questions opened: Q-007 (stale verifier anchors), Q-008 (dominance+gate definitions in task_c3_hpa_rebuild_v4.py), Q-009 (manuscript framing of `passes_gate=False`).
- grep narrowed Q-008's candidate scripts to `task_c3_hpa_rebuild_v4.py` (primary) and `task_c3_hpa_rebuild_bitbrains_bytedance.py` (partial); README at `results/bcf_v2/README_HPA_CANONICAL.md` as Day 3 starting point.

**Tomorrow planned (Day 3, F0-D3):**
- Morning (~30 min): read `results/bcf_v2/README_HPA_CANONICAL.md` + `task_c3_hpa_rebuild_v4.py` → extract dominance + gate definitions → close Q-008 → decide Q-009 / ERRATA-011
- Mid-morning (~50 min): Overleaf batch apply ERRATA-001..010 (plus 011 if opened); ERRATA-010 with §6.1 narrative rewrite (monotonic decline) and `passes_gate=False` footnote
- Afternoon: per DECISION-004 reinvestment, bibliography audit begins (Ch1–Ch6 citations against memory's "Key bibliography corrections" guidance)

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|---|---|---|---|---|
| Q-001 | verify_foundation 184/3 vs memory #24's 211/3 | RESOLVED 2026-05-24 → 211/3 (memory #24 confirmed) | — | — |
| Q-002 | Vast.ai C.37124280 fate (destroyed/stopped/unknown) | No | Jimmy | Check Vast.ai web UI history when convenient |
| Q-003 | Public + MIT repo — supervisor approval status | No (cheap to reverse) | Jimmy | Raise with Dr. Ho at next meeting |
| Q-004 | V4 per-horizon HPA dominance counts | RESOLVED 2026-05-24 → canonical file matches memory exactly | — | — |
| Q-005 | F0 calendar reinvestment path | RESOLVED 2026-05-23 → path (a) | — | — |
| Q-006 | Memory #17 update with v4 per-horizon | RESOLVED 2026-05-24 → no v1 HPA count in memory to update | — | — |
| Q-007 | verify_foundation.py encodes 3 stale BCF Bitbrains anchors (OLD pool expected, NEW pool actual); update literals to clear audit to 214/0 | No (low priority) | Jimmy | Day 3 afternoon or later |
| Q-008 | What dominance + gate definitions does `task_c3_hpa_rebuild_v4.py` actually use? Candidate file located via grep; my Pareto re-derivation disagrees with canonical counts | Yes (blocks ERRATA-011 decision) | Claude + Jimmy | Read script Day 3 morning |
| Q-009 | Does `passes_gate=False` for all 11 cells require manuscript-level disclosure beyond ERRATA-010 (i.e. new ERRATA-011)? | Yes (blocks Day 3 Overleaf batch for §6.1 rewrite) | Claude + Jimmy | Decide after Q-008 closes |

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
| Toto Alibaba results | `results/foundation_comparison/toto_k20_alibaba.json` + `_npz.npz` (D2) | (none) |
| Toto Bitbrains results | `results/foundation_comparison/toto_k20_bitbrains.json` + `_npz.npz` (D2) | (none) |
| Toto Bytedance results | `results/foundation_comparison/toto_k20_bytedance.json` + `_npz.npz` (D2) | (none) |
| Error correlations | `results/bcf_v2/c2_*.csv` | (none) |
| LOO ablation | `results/bcf_v2/loo_ablation_new_pool_v2.csv` | (missing — Q deferred) |

## Pending Overleaf edits (full detail in ERRATA.md)

10 edits queued from Day 1 audit, 0 applied. Status: ERRATA-010 now PENDING (Q-004 closed); ERRATA-011 may be opened Day 3 morning pending Q-008/Q-009. Batch plan: Day 3 mid-morning after Q-008 closes (~50 min combined).

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
| Vast.ai instance C.37423026 | Running (D2 work complete; stop before EOD to preserve disk) | /mnt/project symlink tree intact; D2 scripts at phase_f/scripts/ |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | github.com/HungCuong862003/kubernetes-cpu-ensemble-thesis | Public MIT (pending Q-003) |
| Dashboard repo | Separate, not synced | github.com/HungCuong862003/hpa-thesis-dashboard, deployed at hpa-thesis-hcuong.streamlit.app |
| Drive: phase_f/scripts/ | Synced D2 | 3 .py files (convert_toto, rederive v1, rederive v2) |
| Drive: phase_f/logs/ | Q-001 log synced; Q-004 + Q-004-v2 + d2_toto_npz logs pending | 3 logs need rclone before EOD |
| Drive: results/foundation_comparison/*_npz.npz | Synced D2 | 3 Toto npz files |

## Memory state

- **Slots used:** 30/30
- **Last memory snapshot:** memory_snapshots/memory_snapshot_2026-05-23.md
- **Next snapshot due:** Sunday 2026-05-31 (weekly cadence) — memory_user_edits NOT called today; D2 snapshot skipped despite 2026-05-24 being a Sunday because last snapshot is <24h old and cadence is weekly

## Update protocol

This file is **refreshed at every day-close**, not appended to. Replace stale sections with current state. Historical record lives in handoffs/ and DECISIONS.md.