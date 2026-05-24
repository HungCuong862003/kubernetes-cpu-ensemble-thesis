# THESIS_STATE.md

**Last updated:** 2026-05-31 (F0-D9 close)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F0 (lockdown opening), day 9 of 10
- **Phase F overall:** day 9 of ~135
- **Days until defence:** ~127

## Today's headline

**Today completed (Day 9):**
- Weekly memory snapshot per F0 cadence: `phase_f/memory_snapshots/memory_snapshot_2026-05-31.md` written, covering deltas D2-D9 (8 days) since the 2026-05-23 baseline snapshot. Starting point was the pre-drafted version produced at D8 EOD; three substantive revisions applied: (1) Alibaba h10 delta tuple +0.26→+0.25 to match `comparison_table.csv`; (2) D3 biblio audit narrative re-framed from implied-in-progress to schema-only; (3) D9 lesson #6 added on schema-only progress not equating to audit progress.
- State read for D10 prep: confirmed ERRATA-012 PENDING, all other 11 ERRATA APPLIED, 12 DECISIONS logged, F2 null + DECISION-012 F1 design lock both reflected in canonical files, BCF backbone + HPA v4 framing untouched D6-D9, userMemories 30/30 at D9 start (will free cycles once snapshot landed).

**Tomorrow planned (Day 10, F0-D10, Monday 2026-06-01) — F0 close batch:**
- Task A: Apply ERRATA-012 to Overleaf. Source-of-truth substitution text in ERRATA-012 Correction column + DECISION-011 anchor. Ch4 §4.X prose + Table 4.10 Bitbrains row.
- Task B: Biblio audit decision. Current state is schema-only at `biblio_audit_d3_notes.md`. Honest options: (a) drive paste-by-paste verification at D10 (several hours, may not finish); (b) defer audit work to F1+ batch with explicit "biblio audit deferred" framing. Half-finished is the worst option.
- Task C: F0 close summary + F0→F1 transition state file refresh. Optional D6-D10 combined handoff.

## Active open questions

| Q-ID | Description | Blocking? | Owner | Action |
|------|-------------|-----------|-------|--------|
| Q-002 | Vast.ai C.37124280 fate | No | Jimmy | Check Vast.ai web UI when convenient |
| Q-003 | Public + MIT repo — supervisor approval | No (cheap to reverse) | Jimmy | Raise with Dr. Ho at next meeting |

Q-001 / Q-004 / Q-006 closed D2. Q-008 / Q-009 closed D3. Q-007 closed D5.

No new questions opened today.

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold | Status |
|---|---|---|---|
| F1 | macro-F1 | ≥ 0.55 | design locked D8 per DECISION-012; baseline 0.167; implementation starts D11+ |
| F2 | partial-R²(WPE \| ACF@24h) | ≥ 0.30 | **BELOW (headline 0.079, Bitbrains per-VM 0.056). Null reported per DECISION-005. Chapter outline at v4, 5 method-detail TBVs deferred to D11+.** |
| F3 | Spearman ρ | ≥ 0.6 | not yet tested |
| F3 | \|DFL−Pinball−τ\| | ≤ 5% | not yet tested |

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai instance C.37423026 | Stopped (untouched D6–D9) | Four consecutive no-compute days. statsmodels 0.14.6 + scipy 1.17.1 still installed from D4/D5. Disk preserved. |
| Vast.ai rclone (gdrive:) | Configured | Service-account JSON |
| Local Windows rclone (gdrive:) | Configured | OAuth, working |
| Thesis git repo | Active | D9-close commit covers `phase_f/memory_snapshots/memory_snapshot_2026-05-31.md` + journal + state file refresh. Branch `feature/live-demo`. |
| Dashboard repo | Separate, not synced | Unchanged |
| Drive: phase_f/ | Synced | Mirror of git phase_f/ except data_snapshots/. |
| Overleaf | No commit D6, D7, D8, D9 | ERRATA-012 still PENDING; D10 application target. |

## Memory state

- **Slots used:** 30/30 at D9 start; snapshot written → cycles freed for D10+ updates.
- **Last memory snapshot:** `memory_snapshots/memory_snapshot_2026-05-31.md` (written today).
- **Next snapshot due:** Sunday 2026-06-07 (D16, weekly cadence; first F1-implementation snapshot).

## Update protocol

This file is refreshed at every day-close, not appended to. Replace stale sections with current state. Historical record lives in `handoffs/` and `DECISIONS.md`.