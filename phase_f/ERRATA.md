# ERRATA.md

**Project:** kubernetes-cpu-ensemble-thesis
**Started:** 2026-05-23 (Phase F Day 1 audit)
**Update protocol:** append-only. Never delete or reorder. Status field tracks lifecycle:
- **PENDING** — error identified, not yet applied to manuscript
- **APPLIED** — correction applied in Overleaf, commit pushed
- **BLOCKED** — depends on a Q-ID resolution before it can be applied

---

## How to read this file

Each errata is numbered sequentially (ERRATA-NNN), never reused. Rows are immutable except for status updates. Corrections propagate from this file *into* the chapter, never the reverse. The submitted PDF is the baseline; this file tracks what changed since.

---

## Errata table

| # | Date opened | Location | Original (in submitted PDF) | Correction | Status | Related Q-ID / DECISION |
|---|---|---|---|---|---|---|
| 001 | 2026-05-23 | Ch3 §3.8 | "p < 0.011" | "p = 0.0097" | PENDING | DECISION-002 |
| 002 | 2026-05-23 | Ch3 §3.8 | cites `bcf_pooled_results.json` (4-model file) but reports 3-model numbers | change citation to `bcf_pooled_3model.json` | PENDING | DECISION-002 |
| 003 | 2026-05-23 | Ch3 §3.8 | "95% CI [0.70, 0.88]" (method unlabeled) | "percentile 95% CI [0.71, 0.89] (BCa degenerate)" | PENDING | DECISION-007 |
| 004 | 2026-05-23 | Ch3 §3.8 | (no footnote on CI method) | ADD footnote explaining percentile choice and BCa degeneracy (text drafted in DECISION-007) | PENDING | DECISION-007 |
| 005 | 2026-05-23 | Ch3 §3.10 | cites `hpa_simulation_v2.csv` (v1-sprint, max_replicas=100) | change to `bcf_v2/hpa_simulation_*_v4.csv` (max_replicas=1000) | PENDING | DECISION-003 |
| 006 | 2026-05-23 | Ch4 §4.8 | (no cross-reference to BCa footnote) | ADD reference to Ch3 §3.8 footnote | PENDING | DECISION-007 |
| 007 | 2026-05-23 | Ch4 §4.11 | "800 rows per dataset" | qualify: "800 rows per Alibaba/Bitbrains; 600 rows for Bytedance (3 horizons, no h10)" | PENDING | DECISION-003 |
| 008 | 2026-05-23 | Ch5 §5.1 | "CI crosses the Alibaba 30 min point" | "the Bitbrains 30 min CI lower bound (+3.54) sits 0.15 pp above the Alibaba 30 min point (+3.39) — close enough that the gap is within typical bootstrap variation across cells, but the CI itself does not cover the Alibaba point" | PENDING | — |
| 009 | 2026-05-23 | Ch5 §5.6 | "matched reactive lag-1 point" | "matched reactive lag variant" (generic; the matched-point comparison is at the same (h, τ, s), not lag-1 specifically) | PENDING | — |
| 010 | 2026-05-23 | Ch6 §6.1 + Ch4 Table 4.13 | per-horizon: 128 / 156 / 157 / 92 (from superseded `hpa_simulation_v2.csv`) | per-horizon: TBD from v4 re-derivation (total 533/640 sum unchanged) + narrative shape "rises slightly to peak at h30, then declines monotonically through h60 and h120" | BLOCKED on Q-004 | DECISION-003 |

---

## Day 1 audit summary

10 errata identified through cross-chapter audit on 2026-05-23 (Phase F Day 1). All data-verified on Vast.ai instance C.37423026 against canonical source files:

- BCF stats (ERRATA-001 through 004): verified against `bcf_pooled_3model.json` and `bcf_pooled_results.json`
- HPA file path (ERRATA-005): verified via `c3_hpa_v4_verdict.md` directing v4 as canonical
- Bytedance row count (ERRATA-007): verified by line-counting `hpa_simulation_bytedance_v4.csv` = 601 lines (600 + header)
- CI crossing claim (ERRATA-008): verified Bitbrains 30min CI = [+3.5390, +5.4170] vs Alibaba +3.39 (Alibaba sits 0.149 pp below CI lower bound)
- Lag-1 phrasing (ERRATA-009): verified via inspection of `task2_hpa_v2.py` showing matched comparison is at same (h, τ, s) across all reactive lag variants
- Per-horizon counts (ERRATA-010): submitted PDF's 128/156/157/92 from `hpa_simulation_v2.csv` (superseded); v4 re-derivation pending (Q-004)

## Batch application plan

- **Day 3 morning:** Apply ERRATA-001 through 009 in one Overleaf session (~20 minutes batch)
- **Day 3 afternoon:** After Q-004 resolves, apply ERRATA-010 with confirmed v4 numbers

When applied, status changes from PENDING (or BLOCKED) to APPLIED with date-applied appended in a new column. The row is never deleted.

---

(future errata appended below — keep ID sequence monotonic)
