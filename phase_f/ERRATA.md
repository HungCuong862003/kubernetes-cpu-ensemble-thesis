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
| 010 | 2026-05-23 (opened); 2026-05-24 (unblocked) | Ch6 §6.1 + Ch4 Table 4.13 | per-horizon: 128 / 156 / 157 / 92 (from superseded `hpa_simulation_v2.csv`, v1-sprint per-(cell × lag) scope, sum 533/640) | per-horizon Alibaba ML strict-dominance cells under v4 (denominator 40 per horizon): **40 / 35 / 15 / 1** (sum 91/160). Source: `results/bcf_v2/hpa_v4_dominance_per_dataset.csv` column `n_ml_strictly_dominant`. 11-cell grand total: 225 ML-strict cells / 440 cells. Submitted PDF's narrative "rises slightly to peak at h30, then declines monotonically through h60 and h120" is **false under v4** — pattern is monotonic decline from h10 through h120. Drop the prior errata-draft claim "total 533/640 sum unchanged" (it was wrong: v1 and v4 sums index different scopes). Note: all 11 cells have `passes_gate=False` in the canonical file, which separately constrains any production-readiness claim downstream of this number — see potential ERRATA-011 pending Q-008/Q-009. | PENDING | DECISION-003, Q-007, Q-008, Q-009 |

---

## Day 1 audit summary

10 errata identified through cross-chapter audit on 2026-05-23 (Phase F Day 1).

All data-verified on Vast.ai instance C.37423026 against canonical source files:

- BCF stats (ERRATA-001 through 004): verified against `bcf_pooled_3model.json` and `bcf_pooled_results.json`
- HPA file path (ERRATA-005): verified via `c3_hpa_v4_verdict.md` directing v4 as canonical
- Bytedance row count (ERRATA-007): verified by line-counting `hpa_simulation_bytedance_v4.csv` = 601 lines (600 + header)
- CI crossing claim (ERRATA-008): verified Bitbrains 30min CI = [+3.5390, +5.4170] vs Alibaba +3.39 (Alibaba sits 0.149 pp below CI lower bound)
- Lag-1 phrasing (ERRATA-009): verified via inspection of `task2_hpa_v2.py` showing matched comparison is at same (h, τ, s) across all reactive lag variants
- Per-horizon counts (ERRATA-010): submitted PDF's 128/156/157/92 from `hpa_simulation_v2.csv` (superseded); v4 re-derivation pending (Q-004) — unblocked 2026-05-24

## Day 2 audit note (2026-05-24)

**ERRATA-010 unblocked.** Q-004 closed by reading `results/bcf_v2/hpa_v4_dominance_per_dataset.csv` directly. The canonical file already encodes the per-horizon counts; re-derivation from the underlying v4 grids was unnecessary. Memory anchors match the file exactly:

| dataset | h10 | h30 | h60 | h120 |
|---|---|---|---|---|
| Alibaba ML strict cells | 40/40 | 35/40 | 15/40 | 1/40 |
| Bitbrains ML strict cells | 40/40 | 0/40 | 0/40 | 0/40 |
| Bytedance ML strict cells | — | 33/40 | 32/40 | 29/40 |

11-cell grand total: 225 ML-strict cells / 440 cells (51.1%); 675 reactive-dominated pairs / 1760 pairs (38.4%).

**Two scope distinctions to keep straight:**
- v1-sprint (`hpa_simulation_v2.csv`, max_replicas=100): per-(cell × lag), Alibaba sum 533/640
- v4 (`hpa_v4_dominance_per_dataset.csv`, max_replicas=1000): per-cell, Alibaba sum 91/160, 11-cell sum 225/440

These sums are NOT comparable — they index different scopes AND different grids. The ERRATA-010 draft text originally claimed "total 533/640 sum unchanged"; that claim was wrong and has been removed in the v4 substitution above.

**`passes_gate=False` for all 11 cells.** This includes Alibaba h10 where ML strict-dominates 40/40 cells. The "gate" appears to be a stricter pre-registered criterion than cell-level ML strict-dominance — likely a production-readiness threshold. Until Q-008 closes (reading `task_c3_hpa_rebuild_v4.py` to extract the gate definition), we cannot determine whether this is a manuscript-level disclosure issue (ERRATA-011 candidate) or an exploratory diagnostic that doesn't require new errata.

**Dominance definition drift.** My Pareto-dominance definitions in `phase_f/scripts/rederive_v4_per_horizon.py` (v1, "min on each axis strict") and `rederive_v4_per_horizon_v2.py` (v2, "per-lag Pareto") both produce counts that disagree with the canonical CSV. v1 gave 0/0/0/0 for Alibaba ML strict; v2 gave the same. The canonical script (`task_c3_hpa_rebuild_v4.py`) uses a weaker criterion (likely a combined cost metric). Both scripts are archived in `phase_f/scripts/` as reference but are NOT authoritative — `hpa_v4_dominance_per_dataset.csv` is the source of truth. Q-008 will document the actual criterion.

## Batch application plan

- **Day 3 morning:** Read README_HPA_CANONICAL.md + task_c3_hpa_rebuild_v4.py → close Q-008, decide on Q-009 / ERRATA-011 (~30 min)
- **Day 3 mid-morning:** Apply ERRATA-001 through 010 (and 011 if opened) in one Overleaf session (~20 min batch + ~30 min diff verify)

When applied, status changes from PENDING (or BLOCKED) to APPLIED with date-applied appended in a new column. The row is never deleted.

---

(future errata appended below — keep ID sequence monotonic)