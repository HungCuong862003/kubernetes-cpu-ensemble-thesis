# ERRATA.md

**Project:** kubernetes-cpu-ensemble-thesis
**Started:** 2026-05-23 (Phase F Day 1 audit)
**Update protocol:** append-only. Never delete or reorder. Status field tracks lifecycle:
- **PENDING** — error identified, not yet applied to manuscript
- **APPLIED** — correction applied in Overleaf, commit pushed
- **BLOCKED** — depends on a Q-ID resolution before it can be applied

---

## How to read this file

Each errata is numbered sequentially (ERRATA-NNN), never reused. Rows are immutable except for status updates and the appended `Applied` date. Corrections propagate from this file *into* the chapter, never the reverse. The submitted PDF is the baseline; this file tracks what changed since.

---

## Errata table

| # | Date opened | Location | Original (in submitted PDF) | Correction | Status | Applied | Related Q-ID / DECISION |
|---|---|---|---|---|---|---|---|
| 001 | 2026-05-23 | Ch3 §3.8 | "p < 0.011" | "p = 0.0097" | APPLIED | 2026-05-25 | DECISION-002 |
| 002 | 2026-05-23 | Ch3 §3.8 | cites `bcf_pooled_results.json` (4-model file) but reports 3-model numbers | change citation to `bcf_pooled_3model.json` | APPLIED | 2026-05-25 | DECISION-002 |
| 003 | 2026-05-23 | Ch3 §3.8 | "95% CI [0.70, 0.88]" (method unlabeled) | "percentile 95% CI [0.71, 0.89] (BCa degenerate)" | APPLIED | 2026-05-25 | DECISION-007 |
| 004 | 2026-05-23 | Ch3 §3.8 | (no footnote on CI method) | ADD footnote `fn:bca-degenerate` explaining percentile choice and BCa degeneracy (Efron-Tibshirani 1993) | APPLIED | 2026-05-25 | DECISION-007 |
| 005 | 2026-05-23 | Ch3 §3.10 | cites `hpa_simulation_v2.csv` (v1-sprint, max_replicas=100) | change to `bcf_v2/hpa_simulation_*_v4.csv` (max_replicas=1000); add saturation justification clause | APPLIED | 2026-05-25 | DECISION-003 |
| 006 | 2026-05-23 | Ch4 §4.8 | (no cross-reference to BCa footnote) | ADD reference to Ch3 §3.8 footnote `fn:bca-degenerate` | APPLIED | 2026-05-25 | DECISION-007 |
| 007 | 2026-05-23 | Ch4 §4.11 | "800 rows per dataset" | qualify: "800 rows per Alibaba and Bitbrains; 600 rows for ByteDance (three horizons: h30, h60, h120)" | APPLIED | 2026-05-25 | DECISION-003 |
| 008 | 2026-05-23 | Ch5 §5.1 | "CI crosses the Alibaba 30 min point" | "the Bitbrains 30 min CI lower bound (+3.54 pp) sits 0.15 pp above the Alibaba 30 min point (+3.39 pp) — within typical bootstrap variation across cells, but the interval itself does not cover the Alibaba point" | APPLIED | 2026-05-25 | — |
| 009 | 2026-05-23 | Ch5 §5.6 | "matched reactive lag-1 point" | "matched reactive lag variant at the same $(h, \tau_{\text{util}}, s)$" | APPLIED | 2026-05-25 | — |
| 010 | 2026-05-23 | Ch6 §6.1 + Ch4 Table 4.13 | Original headline: 533 / 640 Alibaba reactive points dominated, with per-horizon breakdown 128 / 156 / 157 / 92, from v1-sprint `hpa_simulation_v2.csv` at max_replicas=100 (saturation-confounded). | Dual-metric replacement from `hpa_v4_dominance_per_dataset.csv` at max_replicas=1000. **reactive_dominated_pct** per horizon (of 160): Alibaba 120/55/21/1; Bitbrains 117/0/0/0; ByteDance —/125/124/112. **ml_strict_dominance_pct** per horizon (of 40): Alibaba 40/35/15/1; Bitbrains 40/0/0/0; ByteDance —/33/32/29. 11-cell pools: 675/1760 (38.4%) and 225/440 (51.1%). Full substitution paragraph and Table 4.13 format in Day 3 application note below. | APPLIED | 2026-05-25 | DECISION-003, DECISION-008 |
| 011 | 2026-05-25 | Ch6 §6.1 + Ch4 §4.11 | (implied population-level Pareto dominance via 533/640 headline; no acknowledgement of the pre-registered 80% reactive-dominated production-readiness gate) | Explicit disclosure that `passes_gate=False` in all eleven cells under the gate defined at `task_c3_hpa_rebuild_v4.py` line 279 (`reactive_dominated_pct >= 0.80`); highest single-cell rate ByteDance h30 at 78.12%. Disclosure embedded in the ERRATA-010 substitution paragraph rather than as a separate footnote. | APPLIED | 2026-05-25 | Q-008, Q-009, DECISION-003, DECISION-008 |

---

## Day 1 audit summary

10 errata identified through cross-chapter audit on 2026-05-23 (Phase F Day 1). All data-verified on Vast.ai instance C.37423026 against canonical source files:

- BCF stats (ERRATA-001 through 004): verified against `bcf_pooled_3model.json` and `bcf_pooled_results.json`
- HPA file path (ERRATA-005): verified via `c3_hpa_v4_verdict.md` directing v4 as canonical
- Bytedance row count (ERRATA-007): verified by line-counting `hpa_simulation_bytedance_v4.csv` = 601 lines (600 + header)
- CI crossing claim (ERRATA-008): verified Bitbrains 30min CI = [+3.5390, +5.4170] vs Alibaba +3.39 (Alibaba sits 0.149 pp below CI lower bound)
- Lag-1 phrasing (ERRATA-009): verified via inspection of `task2_hpa_v2.py` showing matched comparison is at same (h, τ, s) across all reactive lag variants
- Per-horizon counts (ERRATA-010): submitted PDF's 128/156/157/92 from `hpa_simulation_v2.csv` (superseded); v4 re-derivation pending (Q-004 at D1 close, RESOLVED D2)

---

## Day 2 audit note (2026-05-24)

Investigating Q-004 surfaced three findings about the HPA dominance landscape that bear on ERRATA-010 and motivated opening ERRATA-011 on Day 3.

1. **v1-sprint vs v4 scope distinction.** The submitted PDF's 533/640 figure derived from `hpa_simulation_v2.csv` at `max_replicas=100`, which `c3_saturation_verdict.md` documents as saturating 12–40% of grid points (binding cluster-capacity ceiling). The v4 grid at `max_replicas=1000` reduces saturation to <0.07% on every grid point. The two figures are not commensurable; the v4 figure supersedes.
2. **Dominance definition drift.** `hpa_v4_dominance_per_dataset.csv` reports two metrics whose distinction was not preserved in earlier prose: `reactive_dominated_pct` (160-point denominator) and `ml_strict_dominance_pct` (40-configuration denominator). They measure complementary axes of the same Pareto-cloud relationship and can disagree by a factor of two within the same cell. ERRATA-010's substitution reports both explicitly.
3. **`passes_gate=False` for all eleven cells** under the pre-registered ≥80% reactive-dominated production-readiness criterion (highest single-cell rate: ByteDance h30 at 78.12%). The submitted PDF's headline framing implies population-level dominance that the gate refutes. ERRATA-011 opened on Day 3 to disclose this.

ERRATA-010 transitioned BLOCKED → PENDING on D2 close (substitution numbers locked from canonical CSV). The Day 3 SSH read of `README_HPA_CANONICAL.md` + script body for Q-008 confirmed the gate semantics and produced the final substitution wording in the Day 3 application note below.

---

## Day 3 application note (2026-05-25) — ERRATA-010 + ERRATA-011 substitution text

### Ch6 §6.1 replacement paragraph

> The cross-dataset HPA simulation at `max_replicas=1000`, recorded in `results/bcf_v2/hpa_v4_dominance_per_dataset.csv`, reports per-cell dominance counts along two complementary axes. The first counts reactive operating points that are strictly Pareto-dominated by at least one ML-Proactive configuration, out of 160 reactive points per cell (4 reaction lags × 4 target utilisations × 10 safety margins). On Alibaba this count declines monotonically across horizons — 120, 55, 21, and 1 of 160 at h10, h30, h60, and h120, summing to 197 of 640 across the four horizons. On Bitbrains the pattern is not a decline but a cutoff: 117 of 160 at h10 and exactly zero at h30, h60, and h120. On ByteDance the counts are stable-high across its three horizons — 125, 124, and 112 of 160 at h30, h60, and h120. The second axis counts ML-Proactive configurations that strict-dominate at least one reactive point, out of 40 per cell: 40 / 35 / 15 / 1 on Alibaba, 40 / 0 / 0 / 0 on Bitbrains, and 33 / 32 / 29 on ByteDance. Pooled across the eleven cells, 675 of 1760 reactive points (38.4%) are strict-dominated and 225 of 440 ML configurations (51.1%) strict-dominate something. The pre-registered gate of 80% population coverage — the threshold defined in `task_c3_hpa_rebuild_v4.py` and intended as a production-readiness criterion for unqualified replacement of reactive HPA — is not crossed in any of the eleven cells; the highest single-cell rate is ByteDance h30 at 78.12%. The earlier submission reported 533 of 640 Alibaba reactive points dominated, with per-horizon breakdown 128 / 156 / 157 / 92. That figure derived from a v1-sprint grid at `max_replicas=100` whose 12–40% saturation rate, diagnosed in `c3_saturation_verdict.md`, asymmetrically inflated long-horizon dominance counts; the v4 protocol removes this confound and supersedes the earlier figure. The cross-dataset pattern visible in the v4 counts aligns with the ACF@24h ordering (ByteDance 0.489 > Alibaba 0.316 > Bitbrains 0.116), consistent with the boundary condition framework of §3.8 rather than with horizon alone.

### Ch4 Table 4.13 replacement

| Horizon | Alibaba (rd / mld / gate) | Bitbrains (rd / mld / gate) | ByteDance (rd / mld / gate) |
|---|---|---|---|
| h10  | 120 / 40 / No | 117 / 40 / No | — |
| h30  | 55  / 35 / No | 0   / 0  / No | 125 / 33 / No |
| h60  | 21  / 15 / No | 0   / 0  / No | 124 / 32 / No |
| h120 | 1   / 1  / No | 0   / 0  / No | 112 / 29 / No |

Caption:

> Per-cell HPA dominance counts from `hpa_v4_dominance_per_dataset.csv`. `rd` (reactive-dominated) reports the number of the 160 reactive operating points per cell that are strict-Pareto-dominated by at least one ML-Proactive configuration. `mld` (ML strict-dominance) reports the number of the 40 ML configurations per cell that strict-dominate at least one reactive point. `gate` reports whether $\text{rd}/160 \geq 0.80$, the pre-registered production-readiness criterion. No cell crosses the gate; the highest single-cell `rd` rate is ByteDance h30 at 78.12%. ByteDance has no h10 horizon. Pooled across the eleven cells: $\text{rd}/n_{\text{reactive}} = 675/1760 = 38.4\%$; $\text{mld}/n_{\text{ml}} = 225/440 = 51.1\%$.

---

## Batch application record

- **2026-05-25 (Day 3 morning):** ERRATA-001 through 010 applied in one Overleaf session; ERRATA-011 added and applied in the same commit (disclosure embedded in 010's paragraph). Single commit message: "Apply ERRATA-001..011 — see ERRATA.md for source-of-truth substitutions, Phase F D3 (2026-05-25)". Visual diff against submitted-PDF baseline completed for all 8 chapter loci (Ch3 §3.8, Ch3 §3.10, Ch4 §4.8, Ch4 §4.11, Ch4 Table 4.13, Ch5 §5.1, Ch5 §5.6, Ch6 §6.1). PDF rebuilt clean.

---

(future errata appended below — keep ID sequence monotonic)