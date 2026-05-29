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
| 012 | 2026-05-27 | Ch4 BCF cell-level evidence paragraph + Table 4.10 Bitbrains row | Table 4.10 Bitbrains row: delta@30min=`-5.46pp`, delta@120min=`+3.30pp`, verdict "ML wins only @120min"; surrounding §4.X prose claims "ML wins only at h=120 for Bitbrains" under OLD pool per-VM median scope. | Table 4.10 Bitbrains row: delta@30min=`+1.53pp`, delta@120min=`-2.81pp`, verdict reframed to NEW pool semantics (h10=-16.66, h30=+1.53, h60=-1.99, h120=-2.81pp; %VMs NEW>naive: h30=61%, h120=35%). §4.X prose grows from one sentence to ~3 sentences disclosing per-VM win rates and the OLD→NEW pool aggregation scope shift. Alibaba/ByteDance rows unchanged. Canonical source CSV: `reports/tables/boundary_condition_table_corrected.csv` (claimed at NEW pool from a prior session); verifier `thesis_numbers.json` cleared 211/3 → 214/0 on 2026-05-27 under same scope. | **BLOCKED** |  | DECISION-011 (Q-007 Anchor A); **see ERRATA-012 verification note below** |
| 013 | 2026-05-25 | Project storage: `/results/bitbrains/h{030,060,120}/predictions/test_ensemble_hetero.npy` | Files contain Alibaba data (shape ~1.6M rows, mean 10.28), not Bitbrains data (~203K rows, mean 30.4). Confirmed via shape and value distribution against `y_true` and against the Alibaba hetero files at the same horizons. The h010 hetero file does not exist at all (consistent with the known BiLSTM OOF shared-memory failure documented in `run.log`). Likely cause: a pipeline run-script copied Alibaba's hetero output to the Bitbrains path during an earlier session (file-path templating error). | No manuscript change required. The submitted PDF cites per-VM XGBoost (`bitbrains_per_vm_results.csv`) as the Bitbrains ML reference, not the global hetero ensemble; Figure 4.2 caption already states this explicitly. PAR substitutes `test_xgb.npy` aligned to the Chronos-2 K=50 spine for apples-to-apples per-series comparison. Verification: `par_per_series_r2_v2.py` pre-flight check (ii) returns pooled R² = 0.4973 at Bitbrains h030, matching the diagnostic. Detail in standalone file `phase_f/ERRATA-013.md`. | APPLIED | 2026-05-25 | — (PAR D17 diagnostic) |
| 014 | 2026-05-28 | F2 per-series rescue data pipeline | Bitbrains `test_ensemble_hetero.npy` corruption (same files as ERRATA-013) surfaces again during F2 per-series partial-R² computation against `meta_learner/hetero_ensemble.json`. The F2 per-series script cannot read the corrupted file as a Bitbrains source. | Substitute `test_ensemble_homo.npy` for Bitbrains as the canonical F2 per-series ensemble source. BiLSTM is unavailable at all Bitbrains horizons (per `hetero_ensemble.json` field `bilstm_unavailable`), so under the existing pool definition homo ≡ hetero on Bitbrains; the substitution does not change the underlying mathematics, only the file pointer. Pooled cell-level R² recomputed under the substituted file matches `meta_learner/homo_ensemble.json` to ≤ 0.001 across all 11 (dataset, horizon) cells (verification gate in `build_per_series_fulltest_r2.py`). | APPLIED | 2026-05-28 | DECISION-017 |
| 015 | 2026-05-28 | Ch5 §5.7 (new section, drafted in `phase_f/manuscript/manuscript_5_7_F4_evaluation.md`) | The submitted PDF contains no F4 closed-loop autoscaler evaluation. F4 was scoped during Phase F under written supervisor acceptance (Dr. Ho Long Van, 2026-05-22) and ran on 2026-05-28 under pre-registered protocol committed to git at SHA `012786af9d` on 2026-05-27. | Insert §5.7 "Routing alternative evaluation". The section reports a paired stationary bootstrap (Politis and Romano 1994; R=1000; expected block length $b \approx T^{1/3}$) with Holm-Bonferroni correction across six tests, comparing v6 (uniform F3-FT pipeline) against v7 (per-dataset adapter routing) on three datasets at α = 0.95 chance-constrained MPC. Verdicts: Alibaba (n=14,760 paired) shows a Pareto **tradeoff** — violation rate −1.897 pp [−2.190, −1.603] AND replica mean +0.220 [+0.091, +0.353], both Holm-significant in opposite directions. Bitbrains (n=426) shows partial improvement — violation rate −0.231 pp significant, replica mean −0.132 with CI [−0.424, +0.141] straddling zero. ByteDance (n=279) shows neither effect significant. The pre-registered strict-Pareto rule (v7 must dominate Alibaba and be at least indistinguishable elsewhere) is not met because Alibaba's replica-mean CI is wholly positive. Resolution: adopt v6 as canonical F3-FT pipeline; retain v7 routing infrastructure as reproducibility artefact in `phase_f/models/f3_lora_rank8_no_log1p/` and `phase_f/configs/router_v7.yaml`. Source files: `phase_f/data/f4_trajectories.parquet` (30,933 rows), `phase_f/data/f4_bootstrap_results.csv`, `phase_f/data/f4_pareto_verdicts.csv`, `phase_f/data/f4_pareto_decision.json`. Companion figure: `phase_f/data/f4_pareto_combined.pdf`. | PENDING | | DECISION-018 |
| 016 | 2026-05-28 | Ch5 §5.7 prose + `f4_pareto_decision.json` rationale string | The F4 Pareto analysis script (`phase_f/scripts/f4_pareto_analysis.py`) labels all three per-dataset verdicts `indistinguishable` whenever the strict Pareto orthant condition is not met. The JSON file inherits this label, and the script's `rationale` field reads "v7 did not dominate Alibaba; the routing hypothesis is not vindicated." | The single label collapses three distinct empirical patterns. Alibaba shows two Holm-significant effects in opposite directions (Pareto tradeoff). Bitbrains shows one significant effect and one CI straddling zero (partial improvement under uncertainty). ByteDance shows two CIs straddling zero (genuine null result). §5.7 must distinguish the three patterns by name, report the bootstrap CIs directly, and avoid the collapsed label in narrative prose. The single-label verdict drives only the binary routing decision (v6 retained); it does not describe the scientific finding. | PENDING | | DECISION-018 |

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
- **2026-05-28 (Phase F dual closure):** ERRATA-014, 015, 016 opened during F2 per-series rescue and F4 closed-loop MPC evaluation. ERRATA-014 resolves at the computational-pipeline level (homo substitution applied in `build_per_series_fulltest_r2.py`); no Overleaf change required. ERRATA-015 and ERRATA-016 are content for a new §5.7 section drafted in `phase_f/manuscript/manuscript_5_7_F4_evaluation.md` but not yet pasted into Overleaf. Both will flip PENDING → APPLIED during F5 writing.

---

## Closure additions audit (2026-05-25 to 2026-05-28)

Four rows were appended to the table after the Day 3 batch closed. Two of them (ERRATA-013, ERRATA-014) document a project-storage data corruption that surfaced first during PAR Step D17 diagnostic work and resurfaced during the F2 per-series rescue. The other two (ERRATA-015, ERRATA-016) capture the F4 closed-loop MPC evaluation outcome and the verdict-label disambiguation that emerged from reading the analysis script's output.

We note that ERRATA-013 and ERRATA-014 describe the same underlying data corruption (`test_ensemble_hetero.npy` for Bitbrains containing Alibaba data) at different scopes. ERRATA-013 is the project-storage discovery and PAR-side resolution; ERRATA-014 is the F2 per-series rescue's parallel resolution via homo-file substitution. We considered collapsing the two into a single row and decided against it. The two work streams reached the corruption independently — PAR through its per-series argmax pipeline, F2 through partial-R² computation — and produced two distinct substitution decisions tied to two distinct downstream consumers (`test_xgb.npy` aligned to the Chronos-2 K=50 spine for PAR; `test_ensemble_homo.npy` for F2). Numbering the two events separately preserves the audit trail.

ERRATA-015 carries the substantive F4 evaluation finding. The Alibaba result is the only one that warrants careful framing in §5.7: two Holm-significant effects in opposite directions, neither of which dominates under the pre-registered strict-Pareto rule. The verdict label "indistinguishable" in `f4_pareto_decision.json` understates this. ERRATA-016 records the disambiguation we will write into the chapter — Pareto tradeoff, partial improvement, genuine null — and notes that the collapsed label is fine for the routing decision but inadequate for the scientific narrative.

We did not draft Table 5.7.1 replacement text at the time the row was opened; the manuscript draft in `phase_f/manuscript/manuscript_5_7_F4_evaluation.md` already contains the companion table. The row therefore points at the draft file rather than embedding the table inline.

---

## ERRATA-012 verification note (added during state-reconciliation review)

We measured the proposed correction against the canonical per-VM source rather than trust the row text. Two files were on disk: `bitbrains_per_vm.csv` and `boundary_condition_table_corrected.csv`. The result inverts the premise of the erratum, and it contradicts a claim the row makes about its own source.

**What the per-VM data shows.** Computing the per-VM median `r2_delta` over `bitbrains_per_vm.csv` (n=156 VMs per horizon, 624 rows):

| Horizon | per-VM median r2_delta | %VMs where ML wins |
|---|---|---|
| h10  | −8.68 pp | 12.2% |
| h30  | **−5.46 pp** | 35.9% |
| h60  | −0.79 pp | 47.4% |
| h120 | **+3.30 pp** | 51.9% |

`boundary_condition_table_corrected.csv` agrees exactly: the Bitbrains row reads `-5.46pp` at h30, `+3.30pp` at h120, verdict "ML wins only @120min." These are the figures already printed in the submitted manuscript.

**The finding.** The manuscript's existing numbers match the per-VM source. The proposed correction does not. ERRATA-012 was opened on the assumption that Table 4.10 held stale OLD-pool values needing migration to NEW-pool values; the per-VM data shows the manuscript was already correct, and the proposed `+1.53` / `−2.81` would replace a correct row with figures the source contradicts. The proposed win-rates compound the mismatch: the row claims 61% of VMs win at h30 and 35% at h120, while the data gives 35.9% at h30 and 51.9% at h120 — the two horizons roughly transposed.

**A claim in the row that we disproved.** The Correction column asserts that `boundary_condition_table_corrected.csv` is "at NEW pool from a prior session." It is not. That file holds the OLD-pool per-VM values (`-5.46` / `+3.30`). The row's own source citation therefore points at a file that refutes the row's proposed numbers. We suspect the `+1.53` / `−2.81` figures were read from a different file and the citation was attached in error.

**Why we did not simply void the erratum.** The proposed figures are not random. They trace to a genuinely different aggregation. A prior session recorded a per-VM median over a filtered 142-VM set (NEW pool) at exactly h30 = +1.53 / h120 = −2.81, against the 156-VM median we computed here. The 14-VM gap is the likely filter — idle or low-variance VMs excluded from the NEW pool. So three Bitbrains aggregations are in play, and they disagree on sign: the 156-VM per-VM median (−5.46 / +3.30), a 142-VM filtered per-VM median (+1.53 / −2.81 per the prior record), and a pooled full-test scope in `cross_dataset_headline_v2.csv`, which is not present in project knowledge. We did not reconcile the 142-versus-156 VM-count discrepancy directly, and we cannot read the pooled CSV from here.

**Resolution required before this row may be applied.** Read `cross_dataset_headline_v2.csv` on Vast.ai. Establish which VM set Table 4.10 reports and which the surrounding §4.X prose intends. Determine whether `+1.53` / `−2.81` describes a real cell — the 142-VM filtered median or the pooled scope — that belongs somewhere in the manuscript, or whether the erratum was sourced in error. Until that read settles the scope, the Table 4.10 per-VM row stays at `−5.46` / `+3.30`. Applying the proposed correction to that row on current evidence would introduce an error into a correct table.

**Status rationale.** PENDING → BLOCKED. Under this file's own definition, BLOCKED marks a row that depends on a resolution before it can be applied. The resolution here is the scope question above, gated on a source file absent from the working set. The row carries forward to the F5 writing session, where the Bitbrains row is cited and the canonical CSVs will be synced and present. It must not enter any earlier Overleaf batch.

---

(future errata appended below — keep ID sequence monotonic)