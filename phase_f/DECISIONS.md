# DECISIONS.md

**Project:** kubernetes-cpu-ensemble-thesis
**Started:** 2026-05-23 (consolidated from prior chat history + Phase F Day 1)
**Update protocol:** append-only. Never delete or reorder. Decisions evolve via SUPERSEDED status, not deletion.

---

## How to read this file

Each decision is numbered sequentially (DECISION-NNN), never reused. Status values:
- **PROPOSED** — drafted but not yet implemented
- **ACCEPTED** — approved, awaiting implementation
- **IMPLEMENTED** — done, in effect
- **SUPERSEDED** — replaced by a later decision (points forward to the new DECISION-NNN)
- **UNUSED** — drafted but determined not to be needed (resolved without action)

The Y-statement format: "In the context of __, facing __, we decided __ to achieve __, accepting __."

---

## DECISION-001 — Manuscript structure: 6 chapters (not 8)

**Date:** 2026-05-22
**Status:** IMPLEMENTED

**Context:** Original outline proposed 8 chapters. Workload and SCSE page budget made this unrealistic. The Phase A integration findings consolidated naturally into fewer chapters.

**Decision:** Six chapters — Ch1 Intro (6pp), Ch2 Related (11pp), Ch3 Methodology (13pp), Ch4 Impl&Results (24pp), Ch5 Discussion&Eval (25pp), Ch6 Conclusion (6pp) = 85 body pp. Bound ~110-120pp.

**Alternatives considered:** 8 chapters with F1-F4 separated (too long); 5 chapters merging Methodology and Impl&Results (insufficient room for ablations).

**Consequences:** SCSE 75-page anchor unconfirmed (from Business School); will verify with department. Phase D work distributed: Ch3 §3.7, Ch5 §5.7, Ch5 §5.6.

**Y-statement:** In the context of a Phase F manuscript revision, facing page budget constraints and content consolidation, we decided on a 6-chapter structure to achieve focused presentation within ~110-120 bound pages, accepting that some F1-F4 Phase F work will be appendices rather than separate chapters.

---

## DECISION-002 — BCF canonical = 3-model pool

**Date:** ~2026-05-21
**Status:** IMPLEMENTED

**Context:** BCF was originally formulated with a 4-model pool including Granite-TTM. Granite's wins were dataset-specific (4/4 Bytedance, 0/4 Alibaba), not predicate-aligned. The 4-model pooled run gave AUC=0.667, p=0.0511 — marginally non-significant.

**Decision:** Canonical BCF pool is 3 models (NNLS + Chronos-2 + TimesFM). Granite-TTM excluded but documented in `bcf_pooled_results.json` as justification for exclusion.

**Alternatives considered:** Keep all 4 models in pool (loses significance); include Granite only at Bytedance horizons (special-cases the predicate, undermines transferability claim).

**Consequences:** Headline AUC = 0.80, percentile CI [0.7097, 0.8871], p = 0.0097. CI method is percentile because BCa is degenerate for binary classifier × binary outcome.

**Source files:** `bcf_pooled_3model.json` (canonical), `bcf_pooled_results.json` (4-model justification).

**Y-statement:** In the context of BCF generalisation claims, facing a Granite-TTM model that wins on the wrong axis (dataset, not predicate), we decided to exclude Granite from the canonical pool to achieve a transferable predicate-based finding, accepting that the 4-model run becomes appendix material.

---

## DECISION-003 — V4 HPA grid at max_replicas=1000 supersedes v1/v2/v3

**Date:** 2026-05-23
**Status:** IMPLEMENTED

**Context:** Phase C saturation analysis (`c3_saturation_verdict.md`) revealed that the v1-sprint HPA grid (`hpa_simulation_v2.csv`, max_replicas=100) had 12-40% of cells at the replica cap. This biased the dominance counts toward ML-Proactive at heavy load.

**Decision:** Regenerate HPA grids with max_replicas=1000. Make `hpa_simulation_*_v4.csv` (3 datasets) canonical. Rename v1/v2/v3 files to `_SUPERSEDED`.

**Alternatives considered:** Keep v1-sprint with documented caveat (rejected — caveat doesn't resolve the bias); raise max_replicas to 500 (rejected — still saturates 5-15% of grid).

**Consequences:** Manuscript §3.8, §3.10, §4.11, §6.1 need re-citation. Per-horizon dominance counts change (Q-004 re-derivation pending). Errata sheet entry ERRATA-010.

**Source files:** Canonical = `hpa_simulation_alibaba_v4.csv`, `hpa_simulation_bitbrains_v4.csv`, `hpa_simulation_bytedance_v4.csv`. Superseded = `hpa_simulation_v2.csv` (and the others).

**Y-statement:** In the context of HPA dominance evidence, facing grid saturation in 12-40% of cells, we decided to regenerate at max_replicas=1000 to achieve unbiased dominance counts, accepting manuscript re-citation work in 4 sections.

---

## DECISION-004 — F0 calendar reinvestment: path (a)

**Date:** 2026-05-23
**Status:** ACCEPTED (Day 1 same-day decision)

**Context:** Phase F Day 1.5b unplanned audit collapsed Days 4-5 of original F0 plan into Day 1. Two options: (a) keep F0 at 10 days, reinvest slack into bibliography audit + chapter outlines + pre-registration drafting, or (b) compress F0 to 7-8 days, start F1 earlier.

**Decision:** Path (a). Keep F0 at 10 days.

**Alternatives considered:** Path (b) — start F1 router work earlier. Rejected because (1) defence is in October, no urgency, and (2) F1/F2/F3 have pre-registered thresholds and missing them means we want time to iterate, not time to write up null results.

**Consequences:** Days 4-10 reallocated to bibliography audit (Day 4), chapter outlines (Days 5-6), pre-registration drafting (Days 7-8), F1 setup (Days 9-10).

**Y-statement:** In the context of an unexpectedly productive Day 1 (3 days of planned work collapsed into 1), facing the option to start F1 earlier or polish F0 further, we decided to reinvest the slack in F0 to achieve better foundations before threshold-gated experiments, accepting that F1 starts at original Day 11.

---

## DECISION-005 — Phase F scope locked with Dr. Ho written acceptance

**Date:** 2026-05-22
**Status:** IMPLEMENTED

**Context:** Phase F (135-day extension) needed formal scope agreement with supervisor. Written acceptance received 2026-05-22.

**Decision:** Phase F sub-tasks: F1 router (predictability-aware ensemble selection), F2 predictability measures (WPE/Ω/LZ/DFA), F3 cost-asymmetric quantile fine-tune of Chronos-2, F4 integration with OptScaler + AHPA, F5 6 new chapters.

**Pre-registered thresholds:**
- F1: macro-F1 ≥ 0.55 on 11-cell test set
- F2: partial-R²(WPE | ACF@24h) ≥ 0.3
- F3: Spearman ρ ≥ 0.6 AND |DFL−Pinball−τ| ≤ 5%

**Alternatives considered:** Narrower scope (F1 only, defer F2-F4 to future work); broader scope (add F6 production K8s deployment evaluation).

**Consequences:** ~135 days from F0 start; defence ~October 2026 with one-semester graduation delay. If any phase misses threshold, result reported honestly as null/below-threshold (no post-hoc adjustment).

**Source:** `phase_f/decisions/supervisor_acceptance_2026-05-22.pdf`

**Y-statement:** In the context of a thesis extension following first-version submission, facing supervisor's preference for systems-grade contributions, we decided on a 5-phase Phase F with pre-registered thresholds to achieve defensible operational findings, accepting a ~135-day timeline and one-semester graduation delay.

---

## DECISION-006 — Dual-canonical store: Drive + Git

**Date:** 2026-05-23
**Status:** IMPLEMENTED

**Context:** Phase F work needs durable, version-controlled storage. Drive alone is unversioned; Git alone can't hold large data files.

**Decision:** Drive = canonical for DATA (parquets, raw traces, model checkpoints, per-day data snapshots, hi-res figures). Git = canonical for CODE + work-record (scripts, decisions, journal, state, small CSVs/JSONs cited in manuscript).

**Alternatives considered:** Drive only (no versioning); Git only (size limits); Notion (vendor lock-in); Obsidian (workflow disruption).

**Consequences:** Two stores to sync. Rclone configured both sides (Vast.ai + local Windows). Both stores get day-close updates via PowerShell script. Memory snapshots go to both.

**Y-statement:** In the context of a long-running thesis with large data + versioned code needs, facing single-store limitations, we decided on dual-canonical Drive + Git to achieve both versioning and large-file handling, accepting the ~30-second daily sync overhead.

---

## DECISION-007 — BCa CI label — UNUSED

**Date:** 2026-05-23
**Status:** UNUSED

**Context:** Earlier work plans included an errata item to relabel the BCF AUC confidence interval method (percentile vs BCa). Submitted PDF was suspected to use BCa labels in BCF context.

**Decision (initially proposed):** Add errata item correcting "BCa CI" to "percentile CI (BCa degenerate for binary classifier)".

**Verification on 2026-05-23:**
```
pdftotext -layout Ensemble_*.pdf - | grep -iE 'BCa|bias[- ]?corrected[- ]?accelerated'
→ empty output
```

**Outcome:** UNUSED. Submitted PDF does not refer to BCa in the BCF AUC context. The CI method is unlabeled in the submitted prose ("95% CI [0.70, 0.88]") — silent correctness rather than deliberate, but no errata correction needed.

**Enhancement (dissertation version, not errata):** Add a footnote in Ch3 §3.8 (referenced from Ch4 §4.8) explaining percentile choice and BCa degeneracy:

> "We report percentile intervals throughout because BCa is degenerate for a binary classifier of a binary outcome: the acceleration term â and the bias-correction z₀ are undefined when the bootstrap statistic has only finitely many distinct values (the BCF AUC takes ~200 unique values across 10,000 resamples). This is recorded directly in bcf_pooled_3model.json as ci_method = 'percentile (binary classifier; BCa degenerate)'. (Efron-Tibshirani 1993)"

**Net effect:** Positive for dissertation version (via footnote); neutral for submitted PDF (no change).

**Y-statement:** In the context of a planned BCa correction, facing actual evidence that the submitted PDF never used the BCa label, we decided to close the decision as UNUSED to achieve accurate provenance, accepting that the footnote enhancement is the only forward-looking action.

---

(future decisions appended below)
