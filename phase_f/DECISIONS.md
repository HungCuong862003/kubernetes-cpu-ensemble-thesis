---
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

---

## DECISION-008 — Full retirement of v1-sprint 533/640 figure with dual-metric replacement

**Date:** 2026-05-25
**Status:** IMPLEMENTED (Day 3 ERRATA batch)

**Context:** Q-008/Q-009 closure on Day 3 surfaced that the submitted PDF's headline figure "ML-Proactive Pareto-dominates the matched reactive in 533/640 cell-matched comparisons" derives from `hpa_simulation_v2.csv` at `max_replicas=100`, whose 12–40% grid-saturation rate (diagnosed in `c3_saturation_verdict.md`) inflated long-horizon dominance counts asymmetrically. Direct same-metric v1-sprint→v4 comparison on Alibaba shows the per-horizon counts moved 128→120, 156→55, 157→21, 92→1 (totals 533/640 → 197/640, a 45 pp drop). The v4 protocol at `max_replicas=1000` also reports a second metric, `ml_strict_dominance_pct`, that the submitted PDF did not use; and a pre-registered gate at 80% reactive-dominated coverage that fails in all eleven cells.

**Decision:** Retire the 533/640 figure entirely in ERRATA-010, not just the per-horizon breakdown. Replace with a dual-metric paragraph that reports both `reactive_dominated_pct` (the same metric the original used, now at v4 scope) and `ml_strict_dominance_pct` (the complementary axis), discloses `passes_gate=False` everywhere (this is ERRATA-011), names the saturation diagnosis as the reason the v1-sprint figure is irrecoverable, and ties the cross-dataset arc to the BCF predicate via the ACF@24h ordering (ByteDance 0.489 > Alibaba 0.316 > Bitbrains 0.116).

**Alternatives considered:**
- *Swap numbers, keep framing.* Replace 533/640 and 128/156/157/92 with v4 equivalents (197/640 and 120/55/21/1) and leave the rest of the paragraph alone. Rejected because the original implies population-level dominance, the v4 numbers say the gate isn't crossed anywhere, and reporting just one metric makes the discrepancy invisible. Defence-time the gap between 533/640's implication and `passes_gate=False` would surface eventually.
- *Switch to ml_strict_dominance only (40/35/15/1).* Cleaner narrative ("monotonic decline") but the metric isn't the same as the original's, and a careful reader will notice the silent unit-of-measurement shift (from 160 reactive points to 40 ML configs). Honest substitution requires naming the shift.
- *Drop the dominance claim entirely.* Most conservative but loses the systems contribution Dr. Ho explicitly prioritises.

**Consequences:**
- Ch6 §6.1 paragraph grows from ~80 words to ~280 words. Ch4 Table 4.13 gains a `passes_gate` column. ERRATA-011 row added to track the gate disclosure separately for audit traceability, even though the disclosure prose is embedded in ERRATA-010's substitution.
- Manuscript's strongest systems claim is now qualified rather than headline-grade. The honest framing is consistent with the negative-results pattern adopted elsewhere in the thesis (per memory: foundation models winning 11/12, BiLSTM non-diversity, per-VM Bitbrains sign reversal, imputation rates above 5%).
- Defence preparation must include a clear answer to "if `passes_gate=False` everywhere, what is the operational claim?" Suggested answer: "ML-Proactive secures local Pareto improvements at specific operating points in high-ACF@24h cells, particularly at horizons where the BCF predicate is positive; it does not yet sweep the reactive cloud at the 80% saturation level that would warrant unqualified replacement of reactive HPA."

**Source files:**
- Canonical numbers: `results/bcf_v2/hpa_v4_dominance_per_dataset.csv`
- Gate + dominance definitions: `task_c3_hpa_rebuild_v4.py` lines 256–280
- Saturation diagnosis: `results/bcf_v2/c3_saturation_verdict.md`
- README directing supersession: `results/bcf_v2/README_HPA_CANONICAL.md`

**Y-statement:** In the context of the HPA simulation's headline figure being saturation-confounded and the pre-registered gate failing in all eleven cells, facing the choice between cosmetic number-swap and full retirement, we decided to retire 533/640 entirely with a dual-metric substitution and explicit gate disclosure to achieve defensible operational framing, accepting a ~200 word expansion in Ch6 §6.1.
## DECISION-009 — WPE implementation choice for F2 (Fadlallah weighted PE, m=4, tau=1)

**Date:** 2026-05-26
**Status:** IMPLEMENTED (Day 4)

**Context:** Pre-registration commits to "F2 partial-R²(WPE | ACF@24h) ≥ 0.3" but does not specify which permutation entropy variant. "WPE" in the literature is ambiguous between (a) Bandt-Pompe Weighted Permutation Entropy per Fadlallah, Avolio & Mohamed (2013), where motifs are amplitude-weighted by variance, and (b) plain Permutation Entropy per Bandt & Pompe (2002), where motifs are unweighted. Embedding dimension m and lag tau also need to be chosen.

**Decision:** Use Fadlallah's weighted PE with m=4 and tau=1.

**Rationale:**
- Fadlallah's variance weighting handles cloud workload amplitude jumps better than unweighted PE (large spikes get the weight they merit).
- m=4 is the standard middle-ground in the WPE literature: m=3 is too coarse (only 6 ordinal patterns), m=5+ needs much longer series for stable estimates (5!=120 patterns, sparse coverage on short series).
- tau=1 is appropriate because cloud workloads have autocorrelation at every lag; no need to decimate.
- Implementation reproduces Fadlallah's formula directly (no external library) and was sanity-checked against expected behaviour (constant series → NaN; cross-dataset distributions interpretable in the context of dataset cadence).

**Alternatives considered:**
- *Plain (unweighted) Bandt-Pompe PE.* Rejected because amplitude information is signal in cloud traces — a flat-low motif and a sharp-spike motif have very different operational meaning even if they share an ordinal pattern.
- *m=3 or m=5.* m=3 rejected for coarseness; m=5 rejected because ByteDance series are short (~3,500 points) and Bitbrains is shorter still — 120 patterns would be undersampled.
- *Larger tau (e.g. tau=12 = 1 hour).* Considered but rejected — would conflate WPE with hour-scale predictability that ACF@1h already captures; the point of WPE for F2 is to add information beyond what existing predictability metrics provide.

**Consequences:**
- Within-Bitbrains, WPE and ACF@24h are highly correlated (Spearman rho=+0.76). WPE will add little partial-R² on Bitbrains specifically.
- On Alibaba (rho=+0.12) and ByteDance (rho=+0.28), WPE is closer to orthogonal — F2 has a real chance of clearing the 0.3 threshold there.
- Cross-dataset cadence asymmetry (ByteDance 10-min vs others 5-min) means WPE measures slightly different time scales. Document this disclosure in the F2 chapter when D5 work produces the partial-R² number.

**Y-statement:** In the context of computing a permutation-entropy-based predictability metric for F2, facing literature ambiguity about which PE variant "WPE" denotes, we decided to use Fadlallah's amplitude-weighted PE with m=4 and tau=1 to capture amplitude information relevant to cloud workloads while keeping estimates stable on the shortest series, accepting that the choice locks in a specific definition that the F2 results section must disclose explicitly.