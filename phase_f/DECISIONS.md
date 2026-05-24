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
## DECISION-010 — F2 regression framing: per-cell headline + Bitbrains per-VM robustness

**Date:** 2026-05-27
**Status:** IMPLEMENTED (Day 5)

**Context:** Pre-registration (Dr. Ho written acceptance 2026-05-22) commits
to "F2 partial-R²(WPE | ACF@24h) ≥ 0.3" but does not specify regression
granularity. Three valid framings exist: per-cell pooled (n=12), per-series
pooled (n ≈ 5235), per-dataset (3 separate regressions). D4's WPE-ACF
Spearman finding (Bitbrains +0.7553, Alibaba +0.1231, ByteDance +0.2801) is
post-pre-reg and reveals that pooled and per-dataset framings will give
substantively different answers.

**Decision:** Headline test is per-cell pooled, n=12, using NNLS rows from
`bcf_pairs.csv` as outcome (`delta_pp`) joined to dataset-median WPE and
dataset-median ACF@24h, with `horizon_min` as a covariate. Robustness test
is Bitbrains per-VM pooled, n=568 (142 VMs × 4 horizons), using
`r2_delta * 100` from `bitbrains_per_vm.csv` as outcome with per-VM WPE
and per-VM ACF@24h. Both use cluster-bootstrap 95% percentile CI
(cluster = dataset for headline, cluster = vm_id for Bitbrains). Alibaba
and ByteDance per-series robustness deferred to a future Vast.ai session
pending per-container R² extraction.

**Rationale:**
- Per-cell granularity matches the BCF predicate, which is itself cell-
  level (`ACF@24h > 0.2 AND h ≥ 30min` evaluated at cell granularity).
  Partial-R² at the same granularity is the predictive complement of the
  BCF predicate test.
- Per-series headline was rejected because the per-container R² artefacts
  required for Alibaba and ByteDance do not exist locally — only Bitbrains
  has a per-VM R² file. Headlining at a granularity only one dataset can
  meet would create scope-asymmetry at defence.
- Bitbrains per-VM robustness uses the data that IS local and gives a
  legitimate per-series partial-R² for that one dataset, closing the
  cadence-asymmetry concern at least for the dataset where WPE-ACF
  redundancy (Spearman +0.7553) is most acute.

**Alternatives considered:**
- *Per-series pooled headline (n ≈ 5235).* Rejected — requires Vast.ai
  compute to extract per-container Alibaba/ByteDance R²; would also
  conflate three datasets at three cadences (Bitbrains/Alibaba 5-min,
  ByteDance 10-min).
- *Per-dataset (3 regressions).* Rejected as headline (multiple-testing
  problem with 3 separate threshold tests); the per-dataset breakdown is
  present implicitly in the residual diagnostics step of the script.
- *Pooled with dataset fixed effects.* Rejected — would absorb the
  between-dataset variation that is the only source of WPE+ACF variation
  in the per-cell design, leaving zero usable predictor variance.

**Consequences:**
- F2 reports null under both framings: headline partial-R² = 0.0790
  (CI [0.0000, 0.0790]), Bitbrains per-VM partial-R² = 0.0558
  (CI [0.0007, 0.2126]). Both decisively below 0.30.
- Structural diagnosis: R²_reduced (ACF + horizon, no WPE) = 0.903 in the
  headline. ACF@24h + horizon already explain 90% of cell-level delta_pp
  variance — WPE has nothing left to predict on this corpus.
- WPE coefficient is positive cross-dataset, inverting the M4-literature
  prior that PE captures noise ACF misses. Interpretation for the F2
  chapter: WPE and ACF@24h are partial substitutes on this corpus, not
  complements.
- ByteDance h10 IS in the n=12 design (D2-close incidental finding; the
  "11 cells minus ByteDance h10" framing from earlier planning notes was
  carried wrong into the preflight).
- Bitbrains delta_pp scope in bcf_pairs.csv is OLD pool per-VM median,
  matching the canonical BCF table aggregation rule. If a future scope
  rationalisation regenerates bcf_pairs.csv at NEW pool, F2 numbers
  shift quantitatively but the verdict (both framings well below 0.30)
  is robust.
- F2 null actively informs F1 design: drop WPE as a candidate router
  feature without loss.

**Source files:**
- Outcome: `bcf_pairs.csv` (headline) and `bitbrains_per_vm.csv` (robustness)
- Predictors: `phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv`,
  `omega_{alibaba,bitbrains}.csv`, `bytedance_per_instance_stats.csv`
- Script: `phase_f/scripts/f2_partial_r2.py`
- Outputs: `phase_f/data/f2_partial_r2_results.csv`,
  `phase_f/data/per_series_deltas_bitbrains.csv`

**Y-statement:** In the context of pre-registered F2 partial-R² testing
without specified granularity, facing a post-pre-reg discovery that
WPE-ACF correlation varies sharply across datasets, we decided to
headline per-cell at n=12 (matching the cell-level BCF predicate) with
Bitbrains per-VM robustness as a within-dataset sensitivity test, to
achieve a defensible primary number at the same granularity as the
construct F2 is meant to predict, accepting that the per-cell design's
low effective rank produces a degenerate cluster-bootstrap CI and that
the headline reports a null result under the pre-reg threshold.

---

## DECISION-011 — Q-007 Anchor A: verifier expected updated to NEW pool per-VM medians

**Date:** 2026-05-27
**Status:** IMPLEMENTED (Day 5)

**Context:** `verify_foundation.py` audit stuck at 211/3 since the Phase B
pool transition. The 3 FAILs are in `check_section_11_boundary` (Bitbrains
BCF row), where the verifier's `actual` side (read from
`boundary_condition_table_corrected.csv`) was updated to NEW pool per-VM
medians at some prior session, but the `expected` side (read from
`reports/tables/thesis_numbers.json`) still carries OLD pool literals.
The D4 journal misdiagnosed this as "computed live by verifier"; the BCF
section is in fact a static CSV-vs-JSON comparison, and the CSV had
already been migrated to NEW pool.

**Decision:** Update three keys in `reports/tables/thesis_numbers.json` to
match the CSV's NEW pool values:
- `"Section 11: Boundary Conditions.Bitbrains.delta_30min"`: `"-5.46pp"` → `"+1.53pp"`
- `"Section 11: Boundary Conditions.Bitbrains.delta_120min"`: `"+3.30pp"` → `"-2.81pp"`
- `"Section 11: Boundary Conditions.Bitbrains.verdict"`: `"ML wins only @120min"` → the full descriptive NEW-pool string already present in the CSV.

**Rationale:**
- The CSV is the canonical artefact (anchored to the canonical post-Phase-B
  pipeline output). The JSON is a downstream anchor file for the verifier.
  Aligning JSON to CSV preserves a single canonical scope.
- Per-VM median NEW pool is the production-relevant aggregation per
  documented project norms (memory: "must report BOTH in §5"). Anchor A
  locks the verifier to this scope.
- Alternatives — regenerating the CSV back to OLD pool, or extending the
  verifier to check both pools — were rejected as scope-regression and
  audit-noise respectively.

**Alternatives considered:**
- *Restore CSV to OLD pool, leave JSON as is.* Rejected — undoes a prior
  canonical-scope migration; the post-Phase-B pipeline produces NEW pool
  by default.
- *Switch verifier to read directly from canonical NEW pool source files
  (cross_dataset_headline_v2.csv etc.).* Cleaner but bigger refactor than
  Q-007 scope warrants. Queue for future verifier rationalisation.
- *Keep audit at 211/3, defer indefinitely.* Rejected — 3 stale anchors
  create noise in every audit run and obscure new real failures.

**Consequences:**
- Audit clears 211/3 → 214/0.
- bcf_pairs.csv (BCF canonical input, source for F2 today) is still OLD
  pool per-VM median for Bitbrains — Section 2 verifier and bcf_pairs.csv
  remain internally consistent at OLD pool. Q-007 does NOT touch them
  today.
- Submitted manuscript Ch4 BCF prose + Table 4.10 Bitbrains row still
  cite OLD pool (-5.46/+3.30, "ML wins only at h=120"). ERRATA-012 added
  for D6+ Overleaf application.
- No source-code change to verify_foundation.py. No CSV regeneration.
- File-staging convention: the JSON edit is outside `phase_f/` scope.
  Per SYNC_PROTOCOL Revision 4 (proposed today), surgical patches to
  canonical project files outside `phase_f/` can be committed alongside
  `phase_f/` when they are the direct output of a documented Phase F
  decision. Staged explicitly via `git add reports/tables/thesis_numbers.json`.

**Source files:**
- Modified: `reports/tables/thesis_numbers.json` (3 keys)
- Unchanged: `verify_foundation.py`, `boundary_condition_table_corrected.csv`,
  `bitbrains_summary_corrected.csv`, `bcf_pairs.csv`

**Y-statement:** In the context of a 211/3 verifier audit with the FAILs
concentrated on stale OLD-pool literals in `thesis_numbers.json` (while
the corresponding CSV had already migrated to NEW pool), facing the
choice between aligning the JSON forward to match the CSV or reverting
the CSV backward to match the JSON, we decided to update the JSON to NEW
pool per-VM medians to achieve verifier 214/0 with the post-Phase-B
canonical scope intact, accepting that the submitted manuscript's
Bitbrains BCF row now disagrees with the verifier and requires
manuscript-side correction via ERRATA-012.
---

## DECISION-012 — F1 router design lock (D8)

**Date:** 2026-05-30
**Status:** LOCKED (Day 8); implementation D11+

**Context:** F1 is the second of three pre-registered Phase F tests
(DECISION-005, Dr. Ho written acceptance 2026-05-22), threshold
macro-F1 ≥ 0.55. F2's D5 null (partial-R²(WPE | ACF@24h, horizon) =
0.0790 at headline, well below 0.30) actively informs F1 design by
ruling out WPE as a useful router feature on this corpus. D8 Task B
(per the D6-D8 sequence plan) locks the F1 design BEFORE implementation
begins, to enforce pre-registration discipline analogous to DECISION-010
for F2.

**Decision:** F1 router operates at cell-level granularity (cell =
(dataset, horizon) pair). Four input features:
1. ACF@24h (continuous, dataset-median, 3 distinct values across 12 cells)
2. horizon_min (continuous or one-hot, 4 distinct values: 10/30/60/120)
3. CV (continuous, dataset-median, 3 distinct values)
4. ACF@1h (continuous, dataset-median, 3 distinct values)

Training data: 12 NNLS cells from `results/bcf/bcf_pairs.csv` (includes
ByteDance h10 at delta_pp +6.97 per D5 close handoff lesson #4).
Per-cell label = winner foundation model from
`results/foundation_comparison/leaderboard_v1.csv` (winner tally
Chronos-2:6, TimesFM:3, Granite-TTM:2, NNLS:1).

Evaluation: LOO-cell cross-validation (12 folds, each holding one cell
out), aggregating per-cell predictions into a single 4×4 confusion
matrix, computing macro-averaged F1. Pre-registered threshold:
macro-F1 ≥ 0.55 per DECISION-005. Baseline: always-predict-Chronos-2
yields macro-F1 = 0.167 (computed: per-class F1 = [0, 0.667, 0, 0],
macro-average = 0.167; the D6-D8 prompt's worry that the baseline
might exceed 0.55 is not realised under macro-F1).

Classifier architecture deliberately unlocked at D8; locked at F1
implementation (D11+) among default candidates: multinomial logistic
regression with L2 regularisation, k-NN with k=3, shallow decision
tree (depth ≤ 3). NOT deep neural networks or gradient boosting
(over-parameterised for n=12 training points).

LOO-dataset CV (3 folds, 4 cells held out per fold) reserved as
post-hoc robustness check IF LOO-cell clears 0.55. Not part of the
pre-registered threshold.

**Rationale:**

*Feature selection.*
- **ACF@24h IN.** Load-bearing F1 feature. BCF predicate uses ACF@24h
  (AUC 0.80, p=0.0097, n=36, predicate ACF@24h > 0.2 AND h ≥ 30).
  Demonstrated cell-level ordering of ML benefit at h ≥ 60.
- **horizon_min IN.** Free input; router knows what horizon it is
  routing for. Horizon dominates F2's R²_reduced (0.903 with ACF +
  horizon).
- **CV IN.** cv_stratified_skill.csv shows container-level ML win rate
  varies by CV bin (best at CV 0.3–0.5 at h120: 62.3% win rate). Cell
  level granularity collapses to dataset-median but signal persists.
- **ACF@1h IN.** Provides discriminative power ACF@24h does not.
  Bitbrains has HIGH ACF@1h (0.748) and LOW ACF@24h (0.116), the
  cross-dataset pattern the router must detect (Bitbrains is the
  dataset where ML benefit is small and unstable).
- **WPE OUT.** F2 null implies WPE adds no marginal signal beyond
  ACF@24h + horizon at the cell level. Per F2 outline §7 implication.
- **Hurst OUT.** Low cross-dataset variance (Alibaba 0.775, Bitbrains
  0.997, ByteDance 0.956, range 0.22) gives no discriminative signal
  beyond ACF@24h. Collinear with ACF@24h within-dataset.

*Evaluation protocol.* LOO-cell over k-fold because k=12 is small.
LOO is the small-n classification default. Macro-F1 over accuracy
because the 6/3/2/1 class imbalance makes accuracy a weak baseline
(always-predict-Chronos-2 gets accuracy 0.50 vs macro-F1 0.167); the
gap-to-threshold (0.55 - 0.167 = 0.38) is more discriminative than
accuracy gap (0.55 - 0.50 = 0.05) would be.

*Classifier architecture deferral.* Architecture choice is an
implementation question that depends on what the actual training data
looks like in practice (variance structure, feature-target relationships
not fully visible until features are computed for all 12 cells).
Pre-locking the architecture risks the over-engineering the D5+D7
self-examinations flagged. Defaults named (logistic regression, k-NN,
shallow tree) constrain implementation to small-n-appropriate models.

**Alternatives considered:**

- *Include Hurst.* Rejected for low cross-dataset variance + collinearity
  with ACF@24h. Would add a parameter dimension without informative
  signal.
- *Include WPE.* Rejected per F2 null + DECISION-010 framing implication.
- *Use accuracy not macro-F1.* Rejected because always-predict-Chronos-2
  accuracy is 0.50, making the 0.55 threshold a marginal test.
  Macro-F1 makes the test informative.
- *Use LOO-dataset for pre-reg threshold.* Rejected because LOO-dataset
  with k=3 folds (4 cells held out per fold, training on 8) tests a
  harder generalisation question than the pre-registered scope.
  Reserve as robustness check.
- *Lock classifier architecture at D8.* Rejected — implementation-level
  choice better made at D11+ with visibility into actual feature matrix
  properties.
- *Defer DECISION-012 until F1 implementation start (D11+).* Rejected.
  Locking the design pre-implementation enforces pre-registration
  discipline analogous to DECISION-010 for F2. Defending the F1 design
  post-implementation would invite the same "could have been chosen to
  fit the data" concern that pre-registration is designed to prevent.

**Consequences:**

- F1 implementation can begin D11+ with scope clear; no further design
  questions to answer at implementation time other than classifier
  architecture (constrained to the three named defaults).
- F2's structural concern carries over: n=12 training points with 3
  of 4 features dataset-constant at cell level means F1 essentially
  learns a function of (dataset_fixed_effect, horizon). Honest F1
  outcome expectation range: macro-F1 ∈ [0.30, 0.65]. If F1 nulls in
  [0.30, 0.55], the chapter contribution is structural ("dataset-constant
  features dominate; router signal limited by n=12"), analogous to F2's
  "ACF saturates the predictability axis".
- Baseline computation (always-predict-Chronos-2 = macro-F1 0.167) is
  locked. F1 implementation result reports both macro-F1 against
  threshold and uplift over this baseline.
- LOO-cell vs LOO-dataset distinction is locked in the chapter at
  defence: LOO-cell is pre-registered; LOO-dataset is post-hoc
  robustness. Both numbers reported if they diverge.
- F1 implementation produces, at minimum:
  - `phase_f/scripts/f1_router.py` (training + LOO-cell CV)
  - `phase_f/data/f1_router_predictions.csv` (per-fold predictions)
  - `phase_f/data/f1_router_results.csv` (per-class F1, macro-F1,
    architecture used)
  - `phase_f/journal/d11_f1_implementation.md` or similar

**Source files:**

- Created: `phase_f/journal/f1_prep_scope.md` (D8 Task B output, full
  design rationale)
- Modified: `phase_f/DECISIONS.md` (this entry), `phase_f/THESIS_STATE.md`
  (D8-close refresh)
- Referenced unchanged: `results/bcf/bcf_pairs.csv` (training data
  identification), `results/foundation_comparison/leaderboard_v1.csv`
  (labels), `omega_summary.csv` (feature values — ACF@24h, ACF@1h, CV,
  Hurst dataset-medians; verify exact path at file save),
  `phase_f/journal/f2_chapter_outline.md` (F2 implication anchor)

**Y-statement:** In the context of pre-registered F1 threshold
macro-F1 ≥ 0.55 (DECISION-005), F2's structural null at D5 ruling out
WPE as a router feature, and n=12 cells with three of four candidate
features dataset-constant at cell level, facing the choice between
locking the design pre-implementation (mirror DECISION-010 for F2) or
deferring decisions to implementation start (D11+), we decided to lock
the 4-feature LOO-cell macro-F1 design with classifier architecture
deferred among three named small-n-appropriate defaults, to achieve
pre-registration discipline and disclosable design rationale at
defence, accepting that F1 may report null in the [0.30, 0.55] range
for sample-size and feature-constancy reasons rather than feature-set
reasons, and that the structural null finding would itself be the
chapter contribution.