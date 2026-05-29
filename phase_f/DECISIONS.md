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
**Status:** IMPLEMENTED (F3 threshold component SUPERSEDED in part by DECISION-016)

**Context:** Phase F (135-day extension) needed formal scope agreement with supervisor. Written acceptance received 2026-05-22.

**Decision:** Phase F sub-tasks: F1 router (predictability-aware ensemble selection), F2 predictability measures (WPE/Ω/LZ/DFA), F3 cost-asymmetric quantile fine-tune of Chronos-2, F4 integration with OptScaler + AHPA, F5 6 new chapters.

**Pre-registered thresholds:**
- F1: macro-F1 ≥ 0.55 on 11-cell test set
- F2: partial-R²(WPE | ACF@24h) ≥ 0.3
- F3: Spearman ρ ≥ 0.6 AND |DFL−Pinball−τ| ≤ 5%   **[SUPERSEDED — see note below]**

**Supersession note (added 2026-05-26):** The F3 threshold as originally written did not survive the F3 redesign around mean pinball loss at production-realistic operating points. The current F3 thresholds, locked before F3 training began, are: primary mean pinball improvement ≥ 5% at h=60, τ=0.9 across three datasets (DECISION-015 records the close); secondary geometric mean of per-dataset percentage improvements at the same operating point and same thresholds (DECISION-016 locks the secondary). The original Spearman ρ criterion does not appear in the final F3 evaluation. We disclose this in the F3 chapter as a methodology evolution, with the date of supersession (2026-05-26) preceding the F3 training run.

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

---

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

---

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
  has per-VM R² extracted at the OOF stage (576 series × 4 horizons before
  filtering). Lifting Alibaba and ByteDance per-series R² requires a
  full OOF pipeline re-run on Vast.ai, which the timeline does not support
  before Day 6.
- Per-dataset (3 separate regressions) was rejected because partial-R² at
  n ≈ 1750 (Bitbrains) or n ≈ 3500 (Alibaba) per dataset is meaningful but
  the pre-registered threshold (0.3) is calibrated to the structural-finding
  granularity, not to within-dataset n. Reporting three separate per-dataset
  partial-R² values invites a "which one counts?" objection that the per-cell
  framing avoids.
- The Bitbrains per-VM robustness is informative *because* of D4's
  within-dataset WPE-ACF rho = +0.7553 finding: if the headline per-cell
  result clears 0.3 because WPE captures *between-dataset* variation that
  ACF@24h misses, the per-VM (within-Bitbrains) result will be NULL. If the
  headline result is below 0.3 *because* WPE adds nothing beyond ACF@24h at
  the cell level, the per-VM result may or may not be NULL depending on
  whether per-VM WPE captures within-dataset signal. Reporting both is the
  honest disclosure.

**Alternatives considered:**
- *Per-series headline.* Rejected per above (data availability).
- *Average of 3 per-dataset partial-R² values as headline.* Rejected
  (averaging non-commensurable estimates).
- *Bitbrains as headline because it has the per-VM signal.* Rejected
  (cherry-picks the dataset most favourable to a within-dataset WPE-ACF
  rho story; thesis-defence-fragile).
- *Defer all per-VM work to D6.* Rejected (Bitbrains per-VM extracted
  during D5 task A — no incremental cost).

**Consequences:**
- D5 task C (partial-R² computation) runs both regressions and reports
  point estimates + 95% cluster-bootstrap CIs for both.
- F2 chapter §4 reports per-cell as the pre-registered headline; §5
  reports Bitbrains per-VM as planned robustness; §6 reports Alibaba and
  ByteDance per-series as future-work deferral if they can't be lifted by
  D6 end-of-day.
- If per-cell clears 0.3 and Bitbrains per-VM is NULL → "WPE adds
  cross-dataset information beyond ACF@24h, consistent with cadence-
  asymmetry-driven between-dataset variation."
- If per-cell is NULL and Bitbrains per-VM clears 0.3 → "WPE captures
  within-dataset (per-VM) heterogeneity that the dataset-median ACF@24h
  smooths over; the BCF predicate operates at the wrong granularity to
  benefit from WPE."
- If both NULL → "WPE adds no information beyond ACF@24h at either
  granularity tested" (structural conclusion).
- If both clear 0.3 → "WPE clears the pre-registered threshold at the
  pre-registered granularity and at the robustness granularity"
  (positive result).

**Source:** `phase_f/scripts/par_f2_series.py`, `phase_f/data/par_features.parquet`, `bcf_pairs.csv`, `bitbrains_per_vm.csv`.

**Y-statement:** In the context of F2 partial-R² evaluation, facing
multiple valid regression framings and a D4 finding that headline and
robustness framings will likely give different answers, we decided to
run per-cell pooled n=12 as the pre-registered headline and Bitbrains
per-VM as planned robustness to achieve a defensible threshold test plus
mechanistic robustness disclosure, accepting that Alibaba and ByteDance
per-series robustness defers to a future Vast.ai session pending OOF
artefact extraction. Series-level extension (post-hoc, added 2026-05-27). The cell-level F2 partial-R² of 0.079 was extended post-hoc to per-series granularity using the catch22 + DFA + LZC + SampEn feature set computed for PAR. Results recorded in phase_f/data/par_f2_series_results.json (v2 canonical; v1 carried a ByteDance container_id parsing bug that dropped 276 observations and 92 clusters from the regression and prevented the bytedance-only sensitivity from running). Series-level partial-R²(WPE | ACF@24h, horizon_min) = 0.0004, CI [0.0002, 0.0091], n_obs = 17,741, verdict "NULL DEEPENS." R²(reduced) at series scope is 2.56 × 10⁻⁵; the cell-level R²_reduced of 0.903 reflects dataset-identity rather than within-dataset predictability structure. ByteDance-only sensitivity borderline at partial-R² = 0.0433. This extension does not change the cell-level pre-registered outcome; it locates the cell-level result as an aggregation artefact and the series-level result as the stronger structural finding. See DECISION-014 for the close that bundles this with the PAR router outcome.

---

## DECISION-011 — Bitbrains BCF row migration to NEW pool per-VM medians via thesis_numbers.json patch

**Date:** 2026-05-27
**Status:** IMPLEMENTED (Day 5 closeout)

**Context:** Q-007 D5-close anchor A surfaced an internal disagreement
between `thesis_numbers.json` and the source CSV
`boundary_condition_table_corrected.csv` on the Bitbrains BCF row.
The CSV had migrated to NEW pool per-VM medians (h30=+1.53, h120=−2.81,
verdict "ML wins only @30min" per the file's verdict column) in a prior
session, while `thesis_numbers.json` still held the OLD pool literals
(h30=−5.46, h120=+3.30, verdict "ML wins only @120min"). The verifier
(`verify_foundation.py`) reads `thesis_numbers.json` against the CSV
and reported 3 FAILs on the Bitbrains row + verdict pair.

**Decision:** Update `thesis_numbers.json` to NEW pool per-VM medians,
matching the CSV. Three keys patched: `bitbrains_bcf_delta_h30`,
`bitbrains_bcf_delta_h120`, `bitbrains_bcf_verdict`. Verifier cleared
214/0 after the patch (was 211/3).

**Source files patched:**
- `reports/tables/thesis_numbers.json`: 3 keys updated to NEW pool values

**Note from later state-reconciliation review:** the proposed correction
in ERRATA-012 to migrate the manuscript Table 4.10 Bitbrains row to NEW
pool values matches this decision's JSON patch. A 2026-05-28 verification
read of `bitbrains_per_vm.csv` and `boundary_condition_table_corrected.csv`
finds that BOTH FILES report OLD pool values (−5.46/+3.30) for the per-VM
median over 156 VMs. The NEW pool source (the 142-VM filtered per-VM
median producing +1.53/−2.81) is `cross_dataset_headline_v2.csv`, which
is not present in project knowledge. The contents of
`boundary_condition_table_corrected.csv` therefore disagree with what
this decision body claims about that file's scope. Either DECISION-011
was implemented against a different version of the corrected CSV that no
longer exists in the working set, or the patched JSON values are correct
for a NEW pool scope but the CSV cited as the source uses OLD pool. The
F1+ verification window (per memory snapshot) should resolve which is
canonical. Until then, neither ERRATA-012 nor this decision's CSV
citation can be considered settled.

**Consequences:**
- Verifier passes (211/3 → 214/0).
- Submitted manuscript Ch4 BCF prose + Table 4.10 Bitbrains row still
  cite OLD pool (-5.46/+3.30, "ML wins only at h=120"). ERRATA-012 added
  for D6+ Overleaf application; subsequently BLOCKED pending the scope
  resolution noted above.
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
manuscript-side correction via ERRATA-012, and accepting further that a
later verification (2026-05-28) finds the CSV cited here contains OLD
pool values rather than NEW pool, which leaves ERRATA-012 BLOCKED until
the scope is reconciled against `cross_dataset_headline_v2.csv` on Vast.

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

---

## DECISION-013 — PAR pivot: demote F1 cell-level + F3 quantile FT; adopt unified per-series PAR

**Date:** 2026-06-07 (D16)
**Status:** IMPLEMENTED (Dr. Ho Long Van verbal confirmation 2026-06-07)

**Context:**

F1 pre-registered cell-level router (n=12, 4-class, macro-F1 threshold 0.55) returned
NULL at 0.2532 (D12). F2 pre-registered WPE partial-R² test returned NULL at 0.079
(D5). Deep research synthesis (D15/D16) identified a single root cause for both nulls:
cloud workloads sit on an ACF-saturated predictability manifold where all complexity
metrics (WPE, CV, ACF@1h, ACF@24h) collapse to one dimension. This was independently
confirmed by Şen et al. (2024, Methods in Ecology and Evolution) for the WPE-ACF
substitution mechanism, and by Wang et al. (2025, arXiv:2511.08884) for the foundation-
model threshold behaviour. Dr. Ho Long Van confirmed acceptance of the new direction.

**Decision:**

Replace the original F1 cell-level classifier and standalone F3 quantile fine-tuning
with a unified Predictability-Aware Router (PAR) operating at per-series granularity.

Specifically:
1. F1 pre-reg result (NULL, 0.2532) stays in the thesis as the pre-registered finding.
   The cell-level router work is complete and documented. No further cell-level work.
2. PAR is a new post-hoc experiment motivated by the F1+F2 structural diagnosis.
   It implements the per-series routing the original F1 plan (§4) always intended,
   now with corrected features (catch22 + DFA + LZC replacing dataset-level medians).
3. F3 (cost-asymmetric quantile FT of Chronos-2) is demoted from standalone
   contribution to optional ablation within PAR. F3 GPU compute is reallocated to
   PAR per-series feature computation + HPA integration.
4. The unified chapter framing is: BCF (established) + HPA simulation (established)
   + ACF-saturation structural finding (F1+F2 reframed) + PAR (new experiment).

**PAR specification:**
- Per-series features: catch22 (22 features) + DFA/Hurst + Lempel-Ziv Complexity
  + Sample Entropy; computed on all ~5,150 series (Alibaba ~4,900 + Bitbrains 156
  + ByteDance 93)
- Router: shrinkage-LDA (Ledoit-Wolf, sklearn) + XGBoost; trained on per-series
  (series × horizon) pairs, n ≈ 20,600
- Evaluation: Leave-One-Dataset-Out CV (3 folds); regret-based metric (MASE selected
  − MASE oracle) as primary; macro-F1 as secondary
- HPA integration: compare 4 policies — reactive baseline, BCF binary, always-C2, PAR
- Feature library: antropy (DFA, LZC, SampEn), pycatch22
- Timeline: Weeks 1–13 of revised plan (see THESIS_STATE.md)

**Alternatives considered:**
- Re-run F1 cell-level with better classifiers (shrinkage-LDA, NearestCentroid):
  Rejected. Root cause is structural — n=12 with ACF-saturated features cannot support
  4-class discrimination regardless of classifier. Produces another null.
- Keep F3 as standalone contribution: Rejected. Orthogonal to routing story; lower
  marginal value than per-series PAR; 120 GPU-hours better spent on PAR.
- Maintain original F1+F2+F3+F4 plan: Rejected. Both pre-reg tests nulled; continuing
  on the same granularity (dataset cells) while the root cause is per-series granularity
  would produce further nulls with no new insight.

**Consequences:**
- F1 and F2 null results are retained and reframed as the "ACF-saturation structural
  finding" — a positive structural contribution following Karl et al. (ICML 2024)
  NMNR criteria.
- Pivot C+D framing committed: thesis reframed as "When ML Helps Kubernetes Autoscaling:
  Boundary Conditions and Structural Saturation." Six chapters. F3 demoted to optional
  ablation; PAR promoted to candidate empirical headline.
- F3 demotion is later partially reversed by DECISION-014 (PAR close, PARTIAL POSITIVE
  only at series scale): with PAR also demoted to supporting evidence, F3 is re-promoted
  to candidate empirical headline as Chapter 4, evaluated against the new pinball-loss
  pre-registration locked in DECISION-015/016. This is a two-step reorganisation of the
  Phase F empirical hierarchy and the F3 chapter ends up where DECISION-013 wanted to
  put PAR.

**Source files:**
- `phase_f/journal/2026-06-07_d16_par_pivot.md`
- `phase_f/THESIS_STATE.md` (Pivot C+D reframing section)
- Wang et al. 2025 (arXiv:2511.08884) cited in Chapter 3

**Y-statement:** In the context of the F1+F2 structural nulls and the deep-research
identification of ACF saturation as their common root cause, facing the choice between
re-running F1 with better classifiers or pivoting to per-series PAR, we decided to
pivot to per-series PAR with F3 demoted to optional ablation, to achieve a positive
structural contribution at the granularity where the underlying saturation operates,
accepting that F3 standalone work is deprioritised.

---

DECISION-014 — PAR close: PARTIAL POSITIVE at the 0.20 floor; ByteDance failure documented
Date: 2026-05-27 (run date per par_features.parquet; calendar-day labelling deferred to next reconciliation pass)
Status: IMPLEMENTED
Context: DECISION-013 specified three outcome branches for the per-series PAR router. LODO macro-F1 > 0.40 would mark PAR as a headline contribution. The 0.20–0.40 band would mark it partial-positive. Below 0.20 would deepen the ACF-saturation finding at series scope. The router ran under Leave-One-Dataset-Out cross-validation across three folds (test on Alibaba, Bitbrains, ByteDance in turn) with both shrinkage-LDA and XGBoost trained at each fold.
Decision: PAR closes as PARTIAL POSITIVE. XGBoost equal-weight LODO macro-F1 = 0.2166, computed as the mean of per-fold scores 0.2258 (alibaba holdout), 0.1931 (bitbrains holdout), and 0.2310 (bytedance holdout). LDA averages lower at 0.1851. The 0.2166 figure clears the 0.20 floor and sits in the partial-positive band. PAR-XGB is retained as the headline classifier; LDA stays in the chapter as the linear comparison.
Three secondary results lock at the same close.
ByteDance catastrophe. XGB regret reduction relative to always-Chronos-2 reads +4.03 pp on alibaba, +1.14 pp on bitbrains, and −90.90 pp on bytedance. The mean R²-regret across all 189 bytedance test rows is 0.968. A subset of 60 of those 189 series were routed to granite_ttm with mean regret 2.92/row — the bulk of bytedance's failure. Root cause: the training partition for the bytedance holdout fold combines alibaba and bitbrains, both of which contain granite_ttm-favourable cells; the router learns a granite_ttm preference that does not transfer. The mechanism is a granularity mismatch — cell-level routing signal interpreted at series-level inference — not a defect of the feature set.
Series-level F2 partial-R² (post-hoc extension of DECISION-010). par_f2_series_results.json records partial-R²(WPE | ACF@24h, horizon_min) = 0.0004, CI [0.0002, 0.0091], n_obs = 17,741, n_clusters = 4,477. The verdict in the JSON reads "F2 NULL DEEPENS at series scope." R²(reduced) at series scope is 2.56 × 10⁻⁵, against 0.903 at cell scope. The cell-level R²_reduced was a dataset-identity artefact — three datasets, three ACF@24h means, twelve cells, with the regression of delta_pp on ACF reading dataset identity rather than within-dataset predictability variation. The structural saturation claim sharpens at series scope: no predictability metric we measured explains ML benefit at the routing granularity. One sensitivity check sits at the edge: bytedance-only WPE partial-R² = 0.0433, CI [0.0130, 0.1216], borderline against the 0.10 lower threshold. The highest-ACF dataset exits the saturated regime weakly.
PCA structural evidence. PC1 explains 26.4% of total per-series feature variance, and ρ(PC1, ACF@24h) = 0.836. Between-dataset variance on PC1 is 5.9%. Nine principal components are required to reach 80% explained variance. Per-series features therefore exit the ACF-saturation trap that locked F1 at cell scope, but the router gain over always-Chronos-2 is bounded by the same saturation mechanism the F2 series-level result confirms.
Alternatives considered.
Retrain with bytedance-specific holdout strategy. Rejected. The catastrophe is a granularity mismatch, not a sampling defect; ad-hoc holdout adjustment does not address the root cause, and tuning the fold structure on the failed test partition would itself be post-hoc on the evaluation data.
Promote LDA over XGB as headline. Rejected. XGB's per-fold average macro-F1 is higher (0.2166 vs 0.1851), and LDA's bitbrains regret reduction is −17.88 pp against XGB's +1.14 pp on the same fold. LDA's only advantage is interpretability, which is not what PAR is evaluated on under DECISION-013.
Drop PAR from the thesis on the ground that it did not clear 0.40. Rejected per DECISION-013, which specifies all three outcome branches as defensible. The partial-positive outcome combined with the series-level F2 deepening produces a coherent structural finding the chapter can argue.
Consequences.
F5 Ch3 cites PAR-XGB macro-F1 = 0.2166 with the comparison to F1 cell-level (0.2532) printed honestly. The chapter frames PAR as partial against the 0.20 floor, below F1 against the cell-level baseline, and the contradiction between those two readings is itself a structural finding worth a paragraph.
The ByteDance failure receives its own subsection (~1 page target) tracing the 60/189 routing to granite_ttm and the cross-dataset spillover mechanism. The PCA finding follows in the same chapter as evidence that per-series features exit ACF saturation only partially.
The series-level F2 result enters Ch3 as a post-hoc extension of the pre-registered cell-level F2. DECISION-010 acquires a Consequences-section addendum pointing at par_f2_series_results.json (see Part 2 below).
HPA 4-policy comparison (par_hpa_comparison.csv) exists on disk but the headline numbers carried in THESIS_STATE (PAR-XGB 0.1205, AlwaysC2 0.0906) do not match a straight mean over the file (0.231, 0.100). The discrepancy is HPA-integration scope and is deferred to that phase. The PAR block in THESIS_STATE is cleaned to remove the unverified HPA bullets (see Part 3 below).
Y-statement: In the context of the per-series PAR router run under three pre-specified outcome branches per DECISION-013, facing an XGBoost LODO macro-F1 of 0.2166 that clears the 0.20 floor while sitting below the 0.40 headline threshold and below F1 cell-level 0.2532, and a ByteDance fold that fails catastrophically through cross-dataset spillover, we decided to close PAR as PARTIAL POSITIVE with explicit ByteDance failure disclosure and series-level F2 extension included as a downstream structural finding, to achieve a coherent chapter narrative that does not overclaim, accepting that the partial-positive label depends on the pre-reg floor rather than on outperformance of F1.

---

## DECISION-015 — F3 evaluation: SUCCESS on primary, PARTIAL on secondary

**Date:** ~2026-05-28 (close session not visible in working set; date taken from THESIS_STATE)
**Status:** IMPLEMENTED

> **Reconstruction note:** the body below is reconstructed from
> THESIS_STATE evidence (+8.17% primary, 3.44% secondary, v6 canonical,
> epoch 1 best, early stop at epoch 6). The original close-session
> journal is not in the working set. Verify against the actual session
> record on Vast before treating this as canonical.

**Context:** F3 was pre-registered (locked 2026-05-25 at F3.1 baseline
close, with primary criterion defined and secondary recommended) with
two metrics: a primary (mean pinball loss at h=60, τ=0.9 across three
datasets) at SUCCESS ≥ 5% improvement, PARTIAL ≥ 2%, FAILURE < 2%; and
a secondary (geometric mean of per-dataset percentage improvements at
the same operating point, DECISION-016) at the same thresholds. Baseline
value locked at 1.921658. F3 LoRA training and evaluation completed in
a session between 2026-05-25 and 2026-05-28.

**Decision:** F3 is SUCCESS on the primary metric and PARTIAL on the
secondary. F3 confirmed as Chapter 4 empirical contribution within the
Pivot C+D reframing.

**Findings:**

| Metric | Baseline | Fine-tuned | Improvement | Threshold | Verdict |
|---|---|---|---|---|---|
| Primary (mean pinball h=60 τ=0.9) | 1.921658 | 1.764582 | +8.17% | ≥ 5% | **SUCCESS** |
| Secondary (geo mean of per-dataset %) | — | — | 3.44% | ≥ 5% / ≥ 2% | **PARTIAL** |

**Configuration locked as v6 canonical:**
- log1p target transform with cap at 110 (handles Bitbrains heavy-tail outliers
  without losing signal in the normal-range cpu_util_percent regime)
- CFG sort (chronological forecast generation, no future leakage in training
  pair construction)
- Best model: epoch 1
- Early stopping triggered at epoch 6

The model converges within one epoch. We expected three to five before the
training run; observing one is consistent with what the TimesFM quantile-FT
literature reports (Cisana arXiv:2410.11773): asymmetric pinball loss on a
pretrained foundation backbone reaches its best validation pinball within
one to three epochs and then drifts upward as the LoRA adapters overfit the
asymmetry. The early stop at epoch 6 confirms drift after the best epoch.

**Primary-secondary divergence:**

The primary mean was disclosed at F3.1 baseline as Bitbrains-dominated:
Bitbrains contributes 84% (1.620 of 1.922) of the unweighted-mean baseline,
because Bitbrains absolute CPU values are approximately 10× larger than
Alibaba and ByteDance. The +8.17% primary improvement is therefore largely
a Bitbrains improvement, scaled by the Bitbrains baseline weight. The 3.44%
geometric mean of per-dataset percentages reflects more balanced per-dataset
gains, but lands below the 5% SUCCESS threshold and above the 2% PARTIAL.

Both metrics are honest. The primary is the pre-registered decision rule
and stays the official verdict (SUCCESS). The secondary catches scale-invariant
truth and reports PARTIAL. Chapter 4 reports both, with the per-dataset
breakdown that explains the divergence.

**Consequences:**
- F3 confirmed as Chapter 4 of the Pivot C+D thesis. The "primary SUCCESS,
  secondary PARTIAL" framing is defensible: the pre-registered primary
  criterion is met; the scale-invariant secondary lands one threshold below
  the primary because the primary is Bitbrains-weighted.
- Chapter 4 must disclose the Bitbrains scale dominance honestly. The +8.17%
  headline is not misleading once the per-dataset breakdown is shown and
  the secondary's 3.44% is reported alongside.
- v6 retained as canonical model. DECISION-018 (F4 close) is gated on this
  decision: F4 tests v6 against zero-shot Chronos-2 in the HPA simulator,
  and DECISION-018's verdict depends on v6 being the locked fine-tuned model
  here.
- ERRATA-014 (PAR partial-positive framing) remains pending Overleaf
  application; ERRATA-013 (related to PAR) remains pending; the F3 chapter
  draft does not need either to land first.

**Source files:**
- `phase_f/data/f3_finetune_results.json` (per-cell fine-tuned pinball)
- `phase_f/data/f3_zero_shot_baseline.json` (baseline locked 2026-05-25)
- `phase_f/f3_design.md` (pre-registration document with primary + secondary)

**Y-statement:** In the context of pre-registered F3 success criteria
(primary mean pinball ≥ 5% improvement, secondary geometric mean ≥ 5% at the
same operating point), facing primary SUCCESS at +8.17% and secondary PARTIAL
at 3.44% reflecting the Bitbrains scale dominance disclosed at baseline, we
decided to confirm F3 as a Chapter 4 empirical contribution with both metrics
reported, to achieve pre-registration discipline and disclosure of the
primary-secondary asymmetry, accepting that the headline improvement is
Bitbrains-driven and the scale-invariant secondary lands one threshold below
the pre-registered primary.

---

## DECISION-016 — F3 secondary metric: Option C LOCKED (geometric mean of per-dataset percentage improvements)

**Date:** 2026-05-26 (locked before F3 training began)
**Status:** IMPLEMENTED

**Context:** The F3 zero-shot baseline (locked 2026-05-25; see DECISION-015
context for primary criterion) disclosed that Bitbrains contributes 84% of
the unweighted-mean primary metric — 1.620 of 1.922 — because Bitbrains
absolute CPU values are approximately 10× larger than the other two datasets.
Pinball loss is not scale-invariant. The pre-registered primary criterion
has this design flaw: a fine-tune that improves only Bitbrains by 5% clears
SUCCESS without genuinely helping the other two datasets, and a fine-tune
that improves Alibaba and ByteDance by 10% but Bitbrains 0% would report a
~0.8% headline (FAILURE) despite materially helping two-thirds of the corpus.

Three options were tabled at F3.1 close:
- **Option A** — keep pre-reg as is, accept the artifact, disclose Bitbrains
  dominance in the F3 chapter. Risk: defence challenges meaningfulness of an
  unweighted mean of scale-asymmetric quantities.
- **Option B** — renegotiate pre-reg with Dr. Ho Long Van before fine-tune.
  Risk: opening pre-reg looks like p-hacking even when defensible. Rejected.
- **Option C** — keep the pre-registered primary, ADD a scale-invariant
  secondary at the same operating point and same thresholds. Both metrics
  reported. Primary stays the official decision rule.

**Decision:** Option C, locked before F3 training began. The secondary metric
definition:

```
per_dataset_pct(d) = (baseline_d − finetune_d) / baseline_d × 100
secondary = (∏_{d ∈ {alibaba, bitbrains, bytedance}} per_dataset_pct(d))^(1/3)
```

Same thresholds as the primary:
- SUCCESS: ≥ 5%
- PARTIAL: ≥ 2%
- FAILURE: < 2%

Both metrics reported regardless of outcome. The primary remains the official
decision rule per DECISION-005 pre-registration discipline; the secondary
catches scale-invariant truth without retroactively overriding the primary.

**Rationale:**

Geometric mean is appropriate because percentage improvements compound
multiplicatively across datasets and the geometric mean is the unbiased
aggregator for ratio quantities (Fleming-Wallace 1986). Arithmetic mean of
ratios is biased toward the largest ratio, which would partially reproduce
the Bitbrains-domination problem we are trying to avoid. Per-dataset
percentages computed at h=60, τ=0.9 keep the secondary measured at the same
operating point as the primary, which makes the two metrics directly
comparable rather than testing different aspects of the model.

**Alternatives considered:**
- *Arithmetic mean of per-dataset percentages.* Rejected per the bias-toward-
  largest-ratio argument above.
- *Bootstrap per-dataset CIs and require all three to clear 2% individually.*
  More rigorous, but with only three datasets the bootstrap is degenerate at
  the dataset level — no within-dataset resampling structure here that is not
  already captured in the per-cell pinball means. We would be bootstrapping
  three numbers.
- *Drop ByteDance from the secondary because its absolute pinball values are
  the smallest and its percentage improvement is most sensitive to small
  absolute changes.* Rejected: dropping a dataset after baseline disclosure
  looks like cherry-picking. Better to keep all three and disclose any
  sensitivity in Chapter 4.

**Consequences:**
- DECISION-015 reports both primary and secondary at F3 evaluation time.
- F3 design document (`phase_f/f3_design.md`) updated to include the secondary
  definition before fine-tune training begins.
- DECISION-005's F3 threshold (Spearman ρ ≥ 0.6 AND |DFL−Pinball−τ| ≤ 5%) is
  superseded in part: the current F3 criteria are the primary defined at
  DECISION-015 and this secondary, both expressed in mean-pinball-improvement
  terms. The original Spearman criterion did not survive F3 redesign around
  pinball loss. DECISION-005 carries a supersession note pointing to this
  decision.

**Source files:**
- `phase_f/f3_design.md` (pre-registration document, secondary section appended)
- `phase_f/data/f3_zero_shot_baseline.json` (baseline numbers feeding the secondary)
- `phase_f/journal/2026-05-25_f3-day1.md` (Option C recommendation as F3.1 close)
- `phase_f/journal/2026-05-26_f3-day2.md` or equivalent (Option C lock — verify path)

**Y-statement:** In the context of an F3 baseline showing the unweighted-mean
primary metric dominated 84% by Bitbrains because pinball loss is not
scale-invariant across datasets with 10×-different absolute CPU values,
facing the choice between accepting the scale artifact (Option A) or
renegotiating pre-registration (Option B) or adding a scale-invariant
secondary (Option C), we decided to lock Option C — keep the primary as the
official decision rule, add a scale-invariant secondary at the same
operating point and thresholds — to achieve disclosure of scale-invariant
truth without retroactively overriding pre-registration, accepting that the
F3 chapter reports two metrics that may disagree on outcome category and
that the primary stays the headline.

---

## DECISION-017 — Omega post-hoc validation: canonical findings + partial correlation retracted

**Date:** 2026-05-28
**Status:** IMPLEMENTED
**Amended:** D17.4 and D17.5 amended by DECISION-019 (see amendments at end of this entry)

**Context:** Following the F1/PAR structural saturation diagnosis, Wang et al. (2025,
arXiv:2511.08884) was cited as independent validation that spectral predictability Omega
stratifies TSFM fitness. A post-hoc analysis was run to test whether Omega vs ML-benefit
correlations on our corpus are consistent with Wang et al.'s claim, and whether Omega
adds information beyond ACF@24h.

**Script:** phase_f/scripts/par_omega_validation.py
**Outputs:** phase_f/data/par_omega_{cell_results, vm_results, scatter, validation_report}

**Findings (canonical):**

Cell-level (n=48: 4 models × 3 datasets × 4 horizons):
- Raw ρ(Omega, delta_pp) = +0.090, p=0.54 → **NULL** (Omega does not generalise cross-dataset)
- Raw ρ(ACF@24h, delta_pp) = +0.449, p=0.0014 → significant (only generalising feature)
- Partial ρ(Omega | ACF@24h) = +0.370, p=0.0097 → **RETRACTED** (see below)

Per-VM Bitbrains (n=624: 156 VMs × 4 horizons):
- ρ(Omega, ΔR²) = −0.507, p<0.001 → **ROBUST** — consistent across all four horizons
  (10min −0.414 / 30min −0.518 / 60min −0.537 / 120min −0.525)
- ρ(ACF@24h, ΔR²) = −0.295 (weaker than Omega within Bitbrains)

**Partial correlation retraction:**
The partial ρ = +0.370 was tested via jackknife (leave-one-dataset-out):
- Leave out Alibaba (n=32): partial ρ = 0.020, p=0.912
- Leave out Bitbrains (n=32): partial ρ = 0.000, p=1.000
- Leave out ByteDance (n=32): partial ρ = −0.020, p=0.912

The result collapses to exactly zero under every jackknife fold. Cause: with 3 unique
Omega values and 3 unique ACF@24h values, removing one dataset leaves 2 points per
feature — too few to estimate a partial effect. The 0.370 is a fixed geometric consequence
of the 3-dataset triangular arrangement, not a sample estimate of a population parameter.
PSR robustness check (Liu et al. 2018, Biometrics 74:595): rho=0.317, difference=0.053
(exceeds 0.05 threshold). Both checks confirm the partial result is not defensible.
Suppressor variable framing (Conger 1974) DROPPED.

**Decision:** Partial correlation claim dropped. Chapter uses only:
(1) cell-level raw ρ(Omega) = 0.090, null;
(2) cell-level raw ρ(ACF@24h) = 0.449, significant;
(3) per-VM ρ(Omega within Bitbrains) = −0.507, robust.

**Y-statement:** In the context of the structural saturation chapter, facing the need
to ground the ACF-saturation finding in external literature (Wang et al. 2025), we
decided to run a post-hoc Omega validation analysis and report only the robust per-VM
result and null cross-dataset result, dropping the jackknife-fragile partial correlation,
to achieve a defensible and honest characterisation of Omega's role on cloud workloads,
accepting that the chapter cannot claim Omega adds information beyond ACF@24h.

**Chapter framing (canonical sentence):**
"Wang et al. (2025, arXiv:2511.08884) found Omega stratifies TSFM fitness on
general-purpose benchmarks. On cloud workloads, the cross-dataset relationship is null
(ρ=+0.09, p=0.54), while within Bitbrains the direction inverts (ρ=−0.51, p<0.001):
higher spectral concentration on server VM traces reflects short-lag mean-reversion
already captured by naive persistence, not diurnal regularity that learned models can
leverage. ACF@24h — which isolates the 24-hour periodic component — is the only feature
that generalises across workload classes (ρ=+0.45, p=0.0014) and aligns with the BCF
predicate threshold."

### D17.4 — AMENDED 2026-05-28 by DECISION-019

The three exploratory exceedances (SB_TransitionMatrix 0.500, SP_Summaries
0.374, DN_OutlierInclude 0.309 on Bitbrains under clip [−1, +1]) are no longer
merely "disclosed and not promoted." Week 2 formally DISMISSED them as
multiple-comparison + idle-VM artefacts on four independent grounds (BH-FDR +
Westfall-Young, CV-stratification active-only, cluster-leverage, hierarchical
Bayes). See DECISION-019 (D19.1, D19.2). The "uniform null" narrative is
restored and strengthened.

### D17.5 — AMENDED 2026-05-28 by DECISION-019

The proposed Week 5–8 side-experiment to test SB_TransitionMatrix as a 5th F1
router feature is CANCELLED. The feature failed every multiple-comparison
correction and the hierarchical Bayes shows its inclusion reduces out-of-sample
predictive density (ELPD diff = −1236, SE 300). F1 stays at 4 features. See
DECISION-019 (D19.3).

---

## DECISION-018 — F4 close: MPC + HPA simulation Pareto verdict per dataset

**Date:** ~2026-05-28 (close session not visible in working set; date taken from THESIS_STATE)
**Status:** IMPLEMENTED

> **Reconstruction note:** the body below is reconstructed from THESIS_STATE
> evidence and the ERRATA register. The original close-session journal entry
> is not in the working set. Verify against the actual session record on Vast
> before treating this as canonical. We are particularly uncertain about the
> v7 routing-infrastructure provenance — THESIS_STATE refers to "v7 routing
> infra retained as reproducibility artefact" without giving the design, and
> we have not attempted to reconstruct that detail.

**Context:** F4 (MPC + HPA simulation integration of the F3 fine-tuned model
against reactive baselines and the v6 zero-shot Chronos-2 model) tested whether
the LoRA-fine-tuned Chronos-2 v7 produces SLO-violation/cost-Pareto improvements
over the v6 zero-shot baseline at production-realistic operating points.
Pre-registration committed to strict Pareto dominance at the per-dataset cell
level: v7 retained as canonical only if both violation and cost improve. Pre-
registration SHA `012786af9d` recorded in git history.

**Decision:** v6 retained as canonical model. v7 routing infrastructure retained
as reproducibility artefact but not promoted to canonical. The F4 outcome is
mixed across datasets and does not meet the strict Pareto criterion:

| Dataset | n | Violation Δ (pp, 95% CI) | Cost Δ (pp, 95% CI) | Verdict |
|---|---|---|---|---|
| Alibaba | 14,760 | −1.90 [−2.19, −1.60] sig | +0.22 [+0.09, +0.35] sig | **Pareto tradeoff** |
| Bitbrains | 426 | −0.23 sig | −0.13 [−0.42, +0.14] non-sig | Partial improvement |
| ByteDance | 279 | (null) | (null) | Null |

**Strict Pareto pre-registration mandates v6 retention.** Alibaba shows a clean
tradeoff: violation falls and cost rises, both significant; neither variant
dominates the other at the population level. Bitbrains shows partial
improvement: violation falls significantly, cost falls but the CI crosses zero
(non-significant). ByteDance shows no signal in either direction. With the
strict Pareto criterion violated on Alibaba (the largest dataset by sample
weight) and the CI crossing zero on Bitbrains' cost side, neither v6 nor v7
dominates v7's results on a strict Pareto reading.

**Findings:**

The dataset-by-dataset pattern is itself the F4 contribution. Alibaba's
tradeoff is not a defect of the fine-tuned model — it is a property of the
workload. Alibaba's strong diurnal cycle (ACF@24h 0.316) produces predictable
peaks that the fine-tuned model bids ahead of, reducing violations at the
cost of brief over-provisioning during those bids. Bitbrains' partial
improvement aligns with its mixed cell-level ACF@24h profile (median 0.116,
some VMs higher). ByteDance's null is consistent with already-high
zero-shot performance leaving limited room for fine-tune gains.

The trajectories file `f4_trajectories.parquet` (30,933 rows) captures the
per-decision-point violation and cost trace, which is the raw input for any
Chapter 5 figure or statistical re-analysis.

**Consequences:**
- v6 (zero-shot Chronos-2 in the F3 evaluation pipeline) remains the canonical
  Phase F model for downstream comparisons. v7 (the F3 fine-tuned model from
  DECISION-015) is retained in repo for reproducibility but not the
  recommended deployment configuration.
- Chapter 5 framing: "F3 fine-tune produces dataset-dependent Pareto behaviour;
  Alibaba shows a tradeoff, Bitbrains shows partial improvement, ByteDance is
  null. v6 retained per pre-registration." The chapter must disclose the
  three per-dataset outcomes and the strict-Pareto pre-registration that
  produced the v6-retention verdict.
- `f4_trajectories.parquet` is the canonical raw output for Chapter 5 figures
  and statistical tests.
- ERRATA-015 filed for any submitted-PDF Chapter 5/6 prose claiming uniform
  improvement across the F3 → F4 transition.
- ERRATA-016 filed (companion record; we did not verify the specific row
  contents from the working set).

**Source files:**
- `results/bcf_v2/f4_trajectories.parquet` (30,933 rows, canonical raw output)
- `results/bcf_v2/hpa_v4_dominance_per_dataset.csv` (dominance counts, v4 protocol)
- Pre-registration commit `012786af9d` (git history)

**Y-statement:** In the context of testing whether the F3 fine-tuned v7 strict-
Pareto-dominates the v6 zero-shot model at production operating points, facing
dataset-by-dataset mixed results (Alibaba tradeoff with both metrics significant,
Bitbrains partial improvement with one metric's CI crossing zero, ByteDance
null in both metrics), we decided to retain v6 as canonical per the strict
Pareto pre-registration and keep v7 routing infrastructure as a reproducibility
artefact, to achieve consistent pre-registration discipline across all Phase F
threshold-gated decisions, accepting that Chapter 5 reports a tradeoff finding
rather than a strict-dominance finding and that the F3 → F4 chapter transition
must disclose the per-dataset asymmetry honestly.

---

## DECISION-019 — F2 Week 2 methodology hardening: five-front null defence + exceedances formally dismissed

**Date:** 2026-05-28
**Phase:** F Week 2 close
**Status:** IMPLEMENTED
**Supersedes:** nothing
**Amends:** DECISION-017 (D17.4 and D17.5 — see amendment block at end of DECISION-017)
**Relates to:** DECISION-005 (F2 pre-reg), DECISION-009 (WPE method), DECISION-010 (F2 framings), DECISION-013 (PAR pivot)

### Context

DECISION-017 (Week 1) closed the F2 per-series extension but left a loose end:
three catch22 features crossed the 0.30 threshold on Bitbrains under clip
[−1, +1] — SB_TransitionMatrix_3ac_sumdiagcov (0.500), SP_Summaries_welch_rect_area_5_1
(0.374), DN_OutlierInclude_p_001_mdrmd (0.309). D17.4 disclosed these as
exploratory and declined to promote them, but did not formally dismiss them.
D17.5 flagged a possible Week 5–8 side-experiment to test whether
SB_TransitionMatrix should join the F1 router as a 5th feature.

A methodological verification (deep-research review, Dr. Ho accepted all ideas
2026-05-28) found three of five planned Week 2 tasks aimed at the wrong target
and replaced them. Week 2 then executed six analyses that together either
dismiss the exceedances or explain the null analytically.

### Decisions

#### D19.1 — Five-front defence of the F2 null is the canonical Week 2 product

The F2 null (predictability metrics add no incremental information over ACF@24h
and horizon) is defended on five independent empirical fronts plus one analytical
explanation. All six are canonical for §5 of the F2 chapter:

1. **Structural identification** (`identification_limits.py` → `identification_limits.md`):
   the cell-level n=12 test is non-identified for any within-dataset-constant
   regressor. Five arbitrary constant-per-dataset candidates all return
   partial-R² = 0.078960 (SD = 0.0, machine precision). Reproduces the locked
   F2 value (0.0790) exactly. Ibragimov-Müller t(2) = −0.34, p = 0.766
   (no power by design; per-dataset WPE slopes heterogeneous and sign-discordant:
   Alibaba −0.26, Bitbrains −4.19, ByteDance +2.48).

2. **Multiple-comparison correction** (`fdr_corrected_screen.py` →
   `fdr_corrected_screen.csv/.md`): NONE of 26 exploratory candidates survives
   Benjamini-Hochberg FDR at q = 0.05 (all BH_q = 1.0000) or Westfall-Young
   step-down max-T (all WY_p > 0.49). SB_TransitionMatrix raw_p = 0.0775 fails
   even uncorrected α = 0.05; WY_p = 0.494. WPE confirmatory NULL at 0.1844 on
   the Bitbrains clip subset.

3. **CV-stratification** (`cv_stratification.py` → `cv_stratification.csv/.md`):
   under NO clip, in the active-only stratum (137 VMs, idle VMs excluded),
   SB_TransitionMatrix collapses 0.500 → 0.0147, SP_Summaries → 0.0005,
   DN_OutlierInclude → 0.0236, WPE → 0.0177. None approaches 0.30. The
   exceedances are entirely an artefact of the 5 idle VMs.

4. **Cluster-leverage diagnostics** (`loo_cluster_leverage.py` →
   `loo_cluster_leverage*.csv/.md`): for every exceedance, the same 4–5 idle
   VMs (bb_609–613) are simultaneously the top cluster-leverage points AND the
   VMs whose removal most decreases partial-R² (SB_TransitionMatrix: dropping
   bb_611 −0.078, bb_613 −0.076, bb_612 −0.060). Single-VM LOO keeps "100%
   above threshold" only because dropping one idle VM leaves the other 3–4;
   the CV-stratification (removes all 5 at once) is the correct combined test.

5. **Hierarchical Bayesian model comparison** (`hierarchical_bayes.py` →
   `hierarchical_bayes_loo.csv/.md`): random dataset intercepts +
   regularised-horseshoe candidate prior (Piironen-Vehtari 2017), PSIS-LOO.
   Both candidates Pr(partial-R² ≥ 0.30) = 0.0000. WPE: partial-R² = 0.0000
   [−0.016, 0.016], clean sampling. SB_TransitionMatrix: partial-R² = 0.0757
   [−0.010, 0.116] (CI crosses zero), and ELPD diff = −1236.5 (SE 300) —
   **adding the feature actively degrades out-of-sample predictive density.**

**Analytical explanation** (`bandt_shiha_ar1_numerical.py` →
`bandt_shiha_ar1_numerical.md`): for AR(1), ρ(1) = φ exactly, so φ is identified
by ACF alone; PE and WPE are deterministic monotone functions of φ (verified
numerically: PE = 1.0000 at φ=0 → 0.871 at |φ|=0.95; m=3 closed form matches
Monte-Carlo to max error 0.0021). On AR(1)-like short-memory Gaussian processes —
the regime most CPU-utilisation series occupy — WPE is analytically redundant
with ACF. This is *why* the empirical null holds.

#### D19.2 — The three Week 1 exceedances are FORMALLY DISMISSED

Amends D17.4. The exceedances are not merely "disclosed and not promoted" — they
are formally dismissed as multiple-comparison + idle-VM artefacts, on four
independent grounds (D19.1 fronts 2, 3, 4, 5). Chapter §5 states this directly:
"Of 26 exploratory predictability features screened, three crossed the 0.30
threshold on Bitbrains under a [−1, +1] clip. None survives Benjamini-Hochberg
FDR or Westfall-Young correction; all three collapse to below 0.025 once the
five near-zero-mean-utilisation VMs are excluded; cluster-leverage diagnostics
confirm those same VMs drive the entire effect; and a hierarchical Bayesian
model finds zero posterior probability of any candidate crossing the threshold.
The exceedances are artefacts, not signal."

#### D19.3 — F1 router stays at 4 features; SB_TransitionMatrix side-experiment CANCELLED

Amends D17.5. There is no case to add SB_TransitionMatrix (or any catch22
feature) to the F1 router. It failed every correction; the hierarchical Bayes
shows its inclusion *reduces* predictive density (ELPD −1236). The Week 5–8
side-experiment flagged in D17.5 is cancelled. F1 stays at 4 features
(ACF@24h, horizon, CV, ACF@1h), already closed NULL at D15/D12.

#### D19.4 — Three Week 2 task replacements locked (methodology)

The methodological verification replaced three planned tasks; these replacements
are canonical:

- **Wild-cluster bootstrap at G=3 → identification-limits documentation.**
  MacKinnon-Webb (2018) require G ≥ 15–20 for size control; structural
  non-identification cannot be cured by resampling. Cite Mundlak (1978),
  Ibragimov-Müller (2016), Snijders-Bosker (2012).
- **Flat-prior Bayes-R² → hierarchical model + regularised-horseshoe + PSIS-LOO.**
  Cite Piironen-Vehtari (2017), Vehtari-Gelman-Gabry (2017), Gelman et al. (2019).
- **Bonferroni k=26 → BH-FDR + Westfall-Young max-T.** Bonferroni is dominated
  by BH-FDR under the positive dependence among catch22 features. Cite
  Benjamini-Hochberg (1995), Benjamini-Yekutieli (2001), Westfall-Young (1993).

Plus one citation correction: the ordinal-pattern derivation cites
**Bandt & Shiha (2007)**, *J. Time Series Analysis* 28:646–665 — NOT Bandt (2005).
The m=4 / WPE(φ) closed form does not exist in elementary functions; numerical
verification only (out of scope for a bachelor's thesis to derive).

#### D19.5 — Sampling-provenance disclosure for the Bayesian run

The hierarchical Bayes canonical CSV (`hierarchical_bayes_loo.csv`) consolidates
two runs at different sampler settings, recorded in its `target_accept`,
`n_divergences`, and `sampling_note` columns:
- WPE at target_accept = 0.95 (clean, 0 divergences)
- SB_TransitionMatrix at target_accept = 0.99 (6 divergences, down from 870 at
  0.95; minor R-hat/ESS warnings persist on the horseshoe hyperparameters
  tau/lam/c2, which is expected and does not affect the b_cand / sigma / R²
  conclusions). The residual non-convergence is itself consistent with the
  feature's near-degenerate distribution — the same pathology evidenced at
  cell level (structural non-identification) and series level (idle-VM leverage).

The chapter discloses this in a methodology footnote; it does not weaken the
conclusion, which is corroborated on four other fronts.

### Consequences

- F2 chapter §5 gains a "five-front defence" structure; the exploratory
  exceedances move from "open disclosure" to "dismissed with evidence."
- F1 router scope is final at 4 features; no further router experiments.
- Six new Week 2 scripts + six output artefacts enter the repository
  (`phase_f/scripts/`, `phase_f/data/week2/`).
- No manuscript number changes → no new erratum required.
- New citations enter the bibliography (see D19.4); to be added at F5 chapter
  writing, batched with the existing bibliography audit.

### Y-statement

In the context of an F2 null finding with three loose-end exploratory
exceedances, facing a committee statistician's likely objections (multiple
comparisons, small-cluster inference, R²-as-outcome instability, idle-VM
leverage), we decided to defend the null on five independent fronts plus an
analytical explanation and formally dismiss the exceedances, to achieve a
viva-robust F2 chapter, accepting ~2 weeks of additional methodology work and a
modest set of new bibliography entries.