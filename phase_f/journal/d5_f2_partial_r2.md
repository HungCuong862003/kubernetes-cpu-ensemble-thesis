# D5 — F2 partial-R² test + Q-007 Anchor A (2026-05-27)

casual notes. Vast.ai instance C.37423026.

## scope

Today's pre-reg test: F2 partial-R²(WPE | ACF@24h) ≥ 0.30.
Bundled Q-007 because user flipped run venue to Vast.ai → boot tax
already paid → Q-007 cheap to include.

## decisions

- DECISION-010: F2 framing = per-cell pooled (n=12, headline) +
  Bitbrains per-VM (n=568, robustness). Alibaba and ByteDance
  per-series robustness deferred (need Vast.ai pass to extract
  per-container R² that doesn't exist as a local artefact).
- DECISION-011: Q-007 Anchor A = align verifier expected to NEW pool
  per-VM medians via 3-key update to thesis_numbers.json. CSV side
  (boundary_condition_table_corrected.csv) was already at NEW pool
  from a prior session; only the JSON side was stale.

## what got done

- [x] statsmodels 0.14.6 installed (was missing from /venv/main)
- [x] f2_partial_r2.py saved at phase_f/scripts/ (junior-verbose,
      two framings + cluster-bootstrap CI, ~520 lines)
- [x] paths patched after find: results/{bcf,alibaba,bitbrains,bytedance}/
- [x] wpe_*.csv column rename (wpe_m4_t1 → wpe at load time)
- [x] bb_ prefix strip + str→int coercion on container_id for Bitbrains join
- [x] f2_partial_r2_results.csv + per_series_deltas_bitbrains.csv
      written to phase_f/data/
- [x] Q-007 closed via 3-key JSON update — verifier 211/3 → 214/0
- [x] ByteDance stats duplicate diff: byte-identical (a5e25823a28e10c6970c46e022e32ad4); D4 TBD closed

## F2 verdict

| Framing            | n   | partial-R² | 95% CI            | passes ≥ 0.30 |
|--------------------|-----|------------|-------------------|---------------|
| Headline per-cell  | 12  | 0.0790     | [0.0000, 0.0790]  | NO            |
| Bitbrains per-VM   | 568 | 0.0558     | [0.0007, 0.2126]  | NO            |

F2 reports null under both pre-registered framings.

## findings / surprises

- **R²_reduced = 0.903 in headline.** ACF@24h + horizon explain 90.3% of
  cell-level delta_pp variance without WPE. Threshold 0.30 partial-R²
  requires WPE to capture ~a third of the residual 9.7% — near-impossible
  on a corpus where ACF already does this much work. The failure is
  structural, not methodological. Chapter framing: "ACF@24h saturates the
  cell-level predictability axis on production cloud traces; PE-based
  metrics measure the same axis, not a complementary one."

- **WPE coefficient is POSITIVE (+3.45 headline, +1355 Bitbrains).**
  Inverts the M4-literature prior that PE captures noise ACF misses.
  Cross-dataset, WPE and ACF@24h move together: ByteDance is highest in
  both AND has the highest ML benefit. Honest framing: WPE and ACF@24h
  are partial substitutes on this corpus, not complements. This is the
  most defendable chapter-level contribution from today.

- **Bitbrains per-VM coefficient is outlier-driven.** delta_pp range
  [-1100, +222] inherits clipping artefacts from bitbrains_per_vm.csv's
  ml_r2 column (hard floor around -10). Partial-R² robust to this
  (still 0.056 << 0.30), but the +1355 coefficient itself can't be cited
  in the manuscript. Robust re-fit deferred.

- **Headline CI is degenerate at k=3 clusters.** Bootstrap distribution
  bimodal — point mass at 0 (when sample collapses WPE to constant) and
  point mass at 0.079 (when not). Reported CI [0, 0.079] is artefact;
  point estimate is the right number to cite.

- **bcf_pairs.csv Bitbrains scope is OLD pool per-VM median.** Today's
  F2 inherits this scope. Q-007 today did NOT touch bcf_pairs.csv —
  only the verifier expected JSON. If a future scope rationalisation
  regenerates bcf_pairs.csv at NEW pool, F2 numbers shift quantitatively
  but verdict stays (Bitbrains still drags mean delta_pp negative).

## Q-007 Anchor A close

Diagnosis from D4 was wrong on one point: the BCF Bitbrains anchors are
NOT computed live by the verifier. They're a static CSV-vs-JSON
comparison in check_section_11_boundary (line 332+). The CSV side
(boundary_condition_table_corrected.csv) had already been migrated to
NEW pool by some prior session. The JSON side (thesis_numbers.json) was
still OLD pool. Fix = align JSON to CSV. 3 keys:

- "Section 11: Boundary Conditions.Bitbrains.delta_30min":  "-5.46pp" → "+1.53pp"
- "Section 11: Boundary Conditions.Bitbrains.delta_120min": "+3.30pp" → "-2.81pp"
- "Section 11: Boundary Conditions.Bitbrains.verdict":      "ML wins only @120min" → full NEW pool descriptive string

No verify_foundation.py source edit. No CSV regeneration. No
bitbrains_summary_corrected.csv touched (still OLD pool, untouched
since Apr 30, used by Section 2 which is internally consistent and
passes).

Knock-on: manuscript Ch4 BCF prose + Table 4.10 Bitbrains row are now
stale vs both the CSV and the verifier. ERRATA-012 queued for D6+
Overleaf application.

## blockers / forward implications

- F2 null actively informs F1 design: drop WPE as a candidate router
  feature without loss. Slimmer router, fewer hyperparameters.
- ERRATA-012 for D6+ Overleaf batch.
- Alibaba + ByteDance per-series F2 robustness only needed if F1 also
  fails and the F2 chapter requires more support to defend the null.
  Don't preempt; it risks p-hacking optics.

## off-plan items today (lessons)

1. Session-start step 3 (pull state files from Drive) was skipped.
   Surfaced at EOD sync; recovered via post-hoc pull. Bake into a
   phase_f_session_start.sh wrapper for D6+ to prevent recurrence.
2. Script went through 4 patch iterations (paths, column names, prefix
   strip, dtype coercion) that would have been preempted by schema-
   inspect-first on the input CSVs. Lesson: when writing a script that
   joins files I haven't read directly, inspect schemas first.
3. D4 journal's "computed live by verifier" framing for the BCF check
   was wrong. The grep-first-patch-after rigor caught it. ~5x scope
   reduction on the Q-007 fix vs what the preflight estimated.

## time

started ~17:30, finished ~21:00, total ~3.5 h
