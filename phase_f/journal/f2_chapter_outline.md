# F2 Chapter Outline — D7 Sketch (2026-05-29)

Structure only, no prose. Eight sections per the D6-D8 sequence plan.
D8 promotes this sketch to defensible structure and identifies single
strongest + single weakest claim. Actual chapter prose drafting deferred
(probably post-F0 depending on F1+ workload).

Target length: ~8–10 pages standalone, OR one sub-section of a broader
"Phase F extensions" chapter in the defence pack. Format TBC with
Dr. Ho at next supervisor meeting.

---

## §1 Pre-reg recap

**Claim.** F2 was pre-registered at partial-R²(WPE | ACF@24h, horizon)
≥ 0.30 alongside F1 (macro-F1 ≥ 0.55) and F3 (Spearman ρ ≥ 0.6 AND
|DFL−Pinball−τ| ≤ 5%) per Dr. Ho written acceptance 2026-05-22. No
post-hoc threshold adjustment permitted under DECISION-005.

**Anchor.**
- phase_f/DECISIONS.md DECISION-005 (pre-registration thresholds)
- supervisor email thread, 2026-05-22 (written acceptance)

---

## §2 Method

**Claim.** WPE computed per Fadlallah et al. 2013, m=4, τ=1, on
train+val+test concatenation per series, sorted by `time_stamp`.
Partial-R²(WPE | ACF@24h, horizon) estimated by OLS with
cluster-bootstrap CI (1000 resamples at dataset level). Regression
framing locked pre-execution: per-cell pooled (n=12, headline) +
Bitbrains per-VM (n=568, robustness). Alibaba and ByteDance per-series
robustness deferred — would require Vast.ai compute pass to extract
per-container R² that does not exist as a local artefact.

**Anchor.**
- phase_f/DECISIONS.md DECISION-009 (WPE parameter choice)
- phase_f/DECISIONS.md DECISION-010 (regression framing)
- phase_f/scripts/compute_wpe.py
- phase_f/scripts/f2_partial_r2.py
- phase_f/data/wpe_alibaba.csv (5000 containers)
- phase_f/data/wpe_bitbrains.csv (142 VMs)
- phase_f/data/wpe_bytedance.csv (93 instances)
- results/bcf/bcf_pairs.csv (12 NNLS rows used for headline)

---

## §3 Results

**Claim.** Both pre-registered framings return BELOW threshold.
F2 reports honest null per DECISION-005. Headline per-cell (n=12):
partial-R² = 0.0790, percentile CI [0.0000, 0.0790]. Bitbrains per-VM
(n=568): partial-R² = 0.0558, CI [0.0007, 0.2126]. Neither framing
crosses 0.30.

**Anchor.**
- phase_f/data/f2_partial_r2_results.csv (2 rows: headline + Bitbrains robustness)
- phase_f/data/per_series_deltas_bitbrains.csv (568 rows, intermediate)
- phase_f/scripts/f2_partial_r2.log

---

## §4 Diagnosis

**Claim.** R²_reduced (ACF@24h + horizon, no WPE) = 0.903 in the
headline regression. ACF@24h + horizon already explain 90% of
cell-level delta_pp variance on this corpus; the pre-registered 0.30
partial-R² threshold requires WPE to capture ~one-third of the residual
9.7%, near-impossible on a corpus where ACF saturates the
predictability axis. Failure is structural (corpus property), not
methodological (WPE parameter or regression specification).

**Anchor.**
- phase_f/data/f2_partial_r2_results.csv (R²_reduced column)
- phase_f/scripts/f2_partial_r2.log (per-step OLS output)

---

## §5 Interpretation

**Claim.** WPE coefficient is positive cross-dataset (+3.45 headline,
+1355 Bitbrains per-VM — sign, not magnitude, is the interpretable
result; the Bitbrains magnitude is outlier-driven, see §6). Inverts
the M4-literature prior that PE captures noise ACF misses. Cross-dataset,
WPE and ACF@24h move together (ByteDance is highest in both and has
the highest ML benefit). Honest framing: WPE and ACF@24h are partial
substitutes on cloud traces, not complements. This is the chapter's
defendable contribution despite the null on the pre-registered metric.

**Anchor.**
- phase_f/data/f2_partial_r2_results.csv (coefficient column)
- phase_f/journal/d5_f2_partial_r2.md (findings narrative)
- phase_f/journal/d4_wpe_prep.md (within-dataset WPE-ACF correlation sanity)

---

## §6 Limitations

**Claim.** Five named limitations.

1. **CI degeneracy at k=3 clusters.** Bootstrap distribution bimodal —
   point mass at 0 when the sample collapses WPE to near-constant,
   point mass at 0.079 otherwise. Reported CI [0, 0.079] is an
   artefact of low effective rank; the point estimate is the citable
   number.
2. **Bitbrains per-VM coefficient (+1355) outlier-driven** by
   delta_pp clipping artefacts in `bitbrains_per_vm.csv` (`ml_r2`
   hard floor around -10). Partial-R² robust to this; coefficient
   itself uncitable.
3. **Alibaba and ByteDance per-series robustness deferred** — risks
   p-hacking optics if added after the headline null.
4. **Cadence asymmetry.** ByteDance at 10-min intervals vs Alibaba +
   Bitbrains at 5-min. A 4-point WPE motif covers 40 min on ByteDance
   vs 20 min on the others; same metric measures slightly different
   time scales cross-dataset. Disclose explicitly.
5. **`bcf_pairs.csv` Bitbrains scope is OLD pool per-VM median.** F2
   inherits this scope. Future rationalisation may regenerate
   `bcf_pairs` at NEW pool; the verdict (Bitbrains drags mean
   delta_pp negative) holds under either scope but quantitative
   numbers shift.

**Anchor.**
- phase_f/journal/d5_f2_partial_r2.md (limitations 1, 2)
- phase_f/journal/d4_wpe_prep.md (limitation 4)
- results/bitbrains/bitbrains_per_vm.csv (limitation 2)
- results/bcf/bcf_pairs.csv (limitation 5)

---

## §7 F1 implication

**Claim.** F2 null actively informs F1 design: drop WPE as a candidate
router feature without loss of predictive signal. F1 router simplifies —
fewer hyperparameters, fewer feature columns to ablate, smaller search
space. F1 prep scope (D8 Task B) operates under this constraint.

**Anchor.**
- forward reference to phase_f/journal/f1_prep_scope.md (D8 deliverable)
- phase_f/DECISIONS.md DECISION-005 (F1 pre-reg threshold 0.55)

---

## §8 Null framing per DECISION-005

**Claim.** F2 reports honest null under both pre-registered framings.
No post-hoc threshold adjustment, no exploratory framings inserted
after seeing the result, no p-hacking. The structural finding
(R²_reduced = 0.903) IS the chapter contribution — the test failure is
itself the result. Pre-registration honour is the chapter's
credibility anchor at defence.

**Anchor.**
- phase_f/DECISIONS.md DECISION-005 (pre-reg discipline)

---

## Notes for D8 promotion

- §1 + §8 may collapse into one short framing section if they feel
  redundant under the pen.
- §5 is the chapter's headline if pulled forward. Consider whether it
  belongs before §3 (interpretation drives results) or stays in current
  order (results drive interpretation).
- §6 limitations 1+2 are method-internal; 3+4+5 are corpus-internal.
  Possible re-split into two sub-headings.
- §7 is a one-paragraph item; may merge into §8 conclusion.
- **D8 task — identify single strongest + single weakest claim:**
  - Currently strongest candidate: §5 (WPE-ACF substitution finding,
    cross-dataset, inverts M4 prior).
  - Currently weakest candidate: §6 limitation 2 (Bitbrains per-VM
    coefficient uncitable) — affects chapter's headline magnitude
    framing.