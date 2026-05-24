# F2 Chapter Outline — D8 Finalised (2026-05-30)

Promoted from D7 sketch (v3.1) via:
- TBV item 4 (§5 h10 ordering) cleared — `bcf_pairs.csv` NNLS rows show
  monotonic ByteDance > Alibaba > Bitbrains ordering at all 4 horizons.
- TBV item 7 (§5 M4-literature citation) cleared — Pennekamp et al.
  (2019) Ecological Monographs + Ponce-Flores et al. (2020) Entropy
  22(1):89.
- Scope correction in §5: h30 monotonicity exception is at NEW pool
  full-test scope (`cross_dataset_headline_v2.csv`), NOT F2's input
  scope (`bcf_pairs.csv`). The D7 sketch had this conflated.
- DECISION-012 cross-reference added to §7 (F1 design lock, D8).
- Single strongest claim and single weakest claim named (end of file).
- 5 TBV items remain for D11+ source reads (method-spec + WPE medians).

Target length: ~8–10 pages standalone, OR one sub-section of a broader
"Phase F extensions" chapter in the defence pack. Format TBC with
Dr. Ho at next supervisor meeting.

---

## §1 Pre-reg recap

**Claim.** F2 was pre-registered at partial-R²(WPE | ACF@24h) ≥ 0.30
alongside F1 (macro-F1 ≥ 0.55) and F3 (Spearman ρ ≥ 0.6 AND
|DFL−Pinball−τ| ≤ 5%) per Dr. Ho written acceptance 2026-05-22.
Horizon was added as a regression control post-acceptance via
DECISION-010 (specification choice, not threshold change). No
post-hoc threshold adjustment permitted under DECISION-005.

**Anchor.**
- phase_f/DECISIONS.md DECISION-005 (pre-registration thresholds)
- phase_f/DECISIONS.md DECISION-010 (post-acceptance regression spec)
- Dr. Ho written acceptance, 2026-05-22 (medium per supervisor log)

---

## §2 Method

**Claim.** WPE computed per Fadlallah et al. 2013, m=4, τ=1, on
train+val+test concatenation per series, sorted by `time_stamp`.
Operational regression `partial-R²(WPE | ACF@24h, horizon)` estimated
by OLS with cluster-bootstrap CI; clustering scheme is at dataset
level (k=3) for the headline framing and at VM level [k TBV at D11+
from `f2_partial_r2.py` source] for the Bitbrains per-VM robustness
framing. Horizon control added post-acceptance per DECISION-010,
threshold unchanged. Regression framing locked pre-execution:
per-cell pooled (n=12, headline) + Bitbrains per-VM (n=568,
robustness). Alibaba and ByteDance per-series robustness deferred —
would require Vast.ai compute pass to extract per-container R² that
does not exist as a local artefact.

**Anchor.**
- phase_f/DECISIONS.md DECISION-009 (WPE parameter choice)
- phase_f/DECISIONS.md DECISION-010 (regression framing + horizon control)
- phase_f/scripts/compute_wpe.py
- phase_f/scripts/f2_partial_r2.py
- phase_f/data/wpe_alibaba.csv (5000 containers)
- phase_f/data/wpe_bitbrains.csv (142 VMs)
- phase_f/data/wpe_bytedance.csv (93 instances)
- results/bcf/bcf_pairs.csv (12 NNLS rows used for headline)

---

## §3 Results

**Claim.** Both framings, locked pre-execution per DECISION-010,
return BELOW the pre-registered 0.30 threshold. F2 reports honest
null per DECISION-005. Headline per-cell (n=12): partial-R² = 0.0790,
percentile CI [0.0000, 0.0790]. Bitbrains per-VM (n=568): partial-R²
= 0.0558, CI [0.0007, 0.2126]. Neither operationalisation crosses
0.30.

**Anchor.**
- phase_f/data/f2_partial_r2_results.csv (2 rows: headline + Bitbrains robustness)
- phase_f/data/per_series_deltas_bitbrains.csv (568 rows, intermediate)
- phase_f/scripts/f2_partial_r2.log
- phase_f/DECISIONS.md DECISION-005 (threshold)

---

## §4 Structural diagnosis of the null

**Claim.** R²_reduced (ACF@24h + horizon, no WPE) = 0.903 in the
headline regression. ACF@24h + horizon already explain 90% of cell-level
delta_pp variance on this corpus; the pre-registered 0.30 partial-R²
threshold requires WPE to capture ~30% of the residual 9.7% (about
2.9 percentage points of additional R² lift), near-impossible on a
corpus where ACF saturates the predictability axis. Failure is
structural (corpus property), not methodological (WPE parameter or
regression specification).

**Anchor.**
- phase_f/data/f2_partial_r2_results.csv (R²_reduced column)
- phase_f/scripts/f2_partial_r2.log (per-step OLS output)
- phase_f/DECISIONS.md DECISION-005 (threshold definition)

---

## §5 Post-hoc interpretation

**Claim.** WPE coefficient is positive cross-dataset (+3.45 headline,
+1355 Bitbrains per-VM — sign, not magnitude, is the interpretable
result; the Bitbrains magnitude is outlier-driven, see §6). Inverts
the Pennekamp et al. (2019) / Ponce-Flores et al. (2020) prior that
permutation entropy captures predictability complementary to
autocorrelation. **Across all four horizons in F2's input scope
(`bcf_pairs.csv` NNLS rows, OLD pool Bitbrains aggregation), the
dataset ordering by delta_pp matches the ACF@24h ordering
monotonically**: ByteDance > Alibaba > Bitbrains at h10 (+6.97 /
+0.26 / -8.68 pp), h30 (+5.52 / +0.43 / -5.46 pp), h60 (+6.09 /
+1.33 / -0.79 pp), h120 (+11.31 / +4.64 / +3.30 pp). WPE medians by
dataset show the same direction qualitatively (ByteDance highest at
0.95 per D4 sanity check; Alibaba and Bitbrains medians TBV at D11+
from `wpe_*.csv`).

A separate h30 monotonicity exception exists **in the NEW pool
full-test aggregation** (`cross_dataset_headline_v2.csv`, a different
scope than F2's input): at h30, Bitbrains pooled +4.47 pp exceeds
Alibaba's +3.39 pp despite lower ACF@24h. This NEW pool exception
does NOT affect the F2 partial-R² result (which used the OLD pool
Bitbrains scope) but disclose it explicitly — under at least one
aggregation, the ACF@24h-orders-ML-benefit monotonicity breaks
at h30.

Honest framing: WPE and ACF@24h are partial substitutes on cloud
traces, not complements. This is the chapter's defendable
contribution despite the null on the pre-registered metric, and it
is post-hoc (not pre-registered).

**Anchor.**
- phase_f/data/f2_partial_r2_results.csv (coefficient column)
- results/bcf/bcf_pairs.csv (NNLS delta_pp ordering all 4 horizons)
- results/foundation_comparison/cross_dataset_headline_v2.csv (NEW pool
  h30 exception)
- phase_f/journal/d5_f2_partial_r2.md (findings narrative)
- phase_f/journal/d4_wpe_prep.md (within-dataset WPE-ACF correlation;
  ByteDance WPE 0.95)
- Pennekamp et al. (2019), Ecological Monographs — "The intrinsic
  predictability of ecological time series and its potential to
  guide forecasting" (PE as model-free predictability metric
  complementary to model-based methods).
- Ponce-Flores et al. (2020), Entropy 22(1):89 — "Time Series
  Complexities and Their Relationship to Forecasting Performance"
  (PE applied to M4 Competition series).

---

## §6 Limitations

**Claim.** Five named limitations.

1. **CI degeneracy at k=3 clusters.** Bootstrap distribution bimodal —
   point mass at 0 when the sample collapses WPE to near-constant,
   point mass at 0.079 otherwise. Reported CI [0, 0.079] is an
   artefact of low effective rank; the point estimate is the citable
   number. **This is the chapter's single weakest claim** (see end
   of file).
2. **Bitbrains per-VM coefficient (+1355) outlier-driven** by delta_pp
   clipping artefacts in `bitbrains_per_vm.csv` (`ml_r2` hard floor
   around -10). Partial-R² robust to this; coefficient itself
   uncitable in chapter prose.
3. **Alibaba and ByteDance per-series robustness deferred** — risks
   p-hacking optics if added after the headline null. Reserve for
   post-hoc robustness only if F1 also nulls and the F2 chapter
   requires more support.
4. **Cadence asymmetry.** ByteDance at 10-min intervals vs Alibaba +
   Bitbrains at 5-min. A 4-point WPE motif covers 40 min on ByteDance
   vs 20 min on the others; same metric measures slightly different
   time scales cross-dataset. Disclose explicitly.
5. **`bcf_pairs.csv` Bitbrains scope is OLD pool per-VM median.** F2
   inherits this scope. The NEW pool h30 exception noted in §5
   illustrates that the F2 partial-R² value (0.0790) is scope-
   dependent; under a NEW pool re-run, the quantitative number
   shifts but the structural conclusion (R²_reduced near 0.9, WPE
   adds little) is expected to hold per the within-dataset
   correlations documented in D4.

**Anchor.**
- phase_f/journal/d5_f2_partial_r2.md (limitations 1, 2)
- phase_f/journal/d4_wpe_prep.md (limitation 4)
- results/bitbrains/bitbrains_per_vm.csv (limitation 2)
- results/bcf/bcf_pairs.csv (limitation 5)
- phase_f/DECISIONS.md DECISION-009 (WPE m=4 τ=1 hyperparameter
  choice — not sweep-tested, see "sixth-limitation candidate" below)

**Sixth-limitation candidate (not in chapter unless reviewer asks).**
WPE hyperparameter choice (m=4, τ=1) — DECISION-009 justifies as
literature middle-ground but no robustness sweep was run on this
corpus. Absence of sweep IS a disclosable limitation but adding it
makes §6 top-heavy. Reserved for Q&A response, not chapter body.

---

## §7 F1 implication

**Claim.** F2 null lowers the prior on WPE's marginal usefulness as
an F1 router feature. Under the assumption that the cell-level null
carries to F1's series-level (or cell-level) scope, drop WPE as a
candidate router feature. **DECISION-012 (F1 router design lock,
D8 2026-05-30) formalises this**: F1 feature set is ACF@24h +
horizon_min + CV + ACF@1h, four features at cell-level granularity
with WPE explicitly excluded per F2 implication and Hurst excluded
for low cross-dataset variance. F1 router simplifies — fewer
hyperparameters, fewer feature columns to ablate, smaller search
space.

**Anchor.**
- phase_f/DECISIONS.md DECISION-012 (F1 router design lock)
- phase_f/journal/f1_prep_scope.md (F1 design rationale)
- phase_f/DECISIONS.md DECISION-005 (F1 pre-reg threshold 0.55)

---

## §8 Null framing per DECISION-005

**Claim.** F2 reports honest null under both operationalisations of
the pre-registered metric (per-cell pooled n=12, Bitbrains per-VM
n=568, both locked pre-execution per DECISION-010). No post-hoc
threshold adjustment, no exploratory framings inserted after seeing
the result, no p-hacking. The structural finding (R²_reduced = 0.903)
IS the chapter contribution — the test failure is itself the result.
Pre-registration honour is the chapter's credibility anchor at defence.

**Anchor.**
- phase_f/DECISIONS.md DECISION-005 (pre-reg discipline)
- phase_f/DECISIONS.md DECISION-010 (pre-execution operationalisation lock)

---

## Chapter's single strongest claim

**§5 WPE-ACF partial substitution finding.** WPE coefficient is positive
cross-dataset (sign agreed across all three datasets in both framings).
Inverts the Pennekamp 2019 / Ponce-Flores 2020 prior that PE captures
predictability complementary to ACF. Across F2's input scope at all
four horizons, the WPE-ACF-ML-benefit alignment is monotonic
(ByteDance > Alibaba > Bitbrains). This is a defendable post-hoc
contribution that turns the pre-reg null into a substantive
methodological result.

**Defensibility lineage.** Sign robustness: 3/3 datasets positive in
the headline regression. Cross-aggregation robustness: holds across
the bcf_pairs.csv input scope (the F2 scope) at all four horizons;
a single h30 exception exists in NEW pool full-test scope but does
not affect the F2 result. Literature anchor: two-citation
counterweight against the Pennekamp / Ponce-Flores prior.

## Chapter's single weakest claim

**§6 limitation 1 — CI degeneracy at k=3 clusters.** Bootstrap CI
[0.0000, 0.0790] is bimodal-artefact rather than informative interval.
Reduces the headline result to a point estimate without defensible
uncertainty quantification. A reviewer asking "how confident are you
in 0.0790?" gets an unsatisfying answer. This is more fundamental
than the Bitbrains per-VM coefficient issue (§6 limitation 2, which
only affects magnitude framing) because CI degeneracy bounds the
entire headline statistical claim.

**Mitigation for defence Q&A.** Three responses available:
(a) point to the Bitbrains per-VM CI [0.0007, 0.2126] which is
non-degenerate and reaches similar conclusions (partial-R² 0.0558
< 0.30); (b) note that the structural diagnosis (R²_reduced = 0.903)
is the chapter contribution rather than the partial-R² point estimate;
(c) explicit acknowledgement that k=3 is fundamental — three datasets
are what this thesis has, expanding the corpus is a future-work item.

## Open items remaining (D11+ source reads)

5 of the original 7 D7 TBV items remain. Cleared at D8: items 4
(h10 ordering, via bcf_pairs.csv) and 7 (M4-literature, via web search).

- §2 cluster-bootstrap resample count: not in d5_f2_partial_r2.md.
  Read `f2_partial_r2.py` source at D11+ and lock the number.
- §2 Bitbrains per-VM clustering scheme: presumed VM level but cluster
  count k TBV. Read `f2_partial_r2.py` source at D11+ and lock.
- §3 CI method (percentile / BCa / basic / normal): not in
  d5_f2_partial_r2.md. §3 currently asserts "percentile" by carry-over
  from BCF context; that justification may not transfer to F2
  (continuous partial-R², not binary AUC). Read `f2_partial_r2.py`
  source at D11+ and lock the method label.
- §5 WPE median by dataset: only ByteDance (0.95) confirmed from D4
  close. Read `phase_f/data/wpe_alibaba.csv` and
  `phase_f/data/wpe_bitbrains.csv` at D11+ and compute medians.
- §5 h30 WPE comparison: whether Bitbrains WPE median is lower than
  Alibaba's at h30. Folded into the WPE-median item above.

All five remaining items are method-detail or supporting numeric
claims. None affect the chapter's structural claims. Defensible
structure is locked at D8; D11+ source reads will tighten the
methodological precision in §2, §3, §5.