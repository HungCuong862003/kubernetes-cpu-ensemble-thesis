# D8 — F2 outline finalise + F1 prep scope lock (2026-05-30)

casual notes. local Windows session (no Vast.ai today).

## scope

today's tasks per the D6-D8 sequence plan:
- task A: promote F2 chapter outline from D7 sketch v3.1 to defensible
  structure v4. clear what TBV items can be cleared from this
  environment; flag the rest for D11+. name single strongest + single
  weakest claim.
- task B: F1 prep scope design lock at phase_f/journal/f1_prep_scope.md.
  feature set + training data + evaluation + baseline. NO
  implementation. lock as DECISION-012 if scope firm.

## decisions

- DECISION-012 (F1 router design lock, D8): 4-feature classifier
  (ACF@24h + horizon_min + CV + ACF@1h) at cell-level granularity.
  Training: 12 NNLS cells from bcf_pairs.csv with labels from
  leaderboard_v1.csv (Chronos-2:6 / TimesFM:3 / Granite-TTM:2 /
  NNLS:1). Evaluation: LOO-cell CV, macro-F1 >= 0.55 per DECISION-005.
  Baseline: always-predict-Chronos-2 yields macro-F1 = 0.167.
  Classifier architecture deferred to F1 implementation (D11+) among
  small-n-appropriate defaults. Locked.

## what got done

- [x] read bcf_pairs.csv NNLS rows from project mirror. confirmed
  all-four-horizons monotonic dataset ordering ByteDance > Alibaba >
  Bitbrains in F2's input scope. cleared TBV item 4 (§5 h10 ordering).
- [x] web search for M4-literature citation. found two candidates:
  Pennekamp et al. (2019) Ecological Monographs + Ponce-Flores et al.
  (2020) Entropy 22(1):89. cleared TBV item 7.
- [x] scope correction in §5 of F2 outline: h30 monotonicity
  exception is at NEW pool full-test scope (cross_dataset_headline_v2.csv),
  NOT F2's input scope (bcf_pairs.csv). v3.1 had this conflated; v4
  fixes with explicit dual-scope disclosure.
- [x] promoted F2 outline v3.1 → v4. added DECISION-NNN
  cross-references to all relevant sections per Task A spec. named
  single strongest claim (§5 WPE-ACF substitution finding) and
  single weakest claim (§6 limitation 1, CI degeneracy at k=3).
- [x] drafted f1_prep_scope.md. 4-feature set with both-sides arguments
  for Hurst (OUT) and ACF@1h (IN). LOO-cell evaluation protocol.
  always-predict-Chronos-2 baseline computed at 0.167 macro-F1.
- [x] DECISION-012 drafted with Y-statement and pre-mortem honest
  concerns. Y-statement names the F2-null implication and the small-n
  feature-constancy constraint.
- [x] math precision tightened in §4: "~one-third of residual 9.7%" →
  "~30% (about 2.9 percentage points of additional R² lift)".

## findings / surprises

- **scope confusion was real and material.** v3.1 outline §5 cited
  the h30 monotonicity exception (Bitbrains > Alibaba) without
  distinguishing aggregation scope. bcf_pairs.csv (F2's input,
  OLD pool Bitbrains) shows monotonic ordering at all four horizons.
  cross_dataset_headline_v2.csv (NEW pool full-test) shows the
  exception. these are different aggregations. v4 discloses both
  explicitly. would NOT have been caught without the bcf_pairs.csv
  read at D8 — the prior memory notes referencing "+4.47 pp"
  Bitbrains h30 didn't specify scope, so the v3.1 conflation
  carried through unflagged across 4 self-examination passes.

- **always-predict-Chronos-2 baseline = 0.167, not >= 0.55.** the
  D6-D8 prompt warned "baseline may already exceed 0.55" as an
  honest pre-emptive concern. with macro-F1 (not accuracy), the
  baseline is comfortably below. per-class F1 = [0, 0.667, 0, 0];
  macro-average = 0.167. F1 test is informative as designed.

- **n=12 + 3-of-4-features-dataset-constant is the F1 structural
  constraint.** echoes F2's small-n + ACF-saturation pattern.
  Honest expectation range for F1: macro-F1 ∈ [0.30, 0.65]. could
  null analogously to F2. pre-emptive framing built into
  DECISION-012 Y-statement so a null isn't a surprise.

- **5 TBV items deferred to D11+ are all method-detail or
  supporting-numeric.** none affect the chapter's structural claims.
  defensible structure is locked at D8; D11+ source reads tighten
  precision in §2/§3 plus a supporting WPE-median comparison in §5.

## blockers / forward implications

- D9 (Sunday 2026-05-31): memory snapshot due per weekly cadence.
  capture F2 null + F1 design lock as DECISION-012 + the §5 scope
  distinction.
- D10: F0 close batch. apply ERRATA-012 in Overleaf. resume biblio
  audit (deferred from D6+D7). open any ERRATA-013+ rows from
  biblio findings. last day before F0 → F1 transition.
- D11+: F1 implementation begins per DECISION-012. picks classifier
  architecture from named defaults. reads f2_partial_r2.py source
  along the way to clear remaining F2-outline TBVs.

## off-plan items today (lessons)

- bcf_pairs.csv read was unplanned but high-value. caught the §5
  scope conflation that 4 self-examination passes at D7 missed.
  lesson: read canonical CSVs at finalisation pass even if D7 sketch
  appeared clean. inferences from prior notes accumulate scope
  ambiguity faster than they accumulate factual error.

- writing the "single strongest + single weakest claim" sections
  surfaced that the Bitbrains per-VM coefficient (+1355) limitation
  is actually less fundamental than the CI degeneracy. magnitude
  vs uncertainty is the relevant distinction. weakest = uncertainty
  problem, not magnitude problem.

## time

started ~HH:MM, finished ~HH:MM, total ~X h