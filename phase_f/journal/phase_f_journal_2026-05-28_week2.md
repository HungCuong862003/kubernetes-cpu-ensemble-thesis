# Phase F journal — 2026-05-28 — F2 Week 2 methodology hardening (DECISION-019)

**Append to `phase_f/phase_f_journal.md`.**

## What happened

Closed the F2 Week 2 methodology-hardening sprint. Week 1 (D17) had left three
catch22 features crossing the 0.30 threshold on Bitbrains under clip [−1, +1],
disclosed but unresolved. Week 2 set out to (a) defend the F2 null against the
statistical objections a committee would raise, and (b) resolve those
exceedances one way or the other.

A deep-research methodology review found 3 of 5 planned Week 2 tasks aimed at
the wrong target and replaced them (Dr. Ho accepted all ideas 2026-05-28):
- wild-cluster bootstrap at G=3 → identification-limits documentation (G=3 is
  below the G≥15–20 floor; structural non-identification can't be bootstrapped away)
- flat-prior Bayes-R² → hierarchical model + regularised-horseshoe + PSIS-LOO
- Bonferroni → BH-FDR + Westfall-Young (Bonferroni dominated under dependence)
- citation fix: Bandt-Shiha 2007, not Bandt 2005; m=4/WPE(φ) closed form is
  novel research, numerical verification only

## Results — the null held on every front

1. **Structural identification**: 5 arbitrary constant-per-dataset candidates
   all return pr² = 0.078960 (SD = 0.0). Reproduced the locked F2 value exactly.
   Cell-level test is non-identified by design; its null is uninformative, not
   evidence-for-null. Inferential weight transfers to per-series.

2. **BH-FDR + Westfall-Young**: 0/26 candidates survive. SB_TransitionMatrix
   (the strongest, pr² = 0.500) fails even uncorrected α (raw_p = 0.0775),
   WY_p = 0.494. Decisive.

3. **CV-stratification**: under no-clip, active-only stratum, SB_TransitionMatrix
   collapses 0.500 → 0.0147. The exceedances were idle-VM artefacts.

4. **Cluster-leverage**: confirmed the mechanism — same 4–5 idle VMs are
   top-leverage and carry the whole effect.

5. **Hierarchical Bayes**: both Pr(pr² ≥ 0.30) = 0. The standout: SB's ELPD diff
   = −1236 (SE 300), i.e. including the feature *hurts* prediction.

   Bayes was the hardest to run. arviz 1.1.0 (pip pulled pymc 6.0.1) changed
   three APIs vs the 5.x I coded against: ELPD attribute name, az.compare column
   handling, and az.to_netcdf removal. Made all three defensive. The
   SB_TransitionMatrix full model threw 870 divergences at target_accept=0.95;
   bumping to 0.99 cut that to 6 and the partial-R² shifted 0.10 → 0.076 with a
   CI now crossing zero — confirming the 0.10 was a divergent-chain artefact.
   Non-centred reparametrisation of both random intercepts and the horseshoe
   coefficient was essential.

**Analytical backbone** (Bandt-Shiha): PE/WPE are deterministic monotone
functions of φ, and ρ(1) = φ for AR(1), so WPE is analytically redundant with
ACF on the short-memory Gaussian processes that dominate cloud CPU traces.
m=3 closed form verified against Monte-Carlo to max error 0.0021. This explains
*why* the empirical null is not just an accident of these three datasets.

## Decisions

- **DECISION-019** locks the five-front defence, formally dismisses the three
  exceedances (amends D17.4), cancels the SB-as-F1-feature side-experiment
  (amends D17.5), and records the three task replacements + the Bayes sampling
  provenance.

## Corrections logged

- Idle Bitbrains VMs (bb_609–613) are HIGH-CV (bb_609 = 1.85, dataset max), not
  low-CV — near-zero mean inflates CV. Earlier mental model ("low-CV idle VMs")
  was wrong. Chapter prose must say "near-zero mean utilisation producing
  inflated CV."

## What's next

F5 manuscript writing. The F2 chapter §5 now has a clean five-front structure.
No further F2 compute. New bibliography entries to add at chapter-writing time,
batched with the existing biblio audit.

## Time

~1 working session for the six scripts + iterative Bayes debugging + this
documentation pass. Vast.ai instance C.38014225 (stop, don't destroy).
