# DECISION-007 - UNUSED

Status: UNUSED (resolved 2026-05-23)

## Trigger condition
Submitted PDF (Ensemble_Learning_for_Proactive_Resource_Prediction_PhanNguyenHungCuong_ITDSIU21078.pdf)
contains the literal string `BCa` or any case variant of `bias-corrected accelerated`
in the BCF AUC confidence interval context.

## Verification
Command: pdftotext -layout Ensemble_Learning_*.pdf - | Select-String -Pattern 'BCa|bias[- ]?corrected[- ]?accelerated' -CaseSensitive:$false
Result: empty output. Verified 2026-05-23 on local Windows.

## Outcome
No errata item 6 needed. The submitted PDF does not refer to BCa.

The CI method is not labeled in the submitted PDF (the prose reads
"95\% CI [0.70, 0.88]" without method qualifier). This is silent
correctness rather than deliberate correctness. A methodological
footnote added in the dissertation version of Ch3 §3.8 (with a
reference from Ch4 §4.8) converts this to explicit correctness:

"We report percentile intervals throughout because BCa is degenerate
for a binary classifier of a binary outcome: the acceleration term
\$\hat{a}\$ and the bias-correction \\$ are undefined when the
bootstrap statistic has only finitely many distinct values..."

The footnote enters Overleaf as an enhancement (new content), not as
an errata correction. The submitted PDF needs no change.

## Net effect
Positive for the dissertation version; neutral for the submitted PDF.
The footnote pre-empts the obvious "why percentile not BCa" question
at viva.
