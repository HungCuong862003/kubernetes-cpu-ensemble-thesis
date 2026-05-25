# D15 Journal — F1 Close + D3 Batch Retrospective
**Date:** 2026-06-06 (D15)

---

## Task A — D3 batch retrospective diff

The D3 Overleaf commit (2026-05-25) applied ERRATA-001 through 011 in one
session. To verify the diff:

```bash
# In Overleaf git history, find the D3 commit
git log --oneline | grep "ERRATA-001"

# Diff against its parent
git diff <parent_sha> <d3_sha> -- chapters/

# Expected: exactly 8 loci changed
# Ch3 §3.8 (ERRATA-001..004), Ch3 §3.10 (005),
# Ch4 §4.8 (006), Ch4 §4.11 (007), Ch4 Table 4.13 (010+011),
# Ch5 §5.1 (008), Ch5 §5.6 (009), Ch6 §6.1 (010+011)
```

Jimmy executes in Overleaf; report any unexpected loci as ERRATA-013+ candidates.

## Task B — ERRATA-013+ batch Overleaf commit

Deferred. No ERRATA-013+ rows opened during D11–D15 because the .bib file
comparison against the five anchor entries has not yet been performed (the
.bib file is on Overleaf/local, not on Vast.ai). Biblio audit comparison
is a D16+ carry-over: Jimmy pastes .bib entries into the biblio_anchor_
verification.md template and I flag mismatches.

## Task C — F1 close summary

**F1 implementation is complete.** Work product:

| Artefact | Location | Status |
|---|---|---|
| `_paths.py` | `phase_f/scripts/` | ✅ committed |
| `f1_setup.py` | `phase_f/scripts/` | ✅ committed |
| `f1_router.py` | `phase_f/scripts/` | ✅ committed |
| `f1_feature_matrix.csv` | `phase_f/data/` | ✅ saved |
| `f1_router_predictions.csv` | `phase_f/data/` | ✅ saved |
| `f1_loo_cell_results.csv` | `phase_f/data/` | ✅ saved |
| `f1_loo_dataset_results.csv` | `phase_f/data/` | ✅ saved |
| `f1_dual_baseline_comparison.csv` | `phase_f/data/` | ✅ saved |

**F0 paperwork carry-overs status at D15:**
- ERRATA-012: substitution text fully specified in D13 journal. Jimmy applies
  in Overleaf; not yet APPLIED.
- Biblio audit .bib comparison: D16+ carry-over (requires local .bib paste)
- D3 batch retrospective diff: Jimmy executes in Overleaf per Task A above

**Pre-registered result: F1 NULL (macro-F1 = 0.2532, threshold = 0.55)**

## Task D — Forward agenda (D16+)

Options in priority order, pending discussion with Dr. Ho:

1. **F1 biblio audit completion** — paste .bib entries locally, run
   `f1_biblio_verify.py`, open any ERRATA-013+ rows, apply in Overleaf
2. **F2 chapter writing** (F5 scope) — F2 null result is the finding;
   chapter frames WPE-ACF substitution as the contribution
3. **F3 cost-asymmetric quantile fine-tune of Chronos-2** — pre-reg
   threshold Spearman ρ ≥ 0.6 AND |DFL−Pinball−τ| ≤ 5%; needs GPU
4. **F4 OptScaler/AHPA integration** — systems contribution

Recommend starting with biblio audit (D16, local Windows, no GPU), then
F3 (D17+, needs Vast.ai GPU for Chronos-2 fine-tuning).
