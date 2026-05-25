# D11 Journal — Biblio Warm-up + F1 Setup
**Date:** 2026-06-02 (D11)
**Phase:** F1 implementation start

---

## Task A — Biblio anchor warm-up

Five memory-flagged entries verified via web search. Ground truth recorded in
`phase_f/data/biblio_anchor_verification.md`. The `.bib` file is on Overleaf
(not on Vast.ai); per-entry comparison against manuscript requires running
`f1_biblio_verify.py` locally on Windows where the `.bib` lives.

| Key | Verified lead author | Venue | Key watch-for |
|---|---|---|---|
| fremer-pvldb | Hengyu Ye | PVLDB Vol.18 No.11 pp.3812–3825 (2025) | Jiadong Chen leads arXiv version only — not the PVLDB entry |
| fremer-arxiv | Jiadong Chen | arXiv:2507.12908 (2025) | Same paper, different author ordering. Confirm eprint not PVLDB DOI |
| aapa | Guilin Zhang | arXiv:2507.05653 (2025) | Affiliations GWU + Workday + Youngstown State. Code public |
| optscaler | Ding Zou | PVLDB Vol.17 No.12 (2024) | NO public code — delete any url/note field pointing to GitHub |
| hcmiu-qd719 | N/A (regulation) | Internal IU document | Publications optional, NOT graduation gate |

Status: VERIFY_AGAINST_BIB pending until .bib content pasted. No ERRATA-013+
rows opened yet — cannot confirm mismatches without seeing actual .bib entries.

---

## Task B — F1 feature matrix build + structural inspection

Script: `phase_f/scripts/f1_setup.py`
Output: `phase_f/data/f1_feature_matrix.csv`

### Feature matrix (12 rows × 4 features)

| dataset | horizon | acf_24h | horizon_min | cv | acf_1h | label |
|---|---|---|---|---|---|---|
| Alibaba | 10min | 0.316 | 10 | 0.373 | 0.416 | NNLS |
| Alibaba | 30min | 0.316 | 30 | 0.373 | 0.416 | Granite-TTM |
| Alibaba | 60min | 0.316 | 60 | 0.373 | 0.416 | Granite-TTM |
| Alibaba | 120min | 0.316 | 120 | 0.373 | 0.416 | TimesFM |
| Bitbrains | 10min | 0.116 | 10 | 1.056 | 0.748 | Chronos-2 |
| Bitbrains | 30min | 0.116 | 30 | 1.056 | 0.748 | Chronos-2 |
| Bitbrains | 60min | 0.116 | 60 | 1.056 | 0.748 | TimesFM |
| Bitbrains | 120min | 0.116 | 120 | 1.056 | 0.748 | TimesFM |
| ByteDance | 10min | 0.489 | 10 | 0.123 | 0.385 | Chronos-2 |
| ByteDance | 30min | 0.489 | 30 | 0.123 | 0.385 | Chronos-2 |
| ByteDance | 60min | 0.489 | 60 | 0.123 | 0.385 | Chronos-2 |
| ByteDance | 120min | 0.489 | 120 | 0.123 | 0.385 | Chronos-2 |

Tally assertion passed: C2=6, TFM=3, Granite-TTM=2, NNLS=1 ✓

### Critical structural finding

- Matrix rank = 4 (full column rank)
- Spearman ρ = ±1.0 among acf_24h / cv / acf_1h — perfectly rank-correlated
- Only two truly independent discriminative axes: (1) dataset identity
  (any of the three correlated features), (2) horizon_min
- Classifier essentially learns f(dataset_fixed_effect, horizon)
- LOO-cell CV tests within-dataset horizon interpolation, not cross-dataset
  generalisation — the harder question requires LOO-dataset

This finding is chapter material and motivates the dual-baseline comparison (D13).

---

## Infrastructure note — `_paths.py` added

Added `phase_f/scripts/_paths.py` as shared path resolver for all F-task scripts.
All subsequent scripts (`f1_router.py`, future f2_*.py etc.) import from this
rather than defining their own path logic. Prevents the version drift that caused
three failed runs of f1_setup.py today.

## EOD status

- Task A: biblio ground truth verified; .bib comparison deferred to local Windows run
- Task B: feature matrix built and saved; structural inspection complete
- Scripts committed to `phase_f/scripts/`: `_paths.py`, `f1_setup.py`
