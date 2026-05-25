# Memory Snapshot — 2026-06-06 (D15, F1 close)
**Cadence:** weekly (D9 = 2026-05-31 → D15 = 2026-06-06)
**Purpose:** capture F1 implementation findings for continuity across sessions

---

## What happened D11–D15 (F1 implementation)

F1 router implementation complete. Pre-registered result: **NULL**.

### F1 core numbers (canonical, verified on Vast.ai 2026-06-03/04)

| Metric | Value | Source |
|---|---|---|
| Architecture | DecTree-depth3 | f1_router.py sweep |
| LOO-cell macro-F1 | **0.2532** | `phase_f/data/f1_loo_cell_results.csv` |
| Pre-reg threshold | 0.55 | DECISION-005 |
| Pre-reg outcome | **NULL** | 0.2532 < 0.55 |
| LOO-dataset macro-F1 | **0.1000** | `phase_f/data/f1_loo_dataset_results.csv` |
| Structural baseline (ds+h) | 0.2253 | `phase_f/data/f1_dual_baseline_comparison.csv` |
| Full vs structural delta | **+0.028** | same file |
| Trivial baseline | 0.167 | always-Chronos-2, DECISION-012 |

### F1 structural diagnosis (chapter contribution)

Three root causes for the null — all must be disclosed in F5 chapter writing:

1. **Dataset-constant features.** ACF@24h, CV, ACF@1h all have Spearman ρ = ±1.0
   with each other — they bijectively encode dataset identity, not a continuous
   predictability gradient. Only horizon_min varies independently.

2. **n=12 insufficient.** Four-class classification with minority-class counts of
   1 (NNLS) and 2 (Granite-TTM) is reliably unsolvable at LOO n=11 training points.

3. **Zero cross-dataset generalisation.** LOO-dataset macro-F1 = 0.10 < trivial
   baseline 0.167. Patterns learned on two datasets do not transfer to the third.

**Chapter framing:** the F1 null is the finding. A viable router needs per-series
features (not dataset-level medians) and more than 3 training datasets.

BCF (binary classification, AUC=0.80) succeeds where F1 (4-class, macro-F1=0.25)
fails — binary is structurally simpler. This contrast strengthens BCF as the
thesis primary contribution.

### F1 per-class F1 (canonical, DecTree-depth3)

| Class | F1 |
|---|---|
| NNLS | 0.00 |
| Chronos-2 | 0.73 |
| TimesFM | 0.29 |
| Granite-TTM | 0.00 |

### Scripts and outputs location

All scripts: `phase_f/scripts/` — `_paths.py`, `f1_setup.py`, `f1_router.py`
All data: `phase_f/data/` — 5 CSVs: `f1_feature_matrix.csv`,
`f1_router_predictions.csv`, `f1_loo_cell_results.csv`,
`f1_loo_dataset_results.csv`, `f1_dual_baseline_comparison.csv`

`_paths.py` is the shared path resolver for all F-task scripts. Every new
script in `phase_f/scripts/` should import it: `from _paths import P`.
Paths it provides: `P['thesis']`, `P['phase_f']`, `P['data']`, `P['results']`,
`P['bcf']`, `P['bcf_v2']`, `P['foundation']`, `P['src']`, `P['reports']`.

---

## ERRATA state at D15

| # | Status | Notes |
|---|---|---|
| 001–011 | APPLIED 2026-05-25 | D3 batch |
| 012 | **PENDING** | Substitution text in D13 journal. Jimmy applies D16. |

No ERRATA-013+ rows opened D11–D15. Biblio audit .bib comparison pending (D16+).

---

## Biblio anchor ground truth (D11, web-verified)

Five memory-flagged entries verified. .bib comparison against manuscript
requires local Windows run. Key watch-fors:

| Key | Verified lead author | Critical check |
|---|---|---|
| fremer-pvldb | **Hengyu Ye** | Not Jiadong Chen — that is the arXiv ordering only |
| fremer-arxiv | **Jiadong Chen** | Must cite eprint 2507.12908, not PVLDB DOI |
| aapa | **Guilin Zhang** | Affiliations: GWU + Workday + Youngstown State |
| optscaler | **Ding Zou** | **NO public code** — delete any url/note field |
| hcmiu-qd719 | N/A | Publications optional, not mandatory |

---

## Infrastructure notes (D11–D15 learnings)

- Vast.ai `/venv/main` Python 3.14.3: sklearn was missing — installed 1.8.0
- pandas 3.0 with PyArrow backend: `df["col"].values` on string columns returns
  `ArrowStringArray`, not numpy. Use `.to_numpy(dtype=str)` instead.
- `omega_summary.csv` is at `results/omega_summary.csv`, not thesis root
- `.bib` file is on Overleaf/local Windows — not present on Vast.ai

---

## Forward agenda

| Priority | Task | Machine | Phase |
|---|---|---|---|
| 1 | ERRATA-012 Overleaf application | Local | D16 paperwork |
| 2 | Biblio audit .bib comparison | Local Windows | D16 paperwork |
| 3 | D3 batch retro diff | Overleaf | D16 paperwork |
| 4 | F3 quantile FT of Chronos-2 | Vast.ai GPU | F3 |
| 5 | F5 chapter writing (F1+F2 null chapters) | Any | F5 |

---

## Proposed userMemory edits (for Claude memory update)

1. **ADD:** F1 pre-reg NULL: LOO-cell macro-F1=0.2532 (DecTree-depth3), threshold=0.55.
   LOO-dataset=0.1000. Structural diagnosis: dataset-constant features (Spearman ρ=±1.0),
   n=12 insufficient, zero cross-dataset generalisation. Chapter contribution =
   structural diagnosis not the router itself.

2. **ADD:** `_paths.py` at `phase_f/scripts/_paths.py` is the shared path resolver
   for all F-task scripts. All scripts import `from _paths import P`.
   `omega_summary.csv` at `results/omega_summary.csv` (not thesis root).

3. **ADD:** Vast.ai env note: pandas 3.0 ArrowStringArray — use
   `.to_numpy(dtype=str)` not `.values` on string columns. sklearn must be
   installed separately (`pip install scikit-learn`).
