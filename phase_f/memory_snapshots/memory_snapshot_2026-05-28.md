# Memory snapshot — 2026-05-28 (Week 1 close)

## What changed since last snapshot

Week 1 of F2 rescue completed. Three substantive findings, one structural discovery, one methodological choice locked.

## Key new facts to remember

### F2 chapter scope (DECISION-017)

- **Dual framing**: cell-level pre-reg (locked, n=12) + per-series exploratory (NEW, n=20,531 cells split across 3 datasets and 2 clip policies)
- **Pre-registered WPE confirmatory holds as NULL** at both scopes. partial-R² range: 0.0001 (Alibaba clip) to 0.1844 (Bitbrains clip). Never crosses 0.30.
- **THREE Bitbrains catch22 features cross 0.30 under clip [−1, +1]**, exploratory only, not pre-registered:
  - c22_SB_TransitionMatrix_3ac_sumdiagcov: pr² = 0.500, CI [0.185, 0.697]
  - c22_SP_Summaries_welch_rect_area_5_1: pr² = 0.374, CI [0.030, 0.590]
  - c22_DN_OutlierInclude_p_001_mdrmd: pr² = 0.309, CI [0.030, 0.546]
- Only SB_TransitionMatrix's CI lower bound (0.185) is robust enough to survive Bonferroni k=26
- Threshold-crossing is clip-sensitive: no_clip max on Bitbrains is 0.196 (still below threshold)

### Cell-level structural degeneracy (DISCOVERY)

At n=12 with per-dataset-median predictors (3 unique values per feature × 4 horizons), the partial-R² test is structurally degenerate — any constant-per-dataset candidate returns pr² = 0.0790 by construction. The locked F2 was effectively a test of horizon's incremental contribution after marginalising dataset identity, not a WPE-specific test. Disclosed in F2 chapter §3 methodology footnote, NOT a contribution headline.

### Per-series scope: ACF + horizon explains very little

| Scope | r²_reduced |
|---|---|
| Cell-level n=12 (locked F2) | 0.903 |
| Alibaba per-series no_clip | 0.078 |
| Bitbrains per-series no_clip | 0.018 |
| ByteDance per-series no_clip | 0.029 |

Cell-level ACF saturation does NOT generalise to per-series. The chapter must disclose this.

### Per-series data sources (canonical)

- `phase_f/data/week1/per_series_fulltest_r2.csv` (NEW): 20,531 rows = (Alibaba 4921 × 4 + Bitbrains 142 × 4 + ByteDance 93 × 3). Schema: `series_id, dataset, horizon_min, n_points, r2_ensemble, r2_naive, delta_pp, ensemble_file`
- `phase_f/data/par_per_series_r2.csv` (OLD K=20 file, deprecated for F2 chapter use — outcome was pathologically unstable; still canonical for PAR routing dev)

### Per-dataset ensemble file choice

- Alibaba, ByteDance: `test_ensemble_hetero.npy` (NNLS tree+BiLSTM blend, raw scale)
- Bitbrains: `test_ensemble_homo.npy` (BiLSTM unavailable at all Bitbrains horizons per `hetero_ensemble.json` note `bilstm_unavailable`; homo ≡ hetero on Bitbrains. The hetero file at Bitbrains was clobbered Apr 30 with Alibaba data via a file-swap bug; homo survived intact.)
- Verification gate confirmed all 11 cells match canonical R² in JSON to ≤ 0.001

### Bitbrains 5 idle VMs (bb_609 through bb_613)

Real near-constant CPU at measurement floor:
- bb_609: mean 0.49%, std 0.04%, 17-22 unique values across ~1700 measurements
- bb_610: mean 0.37%, std 0.12%
- bb_611-613: similar profile (low CPU, very small variance)

These produce per-series ensemble R² down to −104,159 (bb_609 h120). Real data, not corruption. Clipping at [−1] removes them from the per-series partial-R² regression; without clip they dominate the regression on Bitbrains.

### bitbrains_per_vm.csv: locked F2 used [−10] clip

- 624 rows = 156 VMs × 4 horizons
- ml_r2 column is ml_r2_raw clipped at exactly −10 (29 rows touched)
- min ml_r2_raw: −2.1 × 10³⁶
- Locked F2 Scope B (n=568 after VM filter, WPE pr² = 0.0558) used this clipped column

### F1 router scope (DECISION-012) unchanged

4 features: ACF@24h + horizon + CV + ACF@1h. SB_TransitionMatrix is exploratory finding, NOT added to F1 yet. Week 5-8 evaluation: optional 5-feature variant comparison.

## Files

### Scripts (in `phase_f/scripts/`)
- `compute_features_extended.py`
- `merge_features_v2.py`
- `substitution_battery_v2.py`
- `build_per_series_fulltest_r2.py`
- `per_series_partial_r2.py` (original)
- `per_series_partial_r2_fulltest.py` (Week 1 final)
- `diag_per_series_r2.py`
- `diag_per_series_r2_deep.py`

### Data (in `phase_f/data/week1/`)
- `features_joined_extended_v2.csv` (5151 rows × 57 cols)
- `per_series_fulltest_r2.csv` (20,531 rows)
- `partial_r2_fulltest.csv` (162 rows)
- `partial_r2_clip_robustness.csv` (15 rows)
- `partial_r2_celllevel_v2.csv` (8 rows)
- `partial_r2_bitbrains_v2.csv` (8 rows)
- Various `substitution_*.csv`, `diag_*.md`, `log_*.txt`

### Phase F state files
- `phase_f/DECISIONS.md`: appended DECISION-017
- `phase_f/journal/2026-05-28_week1-close.md`: full journal entry

## Tasks for Week 2

1. Wild-cluster bootstrap on cell-level n=12 partial-R²
2. Bayesian regression (Gelman-Goodrich 2019)
3. Bandt 2005 AR(1) ordinal-pattern derivation for F2 appendix
4. Bonferroni-corrected re-evaluation of SB_TransitionMatrix on Bitbrains
5. Begin F2 chapter prose drafting (§3 methodology, §5 results)

## Forward connections

- F1 router (DECISION-012, Weeks 5-8): may consider SB_TransitionMatrix as 5th feature side-experiment, do NOT modify D12 prematurely
- F2 chapter (Weeks 3-4): dual-framing per D14, full disclosure of exploratory exceedances, structural-degeneracy footnote
- F3, F4: no immediate impact from Week 1 findings
