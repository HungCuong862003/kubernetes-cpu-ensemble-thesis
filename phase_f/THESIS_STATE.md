# THESIS_STATE.md

**Last updated:** 2026-06-07 (D16 — PAR pivot confirmed)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** F1 paperwork close (ERRATA-012 + biblio audit) + PAR Week 1 start
- **Phase F overall:** Day 16 of ~135
- **Days until defence:** ~115

---

## Contribution stack (revised per DECISION-013)

| Tier | Contribution | Status |
|---|---|---|
| 1 | BCF — ACF@24h × horizon threshold, AUC=0.80, p=0.0097 | ✅ Established |
| 1 | HPA simulation — ML-proactive vs reactive, v4 grid, max_replicas=1000 | ✅ Established |
| 2 | ACF-saturation structural finding — F1+F2 unified diagnosis | ✅ Complete (reframed) |
| 2 | F1 pre-reg NULL — cell-level router, macro-F1=0.2532, structural diagnosis | ✅ Complete |
| 2 | F2 pre-reg NULL — WPE partial-R²=0.079, R²_reduced=0.903 | ✅ Complete |
| 3 | PAR — per-series predictability-aware router, n≈5,150 | 🔄 Starting Week 1 |
| 4 | F3 quantile FT ablation | ⏳ Optional, if GPU budget permits after Week 9 |

---

## F1 and F2 summary — pre-registered nulls (FINAL)

**F1 (cell-level router, pre-registered, CLOSED):**
- Architecture: DecTree-depth3
- LOO-cell macro-F1: 0.2532 (threshold 0.55) → **NULL**
- LOO-dataset macro-F1: 0.1000 (below trivial 0.167)
- Root cause: ACF saturation — 3 of 4 features Spearman ρ=±1.0; n=12 too small
- CSVs: phase_f/data/f1_*.csv (5 files, verified 28/28)
- Verify script: phase_f/scripts/f1_verify.py

**F2 (WPE partial-R², pre-registered, CLOSED):**
- Headline partial-R²: 0.0790, CI [0.000, 0.079] → **NULL**
- Bitbrains per-VM: 0.0558, CI [0.0007, 0.2126]
- R²_reduced (ACF + horizon): 0.903
- Root cause: Same ACF saturation — WPE and ACF are partial substitutes on cloud traces
  (Şen et al. 2024 mechanism)
- CSVs: phase_f/data/f2_partial_r2_results.csv, wpe_*.csv

**Unified explanation (DECISION-013):**
Cloud workloads are ACF-saturated. Every predictability metric collapses to one
dimension. F1's features encoded dataset identity not predictability gradient.
F2's WPE added nothing beyond ACF. Same root cause. Framed per Karl et al. (ICML 2024)
NMNR criteria. External validation: Wang et al. (2025, arXiv:2511.08884) found identical
threshold behaviour (foundation models win only at high spectral predictability).

---

## PAR specification (per DECISION-013)

**Goal:** per-series predictability-aware router motivated by F1+F2 structural diagnosis

**Features (per series):**
- catch22 (22 features) — `pycatch22.catch22_all(x)` — ~0.5s per series
- DFA / Hurst exponent — `antropy.detrended_fluctuation(x)`
- Lempel-Ziv Complexity — `antropy.lziv_complexity(x)`
- Sample Entropy — `antropy.sample_entropy(x)`
- ACF@24h (kept for comparison)

**Training data:**
- ~5,150 series (Alibaba ~4,900 + Bitbrains 156 + ByteDance 93)
- Per series × horizon: n ≈ 20,600 training rows

**Classifiers:**
- Shrinkage-LDA (Ledoit-Wolf, `LinearDiscriminantAnalysis(shrinkage='auto')`)
- XGBoost (secondary)

**Evaluation:**
- Primary: Leave-One-Dataset-Out CV (3 folds, true cross-dataset generalisation)
- Metric: regret = MASE(selected) − MASE(oracle); macro-F1 secondary
- Decision thresholds:
  - LODO macro-F1 > 0.40 → PAR headline contribution
  - LODO macro-F1 0.20–0.40 → PAR partial positive
  - LODO macro-F1 < 0.20 → ACF saturation confirmed at series level (structural finding deepens)
  - All branches defensible

**HPA integration:**
- Compare 4 policies: reactive, BCF binary, always-C2, PAR
- Metric: SLO violations × resource cost

---

## Revised 115-day timeline

| Weeks | Work | Machine |
|---|---|---|
| 1–3 | Per-series catch22 + DFA + LZC computation (~5,150 series) | Vast.ai CPU |
| 4–6 | PAR-v0: shrinkage-LDA + XGBoost, LODO evaluation, regret metric | Vast.ai CPU |
| 7–9 | PCA feature analysis; per-series partial-R² for F2 at series level | Vast.ai CPU |
| 10–13 | HPA integration, 4-policy comparison | Vast.ai CPU |
| 14–16 | Optional F3 quantile FT ablation | Vast.ai GPU |
| 17–21 | Chapter writing (F1+F2 structural + PAR) | Local |
| 22–23 | Mock defences | Local |

---

## Pending paperwork (D16 priority — before PAR Week 1)

| Item | Status | Action |
|---|---|---|
| ERRATA-012 Overleaf | **PENDING** | Apply substitution text from D13 journal. Update ERRATA.md PENDING → APPLIED |
| Biblio audit .bib | **PENDING** | Run `f1_biblio_verify.py` on Windows with .bib file |
| D3 batch retro diff | **PENDING** | `git diff <parent> <d3_sha>` in Overleaf history |
| Verifier re-run | Run after ERRATA-012 | `python3 src/analysis/task4_verify_tables.py` post-Overleaf |

---

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold | Status |
|---|---|---|---|
| F1 | macro-F1 | ≥ 0.55 | **NULL — 0.2532 (D12, CLOSED)** |
| F2 | partial-R²(WPE \| ACF@24h) | ≥ 0.30 | **NULL — 0.079 (D5, CLOSED)** |
| F3 | Spearman ρ | ≥ 0.6 | Demoted to ablation per DECISION-013 |
| PAR | LODO macro-F1 | Post-hoc, not pre-reg | Target > 0.40; all branches defensible |

---

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai C.37423026 | Active | sklearn 1.8.0, Python 3.14.3, antropy + pycatch22 to install |
| Git | D15 commits clean (43091cf, 35f06df) | D16 commit pending |
| Overleaf | ERRATA-012 still PENDING | Apply at D16 |
| phase_f/scripts/ | _paths.py, f1_setup.py, f1_router.py, f1_verify.py | All clean |
| phase_f/data/ | 5 F1 CSVs + F2 CSVs | All verified |

---

## Canonical files (current versions, unchanged from D15)

| Topic | File |
|---|---|
| BCF 3-model | results/bcf/bcf_pooled_3model.json |
| HPA dominance | results/bcf_v2/hpa_v4_dominance_per_dataset.csv |
| Foundation leaderboard §4.7 | results/foundation_comparison/leaderboard_v1.csv |
| F1 results | phase_f/data/f1_*.csv (5 files) |
| F2 results | phase_f/data/f2_partial_r2_results.csv |
| THESIS_STATE | phase_f/THESIS_STATE.md (this file) |
| Memory snapshot | phase_f/memory_snapshots/memory_snapshot_2026-06-06.md |

---

## Active open questions

| Q-ID | Description | Blocking? |
|---|---|---|
| Q-002 | Vast.ai C.37124280 fate | No |
| Q-003 | Public + MIT repo supervisor approval | No |
| Q-NEW | Install antropy + pycatch22 on Vast.ai before PAR Week 1 | Yes for PAR start |

---

## ERRATA state (full detail in ERRATA.md)

12 rows total. 11 APPLIED (D3 batch 2026-05-25). 1 PENDING (ERRATA-012).
DECISION-013 does not open new ERRATA rows — no manuscript numbers changed.

---

## Update protocol

Wholesale rewrite at every day-close. Historical record in phase_f/handoffs/ and DECISIONS.md.
# THESIS_STATE.md — update from 2026-05-25 F3 Day 1

This is an UPDATE to be merged into `phase_f/THESIS_STATE.md`. Replace the F3 section with this content, or append at the bottom if F3 had no section yet.

---

## Current phase: F3 (cost-asymmetric Chronos-2 fine-tune)

**As of 2026-05-25 end of session.**

### F3 task progress

| # | Task | Status | Date |
|---|---|---|---|
| F3.1 | Zero-shot baseline | ✅ DONE | 2026-05-25 |
| F3.2 | Setup probe + secondary metric lock | ⏸️ NEXT | — |
| F3.3 | LoRA fine-tune script | Pending | — |
| F3.4 | Run training | Pending | — |
| F3.5 | Evaluation + DECISION-015 lock | Pending | — |
| F3.6 | Robustness ablations | Pending | — |

### F3 pre-registered primary criterion (LOCKED)

Mean pinball loss at h=60min, τ=0.9, averaged across {Alibaba, Bitbrains, ByteDance}.

- Baseline value: **1.921658**
- SUCCESS threshold (≥5% improvement): ≤ 1.825576
- PARTIAL threshold (≥2% improvement): ≤ 1.883225
- FAILURE: > 1.883225 → triggers DECISION-015

### F3 secondary metric (RECOMMENDED, not yet locked)

DECISION-016 OPEN: Option C recommended = geometric mean of per-dataset % improvements at h=60, τ=0.9, same thresholds. To lock at F3.2 by appending to `phase_f/f3_design.md`.

### F3 baseline per-cell pinball at τ=0.9

| Dataset | h=10 | h=30 | h=60 | h=120 |
|---|---|---|---|---|
| Alibaba | 0.237120 | 0.347370 | **0.446603** | 0.473945 |
| Bitbrains | 1.535303 | 2.121677 | **4.858964** | 8.015288 |
| ByteDance | 0.383004 | 0.396275 | **0.459408** | 0.479588 |

Full per-(dataset, horizon, tau) results at `phase_f/data/f3_zero_shot_baseline.json`.

### Critical finding flagged

Bitbrains contributes 84% of the unweighted-mean primary metric (1.620 of 1.922) due to ~10× larger absolute CPU values across datasets. Scale artifact, not quality difference. Drives the Option C secondary-metric recommendation in DECISION-016.

---

## Phase F overall state

| Phase | Status | Notes |
|---|---|---|
| F0 | ✅ Closed 2026-06-01 (D10) | 12 DECISIONS, 12 ERRATA, verifier 214/0 |
| F1 | ✅ Closed 2026-06-06 (D15) | PAR router, STRUCTURAL SATURATION confirmed |
| F2 | ✅ Closed 2026-05-27 (D5) | WPE partial-R² null per pre-reg |
| F3 | 🔄 In progress (Day 1 of ~7) | Baseline locked, fine-tune pending |
| F4 | Pending | OptScaler + AHPA integration; ~5–7 days |
| F5 | Pending | 6-chapter manuscript rewrite under Pivot C+D framing |

---

## Active environment state

### Vast.ai

- Instance: `C.37705458`
- GPU: NVIDIA RTX A4000, 16.6 / 16.8 GB free
- Python 3.12.13, venv `/venv/main`
- torch 2.9.1+cu128, NO torchvision (uninstalled to fix C++ ABI mismatch)
- chronos-forecasting installed, Chronos-2 weights cached (~480 MB)
- AutoGluon 1.5.0 installed but unused by production code
- Modified packages for AutoGluon compatibility: sklearn 1.7.2, pandas 2.3.3, numpy 2.1.3, xgboost 3.1.3, pyarrow 20.0.0, huggingface_hub 0.36.2
- **DO NOT re-run PAR pipeline on this env** — version drift may produce different numbers vs saved versions

### Data layout (canonical)

- Raw time series: `data/processed/<dataset>/{train,val,test}.parquet`
- Per-horizon aligned spine: `results/<dataset>/h<HHH>/predictions/test_spine.parquet`
- Schema: `[container_id, time_stamp, cpu_residual, cpu_target, naive_cpu]`

---

## Pivot C+D framing (committed 2026-05-25 prior session)

6-chapter thesis under new framing "When ML Helps Kubernetes Autoscaling: Boundary Conditions and Structural Saturation":

1. BCF (Boundary Condition Framework) — primary contribution
2. PAR partial-positive + label-shift failure (DECISION-014 LOCKED)
3. Structural Saturation convergence (F2 + PAR)
4. F3 Chronos-2 fine-tune (THIS PHASE — outcome determines weight)
5. F4 OptScaler/AHPA + HPA simulation
6. Discussion + conclusion

Chapter writeup NOT YET STARTED — depends on F3 outcome.

---

## Defence timeline

| Date | Milestone |
|---|---|
| 2026-05-25 | F3 Day 1 ✅ baseline locked |
| 2026-05-26 → 2026-05-31 | F3 Day 2–6: setup, LoRA, eval, ablations |
| 2026-06-01 → 2026-06-15 | F3 chapter writeup + errata sheet |
| 2026-06-15 → 2026-08-30 | F4 + remaining 5 chapters |
| 2026-09 → 2026-12 | Manuscript revision, mock defences |
| ~March 2027 | Revised defence target (one-semester delay accepted) |