# THESIS_STATE.md

**Last updated:** 2026-05-28 (D19 F2 hardening + D18 F4 close + D17 Omega all complete)
**Project:** kubernetes-cpu-ensemble-thesis
**Defence:** ~October 2026 (one-semester delay accepted)

---

## Where we are

- **Phase:** ALL Phase F computational streams CLOSED. F5 chapter writing is next.
- **Phase F overall:** All F0–F4 + D19 F2 methodology hardening + Omega post-hoc done.
- **Days until defence:** ~115

---

## Computational stream status (FINAL)

| Stream | Pre-reg threshold | Result | Status |
|---|---|---|---|
| F0 Audit + errata + biblio | — | 12 ERRATA applied + 4 added (E013–E016) | ✅ CLOSED |
| F1 cell-level router | macro-F1 ≥ 0.55 | 0.2532 — structural NULL | ✅ CLOSED (D12/D15) |
| F2 WPE partial-R² (cell) | partial-R² ≥ 0.30 | 0.0790 — structural NULL | ✅ CLOSED (D5) |
| F2 per-series extension | post-hoc | 162 cells, NULL × 6 (dataset × clip); 3 exceedances exploratory | ✅ CLOSED (D17) |
| **F2 Week 2 hardening (D19)** | **methodology** | **Null defended on 5 fronts + analytical proof; 3 exceedances FORMALLY DISMISSED** | **✅ CLOSED (D19)** |
| PAR per-series router | post-hoc | XGB macro-F1 0.2166; ByteDance catastrophe; structural saturation confirmed at series level | ✅ CLOSED (D14/D18b) |
| F3 Chronos-2 LoRA | primary ≥ 5% | +8.17% primary SUCCESS / 3.44% secondary PARTIAL | ✅ CLOSED (D15) |
| F4 MPC + HPA close | strict-Pareto | Alibaba tradeoff −1.9pp/+0.22pp → v6 retained per pre-reg | ✅ CLOSED (D18) |
| Omega post-hoc validation | — | Cell ρ(Ω)=0.090 NULL; per-VM BB ρ=−0.507; partial RETRACTED (jackknife degenerate) | ✅ CLOSED (D17) |

---

## Structural saturation finding (canonical, FINAL)

**Root cause:** On AR(1)-like short-memory Gaussian processes (the regime most cloud
CPU traces occupy), all linear and nonlinear predictability metrics collapse onto
one dimension aligned with ACF@24h. This is analytical, not empirical accident.

**Five empirical fronts + one analytical proof (D19):**

| Front | Diagnostic | Result |
|---|---|---|
| 1. Structural identification | 5 constant-per-dataset candidates → cell-level partial-R² | All = 0.078960 (SD=0.0); Ibragimov-Müller t(2)=−0.34, p=0.766 |
| 2. BH-FDR + Westfall-Young | 26 catch22 candidates, multiple-comparison correction | 0/26 survives; SB raw_p=0.0775, WY_p=0.494; WPE NULL=0.1844 |
| 3. CV-stratification | Active-only stratum (137 VMs, 5 idle excluded) | SB 0.500→0.0147 / SP→0.0005 / DN→0.0236 / WPE→0.0177 |
| 4. Cluster-leverage | LOO-VM shifts in partial-R² | bb_611 −0.078, bb_613 −0.076, bb_612 −0.060 — same 4–5 idle VMs |
| 5. Hierarchical Bayes | PSIS-LOO + regularised horseshoe | Pr(pr²≥0.30)=0; WPE pr²=0.0000 [−0.016,0.016]; SB ELPD diff = −1236.5 (SE 300) |
| Analytical | Bandt & Shiha (2007) AR(1) ordinal-pattern derivation | PE deterministic monotone of φ; PE=1.0000 at φ=0 → 0.871 at \|φ\|=0.95; m=3 closed-form vs MC max error 0.0021 |

**Per-dataset WPE slopes (sign-discordant, confirms non-identification at cell level):**
Alibaba −0.26 / Bitbrains −4.19 / ByteDance +2.48.

**Convergent evidence from other strands:**
- F1 cell-level: 3/4 features Spearman ρ = ±1.0 (encode dataset identity, not predictability)
- F2 cell-level: R²_reduced(ACF+horizon) = 0.903; partial-R²(WPE) = 0.079 (NULL)
- PAR PC1: ρ(PC1, ACF@24h) = 0.836; between-dataset variance on PC1 = 5.9%
- Omega post-hoc (D17): cell ρ(Ω, Δpp) = +0.090 p=0.54 NULL; ACF@24h ρ = +0.449 p=0.0014
- Within-Bitbrains per-VM ρ(Ω, ΔR²) = −0.507 p<0.001 — direction INVERTS vs Wang 2025 (mean-reversion artefact)

**External corroboration:**
- Wang, Quan, Yang & Srivastava (arXiv:2511.08884, Nov 2025) — Ω stratifies TSFM fitness on general-purpose benchmarks; saturation on cloud workloads is the domain-specific contribution.
- Şen et al. (2024, *Methods in Ecology and Evolution*) — red-noise WPE depression.
- Montanari, Taqqu & Teverovsky (1999, *Math. & Comp. Modelling* 29(10–12):217–228) — Hurst inflation by periodicity.

**Why BCF succeeds where F1/PAR fail:** BCF reduces the saturated manifold to a binary
threshold on ACF@24h. Three datasets give three Ω/ACF/WPE values — too few for
multi-class routing but sufficient for a thresholded predicate. AUC=0.80, n=36, p=0.0097.

---

## Three exceedances FORMALLY DISMISSED (D19.2)

Chapter §5 states: "Of 26 exploratory predictability features screened, three crossed
the 0.30 threshold on Bitbrains under clip[−1,+1]. None survives Benjamini-Hochberg
FDR or Westfall-Young correction; all three collapse below 0.025 once the five
near-zero-mean-utilisation VMs are excluded; cluster-leverage diagnostics confirm those
same VMs drive the entire effect; and a hierarchical Bayesian model finds zero
posterior probability of any candidate crossing the threshold. The exceedances are
artefacts, not signal."

Dismissed features: SB_TransitionMatrix_3ac_sumdiagcov (0.500), SP_Summaries_welch_rect_area_5_1 (0.374), DN_OutlierInclude_p_001_mdrmd (0.309).

**Side-experiment CANCELLED (D19.3):** F1 stays at 4 features. The Week 5–8 plan to
test SB_TransitionMatrix as a 5th F1 feature is cancelled — hierarchical Bayes shows
its inclusion *reduces* predictive density (ELPD diff −1236.5).

---

## Methodology corrections locked (D19.4)

1. **Wild-cluster bootstrap → identification-limits documentation.** MacKinnon-Webb (2018) require G ≥ 15–20; structural non-identification cannot be cured by resampling. Cite Mundlak (1978), Ibragimov-Müller (2016), Snijders-Bosker (2012).
2. **Flat-prior Bayes-R² → hierarchical model + regularised-horseshoe + PSIS-LOO.** Cite Piironen-Vehtari (2017), Vehtari-Gelman-Gabry (2017), Gelman et al. (2019).
3. **Bonferroni k=26 → BH-FDR + Westfall-Young max-T.** Bonferroni dominated by BH under catch22 positive dependence. Cite Benjamini-Hochberg (1995), Benjamini-Yekutieli (2001), Westfall-Young (1993).
4. **Citation correction:** Bandt & Shiha (2007), *J. Time Series Analysis* 28:646–665 — NOT Bandt (2005). The m=4 / WPE(φ) closed form is not elementary; numerical verification only.

---

## Sampling-provenance disclosure (D19.5)

`hierarchical_bayes_loo.csv` consolidates two runs at different sampler settings:
- WPE at target_accept = 0.95 (clean, 0 divergences)
- SB_TransitionMatrix at target_accept = 0.99 (6 divergences, down from 870; minor R-hat/ESS warnings on horseshoe hyperparameters tau/lam/c2, expected, do not affect b_cand/sigma/R² conclusions)

Chapter discloses in methodology footnote. Conclusion corroborated on four other fronts.

---

## Key numbers (canonical, verified)

### BCF
- 3-model pool (NNLS + Chronos-2 + TimesFM): AUC=0.80, CI [0.7097,0.8871], p=0.0097, n=36
- Predicate: ACF@24h > 0.2 AND h ≥ 30 min
- CI method: percentile (BCa degenerate for binary classifier × binary outcome)

### NNLS weights (production canonical, source: run.log)
| Horizon | XGB | LGB | ET | BiLSTM |
|---|---|---|---|---|
| 10 min | 2.4% | 0.0% | 97.6% | N/A |
| 30 min | 0.7% | 4.5% | 72.8% | 22.0% |
| 60 min | 0.0% | 0.0% | 78.4% | 21.6% |
| 120 min | 0.0% | 0.0% | 75.9% | 24.1% |

### Foundation model leaderboard (canonical: leaderboard_v1.csv, K=20/K=50 scope)
| Model | Wins (of 12 cells) |
|---|---|
| Chronos-2 | 6 |
| TimesFM | 3 |
| Granite-TTM | 2 |
| NNLS | 1 |

### PAR (D14 / D18b PARTIAL POSITIVE, CLOSED)
- XGB equal-weight macro-F1 = 0.2166 (threshold 0.20, PARTIAL)
- Sample-weighted regret: AlwaysC2 0.107 beats LDA 0.132 and XGB 0.138
- ByteDance catastrophe: 60/189 series routed to granite_ttm, mean R²-regret 2.92/row
- Root cause: cross-dataset spillover from Bitbrains training; combined covariate + label shift
- HPA 4-policy (R²): PAR-XGB 0.1205 > AlwaysC2 0.0906 > Reactive 0.0000 > BCF −0.0213

### F3 LoRA (D15 SUCCESS, CLOSED)
- Primary (mean pinball h=60 τ=0.9): Baseline 1.921658 → Fine-tuned 1.764582 → +8.17% ✅
- Secondary (geometric mean per-dataset): 3.44% PARTIAL (threshold 5%)
- Best model: epoch 1, early stopped at epoch 6

### F4 MPC + HPA (D18 CLOSED, v6 canonical)
- Alibaba: violation −1.90pp [−2.19,−1.60] AND cost +0.22pp [+0.09,+0.35] → **Pareto tradeoff**
- Bitbrains: violation −0.23pp sig, cost −0.13 non-sig → partial improvement
- ByteDance: null
- v6 retained per strict-Pareto pre-registration; v7 routing infra kept as repro artefact

### Omega post-hoc (D17, CLOSED)
| Result | Value |
|---|---|
| Cell-level ρ(Ω, delta_pp) | +0.090, p=0.54 — **NULL** |
| Cell-level ρ(ACF@24h, delta_pp) | +0.449, p=0.0014 — significant |
| Partial ρ(Ω \| ACF@24h) | +0.370, p=0.0097 → **RETRACTED** (jackknife degenerate, see D17.4–17.5 addendum) |
| Per-VM BB ρ(Ω, ΔR²) overall | −0.507, p<0.001 — **ROBUST** |
| Per-VM BB ρ(Ω, ΔR²) by horizon | 10min −0.414 / 30min −0.518 / 60min −0.537 / 120min −0.525 |

### F2 D19 five-front canonical numbers
| Test | Outcome |
|---|---|
| Structural identification (n=5 candidates) | All pr² = 0.078960, SD=0.0 |
| BH-FDR (q=0.05, k=26) | 0/26 survives (all BH_q=1.0000) |
| Westfall-Young (k=26) | 0/26 survives (all WY_p > 0.49) |
| CV-stratification active-only | SB 0.500→0.0147, SP→0.0005, DN→0.0236, WPE→0.0177 |
| Cluster-leverage (top 3) | bb_611 −0.078, bb_613 −0.076, bb_612 −0.060 |
| Hierarchical Bayes PSIS-LOO | Pr(pr²≥0.30)=0; ELPD diff for SB inclusion = −1236.5 (SE 300) |
| Bandt-Shiha analytical (m=3) | MC vs closed form max error = 0.0021 |

---

## Canonical source files (current versions)

| Topic | File |
|---|---|
| BCF 3-model | results/bcf/bcf_pooled_3model.json |
| BCF pairs | results/bcf/bcf_pairs.csv |
| HPA dominance | results/bcf_v2/hpa_v4_dominance_per_dataset.csv |
| Foundation leaderboard §4.7 | results/foundation_comparison/leaderboard_v1.csv |
| F3 results | phase_f/data/f3_finetune_results.json |
| F4 trajectories | results/bcf_v2/f4_trajectories.parquet |
| F4 bootstrap | phase_f/data/f4_bootstrap_results.csv |
| PAR predictions | phase_f/data/par_router_predictions.csv |
| Omega validation report | phase_f/data/par_omega_validation_report.md |
| Omega cell results | phase_f/data/par_omega_cell_results.csv |
| Omega VM results | phase_f/data/par_omega_vm_results.csv |
| Omega scatter | phase_f/data/par_omega_scatter.png |
| **D19 structural ID** | phase_f/data/week2/identification_limits.md |
| **D19 FDR + WY screen** | phase_f/data/week2/fdr_corrected_screen.csv + .md |
| **D19 CV-stratification** | phase_f/data/week2/cv_stratification.csv + .md |
| **D19 cluster-leverage** | phase_f/data/week2/loo_cluster_leverage.csv + .md |
| **D19 hierarchical Bayes** | phase_f/data/week2/hierarchical_bayes_loo.csv + .md |
| **D19 Bandt-Shiha MC** | phase_f/data/week2/bandt_shiha_ar1_numerical.md |
| NNLS weights | run.log (workspace root) |
| THESIS_STATE | phase_f/THESIS_STATE.md (this file) |

---

## Scripts (D19 Week 2 batch, all 2026-05-28)

| Script | Output | Time |
|---|---|---|
| identification_limits.py | identification_limits.md | 14:03 |
| fdr_corrected_screen.py | fdr_corrected_screen.csv/.md | 13:55 |
| cv_stratification.py | cv_stratification.csv/.md | 14:17 |
| loo_cluster_leverage.py | loo_cluster_leverage*.csv/.md | 14:17 |
| hierarchical_bayes.py | hierarchical_bayes_loo.csv/.md (initial) | 16:45 |
| consolidate_bayes_loo.py | hierarchical_bayes_loo.csv/.md (final consolidated) | 17:01 |
| bandt_shiha_ar1_numerical.py | bandt_shiha_ar1_numerical.md | 14:17 |
| par_omega_validation.py | par_omega_{cell,vm}_results.csv, scatter.png, validation_report.md | 14:01 |

---

## Decisions register summary (D01–D19)

| ID | Date | Status | Topic |
|---|---|---|---|
| D01 | 2026-05-22 | LOCKED | 6-chapter manuscript structure |
| D02 | ~2026-05-21 | LOCKED | BCF canonical = 3-model pool |
| D03–D11 | 2026-05-23 to 2026-06-02 | LOCKED | Pre-F3 lockdown items |
| D12 | 2026-05-25 | LOCKED | F1 router architecture (4 features, cell-level) → NULL |
| D13 | 2026-05-25 | LOCKED | F3 demoted to optional; PAR becomes primary Phase F |
| D14 | 2026-05-25 | LOCKED | PAR PARTIAL POSITIVE (XGB 0.2166) |
| D15 | 2026-05-26 | LOCKED | F3 SUCCESS (+8.17% primary) / PARTIAL secondary (3.44%) |
| D16 | 2026-05-25 | LOCKED | F3 secondary metric definition (geometric mean) |
| **D17** | **2026-05-28** | **LOCKED** | **F2 per-series extension + Omega post-hoc validation (D17.1–17.3 base; D17.4–17.5 AMENDED by D19)** |
| **D18** | **2026-05-28** | **LOCKED** | **F4 close: MPC + HPA simulation Pareto verdict per dataset (v6 retained)** |
| **D19** | **2026-05-28** | **LOCKED** | **F2 Week 2 methodology hardening: five-front null defence + exceedances FORMALLY DISMISSED + F1 stays at 4 features** |

---

## Infrastructure state

| Resource | State | Notes |
|---|---|---|
| Vast.ai C.38014225 | Active | D19 scripts + Omega ran 2026-05-28 13:55–17:01 |
| Git | feature/live-demo branch | D17/D18/D19 + Omega commits pending |
| Overleaf | ERRATA-012 to E016 PENDING | Apply before F5 writing |
| phase_f/data/week2/ | D19 outputs | All synced to Drive |
| phase_f/scripts/ | D19 batch + par_omega_validation.py | Synced |

---

## ERRATA state (full detail in ERRATA.md)

16 rows (E001–E016). E001–E011 APPLIED. E012–E016 PENDING Overleaf.
**D19 produces NO new errata** — no manuscript numbers changed (D19 confirms existing null, adds methodology depth, dismisses exceedances).

---

## Pending paperwork (before F5 writing starts)

| Item | Status | Action |
|---|---|---|
| ERRATA-012 to E016 Overleaf | PENDING | Apply substitutions, update ERRATA.md status |
| Biblio audit .bib | PENDING | Run f1_biblio_verify.py with .bib; +10 new D19 citations |
| §3.5 + Appendix D imputation rates | PENDING | Apply corrected values (Alibaba 7.9–15.6%, Bitbrains 24.4–25.4%, ByteDance 17.3–17.6%) |
| Memory snapshot 2026-05-28 week2 | DONE | phase_f/memory_snapshots/memory_snapshot_2026-05-28_week2.md |

**New D19 citations to add:** Mundlak (1978), Ibragimov-Müller (2016), Snijders-Bosker (2012), Piironen-Vehtari (2017), Vehtari-Gelman-Gabry (2017), Gelman et al. (2019), Benjamini-Hochberg (1995), Benjamini-Yekutieli (2001), Westfall-Young (1993), Bandt-Shiha (2007).

---

## F5 chapter writing plan (next ~5–7 weeks)

Six chapters. Priority: structural saturation chapter (Ch3) is the strongest — D19 gave it analytical backbone.

| Chapter | Topic | Status |
|---|---|---|
| Ch1 | Intro + BCF motivation | ⏳ Not started |
| Ch2 | Related work + PAR + label-shift | ⏳ Not started |
| Ch3 | **Structural saturation (F1+F2+per-series+D19+Omega) — strongest contribution** | ⏳ Not started |
| Ch4 | F3 Chronos-2 LoRA (v6 canonical, D18) | ⏳ Not started |
| Ch5 | F4 MPC+HPA sim | ⏳ Not started |
| Ch6 | Discussion + conclusion | ⏳ Not started |

---

## Pre-registration thresholds (Dr. Ho written acceptance 2026-05-22)

| Phase | Metric | Threshold | Status |
|---|---|---|---|
| F1 | macro-F1 | ≥ 0.55 | **NULL — 0.2532 (CLOSED)** |
| F2 | partial-R²(WPE\|ACF@24h) | ≥ 0.30 | **NULL — 0.079; defended five fronts (D19 CLOSED)** |
| F3 primary | mean pinball improvement | ≥ 5% | **SUCCESS — +8.17% (D15 CLOSED)** |
| F3 secondary | geometric mean improvement | ≥ 5% | **PARTIAL — 3.44% (D15 CLOSED)** |
| F4 | Pareto strict dominance | v6 retained if tradeoff | **TRADEOFF — v6 canonical (D18 CLOSED)** |
| PAR | LODO macro-F1 | post-hoc > 0.40 target | **PARTIAL POSITIVE — 0.2166 (D14 CLOSED)** |

---

## Grade trajectory estimate

| Scenario | Grade (max 10) |
|---|---|
| Current state, F5 not started | 8.0 |
| + F5 chapters at good quality | 8.5–9.0 |
| + Structural saturation chapter exploits D19 analytical backbone | 9.0 |
| + Mock defences with revisions | +0.2–0.4 |

**Primary lever:** the D19 five-front + Bandt-Shiha analytical proof elevates the structural saturation contribution from "null with disclosure" to "null defended on 6 independent grounds + analytical mechanism." This is the strongest section of the thesis and should be Ch3's framing anchor.

---

## Update protocol

Wholesale rewrite at every day-close. Historical record in phase_f/handoffs/ and DECISIONS.md.