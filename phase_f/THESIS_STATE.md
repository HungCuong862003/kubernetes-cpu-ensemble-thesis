# THESIS_STATE.md

**Project:** Hybrid Ensemble Learning for Proactive Resource Prediction in Kubernetes
**Student:** Phan Nguyen Hung Cuong (Jimmy), ITDSIU21078
**Institution:** International University HCMIU, Vietnam National University HCMC
**Supervisor:** Dr. Ho Long Van
**Defence target:** March 2027 (~9 months from current date)
**Last updated:** 2026-05-26 (F3 closure + verification + fix)
**Phase:** F (computational work CLOSED; F5 chapter writing remaining)

---

## 1. Phase status

| Phase | Description | Status | Outcome |
|---|---|---|---|
| F0 | Audit + paperwork | ✅ CLOSED (D10) | 13 DECISIONS, 13 ERRATA initially logged |
| F1 | Cell-level router | ✅ CLOSED (D15) | NULL — intrinsic ACF saturation |
| F2 | WPE partial-R² | ✅ CLOSED (D5) | NULL — intrinsic substitute for ACF@24h |
| PAR | Per-series router | ✅ CLOSED (D58) | PARTIAL POSITIVE — XGB macro-F1=0.2166 |
| PAR PCA | Feature structure | ✅ CLOSED (D59) | Multi-dimensional — 9 PCs for 80% |
| PAR HPA | 4-policy comparison | ✅ CLOSED (D80) | PAR ranks #1 (mean R²=0.1205) |
| F3 | LoRA fine-tune | ✅ **CLOSED 2026-05-26 POST-VERIFICATION + FIX** | SUCCESS — **all 3 datasets SUCCESS per pre-reg threshold** |
| F4 | OptScaler+AHPA integration | ⚠️ Reduced scope | par_hpa_comparison.py R²-proxy; full MPC pending |
| F5 | Chapter writing | ⏳ NOT STARTED | 6-chapter Pivot C+D rewrite remaining |
| Mock defences | ×2 | ⏳ NOT STARTED | ~10 days after F5 |

---

## 2. Locked numbers (canonical, do not regenerate)

### Ensemble vs Naive (Alibaba, source: `comparison_table.csv`)

| Horizon | Naive R² | Hetero Ens R² | Δpp | Residual var reduction |
|---|---|---|---|---|
| 10 min | 0.9188 | 0.9213 | +0.25 | 3.1% |
| 30 min | 0.8361 | 0.8404 | +0.43 | 2.6% |
| 60 min | 0.7878 | 0.8011 | +1.33 | 6.3% |
| 120 min | 0.7178 | 0.7642 | +4.64 | 16.4% |

### NNLS weights (production canonical, source: `run.log`)

| Horizon | XGB | LGB | ET | BiLSTM |
|---|---|---|---|---|
| 10 | 2.4% | 0.0% | 97.6% | N/A |
| 30 | 0.7% | 4.5% | 72.8% | 22.0% |
| 60 | 0.0% | 0.0% | 78.4% | 21.6% |
| 120 | 0.0% | 0.0% | 75.9% | 24.1% |

### BCF (primary contribution)

- 3-model pool (NNLS + Chronos-2 + TimesFM): AUC = 0.80, percentile 95% CI [0.7097, 0.8871], n=36, p=0.0097
- Predicate: ACF@24h > 0.2 AND h ≥ 30 min
- Per-model AUC: NNLS 0.833, Chronos-2 0.773, TimesFM 0.800, Granite 0.500
- CI method: **percentile, not BCa** (DECISION-007)
- External corroboration: Wang, Quan, Yang & Srivastava, arXiv:2511.08884 (Nov 2025)

### F3 LoRA fine-tune (POST-FIX, canonical as of 2026-05-26)

**Primary metric (pooled mean pinball h=60, τ=0.9):**
- Baseline: 1.921658
- Fine-tuned post-fix: 1.70844
- Improvement: **+11.10% SUCCESS** (was +8.17% pre-fix)

**Per-dataset (post-fix, pooled main+holdout weighted by series count):**

| Dataset | Baseline | Post-fix pooled FT | Improvement | Verdict (DECISION-005) |
|---|---|---|---|---|
| Alibaba | 0.4466 | 0.42342 | **+5.19%** | SUCCESS |
| Bitbrains | 4.8590 | 4.26580 | **+12.49%** | SUCCESS |
| ByteDance | 0.4594 | 0.43611 | **+5.07%** | SUCCESS |

**All three datasets SUCCESS per pre-registered threshold (≥5%).**

**Truly-held-out Alibaba cohort (val saw 14% of main, 0% of holdout):**
- Pre-fix Alibaba holdout: −4.72%
- Post-fix Alibaba holdout: −2.90%
- Generalisation improvement: **+1.82pp** (genuine learning of transferable patterns)

**Training details:**
- LoRA rank=8, alpha=16, target modules q/k/v/o/wi/wo/output_layer/residual_layer
- Best checkpoint at epoch 3 (post-fix), early stopped at epoch 8
- Native pinball loss over 21 quantiles per chronos-forecasting 2.2.2 `_compute_loss` (verified 2026-05-26)
- 3-dataset val signal: Alibaba 500 windows + Bitbrains 142 + ByteDance 93

### Foundation model leaderboard (`leaderboard_v1.csv`, K=20/K=50 subsample)

| Model | Wins (of 12 cells) |
|---|---|
| Chronos-2 | 6 |
| TimesFM | 3 |
| Granite-TTM | 2 |
| NNLS | 1 |

NEW pool full-test scope (`cross_dataset_headline_v2.csv`) different tally — do not conflate.

### PAR (Phase F per-series contribution)

- XGB macro-F1 = **0.2166** PARTIAL POSITIVE (threshold 0.20)
- ByteDance catastrophe: 60 series routed to granite_ttm, mean regret 2.92 → publishable finding
- HPA 4-policy: PAR-XGB 0.1205 > AlwaysC2 0.0906 > Reactive 0 > BCF −0.0213
- PCA: PC1 26.4%, 9 PCs for 80%, PC1 ρ(ACF@24h) = 0.84

### HPA simulation v4 (max_replicas=1000)

ML strict dominance per horizon (h10/30/60/120):

| Dataset | h10 | h30 | h60 | h120 |
|---|---|---|---|---|
| Alibaba | 40 (100%) | 35 (87.5%) | 15 (37.5%) | 1 (2.5%) |
| Bitbrains | 40 (100%) | 0 | 0 | 0 |
| ByteDance | N/A | 33 (82.5%) | 32 (80%) | 29 (72.5%) |

11-cell totals: 225 ML strict / 675 reactive dominance.

---

## 3. Honest disclosure register

**Intrinsic nulls (frame as findings, do not modify):**

1. F1 cell-level NULL — n=12 + ACF saturation
2. F2 WPE NULL — partial substitute for ACF@24h
3. Per-VM Bitbrains sign reversal (pooled vs per-VM median)

**F3 disclosures (post-fix; see ERRATA.md for full text):**

4. ERRATA-014 — F3 trained with native SYMMETRIC pinball over 21 quantiles per chronos2/model.py `_compute_loss`; pre-registration specified ASYMMETRIC pinball. Deviation is "symmetric within right loss family", not "wrong loss family".
5. ERRATA-015 — F3 val-cohort bug: Alibaba was silently excluded from val signal due to val region < N_CONTEXT + h_obs. Fixed via adaptive val_frac; re-trained; all 3 datasets now SUCCESS.
6. ERRATA-016 — F3 holdout cohort overlap: for Bitbrains (n=142) and ByteDance (n=93), val cohort = all series in dataset; the alphabetical "holdout" is NOT truly held out. Only Alibaba provides a truly held-out cohort.
7. Bitbrains scale dominance: contributes 88% of pooled absolute improvement (down from 96% pre-fix; still dominant)
8. Asymmetric pinball weighting applied only at evaluation (post-hoc) due to Chronos-2 pipeline using symmetric pinball natively

**Other disclosures:**

9. PAR ByteDance catastrophe (R²-regret 0.97, granite_ttm spillover) — publishable
10. PAR Alibaba-dominant (96% of training data)
11. Imputation rates: Alibaba 7.9–15.6%, Bitbrains 24.4–25.4%, ByteDance 17.3–17.6%
12. Residual ACF(1) 0.387 → 0.838 — unexploited structure
13. MAE/R² divergence at short horizons
14. ERRATA-013 — Bitbrains test_ensemble_hetero.npy storage corruption (project storage only, no manuscript impact)

---

## 4. Outstanding work (priority order)

### Critical path
1. **F5 chapter writing** — 6 chapters under Pivot C+D framing, ~15–20 days
2. **Bibliography audit + errata sheet** — ~6–8 days (DO FIRST)
3. **Abstract + §1.4 reframe** — ~4–6 days

### High-leverage technical (recommended)
4. **Tier 2 OptScaler MPC** — Dr. Ho's priority — ~7–10 days
5. **SPCI/HopCPT residual layer** — exploits ACF(1)=0.838 — ~8–10 days
6. **PAR with abstention** — fixes ByteDance catastrophe — ~6–8 days
7. **Held-out BCF replication** — Azure Public Dataset V2 — ~6–8 days

---

## 5. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Fabricated citations found at defence | Low (if audit done) | Catastrophic | Bibliography audit FIRST |
| Examiner attacks "F3 used wrong loss" | **Now LOW** | High | Verified pinball loss in chronos 2.2.2 source (ERRATA-014) |
| Examiner attacks "val excluded Alibaba" | **Now LOW** | High | Fixed and re-run; disclosed in ERRATA-015 |
| Examiner attacks per-dataset variance | Low | Moderate | All 3 datasets now SUCCESS per pre-reg threshold |
| MPC reimplementation runs late | Moderate | Moderate | Fallback: drop to Fremer-only baseline |
| F5 writing slips | Moderate | High | Start now; cut Tier 2 if needed |

---

## 6. Canonical source files

| Result | Source file (Drive/repo) |
|---|---|
| Ensemble R² | `comparison_table.csv` |
| NNLS weights | `run.log` |
| BCF 3-model | `results/bcf/bcf_pooled_3model.json` |
| Foundation leaderboard (manuscript) | `leaderboard_v1.csv` |
| Foundation leaderboard (full-test) | `cross_dataset_headline_v2.csv` |
| PAR predictions | `phase_f/data/par_router_predictions.csv` |
| PAR HPA | `phase_f/data/par_hpa_comparison.csv` |
| **F3 baseline (zero-shot)** | `phase_f/data/f3_zero_shot_baseline.json` |
| **F3 post-fix results** | `phase_f/data/f3_eval_lora_rank8.json` + `.csv` |
| **F3 pre-fix backup** | `phase_f/data/f3_eval_lora_rank8_prefix.json` + `.csv` |
| **F3 per-dataset diagnosis** | `phase_f/data/f3_postfix_per_dataset.txt` |
| **F3 verification report** | `phase_f/data/f3_verification.json` + `.md` |
| **F3 loss inspection** | `phase_f/data/f3_loss_inspection.txt` |
| **F3 LoRA adapter (post-fix)** | `phase_f/models/f3_lora_rank8/` |
| **F3 LoRA adapter (pre-fix backup)** | `phase_f/models/f3_lora_rank8_prefix/` |
| HPA v4 | `bcf_v2/hpa_simulation_*_v4.csv` |
| BCF v2 | `results/bcf_v2/d4_*_TEST.{csv,md}` |

---

## 7. Phase F decision summary

See `DECISIONS.md` for full detail. Quick reference:

- **DECISION-001 to 014** — pre-F3 lockdown
- **DECISION-015 (REFRAMED 2026-05-26)** — F3 SUCCESS post-fix: pooled +11.10%, all 3 datasets SUCCESS
- **DECISION-016** — F3 secondary metric definition (geometric mean per-dataset, locked SUCCESS)

---

## 8. Grade trajectory estimate

| Scenario | Estimated grade (max 10) |
|---|---|
| Current state (F3 closed with verification) | 8.2–8.7 |
| + F5 chapters written at submitted-PDF quality | 8.7–9.2 |
| + Tier-S interventions (reframe + abstention + MPC) | **9.2–9.4** |
| + Mock defences with revisions | +0.2–0.4 |

**Primary lever:** writing quality + honest framing.
**Secondary lever:** Tier-2 OptScaler MPC.
