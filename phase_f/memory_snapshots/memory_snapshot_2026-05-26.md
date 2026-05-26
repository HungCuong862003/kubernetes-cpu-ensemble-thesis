# memory_snapshot_2026-05-26.md

**Project:** Hybrid Ensemble Learning for Proactive Resource Prediction in Kubernetes
**Snapshot date:** 2026-05-26
**Phase:** F (computational work CLOSED; F5 writing remaining)
**Defence target:** March 2027 (~180 working days available)

---

## What is locked

**Pre-Phase F (BCF + HPA + Ensemble):** Locked since submission.
- BCF: AUC=0.80, percentile CI [0.7097, 0.8871], n=36, p=0.0097, predicate (ACF@24h > 0.2 AND h ≥ 30min)
- HPA v4: 225 ML strict / 675 reactive dominance across 11 cells (max_replicas=1000)
- Ensemble vs Naive (Alibaba): +0.25/+0.43/+1.33/+4.64pp at h10/30/60/120

**Phase F (closed today):**
- F1: NULL (cell-level router macro-F1 ≪ 0.55, intrinsic ACF saturation at n=12)
- F2: NULL (WPE partial-R² 0.079/0.056 vs 0.30 threshold, intrinsic substitute for ACF@24h)
- PAR: PARTIAL POSITIVE (XGB macro-F1=0.2166 vs 0.20 threshold)
- PAR PCA: 9 PCs for 80%, PC1 26.4%, PC1 ρ(ACF@24h)=0.84, within-dataset 94.1%
- PAR HPA: PAR ranks #1 (0.1205) > AlwaysC2 (0.0906) > Reactive (0) > BCF (−0.0213)
- F3 LoRA: **SUCCESS** primary (+8.17%) / **PARTIAL** secondary (3.44%) — DECISION-015

**Decisions:** 16 LOCKED (DECISIONS.md)
**Errata:** 13 logged (11 APPLIED, 2 PENDING: 012 Bitbrains pool source, 013 F3 disclosures)

---

## Three numbers to remember

| Anchor | Value |
|---|---|
| BCF AUC | 0.80 [0.71, 0.89], p=0.0097, n=36 |
| F3 primary improvement | +8.17% (SUCCESS) |
| PAR HPA ranking | PAR-XGB #1 at mean R² 0.1205 |

---

## Open issues (do not forget)

**Critical:**
1. F5 chapter writing not started (6 chapters, ~15–20 days)
2. Bibliography audit not complete (~6–8 days) — DO FIRST
3. Abstract + §1.4 reframe needed (~4–6 days)

**Disclosure obligations (ERRATA-013):**
4. F3 training objective deviation (ForCausalLMLoss not pinball, Chronos-2 API constraint)
5. F3 Bitbrains h=120 degradation (−15% to −19% at τ=0.7/0.8)
6. F3 secondary metric PARTIAL (3.44%)

**Honest framings to preserve (do NOT "fix"):**
7. PAR ByteDance catastrophe (R²-regret 0.97, granite_ttm spillover, 60/189 rows) — publishable finding
8. F1/F2 nulls — structural saturation contribution
9. Bitbrains pooled vs per-VM sign reversal — M5 aggregation precedent
10. ExtraTrees dominates NNLS weights — reframe headline rather than hide

---

## Highest-leverage remaining moves (from research reports)

| Tier | Item | Days | Status |
|---|---|---|---|
| E3 | Reframe headline (BCF+PAR+HPA-sim) | 4–6 | Not started |
| E6 | Bibliography audit + errata sheet | 6–8 | Not started — DO FIRST |
| E1 | OptScaler-style MPC + Fremer benchmark | 18–22 | Optional |
| E2 | SPCI/HopCPT residual conformal layer | 8–10 | Optional |
| E4 | PAR with abstention | 6–8 | Optional |
| E5 | Held-out BCF replication (Azure V2) | 6–8 | Optional |
| F5 | 6 chapter drafts | 15–20 | Critical path |
| Mocks | 2 mock defences | 10 | Final stretch |

**Total Tier 1 (must do):** ~28 days
**Total Tier 1 + 2 (recommended):** ~75–85 days
**Time available:** ~180 days. Plenty of buffer.

---

## Canonical source files (never regenerate, always cite from these)

- `comparison_table.csv` — ensemble R²
- `run.log` — NNLS weights (production)
- `bcf_pooled_3model.json` — BCF AUC + CI + p
- `leaderboard_v1.csv` — 4-model leaderboard (canonical for §4.7)
- `cross_dataset_headline_v2.csv` — NEW pool full-test (only for §5 disclosure)
- `phase_f/data/par_router_predictions.csv` — PAR routing predictions
- `phase_f/data/par_router_summary.json` — PAR aggregate metrics
- `phase_f/data/par_pca_results.csv` — PCA explained variance + Spearman
- `phase_f/data/par_hpa_comparison.csv` — 4-policy HPA table
- `phase_f/data/par_hpa_summary.json` — 4-policy ranking
- `phase_f/data/f3_finetune_results.json` — DECISION-015 (corrected secondary metric)
- `phase_f/data/f3_finetune_results.csv` — per-cell pinball
- `phase_f/data/f3_training_history.csv` — F3 training trajectory
- `phase_f/models/f3_lora_rank8/` — LoRA adapter weights (epoch 1 best)
- `hpa_simulation_*_v4.csv` — HPA grids (max_replicas=1000)
- `hpa_v4_dominance_per_dataset.csv` — ML strict / reactive dominance counts

---

## What NOT to do

- Don't re-run anything to "improve" F1, F2, ByteDance catastrophe. Those are findings.
- Don't reimplement OptScaler/AHPA/Madu from scratch (no public code, no ground truth).
- Don't chase new foundation models post-Sept 2026 (lock literature scope).
- Don't conflate leaderboard_v1 (K=20/K=50 subsample) with cross_dataset_headline_v2 (full-test). Different scopes.
- Don't use BCa CI for BCF (degenerate per DECISION-007).
- Don't apply ERRATA without supervisor co-signing on the consolidated sheet.

---

## Defence readiness checklist

- [ ] All chapters drafted (F5)
- [ ] Bibliography 100% verified (E6)
- [ ] Errata sheet signed by Dr. Ho Long Van
- [ ] Abstract reframed to BCF+PAR+HPA-sim (E3)
- [ ] Disclosures from ERRATA-013 applied in Ch4 + Ch5
- [ ] Mock defence 1 completed + revisions
- [ ] Mock defence 2 completed + final polish
- [ ] Slide deck: 3 killer + 18 main + 10 backup
- [ ] Attack-vector prepared responses

---

## Grade trajectory snapshot

| State | Estimate |
|---|---|
| Today (post-F3 closure) | 8.0–8.5 |
| + F5 chapters written | 8.5–9.0 |
| + Tier-S interventions | 9.0–9.2 |
| + Mock defences | +0.2–0.4 |

Writing quality and honest framing of negative results are the primary lever from this point forward.

---

**End of snapshot. See THESIS_STATE.md for full state, DECISIONS.md for decision register, ERRATA.md for active errata, 2026-05-26_d-today_session-close.md for this session's journal.**
