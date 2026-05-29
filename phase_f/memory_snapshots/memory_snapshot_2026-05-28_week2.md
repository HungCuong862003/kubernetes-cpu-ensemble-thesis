# Memory snapshot — 2026-05-28 — F2 Week 2 close (DECISION-019)

**Save as `phase_f/memory_snapshots/memory_snapshot_2026-05-28_week2.md`.**

## One-line state

F2 null defended on five independent fronts + analytical explanation; three
Week 1 exceedances formally dismissed; F1 stays 4 features; all F2 compute done;
next is F5 manuscript writing.

## Canonical Week 2 numbers (verification-first — every number traces to a file)

**Five-front defence:**

1. Structural identification (`phase_f/data/week2/identification_limits.{md,csv,json}`):
   - 5 constant-per-dataset candidates → pr² = 0.078960 identical (SD = 0.0)
   - reproduces locked F2 (0.0790); r²_full = 0.9110, r²_reduced = 0.9034
   - Ibragimov-Müller t(2) = −0.34, p = 0.766
   - per-dataset WPE slopes: Alibaba −0.26, Bitbrains −4.19, ByteDance +2.48 (sign-discordant)

2. BH-FDR + Westfall-Young (`phase_f/data/week2/fdr_corrected_screen.{csv,md}`):
   - 0/26 survive BH-FDR (all BH_q = 1.0000) or WY (all WY_p > 0.49)
   - SB_TransitionMatrix: pr² = 0.4999, raw_p = 0.0775 (fails uncorrected), BH_q = 1.0, WY_p = 0.494
   - SP_Summaries: pr² = 0.3736, raw_p = 0.336
   - DN_OutlierInclude: pr² = 0.3090, raw_p = 0.477
   - WPE confirmatory on Bitbrains clip: 0.1844 NULL

3. CV-stratification (`phase_f/data/week2/cv_stratification.{csv,md}`), active-only stratum (137 VMs, no clip):
   - SB_TransitionMatrix 0.500 → 0.0147
   - SP_Summaries → 0.0005
   - DN_OutlierInclude → 0.0236
   - WPE → 0.0177
   - CV tertiles: t1 = 0.985, t2 = 1.123; idle VMs bb_609–613 CV = 1.62–1.85 (HIGH, not low)

4. Cluster-leverage (`phase_f/data/week2/loo_cluster_leverage*.{csv,md}`):
   - SB_TransitionMatrix full pr² = 0.4999; dropping bb_611 −0.078, bb_613 −0.076, bb_612 −0.060
   - idle VMs are top cluster-leverage points (e.g. bb_612 leverage 0.16 vs threshold 0.056)
   - single-VM LOO "100% above threshold" because other idle VMs remain; CV-strat is the correct combined test

5. Hierarchical Bayes (`phase_f/data/week2/hierarchical_bayes_loo.csv`, consolidated):
   - WPE: partial-R² = 0.0000 [−0.0163, 0.0161], Pr(≥0.30) = 0.0000, ELPD diff +124.5 (SE 150), target_accept 0.95, 0 divergences — CLEAN
   - SB_TransitionMatrix: partial-R² = 0.0757 [−0.0104, 0.1155] (CI crosses 0), Pr(≥0.30) = 0.0000, ELPD diff −1236.5 (SE 300) → inclusion HURTS, target_accept 0.99, 6 divergences (was 870 at 0.95), minor R-hat/ESS warnings on horseshoe hyperparams (does not affect conclusion)
   - both fit on n = 17,879 pooled series-horizons, 3 datasets

**Analytical** (`phase_f/data/week2/bandt_shiha_ar1_numerical.{csv,md,pdf}`):
   - m=3 up-pattern closed form vs Monte-Carlo: max error 0.0021 (verified)
   - PE(m=4): 1.0000 at φ=0 → 0.8711 at |φ|=0.95, monotone in |φ|
   - WPE(m=4) analogous; ρ(1) = φ exactly → PE/WPE deterministic functions of ACF
   - claim: WPE analytically redundant with ACF on AR(1)-like short-memory Gaussian processes

## Decision numbering (CONFIRMED against live DECISIONS.md)

- D17 = F2 chapter scope / per-series methodology / clip policy (Week 1, 2026-05-28)
- D18 = F4 v6 closure (taken)
- **D19 = F2 Week 2 five-front defence (NEW this session)**
- ERRATA register at 016; ERRATA-013 = Bitbrains test_ensemble_hetero.npy = Alibaba data (Apr 30 swap); ERRATA-014/015/016 = F3 issues. NO new erratum for Week 2.

## Scripts (all in `phase_f/scripts/`, tested + run on Vast.ai C.38014225)

identification_limits.py, fdr_corrected_screen.py, cv_stratification.py,
loo_cluster_leverage.py, bandt_shiha_ar1_numerical.py, hierarchical_bayes.py,
consolidate_bayes_loo.py

## Environment notes (Vast.ai C.38014225)

- pymc 6.0.1 + arviz 1.1.0 (arviz_stats rewrite) — NOT 5.x. Three API breaks
  handled defensively: ELPD attribute (`_get_elpd` tries elpd_loo/elpd/dict/Series),
  az.compare column handling, az.to_netcdf removed (use idata.to_netcdf or skip).
- netCDF traces don't save (no netCDF4/h5netcdf backend) — harmless, numbers in CSV.
  `pip install h5netcdf` if reproducible traces wanted.
- Horseshoe needs non-centred reparam (a_ds AND b_cand) + target_accept ≥ 0.95;
  funnel-prone features may need 0.99. `--target_accept` CLI flag added.
- np.math.factorial removed in numpy 2.x → use math.factorial.

## Corrections to prior memory

- Idle Bitbrains VMs bb_609–613 are HIGH-CV (1.62–1.85), NOT low-CV. Near-zero
  mean utilisation inflates CV. Prior "low-CV idle VMs" framing was wrong.
- SB_TransitionMatrix Bayesian partial-R² is 0.076 (clean, ta=0.99), NOT 0.10
  (the 0.10 was a divergent-chain artefact at ta=0.95).

## Bibliography to add at F5 (batched with existing biblio audit)

Mundlak (1978), Ibragimov-Müller (2016), Snijders-Bosker (2012),
Cameron-Gelbach-Miller (2008), MacKinnon-Webb (2018),
MacKinnon-Nielsen-Webb (2023), Benjamini-Hochberg (1995),
Benjamini-Yekutieli (2001), Westfall-Young (1993), Piironen-Vehtari (2017),
Vehtari-Gelman-Gabry (2017), Gelman-Goodrich-Gabry-Vehtari (2019),
Bandt-Shiha (2007), Bandt-Pompe (2002), Fadlallah et al. (2013),
Zunino et al. (2008), Robinson (1950), King (1997).

## Next

F5 six-chapter manuscript writing. F2 chapter §5 has clean five-front structure.
No further F2 compute.
