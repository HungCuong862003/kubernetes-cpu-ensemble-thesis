# F4 Evaluation Pre-Registration

**Author:** Jimmy (Phan Nguyen Hung Cuong, ITDSIU21078)
**Supervisor:** Dr. Ho Long Van
**Committed:** 2026-05-27 (timestamp via `git commit`)
**Phase:** F4 (chance-constrained MPC integration)
**Status:** LOCKED before any F4 closed-loop simulation runs.

## Purpose

This document pre-registers the methodology for comparing two F3 forecast pipelines (v6 uniform vs v7 per-dataset adapter routing) on downstream Model Predictive Control (MPC) autoscaling outcomes. The pre-registration neutralises selection bias: v7's per-dataset routing rule was constructed *after* observing v6's per-dataset pinball losses on the F3 test split, so any v7 vs v6 forecast-layer comparison on F3 is post-hoc. The F4 closed-loop SLO and cost metrics are the held-out evaluation.

Following Hofman, Chatzimparmpas, Sharma, Watts & Hullman (2023, "Pre-registration for Predictive Modeling," arXiv:2311.18807): the metrics, datasets, statistical tests, multiple-testing correction, and decision rule below are fixed before computing any F4 result. Any deviation from this plan must be documented in `phase_f/ERRATA.md` with rationale.

## Forecast pipelines under comparison

**v6 (uniform):** Single adapter `phase_f/models/f3_lora_rank8/` applied to all three datasets with log1p input + expm1 output + cap=110 + Chernozhukov-Fernandez-Val-Galichon (2010) sort. Canonical parquet: `phase_f/data/f3_lora_quantiles_h060_cad30_sorted.parquet`.

**v7 (per-dataset routing):** Routed via `phase_f/configs/router_v7.yaml`. Alibaba uses `f3_lora_rank8_no_log1p/` (identity transforms); Bitbrains and ByteDance use `f3_lora_rank8/` (log1p + cap=110). Same CFG sort post-hoc. Canonical parquet: `phase_f/data/f3_quantiles_per_dataset_h060_cad30_sorted.parquet`.

## Datasets

Three datasets, evaluated identically:
- **Alibaba** Cluster Trace 2018 (~5,000 containers, 5-min cadence, h=60 min forecast horizon)
- **Bitbrains** GWA-T-12 (142 VMs, 5-min cadence, h=60 min)
- **ByteDance** IaaS (93 containers, 10-min cadence, h=60 min)

## Random seeds

Three seeds: `{42, 1729, 7331}`. Any stochastic component of the autoscaler (e.g., pod startup jitter if added) seeded deterministically per (dataset, pipeline, seed). Forecasts themselves are deterministic given the Chronos-2 quantile output and the saved parquet, so seeds vary only the autoscaler simulation.

## Metrics

### Forecast layer (computed on the F3 test split, descriptive — not the headline)

These are reported for diagnostic completeness. They are NOT the decision metrics, because v7's routing was constructed from v6's pinball loss, so they are contaminated by selection bias.

| Metric | Formula | Source |
|---|---|---|
| Pinball@0.9 | mean over rows of `(τ−1{y<q})·(y−q)` for τ=0.9 | Koenker & Bassett (1978) |
| Weighted Interval Score at α=0.95 | `(upper−lower) + (2/α)·(lower−y)·1{y<lower} + (2/α)·(y−upper)·1{y>upper}` | Gneiting & Raftery (2007, JASA 102:359); Bracher–Gneiting–Reich (2020, PLOS Comp Biol) |
| Empirical coverage at α=0.95 | mean over rows of `1{q_{0.025} ≤ y ≤ q_{0.975}}` | standard |
| Coverage gap (pp) | empirical coverage − nominal (95%) | standard |
| Quantile crossing rate | fraction of rows where any consecutive quantile pair inverts post-sort | sanity check; expected 0 |
| Tail-truncation rate | fraction of rows where `max_output` cap activated | new |

### Closed-loop layer (computed on F4 trajectories, **HEADLINE**)

These are the genuinely held-out metrics — the routing decision did not see them.

| Metric | Definition | Aggregation |
|---|---|---|
| **p95 SLO-violation rate** | percentage of control intervals where realised CPU demand > provisioned capacity, taken at the 95th percentile across containers | per-dataset |
| **Average replica-minutes** | sum over control intervals of `replicas(t) × Δt`, normalised by simulation length | per-dataset |
| Replica churn | (scale-ups + scale-downs) per simulated hour | per-dataset, secondary |
| Empirical chance-constraint satisfaction | did `mean(violation_indicator) ≤ 0.05` on the trajectory? | per-dataset, binary |

Convention follows OptScaler (Zou et al. 2024, PVLDB 17(12):4090–4103), MagicScaler (Pan et al. 2023, PVLDB 16(12):3808–3821), DeepScaler (Meng et al. 2023, ASE).

## Statistical procedure

### Per-dataset hypothesis test (the decision test)

For each dataset D ∈ {Alibaba, Bitbrains, ByteDance} and each objective O ∈ {SLO-violation rate, replica-minutes}, compute the **paired difference**:
```
Δ_D,O(seed) = metric_v7(D, seed) − metric_v6(D, seed)
```

Compute 95% CI for the population mean of Δ via the **stationary bootstrap** of Politis & Romano (1994, JASA 89:1303–1313), with:
- 1,000 resamples
- Expected block length `b ≈ T^{1/3}` where T is the number of control intervals
- Paired resampling (same block indices applied to both pipelines)

The block bootstrap addresses autocorrelation in violation indicators and replica trajectories (Künsch 1989, Annals of Statistics 17:1217; Romano & Wolf 2005, Econometrica 73:1237 endorse stationary bootstrap for forecast-comparison-style multiple testing under dependence).

### Multiple-testing correction

Six simultaneous tests (3 datasets × 2 objectives). Apply **Holm-Bonferroni** (Holm 1979) for family-wise error rate control. With α_family = 0.05:
- Sort the 6 p-values (computed from the bootstrap as `2 · min(P(Δ≥0), P(Δ≤0))`) in ascending order.
- Test the smallest against α/6 = 0.00833; if rejected, test the next against α/5; etc.
- Stop at the first non-rejection.

### Cross-dataset aggregation (descriptive only)

Wilcoxon signed-rank test across the 3 datasets reported as directional commentary, NOT as the headline. With n=3 the test is severely under-powered per Demšar (2006, JMLR 7:1–30) and García & Herrera (2008, JMLR 9:2677–2694). Headline output is per-dataset verdicts.

### Hypervolume rejected

With only two operating points per dataset, hypervolume (Zitzler & Thiele 1998; Guerreiro–Fonseca–Paquete 2021, ACM CSUR 54:119) reduces to a reference-point-sensitive scalar without informational gain over strict dominance with bootstrap CIs.

## Decision rule

After applying Holm-Bonferroni at α_family = 0.05, declare per-dataset verdicts:

- **v7 dominates v6 on D** iff (a) both Δ_SLO and Δ_cost CIs lie in the non-positive orthant AND (b) at least one Δ is strictly negative at the 5% level.
- **v6 dominates v7 on D** iff symmetrically reversed.
- **Indistinguishable on D** otherwise (CIs straddle zero on at least one objective).

**Final routing decision:**
- **Adopt v7 (per-dataset routing) IFF:** v7 dominates v6 on Alibaba AND v7 is indistinguishable from v6 on both Bitbrains and ByteDance.
- **Adopt v6 (uniform):** any other outcome — including v7 dominating everywhere (in which case v6 is still a defensible simpler fallback) AND any outcome where v7 loses on any dataset.

The asymmetry (v7 must win Alibaba; ties elsewhere are acceptable for v7) reflects the design hypothesis: v7 was constructed specifically to recover the Alibaba forecast-layer regression. If that recovery does not translate to closed-loop wins on Alibaba, v7 has no justification.

**If v7 loses on any dataset:** Do NOT iterate to a v8. Adopt v6, document v7 as a failed hypothesis in §Future Work, and frame v6's Alibaba pinball regression as the documented cost of uniform preprocessing (which the BCF v2 framing already absorbs).

## Robustness ablations (mandatory, not optional)

**Cap sensitivity on Bitbrains** (1 dataset, 1 seed, 3 caps, 3 runs, ~1.5 h):
- Cap ∈ {105, 110, 115} for the v6 pipeline on Bitbrains
- Report tail-truncation rate and pinball@0.9 per cap
- If the v6 vs v7 verdict on Bitbrains flips between caps, declare both pipelines indistinguishable on Bitbrains *regardless of cap=110 result*.

**Calibration disclosure** (no extra compute):
- Report coverage gap separately for v6 and v7 per dataset.
- If empirical coverage gap exceeds 5 percentage points on any dataset, flag the pipeline as "calibration-failing" and add CQR wrapping (Romano-Patterson-Candès 2019, NeurIPS 32, arXiv:1905.03222) to §Future Work.

## Compute budget

| Step | Configurations | Time | Total |
|---|---|---|---|
| F4 closed-loop simulations | 3 datasets × 2 pipelines × 3 seeds | ~30 min each | ~9 h |
| Cap-sensitivity ablation | 3 caps × 1 dataset × 1 seed | ~30 min each | ~1.5 h |
| Bootstrap analysis (on saved trajectories) | 1,000 resamples × 6 tests | minutes (CPU) | <30 min |
| **Total compute** | | | **~11 h on RTX 5070 Ti** |

## Deferred to Future Work

These are NOT part of the F4 evaluation. Mentioning them now blocks viva questions of the form "why didn't you also..."

- Conformalised wrapping (CQR — Romano-Patterson-Candès 2019)
- Learned routing (a small classifier on input statistics deciding which adapter, replacing the static `dataset_id` map)
- Hyperparameter sweep beyond the cap = {105, 110, 115} sanity check
- Hypervolume / many-pipeline Pareto analysis (would require N > 2 pipelines)
- Rank ablation of LoRA (rank ∈ {4, 8, 16})
- Cross-architecture comparison against Time-MoE (Shi et al., ICLR 2025) and TimesFM

## Reproducibility commitments

For every F4 run, the harness writes a `router_resolved.json` capturing:
- SHA-256 of each `adapter_model.safetensors`
- SHA-256 of each `transform_metadata.json`
- Git commit SHA of the codebase
- PEFT version, PyTorch version, CUDA version
- Random seed
- GPU model (cross-GPU byte-identical reproducibility NOT guaranteed; see §Caveats below)

## Caveats

1. **GPU-determinism is not byte-exact across hardware generations.** Same RTX 5070 Ti + same driver + `CUBLAS_WORKSPACE_CONFIG=:4096:8` reproduces; cross-GPU does not.
2. **Chronos-2 quantile mode is deterministic** but masking and any stochastic components in the inference path are seeded.
3. **Three datasets is below the threshold** for cross-dataset rank-based tests to be powerful per Demšar (2006); per-dataset verdicts are therefore the honest output.
4. **The PEFT `set_adapter` silent-failure issue (GitHub PEFT #1802)** is guarded against by the per-batch adapter unit test in the resaver harness.

## Sign-off

This plan is committed to git as the v7 hypothesis. Modifications after this commit require a documented errata entry. The F4 closed-loop runs are the held-out evaluation; their numerical results MUST NOT inform any further changes to v6 or v7 design.

Git commit: `<to be filled after commit>`
