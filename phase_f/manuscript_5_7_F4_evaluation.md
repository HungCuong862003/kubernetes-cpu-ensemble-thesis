# Manuscript §5.7 — Routing alternative evaluation

(Draft text for inclusion in Chapter 5 of the thesis manuscript. Word count: ~620. Harvard citations, British spelling, active voice ~70%. Number all CSVs and parquets are traceable to phase_f/data/ canonical files.)

---

## §5.7 Routing alternative evaluation

To assess whether per-dataset adapter routing offered measurable benefits beyond the uniform v6 pipeline, we conducted a pre-registered closed-loop autoscaler evaluation comparing v6 against a v7 alternative. Under v7, the Alibaba dataset routed to a no-log1p adapter (`f3_lora_rank8_no_log1p/`), whilst Bitbrains and ByteDance retained the log1p + cap = 110 adapter (`f3_lora_rank8/`) used in v6. The hypothesis under test, derived from the F3 forecast-layer regression of v6 versus the no-log1p adapter on Alibaba (8% pinball gap; ERRATA-018, ERRATA-019), was that v7 would Pareto-dominate v6 on Alibaba whilst remaining indistinguishable on Bitbrains and ByteDance.

The evaluation was pre-registered to `phase_f/f4_evaluation_plan.md` and committed to git (SHA 012786af9d) on 27 May 2026 before any F4 closed-loop simulation ran, following Hofman et al. (2023). The pre-registration fixed: (i) metrics — per-container SLA violation rate and mean replica count over the simulated trajectory; (ii) datasets and seeds — three datasets, three seeds per (dataset, pipeline), 18 closed-loop simulations total; (iii) statistical procedure — paired stationary bootstrap (Politis and Romano, 1994) with R = 1,000 resamples, expected block length b ≈ T^{1/3}, Holm-Bonferroni correction across the six simultaneous tests (three datasets × two objectives); and (iv) the decision rule — adopt v7 if and only if v7 strictly Pareto-dominates v6 on Alibaba whilst tying or winning on the other two datasets, otherwise adopt v6.

The chance-constrained Model Predictive Control formulation followed Zou et al. (2024) at α = 0.95, with `static_chance` provisioning enough replicas to cover the worst q_{0.95} forecast in the next H = 12 steps. The HPA simulator extends our v2 simulator (§3.7) with a forecast-aware decision callback that consumes the per-dataset routed quantile parquet.

Table 5.7.1 reports the paired effects, expressed as v7 minus v6 per (container, seed) tuple. The Alibaba effect is the largest in absolute terms but partitions across the two objectives in opposite directions: v7 reduces the violation rate by 1.897 percentage points (95% CI [−2.190, −1.603]; p < 0.001 after Holm correction) and simultaneously increases the mean replica count by 0.220 (95% CI [+0.091, +0.353]; p < 0.001). Both effects are statistically robust. This is a Pareto tradeoff, not domination — v7 trades replica cost for violation reduction. On Bitbrains, v7 reduces violations by 0.231 percentage points (CI [−0.322, −0.148]; p < 0.001) with no statistically detectable replica-cost effect (Δ = −0.132; CI [−0.424, +0.141]; p = 0.4). On ByteDance, neither effect reaches significance after correction.

The pre-registered decision rule requires Alibaba dominance: the v7 confidence interval on both objectives must lie in the non-positive orthant. Whilst the Alibaba violation-rate CI is wholly negative, the replica-mean CI is wholly positive, falsifying the orthant condition. The rule therefore mandates v6 as the canonical pipeline. We adopt v6.

This outcome is not a failure of v7 routing. v7 produces statistically significant effects on two of the three datasets. Rather, the pre-registration successfully prevented the cherry-picking that a single-metric (e.g. violations-only) post-hoc evaluation would have rewarded. The Alibaba tradeoff is a finding in its own right: under a cost-asymmetric production objective where SLA violations are penalised more heavily than replica costs (Zhang et al., 2023; Pan et al., 2023), v7 routing becomes preferable on Alibaba. The current pre-registration uses strict Pareto, which is workload-agnostic and conservative.

The v7 routing infrastructure remains in the repository as a reproducibility artefact and supports future work investigating cost-weighted MPC objectives (§6), per-instance routing classifiers (Dange and Sarawagi, 2025), and CQR-wrapped quantile pipelines (Romano et al., 2019). The canonical F3-FT configuration documented in §3.7 (DECISION-018) is unchanged.

---

## Citation notes for §5.7

- Hofman, Chatzimparmpas, Sharma, Watts and Hullman (2023) — pre-registration for predictive modelling — arXiv:2311.18807
- Politis and Romano (1994) — stationary bootstrap — JASA 89(428):1303–1313
- Zou et al. (2024) — OptScaler — PVLDB 17(12):4090–4103
- Romano, Patterson and Candès (2019) — CQR — NeurIPS 32
- Dange and Sarawagi (2025) — TFMAdapter — CIKM 2025
- Pan et al. (2023) — MagicScaler — PVLDB 16(12):3808–3821
- Zhang et al. (2023) — AAPA — workload-aware autoscaling

## Companion table (Table 5.7.1) — F4 closed-loop paired effects

| Dataset | n (container, seed) | Δ violation rate (pp) | 95% CI | Δ replica mean | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| Alibaba | 14,760 | −1.897 | [−2.190, −1.603] | +0.220 | [+0.091, +0.353] | Pareto tradeoff |
| Bitbrains | 426 | −0.231 | [−0.322, −0.148] | −0.132 | [−0.424, +0.141] | Partial improvement |
| ByteDance | 279 | −0.022 | [−0.084, +0.030] | +0.021 | [−0.071, +0.100] | Indistinguishable |

Bootstrap resamples R = 1,000; block length b ≈ T^{1/3}. p-values from two-sided percentile test; Holm-Bonferroni family α = 0.05 across six tests. Verdict labels per ERRATA-021 disambiguation.

Source: `phase_f/data/f4_bootstrap_results.csv` and `phase_f/data/f4_pareto_decision.json`.
