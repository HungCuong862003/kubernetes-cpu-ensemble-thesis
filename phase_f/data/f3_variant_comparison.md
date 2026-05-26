# F3 LoRA / DoRA / Curriculum variant comparison

## Primary metric (mean pinball h=60min tau=0.9 vs baseline 1.921658)

| Variant | Main Delta% | Geom mean Delta% | Alibaba | Bitbrains | ByteDance | Hold-out Delta% | Repl. Delta_pp | Verdict |
|---|---|---|---|---|---|---|---|---|
| lora_rank8 (original v2) | +8.17 | +3.44 | +2.51 | +9.30 | +1.75 | N/A | N/A | SUCCESS |
| lora_rank4 | +29.90 | N/A | N/A | N/A | N/A | -47.13 | 77.03 | SUCCESS |
| lora_rank8 | +36.24 | N/A | N/A | N/A | N/A | -46.76 | 82.99 | SUCCESS |
| lora_rank16 | +33.84 | N/A | N/A | N/A | N/A | -47.09 | 80.93 | SUCCESS |
| dora_rank8 | +32.53 | N/A | N/A | N/A | N/A | -46.30 | 78.83 | SUCCESS |
| curriculum_rank8 | -32.34 | N/A | N/A | N/A | N/A | -80.01 | 47.67 | FAILURE |

## Notes on metrics

- Main Delta%: improvement on first 70% of test series (alphabetically), where
  +5% = SUCCESS, +2-5% = PARTIAL, <+2% = FAILURE per DECISION-005.
- Hold-out Delta%: improvement on last 30% of test series, never seen by val
  checkpoint selection. Large gap to main indicates checkpoint-selection bias.
- Replication Delta_pp: |main - holdout|. Smaller = better generalisation.
- Geom mean Delta%: geometric mean of per-dataset percentage improvements at
  the primary cell (h=60, tau=0.9). N/A when any dataset has negative Delta.

## Methodological caveats

1. Training objective for all variants: native Chronos-2 ForCausalLMLoss
   (Chronos-2 API does not expose differentiable quantile head).
2. Asymmetric pinball applied only at evaluation (post-hoc), not at training.
3. Curriculum variant mixes horizons within each batch (vs rotating per epoch).
4. DoRA variant uses weight-decomposed LoRA (Liu et al., ICML 2024, arXiv:2402.09353).
5. LoRA rank sweep verifies rank=8 from DECISION-016.
6. Original v2 evaluation pooled all series; main-vs-holdout split exposes
   cross-series heterogeneity that the pooled +8.17% headline obscures.