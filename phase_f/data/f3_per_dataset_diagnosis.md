# F3 per-dataset hold-out diagnosis

**Goal:** Determine whether the F3 hold-out failure is uniform (checkpoint-
selection bias) or concentrated on specific datasets (transfer issue).

**Metric:** Pinball loss at h=60min, tau=0.9, per dataset.

## Per-dataset baselines (zero-shot)

| Dataset | Baseline pinball |
|---|---|
| alibaba | 0.45 |
| bitbrains | 4.86 |
| bytedance | 0.46 |

## Main group improvement %

First 70% of series alphabetically — partially overlaps val cohort.

| Variant | alibaba | bitbrains | bytedance |
|---|---|---|---|
| lora_rank4 | +5.54 | +34.56 | +4.31 |
| lora_rank8 | +5.60 | +36.94 | +3.01 |
| lora_rank16 | +5.53 | +39.14 | +5.29 |
| dora_rank8 | +5.61 | +37.69 | +4.12 |
| curriculum_rank8 | +5.62 | -40.52 | +17.30 |

## Hold-out group improvement %

Last 30% of series alphabetically — never seen by val checkpoint selection.

| Variant | alibaba | bitbrains | bytedance |
|---|---|---|---|
| lora_rank4 | -4.76 | -55.65 | +1.72 |
| lora_rank8 | -4.72 | -54.33 | -1.17 |
| lora_rank16 | -4.30 | -55.27 | -2.24 |
| dora_rank8 | -4.55 | -53.93 | -6.17 |
| curriculum_rank8 | -6.51 | -91.96 | -25.06 |

## Replication delta |main - holdout|

Larger = worse generalisation across the within-series alphabetical split.

| Variant | alibaba | bitbrains | bytedance |
|---|---|---|---|
| lora_rank4 | 10.30 | 90.21 | 2.59 |
| lora_rank8 | 10.33 | 91.26 | 4.18 |
| lora_rank16 | 9.83 | 94.41 | 7.53 |
| dora_rank8 | 10.16 | 91.62 | 10.30 |
| curriculum_rank8 | 12.13 | 51.44 | 42.36 |

## Summary statistics

**Per dataset (across variants):**

| Dataset | Mean replication delta | Std |
|---|---|---|
| alibaba | 10.55pp | 0.81pp |
| bitbrains | 83.79pp | 16.23pp |
| bytedance | 13.39pp | 14.73pp |

**Per variant (across datasets):**

| Variant | Mean replication delta | Spread (max - min) |
|---|---|---|
| lora_rank4 | 34.37pp | 87.62pp |
| lora_rank8 | 35.26pp | 87.08pp |
| lora_rank16 | 37.26pp | 86.88pp |
| dora_rank8 | 37.36pp | 81.46pp |
| curriculum_rank8 | 35.31pp | 39.31pp |

## Interpretation rubric

- **Uniform high delta (60-90pp on all 3 datasets):** Checkpoint-selection bias.
  F3 fine-tuning overfits the val cohort (first ~500 series alphabetically).
  PAR per-series approach is the right remedy.
- **Bitbrains delta much higher than others:** Scale-artefact dominance.
  Consistent with memory: Bitbrains contributes 84% of mean pinball.
- **One dataset isolated catastrophe (e.g. only ByteDance fails):** Cross-dataset
  transfer issue. Per-dataset adapters would help; not in current scope.
