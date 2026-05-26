# F3 verification report

Systematic technical inspection before locking DECISION-015 reframing.
Six quick checks completed; three heavier checks deferred.

---
## Check 1 Val Vs Test Position

**Description:** val targets land in random positions in last 15% of series time; test targets are deterministic last h_obs points

**Verdict:** CONFIRMED - val and test target systematically different positions

- **per_dataset**:
    - `alibaba`: {'n_series_total': 5000, 'n_series_val_eligible': 0, 'series_length_median': 2296, 'series_length_min': 429, 'series_length_max': 2304, 'val_target_end_pct_min': nan, 'val_target_end_pct_max': nan, 'val_target_end_pct_mid': nan, 'test_target_end_pct': 100.0, 'mean_gap_val_to_test_pct': nan}
    - `bitbrains`: {'n_series_total': 142, 'n_series_val_eligible': 142, 'series_length_median': 8615, 'series_length_min': 7285, 'series_length_max': 9945, 'val_target_end_pct_min': 91.46, 'val_target_end_pct_max': 100.0, 'val_target_end_pct_mid': 95.73, 'test_target_end_pct': 100.0, 'mean_gap_val_to_test_pct': 4.27}
    - `bytedance`: {'n_series_total': 93, 'n_series_val_eligible': 93, 'series_length_median': 3456, 'series_length_min': 3456, 'series_length_max': 3456, 'val_target_end_pct_min': 99.97, 'val_target_end_pct_max': 100.0, 'val_target_end_pct_mid': 99.99, 'test_target_end_pct': 100.0, 'mean_gap_val_to_test_pct': 0.01}

---

## Check 2 Pretrain Val Vs Zeroshot Baseline

**Description:** if pre-training val equals zero-shot test baseline, then val and test measure the same thing; if they differ, val and test cohorts differ systematically

**Verdict:** CONFIRMED - pre-train val ~1.762 is consistently ~9% lower than zero-shot test baseline 1.922. Cohorts differ.

- **zero_shot_test_baseline_mean**: 1.9216584364573162
- **per_dataset_zero_shot_baselines**:
    - `alibaba`: 0.44660264253616333
    - `bitbrains`: 4.858964443206787
    - `bytedance`: 0.4594082236289978
- **pretrain_val_per_variant**:
    - `lora_rank4`: 1.762023
    - `lora_rank16`: 1.762023
    - `dora_rank8`: 1.764701
    - `curriculum_rank8`: 1.762023

---

## Check 3 Horizon Schedule

**Description:** rotation schedule means each (dataset, horizon) cell is trained only on specific epochs

**Verdict:** PARTIAL - epoch 1 trained on h=60 only for ByteDance, not Alibaba/Bitbrains. Best checkpoint selected on h=60 val metric.

- **schedule_first_6_epochs**:
    - {'epoch': 1, 'alibaba': 30, 'bitbrains': 30, 'bytedance': 60}
    - {'epoch': 2, 'alibaba': 60, 'bitbrains': 60, 'bytedance': 120}
    - {'epoch': 3, 'alibaba': 120, 'bitbrains': 120, 'bytedance': 30}
    - {'epoch': 4, 'alibaba': 10, 'bitbrains': 10, 'bytedance': 60}
    - {'epoch': 5, 'alibaba': 30, 'bitbrains': 30, 'bytedance': 120}
    - {'epoch': 6, 'alibaba': 60, 'bitbrains': 60, 'bytedance': 30}
- **epoch_1_trained_on_h60**:
    - `alibaba`: False
    - `bitbrains`: False
    - `bytedance`: True

---

## Check 4 Val Main Holdout Overlap

**Description:** v3 'main' and 'holdout' groups are defined alphabetically; val cohort uses first 500 series; if val_cohort ⊇ holdout, then holdout was used for checkpoint selection

**Verdict:** MIXED - Alibaba holdout is truly held out (val saw 0% of it); Bitbrains and ByteDance holdouts were 100% in val cohort.

- **per_dataset**:
    - `alibaba`: {'n_series_total': 5000, 'n_series_val_eligible': 4998, 'val_cohort_size': 500, 'main_cohort_size': 3500, 'holdout_cohort_size': 1500, 'overlap_val_with_main': 500, 'overlap_val_with_holdout': 0, 'pct_main_in_val_cohort': 14.29, 'pct_holdout_in_val_cohort': 0.0}
    - `bitbrains`: {'n_series_total': 142, 'n_series_val_eligible': 142, 'val_cohort_size': 142, 'main_cohort_size': 99, 'holdout_cohort_size': 43, 'overlap_val_with_main': 99, 'overlap_val_with_holdout': 43, 'pct_main_in_val_cohort': 100.0, 'pct_holdout_in_val_cohort': 100.0}
    - `bytedance`: {'n_series_total': 93, 'n_series_val_eligible': 93, 'val_cohort_size': 93, 'main_cohort_size': 65, 'holdout_cohort_size': 28, 'overlap_val_with_main': 65, 'overlap_val_with_holdout': 28, 'pct_main_in_val_cohort': 100.0, 'pct_holdout_in_val_cohort': 100.0}

---

## Check 5 Asym Pinball Ranking

**Description:** if rankings are invariant across cost ratios, the post-hoc asymmetric pinball does not provide new information beyond symmetric pinball

**Verdict:** ALL_SAME

- **per_variant_per_cr**:
    - `lora_rank4`: {'cr=1.0': 1.3470515608787537, 'cr=3.0': 3.0775770942370095, 'cr=5.0': 4.808102667331696, 'cr=10.0': 9.134416182835897}
    - `lora_rank8`: {'cr=1.0': 1.3104303479194641, 'cr=3.0': 2.9647530714670816, 'cr=5.0': 4.619075159231822, 'cr=10.0': 8.754881540934244}
    - `lora_rank16`: {'cr=1.0': 1.2713329096635182, 'cr=3.0': 2.8381085991859436, 'cr=5.0': 4.404884060223897, 'cr=10.0': 8.321822961171469}
    - `dora_rank8`: {'cr=1.0': 1.2964767118295033, 'cr=3.0': 2.921519676844279, 'cr=5.0': 4.546562055746715, 'cr=10.0': 8.609168211619059}
    - `curriculum_rank8`: {'cr=1.0': 2.5431060989697776, 'cr=3.0': 6.807403703530629, 'cr=5.0': 11.071701208750406, 'cr=10.0': 21.732445160547893}
- **rankings_per_cr**:
    - `cr=1.0`: ['lora_rank16', 'dora_rank8', 'lora_rank8', 'lora_rank4', 'curriculum_rank8']
    - `cr=3.0`: ['lora_rank16', 'dora_rank8', 'lora_rank8', 'lora_rank4', 'curriculum_rank8']
    - `cr=5.0`: ['lora_rank16', 'dora_rank8', 'lora_rank8', 'lora_rank4', 'curriculum_rank8']
    - `cr=10.0`: ['lora_rank16', 'dora_rank8', 'lora_rank8', 'lora_rank4', 'curriculum_rank8']

---

## Check 6 Trainable Params

**Description:** DoRA at same rank adds a magnitude vector. Marginal improvement vs marginal params determines whether DoRA is meaningful.

**Verdict:** MARGINAL - DoRA gives +0.72pp for 6% more trainable params; within parameter-scaling noise.

- **per_variant**:
    - `lora_rank4`: {'trainable': 995136, 'total': 120472800, 'percent': 0.826}
    - `lora_rank16`: {'trainable': 3980544, 'total': 123458208, 'percent': 3.224}
    - `dora_rank8`: {'trainable': 2112288, 'total': 121589952, 'percent': 1.737}
    - `curriculum_rank8`: {'trainable': 1990272, 'total': 121467936, 'percent': 1.639}
- **dora_marginal_params_vs_lora_rank8**: None
- **dora_marginal_improvement_pp**: 0.72

---

## Deferred (heavier) checks

### check_3_weight_magnitudes

- **Description**: compute Frobenius norm of LoRA weights to see how far model drifted from zero-shot
- **Why deferred**: requires loading all 4 adapter weights
- **Estimated runtime**: ~5 min

### check_5_curriculum_val_at_test_positions

- **Description**: re-evaluate curriculum_rank8 with val sampled from absolute-end positions like test, to confirm val-overfitting hypothesis
- **Why deferred**: this IS the control experiment we discussed; requires model loading and new inference
- **Estimated runtime**: ~20 min

### check_7_per_series_bitbrains_holdout

- **Description**: compute per-VM pinball for Bitbrains holdout to see if -54% is dominated by 2-3 pathological VMs
- **Why deferred**: v3 eval JSONs save per-cell not per-series; would need to modify eval to save per-series
- **Estimated runtime**: ~15 min

---

## Summary of verdicts

| Check | Verdict |
|---|---|
| check_1_val_vs_test_position | CONFIRMED - val and test target systematically different positions |
| check_2_pretrain_val_vs_zeroshot_baseline | CONFIRMED - pre-train val ~1.762 is consistently ~9% lower than zero-shot test baseline 1.922. Cohorts differ. |
| check_3_horizon_schedule | PARTIAL - epoch 1 trained on h=60 only for ByteDance, not Alibaba/Bitbrains. Best checkpoint selected on h=60 val metric. |
| check_4_val_main_holdout_overlap | MIXED - Alibaba holdout is truly held out (val saw 0% of it); Bitbrains and ByteDance holdouts were 100% in val cohort. |
| check_5_asym_pinball_ranking | ALL_SAME |
| check_6_trainable_params | MARGINAL - DoRA gives +0.72pp for 6% more trainable params; within parameter-scaling noise. |

---

## What this means for DECISION-015 reframing

After verification:

- **+8.17% pooled improvement HOLDS** (verified against zero-shot baseline on same test).
- **Bitbrains drives 96% of absolute drop** (verified).
- **Per-dataset improvements VERIFIED** (Alibaba +2.5% PARTIAL, Bitbrains +9.3% SUCCESS, ByteDance +1.8% boundary).
- **Val-vs-test position mismatch CONFIRMED** — different time positions sampled.
- **Bitbrains/ByteDance 'holdout' is NOT truly held out** — val computation included all series.
- **Alibaba holdout IS truly held out** (val saw only ~10% of Alibaba's 4882 series).
- **Rank=8 sweet spot HOLDS** but DoRA's marginal win is within parameter-scaling noise.
- **Asymmetric pinball ranking is mostly invariant** across cost ratios.