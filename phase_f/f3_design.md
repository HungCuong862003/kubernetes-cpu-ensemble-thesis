# F3 design — cost-asymmetric Chronos-2 fine-tune

**Date locked:** 2026-05-25 (Phase F3 Day 1, before any training)
**Status:** PRE-REGISTERED, locked

## Objective

LoRA fine-tune Amazon Chronos-2 (`amazon/chronos-2`, ~120M parameters) with
asymmetric pinball loss to bias predictions toward over-provisioning, evaluated
against zero-shot Chronos-2 on the canonical thesis datasets and horizons.

## Quantile levels

τ ∈ [0.5, 0.7, 0.8, 0.9, 0.95] — τ=0.5 is the median baseline; τ=0.7-0.95 are
over-provisioning quantiles aligned with OptScaler (Lu et al., VLDB 2024) and
the TimesFM quantile-head fine-tune in Cisana (arXiv:2410.11773, 2024-25).

## Datasets and horizons

- Alibaba 2018 (~4,876 containers, 5-min intervals)
- Bitbrains GWA-T-12 (142 VMs, 5-min intervals)
- ByteDance IaaS (93 containers, 10-min intervals)

Horizons: [10, 30, 60, 120] minutes. ByteDance has no h=10 due to 10-min sampling.

## Context configuration

- N_CONTEXT = 512 observations per series
- Batch size = 256
- Per container: context = last (N_CONTEXT + horizon_obs) obs minus final horizon_obs;
  target = final horizon_obs observations.
- Containers with length < N_CONTEXT + horizon_obs are skipped (no padding).

## Pre-registered F3 success criterion (LOCKED)

Primary metric: pinball loss at τ = 0.9, h = 60 min,
averaged across the three datasets. Improvement over zero-shot Chronos-2.

- F3 SUCCESS: ≥ 5.0% pinball-loss improvement.
- F3 PARTIAL: ≥ 2.0% improvement.
- F3 FAILURE: < 2.0% improvement → triggers DECISION-015.

## Method (for the LoRA fine-tune that follows this baseline)

- Backbone: Chronos-2 frozen
- Adaptation: LoRA (rank 8, α = 16, dropout 0.1) on the quantile head + last
  two transformer blocks
- Loss: weighted pinball at each τ ∈ [0.5, 0.7, 0.8, 0.9, 0.95]
- Optimiser: AdamW, lr 1e-4, weight decay 0.01
- 20 epochs, early stopping on val pinball at τ = 0.9
- Vast.ai RTX A4000 (16 GB VRAM)

## Files

- Baseline JSON: `phase_f/data/f3_zero_shot_baseline.json`
- Baseline CSV: `phase_f/data/f3_zero_shot_baseline.csv`
- This design doc: `phase_f/f3_design.md`
