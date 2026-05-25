"""
f3_baseline_chronos2.py — Phase F3 zero-shot Chronos-2 baseline (production).

NO AutoGluon. Direct chronos.BaseChronosPipeline usage based on probe v4 results.

For each (dataset, horizon) combination:
  1. Load test parquet
  2. Per container: split into context (last N_CONTEXT obs before final H) + target (final H obs)
  3. Stack contexts into 3D tensor (n_containers, 1, N_CONTEXT)
  4. Batch through Chronos-2 predict_quantiles
  5. Stack per-series outputs, squeeze n_variates axis, reorder to (n_quantiles, n_series, n_future)
  6. Compute pinball loss at τ ∈ {0.5, 0.7, 0.8, 0.9, 0.95} per (dataset, horizon, tau)

Containers with insufficient length (< N_CONTEXT + H observations) are skipped.

Output:
  phase_f/data/f3_zero_shot_baseline.json
  phase_f/data/f3_zero_shot_baseline.csv
  phase_f/f3_design.md  (pre-registration doc)

F3 success criterion (PRE-REGISTERED, locked before any fine-tune):
  Primary metric: pinball loss at τ=0.9, h=60min, averaged across 3 datasets.
  SUCCESS: ≥ 5% improvement over zero-shot baseline.
  PARTIAL: ≥ 2% improvement.
  FAILURE: < 2% improvement → triggers DECISION-015.

Expected runtime on RTX A4000: ~30–60 min total.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import BaseChronosPipeline

from _paths import P


warnings.filterwarnings('ignore')


# =============================== config ===============================

DATASETS = ['alibaba', 'bitbrains', 'bytedance']
HORIZONS_MIN = [10, 30, 60, 120]
QUANTILE_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]

# Chronos-2 context window: 2048 max; use 512 for speed + coverage
N_CONTEXT = 512
BATCH_SIZE = 256

CHRONOS2_MODEL = 'amazon/chronos-2'

# Interval per dataset (minutes per observation)
INTERVALS_MIN = {
    'alibaba':   5,
    'bitbrains': 5,
    'bytedance': 10,
}

# F3 pre-registered success criterion (LOCKED before any fine-tune)
F3_SUCCESS_PCT = 5.0
F3_PARTIAL_PCT = 2.0
F3_PRIMARY_TAU = 0.9
F3_PRIMARY_HORIZON_MIN = 60

# Output paths
OUT_DIR = Path(P['data'])
OUT_BASELINE_JSON = OUT_DIR / 'f3_zero_shot_baseline.json'
OUT_BASELINE_CSV = OUT_DIR / 'f3_zero_shot_baseline.csv'
OUT_DESIGN_MD = Path(P.get('root', '/workspace/kubernetes-cpu-ensemble-thesis/phase_f')) / 'f3_design.md'


# =============================== helpers ===============================

def load_dataset(dataset):
    """
    Load train + val + test parquets for a dataset, concatenate per container,
    return dict {container_id: 1D numpy array of cpu values sorted by time_stamp}.

    The Alibaba test split alone has ~340 obs per container (insufficient for
    N_CONTEXT=512), so we concatenate all three splits. For zero-shot Chronos-2
    inference there is no training leakage — the model wasn't trained on this
    data. train+val temporally precede test, so using them as context is
    standard time-series forecasting protocol.
    """
    root = Path(P.get('thesis', '/workspace/kubernetes-cpu-ensemble-thesis'))
    parts = []
    for split in ['train', 'val', 'test']:
        path = root / 'data' / 'processed' / dataset / f'{split}.parquet'
        if not path.exists():
            print(f'    warning: {path} missing')
            continue
        parts.append(pd.read_parquet(path))
    if not parts:
        raise FileNotFoundError(
            f'no train/val/test parquets at data/processed/{dataset}/'
        )
    df = pd.concat(parts, ignore_index=True)

    # Detect id column
    if 'container_id' in df.columns:
        id_col = 'container_id'
    elif 'vm_id' in df.columns:
        id_col = 'vm_id'
    else:
        raise ValueError(
            f'no id column in {dataset}; columns: {df.columns.tolist()}'
        )

    # Detect cpu column
    cpu_col = None
    for cand in ['cpu_util_percent', 'cpu_target', 'cpu', 'cpu_percent']:
        if cand in df.columns:
            cpu_col = cand
            break
    if cpu_col is None:
        raise ValueError(
            f'no cpu column in {dataset}; columns: {df.columns.tolist()}'
        )

    df = df[[id_col, 'time_stamp', cpu_col]].dropna()
    df = df.sort_values([id_col, 'time_stamp']).reset_index(drop=True)

    out = {}
    for cid, group in df.groupby(id_col):
        vals = group[cpu_col].values.astype(np.float32)
        out[str(cid)] = vals

    lengths = [len(v) for v in out.values()]
    print(f'    {dataset}: {len(out)} containers, concat(train+val+test), cpu_col={cpu_col}')
    print(f'      length stats: min={min(lengths)}, median={int(np.median(lengths))}, '
          f'mean={int(np.mean(lengths))}, max={max(lengths)}')
    return out


def pinball_loss(y_true, y_pred_q, tau):
    """
    Vectorised pinball loss at quantile tau.
      L_tau(y, q) = max(tau · (y - q), (tau - 1) · (y - q))
    Returns scalar mean.
    """
    diff = y_true - y_pred_q
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return float(np.mean(loss))


def build_contexts_targets(containers, horizon_obs, n_context):
    """
    For each container, build:
      context = last (n_context) observations before the final horizon_obs steps
      target  = the final horizon_obs steps

    Containers with length < n_context + horizon_obs are skipped.
    Returns:
      context_tensor: (n_valid, 1, n_context) float32
      target_array:   (n_valid, horizon_obs) float32
      valid_ids:      list of container_ids retained
      n_skipped:      int
    """
    ctx_list = []
    tgt_list = []
    valid_ids = []
    skipped = 0
    for cid, vals in containers.items():
        min_required = n_context + horizon_obs
        if len(vals) < min_required:
            skipped += 1
            continue
        ctx = vals[-(n_context + horizon_obs):-horizon_obs]
        tgt = vals[-horizon_obs:]
        ctx_list.append(ctx)
        tgt_list.append(tgt)
        valid_ids.append(cid)

    if not ctx_list:
        return None, None, [], skipped

    ctx_array = np.stack(ctx_list, axis=0)
    tgt_array = np.stack(tgt_list, axis=0)
    # Add n_variates=1 axis
    ctx_tensor = torch.tensor(ctx_array, dtype=torch.float32).unsqueeze(1)
    return ctx_tensor, tgt_array, valid_ids, skipped


def run_inference(pipe, ctx_tensor, horizon_obs, batch_size=BATCH_SIZE):
    """
    Send 3D context tensor through Chronos-2 in batches. Returns:
      quantile_array: (n_quantiles, n_series, horizon_obs) float32
    """
    n_series = ctx_tensor.shape[0]
    n_q = len(QUANTILE_LEVELS)
    all_q_per_series = []
    n_batches = (n_series + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch_start = b * batch_size
        batch_end = min((b + 1) * batch_size, n_series)
        batch_ctx = ctx_tensor[batch_start:batch_end]
        q_list, m_list = pipe.predict_quantiles(
            batch_ctx,
            prediction_length=horizon_obs,
            quantile_levels=QUANTILE_LEVELS,
        )
        # Each q_list element is (1, horizon_obs, n_quantiles)
        for qt in q_list:
            all_q_per_series.append(qt.detach().cpu().float().numpy())

    # Stack into (n_series, 1, horizon_obs, n_quantiles)
    stacked = np.stack(all_q_per_series, axis=0)
    # Squeeze n_variates axis (1)
    if stacked.shape[1] == 1:
        stacked = np.squeeze(stacked, axis=1)
    # Now stacked is (n_series, horizon_obs, n_quantiles)
    # Move quantile axis to position 0 → (n_quantiles, n_series, horizon_obs)
    quantile_array = np.moveaxis(stacked, -1, 0)
    return quantile_array


def write_design_doc():
    """Write the F3 pre-registration design doc."""
    OUT_DESIGN_MD.parent.mkdir(parents=True, exist_ok=True)
    content = f'''# F3 design — cost-asymmetric Chronos-2 fine-tune

**Date locked:** 2026-05-25 (Phase F3 Day 1, before any training)
**Status:** PRE-REGISTERED, locked

## Objective

LoRA fine-tune Amazon Chronos-2 (`{CHRONOS2_MODEL}`, ~120M parameters) with
asymmetric pinball loss to bias predictions toward over-provisioning, evaluated
against zero-shot Chronos-2 on the canonical thesis datasets and horizons.

## Quantile levels

τ ∈ {QUANTILE_LEVELS} — τ=0.5 is the median baseline; τ=0.7-0.95 are
over-provisioning quantiles aligned with OptScaler (Lu et al., VLDB 2024) and
the TimesFM quantile-head fine-tune in Cisana (arXiv:2410.11773, 2024-25).

## Datasets and horizons

- Alibaba 2018 (~4,876 containers, 5-min intervals)
- Bitbrains GWA-T-12 (142 VMs, 5-min intervals)
- ByteDance IaaS (93 containers, 10-min intervals)

Horizons: {HORIZONS_MIN} minutes. ByteDance has no h=10 due to 10-min sampling.

## Context configuration

- N_CONTEXT = {N_CONTEXT} observations per series
- Batch size = {BATCH_SIZE}
- Per container: context = last (N_CONTEXT + horizon_obs) obs minus final horizon_obs;
  target = final horizon_obs observations.
- Containers with length < N_CONTEXT + horizon_obs are skipped (no padding).

## Pre-registered F3 success criterion (LOCKED)

Primary metric: pinball loss at τ = {F3_PRIMARY_TAU}, h = {F3_PRIMARY_HORIZON_MIN} min,
averaged across the three datasets. Improvement over zero-shot Chronos-2.

- F3 SUCCESS: ≥ {F3_SUCCESS_PCT}% pinball-loss improvement.
- F3 PARTIAL: ≥ {F3_PARTIAL_PCT}% improvement.
- F3 FAILURE: < {F3_PARTIAL_PCT}% improvement → triggers DECISION-015.

## Method (for the LoRA fine-tune that follows this baseline)

- Backbone: Chronos-2 frozen
- Adaptation: LoRA (rank 8, α = 16, dropout 0.1) on the quantile head + last
  two transformer blocks
- Loss: weighted pinball at each τ ∈ {QUANTILE_LEVELS}
- Optimiser: AdamW, lr 1e-4, weight decay 0.01
- 20 epochs, early stopping on val pinball at τ = {F3_PRIMARY_TAU}
- Vast.ai RTX A4000 (16 GB VRAM)

## Files

- Baseline JSON: `phase_f/data/f3_zero_shot_baseline.json`
- Baseline CSV: `phase_f/data/f3_zero_shot_baseline.csv`
- This design doc: `phase_f/f3_design.md`
'''
    with open(OUT_DESIGN_MD, 'w') as f:
        f.write(content)
    print(f'    wrote {OUT_DESIGN_MD}')


# =============================== main ===============================

def main():
    print('=' * 78)
    print('f3_baseline_chronos2.py — Phase F3 zero-shot baseline')
    print('=' * 78)

    # ---- Environment ----
    print('\n[1] Environment...')
    print(f'    torch:    {torch.__version__}')
    print(f'    cuda:     {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'    device:   {torch.cuda.get_device_name(0)}')
        free, total = torch.cuda.mem_get_info()
        print(f'    GPU mem:  {free / 1e9:.1f} / {total / 1e9:.1f} GB free')

    # ---- Load Chronos-2 ----
    print(f'\n[2] Loading Chronos-2 ({CHRONOS2_MODEL})...')
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS2_MODEL,
        device_map='cuda',
        dtype=torch.bfloat16,
    )
    print(f'    loaded in {time.time() - t0:.1f}s, class={type(pipe).__name__}')

    # ---- Load datasets ----
    print('\n[3] Loading datasets...')
    containers_by_dataset = {}
    for ds in DATASETS:
        containers_by_dataset[ds] = load_dataset(ds)

    # ---- Inference loop ----
    print('\n[4] Running zero-shot inference + pinball loss per (dataset, horizon, tau)...')
    all_results = []
    flat_rows = []
    t_start = time.time()

    for ds in DATASETS:
        interval = INTERVALS_MIN[ds]
        containers = containers_by_dataset[ds]
        for h_min in HORIZONS_MIN:
            horizon_obs = h_min // interval
            if horizon_obs < 1:
                print(f'\n    skip {ds} h={h_min}: less than 1 obs at {interval}-min interval')
                continue

            print(f'\n    --- {ds} h={h_min} min ({horizon_obs} obs) ---')

            ctx_tensor, tgt_array, valid_ids, skipped = build_contexts_targets(
                containers, horizon_obs, N_CONTEXT,
            )
            if ctx_tensor is None:
                print(f'      no valid containers; skipped {skipped}')
                continue
            n_valid = ctx_tensor.shape[0]
            print(f'      valid containers: {n_valid} (skipped {skipped} for length < {N_CONTEXT + horizon_obs})')
            print(f'      context tensor: {tuple(ctx_tensor.shape)}')
            print(f'      target array:   {tuple(tgt_array.shape)}')

            t0 = time.time()
            quantile_array = run_inference(pipe, ctx_tensor, horizon_obs)
            print(f'      inference took {time.time() - t0:.1f}s, output {quantile_array.shape}')

            # Pinball loss per tau
            cell_result = {
                'dataset':            ds,
                'horizon_min':        h_min,
                'horizon_obs':        horizon_obs,
                'interval_min':       interval,
                'n_containers_valid': int(n_valid),
                'n_containers_skipped': int(skipped),
                'n_predictions':      int(n_valid * horizon_obs),
                'pinball_loss':       {},
            }
            for i, tau in enumerate(QUANTILE_LEVELS):
                loss = pinball_loss(tgt_array, quantile_array[i], tau)
                cell_result['pinball_loss'][f'{tau:.2f}'] = loss
                print(f'      tau={tau:.2f}: pinball={loss:.6f}')
                flat_rows.append({
                    'dataset':       ds,
                    'horizon_min':   h_min,
                    'horizon_obs':   horizon_obs,
                    'tau':           tau,
                    'pinball_loss':  loss,
                    'n_containers': n_valid,
                    'n_predictions': n_valid * horizon_obs,
                })
            all_results.append(cell_result)

    print(f'\n    total inference time: {time.time() - t_start:.0f}s')

    # ---- F3 pre-registered criterion ----
    print('\n[5] F3 pre-registered success criterion (locked before any fine-tune):')
    print(f'    PRIMARY: pinball loss at tau={F3_PRIMARY_TAU}, h={F3_PRIMARY_HORIZON_MIN} min')
    print(f'    averaged across {DATASETS}')
    print(f'    SUCCESS: >= {F3_SUCCESS_PCT}% improvement over zero-shot')
    print(f'    PARTIAL: >= {F3_PARTIAL_PCT}% improvement')
    print(f'    FAILURE: <  {F3_PARTIAL_PCT}% improvement -> triggers DECISION-015')

    # Compute primary baseline value for reference
    primary_losses = []
    for r in all_results:
        if r['horizon_min'] == F3_PRIMARY_HORIZON_MIN:
            tau_key = f'{F3_PRIMARY_TAU:.2f}'
            if tau_key in r['pinball_loss']:
                primary_losses.append(r['pinball_loss'][tau_key])
    if primary_losses:
        primary_baseline = float(np.mean(primary_losses))
        print(f'\n    BASELINE primary metric (mean across datasets at h={F3_PRIMARY_HORIZON_MIN}, tau={F3_PRIMARY_TAU}):')
        print(f'      {primary_baseline:.6f}')
        print(f'      F3 SUCCESS threshold (5% better): {primary_baseline * 0.95:.6f}')
        print(f'      F3 PARTIAL threshold (2% better): {primary_baseline * 0.98:.6f}')
    else:
        print(f'    WARNING: no primary metric available (no h={F3_PRIMARY_HORIZON_MIN} cells)')
        primary_baseline = None

    # ---- Save outputs ----
    print('\n[6] Saving outputs...')
    OUT_BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BASELINE_JSON, 'w') as f:
        json.dump({
            'metadata': {
                'date':       '2026-05-25',
                'sprint':     'Phase F3 Day 1',
                'model':      CHRONOS2_MODEL,
                'datasets':   DATASETS,
                'horizons':   HORIZONS_MIN,
                'quantiles':  QUANTILE_LEVELS,
                'n_context':  N_CONTEXT,
                'batch_size': BATCH_SIZE,
                'pre_registration': {
                    'primary_tau':       F3_PRIMARY_TAU,
                    'primary_horizon':   F3_PRIMARY_HORIZON_MIN,
                    'success_pct':       F3_SUCCESS_PCT,
                    'partial_pct':       F3_PARTIAL_PCT,
                    'baseline_primary':  primary_baseline,
                },
            },
            'results': all_results,
        }, f, indent=2, default=str)
    print(f'    wrote {OUT_BASELINE_JSON}')

    pd.DataFrame(flat_rows).to_csv(OUT_BASELINE_CSV, index=False)
    print(f'    wrote {OUT_BASELINE_CSV}')

    write_design_doc()

    print('\n' + '=' * 78)
    print('F3 Day 1 complete. Zero-shot baseline locked.')
    print('Next: write f3_chronos2_lora_finetune.py to train LoRA adapters with')
    print('asymmetric pinball loss against this baseline.')
    print('=' * 78)


if __name__ == '__main__':
    main()
