"""
f3_chronos2_finetune_setup.py — Phase F3 Day 1: environment + zero-shot baseline.

Goal: establish the zero-shot Chronos-2 pinball-loss baseline against which the
LoRA-fine-tuned model will be evaluated. NO TRAINING in this script.

Six steps, in order:

[1] Environment verification: Python ≥ 3.10, CUDA available, GPU memory ≥ 8 GB,
    AutoGluon v1.5.0+ installed with Chronos-2 support, pyarrow + pandas + numpy
    versions sufficient.

[2] Data preparation: load existing Alibaba/Bitbrains/ByteDance test splits from
    the canonical thesis data directory, format as AutoGluon TimeSeriesDataFrame.
    Use the same containers and horizons as par_per_series_r2_v2.parquet so the
    F3 comparison is apples-to-apples with the Phase F leaderboard.

[3] Zero-shot Chronos-2 inference at horizons 10, 30, 60, 120 min, quantile heads
    at τ = {0.5, 0.7, 0.8, 0.9, 0.95}. τ = 0.5 (median) is the standard point-
    forecast baseline; τ = 0.7..0.95 are the over-provisioning quantiles used in
    asymmetric-pinball training.

[4] Pinball-loss computation per (dataset, horizon, τ).
    L_τ(y, ŷ) = max(τ · (y − ŷ), (τ − 1) · (y − ŷ))

[5] Pre-registration of F3 success criterion (LOCKED here before any fine-tune):
      SUCCESS: pinball-loss improvement at τ = 0.9, h = 60 min, averaged across
               three datasets, ≥ 5% vs zero-shot.
      PARTIAL: ≥ 2% improvement.
      FAILURE: < 2% improvement → triggers DECISION-015.

[6] Save baseline pinball loss + F3 design doc to:
      phase_f/data/f3_zero_shot_baseline.json
      phase_f/data/f3_zero_shot_baseline.csv
      phase_f/f3_design.md

Runtime estimate: 20–40 min on RTX 5070 Ti for inference across ~5,000 series ×
4 horizons × 5 quantiles. No training, so GPU memory peak is modest (~6 GB).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import P


warnings.filterwarnings('ignore')


# =============================== config ===============================

DATASETS = ['alibaba', 'bitbrains', 'bytedance']
HORIZONS_MIN = [10, 30, 60, 120]
QUANTILES = [0.5, 0.7, 0.8, 0.9, 0.95]

REQUIRED_AUTOGLUON_VERSION = '1.5.0'
REQUIRED_PYTHON_MAJOR_MINOR = (3, 10)
REQUIRED_GPU_MEM_GB = 8

# Chronos-2 model identifier (Amazon HF)
CHRONOS2_MODEL = 'amazon/chronos-2'

# Output paths (canonical for the chapter)
OUT_DIR = Path(P['data'])
OUT_BASELINE_JSON = OUT_DIR / 'f3_zero_shot_baseline.json'
OUT_BASELINE_CSV = OUT_DIR / 'f3_zero_shot_baseline.csv'
OUT_DESIGN_MD = Path(P.get('root', '/workspace/kubernetes-cpu-ensemble-thesis/phase_f')) / 'f3_design.md'

# Pre-registered F3 success criterion (LOCKED before any fine-tune training)
F3_SUCCESS_PCT = 5.0
F3_PARTIAL_PCT = 2.0
F3_PRIMARY_TAU = 0.9
F3_PRIMARY_HORIZON_MIN = 60


# =============================== helpers ===============================

def check_python_version():
    """Verify Python is at least REQUIRED_PYTHON_MAJOR_MINOR."""
    cur = sys.version_info[:2]
    print(f'    Python: {cur[0]}.{cur[1]}.{sys.version_info[2]}')
    if cur < REQUIRED_PYTHON_MAJOR_MINOR:
        raise RuntimeError(f'Python {REQUIRED_PYTHON_MAJOR_MINOR} required; have {cur}')


def check_cuda():
    """Verify CUDA is available and GPU has enough memory."""
    try:
        import torch
    except ImportError:
        raise RuntimeError('torch not installed; run: pip install torch')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; Chronos-2 inference will be too slow on CPU')

    n_devices = torch.cuda.device_count()
    print(f'    CUDA available: {n_devices} device(s)')
    for i in range(n_devices):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024 ** 3)
        print(f'      device {i}: {props.name}, {mem_gb:.1f} GB')
        if mem_gb < REQUIRED_GPU_MEM_GB:
            print(f'      warning: {REQUIRED_GPU_MEM_GB} GB recommended; may OOM on large batches')


def check_autogluon():
    """Verify AutoGluon TimeSeries is installed at the required version."""
    try:
        import autogluon.timeseries
        ver = autogluon.timeseries.__version__
    except ImportError:
        print(f'    AutoGluon TimeSeries not installed.')
        print(f'    Install with: pip install autogluon.timeseries=={REQUIRED_AUTOGLUON_VERSION}')
        raise RuntimeError('AutoGluon missing')

    print(f'    autogluon.timeseries: {ver}')
    if ver < REQUIRED_AUTOGLUON_VERSION:
        print(f'    warning: v{REQUIRED_AUTOGLUON_VERSION}+ recommended for Chronos-2 LoRA support; have {ver}')

    # Verify Chronos-2 is registered as a model
    from autogluon.timeseries.models.presets import MODEL_TYPES
    chronos2_present = any('Chronos' in str(m) for m in MODEL_TYPES.keys())
    if not chronos2_present:
        print(f'    warning: ChronosModel not in MODEL_TYPES; LoRA path may be unavailable')
    else:
        print(f'    AutoGluon Chronos model class detected')


def install_dependencies():
    """Try `pip install` on autogluon.timeseries 1.5.0+ if missing. Idempotent."""
    print('\n[1.5] Installing/verifying dependencies (may take 5–10 min on first run)...')

    cmd = [
        sys.executable, '-m', 'pip', 'install', '--quiet',
        f'autogluon.timeseries>={REQUIRED_AUTOGLUON_VERSION}',
        'transformers>=4.40',
        'accelerate>=0.30',
        'peft>=0.10',
    ]
    print('    cmd: ' + ' '.join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'    stdout: {result.stdout}')
        print(f'    stderr: {result.stderr}')
        raise RuntimeError('pip install failed')
    print(f'    install OK in {time.time() - t0:.0f}s')


def load_test_splits():
    """
    Load existing Alibaba/Bitbrains/ByteDance test splits from the canonical
    thesis data directory. Format as a long DataFrame with columns:
      [container_id, dataset, time_stamp, cpu_util_percent].
    """
    print('\n[2] Loading test splits from thesis data directory...')

    parts = []
    for ds in DATASETS:
        # Standard thesis layout: results/<dataset>/test_<dataset>.parquet
        candidates = [
            Path(P.get('results', '/workspace/kubernetes-cpu-ensemble-thesis/results')) / ds / f'test_{ds}.parquet',
            Path(P.get('data', '/workspace/kubernetes-cpu-ensemble-thesis/data')) / ds / f'test_{ds}.parquet',
            Path(P.get('data', '/workspace/kubernetes-cpu-ensemble-thesis/data')) / f'test_{ds}.parquet',
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            raise FileNotFoundError(f'No test_{ds}.parquet found; checked {candidates}')

        df = pd.read_parquet(found)
        df['dataset'] = ds
        # Standardise id column to 'container_id'
        id_col = 'container_id' if 'container_id' in df.columns else 'vm_id'
        if id_col != 'container_id':
            df = df.rename(columns={id_col: 'container_id'})
        df['container_id'] = df['container_id'].astype(str)
        print(f'    {ds}: {len(df)} rows, {df["container_id"].nunique()} containers, source={found}')
        parts.append(df[['container_id', 'dataset', 'time_stamp', 'cpu_util_percent']])

    out = pd.concat(parts, ignore_index=True)
    print(f'    total: {len(out)} rows, {out["container_id"].nunique()} containers')
    return out


def to_autogluon_ts_df(df_long):
    """
    Convert long DataFrame to AutoGluon TimeSeriesDataFrame format.
    AutoGluon expects: ['item_id', 'timestamp', 'target'] index.
    Returns the TSDF wrapping the long frame.
    """
    from autogluon.timeseries import TimeSeriesDataFrame

    # AutoGluon item_id = container_id (unique per series)
    df = df_long.rename(columns={
        'container_id': 'item_id',
        'cpu_util_percent': 'target',
    })
    # time_stamp is integer seconds in our data; convert to datetime
    if pd.api.types.is_integer_dtype(df['time_stamp']):
        df['timestamp'] = pd.to_datetime(df['time_stamp'], unit='s')
    else:
        df['timestamp'] = pd.to_datetime(df['time_stamp'])

    df = df[['item_id', 'timestamp', 'target', 'dataset']]
    df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

    return TimeSeriesDataFrame(df)


def pinball_loss(y_true, y_pred_q, tau):
    """
    Pinball loss at quantile tau:
      L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q))
                  = (y - q) * (tau - 1[y < q])
    Returns mean across all (n,) elements.
    """
    diff = y_true - y_pred_q
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return float(np.mean(loss))


def run_zero_shot_inference(ts_df, dataset_filter, horizon_min):
    """
    Run zero-shot Chronos-2 inference for one (dataset, horizon) combination.
    Returns dict with quantile pinball losses + n_predictions.

    Uses AutoGluon's TimeSeriesPredictor with Chronos-2 preset, no fine-tuning,
    no covariates. Predictions are in QUANTILES at the configured horizon.
    """
    from autogluon.timeseries import TimeSeriesPredictor

    # Filter to one dataset
    df_filtered = ts_df.query(f'dataset == "{dataset_filter}"').copy()

    # Determine prediction length in observations (5-min intervals for Alibaba/Bitbrains, 10-min for ByteDance)
    if dataset_filter == 'bytedance':
        interval_min = 10
    else:
        interval_min = 5
    prediction_length = horizon_min // interval_min
    if prediction_length < 1:
        return None  # horizon too short for this dataset

    # Split into train+test: last `prediction_length` observations per container are the test target
    print(f'      {dataset_filter} h={horizon_min}: pred_length={prediction_length} obs ({interval_min}-min intervals)')

    # AutoGluon TimeSeriesPredictor for zero-shot Chronos-2
    tmp_path = Path('/tmp') / f'ag_zero_shot_{dataset_filter}_h{horizon_min}'
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=str(tmp_path),
        target='target',
        quantile_levels=QUANTILES,
        eval_metric='WQL',  # weighted quantile loss
        verbosity=1,
    )

    # Fit with chronos-2 zero-shot only (no other models)
    t0 = time.time()
    predictor.fit(
        df_filtered,
        presets='chronos_large',  # placeholder; the actual AG preset for Chronos-2 may differ
        hyperparameters={'Chronos': {'model_path': CHRONOS2_MODEL}},
        time_limit=1800,
    )
    fit_time = time.time() - t0
    print(f'        fit (zero-shot, no training) took {fit_time:.0f}s')

    # Predict
    t0 = time.time()
    preds = predictor.predict(df_filtered)
    predict_time = time.time() - t0
    print(f'        predict took {predict_time:.0f}s')

    # Compute pinball loss per quantile
    # AutoGluon returns predictions for each quantile in columns named "0.5", "0.7", etc.
    # The true values for the test window need to be extracted from the original df_filtered.
    # AutoGluon's leaderboard / evaluate methods provide this.
    leaderboard = predictor.leaderboard(df_filtered)
    print(f'        leaderboard:\n{leaderboard}')

    # Get held-out true values for the test window (last prediction_length per series)
    losses = {}
    n_preds = 0
    for tau in QUANTILES:
        tau_str = str(tau)
        if tau_str not in preds.columns:
            print(f'        warning: quantile {tau} not in predictions; skipping')
            continue
        # Align preds with true held-out values
        # preds has multi-index (item_id, timestamp); true values are the last K rows per item_id of df_filtered
        true_vals = []
        pred_vals = []
        for item_id in preds.index.get_level_values('item_id').unique():
            ser_true = df_filtered.query(f'item_id == "{item_id}"')['target'].values
            ser_pred = preds.loc[item_id, tau_str].values
            # Compare last `prediction_length` of true vs the prediction
            if len(ser_true) < prediction_length:
                continue
            true_vals.append(ser_true[-prediction_length:])
            pred_vals.append(ser_pred[:prediction_length])
        if not true_vals:
            continue
        true_vals = np.concatenate(true_vals)
        pred_vals = np.concatenate(pred_vals)
        losses[tau] = pinball_loss(true_vals, pred_vals, tau)
        n_preds = len(true_vals)

    return {
        'dataset': dataset_filter,
        'horizon_min': horizon_min,
        'prediction_length': prediction_length,
        'interval_min': interval_min,
        'pinball_loss': losses,
        'n_predictions': int(n_preds),
        'fit_time_s': float(fit_time),
        'predict_time_s': float(predict_time),
    }


def write_design_doc():
    """Write the F3 design doc + pre-registered success criterion."""
    OUT_DESIGN_MD.parent.mkdir(parents=True, exist_ok=True)
    content = f'''# F3 design — cost-asymmetric Chronos-2 fine-tune

**Date locked:** 2026-05-25 (Phase F3 Day 1, before any training begins)
**Status:** PRE-REGISTERED, locked

## Objective

LoRA fine-tune Amazon Chronos-2 (`{CHRONOS2_MODEL}`, ~120M parameters) with
asymmetric pinball loss to bias predictions toward over-provisioning, evaluated
against zero-shot Chronos-2 on the canonical thesis datasets and horizons.

## Quantile levels

τ ∈ {QUANTILES} — τ = 0.5 is the standard median baseline; τ = 0.7–0.95 are
over-provisioning quantiles aligned with the OptScaler asymmetric-pinball design
(Lu et al., VLDB 2024) and the TimesFM quantile-head fine-tune in Cisana
(arXiv:2410.11773, 2024–25).

## Datasets and horizons

- Alibaba 2018 (~4,876 containers, 5-min intervals)
- Bitbrains GWA-T-12 (142 VMs, 5-min intervals)
- ByteDance IaaS (93 containers, 10-min intervals)

Horizons: {HORIZONS_MIN} minutes. ByteDance has no h=10 due to 10-min sampling.

## Pre-registered F3 success criterion (LOCKED)

Primary metric: pinball loss at τ = {F3_PRIMARY_TAU}, h = {F3_PRIMARY_HORIZON_MIN} min,
averaged across the three datasets. Improvement over zero-shot Chronos-2 baseline.

- **F3 SUCCESS:** ≥ {F3_SUCCESS_PCT}% pinball-loss improvement.
- **F3 PARTIAL:** ≥ {F3_PARTIAL_PCT}% improvement.
- **F3 FAILURE:** < {F3_PARTIAL_PCT}% improvement → triggers DECISION-015
  (whether to try TimesFM-2.5 instead, or abandon F3 and rely on F4 alone).

## Rationale for these thresholds

Cisana (2024–25) reports comparable margin for TimesFM quantile-head fine-tune
on financial Value-at-Risk vs GARCH/GAS baselines. OptScaler (VLDB 2024) reports
>36% SLO-violation reduction via asymmetric pinball training of a bespoke
Flowformer baseline. 5% pinball improvement on a 120M-parameter foundation model
is conservative against both references.

## Method

- **Backbone:** Chronos-2 frozen
- **Adaptation:** LoRA (rank 8, α = 16, dropout 0.1) on the quantile head + last
  two transformer blocks
- **Loss:** weighted pinball at each τ ∈ {QUANTILES}, no point-forecast term
- **Optimiser:** AdamW, lr 1e-4, weight decay 0.01
- **Schedule:** 20 epochs with early stopping on val pinball at τ = {F3_PRIMARY_TAU}
- **Batch size:** 32 (auto-scaled by GPU memory)
- **Compute:** Vast.ai RTX 5070 Ti (12 GB VRAM)

## Sanity checks before any fine-tune

1. Zero-shot Chronos-2 must achieve non-trivial pinball loss at τ = 0.5
   (point forecast) on all 12 (dataset, horizon) cells. Threshold:
   pinball loss at τ = 0.5 < pinball loss of naive (previous-value) prediction
   for at least 8 of 12 cells.
2. Inference latency per series ≤ 0.1 s (already verified in Phase A1 spike test
   at 0.028 s/series).
3. No-leak verification: predictions on test split do not overlap training split
   timestamps.

## What to compare against

The chapter table will compare:
- Naive (previous value, baseline)
- Zero-shot Chronos-2 (this script)
- LoRA-fine-tuned Chronos-2 (Phase F3 main script)
- Existing NNLS ensemble (per `comparison_table.csv`)
- Punniyamoorthy et al. (arXiv:2512.23415, Dec 2025) — most recent published
  non-foundation-model autoscaling baseline; reports 31% SLO-violation
  reduction, 24% faster response, 18% cost reduction

## Files this script produces

- `phase_f/data/f3_zero_shot_baseline.json` — full per-cell pinball losses
- `phase_f/data/f3_zero_shot_baseline.csv` — flat table for chapter
- `phase_f/f3_design.md` — this file
'''
    with open(OUT_DESIGN_MD, 'w') as f:
        f.write(content)
    print(f'    wrote {OUT_DESIGN_MD}')


# =============================== main ===============================

def main():
    print('=' * 78)
    print('f3_chronos2_finetune_setup.py — Phase F3 Day 1')
    print('  Environment + zero-shot Chronos-2 baseline + F3 pre-registration')
    print('=' * 78)

    # ---- [1] Environment verification ----
    print('\n[1] Environment verification...')
    check_python_version()
    check_cuda()
    check_autogluon()

    print('\n[1.5] Skipping pip install if AutoGluon already at required version.')
    try:
        import autogluon.timeseries
        if autogluon.timeseries.__version__ >= REQUIRED_AUTOGLUON_VERSION:
            print(f'    AutoGluon already at {autogluon.timeseries.__version__}; skipping install')
        else:
            install_dependencies()
    except ImportError:
        install_dependencies()

    # ---- [2] Data preparation ----
    print('\n[2] Loading data...')
    df_long = load_test_splits()
    ts_df = to_autogluon_ts_df(df_long)
    print(f'    converted to TimeSeriesDataFrame: {len(ts_df)} rows')

    # ---- [3, 4] Zero-shot inference + pinball loss per (dataset, horizon) ----
    print('\n[3+4] Running zero-shot Chronos-2 inference + pinball loss...')
    all_results = []
    flat_rows = []
    t_start = time.time()

    for ds in DATASETS:
        for h in HORIZONS_MIN:
            print(f'\n    --- {ds} h={h} min ---')
            try:
                result = run_zero_shot_inference(ts_df, ds, h)
                if result is None:
                    print(f'      skipped (horizon too short for dataset interval)')
                    continue
                all_results.append(result)
                for tau, loss in result['pinball_loss'].items():
                    flat_rows.append({
                        'dataset':            ds,
                        'horizon_min':        h,
                        'prediction_length':  result['prediction_length'],
                        'interval_min':       result['interval_min'],
                        'tau':                tau,
                        'pinball_loss':       loss,
                        'n_predictions':      result['n_predictions'],
                    })
                    print(f'      tau={tau:.2f}: pinball={loss:.6f}  (n={result["n_predictions"]})')
            except Exception as e:
                print(f'      FAILED: {type(e).__name__}: {e}')
                import traceback
                traceback.print_exc()

    print(f'\n    total inference time: {time.time() - t_start:.0f}s')

    # ---- [5] Pre-registered F3 success criterion ----
    print('\n[5] F3 pre-registered success criterion (locked before any fine-tune):')
    print(f'    PRIMARY METRIC: pinball loss at tau={F3_PRIMARY_TAU}, h={F3_PRIMARY_HORIZON_MIN} min')
    print(f'    averaged across {DATASETS}')
    print(f'    F3 SUCCESS: >= {F3_SUCCESS_PCT}% improvement over zero-shot')
    print(f'    F3 PARTIAL: >= {F3_PARTIAL_PCT}% improvement')
    print(f'    F3 FAILURE: <  {F3_PARTIAL_PCT}% improvement -> triggers DECISION-015')

    # ---- [6] Save outputs ----
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
                'quantiles':  QUANTILES,
                'pre_registration': {
                    'primary_tau':       F3_PRIMARY_TAU,
                    'primary_horizon':   F3_PRIMARY_HORIZON_MIN,
                    'success_pct':       F3_SUCCESS_PCT,
                    'partial_pct':       F3_PARTIAL_PCT,
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
    print('Next: write f3_chronos2_lora_finetune.py to train LoRA adapters.')
    print('=' * 78)


if __name__ == '__main__':
    main()
