"""
f3_probe_v2.py — direct chronos-forecasting library probe.

Bypasses AutoGluon entirely. Uses the chronos-forecasting library
(already installed as a dependency of autogluon.timeseries) to load Chronos-2
directly and run zero-shot inference on tiny synthetic data.

Why direct over AutoGluon: AutoGluon's wrapper for Chronos-2 was breaking on
the torchvision/torch ABI mismatch. The chronos-forecasting library
(BaseChronosPipeline) is the same underlying object AutoGluon wraps, so we
lose nothing by going direct.

Goals:
  1. Confirm chronos.BaseChronosPipeline loads after torchvision uninstall
  2. Verify Chronos-2 weights download and load on GPU
  3. Run zero-shot inference, check output shape
  4. Compute quantile predictions
  5. Compute pinball loss on a tiny known target
"""

import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

print('=' * 70)
print('f3_probe_v2.py — direct chronos-forecasting probe')
print('=' * 70)

# ---- [1] Verify torch + CUDA ----
print('\n[1] Torch / CUDA:')
print(f'    torch:        {torch.__version__}')
print(f'    cuda avail:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    device:       {torch.cuda.get_device_name(0)}')
    print(f'    bfloat16 ok:  {torch.cuda.is_bf16_supported()}')

# ---- [2] Verify transformers + chronos import after torchvision removal ----
print('\n[2] Importing transformers and chronos...')
try:
    import transformers
    print(f'    transformers: {transformers.__version__}')
except Exception as e:
    print(f'    transformers IMPORT FAILED: {e}')
    print('    fix: pip uninstall -y torchvision')
    sys.exit(1)

try:
    import chronos
    print(f'    chronos:      {chronos.__version__ if hasattr(chronos, "__version__") else "(no __version__)"}')
    from chronos import BaseChronosPipeline
    print(f'    BaseChronosPipeline imports OK')
except Exception as e:
    print(f'    chronos IMPORT FAILED: {e}')
    sys.exit(1)

# ---- [3] Load Chronos-2 ----
print('\n[3] Loading Chronos-2 from HuggingFace (may download ~500 MB on first call)...')
t0 = time.time()
try:
    pipe = BaseChronosPipeline.from_pretrained(
        'amazon/chronos-2',
        device_map='cuda',
        torch_dtype=torch.bfloat16,
    )
    print(f'    loaded in {time.time() - t0:.0f}s')
    print(f'    pipeline class: {type(pipe).__name__}')
    if hasattr(pipe, 'model'):
        print(f'    underlying model: {type(pipe.model).__name__}')
except Exception as e:
    print(f'    LOAD FAILED: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---- [4] Tiny synthetic context ----
print('\n[4] Building synthetic context: 10 series, 100 timesteps each...')
rng = np.random.default_rng(42)
n_series = 10
n_context = 100
context_np = np.array([
    50.0 + 20.0 * np.sin(np.arange(n_context) / 10.0 + i) + rng.normal(0, 2, n_context)
    for i in range(n_series)
])
# Also build the "future" we want to predict (next 6 steps) for pinball loss validation
n_future = 6
future_np = np.array([
    50.0 + 20.0 * np.sin(np.arange(n_context, n_context + n_future) / 10.0 + i) + rng.normal(0, 2, n_future)
    for i in range(n_series)
])
context = torch.tensor(context_np, dtype=torch.float32)
print(f'    context shape: {context.shape}')
print(f'    future shape:  {future_np.shape}')

# ---- [5] Zero-shot prediction ----
print('\n[5] Running zero-shot inference, prediction_length=6, num_samples=20...')
t0 = time.time()
try:
    # Chronos-2 API may differ between predict() and predict_quantiles()
    # First try the simpler predict() with num_samples
    try:
        preds = pipe.predict(
            context=context,
            prediction_length=n_future,
            num_samples=20,
        )
        print(f'    predict() returned shape: {preds.shape}')
        print(f'    predict() dtype: {preds.dtype}')
        used_api = 'predict'
    except (TypeError, AttributeError) as e:
        print(f'    predict() with num_samples failed: {e}')
        # Fallback: predict_quantiles
        if hasattr(pipe, 'predict_quantiles'):
            print(f'    trying predict_quantiles()...')
            quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
            quantile_preds, mean_preds = pipe.predict_quantiles(
                context=context,
                prediction_length=n_future,
                quantile_levels=quantile_levels,
            )
            print(f'    predict_quantiles() returned:')
            print(f'      quantiles shape: {quantile_preds.shape}')
            print(f'      mean shape:      {mean_preds.shape}')
            preds = quantile_preds
            used_api = 'predict_quantiles'
        else:
            raise
    print(f'    inference took {time.time() - t0:.1f}s')
except Exception as e:
    print(f'    PREDICT FAILED: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---- [6] Compute quantiles + pinball loss ----
print('\n[6] Computing quantiles + pinball loss against known future...')
quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
preds_np = preds.detach().cpu().float().numpy()

if used_api == 'predict':
    # preds shape: (n_series, num_samples, n_future) → compute quantiles
    q_preds = np.quantile(preds_np, quantile_levels, axis=1)
    # q_preds shape: (n_quantiles, n_series, n_future)
elif used_api == 'predict_quantiles':
    # preds_np already (n_series, n_future, n_quantiles) or similar; reshape if needed
    # Check shape and transpose if needed
    if preds_np.ndim == 3 and preds_np.shape[2] == len(quantile_levels):
        # (n_series, n_future, n_quantiles) → (n_quantiles, n_series, n_future)
        q_preds = preds_np.transpose(2, 0, 1)
    else:
        q_preds = preds_np
print(f'    q_preds shape (n_quantiles, n_series, n_future): {q_preds.shape}')

print('\n    pinball loss per quantile (averaged over all 10 series × 6 timesteps):')
for i, tau in enumerate(quantile_levels):
    diff = future_np - q_preds[i]
    pinball = np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()
    print(f'      tau={tau:.2f}: pinball={pinball:.4f}')

print(f'\n    sample predictions (first series, first 3 timesteps):')
for t in range(3):
    print(f'      t={t} | true={future_np[0, t]:6.2f} | ' + ' | '.join(
        [f'q{q*100:.0f}={q_preds[i, 0, t]:6.2f}' for i, q in enumerate(quantile_levels)]
    ))

print('\n' + '=' * 70)
print('PROBE v2 SUCCESS — chronos-forecasting direct API works.')
print('Next: rewrite f3_chronos2_finetune_setup.py to use BaseChronosPipeline directly.')
print('=' * 70)
