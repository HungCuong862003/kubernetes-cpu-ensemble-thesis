"""
f3_probe_v3.py — fixed kwarg name + signature inspection.

Difference from v2: passes `inputs=context` (or positional) instead of
`context=context`. The Chronos2Pipeline API uses `inputs` as the first
positional argument, not `context`.

Also explicitly inspects predict() and predict_quantiles() signatures so we
know what the production code needs.
"""

import inspect
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

from chronos import BaseChronosPipeline


print('=' * 70)
print('f3_probe_v3.py — Chronos-2 API call with correct kwarg name')
print('=' * 70)

# ---- Load Chronos-2 ----
print('\n[1] Loading Chronos-2...')
pipe = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-2',
    device_map='cuda',
    dtype=torch.bfloat16,  # 'dtype' is the modern arg name
)
print(f'    pipeline class: {type(pipe).__name__}')


# ---- Inspect signatures ----
print('\n[2] Method signatures (so we know what production code needs):')

method_names = ['predict', 'predict_quantiles', 'predict_samples', 'embed']
for m in method_names:
    if hasattr(pipe, m):
        try:
            sig = inspect.signature(getattr(pipe, m))
            print(f'    {m}{sig}')
        except (ValueError, TypeError):
            print(f'    {m}: (signature unavailable)')
    else:
        print(f'    {m}: NOT AVAILABLE')


# ---- Tiny synthetic data ----
print('\n[3] Building synthetic context: 10 series, 100 timesteps each...')
rng = np.random.default_rng(42)
n_series = 10
n_context = 100
n_future = 6

context_np = np.array([
    50.0 + 20.0 * np.sin(np.arange(n_context) / 10.0 + i) + rng.normal(0, 2, n_context)
    for i in range(n_series)
])
future_np = np.array([
    50.0 + 20.0 * np.sin(np.arange(n_context, n_context + n_future) / 10.0 + i) + rng.normal(0, 2, n_future)
    for i in range(n_series)
])
context = torch.tensor(context_np, dtype=torch.float32)
print(f'    context tensor: {context.shape}, dtype={context.dtype}')


# ---- predict_quantiles() — the primary target API for pinball loss ----
print('\n[4] Trying predict_quantiles(context, prediction_length=6, quantile_levels=...)...')
try:
    quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    result = pipe.predict_quantiles(
        context,
        prediction_length=n_future,
        quantile_levels=quantile_levels,
    )
    print(f'    result type: {type(result).__name__}')

    if isinstance(result, tuple):
        print(f'    tuple of length {len(result)}:')
        for i, r in enumerate(result):
            if hasattr(r, 'shape'):
                print(f'      [{i}] shape: {tuple(r.shape)}, dtype: {r.dtype}')
            else:
                print(f'      [{i}] type: {type(r).__name__}')
        # First element is usually quantile predictions
        quantile_preds = result[0]
        mean_preds = result[1] if len(result) > 1 else None
    else:
        quantile_preds = result
        mean_preds = None
        print(f'    shape: {tuple(quantile_preds.shape)}, dtype: {quantile_preds.dtype}')

    qp_np = quantile_preds.detach().cpu().float().numpy()
    print(f'\n    quantile_preds (numpy): {qp_np.shape}')

    # Layout detection: which axis is quantiles?
    n_q = len(quantile_levels)
    q_axis = None
    for ax, size in enumerate(qp_np.shape):
        if size == n_q:
            q_axis = ax
            break
    print(f'    quantile axis: {q_axis}')

    # Reorder to (n_quantiles, n_series, n_future)
    if q_axis == 2:
        # (series, future, quantiles) → (quantiles, series, future)
        qp_ordered = qp_np.transpose(2, 0, 1)
    elif q_axis == 1:
        # (series, quantiles, future) → (quantiles, series, future)
        qp_ordered = qp_np.transpose(1, 0, 2)
    elif q_axis == 0:
        qp_ordered = qp_np
    else:
        print(f'    WARNING: could not determine quantile axis; using as-is')
        qp_ordered = qp_np

    print(f'    qp_ordered shape (quantiles, series, future): {qp_ordered.shape}')

    # ---- Pinball loss against known future ----
    print('\n[5] Pinball loss per quantile (averaged across 10 series × 6 timesteps):')
    for i, tau in enumerate(quantile_levels):
        diff = future_np - qp_ordered[i]
        pinball = np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()
        print(f'      tau={tau:.2f}: pinball={pinball:7.4f}')

    print('\n[6] Sample predictions vs truth (first series, all 6 future timesteps):')
    print('       t |  true   |  q50    q70    q80    q90    q95')
    for t in range(n_future):
        row = f'      {t}  | {future_np[0, t]:7.2f} | '
        row += ' '.join(f'{qp_ordered[i, 0, t]:6.2f}' for i, _ in enumerate(quantile_levels))
        print(row)

    print('\n' + '=' * 70)
    print('PROBE v3 SUCCESS — production-ready API confirmed.')
    print('Calling convention:')
    print('  pipe.predict_quantiles(')
    print('      context_tensor,                  # positional, shape (n_series, n_context)')
    print('      prediction_length=H,')
    print('      quantile_levels=[0.5, 0.7, 0.8, 0.9, 0.95],')
    print('  )')
    print('=' * 70)

except Exception as e:
    print(f'    FAILED: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
