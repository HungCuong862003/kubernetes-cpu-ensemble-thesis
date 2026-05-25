"""
f3_probe_v4.py — final API probe with 3D input + list-of-tensors output.

Two fixes from v3:
  (a) Input tensor reshaped to 3D (n_series, n_variates=1, n_context).
      Chronos-2 supports multivariate; for univariate CPU forecasting, n_variates=1.
  (b) Output is tuple[list[torch.Tensor], list[torch.Tensor]] = (quantile_lists, mean_lists),
      one tensor per series. Production code iterates the list.

If this works, write the production f3 setup script.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

from chronos import BaseChronosPipeline


print('=' * 70)
print('f3_probe_v4.py — 3D input + list output')
print('=' * 70)

# ---- Load ----
print('\n[1] Loading Chronos-2...')
pipe = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-2',
    device_map='cuda',
    dtype=torch.bfloat16,
)
print(f'    pipeline class: {type(pipe).__name__}')

# ---- Build 3D context ----
print('\n[2] Building synthetic 3D context: (n_series=10, n_variates=1, n_context=100)...')
rng = np.random.default_rng(42)
n_series, n_context, n_future = 10, 100, 6

context_2d = np.array([
    50.0 + 20.0 * np.sin(np.arange(n_context) / 10.0 + i) + rng.normal(0, 2, n_context)
    for i in range(n_series)
])
future_2d = np.array([
    50.0 + 20.0 * np.sin(np.arange(n_context, n_context + n_future) / 10.0 + i)
    + rng.normal(0, 2, n_future)
    for i in range(n_series)
])

# Reshape to 3D: (n_series, n_variates=1, n_context)
context_3d = torch.tensor(context_2d, dtype=torch.float32).unsqueeze(1)
print(f'    context_3d shape: {context_3d.shape}')

# ---- predict_quantiles ----
print('\n[3] Calling predict_quantiles(context_3d, prediction_length=6, quantile_levels=[0.5,0.7,0.8,0.9,0.95])...')
quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95]

try:
    result = pipe.predict_quantiles(
        context_3d,
        prediction_length=n_future,
        quantile_levels=quantile_levels,
    )
except Exception as e:
    print(f'    FAILED: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---- Inspect output structure ----
print(f'\n[4] Output structure:')
print(f'    type(result):     {type(result).__name__}')
if isinstance(result, tuple):
    print(f'    len(result):      {len(result)}')
    for i, part in enumerate(result):
        print(f'    result[{i}] type: {type(part).__name__}')
        if isinstance(part, list):
            print(f'      list len:       {len(part)}')
            if len(part) > 0:
                first = part[0]
                if hasattr(first, 'shape'):
                    print(f'      first elem:     torch.Tensor, shape {tuple(first.shape)}, dtype {first.dtype}')
                else:
                    print(f'      first elem:     {type(first).__name__}')
                if len(part) > 1:
                    last = part[-1]
                    if hasattr(last, 'shape'):
                        print(f'      last elem:      torch.Tensor, shape {tuple(last.shape)}')

# Unpack
quantile_preds_list, mean_preds_list = result
print(f'\n    quantile_preds_list: {len(quantile_preds_list)} tensors')
print(f'    mean_preds_list:     {len(mean_preds_list)} tensors')

# ---- Compute pinball loss ----
print('\n[5] Pinball loss per quantile (averaged across 10 series × 6 timesteps):')

# Each quantile tensor in the list is (n_variates, n_future, n_quantiles)
# OR (n_future, n_quantiles) - we need to figure out
qp0 = quantile_preds_list[0]
print(f'    inspecting first series quantile tensor: shape {tuple(qp0.shape)}')

# Stack into (n_series, ...) for analysis
all_q = torch.stack([q for q in quantile_preds_list], dim=0).detach().cpu().float().numpy()
all_m = torch.stack([m for m in mean_preds_list], dim=0).detach().cpu().float().numpy()
print(f'    stacked quantile tensor: {all_q.shape}')
print(f'    stacked mean tensor:     {all_m.shape}')

# Detect axes: which is n_quantiles=5?
n_q = len(quantile_levels)
q_axis = None
for ax, size in enumerate(all_q.shape):
    if size == n_q:
        q_axis = ax
        break
print(f'    quantile axis in stacked tensor: {q_axis}')

# Move quantile axis to position 0: (n_quantiles, n_series, ...)
all_q_reord = np.moveaxis(all_q, q_axis, 0)
# If there's a singleton n_variates axis, squeeze it
if all_q_reord.ndim == 4:
    print(f'    squeezing n_variates axis...')
    # Could be at axis 1 or 2 now
    for ax in range(1, all_q_reord.ndim):
        if all_q_reord.shape[ax] == 1:
            all_q_reord = np.squeeze(all_q_reord, axis=ax)
            break
print(f'    reordered shape (n_q, n_series, n_future): {all_q_reord.shape}')

for i, tau in enumerate(quantile_levels):
    diff = future_2d - all_q_reord[i]
    pinball = np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()
    print(f'      tau={tau:.2f}: pinball={pinball:7.4f}')

# ---- Sanity-check sample predictions ----
print(f'\n[6] First series, all 6 future timesteps, predicted vs true:')
print('       t |  true   |  q50    q70    q80    q90    q95')
for t in range(n_future):
    row = f'      {t}  | {future_2d[0, t]:7.2f} | '
    row += ' '.join(f'{all_q_reord[i, 0, t]:6.2f}' for i, _ in enumerate(quantile_levels))
    print(row)

print('\n' + '=' * 70)
print('PROBE v4 SUCCESS — production-ready calling convention locked.')
print()
print('Calling convention for production:')
print('  # input shape: (n_series, 1, n_context_obs), float32')
print('  quantile_preds_list, mean_preds_list = pipe.predict_quantiles(')
print('      context_3d,                       # 3D tensor, dtype=float32')
print('      prediction_length=H,')
print('      quantile_levels=[0.5, 0.7, 0.8, 0.9, 0.95],')
print('  )')
print('  # Each element of quantile_preds_list is a per-series tensor.')
print('  # Stack and reorder to (n_quantiles, n_series, n_future) for pinball loss.')
print('=' * 70)
