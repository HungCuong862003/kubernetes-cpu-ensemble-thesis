"""
f3_probe.py — minimal AutoGluon Chronos-2 API probe.

Run BEFORE the full F3 setup. Tries three different API patterns on tiny
synthetic data (10 series × 100 timesteps), reports which one works, and prints
the prediction column layout we need for pinball-loss computation.

This is a discovery script, not a thesis-output script. Output goes to stdout
only. Total runtime: <5 min (including HF model download on first call).
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


print('=' * 70)
print('f3_probe.py — AutoGluon Chronos-2 API discovery')
print('=' * 70)


# ---- Step 1: import test ----
print('\n[1] Importing AutoGluon TimeSeries...')
try:
    import autogluon.timeseries
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    print(f'    autogluon.timeseries: {autogluon.timeseries.__version__}')
except ImportError as e:
    print(f'    IMPORT FAILED: {e}')
    sys.exit(1)


# ---- Step 2: inspect the fit signature ----
print('\n[2] Inspecting TimeSeriesPredictor.fit signature...')
import inspect
try:
    fit_sig = inspect.signature(TimeSeriesPredictor.fit)
    params = list(fit_sig.parameters.keys())
    print(f'    parameters: {params}')
except Exception as e:
    print(f'    inspect failed: {e}')


# ---- Step 3: tiny synthetic data ----
print('\n[3] Building synthetic data: 10 series × 100 timesteps...')
rng = np.random.default_rng(42)
rows = []
for sid in range(10):
    base_ts = pd.Timestamp('2025-01-01')
    for t in range(100):
        rows.append({
            'item_id':    f'series_{sid}',
            'timestamp':  base_ts + pd.Timedelta(minutes=5 * t),
            'target':     float(50 + 20 * np.sin(t / 10 + sid) + rng.normal(0, 2)),
        })
df = pd.DataFrame(rows)
tsdf = TimeSeriesDataFrame(df)
print(f'    TSDF: {len(tsdf)} rows, {tsdf.num_items} items')


# ---- Step 4: try three API patterns ----
print('\n[4] Trying Chronos-2 zero-shot via three API patterns...')

attempts = [
    # Pattern A: AutoGluon HF org + hyperparameters as list
    {
        'name':            'autogluon/chronos-2 (HF list)',
        'hyperparameters': {'Chronos': [{'model_path': 'autogluon/chronos-2'}]},
        'presets':         None,
    },
    # Pattern B: Amazon HF org + hyperparameters as list
    {
        'name':            'amazon/chronos-2 (HF list)',
        'hyperparameters': {'Chronos': [{'model_path': 'amazon/chronos-2'}]},
        'presets':         None,
    },
    # Pattern C: hyperparameters as dict (some AG versions use this)
    {
        'name':            'autogluon/chronos-2 (HF dict)',
        'hyperparameters': {'Chronos': {'model_path': 'autogluon/chronos-2'}},
        'presets':         None,
    },
    # Pattern D: bolt preset (Chronos-Bolt, simpler than Chronos-2 but related)
    {
        'name':            'bolt_small preset',
        'hyperparameters': None,
        'presets':         'bolt_small',
    },
    # Pattern E: chronos_small from hyperparameters
    {
        'name':            'Chronos with simple path',
        'hyperparameters': {'Chronos': {}},
        'presets':         None,
    },
]

success = None
for i, attempt in enumerate(attempts):
    print(f'\n    --- attempt {i + 1}: {attempt["name"]} ---')
    try:
        path = f'/tmp/probe_tsp_{i}'
        import shutil, os
        if os.path.exists(path):
            shutil.rmtree(path)

        predictor = TimeSeriesPredictor(
            prediction_length=6,
            target='target',
            quantile_levels=[0.5, 0.7, 0.8, 0.9, 0.95],
            verbosity=2,
            path=path,
        )

        fit_kwargs = {'time_limit': 300}
        if attempt['hyperparameters'] is not None:
            fit_kwargs['hyperparameters'] = attempt['hyperparameters']
        if attempt['presets'] is not None:
            fit_kwargs['presets'] = attempt['presets']

        print(f'    fit_kwargs: {fit_kwargs}')
        predictor.fit(tsdf, **fit_kwargs)
        print(f'    SUCCESS')
        print(f'    fitted models: {predictor.model_names()}')

        preds = predictor.predict(tsdf)
        print(f'    predictions shape: {preds.shape}')
        print(f'    prediction columns: {preds.columns.tolist()}')
        print(f'    sample (first 5 rows):')
        print(preds.head().to_string())

        success = attempt
        break
    except Exception as e:
        print(f'    FAILED: {type(e).__name__}: {str(e)[:200]}')
        import traceback
        # Only print short traceback to keep output readable
        tb_lines = traceback.format_exc().split('\n')
        last_lines = [l for l in tb_lines[-8:] if l.strip()]
        for ln in last_lines:
            print(f'      {ln}')

print('\n' + '=' * 70)
if success is not None:
    print(f'WORKING API PATTERN: {success["name"]}')
    print(f'  hyperparameters = {success["hyperparameters"]}')
    print(f'  presets = {success["presets"]}')
else:
    print('NO API PATTERN WORKED — fall back to chronos-forecasting library directly')
    print('  pip already installed chronos-forecasting==2.2.2')
    print('  Use: from chronos import BaseChronosPipeline; pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", ...)')
print('=' * 70)
