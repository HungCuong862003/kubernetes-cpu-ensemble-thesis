# %% [markdown]
# # Thesis Final Analyses v3: Seasonal Naive, MASE, Krogh-Vedelsby, MDA
#
# **v3 fixes** (from first run):
# - KV: pipeline operates in RESIDUAL space (pred_ens = naive + NNLS(oof_residuals)).
#   Decomposition now done in residual space with correct reconstruction.
# - MASE: container_index.csv has start_idx/end_idx columns, not per-point rows.
#   Expanded correctly now.
# - MDA: 71% of 10min samples have zero change. Now filters those out and
#   reports MDA only on samples where CPU actually moved.

# %% Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %% Imports
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import nnls as scipy_nnls
import warnings
warnings.filterwarnings('ignore')

# %% ===== CONFIGURATION =====
BASE = "/content/drive/MyDrive/k8s-ensemble-forecast"

ALI_TRAIN = os.path.join(BASE, "data/alibaba/train.parquet")
ALI_VAL   = os.path.join(BASE, "data/alibaba/val.parquet")
ALI_TEST  = os.path.join(BASE, "data/alibaba/test.parquet")
ALI_RESULTS = os.path.join(BASE, "results/alibaba")
PER_CONTAINER = os.path.join(BASE,
    "bitbrains-20260402T054032Z-3-001/thesis_upgrade/per_container")
OUT_DIR = os.path.join(BASE, "results/thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

HORIZONS = ["10min", "30min", "60min", "120min"]
H_STEPS  = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}
SEASONAL_PERIOD = 288

# NNLS weights from run.log — HOMO (3 tree models only)
HOMO_WEIGHTS = {
    "10min":  {"xgboost": 0.024, "lightgbm": 0.000, "extratrees": 0.976},
    "30min":  {"xgboost": 0.053, "lightgbm": 0.000, "extratrees": 0.947},
    "60min":  {"xgboost": 0.000, "lightgbm": 0.000, "extratrees": 1.000},
    "120min": {"xgboost": 0.000, "lightgbm": 0.000, "extratrees": 1.000},
}

# NNLS weights — HETERO (tree + BiLSTM, where BiLSTM OOF succeeded)
HETERO_WEIGHTS = {
    "10min":  None,  # BiLSTM OOF failed at 10min
    "30min":  {"xgboost": 0.007, "lightgbm": 0.045, "extratrees": 0.728, "bilstm": 0.220},
    "60min":  {"xgboost": 0.000, "lightgbm": 0.000, "extratrees": 0.784, "bilstm": 0.216},
    "120min": {"xgboost": 0.000, "lightgbm": 0.000, "extratrees": 0.759, "bilstm": 0.241},
}

EXPECTED_NAIVE_R2 = {
    "10min": 0.9188, "30min": 0.8361, "60min": 0.7878, "120min": 0.7178
}
EXPECTED_ENS_R2 = {
    "10min": 0.9213, "30min": 0.8404, "60min": 0.8011, "120min": 0.7642
}


# %% ===== CHECKPOINT HELPERS =====
def ckpt_exists(name):
    return os.path.exists(os.path.join(CKPT_DIR, name))

def ckpt_save(name, obj):
    path = os.path.join(CKPT_DIR, name)
    tmp = path + ".tmp"
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(tmp, index=False)
    elif isinstance(obj, dict):
        with open(tmp, 'w') as f:
            json.dump(obj, f)
    else:
        raise ValueError(f"Don't know how to checkpoint type {type(obj)}")
    os.rename(tmp, path)
    print(f"  [CKPT] Saved: {name}")

def ckpt_load_csv(name):
    return pd.read_csv(os.path.join(CKPT_DIR, name))

def ckpt_load_json(name):
    with open(os.path.join(CKPT_DIR, name)) as f:
        return json.load(f)


# %% Path check
print("Checking paths...")
all_ok = True
for p in [ALI_TRAIN, ALI_VAL, ALI_TEST, ALI_RESULTS, PER_CONTAINER]:
    if not os.path.exists(p):
        print(f"  MISSING: {p}")
        all_ok = False
    else:
        print(f"  OK: {p}")
if not all_ok:
    raise FileNotFoundError("Fix missing paths before continuing.")


# %% Column detection helper
def find_columns(df):
    ts_col = None
    cpu_col = None
    for c in df.columns:
        if c in ('time_stamp', 'timestamp', 'ts', 'date'):
            ts_col = c
        if c == 'cpu_util_percent':
            cpu_col = c
    if ts_col is None:
        raise KeyError(f"No timestamp column in {list(df.columns)}")
    if cpu_col is None:
        raise KeyError(f"No cpu column in {list(df.columns)}")
    return ts_col, cpu_col


# %% [markdown]
# ---
# ## Part 1: MASE denominators (unchanged from v2)

# %%
CKPT_DENOM = "mase_denoms.json"

if ckpt_exists(CKPT_DENOM):
    print(f"[SKIP] MASE denominators loaded from checkpoint")
    mase_denoms = ckpt_load_json(CKPT_DENOM)
    print(f"  {len(mase_denoms)} containers")
else:
    print("Loading train.parquet for MASE denominators...")
    train_df = pd.read_parquet(ALI_TRAIN)
    ts_col, cpu_col = find_columns(train_df)
    m = SEASONAL_PERIOD

    mase_denoms = {}
    skipped_short, skipped_flat, skipped_nan = 0, 0, 0

    for cid, grp in train_df.groupby('container_id'):
        series = grp.sort_values(ts_col)[cpu_col].values
        if np.any(np.isnan(series)):
            skipped_nan += 1; continue
        if len(series) <= m:
            skipped_short += 1; continue
        denom = np.mean(np.abs(series[m:] - series[:-m]))
        if denom < 1e-10:
            skipped_flat += 1; continue
        mase_denoms[str(cid)] = float(denom)

    print(f"  {len(mase_denoms)} containers OK, "
          f"skipped: {skipped_short} short + {skipped_flat} flat + {skipped_nan} NaN")
    ckpt_save(CKPT_DENOM, mase_denoms)
    del train_df


# %% [markdown]
# ---
# ## Part 2: Seasonal naive (unchanged from v2)

# %%
CKPT_SEASONAL = "seasonal_naive_results.csv"
CKPT_SEASONAL_PC = "seasonal_naive_per_container.csv"

if ckpt_exists(CKPT_SEASONAL) and ckpt_exists(CKPT_SEASONAL_PC):
    print(f"[SKIP] Seasonal naive loaded from checkpoint")
    seasonal_df = ckpt_load_csv(CKPT_SEASONAL)
    pc_df = ckpt_load_csv(CKPT_SEASONAL_PC)
else:
    print("Loading parquet files for seasonal naive...")
    train_df = pd.read_parquet(ALI_TRAIN)
    val_df   = pd.read_parquet(ALI_VAL)
    test_df  = pd.read_parquet(ALI_TEST)
    ts_col, cpu_col = find_columns(train_df)
    keep_cols = ['container_id', ts_col, cpu_col]

    train_df['_split'] = 'train'
    val_df['_split']   = 'val'
    test_df['_split']  = 'test'
    tagged = pd.concat([
        train_df[keep_cols + ['_split']],
        val_df[keep_cols + ['_split']],
        test_df[keep_cols + ['_split']]
    ], ignore_index=True)
    print(f"  Concatenated: {len(tagged):,} rows")
    train_df.drop(columns=['_split'], inplace=True)
    val_df.drop(columns=['_split'], inplace=True)
    test_df.drop(columns=['_split'], inplace=True)
    del train_df, val_df, test_df

    test_cids = set(tagged.loc[tagged['_split'] == 'test', 'container_id'].unique())
    valid_cids = sorted(test_cids & set(mase_denoms.keys()))
    if len(valid_cids) == 0:
        tagged['container_id'] = tagged['container_id'].astype(str)
        test_cids = set(tagged.loc[tagged['_split'] == 'test', 'container_id'].unique())
        valid_cids = sorted(test_cids & set(mase_denoms.keys()))
    print(f"  {len(valid_cids)} containers with valid MASE denoms")
    assert len(valid_cids) > 0, "No container overlap! Check types."

    m = SEASONAL_PERIOD
    grouped = tagged.groupby('container_id')
    seasonal_rows = []
    per_container_rows = []

    for hz in HORIZONS:
        h = H_STEPS[hz]
        print(f"\n--- {hz} (h={h}) ---")
        all_act, all_seas, all_pers = [], [], []
        s_r2s, s_maes, s_mases = [], [], []
        p_r2s, p_maes, p_mases = [], [], []
        n_pts, n_err = 0, 0

        for cid in valid_cids:
            try:
                grp = grouped.get_group(cid).sort_values(ts_col)
                full = grp[cpu_col].values
                splits = grp['_split'].values
                if np.any(np.isnan(full)): continue
                test_mask = (splits == 'test')
                if not test_mask.any(): continue
                test_start = np.argmax(test_mask)

                min_o = max(test_start, m - h)
                max_o = len(full) - h - 1
                if min_o > max_o: continue

                origins = np.arange(min_o, max_o + 1)
                targets = origins + h
                s_refs = targets - m
                ok = s_refs >= 0
                if not ok.all():
                    origins, targets, s_refs = origins[ok], targets[ok], s_refs[ok]
                if len(targets) < 5: continue

                act = full[targets]
                s_pred = full[s_refs]
                p_pred = full[origins]
                if np.std(act) < 1e-12: continue

                sm = np.mean(np.abs(act - s_pred))
                sr = r2_score(act, s_pred)
                pm = np.mean(np.abs(act - p_pred))
                pr = r2_score(act, p_pred)
                d = mase_denoms[str(cid)]

                s_r2s.append(sr); s_maes.append(sm); s_mases.append(sm/d)
                p_r2s.append(pr); p_maes.append(pm); p_mases.append(pm/d)
                n_pts += len(act)
                all_act.append(act); all_seas.append(s_pred); all_pers.append(p_pred)
                per_container_rows.append({
                    'horizon': hz, 'container_id': cid,
                    'seasonal_r2': sr, 'seasonal_mae': sm, 'seasonal_mase': sm/d,
                    'persist_r2': pr, 'persist_mae': pm, 'persist_mase': pm/d,
                })
            except Exception as e:
                n_err += 1
                if n_err <= 3: print(f"  ERROR {cid}: {e}")

        if len(s_r2s) == 0:
            print("  WARNING: no valid containers"); continue

        pa = np.concatenate(all_act)
        ps = np.concatenate(all_seas)
        pp = np.concatenate(all_pers)
        row = {
            'Horizon': hz, 'N_containers': len(s_r2s), 'N_points': n_pts,
            'Seasonal_R2_pooled': round(r2_score(pa, ps), 6),
            'Seasonal_MAE_pooled': round(mean_absolute_error(pa, ps), 6),
            'Persist_R2_pooled': round(r2_score(pa, pp), 6),
            'Persist_MAE_pooled': round(mean_absolute_error(pa, pp), 6),
            'Seasonal_R2_median': round(np.median(s_r2s), 6),
            'Seasonal_MAE_median': round(np.median(s_maes), 6),
            'Seasonal_MASE_median': round(np.median(s_mases), 6),
            'Seasonal_MASE_mean': round(np.mean(s_mases), 6),
            'Persist_R2_median': round(np.median(p_r2s), 6),
            'Persist_MAE_median': round(np.median(p_maes), 6),
            'Persist_MASE_median': round(np.median(p_mases), 6),
            'Persist_MASE_mean': round(np.mean(p_mases), 6),
        }
        seasonal_rows.append(row)
        print(f"  {row['N_containers']} containers, {n_pts:,} points")
        print(f"  POOLED: seasonal R2={row['Seasonal_R2_pooled']:.4f}, "
              f"persist R2={row['Persist_R2_pooled']:.4f}")
        exp = EXPECTED_NAIVE_R2[hz]
        diff = abs(row['Persist_R2_pooled'] - exp)
        print(f"  Sanity: persist R2 diff={diff:.4f}pp vs expected -> "
              f"{'OK' if diff < 0.05 else 'CHECK'}")

    seasonal_df = pd.DataFrame(seasonal_rows)
    pc_df = pd.DataFrame(per_container_rows)
    if len(seasonal_df) > 0:
        ckpt_save(CKPT_SEASONAL, seasonal_df)
        ckpt_save(CKPT_SEASONAL_PC, pc_df)
        seasonal_df.to_csv(os.path.join(OUT_DIR, "seasonal_naive_results.csv"), index=False)
        pc_df.to_csv(os.path.join(OUT_DIR, "seasonal_naive_per_container.csv"), index=False)
    else:
        print("  WARNING: no results, checkpoint NOT saved.")
    del tagged, grouped

print("\n--- Seasonal Naive ---")
if len(seasonal_df) > 0:
    print(seasonal_df.to_string(index=False))
else:
    print("  (empty)")


# %% [markdown]
# ---
# ## Part 3: Load .npy arrays (FIXED: load both direct + OOF predictions)
#
# **Key insight from sprint1_Main.py source code:**
# The ensemble operates in RESIDUAL space:
#   `pred_ensemble = pred_naive + NNLS(oof_xgb_test, oof_lgb_test, oof_et_test[, bilstm_test])`
# The OOF .npy files predict `cpu_residual`, not `cpu_target`.

# %%
print("\n" + "="*60)
print("LOADING .NPY PREDICTION ARRAYS")
print("="*60)

data = {}

for hz in HORIZONS:
    print(f"\n--- {hz} ---")
    hz_dir = os.path.join(ALI_RESULTS, hz)
    pc_dir = os.path.join(PER_CONTAINER, hz)
    d = {}

    # y_true from per_container
    yt_path = os.path.join(pc_dir, "y_true.npy")
    if not os.path.exists(yt_path):
        print(f"  y_true NOT FOUND — skipping {hz}")
        data[hz] = d; continue
    d['y_true'] = np.load(yt_path)
    n = len(d['y_true'])
    print(f"  y_true: {n:,} points")

    # load everything we might need
    file_map = [
        ('pred_naive',       'pred_naive.npy'),
        ('pred_ens',         'pred_hetero_ensemble.npy'),
        ('pred_homo',        'pred_homo_ensemble.npy'),
        # direct model predictions (absolute CPU values, for MDA + MASE)
        ('pred_xgb',         'pred_xgboost.npy'),
        ('pred_lgb',         'pred_lightgbm.npy'),
        ('pred_et',          'pred_extratrees.npy'),
        # OOF test predictions (residual space, for Krogh-Vedelsby)
        ('oof_xgb',          'pred_oof_xgb_test.npy'),
        ('oof_lgb',          'pred_oof_lgb_test.npy'),
        ('oof_et',           'pred_oof_et_test.npy'),
        # BiLSTM (may have length mismatch)
        ('bilstm_test',      'pred_bilstm_test.npy'),
    ]

    for name, fname in file_map:
        fpath = os.path.join(hz_dir, fname)
        if not os.path.exists(fpath):
            print(f"  {name}: NOT FOUND")
            continue
        arr = np.load(fpath)
        # allow bilstm to have different length (it will be handled later)
        if name == 'bilstm_test':
            d[name] = arr
            if len(arr) != n:
                print(f"  {name}: {len(arr):,} points (offset={n - len(arr)})")
            else:
                print(f"  {name}: {len(arr):,} points OK")
        elif len(arr) != n:
            print(f"  {name}: LENGTH MISMATCH ({len(arr)} vs {n}) — SKIPPING")
        else:
            d[name] = arr
            print(f"  {name}: {len(arr):,} points OK")

    # R2 sanity checks
    if 'pred_naive' in d:
        r2 = r2_score(d['y_true'], d['pred_naive'])
        exp = EXPECTED_NAIVE_R2[hz]
        print(f"  Naive R2: {r2:.4f} vs expected {exp:.4f} -> "
              f"{'OK' if abs(r2-exp)<0.01 else 'MISMATCH'}")
    if 'pred_ens' in d:
        r2 = r2_score(d['y_true'], d['pred_ens'])
        exp = EXPECTED_ENS_R2[hz]
        print(f"  Ensemble R2: {r2:.4f} vs expected {exp:.4f} -> "
              f"{'OK' if abs(r2-exp)<0.01 else 'MISMATCH'}")

    # verify residual-space reconstruction:
    # pred_naive + NNLS(oof_residuals) ≈ pred_homo_ensemble
    if all(k in d for k in ['pred_naive', 'pred_homo', 'oof_xgb', 'oof_lgb', 'oof_et']):
        hw = HOMO_WEIGHTS[hz]
        w_h = np.array([hw['xgboost'], hw['lightgbm'], hw['extratrees']])
        ws = w_h.sum()
        if ws > 1e-10:
            w_h = w_h / ws
        P_oof = np.array([d['oof_xgb'], d['oof_lgb'], d['oof_et']])
        recon = d['pred_naive'] + w_h @ P_oof
        maxd = np.max(np.abs(recon - d['pred_homo']))
        print(f"  Homo reconstruction check: max_diff={maxd:.6f} "
              f"-> {'OK' if maxd < 0.01 else 'MISMATCH'}")

    data[hz] = d


# %% [markdown]
# ---
# ## Part 4: Krogh-Vedelsby (FIXED: residual-space decomposition)
#
# The pipeline does: `ensemble = naive + NNLS(residual_predictions)`
# So the NNLS combination happens in RESIDUAL space.
# We decompose there: `actual_residual = y_true - pred_naive`
#
# The identity: `E_ens_res = E_bar_res - Ambiguity_res` (exact for MSE)

# %%
CKPT_KV = "krogh_vedelsby.csv"

if ckpt_exists(CKPT_KV):
    print(f"[SKIP] Krogh-Vedelsby loaded from checkpoint")
    kv_df = ckpt_load_csv(CKPT_KV)
else:
    print("\n" + "="*60)
    print("KROGH-VEDELSBY (RESIDUAL SPACE)")
    print("="*60)

    kv_rows = []

    for hz in HORIZONS:
        print(f"\n--- {hz} ---")
        d = data[hz]
        if not all(k in d for k in ['y_true', 'pred_naive']):
            print("  Missing y_true or pred_naive, skipping")
            continue

        y_true = d['y_true']
        naive = d['pred_naive']
        actual_res = y_true - naive  # what the OOF models try to predict

        # --- HOMO decomposition (3 tree models, all test points) ---
        if all(k in d for k in ['oof_xgb', 'oof_lgb', 'oof_et']):
            hw = HOMO_WEIGHTS[hz]
            w = np.array([hw['xgboost'], hw['lightgbm'], hw['extratrees']])
            ws = w.sum()
            if ws < 1e-10:
                print("  Homo: all weights zero, skipping")
            else:
                w = w / ws
                P = np.array([d['oof_xgb'], d['oof_lgb'], d['oof_et']])
                f_ens = w @ P  # ensemble residual prediction
                model_names = ['XGB', 'LGB', 'ET']

                E_ens = np.mean((actual_res - f_ens) ** 2)
                indiv_mses = np.mean((P - actual_res) ** 2, axis=1)
                E_bar = np.dot(w, indiv_mses)
                A_bar = np.mean(np.dot(w, (P - f_ens) ** 2))
                check = abs(E_ens - (E_bar - A_bar))
                ratio = A_bar / E_bar if E_bar > 0 else 0.0

                # also report the absolute-space ensemble R2
                abs_ens = naive + f_ens
                abs_r2 = r2_score(y_true, abs_ens)

                print(f"  HOMO ({'+'.join(model_names)}):")
                print(f"    Weights: {[f'{x:.3f}' for x in w]}")
                print(f"    E_bar={E_bar:.4f}, Ambiguity={A_bar:.6f}, E_ens={E_ens:.4f}")
                print(f"    Identity check: {check:.2e}")
                print(f"    Ambiguity ratio: {ratio:.6f} ({ratio*100:.4f}%)")
                print(f"    Absolute R2 of reconstructed ensemble: {abs_r2:.4f}")
                for i, nm in enumerate(model_names):
                    print(f"      {nm}: w={w[i]:.3f}, residual_MSE={indiv_mses[i]:.4f}")

                kv_rows.append({
                    'Horizon': hz, 'Type': 'homo',
                    'E_bar': round(E_bar, 6), 'Ambiguity': round(A_bar, 8),
                    'E_ens': round(E_ens, 6), 'Ambiguity_Ratio': round(ratio, 8),
                    'Identity_Check': check, 'Abs_R2': round(abs_r2, 6),
                    'Models': '+'.join(model_names),
                })

        # --- HETERO decomposition (tree + BiLSTM, BiLSTM subset only) ---
        het_w = HETERO_WEIGHTS[hz]
        if het_w is not None and 'bilstm_test' in d:
            n_bilstm = len(d['bilstm_test'])
            n_total = len(y_true)
            n_pad = n_total - n_bilstm

            if n_pad < 0 or n_bilstm < 100:
                print(f"  Hetero: BiLSTM alignment issue (pad={n_pad}), skipping")
            elif not all(k in d for k in ['oof_xgb', 'oof_lgb', 'oof_et']):
                print(f"  Hetero: missing OOF arrays, skipping")
            else:
                # work on the last n_bilstm points (where BiLSTM has predictions)
                sl = slice(n_pad, None)
                ar_sl = actual_res[sl]
                P4 = np.array([
                    d['oof_xgb'][sl], d['oof_lgb'][sl],
                    d['oof_et'][sl], d['bilstm_test']
                ])
                w4 = np.array([het_w['xgboost'], het_w['lightgbm'],
                               het_w['extratrees'], het_w['bilstm']])
                w4s = w4.sum()
                if w4s > 1e-10:
                    w4 = w4 / w4s

                f4 = w4 @ P4
                E4_ens = np.mean((ar_sl - f4) ** 2)
                indiv4 = np.mean((P4 - ar_sl) ** 2, axis=1)
                E4_bar = np.dot(w4, indiv4)
                A4_bar = np.mean(np.dot(w4, (P4 - f4) ** 2))
                check4 = abs(E4_ens - (E4_bar - A4_bar))
                ratio4 = A4_bar / E4_bar if E4_bar > 0 else 0.0

                abs_ens4 = d['pred_naive'][sl] + f4
                abs_r2_4 = r2_score(y_true[sl], abs_ens4)

                names4 = ['XGB', 'LGB', 'ET', 'BiLSTM']
                print(f"  HETERO ({'+'.join(names4)}, last {n_bilstm:,} points):")
                print(f"    Weights: {[f'{x:.3f}' for x in w4]}")
                print(f"    E_bar={E4_bar:.4f}, Ambiguity={A4_bar:.6f}, E_ens={E4_ens:.4f}")
                print(f"    Identity check: {check4:.2e}")
                print(f"    Ambiguity ratio: {ratio4:.6f} ({ratio4*100:.4f}%)")
                print(f"    Absolute R2: {abs_r2_4:.4f}")
                for i, nm in enumerate(names4):
                    print(f"      {nm}: w={w4[i]:.3f}, residual_MSE={indiv4[i]:.4f}")

                kv_rows.append({
                    'Horizon': hz, 'Type': 'hetero',
                    'E_bar': round(E4_bar, 6), 'Ambiguity': round(A4_bar, 8),
                    'E_ens': round(E4_ens, 6), 'Ambiguity_Ratio': round(ratio4, 8),
                    'Identity_Check': check4, 'Abs_R2': round(abs_r2_4, 6),
                    'Models': '+'.join(names4),
                })

    kv_df = pd.DataFrame(kv_rows)
    if len(kv_df) > 0:
        ckpt_save(CKPT_KV, kv_df)
        kv_df.to_csv(os.path.join(OUT_DIR, "krogh_vedelsby.csv"), index=False)
    else:
        print("  WARNING: no KV results.")

print("\n--- Krogh-Vedelsby ---")
if len(kv_df) > 0:
    print(kv_df[['Horizon','Type','E_bar','Ambiguity','E_ens','Ambiguity_Ratio','Abs_R2']].to_string(index=False))
else:
    print("  (empty)")


# %% KV bar chart (always regenerate)
if len(kv_df) > 0:
    # plot homo and hetero side by side per horizon
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ens_type, title in [
        (axes[0], 'homo', 'Homo Ensemble (3 tree models)'),
        (axes[1], 'hetero', 'Hetero Ensemble (tree + BiLSTM)'),
    ]:
        sub = kv_df[kv_df['Type'] == ens_type]
        if len(sub) == 0:
            ax.set_title(f'{title}\n(no data)')
            continue
        labels = sub['Horizon'].tolist()
        e_ens = sub['E_ens'].tolist()
        ambig = sub['Ambiguity'].tolist()
        x = np.arange(len(labels))
        ax.bar(x, e_ens, 0.5, label='Ensemble MSE (residual)', color='#2196F3')
        ax.bar(x, ambig, 0.5, bottom=e_ens, label='Ambiguity', color='#FF9800')
        for i in range(len(sub)):
            pct = sub.iloc[i]['Ambiguity_Ratio'] * 100
            total = sub.iloc[i]['E_bar']
            ax.text(i, total + total * 0.02, f'{pct:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Residual MSE')
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle('Krogh-Vedelsby Decomposition (Residual Space)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "krogh_vedelsby_bar.pdf"), dpi=150, bbox_inches='tight')
    print(f"Saved: krogh_vedelsby_bar.pdf")
    plt.show()


# %% [markdown]
# ---
# ## Part 5: MDA (FIXED: filter zero-change, use direct predictions)
#
# At 10min, 71% of samples have y_true == pred_naive (CPU didn't move).
# np.sign(0) = 0 never matches any model's predicted direction,
# crushing MDA artificially. We now exclude zero-change samples.

# %%
CKPT_MDA = "mda_results.csv"

if ckpt_exists(CKPT_MDA):
    print(f"[SKIP] MDA loaded from checkpoint")
    mda_df = ckpt_load_csv(CKPT_MDA)
else:
    print("\n" + "="*60)
    print("DIRECTIONAL ACCURACY (MDA)")
    print("="*60)

    mda_rows = []

    # use DIRECT predictions (absolute CPU values) for MDA
    models_to_eval = {
        'Hetero Ensemble': 'pred_ens',
        'Homo Ensemble':   'pred_homo',
        'XGBoost':         'pred_xgb',
        'LightGBM':        'pred_lgb',
        'ExtraTrees':      'pred_et',
    }

    for hz in HORIZONS:
        print(f"\n--- {hz} ---")
        d = data[hz]
        if 'y_true' not in d or 'pred_naive' not in d:
            print("  Skipping (missing data)")
            continue

        y_true = d['y_true']
        y_origin = d['pred_naive']  # current value at prediction origin
        actual_dir = np.sign(y_true - y_origin)

        # identify non-zero-change samples
        nonzero = (actual_dir != 0)
        n_total = len(actual_dir)
        n_nonzero = int(nonzero.sum())
        frac_zero = 1.0 - n_nonzero / n_total
        print(f"  {n_total:,} total, {n_nonzero:,} with actual change "
              f"({frac_zero*100:.1f}% zero-change excluded)")

        for model_label, arr_key in models_to_eval.items():
            if arr_key not in d:
                continue
            pred = d[arr_key]
            pred_dir = np.sign(pred - y_origin)
            match = (actual_dir == pred_dir)

            # MDA on ALL samples (including zero-change)
            mda_all = float(np.mean(match))

            # MDA on NON-ZERO samples only (the meaningful metric)
            if n_nonzero > 0:
                mda_nz = float(np.mean(match[nonzero]))
            else:
                mda_nz = float('nan')

            print(f"  {model_label:20s}: MDA_nonzero={mda_nz:.4f}  "
                  f"(MDA_all={mda_all:.4f})")

            mda_rows.append({
                'Horizon': hz,
                'Model': model_label,
                'MDA_nonzero': round(mda_nz, 6),
                'MDA_all': round(mda_all, 6),
                'N_total': n_total,
                'N_nonzero': n_nonzero,
                'Frac_zero': round(frac_zero, 4),
            })

    mda_df = pd.DataFrame(mda_rows)
    if len(mda_df) > 0:
        ckpt_save(CKPT_MDA, mda_df)
        mda_df.to_csv(os.path.join(OUT_DIR, "mda_results.csv"), index=False)
    else:
        print("  WARNING: no MDA results.")

print("\n--- MDA (non-zero change only) ---")
if len(mda_df) > 0:
    pivot = mda_df.pivot(index='Horizon', columns='Model', values='MDA_nonzero')
    hz_order = [h for h in HORIZONS if h in pivot.index]
    if hz_order:
        pivot = pivot.reindex(hz_order)
    print(pivot.to_string())
    print(f"\nZero-change fractions:")
    for hz in HORIZONS:
        sub = mda_df[mda_df['Horizon'] == hz]
        if len(sub) > 0:
            print(f"  {hz}: {sub.iloc[0]['Frac_zero']*100:.1f}% of samples had no CPU change")
else:
    print("  (empty)")


# %% [markdown]
# ---
# ## Part 6: MASE for ML models (FIXED: expand container_index correctly)
#
# container_index.csv has columns: container_id, start_idx, end_idx, n_points
# One row per container, not per prediction point.
# Need to expand start_idx:end_idx into a per-point container_id array.

# %%
CKPT_MASE = "mase_comparison.csv"

if ckpt_exists(CKPT_MASE):
    print(f"[SKIP] MASE loaded from checkpoint")
    mase_model_df = ckpt_load_csv(CKPT_MASE)
else:
    print("\n" + "="*60)
    print("MASE FOR ML MODELS")
    print("="*60)

    mase_model_rows = []

    for hz in HORIZONS:
        print(f"\n--- {hz} ---")
        d = data[hz]
        if 'y_true' not in d:
            print("  Skipping"); continue

        ci_path = os.path.join(PER_CONTAINER, hz, "container_index.csv")
        if not os.path.exists(ci_path):
            print(f"  container_index.csv not found, skipping"); continue

        ci_df = pd.read_csv(ci_path)
        print(f"  container_index: {len(ci_df)} containers, cols={list(ci_df.columns)}")

        y_true = d['y_true']
        n = len(y_true)

        # expand container_index into per-point array using start_idx/end_idx
        if 'start_idx' in ci_df.columns and 'end_idx' in ci_df.columns:
            container_ids = np.full(n, '', dtype=object)
            n_mapped = 0
            for _, row in ci_df.iterrows():
                s = int(row['start_idx'])
                e = int(row['end_idx'])
                cid_str = str(row['container_id'])
                if s < 0 or e > n:
                    continue  # skip out-of-bounds
                container_ids[s:e] = cid_str
                n_mapped += (e - s)
            n_unmapped = int(np.sum(container_ids == ''))
            print(f"  Expanded: {n_mapped:,} mapped, {n_unmapped:,} unmapped out of {n:,}")
            if n_unmapped > n * 0.1:
                print(f"  WARNING: >10% unmapped points — check container_index alignment")
        else:
            # fallback: if columns are different, try first column as per-point
            print(f"  No start_idx/end_idx columns, trying first column as per-point...")
            cid_col = ci_df.columns[0]
            if len(ci_df) == n:
                container_ids = np.array([str(x) for x in ci_df[cid_col].values])
            else:
                print(f"  FATAL: can't figure out container mapping. Skipping {hz}.")
                continue

        # compute per-model MASE
        model_arrays = {
            'Hetero Ensemble': d.get('pred_ens'),
            'Homo Ensemble':   d.get('pred_homo'),
            'XGBoost':         d.get('pred_xgb'),
            'LightGBM':        d.get('pred_lgb'),
            'ExtraTrees':      d.get('pred_et'),
            'Persistence':     d.get('pred_naive'),
        }

        for model_label, pred_arr in model_arrays.items():
            if pred_arr is None: continue
            if len(pred_arr) != n:
                print(f"  {model_label}: length mismatch, skipping"); continue

            mase_values = []
            for cid_str in np.unique(container_ids):
                if cid_str == '' or cid_str not in mase_denoms:
                    continue
                mask = (container_ids == cid_str)
                if mask.sum() < 3: continue

                yt = y_true[mask]
                yp = pred_arr[mask]
                if np.any(np.isnan(yt)) or np.any(np.isnan(yp)): continue

                mae_m = np.mean(np.abs(yt - yp))
                mase_values.append(mae_m / mase_denoms[cid_str])

            if len(mase_values) > 0:
                med = np.median(mase_values)
                mn = np.mean(mase_values)
                print(f"  {model_label:20s}: MASE median={med:.4f}, mean={mn:.4f} "
                      f"({len(mase_values)} containers)")
                mase_model_rows.append({
                    'Horizon': hz, 'Model': model_label,
                    'MASE_median': round(med, 6), 'MASE_mean': round(mn, 6),
                    'N_containers': len(mase_values),
                })
            else:
                print(f"  {model_label:20s}: no valid containers")

    mase_model_df = pd.DataFrame(mase_model_rows)
    if len(mase_model_df) > 0:
        ckpt_save(CKPT_MASE, mase_model_df)
        mase_model_df.to_csv(os.path.join(OUT_DIR, "mase_comparison.csv"), index=False)
    else:
        print("  WARNING: no MASE results.")

print("\n--- MASE Comparison ---")
if len(mase_model_df) > 0:
    pivot = mase_model_df.pivot(index='Horizon', columns='Model', values='MASE_median')
    hz_order = [h for h in HORIZONS if h in pivot.index]
    if hz_order:
        pivot = pivot.reindex(hz_order)
    print(pivot.to_string())
else:
    print("  (empty)")


# %% [markdown]
# ---
# ## Part 7: Summary

# %%
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print("\n--- Seasonal Naive ---")
if len(seasonal_df) > 0:
    print(seasonal_df.to_string(index=False))
else:
    print("  (empty)")

print("\n--- Krogh-Vedelsby ---")
if len(kv_df) > 0:
    print(kv_df[['Horizon','Type','E_bar','Ambiguity','Ambiguity_Ratio','Abs_R2']].to_string(index=False))
else:
    print("  (empty)")

print("\n--- MASE ---")
if len(mase_model_df) > 0:
    pivot = mase_model_df.pivot(index='Horizon', columns='Model', values='MASE_median')
    hz_order = [h for h in HORIZONS if h in pivot.index]
    if hz_order: pivot = pivot.reindex(hz_order)
    print(pivot.to_string())
else:
    print("  (empty)")

print("\n--- MDA (non-zero change only) ---")
if len(mda_df) > 0:
    pivot2 = mda_df.pivot(index='Horizon', columns='Model', values='MDA_nonzero')
    hz_order2 = [h for h in HORIZONS if h in pivot2.index]
    if hz_order2: pivot2 = pivot2.reindex(hz_order2)
    print(pivot2.to_string())
else:
    print("  (empty)")

print("\n\nOutputs:", OUT_DIR)
print("Checkpoints:", CKPT_DIR)
for f in ['seasonal_naive_results.csv', 'seasonal_naive_per_container.csv',
          'krogh_vedelsby.csv', 'krogh_vedelsby_bar.pdf',
          'mda_results.csv', 'mase_comparison.csv']:
    full = os.path.join(OUT_DIR, f)
    print(f"  [{'OK' if os.path.exists(full) else 'MISSING'}] {f}")

print("\nTo rerun: delete", CKPT_DIR)
print("Done!")
