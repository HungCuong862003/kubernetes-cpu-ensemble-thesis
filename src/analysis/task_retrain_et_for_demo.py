"""
task_retrain_et_for_demo.py
============================
Retrain ExtraTrees on a subsample of train.parquet for the live HPA demo
(FastAPI predictor service in Day 10 of thesis_plan.md).

Goal: a serialized ET model that loads in <2s, predicts in <50ms per call,
and whose R2 stays close enough to the thesis numbers (in comparison_table.csv)
to defend at the June 2026 oral.

This script does NOT replicate OOF. Live inference doesn't need OOF.
A single fit on the subsample is sufficient. We compare to the full-pipeline
thesis numbers loaded from comparison_table.csv.

Hyperparameters: matched to sprint1_Main.py Config.ET_PARAMS exactly.
Training target:  cpu_residual = cpu_target - naive_cpu (NOT cpu_target).
Inference:        pred = naive_cpu + et.predict(X)
Dtype:            float32 (matches sprint1_Main.py safe_extract_array)
Inf handling:     nan_to_num with posinf=0.0, neginf=0.0 (also sprint1)

Pipeline:
    1. env check (sklearn version, file paths)
    2. load thesis baselines from comparison_table.csv
    3. sample K container_ids from raw train -- FE only this subset
    4. FE the full val and full test (no subsample, evaluation must be honest)
    5. fail-fast: compare our naive R2 to thesis naive R2 BEFORE training
    6. train ET on subsampled train, save as joblib + JSON manifest
    7. evaluate on val (hold-out sanity), then test (vs thesis numbers)
    8. round-trip self-test: reload saved model from disk, predict, compare
    9. write verification CSV

Convention: matches base_model_correlations.py / residual_diagnostics_figure.py
-- runtime import of sprint1_Main.py for create_features() and friends.

Run on Windows (E:) or Linux. CPU only, no GPU.
"""

import os
import sys
import json
import time
import importlib.util

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


# =====================================================================
# CONFIG -- edit these for your machine
# =====================================================================

# data dir containing train.parquet, val.parquet, test.parquet
DATA_DIR = r"E:\k8s-ensemble-forecast\data\alibaba"

# where sprint1_Main.py lives -- imported at runtime for create_features
SPRINT1_DIR = r"E:\k8s-ensemble-forecast\src"

# where to write outputs (model, manifest, verification CSV)
OUTPUT_DIR = r"E:\k8s-ensemble-forecast\demo_models"

# path to comparison_table.csv -- thesis ground-truth R2 baselines
COMPARISON_TABLE_CSV = r"E:\k8s-ensemble-forecast\results\thesis_figures\comparison_table.csv"


# how many container_ids to sample from raw train
# Alibaba: ~4900 containers, ~1500 post-FE rows each at 10min,
# ~1200 at 120min (because hist_lag drops the first ~313 rows).
# K=200 -> ~240K post-FE rows at the worst horizon. Plenty of headroom.
N_TRAIN_CONTAINERS = 200

# how many rows to keep after FE (before training ET)
# 200K is well within laptop RAM; ET with squared_error criterion is
# fast enough that going bigger isn't a clear win. The 50K target in
# my project memory was for the MAE criterion variant, which scales
# O(n^2) -- this script uses the default squared_error so 200K is fine.
N_SUBSAMPLE = 200_000

# random seed (single seed for everything for reproducibility)
SEED = 42

# horizon name -> number of 5-min steps ahead (must match sprint1_Main.py)
HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}

# rows that sprint1 drops NaN on -- must match exactly
REQUIRED_COLS = [
    "cpu_target", "cpu_residual", "naive_cpu",
    "mem_target", "mem_residual", "naive_mem",
]

# match sprint1_Main.py Config.ET_PARAMS EXACTLY -- do not change
ET_PARAMS = {
    "n_estimators":     200,
    "max_depth":         25,
    "min_samples_leaf":   5,
    "max_features":     0.8,
    "bootstrap":       True,
    "max_samples":      0.3,
    "random_state":      42,
    "n_jobs":            -1,
}

# pinned sklearn version -- the project trains and serves both on 1.5.2
EXPECTED_SKLEARN = "1.5.2"

# naive_R2 mismatch tolerance vs thesis (percentage points)
# bigger than this means our test split differs from the thesis test split
NAIVE_GAP_FAIL_PP = 0.5

# ET R2 parity warning vs thesis (percentage points)
# the thesis ET trains on ~7.4M rows; this script trains on ~200K, so
# a small gap is expected. 2.0 pp is loose enough to not cry wolf,
# tight enough that a 5pp gap (real model bug) still gets flagged.
PARITY_WARN_PP = 2.0

# self-test: how many test rows to round-trip through the saved model
SELFTEST_ROWS = 100

# self-test: max allowed prediction difference between in-memory model
# and freshly loaded model (should be 0.0 exactly for ET)
SELFTEST_MAX_DIFF = 1e-6


# =====================================================================
# DYNAMIC IMPORT of sprint1_Main.py
# =====================================================================
# Same pattern as base_model_correlations.py and residual_diagnostics_figure.py.

_s1 = None

def load_sprint1():
    global _s1
    for name in ("sprint1_Main", "sprint1_v9"):
        path = os.path.join(SPRINT1_DIR, f"{name}.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _s1 = mod
            print(f"loaded {name}.py from {SPRINT1_DIR}")

            # sanity: the loaded module's ET_PARAMS must match ours.
            # if SPRINT1_DIR points at a stale checkout we want to know
            # NOW, not after producing a model that disagrees with the
            # thesis numbers.
            if hasattr(_s1, "cfg") and hasattr(_s1.cfg, "ET_PARAMS"):
                # ignore n_jobs since cfg may have it overridden at runtime
                ours = {k: v for k, v in ET_PARAMS.items() if k != "n_jobs"}
                theirs = {k: v for k, v in _s1.cfg.ET_PARAMS.items() if k != "n_jobs"}
                if ours != theirs:
                    diffs = []
                    for k in set(ours) | set(theirs):
                        if ours.get(k) != theirs.get(k):
                            diffs.append(f"{k}: script={ours.get(k)} sprint1={theirs.get(k)}")
                    raise RuntimeError(
                        f"loaded sprint1's cfg.ET_PARAMS does not match this "
                        f"script's ET_PARAMS:\n  " + "\n  ".join(diffs) + "\n"
                        f"either update ET_PARAMS in this script or point "
                        f"SPRINT1_DIR at a sprint1_Main.py that matches the thesis.")
                print(f"  ok: cfg.ET_PARAMS matches script's ET_PARAMS")
            else:
                print(f"  WARN: loaded sprint1 has no cfg.ET_PARAMS to verify")
            return
    raise FileNotFoundError(
        f"sprint1_Main.py / sprint1_v9.py not found in {SPRINT1_DIR}")


# =====================================================================
# ARRAY EXTRACTION (matches sprint1_Main.py safe_extract_array exactly)
# =====================================================================
# Sprint1's helper does .values.astype(float32) then nan_to_num with
# nan=0, posinf=0, neginf=0. We reproduce it byte-for-byte so the
# retrained ET trains on the SAME numerical inputs as the thesis ET.

def safe_extract(df, cols):
    """Match sprint1_Main.safe_extract_array exactly."""
    arr = df[cols].values.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# =====================================================================
# ENVIRONMENT CHECK
# =====================================================================

def env_check():
    print(f"\n--- env check ---")
    print(f"  python:  {sys.version.split()[0]}")
    print(f"  sklearn: {sklearn.__version__}")
    if sklearn.__version__ != EXPECTED_SKLEARN:
        print(f"  WARN: sklearn != {EXPECTED_SKLEARN}")
        print(f"        the production pipeline uses {EXPECTED_SKLEARN}")
        print(f"        a model trained here may behave differently in serving")

    for label, path in [
        ("DATA_DIR",             DATA_DIR),
        ("SPRINT1_DIR",          SPRINT1_DIR),
        ("COMPARISON_TABLE_CSV", COMPARISON_TABLE_CSV),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")
        print(f"  ok: {label} = {path}")

    for fn in ("train.parquet", "val.parquet", "test.parquet"):
        p = os.path.join(DATA_DIR, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing data file: {p}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  ok: OUTPUT_DIR = {OUTPUT_DIR}")


# =====================================================================
# LOAD THESIS BASELINES (no hardcoded numbers from memory!)
# =====================================================================

def load_thesis_baselines():
    """Load ET R2 and Naive R2 per horizon from comparison_table.csv.
    These are the ground-truth numbers used in Chapter 5."""
    df = pd.read_csv(COMPARISON_TABLE_CSV)

    # find the ExtraTrees column. need to be careful here -- a naive
    # startswith("et") would also match ETS R², which is a totally
    # different model. only accept "et" as a complete token.
    et_col = None
    for c in df.columns:
        cl = c.strip().lower()
        # accepts "ET R²", "ET R2", "ET_R2", "ET" -- rejects ETS / ETC / ETA
        if cl == "et" or cl.startswith("et ") or cl.startswith("et_"):
            et_col = c; break
    if et_col is None:
        raise ValueError(
            f"can't find ExtraTrees column in {COMPARISON_TABLE_CSV}. "
            f"columns: {list(df.columns)}. "
            f"expected something like 'ET R²' or 'ET R2'.")

    # find the Naive column. same care: only accept "naive" as a token
    naive_col = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "naive" or cl.startswith("naive ") or cl.startswith("naive_"):
            naive_col = c; break
    if naive_col is None:
        raise ValueError(
            f"can't find Naive column in {COMPARISON_TABLE_CSV}. "
            f"columns: {list(df.columns)}")

    print(f"  using comparison_table.csv columns:")
    print(f"    Naive R2 -> {naive_col}")
    print(f"    ET R2    -> {et_col}")

    out = {}
    for _, row in df.iterrows():
        hz = str(row["Horizon"]).strip()
        out[hz] = {
            "thesis_naive_R2": float(row[naive_col]),
            "thesis_ET_R2":    float(row[et_col]),
        }

    # validate every horizon we plan to train is in the CSV
    missing = [hz for hz in HORIZONS if hz not in out]
    if missing:
        raise ValueError(
            f"horizons {missing} not present in {COMPARISON_TABLE_CSV}. "
            f"CSV horizons: {list(out.keys())}. "
            f"either fix the CSV or remove these from HORIZONS.")

    for hz, v in out.items():
        print(f"    {hz:6s}  thesis ET R2 = {v['thesis_ET_R2']:.4f}   "
              f"naive = {v['thesis_naive_R2']:.4f}")
    return out


# =====================================================================
# CONTAINER-FIRST SUBSAMPLING
# =====================================================================
# 1) sample K container_ids uniformly (NOT weighted by size -- diversity > volume)
# 2) FE only those containers (laptop-friendly RAM)
# 3) random-sample post-FE rows down to N_SUBSAMPLE
#
# Avoids running FE on the full 7.4M-row train set, which needed >10GB RAM
# on Vast.ai (see "rolling cache 'train' ready [10.0GB]" in run.log).

def sample_train_containers(train_raw, k, seed=42):
    """Pick k container_ids uniformly at random."""
    if "container_id" not in train_raw.columns:
        raise ValueError("train_raw has no container_id column")

    # drop NaN cids -- otherwise unique() returns NaN as a candidate, and
    # isin([NaN]) silently returns 0 rows because NaN != NaN in pandas
    cids = train_raw["container_id"].dropna().unique()
    if len(cids) == 0:
        raise ValueError("train_raw has no valid container_ids after dropna")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(cids, size=min(k, len(cids)), replace=False)

    sub = train_raw[train_raw["container_id"].isin(chosen)].copy()
    sub = sub.reset_index(drop=True)

    print(f"  sampled {len(chosen):,} of {len(cids):,} container_ids "
          f"({len(sub):,} raw rows)")
    return sub


def random_subsample_rows(df, n_target, seed=42):
    """Pick n_target rows uniformly at random; preserve temporal order."""
    if len(df) <= n_target:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n_target, replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


# =====================================================================
# FEATURE ENGINEERING WRAPPER
# =====================================================================

def fe_for_horizon(raw_df, horizon_steps, split_id):
    """Run sprint1's create_features and drop rows with NaN targets.
    Reproduces the row alignment used during the original training."""
    fe = _s1.create_features(raw_df, horizon_steps, split_id=split_id)
    fe = fe.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    return fe


def get_feature_columns(df, horizon_steps):
    """Same column list ExtraTrees uses in sprint1_Main.py."""
    return _s1.get_all_feature_columns(df, horizon_steps)


# =====================================================================
# TRAIN + EVAL FOR ONE HORIZON
# =====================================================================

def train_one_horizon(hz_name, hz_steps,
                      train_raw_sub, val_raw, test_raw,
                      thesis_baselines):
    """Run the whole pipeline for one horizon. Returns a dict of metrics."""
    print(f"\n{'='*60}")
    print(f"  horizon = {hz_name}  (steps={hz_steps})")
    print(f"{'='*60}")

    # ---- FE on the subsampled train ----
    t0 = time.time()
    print("create_features() on subsampled train ...")
    train_fe = fe_for_horizon(train_raw_sub, hz_steps, split_id="train")
    print(f"  train_fe: {len(train_fe):,} rows  ({time.time()-t0:.1f}s)")

    if len(train_fe) > N_SUBSAMPLE:
        train_fe = random_subsample_rows(train_fe, N_SUBSAMPLE, seed=SEED)
        print(f"  random subsample -> {len(train_fe):,} rows")
    else:
        # we wanted N_SUBSAMPLE but didn't have that many to begin with.
        # bump N_TRAIN_CONTAINERS if you want closer to the target.
        print(f"  WARN: only {len(train_fe):,} rows available "
              f"(target N_SUBSAMPLE={N_SUBSAMPLE:,}) -- using all")

    # ---- FE on full val and full test (do NOT subsample evaluation sets) ----
    t0 = time.time()
    print("create_features() on full val ...")
    val_fe = fe_for_horizon(val_raw, hz_steps, split_id="val")
    print(f"  val_fe:  {len(val_fe):,} rows  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("create_features() on full test ...")
    test_fe = fe_for_horizon(test_raw, hz_steps, split_id="test")
    print(f"  test_fe: {len(test_fe):,} rows  ({time.time()-t0:.1f}s)")

    # ---- FAIL-FAST: naive R2 check BEFORE training ----
    # naive_R2 doesn't need a trained model. If it diverges from the thesis
    # number, our test split is wrong; bail out before wasting training time.
    y_cpu_te  = test_fe["cpu_target"].values
    naive_te  = test_fe["naive_cpu"].values
    r2_naive_te = r2_score(y_cpu_te, naive_te)

    thesis_naive_r2 = thesis_baselines[hz_name]["thesis_naive_R2"]
    naive_gap_pp    = (r2_naive_te - thesis_naive_r2) * 100
    print(f"\n  fail-fast naive check:")
    print(f"    thesis naive R2: {thesis_naive_r2:.4f}")
    print(f"    our naive R2:    {r2_naive_te:.4f}")
    print(f"    gap:             {naive_gap_pp:+.2f} pp")
    if abs(naive_gap_pp) > NAIVE_GAP_FAIL_PP:
        raise RuntimeError(
            f"naive gap {naive_gap_pp:+.2f}pp exceeds {NAIVE_GAP_FAIL_PP}pp "
            f"on horizon {hz_name}. our test split likely differs from "
            f"the thesis test split. fix the test split before continuing.")
    print(f"    OK")

    # ---- feature columns (use train_fe for the column list) ----
    feat_cols = get_feature_columns(train_fe, hz_steps)
    print(f"\n  features: {len(feat_cols)}")

    if len(feat_cols) == 0:
        raise RuntimeError(
            f"get_all_feature_columns() returned an empty list for {hz_name}. "
            f"this should never happen with real sprint1_Main.py -- check "
            f"that the loaded sprint1 module is the one used to train the "
            f"thesis pipeline, and that train_fe has the expected lag/roll "
            f"columns (got: {[c for c in train_fe.columns if 'lag' in c or 'roll' in c][:5]})")

    # sanity: same columns must exist in val/test
    for split_name, split_df in [("val", val_fe), ("test", test_fe)]:
        missing = [c for c in feat_cols if c not in split_df.columns]
        if missing:
            raise ValueError(
                f"{split_name}_fe is missing {len(missing)} feature columns "
                f"that exist in train_fe -- e.g. {missing[:3]}. "
                f"this means create_features() produced different columns "
                f"for this split, which should never happen.")

    # ---- arrays (float32, sprint1-compatible nan/inf handling) ----
    X_tr = safe_extract(train_fe, feat_cols)
    y_tr = train_fe["cpu_residual"].values.astype(np.float32)

    X_va = safe_extract(val_fe, feat_cols)
    y_cpu_va = val_fe["cpu_target"].values
    naive_va = val_fe["naive_cpu"].values

    X_te = safe_extract(test_fe, feat_cols)
    # y_cpu_te, naive_te already extracted above for the fail-fast check

    print(f"  X_tr: {X_tr.shape} {X_tr.dtype}   "
          f"X_va: {X_va.shape}   X_te: {X_te.shape}")

    # ---- train ----
    print(f"\n  training ExtraTreesRegressor ...")
    print(f"    n_est={ET_PARAMS['n_estimators']}  "
          f"depth={ET_PARAMS['max_depth']}  "
          f"min_leaf={ET_PARAMS['min_samples_leaf']}  "
          f"max_samples={ET_PARAMS['max_samples']}")
    t0 = time.time()
    et = ExtraTreesRegressor(**ET_PARAMS)
    et.fit(X_tr, y_tr)
    train_s = time.time() - t0
    print(f"    done in {train_s:.1f}s")

    # ---- predict on val (hold-out sanity check) ----
    t0 = time.time()
    pred_va = naive_va + et.predict(X_va)   # add naive back!
    val_infer_s = time.time() - t0
    r2_val  = r2_score(y_cpu_va, pred_va)
    mae_val = mean_absolute_error(y_cpu_va, pred_va)
    print(f"\n  val:  R2={r2_val:.4f}  MAE={mae_val:.4f}  "
          f"({val_infer_s:.2f}s for {len(X_va):,} rows)")

    # ---- predict on test (compare to thesis ET R2) ----
    t0 = time.time()
    pred_te = naive_te + et.predict(X_te)
    test_infer_s = time.time() - t0
    r2_te  = r2_score(y_cpu_te, pred_te)
    mae_te = mean_absolute_error(y_cpu_te, pred_te)

    print(f"  test: R2={r2_te:.4f}  MAE={mae_te:.4f}  "
          f"({test_infer_s:.2f}s for {len(X_te):,} rows)")

    # ---- compare to thesis ET R2 ----
    thesis_et_r2 = thesis_baselines[hz_name]["thesis_ET_R2"]
    gap_pp = (r2_te - thesis_et_r2) * 100

    print(f"\n  parity check vs comparison_table.csv:")
    print(f"    thesis ET R2:    {thesis_et_r2:.4f}")
    print(f"    retrained R2:    {r2_te:.4f}")
    print(f"    gap:             {gap_pp:+.2f} pp")
    if abs(gap_pp) > PARITY_WARN_PP:
        print(f"    WARN: gap exceeds {PARITY_WARN_PP}pp threshold")
        print(f"          consider increasing N_TRAIN_CONTAINERS or N_SUBSAMPLE")

    # ---- save model + feature manifest ----
    model_path = os.path.join(OUTPUT_DIR, f"et_demo_{hz_name}.joblib")
    joblib.dump(et, model_path, compress=3)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"\n  saved: {model_path}  ({size_mb:.1f} MB)")

    feat_path = os.path.join(OUTPUT_DIR, f"et_demo_features_{hz_name}.json")
    with open(feat_path, "w") as f:
        json.dump({
            "horizon":            hz_name,
            "horizon_steps":      hz_steps,
            "n_features":         len(feat_cols),
            "feature_columns":    feat_cols,
            "et_params":          ET_PARAMS,
            "n_train_rows":       int(X_tr.shape[0]),
            "n_train_containers": N_TRAIN_CONTAINERS,
            "subsample_seed":     SEED,
            "sklearn_version":    sklearn.__version__,
            "input_dtype":        "float32",
        }, f, indent=2)
    print(f"  saved: {feat_path}")

    # ---- round-trip self-test: reload from disk and predict ----
    print(f"\n  round-trip self-test ({SELFTEST_ROWS} rows) ...")
    et_reloaded = joblib.load(model_path)
    n = min(SELFTEST_ROWS, len(X_te))
    in_mem_pred = et.predict(X_te[:n])
    reload_pred = et_reloaded.predict(X_te[:n])
    max_diff = float(np.max(np.abs(in_mem_pred - reload_pred)))
    print(f"    max |in_memory - reloaded| = {max_diff:.2e}")
    if max_diff > SELFTEST_MAX_DIFF:
        raise RuntimeError(
            f"self-test FAILED: predictions diverge after joblib roundtrip. "
            f"this means joblib serialisation lost state -- do not deploy.")
    print(f"    OK (within {SELFTEST_MAX_DIFF:.0e})")

    return {
        "horizon":         hz_name,
        "n_train_rows":    int(X_tr.shape[0]),
        "n_val_rows":      int(X_va.shape[0]),
        "n_test_rows":     int(X_te.shape[0]),
        "n_features":      len(feat_cols),
        "val_R2":          round(float(r2_val), 4),
        "val_MAE":         round(float(mae_val), 4),
        "test_R2":         round(float(r2_te), 4),
        "test_MAE":        round(float(mae_te), 4),
        "naive_R2":        round(float(r2_naive_te), 4),
        "thesis_ET_R2":    round(float(thesis_et_r2), 4),
        "gap_pp":          round(float(gap_pp), 2),
        "thesis_naive_R2": round(float(thesis_naive_r2), 4),
        "naive_gap_pp":    round(float(naive_gap_pp), 2),
        "train_seconds":   round(train_s, 1),
        "infer_ms_per_row": round(1000 * test_infer_s / len(X_te), 3),
        "selftest_max_diff": float(max_diff),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print(f"task_retrain_et_for_demo.py")
    print(f"---")
    env_check()
    load_sprint1()

    print(f"\n--- loading thesis baselines ---")
    thesis_baselines = load_thesis_baselines()

    # ---- load raw parquets ----
    print(f"\n--- loading parquets ---")
    t0 = time.time()
    train_raw = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    print(f"  train.parquet: {len(train_raw):,} rows  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    val_raw = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    print(f"  val.parquet:   {len(val_raw):,} rows  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    test_raw = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    print(f"  test.parquet:  {len(test_raw):,} rows  ({time.time()-t0:.1f}s)")

    # ---- container-level subsample of train ----
    print(f"\n--- subsampling train containers ---")
    train_raw_sub = sample_train_containers(
        train_raw, k=N_TRAIN_CONTAINERS, seed=SEED)
    del train_raw
    import gc; gc.collect()

    # ---- run all four horizons ----
    rows = []
    for hz_name, hz_steps in HORIZONS.items():
        row = train_one_horizon(
            hz_name, hz_steps,
            train_raw_sub, val_raw, test_raw,
            thesis_baselines,
        )
        rows.append(row)
        gc.collect()

    # ---- write verification CSV ----
    ver_df = pd.DataFrame(rows)
    ver_path = os.path.join(OUTPUT_DIR, "et_demo_verification.csv")
    ver_df.to_csv(ver_path, index=False)

    print(f"\n{'='*60}")
    print(f"  verification summary")
    print(f"{'='*60}")
    show_cols = ["horizon", "test_R2", "thesis_ET_R2", "gap_pp",
                 "test_MAE", "infer_ms_per_row", "train_seconds"]
    print(ver_df[show_cols].to_string(index=False))
    print(f"\nsaved: {ver_path}")

    # ---- final summary ----
    max_gap = ver_df["gap_pp"].abs().max()
    print(f"\nmax |ET gap| across horizons: {max_gap:.2f} pp")
    if max_gap > PARITY_WARN_PP:
        print(f"\nWARN: ET gap exceeds {PARITY_WARN_PP}pp on some horizon.")
        print(f"      defense answer: 'this is a deployable variant trained")
        print(f"      on a stratified subsample; Chapter 5 numbers use the")
        print(f"      full training set.'")
    else:
        print(f"OK: all horizons within {PARITY_WARN_PP}pp of thesis numbers.")


if __name__ == "__main__":
    main()


# =====================================================================
# INFERENCE HELPER (copy this into the FastAPI predictor service)
# =====================================================================
# Live demo flow:
#   1. service starts -> load et_demo_{hz}.joblib + et_demo_features_{hz}.json
#   2. on /predict request:
#        - pull last (288 + horizon_steps) timesteps from Prometheus
#        - build a DataFrame with cols: container_id, time_stamp,
#                                       cpu_util_percent, mem_util_percent
#        - call predict_one() below
#        - return naive + residual_pred
#
# !!! IMPORTANT WHEN COPY-PASTING INTO THE FASTAPI SERVICE !!!
# This function references _s1, the module loaded by load_sprint1().
# In a FastAPI service this MUST be initialized before predict_one() is
# ever called -- typically in the @app.on_event("startup") handler.
# A cleaner fix is to refactor sprint1_Main.create_features() into a
# standalone module the service imports directly, which avoids the
# fragile dynamic import in production code.
#
# Other gotchas:
# - time_stamp must be the actual unix-epoch timestamp (or whatever
#   convention train.parquet uses). create_features() derives
#   hour_sin/cos from it. Wrong time_stamp -> wrong diurnal features
#   -> silently degraded predictions.
# - History of (288 + horizon_steps + 1) rows is the MINIMUM. Without
#   24h of history, cpu_same_time_1d is NaN -> filled with cpu_lag_*
#   (sprint1's fallback) -> degraded but not catastrophic. The HPA
#   controller should still fall back to reactive mode until enough
#   history accumulates.
# - We pass split_id=None to create_features() inside predict_one()
#   on purpose -- see the comment in predict_one().

def predict_one(et_model, feature_cols, raw_df, horizon_steps):
    """Single-prediction inference helper for the FastAPI service.

    raw_df: DataFrame with at least these columns
            - container_id
            - time_stamp
            - cpu_util_percent
            - mem_util_percent
        and at least (288 + horizon_steps + 1) rows of history for the
        target container.

    Returns: float, predicted CPU at +horizon_steps timesteps.
    """
    # IMPORTANT: pass split_id=None, NOT split_id="live".
    # sprint1_Main._rolling_cache is keyed by split_id. if we passed
    # "live", request #1's data would be cached under that key and every
    # later request with new raw_df would silently get request #1's
    # rolling features back -- a quiet correctness bug. split_id=None
    # forces _compute_rolling_only(df) to run every call.
    fe = _s1.create_features(raw_df, horizon_steps, split_id=None)
    # we want a forecast for the most recent timestamp where naive_cpu is
    # defined (which is just current cpu, so almost always the last row)
    fe = fe.dropna(subset=["naive_cpu"]).reset_index(drop=True)
    if len(fe) == 0:
        raise ValueError("not enough history for feature engineering")

    row = fe.iloc[-1:]
    # match the script's training-time array extraction exactly
    X = row[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    naive = float(row["naive_cpu"].iloc[0])

    residual_pred = float(et_model.predict(X)[0])
    return naive + residual_pred
