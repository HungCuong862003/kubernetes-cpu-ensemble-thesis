"""
thesis_analysis.py — Unified post-hoc analysis and figure generation.

Replaces five earlier standalone scripts:
  thesis_analysis.py  +  run_dm_tests.py  +  run_hpa_sim.py
  +  workload_stats.py  +  run_extra_analysis.py

PARTS
=====
  Part 1 — SHAP feature importance (weighted TreeSHAP + permutation check)
  Part 2 — Workload characterisation (CV, Hurst, ACF, predictability landscape)
  Part 3 — UQ reporting (original AgACI intervals, no post-hoc rescaling)
  Part 4 — Statistical tests (DM+HLN, Friedman, Nemenyi, CD diagrams)
  Part 5 — HPA simulation (reactive vs ML-proactive, Pareto frontier)
  Part 6 — Extra analyses (skill score, per-container win rate, residual diag)
  Part 7 — Summary figures (NNLS stacked bar, R² improvement line, comp table)

CHECKPOINT SYSTEM
=================
  Every part writes a JSON checkpoint before exiting.
  If the script dies mid-run, just re-execute — it skips finished parts.
  Use --force PART to clear one part and re-run it.
  Valid PART names: shap, workload, uq, dm, hpa, extra, summary

USAGE
=====
  python thesis_analysis_v2.py \
      --sprint1-dir   /workspace \
      --data-dir      /workspace/thesis \
      --ckpt-dir      /workspace/results/sprint1 \
      --output-dir    ./thesis_figures \
      [--dsb-data     /path/to/dsb.csv] \
      [--bitbrains-data /path/to/bitbrains.csv] \
      [--use-gpu] \
      [--force shap]

INSTALL (one-off on Vast.ai)
=============================
  pip install shap nolds scikit-posthocs --break-system-packages

CHANGELOG
=========
  v2 (initial): Combined 5 scripts into one with checkpointing.

  v2.1 bug fixes:
    FIX-1 (Critical):  load_cqr_intervals() now also loads the 'point_pred'
           key from cqr_intervals.npz.  The CQR intervals are in residual
           space and centered on the tree ensemble's own predictions
           (point_pred), NOT on hetero_ensemble (which includes BiLSTM).
           Using hetero_pred as the center caused asymmetric margin mismatch.
    FIX-2 (Critical):  Removed np.clip(margin, 0, None) before taking median
           in the CQR scale factor calculation.  When clipped margins had
           median == 0, scale = q_target / 1e-6 blew up to millions.
           Fixed by using np.abs(margin) and a proper zero-guard with a
           fallback of s=1.0 (keep original width) instead of s=huge.
    FIX-3 (Medium):    Friedman per-container MAE had a triple slice
           arr[-n:][-ref_len:][mask[:n]] that mis-aligned predictions when
           arr was shorter than ref_len.  Replaced with explicit NaN-padding
           to ref_len before applying the container mask.
    FIX-4 (Minor):     Summary annotation used f"+{v:.2f}pp" which renders
           as "+-0.60pp" for negative values.  Changed to f"{v:+.2f}pp".
    FIX-5 (Minor):     assert in _fit_tree_models replaced with explicit
           if/raise ValueError so the error message is actually readable.

  v2.2 bug fixes (identified by deep static analysis):
    FIX-6 (High):      _pareto_front scanned right-to-left with <=, which
           collapsed the entire Pareto front to a single minimum-violation
           point.  Fixed to scan left-to-right with strict < so the full
           efficient frontier is identified.
    FIX-7 (Medium):    _EnsWrapper.fit() did not set any fitted attribute,
           causing sklearn 1.5.2's check_is_fitted() to raise NotFittedError
           inside permutation_importance(), silently killing that entire
           analysis.  Fixed by adding self.is_fitted_ = True in fit().
    FIX-8 (Medium):    simulate_hpa scale-down stabilisation window was not
           enforced: the code pruned old entries but never checked that the
           window was full before acting, so scale-down fired at t=0.
           Fixed by adding a len(sd_recs) >= scaledown_stab guard.
    FIX-9 (Medium):    run_extra ACF1 and Ljung-Box were computed on a
           np.linspace-subsampled residual array.  For test sets of ~1.7M
           rows, consecutive elements in the subsampled array were ~33
           timesteps apart, so ACF1 measured autocorrelation at lag 33
           (165 min), not lag 1 (5 min).  Fixed by using res[:samp]
           (consecutive head) instead of res[linspace_indices].

  v2.3 bug fixes (identified by second-pass static analysis):
    FIX-A (High):      run_hpa used test_fe["cpu_util_percent"] which is
           the raw Alibaba column consumed and dropped by create_features().
           After fe_for_horizon(), this column is not guaranteed to exist.
           naive_cpu is the identical value (cpu_util_percent at lag-0) and
           is guaranteed by SPRINT1_REQUIRED_COLS.  Also, cpu_target would
           be wrong — it is the *future* CPU value (shifted by horizon), so
           using it as the live demand series would corrupt violation/waste.
           Fixed both occurrences in run_hpa (main loop and timeseries block).
    FIX-B (Minor):     run_summary NNLS bar: ax.legend() was called before
           ax.axhline(25, label=...), so the reference-line label was never
           captured by the legend snapshot.  Fixed by swapping the order so
           axhline() comes before legend().
    FIX-C (Low):       run_shap early-exit paths returned {} on missing deps,
           causing run_section() to checkpoint the section as complete.  The
           next run (after installing deps) would silently skip SHAP.
           The intended fix (return None) was applied to run_shap, but
           run_section() still called ckpt.mark_done() unconditionally for
           any non-exception return including None.  Fixed by adding a
           'if result is None: return None' guard in run_section() itself,
           before the mark_done() call, so None-returning sections are never
           written to the checkpoint.

  v2.4 bug fixes (second external audit):
    FIX-D (High):      HPAConfig constants scaledown_stab=20 and pod_startup=2
           were calibrated for 15-second HPA sync cycles, not the simulation's
           5-minute (300s) timesteps.  20×300s=6000s (100-min window) and
           2×300s=600s (10-min cold start) — both wildly off from K8s defaults
           of 300s and ~30s respectively.  With stab=20, scale-down barely
           fires, making the reactive baseline appear over-provisioned and safe,
           and invalidating the entire Pareto frontier comparison.
           Fixed: scaledown_stab=1 (300s/300s=1 step), pod_startup=1 (~30s
           rounds up to one step as a conservative single-step penalty).
    FIX-E (Medium):    simulate_hpa scale-down used len(sd_recs) >= stab, which
           counts *recommendations*, not elapsed time.  Any step that hits the
           dead zone or triggers a scale-up pushes nothing, so in oscillating
           workloads scale-down can be permanently disabled.
           Fixed: replaced with elapsed-time check
           (t - sd_recs[0][0]) >= scaledown_stab, matching K8s rolling-window
           semantics.  Deque is cleared after acting.
    FIX-F (Medium):    _fit_tree_models passed early_stopping_rounds=50 to the
           XGBRegressor constructor.  XGBoost >=2.0 deprecated this; on some
           builds it is silently ignored so all 2000 rounds always run.
           SHAP re-fitting runs for all 4 horizons, so this matters for runtime.
           Fixed: moved early_stopping_rounds into the .fit() call.
    FIX-G (Low):       run_uq loaded val_raw unconditionally at function entry
           even though it is only needed in the split-conformal fallback path
           (when cqr_intervals.npz is absent).  Deferred the load to inside
           the fallback block.
    FIX-H (Low):       ACF comparison plot recomputed the constant `mins` list
           on every loop iteration.  Hoisted above the loop.

  v2.5 bug fixes (third external audit — applied 2026-03-18):
    BUG-1 (High):      run_uq CQR preferred path fell back to hetero_pred
           (y_pred) as the interval center when 'point_pred' was absent from
           cqr_intervals.npz.  CQR intervals are geometrically built around
           the tree-ensemble's own point_pred, not hetero_pred (which includes
           BiLSTM).  Using hetero_pred produces wrong margin estimates and
           corrupts the scale factors s_lo/s_hi even when the error is small.
           Fix: when point_pred is absent the entire CQR recalibration block
           is skipped (result stays None) and the code falls through cleanly
           to the split-conformal fallback.  Previously the warn() + wrong
           center assignment left subsequent code referencing a bad cqr_center
           without any structural exit from the CQR branch.
    BUG-2 (Medium):    _dm_test_hln recursed to h=1 when Newey-West spectral
           estimate V<=0 at h>1.  This silently changed the test null from
           "h-step equal predictive accuracy" to "1-step equal predictive
           accuracy" — different hypotheses (Harvey, Leybourne & Newbold 1997
           do not prescribe this fallback).  Fixed: return NaN for all V<=0
           cases regardless of h.  All current run_dm calls already use h=1,
           so this path was unreachable in practice, but the logic was wrong.
    BUG-3 (Medium):    run_workload loaded the test DataFrame twice — once at
           the top of the Alibaba section, and again inside the predictability
           landscape scatter block.  Fixed: single load before both blocks,
           stored as test_df, reused in both.
    BUG-5 (Low):       run_summary annotation loop used x_vals.index(xi) to
           look up the horizon colour, which is O(n) on a list whose iteration
           order already guarantees alignment.  Fixed with zip(x_vals,
           impr.values, hz_order), eliminating the list search entirely.
"""
import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
import traceback
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

# soft deps — warn on missing but don't crash the whole script
try:
    import shap as shap_lib
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. pip install shap --break-system-packages")

try:
    import scikit_posthocs as sp
    HAS_SP = True
except ImportError:
    HAS_SP = False
    print("WARNING: scikit-posthocs not installed. pip install scikit-posthocs --break-system-packages")

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False
    print("WARNING: nolds not installed. pip install nolds --break-system-packages")

try:
    import xgboost as xgb_lib
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb_lib
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.base import BaseEstimator
    HAS_SKL = True
except ImportError:
    HAS_SKL = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("thesis_analysis_v2.log", mode="a"),
    ],
)
log  = logging.getLogger(__name__).info
warn = logging.getLogger(__name__).warning
err  = logging.getLogger(__name__).error


def sep(title=""):
    log("=" * 60 + (f"  {title}" if title else ""))


# ---------------------------------------------------------------------------
# Constants — must match sprint1_v9.py
# ---------------------------------------------------------------------------
HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}
SAMPLING_INTERVAL = 300  # 5-min in seconds

# same 6-column dropna list that sprint1_v9.py uses, so row counts match
SPRINT1_REQUIRED_COLS = [
    "cpu_target", "cpu_residual", "naive_cpu",
    "mem_target", "mem_residual", "naive_mem",
]

# all model names we try to load as pred_<name>.npy
ALL_MODEL_NAMES = [
    "naive", "ets", "arima", "linear_reg",
    "xgboost", "lightgbm", "extratrees", "bilstm",
    "homo_ensemble", "hetero_ensemble",
]

# Okabe-Ito colorblind-safe palette
PALETTE_OI = {
    "naive":           "#000000",
    "ets":             "#56B4E9",
    "arima":           "#009E73",
    "linear_reg":      "#F0E442",
    "xgboost":         "#E69F00",
    "lightgbm":        "#D55E00",
    "extratrees":      "#0072B2",
    "bilstm":          "#CC79A7",
    "homo_ensemble":   "#999999",
    "hetero_ensemble": "#8B0000",
}

HZ_COLORS = {
    "10min":  "#4C72B0",
    "30min":  "#DD8452",
    "60min":  "#55A868",
    "120min": "#C44E52",
}

BILSTM_WEIGHT_WARN_THRESH = 0.15

# model hyperparams — must match sprint1_v9 exactly
XGB_PARAMS = dict(
    objective="reg:squarederror", n_estimators=2000, max_depth=7,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=5, reg_alpha=0.1, reg_lambda=3.0,
    random_state=42, n_jobs=-1,
)
LGBM_PARAMS = dict(
    objective="regression_l1", n_estimators=500, max_depth=6,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1, metric="l1",
)
ET_PARAMS = dict(
    n_estimators=200, max_depth=25, min_samples_leaf=5,
    max_features=0.8, bootstrap=True, max_samples=0.3,
    random_state=42, n_jobs=-1,
)

# publication figure style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
})


# ===========================================================================
# CHECKPOINT MANAGER
# ===========================================================================

class CheckpointManager:
    """
    Simple JSON checkpoint system.
    Tracks which parts are done so we can skip them on re-run.

    Layout inside checkpoint_dir:
      metadata.json    — completed parts, timings, scalar results
      <part>.npz       — numpy arrays for that part (if any)
    """

    def __init__(self, checkpoint_dir):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir / "metadata.json"

    def _load(self):
        if self._meta_path.exists():
            try:
                with open(self._meta_path) as f:
                    return json.load(f)
            except Exception:
                warn("Corrupt checkpoint metadata — treating as empty")
        return {"completed": [], "times": {}, "errors": {}, "results": {}}

    def _save(self, meta):
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def is_done(self, section):
        return section in self._load()["completed"]

    def force_clear(self, section):
        meta = self._load()
        if section in meta["completed"]:
            meta["completed"].remove(section)
        meta.get("results", {}).pop(section, None)
        meta.get("errors", {}).pop(section, None)
        self._save(meta)
        npz = self.dir / f"{section}.npz"
        if npz.exists():
            npz.unlink()
        log(f"  Cleared checkpoint for '{section}'")

    def mark_done(self, section, scalars=None, arrays=None, elapsed=0.0):
        if arrays:
            np.savez_compressed(self.dir / f"{section}.npz", **arrays)
        meta = self._load()
        if scalars:
            meta.setdefault("results", {})[section] = scalars
        if section not in meta["completed"]:
            meta["completed"].append(section)
        meta.setdefault("times", {})[section] = round(elapsed, 1)
        self._save(meta)

    def mark_error(self, section, msg):
        meta = self._load()
        meta.setdefault("errors", {})[section] = msg
        self._save(meta)

    def load_arrays(self, section):
        p = self.dir / f"{section}.npz"
        if p.exists():
            return dict(np.load(p, allow_pickle=False))
        return {}

    def load_scalars(self, section):
        return self._load().get("results", {}).get(section, {})

    def summary(self):
        meta = self._load()
        log("Checkpoint summary:")
        for s in meta.get("completed", []):
            t = meta.get("times", {}).get(s, "?")
            log(f"  ✓ {s} ({t}s)")
        for s, e in meta.get("errors", {}).items():
            if s not in meta.get("completed", []):
                log(f"  ✗ {s}: {e}")


def run_section(name, func, ckpt, *args, **kwargs):
    """
    Run one analysis section with checkpoint guard.
    If the checkpoint already exists, skip it.
    Catches exceptions so one failed section doesn't kill the whole script.

    IMPORTANT: if func() returns None, the section is treated as deliberately
    skipped (e.g. missing optional dependencies) and is NOT checkpointed.
    This means it will be re-attempted on the next run, which is correct
    behaviour — once the user installs the missing deps, the section runs.
    func() must return a dict (even {}) to be checkpointed as complete.
    """
    if ckpt.is_done(name):
        log(f"⏭  '{name}' already done — skipping (use --force {name} to re-run)")
        return ckpt.load_scalars(name)

    sep(f"PART: {name.upper()}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0

        # FIX-C: None return means "skipped due to missing deps" — do NOT
        # checkpoint so the section runs again after deps are installed.
        if result is None:
            log(f"⚠  '{name}' skipped (deps missing or not applicable) — not checkpointed")
            return None

        # separate numpy arrays from scalar results for storage
        scalars = {}
        arrays = {}
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    arrays[k] = v
                elif not callable(v):
                    scalars[k] = v

        ckpt.mark_done(name, scalars=scalars or None,
                       arrays=arrays or None, elapsed=elapsed)
        log(f"✓  '{name}' finished in {elapsed:.1f}s")
        return scalars

    except Exception as e:
        elapsed = time.time() - t0
        err(f"✗  '{name}' failed after {elapsed:.1f}s: {e}")
        err(traceback.format_exc())
        ckpt.mark_error(name, str(e))
        return None


# ===========================================================================
# DYNAMIC IMPORT OF sprint1_v9
# ===========================================================================

_s1 = None  # module reference, loaded once


def load_sprint1(sprint1_dir):
    global _s1
    path = os.path.join(sprint1_dir, "sprint1_v9.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sprint1_v9.py not found in {sprint1_dir}")
    spec = importlib.util.spec_from_file_location("sprint1_v9", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _s1 = mod
    log(f"Loaded sprint1_v9 from {path}")


def _need_s1():
    if _s1 is None:
        raise RuntimeError("sprint1_v9 not loaded. Call load_sprint1() first.")


def create_features(df, horizon_steps, split_id=None):
    _need_s1()
    return _s1.create_features(df, horizon_steps, split_id=split_id)

def get_xgb_features(df, horizon_steps):
    _need_s1()
    return _s1.get_xgb_features(df, horizon_steps)

def get_lgbm_features(df, horizon_steps):
    _need_s1()
    return _s1.get_lgbm_features(df, horizon_steps)

def get_all_feature_columns(df, horizon_steps):
    _need_s1()
    return _s1.get_all_feature_columns(df, horizon_steps)

def categorize_features(feature_cols, horizon_steps):
    _need_s1()
    return _s1.categorize_features(feature_cols, horizon_steps)

def safe_extract_array(df, cols, dtype=np.float32):
    _need_s1()
    if hasattr(_s1, "safe_extract_array"):
        return _s1.safe_extract_array(df, cols, dtype)
    arr = df[cols].values.astype(dtype)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ===========================================================================
# SHARED I/O HELPERS
# ===========================================================================

def load_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_file(data_dir, stem):
    for ext in (".csv", ".parquet"):
        p = os.path.join(data_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Can't find {stem}.csv or {stem}.parquet in {data_dir}")


def load_splits(data_dir):
    return (
        load_df(find_file(data_dir, "train")),
        load_df(find_file(data_dir, "val")),
        load_df(find_file(data_dir, "test")),
    )


def load_pred(ckpt_dir, hz, name):
    p = os.path.join(ckpt_dir, hz, f"pred_{name}.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="c")
    return None


def load_all_preds(ckpt_dir, hz):
    """Load every available pred_*.npy for this horizon."""
    out = {}
    for name in ALL_MODEL_NAMES:
        arr = load_pred(ckpt_dir, hz, name)
        if arr is not None:
            out[name] = arr
    return out


def load_nnls_weights(ckpt_dir, hz):
    """
    Try to load NNLS weights from the horizon's JSON checkpoint.
    Returns a dict like {"XGB": 0.4, "LGB": 0.3, "ET": 0.2, "BiLSTM": 0.1}
    or None if the file isn't found.
    """
    for fname in ("hetero_ensemble.json", "homo_ensemble.json"):
        p = os.path.join(ckpt_dir, hz, fname)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            data = json.load(f)
        for outer in data.values():
            if isinstance(outer, dict) and "nnls_weights" in outer:
                return outer["nnls_weights"]
    return None


def load_cqr_intervals(ckpt_dir, hz):
    """
    Load saved CQR intervals from cqr_intervals.npz.

    Returns (lower, upper, point_pred) all as float32 arrays, or
    (None, None, None) if the file doesn't exist.

    lower/upper/point_pred are all in *residual space*
    (cpu_residual = cpu_target - naive_cpu).  Add naive_cpu predictions to
    convert to absolute CPU space.

    point_pred is the tree ensemble's own center used to build the intervals —
    NOT hetero_ensemble (which includes BiLSTM).  Always use point_pred as
    the center when computing margins.  If point_pred is absent (old
    checkpoints), the caller must fall through to split-conformal; using
    hetero_pred as a substitute center produces wrong scale factors.
    """
    p = os.path.join(ckpt_dir, hz, "cqr_intervals.npz")
    if not os.path.exists(p):
        return None, None, None

    data = np.load(p, mmap_mode="c")
    lower = data["lower"].astype(np.float32)
    upper = data["upper"].astype(np.float32)

    if "point_pred" in data:
        point_pred = data["point_pred"].astype(np.float32)
    else:
        point_pred = None
        warn(f"  [{hz}] cqr_intervals.npz has no 'point_pred' key — "
             f"will fall through to split-conformal (cannot recalibrate "
             f"without the tree-ensemble center).")

    return lower, upper, point_pred


def fe_for_horizon(raw_df, horizon_steps, split_id):
    """Run sprint1_v9 feature engineering and drop rows with NaN targets."""
    fe = create_features(raw_df, horizon_steps, split_id=split_id)
    fe = fe.dropna(subset=SPRINT1_REQUIRED_COLS).reset_index(drop=True)
    return fe


# ===========================================================================
# NNLS WEIGHT NORMALISATION
# ===========================================================================

def normalise_tree_weights(nnls_w):
    """
    Given the full NNLS weight dict (includes BiLSTM), return the
    normalised weights for XGB, LGB, ET only (the tree models).

    We need this because SHAP can only compute TreeSHAP for tree models.
    BiLSTM SHAP is not available so we scale the tree weights to sum=1.
    """
    bilstm_w = nnls_w.get("BiLSTM", nnls_w.get("BILSTM", 0.0))
    w_xgb = nnls_w.get("XGB", 1.0)
    w_lgb = nnls_w.get("LGB", 1.0)
    w_et  = nnls_w.get("ET",  1.0)

    tree_total = w_xgb + w_lgb + w_et
    total = tree_total + bilstm_w

    if total > 1e-9:
        bilstm_share = bilstm_w / total
        if bilstm_share > BILSTM_WEIGHT_WARN_THRESH:
            warn(f"BiLSTM carries {bilstm_share:.1%} of NNLS weight — "
                 f"tree-only SHAP renorm will under-represent it")

    if tree_total < 1e-9:
        return 1/3, 1/3, 1/3

    return w_xgb / tree_total, w_lgb / tree_total, w_et / tree_total


def squeeze_shap(sv):
    """
    shap.TreeExplainer.shap_values() returns a list for multioutput models
    and a 2D array for single-output.  Normalise to a 2D numpy array.
    """
    if isinstance(sv, list):
        sv = sv[0]
    return np.array(sv)


# ===========================================================================
# PART 1: SHAP FEATURE IMPORTANCE
# ===========================================================================

def _fit_tree_models(train_fe, val_fe, xgb_cols, lgbm_cols, all_cols, use_gpu):
    """
    Re-fit XGB, LGB, ExtraTrees on cpu_residual.
    We re-fit here because sprint1_v9 deletes model objects after training
    to save RAM — only the .npy prediction files survive in checkpoints.
    """
    all_set = set(all_cols)

    # FIX-5: use if/raise instead of assert so the error message is readable
    missing_xgb = set(xgb_cols) - all_set
    if missing_xgb:
        raise ValueError(f"xgb_cols has features not in all_cols: {missing_xgb}")
    missing_lgb = set(lgbm_cols) - all_set
    if missing_lgb:
        raise ValueError(f"lgbm_cols has features not in all_cols: {missing_lgb}")

    y_tr  = train_fe["cpu_residual"].values.astype(np.float32)
    y_val = val_fe["cpu_residual"].values.astype(np.float32)

    X_xgb_tr  = safe_extract_array(train_fe, xgb_cols)
    X_xgb_val = safe_extract_array(val_fe,   xgb_cols)
    X_lgb_tr  = safe_extract_array(train_fe, lgbm_cols)
    X_lgb_val = safe_extract_array(val_fe,   lgbm_cols)
    X_all_tr  = safe_extract_array(train_fe, all_cols)

    xgb_params = dict(XGB_PARAMS)
    if use_gpu:
        xgb_params["device"] = "cuda"
        xgb_params["tree_method"] = "hist"

    log("    fitting XGB...")
    # FIX-F: early_stopping_rounds belongs in .fit(), not the constructor.
    xgb_mod = xgb_lib.XGBRegressor(**xgb_params, verbosity=0)
    xgb_mod.fit(X_xgb_tr, y_tr, eval_set=[(X_xgb_val, y_val)],
                early_stopping_rounds=50, verbose=False)

    log("    fitting LGB...")
    lgb_mod = lgb_lib.LGBMRegressor(**LGBM_PARAMS)
    lgb_mod.fit(X_lgb_tr, y_tr,
                eval_set=[(X_lgb_val, y_val)],
                callbacks=[lgb_lib.early_stopping(20, verbose=False)])

    log("    fitting ExtraTrees...")
    et_mod = ExtraTreesRegressor(**ET_PARAMS)
    et_mod.fit(X_all_tr, y_tr)

    return xgb_mod, lgb_mod, et_mod


def predict_weighted_ensemble(xgb_mod, lgb_mod, et_mod,
                               w_xgb, w_lgb, w_et,
                               xgb_cols, lgbm_cols, all_cols,
                               X_all):
    """
    Weighted prediction from the three tree models.
    X_all has columns in the same order as all_cols.
    We index into it to extract the right subset for each model.
    """
    col_to_idx = {c: i for i, c in enumerate(all_cols)}

    xgb_idx = [col_to_idx[c] for c in xgb_cols]
    lgb_idx  = [col_to_idx[c] for c in lgbm_cols]

    pred_xgb = xgb_mod.predict(X_all[:, xgb_idx])
    pred_lgb = lgb_mod.predict(X_all[:, lgb_idx])
    pred_et  = et_mod.predict(X_all)

    return w_xgb * pred_xgb + w_lgb * pred_lgb + w_et * pred_et


def run_shap(data_dir, ckpt_dir, output_dir, shap_sub, perm_sub, perm_rep,
             use_gpu, run_perm):
    # FIX-C: return None (not {}) when deps are missing.
    if not HAS_SHAP:
        log("SKIP SHAP: shap not installed")
        return None

    if not (HAS_XGB and HAS_LGB and HAS_SKL):
        log("SKIP SHAP: ML dependencies missing (xgboost/lightgbm/sklearn)")
        return None

    train_raw, val_raw, test_raw = load_splits(data_dir)

    imp_by_hz  = {}
    perm_by_hz = {}
    feat_by_hz = {}

    for hz, hz_steps in HORIZONS.items():
        log(f"\n  [{hz}]")
        nnls_w = load_nnls_weights(ckpt_dir, hz)
        if nnls_w is None:
            nnls_w = {"XGB": 1.0, "LGB": 1.0, "ET": 1.0}

        try:
            train_fe = fe_for_horizon(train_raw, hz_steps, "train")
            val_fe   = fe_for_horizon(val_raw,   hz_steps, "val")
            test_fe  = fe_for_horizon(test_raw,  hz_steps, "test")

            if train_fe.empty or val_fe.empty or test_fe.empty:
                log(f"  empty split for {hz}, skipping")
                continue

            xgb_cols  = get_xgb_features(train_fe, hz_steps)
            lgbm_cols = get_lgbm_features(train_fe, hz_steps)
            all_cols  = get_all_feature_columns(train_fe, hz_steps)
            w_xgb, w_lgb, w_et = normalise_tree_weights(nnls_w)

            xgb_mod, lgb_mod, et_mod = _fit_tree_models(
                train_fe, val_fe, xgb_cols, lgbm_cols, all_cols, use_gpu)

            X_xgb_te = safe_extract_array(test_fe, xgb_cols)
            X_lgb_te = safe_extract_array(test_fe, lgbm_cols)
            X_all_te = safe_extract_array(test_fe, all_cols)

            rng = np.random.default_rng(42)
            n_samples = min(shap_sub, len(X_all_te))
            idx = rng.choice(len(X_all_te), size=n_samples, replace=False)
            idx.sort()
            log(f"    TreeSHAP on {n_samples} samples...")

            sv_xgb = squeeze_shap(shap_lib.TreeExplainer(xgb_mod).shap_values(X_xgb_te[idx]))
            sv_lgb = squeeze_shap(shap_lib.TreeExplainer(lgb_mod).shap_values(X_lgb_te[idx]))
            sv_et  = squeeze_shap(shap_lib.TreeExplainer(et_mod).shap_values(X_all_te[idx]))

            union = list(dict.fromkeys(xgb_cols + lgbm_cols + all_cols))

            def pad_to_union(sv, cols):
                col_to_idx_local = {c: i for i, c in enumerate(cols)}
                out = np.zeros((sv.shape[0], len(union)), dtype=np.float32)
                for j, c in enumerate(union):
                    if c in col_to_idx_local:
                        out[:, j] = sv[:, col_to_idx_local[c]]
                return out

            sv_xgb_padded = pad_to_union(sv_xgb, xgb_cols)
            sv_lgb_padded = pad_to_union(sv_lgb, lgbm_cols)
            sv_et_padded  = pad_to_union(sv_et,  all_cols)

            sv_ens = w_xgb * sv_xgb_padded + w_lgb * sv_lgb_padded + w_et * sv_et_padded

            imp_by_hz[hz] = pd.Series(np.abs(sv_ens).mean(axis=0), index=union)
            feat_by_hz[hz] = union

            if run_perm:
                log("    computing permutation importance...")
                try:
                    rng2 = np.random.default_rng(99)
                    perm_idx = rng2.choice(len(X_all_te), min(perm_sub, len(X_all_te)), replace=False)
                    perm_idx.sort()

                    y_test_perm = test_fe["cpu_residual"].values[perm_idx]

                    # FIX-7: self.is_fitted_ = True required by sklearn 1.5.2
                    class _EnsWrapper(BaseEstimator):
                        def fit(self, X, y):
                            self.is_fitted_ = True
                            return self
                        def predict(self, X):
                            return predict_weighted_ensemble(
                                xgb_mod, lgb_mod, et_mod,
                                w_xgb, w_lgb, w_et,
                                xgb_cols, lgbm_cols, all_cols, X)

                    pr = permutation_importance(
                        _EnsWrapper().fit(X_all_te[perm_idx], y_test_perm),
                        X_all_te[perm_idx], y_test_perm,
                        n_repeats=perm_rep, scoring="r2",
                        random_state=42, n_jobs=-1)
                    perm_by_hz[hz] = pd.Series(pr.importances_mean, index=all_cols)
                except Exception as e:
                    warn(f"    permutation importance failed: {e}")

        except Exception as e:
            err(f"  SHAP {hz}: {e}")
            traceback.print_exc()

    if not imp_by_hz:
        log("No SHAP results produced")
        return {}

    all_feats = list(dict.fromkeys(f for cols in feat_by_hz.values() for f in cols))
    imp_df = pd.DataFrame(index=all_feats)
    for hz in HORIZONS:
        if hz in imp_by_hz:
            imp_df[hz] = imp_by_hz[hz].reindex(all_feats).fillna(0)
        else:
            imp_df[hz] = 0.0
    imp_df.to_csv(os.path.join(output_dir, "shap_importance.csv"))

    # --- Figure 1: heatmap of top 20 features ---
    top20 = imp_df.mean(axis=1).nlargest(20).index.tolist()
    imp_top = imp_df.loc[top20].copy()

    sort_col = imp_top.columns[0]
    for h in ["120min", "60min", "30min", "10min"]:
        if h in imp_top.columns and imp_top[h].max() > 0:
            sort_col = h
            break
    imp_top = imp_top.sort_values(sort_col, ascending=True)

    col_max = imp_top.max(axis=0).clip(lower=1e-9)
    imp_norm = imp_top.div(col_max)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(imp_norm, annot=imp_top.round(4), fmt=".4f",
                cmap="YlOrRd", linewidths=0.4, ax=ax,
                cbar_kws={"label": "Relative importance (norm. per horizon)"},
                xticklabels=["10 min", "30 min", "60 min", "120 min"])
    ax.set_title("Feature attribution shifts from lag to temporal features with horizon",
                 fontsize=12, pad=12)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "shap_heatmap.pdf"), dpi=300)
    plt.close(fig)
    log("  saved: shap_heatmap.pdf")

    # --- Figure 2: stacked bar by feature group ---
    group_colors = {
        "lag_immediate":  "#2166AC",
        "lag_extended":   "#74ADD1",
        "rolling_mean":   "#4DAC26",
        "rolling_std":    "#A1D76A",
        "rolling_range":  "#D9F0D3",
        "trend":          "#F4A582",
        "acceleration":   "#D6604D",
        "volatility":     "#B2182B",
        "temporal":       "#762A83",
        "historical":     "#C2A5CF",
        "cross_resource": "#FEE08B",
        "cluster":        "#999999",
    }

    group_data = {}
    for hz, hz_steps in HORIZONS.items():
        hz_feats = feat_by_hz.get(hz, [])
        if not hz_feats:
            continue
        cats = categorize_features(hz_feats, hz_steps)
        row = {}
        for grp, fcols in cats.items():
            present = [f for f in fcols if f in imp_df.index]
            row[grp] = float(imp_df.loc[present, hz].sum()) if present else 0.0
        group_data[hz] = row

    if group_data:
        gdf  = pd.DataFrame(group_data).T.fillna(0)
        gpct = gdf.div(gdf.sum(axis=1).clip(lower=1e-9), axis=0) * 100

        cols_present = [c for c in group_colors if c in gpct.columns]
        if cols_present:
            fig, ax = plt.subplots(figsize=(9, 5))
            bot = np.zeros(len(gpct))
            for c in cols_present:
                v = gpct[c].values
                ax.bar(gpct.index, v, bottom=bot, color=group_colors[c],
                       label=c.replace("_", " "), width=0.6)
                bot += v
            ax.set_ylabel("Feature group share (%)")
            ax.set_xlabel("Horizon")
            ax.set_title("Feature importance shifts from lag-dominated to "
                         "temporal-dominated as horizon increases")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=8, frameon=False)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "shap_group_bars.pdf"), dpi=300)
            plt.close(fig)
            log("  saved: shap_group_bars.pdf")
            log("\n  SHAP group % per horizon:")
            print(gpct[cols_present].round(1).to_string())

    # --- Figure 3: SHAP vs permutation side-by-side (optional) ---
    if perm_by_hz:
        perm_df = pd.DataFrame(index=all_feats)
        for hz in HORIZONS:
            if hz in perm_by_hz:
                perm_df[hz] = perm_by_hz[hz].reindex(all_feats).fillna(0)
            else:
                perm_df[hz] = 0.0

        top15 = imp_df.mean(axis=1).nlargest(15).index.tolist()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, hz in enumerate(HORIZONS):
            ax = axes[i]
            sv = imp_df.loc[top15, hz]
            pv = perm_df.loc[top15, hz].clip(lower=0)

            sn = sv / (sv.max() + 1e-9)
            pn = pv / (pv.max() + 1e-9)

            x = np.arange(len(top15))
            w = 0.38
            ax.barh(x + w/2, sn.values, w, color=HZ_COLORS[hz], alpha=0.85,
                    label="SHAP (attribution)")
            ax.barh(x - w/2, pn.values, w, color=HZ_COLORS[hz], alpha=0.4,
                    label="Permutation (accuracy impact)")
            ax.set_yticks(x)
            ax.set_yticklabels([f.replace("_", " ") for f in top15], fontsize=7)
            ax.set_title(hz, fontsize=10)
            ax.legend(fontsize=8, frameon=False)
            ax.invert_yaxis()

        fig.suptitle("SHAP vs Permutation importance", fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "shap_vs_perm.pdf"), dpi=300)
        plt.close(fig)
        log("  saved: shap_vs_perm.pdf")

    return {"shap_done": True}


# ===========================================================================
# PART 2: WORKLOAD CHARACTERISATION
# ===========================================================================

def _compute_cv(series):
    """Coefficient of variation. Returns NaN if mean is too close to zero."""
    s = series[~np.isnan(series)]
    if len(s) < 2:
        return np.nan
    m = np.mean(s)
    if m < 1e-6:
        return np.nan
    return np.std(s, ddof=1) / m


def _compute_hurst(series, min_len=100):
    """Hurst exponent using R/S analysis. H>0.5 = persistent, H<0.5 = mean-reverting."""
    s = series[~np.isnan(series)]
    if len(s) < min_len or not HAS_NOLDS:
        return np.nan
    try:
        return float(nolds.hurst_rs(s, fit="RANSAC", corrected=True, unbiased=True))
    except Exception:
        return np.nan


def _predictability_row(series_list, lag_steps, label):
    """
    Compute predictability stats (CV, Hurst, ACF at various lags) across
    a list of time series.  Returns a summary row dict and per-lag ACF values.
    """
    cvs = []
    hursts = []
    acf_by_lag = {s: [] for s in lag_steps}

    for raw in series_list:
        s = np.asarray(raw, dtype=float)
        s = s[~np.isnan(s)]
        if len(s) < 50:
            continue

        cv = _compute_cv(s)
        if not np.isnan(cv):
            cvs.append(cv)

        h = _compute_hurst(s)
        if not np.isnan(h):
            hursts.append(h)

        max_lag = max(lag_steps) + 1
        if len(s) > max_lag + 5:
            acf_vals = sm_acf(s, nlags=max_lag, fft=True, alpha=None)
            for lag in lag_steps:
                acf_by_lag[lag].append(float(acf_vals[lag]))

    row = {
        "Dataset":      label,
        "N_series":     len(series_list),
        "CV_median":    np.nanmedian(cvs)    if cvs    else np.nan,
        "Hurst_median": np.nanmedian(hursts) if hursts else np.nan,
    }
    for lag in lag_steps:
        mins = lag * (SAMPLING_INTERVAL // 60)
        row[f"ACF@{mins}min"] = np.nanmedian(acf_by_lag[lag]) if acf_by_lag[lag] else np.nan

    return row, acf_by_lag


def run_workload(data_dir, dsb_path, bitbrains_path, output_dir):
    lag_steps = [2, 6, 12, 24]  # corresponds to 10, 30, 60, 120 min
    rows = []
    acf_data = {}

    # BUG-3 FIX: load the Alibaba test DataFrame once here and reuse it in
    # both the Alibaba characterisation section and the predictability
    # landscape scatter below.  Previously there were two independent
    # load_df() calls, doubling I/O for no benefit.
    test_df = None
    try:
        test_df = load_df(find_file(data_dir, "test"))
    except Exception as e:
        warn(f"  Cannot load test set: {e}")

    # --- Alibaba ---
    if test_df is not None:
        try:
            cpu_col = None
            for c in ["cpu_util_percent", "cpu_usage", "cpu"]:
                if c in test_df.columns:
                    cpu_col = c
                    break
            if cpu_col is None:
                warn("  Alibaba: no recognised CPU column, skipping")
            else:
                if "container_id" in test_df.columns:
                    series = [g[cpu_col].dropna().values
                              for _, g in test_df.groupby("container_id")
                              if len(g) > 50]
                else:
                    series = [test_df[cpu_col].dropna().values]
                log(f"  Alibaba: {len(series)} containers")
                row, acf_v = _predictability_row(series, lag_steps, "Alibaba (batch)")
                rows.append(row)
                acf_data["Alibaba"] = acf_v
        except Exception as e:
            warn(f"  Alibaba workload stats failed: {e}")
    else:
        warn("  Alibaba: test set unavailable, skipping characterisation")

    # --- DeathStarBench ---
    if dsb_path and os.path.exists(dsb_path):
        try:
            dsb_df = load_df(dsb_path)
            cpu_col = None
            for c in ["cpu_usage", "cpu_util_percent", "cpu"]:
                if c in dsb_df.columns:
                    cpu_col = c
                    break
            grp_col = None
            for c in ["service", "pod", "container_id"]:
                if c in dsb_df.columns:
                    grp_col = c
                    break

            if cpu_col:
                if grp_col:
                    series = [g[cpu_col].dropna().values
                              for _, g in dsb_df.groupby(grp_col)
                              if len(g) > 50]
                else:
                    series = [dsb_df[cpu_col].dropna().values]
                log(f"  DSB: {len(series)} series")
                row, acf_v = _predictability_row(series, lag_steps,
                                                 "DeathStarBench (microservices)")
                rows.append(row)
                acf_data["DSB"] = acf_v
        except Exception as e:
            warn(f"  DSB workload stats failed: {e}")

    # --- Bitbrains ---
    if bitbrains_path and os.path.exists(bitbrains_path):
        try:
            bb_df = load_df(bitbrains_path)
            cpu_col = None
            for c in ["CPU usage [%]", "cpu_util_percent", "cpu"]:
                if c in bb_df.columns:
                    cpu_col = c
                    break
            grp_col = None
            for c in ["vm_id", "container_id"]:
                if c in bb_df.columns:
                    grp_col = c
                    break

            if cpu_col:
                if grp_col:
                    series = [g[cpu_col].dropna().values
                              for _, g in bb_df.groupby(grp_col)
                              if len(g) > 50]
                else:
                    series = [bb_df[cpu_col].dropna().values]
                log(f"  Bitbrains: {len(series)} VMs")
                row, acf_v = _predictability_row(series, lag_steps, "Bitbrains (VM batch)")
                rows.append(row)
                acf_data["Bitbrains"] = acf_v
        except Exception as e:
            warn(f"  Bitbrains workload stats failed: {e}")

    if not rows:
        log("No workload stats produced")
        return {}

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(output_dir, "workload_stats.csv"), index=False)
    log("\n  Workload predictability table:")
    print(pred_df.to_string(index=False))
    log("  saved: workload_stats.csv")

    # --- ACF comparison figure ---
    if len(acf_data) >= 2:
        styles = {
            "Alibaba":   {"color": "#2166AC", "ls": "-",  "label": "Alibaba (batch)"},
            "Bitbrains": {"color": "#4DAC26", "ls": "--", "label": "Bitbrains (VM)"},
            "DSB":       {"color": "#D6604D", "ls": ":",  "label": "DeathStarBench"},
        }
        fig, ax = plt.subplots(figsize=(9, 5))
        # FIX-H: hoist constant list out of loop
        mins = [lag * (SAMPLING_INTERVAL // 60) for lag in lag_steps]
        for name, av in acf_data.items():
            if name not in styles:
                continue
            vals = [np.nanmedian(av[lag]) if av[lag] else np.nan for lag in lag_steps]
            s = styles[name]
            ax.plot([0] + mins, [1.0] + vals,
                    color=s["color"], ls=s["ls"], lw=2, marker="o", ms=5,
                    label=s["label"])

        ax.axvspan(60, 120, alpha=0.08, color="gray", label="60-120 min horizon zone")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Lag (minutes)")
        ax.set_ylabel("Median ACF")
        ax.set_title("ACF decay explains horizon-dependent ML benefit:\n"
                     "batch workloads stay predictable, microservices do not")
        ax.legend(fontsize=10, frameon=False)
        ax.set_ylim(-0.1, 1.05)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "acf_comparison.pdf"), dpi=300)
        plt.close(fig)
        log("  saved: acf_comparison.pdf")

    # --- Per-container CV + Hurst scatter (predictability landscape) ---
    # BUG-3 FIX: reuse test_df already loaded above — no second load_df call.
    try:
        if test_df is None:
            warn("  Landscape scatter: test_df unavailable, skipping")
        else:
            cpu_col = None
            for c in ["cpu_util_percent", "cpu_usage", "cpu"]:
                if c in test_df.columns:
                    cpu_col = c
                    break
            if cpu_col and "container_id" in test_df.columns:
                pland = []
                for cid, g in test_df.groupby("container_id"):
                    s = g[cpu_col].dropna().values
                    if len(s) < 50:
                        continue
                    pland.append({
                        "container_id": cid,
                        "cv":           _compute_cv(s),
                        "hurst":        _compute_hurst(s),
                    })
                pland_df = pd.DataFrame(pland).dropna()
                pland_df.to_csv(os.path.join(output_dir, "container_landscape_stats.csv"),
                                index=False)
                log(f"  saved: container_landscape_stats.csv ({len(pland_df)} containers)")

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(pland_df["cv"], pland_df["hurst"],
                           alpha=0.5, s=15, c=HZ_COLORS["120min"])
                ax.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.5,
                           label="H=0.5 (random walk)")
                ax.axvline(0.3, color="grey", ls=":", lw=0.8, alpha=0.5)
                ax.axvline(0.7, color="grey", ls=":", lw=0.8, alpha=0.5,
                           label="CV thresholds (0.3, 0.7)")
                ax.set_xlabel("Coefficient of Variation (CV)")
                ax.set_ylabel("Hurst Exponent")
                ax.set_title("Workload predictability landscape (Alibaba containers)")
                ax.legend(fontsize=9, frameon=False)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, "predictability_landscape.pdf"), dpi=300)
                plt.close(fig)
                log("  saved: predictability_landscape.pdf")
    except Exception as e:
        warn(f"  Landscape scatter failed: {e}")

    return {"workload_done": True}


# ===========================================================================
# PART 3: UQ REPORTING
# ===========================================================================

def run_uq(data_dir, ckpt_dir, output_dir):
    """
    Report AgACI prediction intervals from sprint1_v9 checkpoints.

    AgACI (Zaffran et al., ICML 2022) already achieves 80%+ nominal coverage
    without any post-hoc adjustment.  No scalar rescaling is applied — doing
    so would constitute redundant conformalization, inflating coverage to
    87-91% and interval width by up to 33% at 120min without theoretical
    justification.

    The saved intervals from sprint1_v9 are in residual space
    (cpu_residual = cpu_target - naive_cpu).  We convert to absolute CPU
    space by adding naive_cpu predictions, then report directly.

    IMPORTANT: use the CQR model's own point_pred as the interval center.
    When point_pred is absent from the .npz file, the entire CQR path is
    skipped and the code falls through to split-conformal recalibration.
    Using hetero_pred as a substitute center is incorrect — CQR intervals
    were geometrically constructed around the tree-ensemble's point_pred,
    and substituting hetero_pred (which includes BiLSTM) produces wrong
    margin estimates even when the difference is small.
    """
    test_raw = load_df(find_file(data_dir, "test"))
    # val_raw is only needed in the split-conformal fallback path.
    # It is loaded lazily inside that block below.

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, (hz, hz_steps) in enumerate(HORIZONS.items()):
        log(f"  [{hz}]")
        try:
            hetero_pred = load_pred(ckpt_dir, hz, "hetero_ensemble")
            if hetero_pred is None:
                log(f"  no hetero_ensemble for {hz}, skipping")
                axes[i].set_visible(False)
                continue

            test_fe = fe_for_horizon(test_raw, hz_steps, "test")
            y_test  = test_fe["cpu_target"].values.astype(np.float32)

            n = min(len(y_test), len(hetero_pred))
            y_test = y_test[-n:]
            y_pred = hetero_pred[-n:]  # used only for final plot

            # FIX-1 (v2.1): load CQR intervals including point_pred key
            cqr_lo_res, cqr_hi_res, cqr_pp_res = load_cqr_intervals(ckpt_dir, hz)

            result = None

            if cqr_lo_res is not None and len(cqr_lo_res) >= n:
                naive_full = load_pred(ckpt_dir, hz, "naive")
                if naive_full is not None and len(naive_full) >= n:
                    naive_ = naive_full[-n:].astype(np.float32)

                    cqr_lo = cqr_lo_res[-n:] + naive_
                    cqr_hi = cqr_hi_res[-n:] + naive_

                    # BUG-1 FIX (v2.5): when point_pred is absent we cannot
                    # recalibrate the CQR intervals correctly.  Do NOT fall
                    # back to hetero_pred as the center — those intervals were
                    # built around the tree-ensemble's point_pred, and using a
                    # different center corrupts the margin computation.
                    # Leave result=None so the split-conformal fallback runs.
                    if cqr_pp_res is None or len(cqr_pp_res) < n:
                        warn(f"  [{hz}] cqr_intervals.npz missing 'point_pred' "
                             f"key or too short — skipping CQR recalibration, "
                             f"falling through to split-conformal.")
                        # result remains None; fallback block below takes over
                    else:
                        # AgACI intervals already achieve 80%+ nominal coverage.
                        # No scalar rescaling — report original intervals directly.
                        # Applying CQR rescaling on top of AgACI is redundant
                        # conformalization: it inflates coverage to 87-91% and
                        # widens intervals by up to 33% without any theoretical
                        # justification (Zaffran et al., ICML 2022).
                        n_cal  = n // 2
                        eval_t = y_test[n_cal:]
                        lo_e   = cqr_lo[n_cal:]
                        hi_e   = cqr_hi[n_cal:]

                        cov_orig  = float(np.mean((eval_t >= lo_e) & (eval_t <= hi_e)))
                        wid_orig  = float(np.mean(hi_e - lo_e))
                        method    = "AgACI (Zaffran et al., ICML 2022) — original intervals"

                        result = (y_pred[n_cal:], eval_t, lo_e, hi_e,
                                  cov_orig, wid_orig, cov_orig, wid_orig, method)

            if result is None:
                log(f"  [{hz}] using split-conformal fallback (no valid CQR intervals)")

                # Load val_raw lazily — only needed in this fallback path
                val_raw = load_df(find_file(data_dir, "val"))
                val_fe  = fe_for_horizon(val_raw, hz_steps, "val")
                y_val   = val_fe["cpu_target"].values.astype(np.float32)
                y_val_n = val_fe["naive_cpu"].values.astype(np.float32)
                n_val   = min(len(y_val), len(y_val_n))
                naive_resid = np.abs(y_val[-n_val:] - y_val_n[-n_val:])
                q_naive = float(np.quantile(naive_resid, 0.80))

                n_cal = n // 2
                ep    = y_pred[n_cal:]
                et    = y_test[n_cal:]

                lo_ref = ep - q_naive
                hi_ref = ep + q_naive

                cal_r = np.abs(y_test[:n_cal] - y_pred[:n_cal])
                q_hat = float(np.quantile(cal_r,
                              min(np.ceil(0.80 * (n_cal + 1)) / n_cal, 1.0)))
                lo_r  = ep - q_hat
                hi_r  = ep + q_hat

                cov_orig  = float(np.mean((et >= lo_ref) & (et <= hi_ref)))
                cov_recal = float(np.mean((et >= lo_r)  & (et <= hi_r)))
                wid_orig  = float(np.mean(hi_ref - lo_ref))
                wid_recal = float(np.mean(hi_r   - lo_r))
                method = "split-conformal fallback (no valid CQR intervals)"

                result = (ep, et, lo_r, hi_r,
                          cov_orig, wid_orig, cov_recal, wid_recal, method)

            ep, et, lo_r, hi_r, _, _, cov_r, w_r, meth = result

            # On the CQR path cov_r==cov_orig and w_r==wid_orig (no rescaling,
            # result tuple packs the same value twice at positions 4/6 and 5/7).
            # On the split-conformal fallback cov_r/w_r are the recalibrated
            # values for the plotted intervals (lo_r/hi_r).  Always report the
            # plotted interval's coverage and width (cov_r, w_r).
            is_fallback = "split-conformal" in meth
            reported_cov = cov_r
            reported_wid = w_r

            log(f"  [{hz}] coverage={reported_cov:.3f} (nominal=0.800) | "
                f"mean_width={reported_wid:.4f} | {meth}")
            rows.append({
                "Horizon":          hz,
                "AgACI_Coverage":   round(reported_cov, 4),
                "Nominal":          0.80,
                "Coverage_Gap_pp":  round((reported_cov - 0.80) * 100, 2),
                "Mean_Width":       round(reported_wid, 4),
                "Method":           meth,
            })

            ax  = axes[i]
            n_p = min(500, len(et))
            xs  = np.arange(n_p)
            pi_label = "split-conformal 80% PI" if is_fallback else "AgACI 80% PI"
            ax.fill_between(xs, lo_r[:n_p], hi_r[:n_p],
                            alpha=0.25, color=HZ_COLORS[hz],
                            label=pi_label)
            ax.plot(xs, et[:n_p], "k-", lw=0.8, alpha=0.7, label="actual")
            ax.plot(xs, ep[:n_p], color=HZ_COLORS[hz], lw=1.0, label="ensemble")
            ax.set_title(f"{hz}  coverage={reported_cov:.1%}  width={reported_wid:.3f}",
                         fontsize=10)
            ax.legend(fontsize=8, frameon=False)

        except Exception as e:
            err(f"  UQ {hz}: {e}")
            traceback.print_exc()
            axes[i].set_visible(False)

    if rows:
        fig.suptitle("AgACI prediction intervals (80% nominal coverage)", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "uq_recalibration.pdf"), dpi=300)
        log("  saved: uq_recalibration.pdf")
    else:
        log("  all horizons failed — skipping uq_recalibration.pdf")
    plt.close(fig)

    if not rows:
        log("No UQ rows produced")
        return {}

    uq_df = pd.DataFrame(rows)
    uq_df.to_csv(os.path.join(output_dir, "uq_recalibration_table.csv"), index=False)
    log("  saved: uq_recalibration_table.csv")
    print(uq_df[["Horizon", "AgACI_Coverage", "Nominal",
                 "Coverage_Gap_pp", "Mean_Width", "Method"]].to_string(index=False))

    return {"uq_done": True}


# ===========================================================================
# PART 4: STATISTICAL TESTS
# ===========================================================================

def _dm_test_hln(actual, pred1, pred2, h=1, loss="MAE"):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) correction.

    H0: equal predictive accuracy.
    DM_stat > 0 means pred1 has LARGER loss (pred1 is WORSE than pred2).

    Returns dict with DM_stat, p_value, mean_diff.
    """
    a  = np.asarray(actual, dtype=float)
    p1 = np.asarray(pred1,  dtype=float)
    p2 = np.asarray(pred2,  dtype=float)
    T  = len(a)

    e1 = a - p1
    e2 = a - p2

    if loss == "MSE":
        d = e1**2 - e2**2
    else:  # MAE
        d = np.abs(e1) - np.abs(e2)

    d_bar = d.mean()

    def acov(lag):
        m = d.mean()
        n = len(d)
        return np.sum((d[lag:] - m) * (d[:n - lag] - m)) / n

    V = acov(0) + 2 * sum(acov(k) for k in range(1, h))
    V = V / T

    # BUG-2 FIX (v2.5): when V <= 0 the Newey-West spectral estimate is
    # non-positive.  Return NaN unconditionally regardless of h.
    # The previous code recursed to h=1 when h>1, silently changing the
    # test null from "h-step equal accuracy" to "1-step equal accuracy" —
    # these are different hypotheses and the substitution is not prescribed
    # by Harvey, Leybourne & Newbold (1997).  All current run_dm() calls
    # use h=1, so this path was unreachable in practice, but the logic was
    # wrong and would corrupt results if h were ever changed.
    if V <= 0:
        return {"DM_stat": np.nan, "p_value": np.nan, "mean_diff": d_bar}

    DM = d_bar / math.sqrt(V)

    k = math.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    DM_hln = DM * k

    p = 2 * float(stats.t.cdf(-abs(DM_hln), df=T-1))
    return {"DM_stat": DM_hln, "p_value": p, "mean_diff": d_bar}


def _sig_star(p):
    """Return significance stars for a p-value."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def run_dm(data_dir, ckpt_dir, output_dir):
    test_raw = load_df(find_file(data_dir, "test"))

    for hz, hz_steps in HORIZONS.items():
        log(f"\n  [{hz}]")
        preds = load_all_preds(ckpt_dir, hz)
        if "hetero_ensemble" not in preds:
            log(f"  no hetero_ensemble for {hz}, skipping")
            continue

        test_fe     = fe_for_horizon(test_raw, hz_steps, "test")
        y_test_full = test_fe["cpu_target"].values.astype(np.float32)

        ref_len = min(len(y_test_full), len(preds["hetero_ensemble"]))
        y_test  = y_test_full[-ref_len:]

        aligned = {}
        for name, arr in preds.items():
            if len(arr) >= ref_len:
                aligned[name] = arr[-ref_len:].astype(np.float32)
            else:
                pad = np.full(ref_len - len(arr), np.nan, dtype=np.float32)
                aligned[name] = np.concatenate([pad, arr.astype(np.float32)])

        names  = list(aligned.keys())
        N      = len(names)
        dm_mat = np.full((N, N), np.nan)
        pv_mat = np.full((N, N), np.nan)

        all_pvals = []
        pairs     = []

        for i in range(N):
            dm_mat[i, i] = 0.0
            pv_mat[i, i] = 1.0
            for j in range(i + 1, N):
                mask = ~(np.isnan(aligned[names[i]]) | np.isnan(aligned[names[j]]))
                if mask.sum() < 30:
                    continue
                r = _dm_test_hln(y_test[mask],
                                 aligned[names[i]][mask],
                                 aligned[names[j]][mask],
                                 h=1, loss="MAE")
                dm_mat[i, j] = r["DM_stat"]
                dm_mat[j, i] = -r["DM_stat"]
                pv_mat[i, j] = r["p_value"]
                pv_mat[j, i] = r["p_value"]
                all_pvals.append(r["p_value"])
                pairs.append((i, j))

        if all_pvals:
            _, pv_corr, _, _ = multipletests(all_pvals, method="holm")
            for k, (i, j) in enumerate(pairs):
                pv_mat[i, j] = pv_corr[k]
                pv_mat[j, i] = pv_corr[k]

        dm_df = pd.DataFrame(dm_mat, index=names, columns=names)
        pv_df = pd.DataFrame(pv_mat, index=names, columns=names)
        dm_df.to_csv(os.path.join(output_dir, f"dm_stats_{hz}.csv"))
        pv_df.to_csv(os.path.join(output_dir, f"dm_pvalues_{hz}.csv"))

        mask_upper = np.triu(np.ones((N, N), dtype=bool), k=0)
        annot = np.empty((N, N), dtype=object)
        for i in range(N):
            for j in range(N):
                if mask_upper[i, j]:
                    annot[i, j] = ""
                else:
                    dm = dm_mat[i, j]
                    p  = pv_mat[i, j]
                    if np.isnan(dm):
                        annot[i, j] = "n/a"
                    else:
                        annot[i, j] = f"{dm:.2f}\n{p:.3f}{_sig_star(p)}"

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(pv_df.values, mask=mask_upper, annot=annot, fmt="",
                    cmap="RdYlGn_r", vmin=0, vmax=1, center=0.05,
                    linewidths=0.5, square=True, ax=ax,
                    cbar_kws={"label": "p-value (Holm-corrected)", "shrink": 0.8},
                    xticklabels=names, yticklabels=names,
                    annot_kws={"size": 7})
        ax.set_title(f"DM test pairwise (HLN + Holm) — {hz}",
                     fontsize=12, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        ax.text(0.5, -0.13,
                "*** p<0.001  ** p<0.01  * p<0.05  (Holm-Bonferroni)",
                transform=ax.transAxes, ha="center", fontsize=7, style="italic")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"dm_heatmap_{hz}.pdf"), dpi=300)
        plt.close(fig)
        log(f"  saved: dm_heatmap_{hz}.pdf")

    # --- Friedman + Nemenyi + CD diagrams ---
    if not HAS_SP:
        log("scikit-posthocs not installed — skipping Friedman/CD diagrams")
        return {"dm_done": True}

    log("\n  Friedman + Nemenyi + CD diagrams")
    friedman_rows = []

    for hz, hz_steps in HORIZONS.items():
        preds = load_all_preds(ckpt_dir, hz)
        if len(preds) < 3:
            log(f"  [{hz}] fewer than 3 models, skipping Friedman")
            continue

        test_fe     = fe_for_horizon(test_raw, hz_steps, "test")
        y_test_full = test_fe["cpu_target"].values.astype(np.float32)

        if "container_id" not in test_fe.columns:
            log(f"  [{hz}] no container_id column, skipping Friedman")
            continue

        model_names = list(preds.keys())
        ref_len = min(len(y_test_full), min(len(v) for v in preds.values()))
        y_test  = y_test_full[-ref_len:]
        cids    = test_fe["container_id"].values[-ref_len:]

        per_cid = {}
        for cid in np.unique(cids):
            mask = cids == cid
            if mask.sum() < 10:
                continue
            per_cid[cid] = {}

            for nm in model_names:
                arr = preds[nm]

                # FIX-3: align arr to ref_len with NaN padding before masking
                if len(arr) >= ref_len:
                    arr_aligned = arr[-ref_len:].astype(np.float32)
                else:
                    pad = np.full(ref_len - len(arr), np.nan, dtype=np.float32)
                    arr_aligned = np.concatenate([pad, arr.astype(np.float32)])

                p_cid = arr_aligned[mask]
                valid = ~np.isnan(p_cid)

                if valid.sum() < 5:
                    continue

                per_cid[cid][nm] = float(
                    np.mean(np.abs(y_test[mask][valid] - p_cid[valid])))

        valid_cids = [c for c, d in per_cid.items() if len(d) == len(model_names)]
        if len(valid_cids) < 10:
            log(f"  [{hz}] only {len(valid_cids)} complete containers, "
                f"skipping Friedman")
            continue

        perf_mat = np.array(
            [[per_cid[c][m] for m in model_names] for c in valid_cids])

        fstat, fp = friedmanchisquare(*[perf_mat[:, i]
                                        for i in range(len(model_names))])
        log(f"  [{hz}] Friedman χ²={fstat:.3f} p={fp:.6f}")
        friedman_rows.append({
            "Horizon":      hz,
            "chi2":         round(fstat, 4),
            "p_value":      round(fp, 6),
            "n_containers": len(valid_cids),
        })

        if fp >= 0.05:
            log(f"  [{hz}] Friedman non-significant — CD diagram not meaningful")
            continue

        perf_df   = pd.DataFrame(perf_mat, columns=model_names)
        nemenyi_p = sp.posthoc_nemenyi_friedman(perf_df)

        ranks     = np.array([rankdata(row) for row in perf_mat])
        avg_ranks = pd.Series(ranks.mean(axis=0), index=model_names)

        fig, ax = plt.subplots(figsize=(10, 2.8), dpi=150)
        try:
            sp.critical_difference_diagram(
                ranks=avg_ranks, sig_matrix=nemenyi_p, ax=ax,
                label_fmt_left="{label} ({rank:.2f})",
                label_fmt_right="({rank:.2f}) {label}",
                label_props={"fontsize": 10},
                marker_props={"s": 80, "zorder": 10, "edgecolors": "black"},
                crossbar_props={"linewidth": 3.0, "color": "#2c3e50"},
            )
        except Exception as e:
            warn(f"  CD diagram draw failed: {e}")

        ax.set_title(
            f"Critical difference diagram — {hz}  "
            f"(n={len(valid_cids)} containers, Nemenyi α=0.05)",
            fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"cd_diagram_{hz}.pdf"),
                    bbox_inches="tight", dpi=300)
        plt.close(fig)
        log(f"  saved: cd_diagram_{hz}.pdf")

    if friedman_rows:
        pd.DataFrame(friedman_rows).to_csv(
            os.path.join(output_dir, "friedman_table.csv"), index=False)
        log("  saved: friedman_table.csv")

    return {"dm_done": True}


# ===========================================================================
# PART 5: HPA SIMULATION
# ===========================================================================

class HPAConfig:
    """Kubernetes HPA configuration parameters."""
    def __init__(self):
        self.min_replicas    = 1
        self.max_replicas    = 100
        self.target_util     = 0.50   # 50% target CPU utilisation
        self.cpu_request     = 0.5    # vCPU per pod
        self.tolerance       = 0.10   # ±10% dead zone (K8s default)
        self.sync_steps      = 1      # evaluate every step
        # FIX-D: constants calibrated for 300s timesteps, not 15s sync cycles.
        self.scaledown_stab  = 1      # 300s window / 300s per step = 1 step
        self.pod_startup     = 1      # ~30s cold start rounds up to 1 step
        self.safety_margin   = 1.0    # headroom multiplier on predicted demand


class HPAState:
    """Mutable HPA controller state for one simulation run."""
    def __init__(self, min_replicas):
        self.cur_replicas = min_replicas
        self.ready        = min_replicas
        self.pending      = []       # list of (ready_at_step, count)
        self.sd_recs      = deque()  # scale-down recommendation history


def simulate_hpa(cpu_series, cfg, pred_cpu=None, reaction_lag=2):
    """
    Simplified K8s HPA state machine.

    Two modes:
      pred_cpu=None  →  reactive (uses observation from reaction_lag steps ago)
      pred_cpu=array →  ML-proactive (uses model predictions)
    """
    T     = len(cpu_series)
    state = HPAState(cfg.min_replicas)
    alloc = np.zeros(T, dtype=int)
    ready = np.zeros(T, dtype=int)

    for t in range(T):
        still_pending = []
        new_ready = 0
        for (ready_at, count) in state.pending:
            if t >= ready_at:
                new_ready += count
            else:
                still_pending.append((ready_at, count))
        state.pending = still_pending
        state.ready   = min(state.ready + new_ready, state.cur_replicas)

        alloc[t] = state.cur_replicas
        ready[t] = state.ready

        if t % cfg.sync_steps != 0:
            continue

        if pred_cpu is not None:
            demand = pred_cpu[t] * cfg.safety_margin
        else:
            demand = cpu_series[max(0, t - reaction_lag)]

        util  = (demand / state.ready) / cfg.cpu_request if state.ready > 0 else 1.0
        ratio = util / cfg.target_util

        if abs(ratio - 1.0) <= cfg.tolerance:
            continue

        raw     = math.ceil(state.cur_replicas * ratio)
        desired = max(cfg.min_replicas, min(cfg.max_replicas, raw))

        if desired > state.cur_replicas:
            add = desired - state.cur_replicas
            state.pending.append((t + cfg.pod_startup, add))
            state.cur_replicas = desired
            state.sd_recs.clear()

        elif desired < state.cur_replicas:
            # FIX-E: elapsed-time check matches K8s rolling-window semantics
            state.sd_recs.append((t, desired))
            if (t - state.sd_recs[0][0]) >= cfg.scaledown_stab:
                stabilised = max(d for _, d in state.sd_recs)
                if stabilised < state.cur_replicas:
                    state.cur_replicas = stabilised
                    state.ready = min(state.ready, stabilised)
                    state.sd_recs.clear()

    capacity  = ready * cfg.cpu_request
    viol_mask = cpu_series > capacity
    waste     = np.maximum(0, capacity - cpu_series)

    return {
        "alloc":              alloc,
        "ready":              ready,
        "violation_rate":     float(np.mean(viol_mask)),
        "waste_rate":         float(np.sum(waste) / max(np.sum(capacity), 1e-9)),
        "violation_severity": float(
            np.sum(np.maximum(0, cpu_series - capacity)) /
            max(np.sum(cpu_series), 1e-9)),
    }


def _pareto_front(points):
    """
    Return indices of Pareto-optimal points when minimising both axes.

    FIX-6: Scan left-to-right (ascending x) with strict <, keeping points
    where y strictly decreases.
    """
    sort_ix = np.argsort(points[:, 0])
    pareto  = []
    min_y   = float("inf")
    for i in range(len(sort_ix)):
        idx = sort_ix[i]
        if points[idx, 1] < min_y:
            min_y = points[idx, 1]
            pareto.append(idx)
    return np.array(pareto)


def run_hpa(data_dir, ckpt_dir, output_dir):
    test_raw = load_df(find_file(data_dir, "test"))

    safety_margins = np.linspace(1.0, 1.6, 10)
    reaction_lags  = [1, 2, 3, 4]
    all_rows = []

    for hz, hz_steps in HORIZONS.items():
        log(f"\n  [{hz}]")
        try:
            naive_arr = load_pred(ckpt_dir, hz, "naive")
            ml_arr    = load_pred(ckpt_dir, hz, "hetero_ensemble")
            if naive_arr is None or ml_arr is None:
                log(f"  missing predictions for {hz}")
                continue

            test_fe = fe_for_horizon(test_raw, hz_steps, "test")

            # FIX-A: use naive_cpu (guaranteed present) not cpu_util_percent
            y_test_full = test_fe["naive_cpu"].values.astype(np.float32)

            ref        = min(len(y_test_full), len(naive_arr), len(ml_arr))
            cpu_series = y_test_full[-ref:]
            ml_s       = ml_arr[-ref:]

            for lag in reaction_lags:
                for sm in safety_margins:
                    cfg = HPAConfig()
                    cfg.safety_margin = sm
                    r = simulate_hpa(cpu_series, cfg,
                                     pred_cpu=None, reaction_lag=lag)
                    all_rows.append({
                        "Horizon":        hz,
                        "Strategy":       "Reactive",
                        "variant":        f"lag={lag}",
                        "safety_margin":  round(float(sm), 3),
                        "violation_rate": r["violation_rate"],
                        "waste_rate":     r["waste_rate"],
                        "viol_severity":  r["violation_severity"],
                    })

            for sm in safety_margins:
                cfg = HPAConfig()
                cfg.safety_margin = sm
                r = simulate_hpa(cpu_series, cfg, pred_cpu=ml_s, reaction_lag=2)
                all_rows.append({
                    "Horizon":        hz,
                    "Strategy":       "ML-Proactive",
                    "variant":        "ML",
                    "safety_margin":  round(float(sm), 3),
                    "violation_rate": r["violation_rate"],
                    "waste_rate":     r["waste_rate"],
                    "viol_severity":  r["violation_severity"],
                })

        except Exception as e:
            warn(f"  [{hz}] HPA simulation failed: {e}")
            traceback.print_exc()
            continue

    if not all_rows:
        log("No HPA rows produced")
        return {}

    hpa_df = pd.DataFrame(all_rows)
    hpa_df.to_csv(os.path.join(output_dir, "hpa_simulation.csv"), index=False)
    log("  saved: hpa_simulation.csv")

    # --- Pareto frontier plot (2×2 per horizon) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, hz in enumerate(HORIZONS):
        ax  = axes[i]
        sub = hpa_df[hpa_df["Horizon"] == hz]
        if sub.empty:
            ax.set_visible(False)
            continue

        for strat, color, marker in [
            ("Reactive",     "#D55E00", "o"),
            ("ML-Proactive", "#0072B2", "s"),
        ]:
            s = sub[sub["Strategy"] == strat]
            if s.empty:
                continue

            w   = s["waste_rate"].values
            v   = s["violation_rate"].values
            pts = np.column_stack([w, v])

            ax.scatter(w, v, c=color, marker=marker, alpha=0.25, s=20)

            pidx = _pareto_front(pts)
            pp   = pts[pidx][np.argsort(pts[pidx, 0])]
            ax.scatter(pp[:, 0], pp[:, 1], c=color, marker=marker,
                       s=90, edgecolors="black", linewidths=1.0, zorder=5,
                       label=f"{strat} (Pareto)")
            ax.step(pp[:, 0], pp[:, 1], where="post", color=color, lw=2)

        ax.set_xlabel("Waste rate (over-provisioning)")
        ax.set_ylabel("Violation rate (QoS loss)")
        ax.set_title(hz, fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle("HPA simulation: Reactive vs ML-Proactive — Pareto frontier",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "hpa_pareto.pdf"), dpi=300)
    plt.close(fig)
    log("  saved: hpa_pareto.pdf")

    # --- 24-hour time-series overlay for 120min horizon ---
    try:
        hz       = "120min"
        hz_steps = HORIZONS[hz]
        ml_arr   = load_pred(ckpt_dir, hz, "hetero_ensemble")
        if ml_arr is not None:
            test_fe = fe_for_horizon(test_raw, hz_steps, "test")
            # FIX-A: use naive_cpu
            y_full  = test_fe["naive_cpu"].values.astype(np.float32)
            ref     = min(len(y_full), len(ml_arr))
            cpu_s   = y_full[-ref:]
            m_s     = ml_arr[-ref:]

            cfg_r = HPAConfig()
            cfg_r.safety_margin = 1.1
            cfg_m = HPAConfig()
            cfg_m.safety_margin = 1.1

            r_r = simulate_hpa(cpu_s, cfg_r, pred_cpu=None, reaction_lag=2)
            r_m = simulate_hpa(cpu_s, cfg_m, pred_cpu=m_s)

            w = 288  # 24 hours at 5-min intervals
            fig, axes2 = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            xs = np.arange(w)

            for ax2, r, cfg_used, label, color in [
                (axes2[0], r_r, cfg_r, "Reactive HPA", "#D55E00"),
                (axes2[1], r_m, cfg_m, "ML-Proactive", "#0072B2"),
            ]:
                cap  = r["ready"][:w] * cfg_used.cpu_request
                viol = cpu_s[:w] > cap

                ax2.plot(xs, cpu_s[:w], "k-", lw=1.0, alpha=0.7,
                         label="Actual CPU")
                ax2.plot(xs, cap, color=color, lw=1.5,
                         label=f"{label} capacity")
                ax2.fill_between(xs, 0, cpu_s[:w],
                                 where=viol, color="red", alpha=0.2,
                                 label="Violation")
                ax2.fill_between(xs, cpu_s[:w], cap,
                                 where=~viol, color="green", alpha=0.1,
                                 label="Waste")
                ax2.set_ylabel("CPU (vCPU)")
                ax2.set_title(label, fontsize=10)
                ax2.legend(fontsize=8, frameon=False, loc="upper right")
                ax2.text(0.01, 0.97,
                         f"viol={r['violation_rate']:.1%}  "
                         f"waste={r['waste_rate']:.1%}",
                         transform=ax2.transAxes, va="top", fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white", alpha=0.8))

            axes2[1].set_xlabel("5-minute step (24h window)")
            fig.suptitle(
                "HPA time-series: reactive lag vs ML proactive scaling "
                "(120min horizon)", fontsize=12)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "hpa_timeseries.pdf"), dpi=300)
            plt.close(fig)
            log("  saved: hpa_timeseries.pdf")

    except Exception as e:
        warn(f"  HPA timeseries plot failed: {e}")

    return {"hpa_done": True}


# ===========================================================================
# PART 6: EXTRA ANALYSES
# ===========================================================================

def run_extra(data_dir, ckpt_dir, output_dir):
    test_raw = load_df(find_file(data_dir, "test"))

    # Cache feature-engineered test splits per horizon once.
    # run_extra calls fe_for_horizon(test_raw, ...) across three separate
    # sub-sections (skill loop, per-container scatter, CV stratification).
    # Without caching that is 12 full feature-engineering passes over the
    # test dataset.  Build the cache here and reuse throughout.
    log("  Pre-computing feature-engineered test splits for all horizons...")
    test_fe_cache = {}
    for hz, hz_steps in HORIZONS.items():
        try:
            test_fe_cache[hz] = fe_for_horizon(test_raw, hz_steps, "test")
        except Exception as e:
            warn(f"  [{hz}] fe_for_horizon failed: {e}")

    strat_rows = []
    lb_rows    = []

    for hz, hz_steps in HORIZONS.items():
        log(f"\n  [{hz}]")
        naive_arr = load_pred(ckpt_dir, hz, "naive")
        ens_arr   = load_pred(ckpt_dir, hz, "hetero_ensemble")

        if naive_arr is None or ens_arr is None:
            log(f"  missing arrays for {hz}")
            continue

        test_fe = test_fe_cache.get(hz)
        if test_fe is None:
            warn(f"  [{hz}] no cached test_fe, skipping")
            continue
        y_full  = test_fe["cpu_target"].values.astype(np.float32)

        n       = min(len(y_full), len(naive_arr), len(ens_arr))
        y_true  = y_full[-n:]
        naive_p = naive_arr[-n:]
        ens_p   = ens_arr[-n:]

        mae_n = float(np.mean(np.abs(naive_p - y_true)))
        mae_e = float(np.mean(np.abs(ens_p   - y_true)))

        skill = 1.0 - mae_e / (mae_n + 1e-9)
        win   = float((np.abs(ens_p - y_true) < np.abs(naive_p - y_true)).mean())

        strat_rows.append({
            "Horizon":              hz,
            "MAE_naive":            round(mae_n, 4),
            "MAE_ensemble":         round(mae_e, 4),
            "Skill_score":          round(skill, 4),
            "Timestep_Win_rate_%":  round(win * 100, 2),
        })
        log(f"  skill={skill:.4f}  win={win:.1%}")

        # FIX-9: use consecutive head of residual array (res[:samp])
        res_n = (y_true - naive_p).astype(np.float64)
        res_e = (y_true - ens_p).astype(np.float64)
        samp  = min(50_000, n)

        res_n_sub = res_n[:samp]
        res_e_sub = res_e[:samp]

        lb_n   = acorr_ljungbox(res_n_sub, lags=[10], return_df=True)
        lb_e   = acorr_ljungbox(res_e_sub, lags=[10], return_df=True)

        acf1_n = float(np.corrcoef(res_n_sub[:-1], res_n_sub[1:])[0, 1])
        acf1_e = float(np.corrcoef(res_e_sub[:-1], res_e_sub[1:])[0, 1])

        lb_rows.append({
            "Horizon":           hz,
            "Bias_naive":        round(float(res_n.mean()), 4),
            "Bias_ensemble":     round(float(res_e.mean()), 4),
            "ACF1_naive":        round(acf1_n, 4),
            "ACF1_ensemble":     round(acf1_e, 4),
            "LjungBox_p_naive":  round(float(lb_n["lb_pvalue"].iloc[0]), 6),
            "LjungBox_p_ens":    round(float(lb_e["lb_pvalue"].iloc[0]), 6),
        })

    if strat_rows:
        sk_df = pd.DataFrame(strat_rows)
        sk_df.to_csv(os.path.join(output_dir, "stratified_skill.csv"), index=False)
        log("\n  saved: stratified_skill.csv")
        print(sk_df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#d62728" if v < 0 else "#1f77b4"
                  for v in sk_df["Skill_score"]]
        bars   = ax.bar(sk_df["Horizon"], sk_df["Skill_score"] * 100,
                        color=colors, width=0.55, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("Skill score (%) — positive = ML beats naive")
        ax.set_title("ML ensemble skill score over naive baseline by horizon")

        for b, v in zip(bars, sk_df["Skill_score"] * 100):
            yp = b.get_height() + 0.3 if v >= 0 else b.get_height() - 1.5
            ax.text(b.get_x() + b.get_width() / 2, yp, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "skill_score_by_horizon.pdf"),
                    dpi=300)
        plt.close(fig)
        log("  saved: skill_score_by_horizon.pdf")

    if lb_rows:
        pd.DataFrame(lb_rows).to_csv(
            os.path.join(output_dir, "residual_diagnostics.csv"), index=False)
        log("  saved: residual_diagnostics.csv")

    # --- Per-container win-rate scatter ---
    log("\n  Per-container win-rate scatter")
    try:
        cid_col = "container_id"
        if cid_col not in test_raw.columns:
            log("  no container_id column in raw test data, "
                "skipping win-rate scatter")
        else:
            fig, axs = plt.subplots(2, 2, figsize=(10, 9))
            axs = axs.flatten()

            for idx, (hz, hz_steps) in enumerate(HORIZONS.items()):
                naive_arr = load_pred(ckpt_dir, hz, "naive")
                ens_arr   = load_pred(ckpt_dir, hz, "hetero_ensemble")
                if naive_arr is None or ens_arr is None:
                    axs[idx].set_visible(False)
                    continue

                test_fe = test_fe_cache.get(hz)
                if test_fe is None:
                    axs[idx].set_visible(False)
                    continue
                y_full  = test_fe["cpu_target"].values.astype(np.float32)
                cids    = test_fe[cid_col].values

                n       = min(len(y_full), len(naive_arr), len(ens_arr))
                y_true  = y_full[-n:]
                naive_p = naive_arr[-n:]
                ens_p   = ens_arr[-n:]
                cids_t  = cids[-n:]

                n_maes = []
                e_maes = []
                for cid in np.unique(cids_t):
                    m = cids_t == cid
                    if m.sum() < 5:
                        continue
                    n_maes.append(float(
                        np.mean(np.abs(naive_p[m] - y_true[m]))))
                    e_maes.append(float(
                        np.mean(np.abs(ens_p[m]   - y_true[m]))))

                nv = np.array(n_maes)
                ev = np.array(e_maes)

                if len(nv) == 0:
                    axs[idx].set_visible(False)
                    continue

                win_pct = float((ev < nv).mean() * 100)
                ax      = axs[idx]
                lim     = max(nv.max(), ev.max()) * 1.05

                point_colors = ["#1f77b4" if e < n_ else "#d62728"
                                for e, n_ in zip(ev, nv)]
                ax.scatter(nv, ev, alpha=0.4, s=12, c=point_colors)
                ax.plot([0, lim], [0, lim], "k--", lw=0.8)
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                ax.set_xlabel("Naive MAE")
                ax.set_ylabel("Ensemble MAE")
                ax.set_title(f"{hz}  —  ML wins on {win_pct:.1f}% of containers")

            fig.suptitle("Per-container MAE: ensemble vs naive\n"
                         "(blue = ML wins, red = naive wins)",
                         fontsize=11, y=1.01)
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, "per_container_winrate.pdf"),
                        dpi=300)
            plt.close(fig)
            log("  saved: per_container_winrate.pdf")

    except Exception as e:
        warn(f"  Win-rate scatter failed: {e}")

    # --- CV-stratified skill scores ---
    log("\n  CV-stratified skill scores")
    try:
        landscape_path = os.path.join(output_dir, "container_landscape_stats.csv")
        if not os.path.exists(landscape_path):
            warn("  container_landscape_stats.csv not found — skipping CV stratification")
        elif "container_id" not in test_raw.columns:
            warn("  no container_id in test_raw — skipping CV stratification")
        else:
            landscape = pd.read_csv(landscape_path)
            cv_map = dict(zip(landscape["container_id"], landscape["cv"]))

            cv_bins   = [0.0, 0.1, 0.3, 0.5, 0.7, 9999.0]
            cv_labels = ["CV<0.1", "CV 0.1-0.3", "CV 0.3-0.5", "CV 0.5-0.7", "CV>0.7"]

            per_cid_rows = []
            for hz, hz_steps in HORIZONS.items():
                naive_arr = load_pred(ckpt_dir, hz, "naive")
                ens_arr   = load_pred(ckpt_dir, hz, "hetero_ensemble")
                if naive_arr is None or ens_arr is None:
                    warn(f"  [{hz}] predictions not found, skipping")
                    continue

                test_fe = test_fe_cache.get(hz)
                if test_fe is None:
                    warn(f"  [{hz}] no cached test_fe, skipping CV bin")
                    continue
                y_full  = test_fe["cpu_target"].values.astype(np.float32)
                cids    = test_fe["container_id"].values

                n      = min(len(y_full), len(naive_arr), len(ens_arr))
                y_true = y_full[-n:]
                naive_p = naive_arr[-n:]
                ens_p   = ens_arr[-n:]
                cids_t  = cids[-n:]

                for cid in np.unique(cids_t):
                    mask = cids_t == cid
                    if mask.sum() < 5:
                        continue
                    cv_val = cv_map.get(cid, np.nan)
                    if np.isnan(cv_val):
                        continue
                    bin_idx   = min(np.searchsorted(cv_bins[1:], cv_val, side='right'), len(cv_labels) - 1)
                    bin_label = cv_labels[bin_idx]
                    mae_n = float(np.mean(np.abs(naive_p[mask] - y_true[mask])))
                    mae_e = float(np.mean(np.abs(ens_p[mask]   - y_true[mask])))
                    per_cid_rows.append({
                        "Horizon":      hz,
                        "CV_bin":       bin_label,
                        "container_id": cid,
                        "MAE_naive":    mae_n,
                        "MAE_ensemble": mae_e,
                    })

            if per_cid_rows:
                raw_df = pd.DataFrame(per_cid_rows)
                agg = (raw_df
                       .groupby(["Horizon", "CV_bin"])
                       .agg(
                           MAE_naive    =("MAE_naive",    "mean"),
                           MAE_ensemble =("MAE_ensemble", "mean"),
                           N_containers =("container_id", "nunique"),
                       )
                       .reset_index())
                agg["Skill_score"]      = 1.0 - agg["MAE_ensemble"] / (agg["MAE_naive"] + 1e-9)
                agg["Container_win_%"]  = (
                    raw_df.groupby(["Horizon", "CV_bin"])
                    .apply(lambda g: (g["MAE_ensemble"] < g["MAE_naive"]).mean() * 100)
                    .reset_index(name="Container_win_%")["Container_win_%"].values
                )

                # sort by horizon then CV bin order
                hz_order  = list(HORIZONS.keys())
                bin_order = cv_labels
                agg["_hz_idx"]  = agg["Horizon"].map({h: i for i, h in enumerate(hz_order)})
                agg["_bin_idx"] = agg["CV_bin"].map({b: i for i, b in enumerate(bin_order)})
                agg = agg.sort_values(["_hz_idx", "_bin_idx"]).drop(columns=["_hz_idx", "_bin_idx"])

                out_path = os.path.join(output_dir, "cv_stratified_skill.csv")
                agg.to_csv(out_path, index=False)
                log(f"  saved: cv_stratified_skill.csv ({len(agg)} rows)")
                print(agg.to_string(index=False))
            else:
                warn("  No per-container rows produced — check container_id alignment")

    except Exception as e:
        warn(f"  CV-stratified skill failed: {e}")
        import traceback; traceback.print_exc()

    return {"extra_done": True}


# ===========================================================================
# PART 7: SUMMARY FIGURES
# ===========================================================================

def run_summary(ckpt_dir, output_dir, results_csv=None):
    """
    Generate summary figures from confirmed result numbers.
    Uses results_summary_5.csv if available.
    """
    res_df = None
    if results_csv and os.path.exists(results_csv):
        res_df = pd.read_csv(results_csv)
        log(f"  Loaded results from {results_csv}")

    # --- NNLS weights stacked bar ---
    log("  NNLS weights stacked bar")
    nnls_all = {}
    for hz in HORIZONS:
        w = load_nnls_weights(ckpt_dir, hz)
        if w:
            nnls_all[hz] = w

    if nnls_all:
        all_models = list(dict.fromkeys(k for d in nnls_all.values() for k in d))

        weights_pct = {}
        for hz, wd in nnls_all.items():
            total = sum(wd.values())
            if total < 1e-9:
                total = 1.0
            weights_pct[hz] = {m: wd.get(m, 0.0) / total * 100
                               for m in all_models}

        color_map = {
            "XGB":    "#E69F00",
            "LGB":    "#56B4E9",
            "ET":     "#009E73",
            "BiLSTM": "#CC79A7",
        }
        colors_list = [color_map.get(m, "#999999") for m in all_models]

        horizons_list = list(nnls_all.keys())

        fig, ax = plt.subplots(figsize=(8, 4))
        x    = np.arange(len(horizons_list))
        bots = np.zeros(len(horizons_list))

        for j, model in enumerate(all_models):
            pcts = np.array([weights_pct[hz][model] for hz in horizons_list])
            ax.bar(x, pcts, bottom=bots, color=colors_list[j],
                   label=model, width=0.65, edgecolor="white", linewidth=0.5)

            for k, pct in enumerate(pcts):
                if pct >= 4:
                    y_pos = bots[k] + pct / 2
                    hex_c = colors_list[j].lstrip("#")
                    r_ = int(hex_c[0:2], 16)
                    g_ = int(hex_c[2:4], 16)
                    b_ = int(hex_c[4:6], 16)
                    lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
                    tc  = "white" if lum < 128 else "black"
                    ax.text(k, y_pos, f"{pct:.0f}%",
                            ha="center", va="center", fontsize=7,
                            fontweight="bold", color=tc)
            bots += pcts

        ax.set_xticks(x)
        ax.set_xticklabels(horizons_list)
        ax.set_ylabel("NNLS weight (%)")
        ax.set_ylim(0, 100)
        ax.set_title("NNLS meta-learner weights: ExtraTrees dominates at short "
                     "horizons,\nBiLSTM contributes at longer horizons")

        # FIX-B: axhline MUST come before ax.legend() so the label is captured
        ax.axhline(25, color="grey", ls=":", lw=0.7, alpha=0.5,
                   label="equal-weight reference (4 models)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=9, frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "nnls_weights_bar.pdf"), dpi=300)
        plt.close(fig)
        log("  saved: nnls_weights_bar.pdf")

    # --- R² improvement line plot ---
    if res_df is not None:
        log("  R² improvement line plot")
        hz_order = ["10min", "30min", "60min", "120min"]
        x_vals   = [10, 30, 60, 120]

        impr     = res_df.set_index("Horizon")["cpu_improvement_pp"].reindex(hz_order)
        naive_r2 = res_df.set_index("Horizon")["naive_R2"].reindex(hz_order)
        ens_r2   = res_df.set_index("Horizon")["hetero_ensemble_R2"].reindex(hz_order)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(x_vals, naive_r2.values, "o--", color="#7f7f7f",
                 lw=2, ms=7, label="Naive baseline")
        ax1.plot(x_vals, ens_r2.values, "s-", color="#1f77b4",
                 lw=2.5, ms=8, label="Hetero ensemble")
        ax1.fill_between(x_vals, naive_r2.values, ens_r2.values,
                         alpha=0.15, color="#1f77b4")
        ax1.set_xlabel("Prediction horizon (minutes)")
        ax1.set_ylabel("R²")
        ax1.set_title("R² by horizon: ensemble vs naive")
        ax1.legend(fontsize=9)
        ax1.set_xscale("log")
        ax1.set_xticks([10, 30, 60, 120])
        ax1.set_xticklabels(["10", "30", "60", "120"])

        bar_colors = [HZ_COLORS[h] for h in hz_order]
        ax2.bar(x_vals, impr.values, color=bar_colors, width=5,
                edgecolor="white")
        ax2.axhline(0, color="black", lw=0.8)

        # BUG-5 FIX (v2.5): use zip(x_vals, impr.values, hz_order) instead
        # of x_vals.index(xi), which is an O(n) list search on a list whose
        # iteration order already guarantees alignment with hz_order.
        for xi, v, hz_label in zip(x_vals, impr.values, hz_order):
            if not np.isnan(v):
                yp = v + 0.05 if v >= 0 else v - 0.15
                # FIX-4: {:+.2f} renders "-0.60pp" not "+-0.60pp"
                ax2.text(xi, yp, f"{v:+.2f}pp",
                         ha="center", fontsize=9, fontweight="bold",
                         color=HZ_COLORS[hz_label])

        ax2.set_xlabel("Prediction horizon (minutes)")
        ax2.set_ylabel("R² improvement over naive (pp)")
        ax2.set_title("ML improvement scales monotonically with horizon")
        ax2.set_xticks(x_vals)

        fig.suptitle("Core finding: ML value grows with prediction horizon",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "r2_improvement_line.pdf"), dpi=300)
        plt.close(fig)
        log("  saved: r2_improvement_line.pdf")

        # --- comparison table as CSV (LaTeX-ready) ---
        log("  Building comparison table")
        cols_map = {
            "Horizon":            "Horizon",
            "naive_R2":           "Naive R²",
            "ets_R2":             "ETS R²",
            "arima_R2":           "ARIMA R²",
            "linear_reg_R2":      "LinReg R²",
            "xgboost_R2":         "XGB R²",
            "lightgbm_R2":        "LGB R²",
            "extratrees_R2":      "ET R²",
            "bilstm_R2":          "BiLSTM R²",
            "homo_ensemble_R2":   "Homo Ens R²",
            "hetero_ensemble_R2": "Hetero Ens R²",
            "cpu_improvement_pp": "Δpp (Hetero vs Naive)",
        }
        present = [c for c in cols_map if c in res_df.columns]
        tbl = res_df[present].rename(columns=cols_map)
        tbl.to_csv(os.path.join(output_dir, "comparison_table.csv"), index=False)
        log("  saved: comparison_table.csv")
        print(tbl.to_string(index=False))

    # --- Ablation disclosure ---
    log("\n  Ablation baseline disclosure:")
    print("""
  Ablation experiments use a single LightGBM proxy trained on naive residuals.
  Reported R² drops are relative to this proxy and are NOT directly comparable
  to stacked ensemble metrics.  Three things differ simultaneously:
    (1) target (residuals vs absolute CPU)
    (2) model (single LGB vs stacked ensemble)
    (3) feature set
  Pairwise comparisons within the ablation table remain internally valid.
""")

    return {"summary_done": True}


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    pa = argparse.ArgumentParser(
        description="Unified thesis analysis: SHAP, workload, UQ, DM, HPA, "
                    "extras, summary"
    )
    pa.add_argument("--sprint1-dir",    default=".",
                    help="folder with sprint1_v9.py")
    pa.add_argument("--data-dir",       required=True,
                    help="folder with train/val/test parquet or csv files")
    pa.add_argument("--ckpt-dir",       required=True,
                    help="sprint1 checkpoint directory")
    pa.add_argument("--output-dir",     default="./thesis_figures")
    pa.add_argument("--results-csv",    default=None,
                    help="path to results_summary_5.csv")
    pa.add_argument("--dsb-data",       default=None,
                    help="path to DSB csv/parquet")
    pa.add_argument("--bitbrains-data", default=None,
                    help="path to Bitbrains csv/parquet")
    pa.add_argument("--shap-subsample", type=int, default=500,
                    help="SHAP sample count")
    pa.add_argument("--perm-subsample", type=int, default=5000,
                    help="permutation importance sample count")
    pa.add_argument("--perm-repeats",   type=int, default=10,
                    help="permutation repeats")
    pa.add_argument("--use-gpu",        action="store_true")
    pa.add_argument("--skip-shap",      action="store_true")
    pa.add_argument("--skip-perm",      action="store_true")
    pa.add_argument("--skip-workload",  action="store_true")
    pa.add_argument("--skip-uq",        action="store_true")
    pa.add_argument("--skip-dm",        action="store_true")
    pa.add_argument("--skip-hpa",       action="store_true")
    pa.add_argument("--skip-extra",     action="store_true")
    pa.add_argument("--skip-summary",   action="store_true")
    pa.add_argument("--force", metavar="SECTION", default=None,
                    help="clear checkpoint for SECTION and re-run it. "
                         "Valid values: shap workload uq dm hpa extra summary")
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_mgr = CheckpointManager(
        os.path.join(args.output_dir, "checkpoints"))

    if args.force:
        ckpt_mgr.force_clear(args.force)

    log(f"Output dir  : {args.output_dir}")
    log(f"Checkpoint  : {ckpt_mgr.dir}")

    needs_s1 = not (args.skip_shap and args.skip_uq and args.skip_dm
                    and args.skip_hpa and args.skip_extra)
    if needs_s1:
        load_sprint1(args.sprint1_dir)

    ckpt_mgr.summary()

    if not args.skip_shap:
        run_section("shap", run_shap, ckpt_mgr,
                    args.data_dir, args.ckpt_dir, args.output_dir,
                    args.shap_subsample, args.perm_subsample,
                    args.perm_repeats, args.use_gpu,
                    run_perm=not args.skip_perm)
    else:
        log("⏭  shap skipped (--skip-shap)")

    if not args.skip_workload:
        run_section("workload", run_workload, ckpt_mgr,
                    args.data_dir, args.dsb_data, args.bitbrains_data,
                    args.output_dir)
    else:
        log("⏭  workload skipped (--skip-workload)")

    if not args.skip_uq:
        run_section("uq", run_uq, ckpt_mgr,
                    args.data_dir, args.ckpt_dir, args.output_dir)
    else:
        log("⏭  uq skipped (--skip-uq)")

    if not args.skip_dm:
        run_section("dm", run_dm, ckpt_mgr,
                    args.data_dir, args.ckpt_dir, args.output_dir)
    else:
        log("⏭  dm skipped (--skip-dm)")

    if not args.skip_hpa:
        run_section("hpa", run_hpa, ckpt_mgr,
                    args.data_dir, args.ckpt_dir, args.output_dir)
    else:
        log("⏭  hpa skipped (--skip-hpa)")

    if not args.skip_extra:
        run_section("extra", run_extra, ckpt_mgr,
                    args.data_dir, args.ckpt_dir, args.output_dir)
    else:
        log("⏭  extra skipped (--skip-extra)")

    if not args.skip_summary:
        run_section("summary", run_summary, ckpt_mgr,
                    args.ckpt_dir, args.output_dir, args.results_csv)
    else:
        log("⏭  summary skipped (--skip-summary)")

    sep()
    ckpt_mgr.summary()
    log(f"\nAll done. Figures in: {args.output_dir}")


if __name__ == "__main__":
    main()