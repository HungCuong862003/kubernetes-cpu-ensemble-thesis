import os
import gc
import json
import time
import signal
import shutil
import logging
import argparse
import warnings
import functools
import threading
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import ttest_rel, t as scipy_t
from scipy.optimize import nnls as scipy_nnls

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)


# try to import torch - if it's not there, BiLSTM just won't run
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_CUDA = torch.cuda.is_available()
    print(f"PyTorch {torch.__version__} -- {DEVICE}")
    if HAS_CUDA:
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({props.total_memory/1e9:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False
    DEVICE = None
    print("WARNING: PyTorch not found. BiLSTM disabled.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not found. HPO disabled.")

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class Config:
    DATA_DIR   = "/workspace/thesis"
    OUTPUT_DIR = "/workspace/results/sprint1"
    TMP_DIR    = "/tmp/sprint1"

    # horizon name -> number of 5-min steps ahead
    HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}
    SAMPLING_INTERVAL = 300   # seconds per step
    N_CV_SPLITS = 5
    N_BOOTSTRAP = 1000

    XGB_PARAMS = {
        "objective": "reg:squarederror",
        "n_estimators": 2000, "max_depth": 7, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
        "reg_alpha": 0.1, "reg_lambda": 3.0,
        "random_state": 42, "n_jobs": -1,
        "early_stopping_rounds": 50,
    }
    LGBM_PARAMS = {
        "objective": "regression_l1",
        "n_estimators": 500,
        "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.7, "min_child_samples": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
        "metric": "l1",
    }
    LGBM_EARLY_STOP = 20

    ET_PARAMS = {
        "n_estimators": 200,
        "max_depth": 25,
        "min_samples_leaf": 5,
        "max_features": 0.8,
        "bootstrap": True,
        "max_samples": 0.3,
        "random_state": 42,
        "n_jobs": -1,
    }

    # lighter params for ablation so it doesn't take forever
    ABLATION_PARAMS = {
        "objective": "regression_l1",
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": 42, "n_jobs": -1, "verbose": -1,
        "metric": "l1",
    }

    BILSTM_HIDDEN   = 64
    BILSTM_LAYERS   = 2
    BILSTM_DROPOUT  = 0.2
    BILSTM_LOOKBACK = 12
    BILSTM_BATCH    = 4096
    BILSTM_LR       = 1e-3
    BILSTM_EPOCHS   = 50
    BILSTM_PATIENCE = 10
    BILSTM_STRIDE   = 2

    USE_AMP = True
    TF32    = True
    DATALOADER_WORKERS = 4
    DATALOADER_PIN_MEMORY = True
    DATALOADER_PERSISTENT_WORKERS = True
    DATALOADER_PREFETCH_FACTOR = 2

    # run all 4 horizons at once using separate processes
    PARALLEL_HORIZONS = True
    N_CPU_WORKERS     = 4
    CPUS_PER_WORKER   = 20

    NO_BILSTM = False

    OPTUNA_TRIALS  = 10
    OPTUNA_TIMEOUT = 600

    MC_PASSES = 30
    CQR_ALPHA = 0.2
    RIDGE_ALPHA = 1.0

    STAT_MAX_CONTAINERS = 20
    STAT_MIN_HISTORY    = 48
    STAT_MAX_HISTORY    = 576
    STAT_N_JOBS         = -1

    # keep subsample at 1.0 so OOF covers the whole training set
    OOF_SUBSAMPLE      = 1.0
    OOF_SUBSAMPLE_SEED = 42

    ACI_GAMMA        = 0.05
    ACI_GAMMA_SEARCH = [0.01, 0.02, 0.05, 0.10]

    # cap the residual buffer so memory doesn't blow up on long runs
    ACI_RESIDUAL_BUFFER_SIZE = 2000

    IO_MAX_RETRIES  = 3
    IO_RETRY_BASE_S = 2

    def _find(self, stem):
        for ext in (".parquet", ".csv"):
            p = os.path.join(self.DATA_DIR, f"{stem}{ext}")
            if os.path.exists(p):
                return p
        return os.path.join(self.DATA_DIR, f"{stem}.csv")

    @property
    def train_file(self): return self._find("train")
    @property
    def val_file(self):   return self._find("val")
    @property
    def test_file(self):  return self._find("test")


cfg = Config()

if HAS_CUDA:
    cfg.XGB_PARAMS["device"] = "cuda"
    cfg.XGB_PARAMS["tree_method"] = "hist"
    try:
        _t = xgb.XGBRegressor(device="cuda", tree_method="hist",
                               n_estimators=2, verbosity=0)
        _t.fit(np.random.randn(100, 5).astype(np.float32),
               np.random.randn(100).astype(np.float32))
        del _t
        print("XGBoost GPU: OK")
    except Exception as e:
        print(f"XGBoost GPU failed ({e}), using CPU")
        cfg.XGB_PARAMS.pop("device", None)
        cfg.XGB_PARAMS.pop("tree_method", None)
else:
    print("No CUDA -- XGBoost on CPU")

# LightGBM GPU - n_jobs=1 when using GPU to avoid thread pool issues
if HAS_CUDA:
    try:
        _lgb_t = lgb.LGBMRegressor(device="gpu", n_estimators=2, verbose=-1)
        _lgb_t.fit(np.random.randn(100, 5).astype(np.float32),
                   np.random.randn(100).astype(np.float32))
        del _lgb_t
        cfg.LGBM_PARAMS["device"] = "gpu"
        cfg.LGBM_PARAMS["n_jobs"] = 1
        print("LightGBM GPU: OK (n_jobs set to 1)")
    except Exception as e:
        print(f"LightGBM GPU failed ({e}), using CPU")


# logging setup - each parallel worker gets its own log file so they don't
# write over each other
_logger = logging.getLogger("sprint1")
_logger.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

def _setup_logging(output_dir, log_suffix=""):
    """Set up file + stream logging.

    Pass log_suffix like "_10min" when running parallel workers so each
    one writes to its own file instead of fighting over run.log.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_name = f"run{log_suffix}.log"
    fh = logging.FileHandler(os.path.join(output_dir, log_name), mode="a")
    fh.setFormatter(_fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(_fmt)
    _logger.handlers.clear()
    _logger.addHandler(fh)
    _logger.addHandler(sh)

def log(msg):
    _logger.info(msg)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def bilstm_enabled():
    return HAS_TORCH and not cfg.NO_BILSTM

def mem_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return -1

def to_serializable(obj):
    """Recursively convert numpy types to plain python so json.dump works."""
    if obj is None: return None
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, pd.DataFrame): return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series): return obj.tolist()
    try:
        if pd.isna(obj): return None
    except (TypeError, ValueError):
        pass
    return obj


def retry_io(func):
    """Retry a file operation a few times before giving up - disk can be flaky."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_err = None
        for attempt in range(cfg.IO_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (IOError, OSError, shutil.Error) as e:
                last_err = e
                wait = cfg.IO_RETRY_BASE_S * (2 ** attempt)
                log(f"  IO retry {attempt+1}/{cfg.IO_MAX_RETRIES} ({e}), wait {wait}s")
                time.sleep(wait)
        raise last_err
    return wrapper


@retry_io
def safe_write_npy(path, arr):
    """Write a numpy array atomically - write to temp then rename so we never
    end up with a half-written checkpoint file."""
    ensure_dir(os.path.dirname(path))
    tmp = os.path.join(os.path.dirname(path), f".tmp_{os.path.basename(path)}")
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except OSError: pass
        raise


@retry_io
def atomic_json_write(path, data):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_serializable(data), f, indent=2)
    os.replace(tmp, path)


_sync_thread = None

def start_rsync_backup(interval_minutes=15, rsync_dest=None):
    global _sync_thread
    if rsync_dest is None:
        log("No rsync dest specified, skipping backup")
        return

    def _do_sync():
        while True:
            time.sleep(interval_minutes * 60)
            cmd = f"rsync -aq --no-perms {cfg.OUTPUT_DIR}/ {rsync_dest}/"
            ret = os.system(cmd)
            if ret == 0:
                log(f"[sync] backed up to {rsync_dest}")
            else:
                log(f"[sync] rsync failed (exit {ret})")

    if _sync_thread is None or not _sync_thread.is_alive():
        _sync_thread = threading.Thread(target=_do_sync, daemon=True)
        _sync_thread.start()
        log(f"Rsync backup thread started (every {interval_minutes}min -> {rsync_dest})")


def health_check():
    log("Health check:")
    free = shutil.disk_usage("/workspace").free / 1e9 if os.path.exists("/workspace") else -1
    if free > 0:
        log(f"  Disk free: {free:.1f} GB")
        if free < 10:
            log("  WARNING: low disk space (<10GB)")
    ram = mem_gb()
    if ram > 0:
        log(f"  RAM: {ram:.1f} GB used")
    if HAS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = cfg.TF32
        torch.backends.cudnn.allow_tf32 = cfg.TF32
        log(f"  GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
        log(f"  TF32: {cfg.TF32}")
        log(f"  XGBoost device: {cfg.XGB_PARAMS.get('device', 'cpu')}")
        log(f"  LightGBM device: {cfg.LGBM_PARAMS.get('device', 'cpu')}")
        log(f"  AMP: {'BF16' if cfg.USE_AMP else 'off'}")
    else:
        log("  GPU: none")
    log(f"  CPU cores: {os.cpu_count()}")
    log(f"  OOF subsample: {cfg.OOF_SUBSAMPLE}")
    log(f"  ACI buffer size: {cfg.ACI_RESIDUAL_BUFFER_SIZE}")
    log(f"  BiLSTM batch: {cfg.BILSTM_BATCH}  stride: {cfg.BILSTM_STRIDE}")
    log(f"  ET max_depth: {cfg.ET_PARAMS['max_depth']}  max_samples: {cfg.ET_PARAMS['max_samples']}")


_INTERRUPTED = False

def _sigint_handler(sig, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    if cfg.PARALLEL_HORIZONS:
        log("SIGINT -- parent stopping; parallel workers will finish current stage "
            "(use 'pkill -9 python3' for immediate stop)")
    else:
        log("SIGINT -- will stop after current stage")

def check_interrupted():
    return _INTERRUPTED


class Checkpoint:
    """Simple file-based checkpoint system. Each horizon gets its own folder
    and a _state.json that tracks which stages finished."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        ensure_dir(output_dir)

    def _hz_dir(self, hz):
        d = os.path.join(self.output_dir, hz)
        ensure_dir(d)
        return d

    def _state_path(self, hz):
        return os.path.join(self._hz_dir(hz), "_state.json")

    def get_completed(self, hz):
        p = self._state_path(hz)
        if not os.path.exists(p):
            return set()
        try:
            with open(p) as f:
                return set(json.load(f).get("completed", []))
        except (json.JSONDecodeError, IOError):
            return set()

    def _mark_stage(self, hz, stage, elapsed_s=None):
        p = self._state_path(hz)
        state = {}
        if os.path.exists(p):
            try:
                with open(p) as f:
                    state = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        completed = set(state.get("completed", []))
        completed.add(stage)
        state["completed"] = sorted(completed)
        state["last_update"] = datetime.now().isoformat()
        timings = state.get("timings", {})
        if elapsed_s is not None:
            timings[stage] = round(elapsed_s, 1)
        state["timings"] = timings
        atomic_json_write(p, state)

    def mark_done(self, hz, stage, metrics_data=None, elapsed_s=None):
        if metrics_data is not None:
            atomic_json_write(
                os.path.join(self._hz_dir(hz), f"{stage}.json"), metrics_data)
        self._mark_stage(hz, stage, elapsed_s=elapsed_s)

    def save_pred(self, hz, name, arr):
        safe_write_npy(os.path.join(self._hz_dir(hz), f"pred_{name}.npy"), arr)

    def load_pred(self, hz, name, expected_len=None):
        p = os.path.join(self._hz_dir(hz), f"pred_{name}.npy")
        if not os.path.exists(p):
            return None
        try:
            arr = np.load(p, mmap_mode="c")
            if expected_len is not None and len(arr) != expected_len:
                log(f"  Stale cache {p} (len={len(arr)}, expected={expected_len}), dropping")
                os.remove(p)
                return None
            return arr
        except Exception as e:
            log(f"  Corrupted {p}: {e}, dropping")
            try: os.remove(p)
            except OSError: pass
            return None

    def save_df(self, hz, name, df):
        df.to_csv(os.path.join(self._hz_dir(hz), f"{name}.csv"), index=False)

    def save_json(self, hz, name, data):
        atomic_json_write(os.path.join(self._hz_dir(hz), f"{name}.json"), data)

    def load_json(self, hz, name):
        p = os.path.join(self._hz_dir(hz), f"{name}.json")
        if not os.path.exists(p):
            return {}
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def load_metrics(self, hz, stage):
        return self.load_json(hz, stage)

    def is_done(self, hz, stage):
        return stage in self.get_completed(hz)

    def horizon_complete(self, hz):
        return self.is_done(hz, "COMPLETE")


def _read_file(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_data():
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    for name, path in [("train", cfg.train_file),
                        ("val",   cfg.val_file),
                        ("test",  cfg.test_file)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}")
        fmt = "parquet" if path.endswith(".parquet") else "csv"
        log(f"  {name}.{fmt}  ({os.path.getsize(path)/1e6:.1f} MB)")

    train = _read_file(cfg.train_file)
    val   = _read_file(cfg.val_file)
    test  = _read_file(cfg.test_file)

    for df in [train, val, test]:
        df.sort_values(["container_id", "time_stamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    log(f"  sizes: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    assert train["time_stamp"].max() < val["time_stamp"].min(), "train/val overlap!"
    assert val["time_stamp"].max()   < test["time_stamp"].min(), "val/test overlap!"
    log("  temporal ordering: OK")
    log(f"  RAM after load: {mem_gb():.1f} GB")
    return train, val, test


def convert_csv_to_parquet():
    for name in ["train", "val", "test"]:
        csv_p = os.path.join(cfg.DATA_DIR, f"{name}.csv")
        pq_p  = os.path.join(cfg.DATA_DIR, f"{name}.parquet")
        if os.path.exists(csv_p) and not os.path.exists(pq_p):
            log(f"Converting {name}.csv -> .parquet ...")
            df = pd.read_csv(csv_p)
            df[df.select_dtypes("float64").columns] = \
                df.select_dtypes("float64").astype("float32")
            for col in df.select_dtypes("int64").columns:
                if df[col].min() >= -(2**31) and df[col].max() < 2**31:
                    df[col] = df[col].astype("int32")
            df.to_parquet(pq_p, index=False, compression="zstd")
            log(f"  {os.path.getsize(csv_p)/1e6:.0f} MB -> "
                f"{os.path.getsize(pq_p)/1e6:.0f} MB")


# rolling feature cache - computing rolling stats is expensive so we cache
# the result per split and reuse it across all 4 horizons
_rolling_cache = {}


def clear_rolling_cache(split_id=None):
    global _rolling_cache
    if split_id is None:
        _rolling_cache.clear()
        gc.collect()
        log("Rolling cache cleared (all splits)")
    elif split_id in _rolling_cache:
        del _rolling_cache[split_id]
        gc.collect()
        log(f"Rolling cache cleared for split '{split_id}'")


def _compute_rolling_only(df):
    """Compute the horizon-agnostic features: rolling stats, temporal encodings,
    and cross-resource ratios. We copy df first so we don't mess up the caller."""
    df = df.copy()
    df = df.sort_values(["container_id", "time_stamp"])
    g = df.groupby("container_id")

    roll_ops = []
    for metric in ["cpu", "mem"]:
        col = f"{metric}_util_percent"
        pre = f"{metric}_roll"
        for window in [6, 12, 24, 48]:
            for stat in ["mean", "std", "min", "max"]:
                roll_ops.append((f"{pre}_{stat}_{window}", col, window, stat))

    for col_name, col, window, fn in roll_ops:
        df[col_name] = g[col].transform(
            lambda x, w=window, f=fn: getattr(x.rolling(w, min_periods=1), f)()
        )

    # cyclic encoding for time features
    hour = (df["time_stamp"] % 86400) / 3600
    dow  = (df["time_stamp"] // 86400) % 7
    df["hour_sin"]       = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"]       = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin"]        = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos"]        = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    df["business_hours"] = ((hour >= 9) & (hour <= 17)).astype(np.float32)

    return df


def _get_cached_rolling(split_id, df):
    if split_id not in _rolling_cache:
        log(f"    computing rolling cache for '{split_id}'...")
        _rolling_cache[split_id] = _compute_rolling_only(df)
        log(f"    rolling cache '{split_id}' ready  [{mem_gb():.1f}GB]")
    return _rolling_cache[split_id]


def create_features(df, horizon_steps, split_id=None):
    """Build the full feature set for a given prediction horizon.

    All lag features are shifted by at least horizon_steps+1 so we never
    accidentally leak future information into training.
    """
    min_lag = horizon_steps + 1

    if split_id is not None:
        base = _get_cached_rolling(split_id, df).copy()
    else:
        base = _compute_rolling_only(df).copy()

    g = base.groupby("container_id")

    # lag features - we skip the recent lags (lag_1 through lag_horizon)
    # because those would be from the future at prediction time
    for offset in [0, 1, 2, 3, 5, 11, 23, 47]:
        lag = min_lag + offset
        if lag <= 96:
            base[f"cpu_lag_{lag}"] = g["cpu_util_percent"].shift(lag)
            base[f"mem_lag_{lag}"] = g["mem_util_percent"].shift(lag)
    if "disk_io_percent" in base.columns:
        for offset in [0, 5, 11]:
            lag = min_lag + offset
            if lag <= 96:
                base[f"disk_lag_{lag}"] = g["disk_io_percent"].shift(lag)

    # shift rolling features too - they're based on current values otherwise
    roll_cols = [c for c in base.columns if "_roll_" in c]
    for col in roll_cols:
        base[col] = g[col].shift(min_lag)

    cpu_lags = sorted([c for c in base.columns if c.startswith("cpu_lag_")],
                      key=lambda x: int(x.split("_")[-1]))
    mem_lags = sorted([c for c in base.columns if c.startswith("mem_lag_")],
                      key=lambda x: int(x.split("_")[-1]))

    # trend features - differences between lag values
    if len(cpu_lags) >= 2:
        base["cpu_trend_short"] = base[cpu_lags[0]] - base[cpu_lags[1]]
        base["mem_trend_short"] = base[mem_lags[0]] - base[mem_lags[1]]
    if len(cpu_lags) >= 3:
        base["cpu_trend_medium"] = base[cpu_lags[0]] - base[cpu_lags[2]]
        base["mem_trend_medium"] = base[mem_lags[0]] - base[mem_lags[2]]
    if len(cpu_lags) >= 4:
        base["cpu_trend_long"] = base[cpu_lags[0]] - base[cpu_lags[-1]]
        base["mem_trend_long"] = base[mem_lags[0]] - base[mem_lags[-1]]
    if "cpu_roll_mean_6" in base.columns and "cpu_roll_mean_24" in base.columns:
        base["cpu_momentum"] = base["cpu_roll_mean_6"] - base["cpu_roll_mean_24"]
        base["mem_momentum"] = base["mem_roll_mean_6"] - base["mem_roll_mean_24"]

    base["cpu_accel"] = g["cpu_util_percent"].transform(
        lambda x: x.diff().diff().shift(min_lag))
    base["mem_accel"] = g["mem_util_percent"].transform(
        lambda x: x.diff().diff().shift(min_lag))
    base["cpu_jerk"]  = g["cpu_util_percent"].transform(
        lambda x: x.diff().diff().diff().shift(min_lag))
    base["mem_jerk"]  = g["mem_util_percent"].transform(
        lambda x: x.diff().diff().diff().shift(min_lag))

    if "cpu_roll_std_12" in base.columns:
        base["cpu_volatility"] = base["cpu_roll_std_12"]
        base["mem_volatility"] = base["mem_roll_std_12"]
        base["cpu_cv"] = base["cpu_roll_std_12"] / (base["cpu_roll_mean_12"] + 1e-6)
        base["mem_cv"] = base["mem_roll_std_12"] / (base["mem_roll_mean_12"] + 1e-6)
    if "cpu_roll_max_12" in base.columns:
        base["cpu_range"] = base["cpu_roll_max_12"] - base["cpu_roll_min_12"]
        base["mem_range"] = base["mem_roll_max_12"] - base["mem_roll_min_12"]

    # cross-resource interaction features
    if len(cpu_lags) > 0 and len(mem_lags) > 0:
        fc, fm = base[cpu_lags[0]], base[mem_lags[0]]
        base["cpu_mem_ratio"]       = fc / (fm + 1e-6)
        base["cpu_mem_product"]     = fc * fm / 100
        base["cpu_mem_diff"]        = fc - fm
        base["resource_saturation"] = ((fc > 70) & (fm > 70)).astype(np.float32)
        if "cpu_trend_short" in base.columns:
            base["cpu_mem_corr_proxy"]  = (
                base["cpu_trend_short"] * base["mem_trend_short"]).clip(-100, 100)
            base["resource_divergence"] = (
                np.sign(base["cpu_trend_short"]) != np.sign(base["mem_trend_short"])
            ).astype(np.float32)

    # same-time-yesterday features for daily seasonality
    ppd = 288
    hist_lag = ppd + min_lag
    base["cpu_same_time_1d"] = g["cpu_util_percent"].shift(hist_lag)
    base["mem_same_time_1d"] = g["mem_util_percent"].shift(hist_lag)
    if len(cpu_lags) > 0:
        base["cpu_diff_1d"] = base[cpu_lags[0]] - base["cpu_same_time_1d"]
        base["mem_diff_1d"] = base[mem_lags[0]] - base["mem_same_time_1d"]

    # targets and naive baseline
    base["cpu_target"]   = g["cpu_util_percent"].shift(-horizon_steps)
    base["mem_target"]   = g["mem_util_percent"].shift(-horizon_steps)
    base["naive_cpu"]    = base["cpu_util_percent"]
    base["naive_mem"]    = base["mem_util_percent"]
    base["cpu_residual"] = base["cpu_target"] - base["naive_cpu"]
    base["mem_residual"] = base["mem_target"] - base["naive_mem"]

    # fill nans for features that are expected to sometimes be zero
    for pat in ["_std_", "volatility", "_cv", "_range", "accel", "jerk"]:
        cols = [c for c in base.columns if pat in c]
        if cols:
            base[cols] = base[cols].fillna(0)
    if len(cpu_lags) > 0:
        base["cpu_same_time_1d"] = base["cpu_same_time_1d"].fillna(base[cpu_lags[0]])
        base["mem_same_time_1d"] = base["mem_same_time_1d"].fillna(base[mem_lags[0]])
    base["cpu_diff_1d"] = base.get("cpu_diff_1d", pd.Series(0, index=base.index)).fillna(0)
    base["mem_diff_1d"] = base.get("mem_diff_1d", pd.Series(0, index=base.index)).fillna(0)

    return base


def create_features_sequential(train_df, val_df, test_df, horizon_steps):
    log("    train...")
    tr = create_features(train_df, horizon_steps, split_id="train")
    log("    val...")
    va = create_features(val_df,   horizon_steps, split_id="val")
    log("    test...")
    te = create_features(test_df,  horizon_steps, split_id="test")
    return tr, va, te


def get_all_feature_columns(df, horizon_steps):
    """Return the main feature column list used by ExtraTrees and the ensemble."""
    min_lag = horizon_steps + 1
    cols = []
    for c in df.columns:
        if c.startswith(("cpu_lag_", "mem_lag_", "disk_lag_")):
            try:
                if int(c.rsplit("_", 1)[1]) >= min_lag:
                    cols.append(c)
            except ValueError:
                pass
    for w in [6, 12, 24, 48]:
        for m in ["cpu", "mem"]:
            for s in ["mean", "std", "min", "max"]:
                c = f"{m}_roll_{s}_{w}"
                if c in df.columns: cols.append(c)
    # note: cross-resource features are intentionally not included here;
    # they go into the ablation superset but not the main ensemble training
    extras = [
        "cpu_trend_short", "cpu_trend_medium", "cpu_trend_long",
        "mem_trend_short", "mem_trend_medium", "mem_trend_long",
        "cpu_momentum", "mem_momentum",
        "cpu_accel", "mem_accel", "cpu_jerk", "mem_jerk",
        "cpu_volatility", "mem_volatility", "cpu_cv", "mem_cv",
        "cpu_range", "mem_range",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "business_hours",
        "cpu_same_time_1d", "mem_same_time_1d", "cpu_diff_1d", "mem_diff_1d",
    ]
    cols += [c for c in extras if c in df.columns]
    if "cluster_id" in df.columns:
        cols.append("cluster_id")
    return cols


def get_ablation_feature_columns(df, horizon_steps):
    """Superset of get_all_feature_columns - adds cross-resource features back
    so that ablation can show what happens when we remove them."""
    base_cols = get_all_feature_columns(df, horizon_steps)
    cross = [
        "cpu_mem_ratio", "cpu_mem_product", "cpu_mem_diff", "cpu_mem_corr_proxy",
        "resource_saturation", "resource_divergence",
    ]
    extra = [c for c in cross if c in df.columns and c not in base_cols]
    return base_cols + extra


def get_xgb_features(df, horizon_steps):
    """XGBoost gets lag and trend/accel features - no rolling stats (LightGBM handles those)."""
    min_lag = horizon_steps + 1
    cols = []
    for c in df.columns:
        if c.startswith(("cpu_lag_", "mem_lag_", "disk_lag_")):
            try:
                if int(c.rsplit("_", 1)[1]) >= min_lag: cols.append(c)
            except ValueError:
                pass
    for group in [
        ["cpu_trend_short", "cpu_trend_medium", "cpu_trend_long",
         "mem_trend_short", "mem_trend_medium", "mem_trend_long",
         "cpu_accel", "mem_accel", "cpu_jerk", "mem_jerk"],
        ["cpu_same_time_1d", "mem_same_time_1d", "cpu_diff_1d", "mem_diff_1d"],
    ]:
        cols += [c for c in group if c in df.columns]
    if "cluster_id" in df.columns:
        cols.append("cluster_id")
    return cols


def get_lgbm_features(df, horizon_steps):
    """LightGBM gets rolling stats and momentum - complementary to XGBoost."""
    min_lag = horizon_steps + 1
    cols = []
    for c in df.columns:
        if c.startswith(("cpu_lag_", "mem_lag_")):
            try:
                lag = int(c.rsplit("_", 1)[1])
                if min_lag <= lag <= min_lag + 5: cols.append(c)
            except ValueError:
                pass
    for w in [6, 12, 24, 48]:
        for m in ["cpu", "mem"]:
            for s in ["mean", "std", "min", "max"]:
                c = f"{m}_roll_{s}_{w}"
                if c in df.columns: cols.append(c)
    for group in [
        ["cpu_momentum", "mem_momentum"],
        ["cpu_volatility", "mem_volatility", "cpu_cv", "mem_cv",
         "cpu_range", "mem_range"],
        ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "business_hours"],
        ["cpu_same_time_1d", "mem_same_time_1d", "cpu_diff_1d", "mem_diff_1d"],
    ]:
        cols += [c for c in group if c in df.columns]
    if "cluster_id" in df.columns:
        cols.append("cluster_id")
    return cols


def categorize_features(feature_cols, horizon_steps):
    """Group features into named buckets for ablation analysis."""
    min_lag = horizon_steps + 1
    cats = {
        "lag_immediate": [], "lag_extended": [],
        "rolling_mean": [], "rolling_std": [], "rolling_range": [],
        "trend": [], "acceleration": [], "volatility": [],
        "temporal": [], "historical": [], "cross_resource": [], "cluster": [],
    }
    for col in feature_cols:
        if col == "cluster_id":
            cats["cluster"].append(col)
        elif "lag_" in col:
            try:
                lag = int(col.rsplit("_", 1)[1])
                bucket = "lag_immediate" if lag <= min_lag + 3 else "lag_extended"
                cats[bucket].append(col)
            except ValueError:
                cats["lag_extended"].append(col)
        elif "roll_std" in col:
            cats["rolling_std"].append(col)
        elif "roll_min" in col or "roll_max" in col:
            cats["rolling_range"].append(col)
        elif "roll_" in col:
            cats["rolling_mean"].append(col)
        elif "trend" in col or "momentum" in col:
            cats["trend"].append(col)
        elif "accel" in col or "jerk" in col:
            cats["acceleration"].append(col)
        elif "volatility" in col or "_range" in col or "_cv" in col:
            cats["volatility"].append(col)
        elif "hour" in col or "dow" in col or "business" in col:
            cats["temporal"].append(col)
        elif "same_time" in col or "diff_1d" in col:
            cats["historical"].append(col)
        elif "cpu_mem" in col or "saturation" in col or "divergence" in col:
            cats["cross_resource"].append(col)
    return {k: v for k, v in cats.items() if v}


def safe_extract_array(df, cols, dtype=np.float32):
    arr = df[cols].values.astype(dtype)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def calc_metrics(y_true, y_pred):
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def calc_metrics_with_ci(y_true, y_pred, n_boot=1000):
    """Bootstrap confidence intervals for R2 and MAE. Vectorized - all
    bootstrap samples at once to avoid a slow python loop."""
    yt = np.asarray(y_true, np.float64)
    yp = np.asarray(y_pred, np.float64)
    n  = len(yt)
    rng = np.random.RandomState(42)

    idx = rng.randint(0, n, size=(n_boot, n))
    ys  = yt[idx]
    ps  = yp[idx]

    ss_res = np.sum((ys - ps) ** 2, axis=1)
    ys_mean = ys.mean(axis=1, keepdims=True)
    ss_tot = np.sum((ys - ys_mean) ** 2, axis=1)
    r2s  = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
    maes = np.mean(np.abs(ys - ps), axis=1)

    return {
        "R2":  {"point":    float(r2_score(yt, yp)),
                "ci_lower": float(np.percentile(r2s,  2.5)),
                "ci_upper": float(np.percentile(r2s, 97.5))},
        "MAE": {"point":    float(mean_absolute_error(yt, yp)),
                "ci_lower": float(np.percentile(maes,  2.5)),
                "ci_upper": float(np.percentile(maes, 97.5))},
    }


def paired_ttest(y_true, y_baseline, y_model):
    yt  = np.asarray(y_true,     np.float64)
    yb  = np.asarray(y_baseline, np.float64)
    ym  = np.asarray(y_model,    np.float64)
    if np.allclose(yb, ym, atol=1e-10):
        return {"t_stat_mse": 0.0, "p_value_mse": 1.0,
                "t_stat_mae": 0.0, "p_value_mae": 1.0,
                "significant_005": False, "significant_001": False,
                "note": "predictions_identical"}
    t_mse, p2_mse = ttest_rel((yt-yb)**2, (yt-ym)**2)
    p_mse = p2_mse/2 if t_mse > 0 else 1 - p2_mse/2
    t_mae, p2_mae = ttest_rel(np.abs(yt-yb), np.abs(yt-ym))
    p_mae = p2_mae/2 if t_mae > 0 else 1 - p2_mae/2
    if not np.isfinite(p_mse): p_mse = 1.0
    if not np.isfinite(p_mae): p_mae = 1.0
    return {
        "t_stat_mse": float(t_mse), "p_value_mse": float(p_mse),
        "t_stat_mae": float(t_mae), "p_value_mae": float(p_mae),
        "significant_005": bool(p_mse < 0.05),
        "significant_001": bool(p_mse < 0.01),
    }


def diebold_mariano(y_true, pred1, pred2, h=1, power=2):
    """DM test with Harvey-Leybourne-Newbold small-sample correction."""
    e1 = np.asarray(y_true - pred1, np.float64)
    e2 = np.asarray(y_true - pred2, np.float64)
    d  = np.abs(e1)**power - np.abs(e2)**power
    n  = len(d)

    if n < 3:
        return {
            "dm_stat": 0.0, "dm_stat_hln": 0.0, "hln_factor": 0.0,
            "dm_p_one": 1.0, "dm_p_two": 1.0,
            "dm_sig005": False, "dm_sig001": False,
            "dm_bandwidth": 0, "dm_df": n-1,
            "note": f"insufficient_n={n}",
        }

    d_bar    = d.mean()
    d_demean = d - d_bar
    gamma_0  = np.mean(d_demean**2)
    max_lags = min(max(h-1, 1), n//4)
    gamma_sum = 0.0
    for k in range(1, max_lags+1):
        if k >= n: break
        gamma_sum += 2 * np.mean(d_demean[k:] * d_demean[:-k])
    var_d = (gamma_0 + gamma_sum) / n

    dm_stat   = d_bar / np.sqrt(max(var_d, 1e-12))
    hln_inner = (n + 1.0 - 2.0*h + h*(h-1.0)/n) / n
    hln_factor = np.sqrt(max(hln_inner, 0.0))
    dm_hln = hln_factor * dm_stat

    p_one = float(scipy_t.sf(dm_hln,      df=n-1))
    p_two = float(2.0 * scipy_t.sf(abs(dm_hln), df=n-1))

    return {
        "dm_stat":     float(dm_stat),
        "dm_stat_hln": float(dm_hln),
        "hln_factor":  float(hln_factor),
        "dm_p_one":    p_one, "dm_p_two": p_two,
        "dm_sig005":   bool(p_one < 0.05),
        "dm_sig001":   bool(p_one < 0.01),
        "dm_bandwidth": max_lags,
        "dm_df":        n-1,
    }


def directional_accuracy(y_true, y_pred, y_current):
    true_dir = np.sign(y_true - y_current)
    pred_dir = np.sign(y_pred - y_current)
    da_all   = float(np.mean(true_dir == pred_dir) * 100)
    rel_chg  = np.abs((y_true - y_current) / (y_current + 1e-8) * 100)
    sig_mask = rel_chg > 1.0
    n_sig    = int(sig_mask.sum())
    da_sig   = float(np.mean(true_dir[sig_mask] == pred_dir[sig_mask]) * 100
                     ) if n_sig > 0 else None
    return {"DA_all": da_all, "DA_significant": da_sig, "n_significant": n_sig}


def error_correlation_matrix(y_true, preds_dict):
    """Compute pairwise error correlations between models.
    Useful for checking whether ensemble members are actually diverse."""
    errors = {}
    for name, pred in preds_dict.items():
        if pred is None: continue
        valid = np.isfinite(pred)
        if valid.sum() <= 100: continue
        err = np.full_like(pred, np.nan)
        err[valid] = y_true[valid] - pred[valid]
        errors[name] = err

    result = {}
    names = list(errors.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            v = np.isfinite(errors[names[i]]) & np.isfinite(errors[names[j]])
            if v.sum() < 50: continue
            ei, ej = errors[names[i]][v], errors[names[j]][v]
            if np.std(ei) < 1e-12 or np.std(ej) < 1e-12:
                result[f"{names[i]}_vs_{names[j]}"] = None
                continue
            r = np.corrcoef(ei, ej)[0, 1]
            result[f"{names[i]}_vs_{names[j]}"] = float(r) if np.isfinite(r) else None
    return result


class NNLSEnsemble:
    """Non-negative least squares meta-learner. Weights are non-negative and
    sum to 1 so the ensemble is a convex combination of base model predictions."""

    def __init__(self):
        self.weights = None
        self.names   = None

    def fit(self, base_preds, y_true, names=None):
        self.names = names
        w, _ = scipy_nnls(base_preds, y_true)
        s = w.sum()
        self.weights = w / s if s > 1e-8 else \
            np.ones(base_preds.shape[1]) / base_preds.shape[1]
        return self

    def predict(self, base_preds):
        return base_preds @ self.weights

    def get_weights_dict(self):
        if self.names is not None and self.weights is not None:
            return {n: round(float(w), 4) for n, w in zip(self.names, self.weights)}
        return self.weights.tolist() if self.weights is not None else None

    def describe(self):
        d = self.get_weights_dict()
        if isinstance(d, dict):
            return "  ".join(f"{k}={v:.3f}" for k, v in d.items())
        return str(d)


class NNLSEnsembleEstimator(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper around the NNLS ensemble so we can pass it
    to the ACI calibration routine."""

    def __init__(self, xgb_params=None, lgbm_params=None, et_params=None,
                 lgbm_early_stop=20):
        self.xgb_params      = xgb_params or {}
        self.lgbm_params     = lgbm_params or {}
        self.et_params       = et_params or {}
        self.lgbm_early_stop = lgbm_early_stop

    def fit(self, X, y):
        X_clean = np.nan_to_num(X, nan=0.0)
        split   = int(len(X_clean) * 0.85)
        X_tr, X_ev = X_clean[:split], X_clean[split:]
        y_tr, y_ev = y[:split],       y[split:]

        self.xgb_model_ = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model_.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)

        self.lgb_model_ = lgb.LGBMRegressor(**self.lgbm_params)
        self.lgb_model_.fit(
            X_tr, y_tr, eval_set=[(X_ev, y_ev)],
            callbacks=[lgb.early_stopping(self.lgbm_early_stop, verbose=False)])

        self.et_model_ = ExtraTreesRegressor(**self.et_params)
        self.et_model_.fit(X_tr, y_tr)

        p_xgb = self.xgb_model_.predict(X_ev)
        p_lgb = self.lgb_model_.predict(X_ev)
        p_et  = self.et_model_.predict(X_ev)
        self.nnls_ = NNLSEnsemble()
        self.nnls_.fit(np.column_stack([p_xgb, p_lgb, p_et]), y_ev,
                       names=["XGB", "LGB", "ET"])
        return self

    def predict(self, X):
        X_clean = np.nan_to_num(X, nan=0.0)
        p_xgb = self.xgb_model_.predict(X_clean)
        p_lgb = self.lgb_model_.predict(X_clean)
        p_et  = self.et_model_.predict(X_clean)
        return self.nnls_.predict(np.column_stack([p_xgb, p_lgb, p_et]))


if HAS_TORCH:

    class SlidingWindowDataset(Dataset):
        def __init__(self, X, y, lookback, stride=1):
            self.X       = torch.FloatTensor(X)
            self.y       = torch.FloatTensor(y)
            self.lookback = lookback
            self.stride   = stride
            max_start = len(X) - lookback + 1
            self.indices = list(range(0, max(0, max_start), stride))

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            start = self.indices[idx]
            return (self.X[start: start + self.lookback],
                    self.y[start + self.lookback - 1])

    class BiLSTMForecaster(nn.Module):
        def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
            super().__init__()
            # round hidden to multiple of 8 for GPU efficiency
            hidden = max(8, (hidden // 8) * 8)
            self.lstm = nn.LSTM(
                n_features, hidden, layers, batch_first=True,
                bidirectional=True,
                dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.fc   = nn.Linear(hidden * 2, 1)
            self._hidden = hidden

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.drop(out[:, -1, :])
            return self.fc(out).squeeze(-1)

        def forward_lstm_only(self, x):
            """Return the last hidden state without dropout or the final linear
            layer - used by MC dropout to cache the LSTM pass and only re-run
            the cheap dropout+fc step multiple times."""
            with torch.no_grad():
                out, _ = self.lstm(x)
            return out[:, -1, :]

    def _make_dataloader(dataset, batch_size, shuffle=False):
        kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)
        if HAS_CUDA and cfg.DATALOADER_WORKERS > 0:
            kwargs.update(
                pin_memory=cfg.DATALOADER_PIN_MEMORY,
                num_workers=cfg.DATALOADER_WORKERS,
                persistent_workers=cfg.DATALOADER_PERSISTENT_WORKERS,
                prefetch_factor=cfg.DATALOADER_PREFETCH_FACTOR,
            )
        elif HAS_CUDA:
            kwargs["pin_memory"] = cfg.DATALOADER_PIN_MEMORY
        return DataLoader(dataset, **kwargs)

    def train_bilstm(X_train, y_train, X_val, y_val, n_features,
                     hidden=None, layers=None, dropout=None, lr=None,
                     batch_size=None, epochs=None, patience=None, verbose=True):
        hidden     = hidden     or cfg.BILSTM_HIDDEN
        layers     = layers     or cfg.BILSTM_LAYERS
        dropout    = dropout    or cfg.BILSTM_DROPOUT
        lr         = lr         or cfg.BILSTM_LR
        batch_size = batch_size or cfg.BILSTM_BATCH
        epochs     = epochs     or cfg.BILSTM_EPOCHS
        patience   = patience   or cfg.BILSTM_PATIENCE
        lookback   = cfg.BILSTM_LOOKBACK
        stride     = cfg.BILSTM_STRIDE
        use_amp    = cfg.USE_AMP and HAS_CUDA

        X_tr_c = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_va_c = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_c)
        X_va = sc.transform(X_va_c)

        train_ds = SlidingWindowDataset(X_tr, y_train, lookback, stride=stride)
        val_ds   = SlidingWindowDataset(X_va, y_val,   lookback, stride=1)

        if len(train_ds) < 100:
            if verbose: log("      too few samples for BiLSTM, skipping")
            return None, sc, -999.0

        train_dl = _make_dataloader(train_ds, batch_size, shuffle=True)
        val_dl   = _make_dataloader(val_ds,   batch_size)

        model   = BiLSTMForecaster(n_features, hidden, layers, dropout).to(DEVICE)
        opt     = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best_loss, best_state, wait = float("inf"), None, 0

        for epoch in range(epochs):
            model.train()
            t_loss, nb = 0.0, 0
            for xb, yb in train_dl:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = loss_fn(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                else:
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                t_loss += loss.item(); nb += 1

            model.eval()
            vp_list, vt_list = [], []
            with torch.no_grad():
                for xb, yb in val_dl:
                    vp_list.append(model(xb.to(DEVICE, non_blocking=True)).cpu().numpy())
                    vt_list.append(yb.numpy())
            vp = np.concatenate(vp_list)
            vt = np.concatenate(vt_list)
            v_loss = float(np.mean((vp - vt)**2))

            if v_loss < best_loss:
                best_loss  = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
                tag = " *"
            else:
                wait += 1
                tag = ""
            if verbose and ((epoch+1) % 10 == 0 or epoch == 0 or tag):
                v_r2 = float(r2_score(vt, vp)) if len(vt) > 1 else 0.0
                log(f"      ep {epoch+1:3d}/{epochs}  train={t_loss/max(nb,1):.6f}  "
                    f"val={v_loss:.6f}  R2={v_r2:.4f}{tag}")
            if wait >= patience:
                if verbose: log(f"      early stop at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
        model = model.cpu().eval()

        preds    = _predict_bilstm_raw(model, X_va, lookback, batch_size)
        n_pred   = len(preds)
        final_r2 = float(r2_score(y_val[lookback-1:lookback-1+n_pred], preds)
                         ) if n_pred > 0 else -999.0
        if verbose: log(f"      BiLSTM final val R2: {final_r2:.4f}")
        return model, sc, final_r2

    def _predict_bilstm_raw(model, X_scaled, lookback, batch_size):
        ds = SlidingWindowDataset(X_scaled, np.zeros(len(X_scaled)), lookback, stride=1)
        dl = _make_dataloader(ds, batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                preds.append(model(xb).numpy())
        return np.concatenate(preds) if preds else np.array([])

    def predict_bilstm(model, scaler, X, batch_size=None):
        batch_size = batch_size or cfg.BILSTM_BATCH
        X_sc = scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        return _predict_bilstm_raw(model, X_sc, cfg.BILSTM_LOOKBACK, batch_size)

    def mc_dropout_predict(model, scaler, X, n_passes=None):
        """Fast MC dropout - we run the LSTM forward pass once to get hidden
        states, then re-apply dropout+fc multiple times without re-running LSTM.
        Much faster than running the whole model N times."""
        n_passes = n_passes or cfg.MC_PASSES
        X_sc  = scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
        ds    = SlidingWindowDataset(X_sc, np.zeros(len(X_sc)), cfg.BILSTM_LOOKBACK, stride=1)
        dl    = _make_dataloader(ds, cfg.BILSTM_BATCH, shuffle=False)

        model.eval()
        hidden_states = []
        with torch.no_grad():
            for xb, _ in dl:
                h = model.forward_lstm_only(xb)
                hidden_states.append(h)
        if not hidden_states:
            return np.array([]), np.array([])
        H = torch.cat(hidden_states, dim=0)

        all_preds = []
        for _ in range(n_passes):
            with torch.no_grad():
                h_drop = F.dropout(H, p=model.drop.p, training=True)
                preds  = model.fc(h_drop).squeeze(-1).numpy()
            all_preds.append(preds)

        arr = np.array(all_preds)
        return arr.mean(axis=0), arr.std(axis=0)


def optuna_bilstm(X_train, y_train, X_val, y_val, n_features,
                  n_trials=None, timeout=None):
    n_trials = n_trials or cfg.OPTUNA_TRIALS
    timeout  = timeout  or cfg.OPTUNA_TIMEOUT
    if not (bilstm_enabled() and HAS_OPTUNA) or n_trials <= 0:
        log("    Optuna or PyTorch unavailable -- using default BiLSTM params")
        return {}

    log(f"    Optuna: {n_trials} trials, timeout={timeout}s, HyperbandPruner")

    def objective(trial):
        hidden  = trial.suggest_categorical("hidden", [32, 64, 128])
        layers  = trial.suggest_int("layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        lr_val  = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch   = trial.suggest_categorical("batch_size", [2048, 4096, 8192])

        m, _, r2 = train_bilstm(
            X_train, y_train, X_val, y_val, n_features,
            hidden=hidden, layers=layers, dropout=dropout,
            lr=lr_val, batch_size=batch, epochs=20, patience=5, verbose=False)

        if HAS_OPTUNA and hasattr(trial, "report"):
            trial.report(r2 if np.isfinite(r2) else -999.0, step=0)
            if trial.should_prune():
                del m; gc.collect()
                if HAS_CUDA: torch.cuda.empty_cache()
                raise optuna.TrialPruned()

        del m; gc.collect()
        if HAS_CUDA: torch.cuda.empty_cache()
        return r2

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=3, max_resource=20, reduction_factor=3)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=3, multivariate=True),
        pruner=pruner)

    # seed with defaults so we always at least evaluate the known-good config
    study.enqueue_trial({
        "hidden": cfg.BILSTM_HIDDEN,
        "layers": cfg.BILSTM_LAYERS,
        "dropout": cfg.BILSTM_DROPOUT,
        "lr": cfg.BILSTM_LR,
        "batch_size": cfg.BILSTM_BATCH,
    })

    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    if study.best_trial:
        bp = study.best_params
        log(f"    best: R2={study.best_value:.4f}  "
            f"hidden={bp['hidden']}  layers={bp['layers']}  "
            f"dropout={bp['dropout']:.2f}  lr={bp['lr']:.1e}  "
            f"batch={bp['batch_size']}")
        return bp
    return {}


def _aci_single(fitted_estimator, X_val, y_val, X_test, y_test, alpha, gamma):
    """Single ACI run at a fixed gamma value. Uses a circular numpy buffer
    to keep the residual history bounded in memory."""
    X_val_c  = np.nan_to_num(X_val,  nan=0.0)
    X_test_c = np.nan_to_num(X_test, nan=0.0)

    cal_preds  = fitted_estimator.predict(X_val_c)
    cal_resids = np.abs(y_val - cal_preds).astype(np.float64)
    test_preds = fitted_estimator.predict(X_test_c)

    K = cfg.ACI_RESIDUAL_BUFFER_SIZE

    buf = np.empty(K, dtype=np.float64)
    n_init = min(K, len(cal_resids))
    buf[:n_init] = cal_resids[-n_init:]
    buf_fill = n_init
    buf_ptr  = n_init % K

    n_test  = len(y_test)
    lower   = np.empty(n_test, dtype=np.float32)
    upper   = np.empty(n_test, dtype=np.float32)
    alpha_t = float(alpha)

    for i in range(n_test):
        active = buf[:buf_fill] if buf_fill < K else buf
        q_t    = float(np.quantile(active, np.clip(1.0 - alpha_t, 0.0, 1.0))) if len(active) > 0 else 0.0
        lower[i] = test_preds[i] - q_t
        upper[i] = test_preds[i] + q_t
        err_t    = 1 if (y_test[i] < lower[i] or y_test[i] > upper[i]) else 0
        alpha_t  = float(np.clip(alpha_t + gamma * (alpha - err_t), 0.01, 0.99))
        buf[buf_ptr % K] = abs(float(y_test[i]) - float(test_preds[i]))
        buf_ptr += 1
        if buf_fill < K:
            buf_fill += 1

    return lower, upper, test_preds


def run_cqr(fitted_estimator, X_val, y_val, X_test, y_test, alpha=None):
    """AgACI (Zaffran et al., ICML 2022) - adaptive conformal intervals.
    Runs ACI at several gamma values and aggregates using exponential weights
    based on which gamma had the best coverage so far."""
    alpha = alpha or cfg.CQR_ALPHA
    log("    ACI calibration (AgACI -- Zaffran et al. ICML 2022)...")

    gamma_list    = cfg.ACI_GAMMA_SEARCH
    all_lower     = []
    all_upper     = []
    all_preds     = []
    expert_losses = np.zeros(len(gamma_list), dtype=np.float64)

    # find the index of the default gamma by value comparison
    default_idx = None
    for gi, gv in enumerate(gamma_list):
        if abs(gv - cfg.ACI_GAMMA) < 1e-9:
            default_idx = gi
            break
    if default_idx is None:
        default_idx = 0

    for gidx, gamma in enumerate(gamma_list):
        lo, hi, tp = _aci_single(
            fitted_estimator, X_val, y_val, X_test, y_test, alpha, gamma)
        all_lower.append(lo)
        all_upper.append(hi)
        all_preds.append(tp)
        missed = ((y_test < lo) | (y_test > hi)).astype(np.float64)
        expert_losses[gidx] = missed.mean()

    tp_default = all_preds[default_idx]

    eta    = 10.0
    log_w  = -eta * expert_losses
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()

    lower = sum(w * lo for w, lo in zip(weights, all_lower))
    upper = sum(w * hi for w, hi in zip(weights, all_upper))

    coverage  = float(np.mean((y_test >= lower) & (y_test <= upper)) * 100)
    widths    = upper - lower
    mean_w    = float(np.mean(widths))
    below     = np.maximum(lower - y_test, 0)
    above     = np.maximum(y_test - upper, 0)
    int_score = float(np.mean(widths + (2/alpha)*below + (2/alpha)*above))
    target    = (1 - alpha) * 100
    cal_err   = abs(coverage - target)

    per_gamma = {}
    for gidx, gamma in enumerate(gamma_list):
        lo_g, hi_g = all_lower[gidx], all_upper[gidx]
        cov_g = float(np.mean((y_test >= lo_g) & (y_test <= hi_g)) * 100)
        per_gamma[f"gamma_{gamma}"] = round(cov_g, 2)

    log(f"      AgACI coverage: {coverage:.1f}%  target: {target:.0f}%  "
        f"cal_error: {cal_err:.1f}pp  mean_width: {mean_w:.3f}")
    log(f"      expert weights: {dict(zip(gamma_list, weights.round(3).tolist()))}")
    log(f"      per-gamma cov:  {per_gamma}")
    log(f"      ACI buffer size: {cfg.ACI_RESIDUAL_BUFFER_SIZE} residuals")

    return {
        "coverage":           coverage,
        "target_coverage":    target,
        "calibration_error":  cal_err,
        "mean_width":         mean_w,
        "median_width":       float(np.median(widths)),
        "interval_score":     int_score,
        "lower":              lower,
        "upper":              upper,
        "point_pred":         tp_default,
        "method":             "AgACI",
        "aci_gamma_weights":  dict(zip(gamma_list, weights.round(4).tolist())),
        "per_gamma_coverage": per_gamma,
        "buffer_size":        cfg.ACI_RESIDUAL_BUFFER_SIZE,
    }


def _fit_stat_model(method, history, horizon_steps):
    if len(history) < cfg.STAT_MIN_HISTORY: return None
    if method == "ets":
        try:
            n = len(history)
            if n > 576:
                m = ExponentialSmoothing(history, trend="add", damped_trend=True,
                    seasonal="add", seasonal_periods=288,
                    initialization_method="estimated")
            elif n > 20:
                m = ExponentialSmoothing(history, trend="add", damped_trend=True,
                    initialization_method="estimated")
            else:
                m = ExponentialSmoothing(history, initialization_method="estimated")
            return m.fit(optimized=True, method="L-BFGS-B")
        except Exception:
            return None
    elif method == "arima":
        try:
            return pm.auto_arima(history, start_p=1, max_p=3, start_q=1, max_q=3,
                max_d=2, seasonal=False, stepwise=True, suppress_warnings=True,
                error_action="ignore", n_fits=20)
        except Exception:
            return None


def _forecast_h_steps(method, model_obj, horizon_steps):
    try:
        if method == "ets":   return float(model_obj.forecast(horizon_steps)[-1])
        if method == "arima": return float(model_obj.predict(n_periods=horizon_steps)[-1])
    except Exception:
        return None


def _process_one_container(cid, method, hist, test_idx, test_vals,
                            horizon_steps, default_preds):
    preds = default_preds.copy()
    stride = max(1, len(test_idx) // 20)
    expanding = list(hist[-cfg.STAT_MAX_HISTORY:])
    n_ok = 0
    try:
        for i in range(0, len(test_idx), stride):
            h_arr = np.array(expanding[-cfg.STAT_MAX_HISTORY:])
            if len(h_arr) < cfg.STAT_MIN_HISTORY: continue
            mdl = _fit_stat_model(method, h_arr, horizon_steps)
            if mdl is None: continue
            fc = _forecast_h_steps(method, mdl, horizon_steps)
            if fc is None or not np.isfinite(fc): continue
            end_i = min(i + stride, len(test_idx))
            for j in range(i, end_i):
                preds[test_idx[j]] = fc
            n_ok += 1
            expanding.extend(test_vals[i:end_i].tolist())
        return preds, n_ok > 0, n_ok
    except Exception:
        return default_preds.copy(), False, 0


def run_statistical_baselines(train_df, test_df, horizon_steps,
                               y_target_test, naive_pred):
    """Run ETS and ARIMA on a subset of containers in parallel.
    These are slow so we only run on STAT_MAX_CONTAINERS containers."""
    stat_results = {}
    ets_preds    = naive_pred.copy()
    arima_preds  = naive_pred.copy()

    test_cids = test_df["container_id"].values
    test_cpu  = test_df["cpu_util_percent"].values.astype(np.float32)

    unique_cids, counts = np.unique(test_cids, return_counts=True)
    container_ids = unique_cids[np.argsort(-counts)][:cfg.STAT_MAX_CONTAINERS]
    log(f"  ETS/ARIMA on {len(container_ids)} containers (parallel, n_jobs={cfg.STAT_N_JOBS})")

    train_by_cid = {
        cid: train_df[train_df["container_id"] == cid]
                 .sort_values("time_stamp")["cpu_util_percent"].values
        for cid in container_ids
        if (train_df["container_id"] == cid).sum() >= cfg.STAT_MIN_HISTORY
    }
    test_by_cid = {
        cid: (np.where(test_cids == cid)[0], test_cpu[test_cids == cid])
        for cid in container_ids
        if (test_cids == cid).sum() > 0
    }

    for method, has_lib in [("ets", HAS_STATSMODELS), ("arima", HAS_PMDARIMA)]:
        if not has_lib:
            stat_results[method] = {**calc_metrics(y_target_test, naive_pred),
                                    "DA": 0.0, "note": "library_unavailable"}
            continue

        work = []
        for cid in container_ids:
            if cid not in train_by_cid or cid not in test_by_cid:
                continue
            hist = train_by_cid[cid]
            test_idx, test_vals = test_by_cid[cid]
            work.append((cid, hist, test_idx, test_vals))

        results = Parallel(n_jobs=cfg.STAT_N_JOBS, backend="loky")(
            delayed(_process_one_container)(
                cid, method, hist, test_idx, test_vals,
                horizon_steps, naive_pred.copy())
            for cid, hist, test_idx, test_vals in work
        )

        preds_arr = naive_pred.copy()
        n_ok, n_fail, n_fc = 0, 0, 0
        for (container_preds, ok, nfc) in results:
            changed = container_preds != naive_pred
            preds_arr[changed] = container_preds[changed]
            n_ok   += ok
            n_fail += not ok
            n_fc   += nfc
        n_fail += len(container_ids) - len(work)

        log(f"  {method.upper()}: {n_ok} ok, {n_fc} forecasts, {n_fail} fallback")
        if method == "ets":   ets_preds   = preds_arr
        else:                 arima_preds = preds_arr

        m  = calc_metrics(y_target_test, preds_arr)
        da = directional_accuracy(y_target_test, preds_arr, test_cpu)
        stat_results[method] = {
            **m, "DA": da["DA_significant"],
            "n_success": n_ok, "n_fail": n_fail, "n_rolling_forecasts": n_fc}

    return stat_results, ets_preds, arima_preds


def train_linear_regression(X_train, y_train, X_test):
    pipe = Pipeline([
        ("sc",    StandardScaler()),
        ("ridge", Ridge(alpha=cfg.RIDGE_ALPHA, solver="lsqr", max_iter=1000))
    ])
    pipe.fit(np.nan_to_num(X_train, nan=0.0), y_train)
    return pipe.predict(np.nan_to_num(X_test, nan=0.0))


def get_oof_tree(ModelClass, params, X_train, y_train, X_test, name,
                 n_splits=5, model_type="xgb"):
    """Out-of-fold predictions using time series cross-validation.
    Returns (oof_predictions, averaged_test_predictions)."""
    n   = len(y_train)
    oof = np.full(n, np.nan)
    test_preds = []

    if model_type == "et":
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0)

    sub_ratio = cfg.OOF_SUBSAMPLE
    if sub_ratio < 1.0:
        n_sub = int(n * sub_ratio)
        step  = max(1, n // n_sub)
        sub_idx = np.arange(0, n, step)[:n_sub]
        X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
        log(f"    {name} OOF subsampled: {n:,} -> {len(sub_idx):,}")
    else:
        X_sub, y_sub, sub_idx = X_train, y_train, np.arange(n)

    first_val_start = None
    for fold, (tr_idx, val_idx) in enumerate(
            TimeSeriesSplit(n_splits=n_splits).split(X_sub)):
        if fold == 0: first_val_start = val_idx[0]
        model = ModelClass(**params)
        if model_type == "lgbm":
            model.fit(X_sub[tr_idx], y_sub[tr_idx],
                      eval_set=[(X_sub[val_idx], y_sub[val_idx])],
                      callbacks=[lgb.early_stopping(cfg.LGBM_EARLY_STOP, verbose=False)])
        elif model_type == "xgb":
            model.fit(X_sub[tr_idx], y_sub[tr_idx],
                      eval_set=[(X_sub[val_idx], y_sub[val_idx])], verbose=False)
        else:
            model.fit(X_sub[tr_idx], y_sub[tr_idx])

        orig_val = sub_idx[val_idx] if sub_ratio < 1.0 else val_idx
        oof[orig_val] = model.predict(X_sub[val_idx])
        test_preds.append(model.predict(X_test))

    # fill the first fold gap (no validation data for those rows) by
    # training a small separate model on an 80/20 split of that window
    nan_mask = np.isnan(oof)
    n_nan    = int(nan_mask.sum())
    if n_nan > 0 and first_val_start is not None and first_val_start > 0:
        try:
            fill_model = ModelClass(**params)
            fill_idx   = np.arange(0, first_val_start)
            split_pt   = int(len(fill_idx) * 0.8)
            if split_pt > 50:
                tr_i, va_i = fill_idx[:split_pt], fill_idx[split_pt:]
                if model_type == "lgbm":
                    fill_model.fit(X_sub[tr_i], y_sub[tr_i],
                                   eval_set=[(X_sub[va_i], y_sub[va_i])],
                                   callbacks=[lgb.early_stopping(
                                       cfg.LGBM_EARLY_STOP, verbose=False)])
                elif model_type == "xgb":
                    fill_model.fit(X_sub[tr_i], y_sub[tr_i],
                                   eval_set=[(X_sub[va_i], y_sub[va_i])],
                                   verbose=False)
                else:
                    fill_model.fit(X_sub[tr_i], y_sub[tr_i])
                fill_X = X_train[nan_mask] if sub_ratio < 1.0 else X_sub[nan_mask]
                if model_type == "et":
                    fill_X = np.nan_to_num(fill_X, nan=0.0)
                oof[nan_mask] = fill_model.predict(fill_X)
                del fill_model
        except Exception:
            oof[nan_mask] = 0.0

    valid = ~np.isnan(oof)
    if valid.sum() > 0:
        log(f"    {name} OOF R2: {r2_score(y_train[valid], oof[valid]):.4f}"
            f"  (infilled {n_nan})")
    if not test_preds:
        log(f"    WARNING: {name} produced no test predictions, returning zeros")
        return oof, np.zeros(len(X_test), dtype=np.float32)
    return oof, np.mean(test_preds, axis=0)


def get_oof_bilstm(X_train, y_train, X_test, n_features, bilstm_params,
                   n_splits=3):
    if not bilstm_enabled(): return None, None

    n   = len(y_train)
    oof = np.full(n, np.nan)
    test_preds = []
    lb = cfg.BILSTM_LOOKBACK

    hidden  = bilstm_params.get("hidden",     cfg.BILSTM_HIDDEN)
    layers  = bilstm_params.get("layers",     cfg.BILSTM_LAYERS)
    dropout = bilstm_params.get("dropout",    cfg.BILSTM_DROPOUT)
    lr_val  = bilstm_params.get("lr",         cfg.BILSTM_LR)
    batch   = bilstm_params.get("batch_size", cfg.BILSTM_BATCH)

    for fold, (tr_idx, va_idx) in enumerate(
            TimeSeriesSplit(n_splits=n_splits).split(X_train)):
        log(f"    BiLSTM OOF fold {fold+1}/{n_splits}")
        model, scaler, _ = train_bilstm(
            X_train[tr_idx], y_train[tr_idx],
            X_train[va_idx], y_train[va_idx], n_features,
            hidden=hidden, layers=layers, dropout=dropout,
            lr=lr_val, batch_size=batch,
            epochs=cfg.BILSTM_EPOCHS, patience=cfg.BILSTM_PATIENCE,
            verbose=False)
        if model is None: continue

        va_pred = predict_bilstm(model, scaler, X_train[va_idx], batch)
        start   = lb - 1
        n_pred  = min(len(va_pred), len(va_idx) - start)
        if n_pred > 0:
            oof[va_idx[start: start + n_pred]] = va_pred[:n_pred]

        te_pred = predict_bilstm(model, scaler, X_test, batch)
        test_preds.append(te_pred)
        del model; gc.collect()
        if HAS_CUDA: torch.cuda.empty_cache()

    valid = ~np.isnan(oof)
    if valid.sum() > 0:
        log(f"    BiLSTM OOF R2: {r2_score(y_train[valid], oof[valid]):.4f}")

    if test_preds:
        min_len  = min(len(p) for p in test_preds)
        test_avg = np.mean([p[:min_len] for p in test_preds], axis=0)
    else:
        test_avg = None
    return oof, test_avg


def run_ablation(X_train, y_train_res, X_val, y_val_res, X_test, y_test_res,
                 feature_cols, categories, target_name="cpu"):
    """Leave-one-group-out ablation. Train with all features, then train
    with each feature group removed and report the R2 drop."""
    results = []
    baseline = lgb.LGBMRegressor(**cfg.ABLATION_PARAMS)
    baseline.fit(X_train, y_train_res,
                 eval_set=[(X_val, y_val_res)],
                 callbacks=[lgb.early_stopping(20, verbose=False)])
    base_pred = baseline.predict(X_test)
    base_r2   = r2_score(y_test_res, base_pred)
    base_mae  = mean_absolute_error(y_test_res, base_pred)
    results.append({
        "Experiment": "Baseline", "Features_Removed": "None",
        "N_Features": len(feature_cols), "R2": base_r2,
        "MAE": base_mae, "R2_Drop": 0.0, "MAE_Increase": 0.0})

    for cat, cols in categories.items():
        if not cols: continue
        keep = [i for i, c in enumerate(feature_cols) if c not in cols]
        if not keep: continue
        m = lgb.LGBMRegressor(**cfg.ABLATION_PARAMS)
        m.fit(X_train[:, keep], y_train_res,
              eval_set=[(X_val[:, keep], y_val_res)],
              callbacks=[lgb.early_stopping(20, verbose=False)])
        pred = m.predict(X_test[:, keep])
        r2   = r2_score(y_test_res, pred)
        mae  = mean_absolute_error(y_test_res, pred)
        results.append({
            "Experiment": f"Without {cat}", "Features_Removed": cat,
            "N_Features": len(keep), "R2": r2, "MAE": mae,
            "R2_Drop": base_r2 - r2, "MAE_Increase": mae - base_mae})
    return pd.DataFrame(results)


def _load_cached(ckpt, hz, model_name, stage_name=None, expected_len=None):
    """Try to load a cached prediction + its metrics dict.
    Returns (pred, metrics) or (None, None) if not available.
    Logs a warning if the prediction file exists but the metrics key is missing,
    which can happen if the process was killed mid-write."""
    sn = stage_name or model_name
    if not ckpt.is_done(hz, sn): return None, None
    pred = ckpt.load_pred(hz, model_name, expected_len=expected_len)
    m    = ckpt.load_metrics(hz, sn)
    if pred is not None:
        if model_name in m:
            return pred, m[model_name]
        else:
            log(f"  WARNING: cache hit for pred '{model_name}' but key missing "
                f"from '{sn}.json' -- likely partial write; recomputing stage")
    return None, None


def cleanup():
    gc.collect()
    if HAS_TORCH and HAS_CUDA: torch.cuda.empty_cache()


def train_horizon(horizon_name, train_df, val_df, test_df, ckpt):
    """Main training function for a single prediction horizon.
    Runs all 14 stages in order, skipping any that are already checkpointed."""
    if ckpt.horizon_complete(horizon_name):
        log(f"  {horizon_name}: already complete, skipping")
        return None

    horizon_steps = cfg.HORIZONS[horizon_name]
    horizon_min   = horizon_steps * (cfg.SAMPLING_INTERVAL // 60)
    min_lag       = horizon_steps + 1
    t0 = time.time()

    print(f"\n{'=' * 70}")
    print(f"  {horizon_name} ({horizon_min}min ahead, min_lag={min_lag})")
    print(f"{'=' * 70}")

    metrics, stat_tests = {}, {}

    log("Building features...")
    train_fe, val_fe, test_fe = create_features_sequential(
        train_df, val_df, test_df, horizon_steps)

    all_cols  = get_all_feature_columns(train_fe, horizon_steps)
    xgb_cols  = get_xgb_features(train_fe, horizon_steps)
    lgbm_cols = get_lgbm_features(train_fe, horizon_steps)
    log(f"  Features: all={len(all_cols)}  XGB={len(xgb_cols)}  LGBM={len(lgbm_cols)}")

    required = ["cpu_target", "cpu_residual", "naive_cpu",
                "mem_target", "mem_residual", "naive_mem"]
    train_fe = train_fe.dropna(subset=required).reset_index(drop=True)
    val_fe   = val_fe.dropna(subset=required).reset_index(drop=True)
    test_fe  = test_fe.dropna(subset=required).reset_index(drop=True)
    log(f"  Clean: train={len(train_fe):,}  val={len(val_fe):,}  "
        f"test={len(test_fe):,}  [{mem_gb():.1f}GB]")

    log("Extracting arrays...")
    X_xgb_tr  = train_fe[xgb_cols].values.astype(np.float32)
    X_xgb_val = val_fe[xgb_cols].values.astype(np.float32)
    X_xgb_te  = test_fe[xgb_cols].values.astype(np.float32)
    X_lgb_tr  = train_fe[lgbm_cols].values.astype(np.float32)
    X_lgb_val = val_fe[lgbm_cols].values.astype(np.float32)
    X_lgb_te  = test_fe[lgbm_cols].values.astype(np.float32)
    X_all_tr  = safe_extract_array(train_fe, all_cols)
    X_all_val = safe_extract_array(val_fe, all_cols)
    X_all_te  = safe_extract_array(test_fe, all_cols)

    # extract ablation arrays before freeing the dataframes
    abl_cols  = get_ablation_feature_columns(train_fe, horizon_steps)
    X_abl_tr  = safe_extract_array(train_fe, abl_cols)
    X_abl_val = safe_extract_array(val_fe,   abl_cols)
    X_abl_te  = safe_extract_array(test_fe,  abl_cols)

    y_cpu_te   = test_fe["cpu_target"].values.astype(np.float32)
    y_cres_tr  = train_fe["cpu_residual"].values.astype(np.float32)
    y_cres_val = val_fe["cpu_residual"].values.astype(np.float32)
    y_cres_te  = test_fe["cpu_residual"].values.astype(np.float32)
    y_cur_cpu  = test_fe["cpu_util_percent"].values.astype(np.float32)
    naive_cpu  = test_fe["naive_cpu"].values.astype(np.float32)

    y_mem_te   = test_fe["mem_target"].values.astype(np.float32)
    y_mres_tr  = train_fe["mem_residual"].values.astype(np.float32)
    y_mres_val = val_fe["mem_residual"].values.astype(np.float32)
    y_mres_te  = test_fe["mem_residual"].values.astype(np.float32)
    y_cur_mem  = test_fe["mem_util_percent"].values.astype(np.float32)
    naive_mem  = test_fe["naive_mem"].values.astype(np.float32)

    te_cids_raw = test_fe["container_id"].values.copy()
    has_cluster = "cluster_id" in test_fe.columns
    cluster_ids = test_fe["cluster_id"].values.copy() if has_cluster else None
    n_test = len(y_cpu_te)
    n_feat = len(all_cols)

    test_fe_for_stat = test_fe[["container_id", "cpu_util_percent", "time_stamp"]].copy()
    del train_fe, val_fe, test_fe
    gc.collect()
    log(f"  DFs freed  [{mem_gb():.1f}GB]")

    all_preds = {}

    # ---- 1/14  NAIVE ----
    stage_t = time.time()
    log("1/14  Naive (CPU + MEM)")
    naive_cpu_m  = calc_metrics(y_cpu_te, naive_cpu)
    naive_cpu_da = directional_accuracy(y_cpu_te, naive_cpu, y_cur_cpu)
    metrics["naive"] = {**naive_cpu_m, "DA": naive_cpu_da["DA_significant"]}
    all_preds["naive"] = naive_cpu
    ckpt.save_pred(horizon_name, "naive", naive_cpu)

    naive_mem_m  = calc_metrics(y_mem_te, naive_mem)
    naive_mem_da = directional_accuracy(y_mem_te, naive_mem, y_cur_mem)
    metrics["naive_mem"] = {**naive_mem_m, "DA": naive_mem_da["DA_significant"]}
    all_preds["naive_mem"] = naive_mem
    ckpt.save_pred(horizon_name, "naive_mem", naive_mem)
    ckpt.mark_done(horizon_name, "naive",
                   {"naive": metrics["naive"], "naive_mem": metrics["naive_mem"]},
                   elapsed_s=time.time()-stage_t)
    log(f"  CPU  R2={naive_cpu_m['R2']:.4f}  MAE={naive_cpu_m['MAE']:.3f}")
    log(f"  MEM  R2={naive_mem_m['R2']:.4f}  MAE={naive_mem_m['MAE']:.3f}")

    if check_interrupted(): return metrics

    # ---- 2-3/14  ETS + ARIMA ----
    cached_ets,   m_ets   = _load_cached(ckpt, horizon_name, "ets",
                                          "stat_baselines", expected_len=n_test)
    cached_arima, m_arima = _load_cached(ckpt, horizon_name, "arima",
                                          "stat_baselines", expected_len=n_test)
    if cached_ets is not None and cached_arima is not None:
        log("2-3/14  ETS + ARIMA: cached")
        all_preds["ets"]   = cached_ets;   metrics["ets"]   = m_ets
        all_preds["arima"] = cached_arima; metrics["arima"] = m_arima
    else:
        stage_t = time.time()
        log("2-3/14  ETS + ARIMA (parallel rolling)...")
        stat_res, ets_preds, arima_preds = run_statistical_baselines(
            train_df, test_fe_for_stat, horizon_steps, y_cpu_te, naive_cpu)
        metrics["ets"]   = stat_res.get("ets",   {**naive_cpu_m, "DA": 0.0})
        metrics["arima"] = stat_res.get("arima", {**naive_cpu_m, "DA": 0.0})
        all_preds["ets"]   = ets_preds
        all_preds["arima"] = arima_preds
        ckpt.save_pred(horizon_name, "ets",   ets_preds)
        ckpt.save_pred(horizon_name, "arima", arima_preds)
        ckpt.mark_done(horizon_name, "stat_baselines",
                       {"ets": metrics["ets"], "arima": metrics["arima"]},
                       elapsed_s=time.time()-stage_t)
        for n in ["ets", "arima"]:
            log(f"  {n.upper()}: R2={metrics[n]['R2']:.4f}  MAE={metrics[n]['MAE']:.3f}")

    del test_fe_for_stat; gc.collect()
    if check_interrupted(): return metrics

    # ---- 4/14  LINEAR REGRESSION ----
    cached_lr, m_lr = _load_cached(ckpt, horizon_name, "linear_reg",
                                    expected_len=n_test)
    if cached_lr is not None:
        log("4/14  Linear Regression: cached")
        all_preds["linear_reg"] = cached_lr; metrics["linear_reg"] = m_lr
    else:
        stage_t = time.time()
        log("4/14  Linear Regression (Ridge, solver=lsqr)")
        lr_res  = train_linear_regression(X_all_tr, y_cres_tr, X_all_te)
        lr_pred = naive_cpu + lr_res
        lr_m    = calc_metrics(y_cpu_te, lr_pred)
        lr_da   = directional_accuracy(y_cpu_te, lr_pred, y_cur_cpu)
        metrics["linear_reg"] = {**lr_m, "DA": lr_da["DA_significant"]}
        all_preds["linear_reg"] = lr_pred
        ckpt.save_pred(horizon_name, "linear_reg", lr_pred)
        ckpt.mark_done(horizon_name, "linear_reg",
                       {"linear_reg": metrics["linear_reg"]},
                       elapsed_s=time.time()-stage_t)
        log(f"  R2={lr_m['R2']:.4f}  vs naive: {(lr_m['R2']-naive_cpu_m['R2'])*100:+.2f}pp")

    if check_interrupted(): return metrics

    # ---- 5/14  XGBOOST ----
    cached_xgb, m_xgb = _load_cached(ckpt, horizon_name, "xgboost", expected_len=n_test)
    if cached_xgb is not None:
        log("5/14  XGBoost: cached")
        all_preds["xgboost"] = cached_xgb; metrics["xgboost"] = m_xgb
    else:
        stage_t = time.time()
        gpu_tag = " [GPU]" if cfg.XGB_PARAMS.get("device") == "cuda" else ""
        log(f"5/14  XGBoost{gpu_tag}")
        xgb_mod = xgb.XGBRegressor(**cfg.XGB_PARAMS)
        xgb_mod.fit(X_xgb_tr, y_cres_tr,
                    eval_set=[(X_xgb_val, y_cres_val)], verbose=False)
        xgb_pred = naive_cpu + xgb_mod.predict(X_xgb_te)
        xgb_m    = calc_metrics(y_cpu_te, xgb_pred)
        xgb_da   = directional_accuracy(y_cpu_te, xgb_pred, y_cur_cpu)
        metrics["xgboost"] = {**xgb_m, "DA": xgb_da["DA_significant"]}
        all_preds["xgboost"] = xgb_pred
        ckpt.save_pred(horizon_name, "xgboost", xgb_pred)
        elapsed_xgb = time.time() - stage_t
        ckpt.mark_done(horizon_name, "xgboost",
                       {"xgboost": metrics["xgboost"]}, elapsed_s=elapsed_xgb)
        log(f"  R2={xgb_m['R2']:.4f}  vs naive: "
            f"{(xgb_m['R2']-naive_cpu_m['R2'])*100:+.2f}pp  ({elapsed_xgb:.1f}s)")
        del xgb_mod

    if HAS_CUDA: torch.cuda.empty_cache()
    if check_interrupted(): return metrics

    # ---- 6/14  LIGHTGBM ----
    cached_lgb, m_lgb = _load_cached(ckpt, horizon_name, "lightgbm", expected_len=n_test)
    if cached_lgb is not None:
        log("6/14  LightGBM: cached")
        all_preds["lightgbm"] = cached_lgb; metrics["lightgbm"] = m_lgb
    else:
        stage_t = time.time()
        gpu_tag = " [GPU]" if cfg.LGBM_PARAMS.get("device") == "gpu" else ""
        log(f"6/14  LightGBM{gpu_tag}")
        lgb_mod = lgb.LGBMRegressor(**cfg.LGBM_PARAMS)
        lgb_mod.fit(X_lgb_tr, y_cres_tr,
                    eval_set=[(X_lgb_val, y_cres_val)],
                    callbacks=[lgb.early_stopping(cfg.LGBM_EARLY_STOP, verbose=False)])
        lgb_pred = naive_cpu + lgb_mod.predict(X_lgb_te)
        lgb_m    = calc_metrics(y_cpu_te, lgb_pred)
        lgb_da   = directional_accuracy(y_cpu_te, lgb_pred, y_cur_cpu)
        metrics["lightgbm"] = {**lgb_m, "DA": lgb_da["DA_significant"]}
        all_preds["lightgbm"] = lgb_pred
        ckpt.save_pred(horizon_name, "lightgbm", lgb_pred)
        ckpt.mark_done(horizon_name, "lightgbm",
                       {"lightgbm": metrics["lightgbm"]},
                       elapsed_s=time.time()-stage_t)
        log(f"  R2={lgb_m['R2']:.4f}  vs naive: "
            f"{(lgb_m['R2']-naive_cpu_m['R2'])*100:+.2f}pp")
        del lgb_mod

    if check_interrupted(): return metrics

    # ---- 7/14  EXTRATREES ----
    cached_et, m_et = _load_cached(ckpt, horizon_name, "extratrees", expected_len=n_test)
    if cached_et is not None:
        log("7/14  ExtraTrees: cached")
        all_preds["extratrees"] = cached_et; metrics["extratrees"] = m_et
    else:
        stage_t = time.time()
        log(f"7/14  ExtraTrees (max_depth={cfg.ET_PARAMS['max_depth']}, "
            f"max_samples={cfg.ET_PARAMS['max_samples']})")
        et_mod  = ExtraTreesRegressor(**cfg.ET_PARAMS)
        et_mod.fit(np.nan_to_num(X_all_tr, nan=0.0), y_cres_tr)
        et_pred = naive_cpu + et_mod.predict(np.nan_to_num(X_all_te, nan=0.0))
        et_m    = calc_metrics(y_cpu_te, et_pred)
        et_da   = directional_accuracy(y_cpu_te, et_pred, y_cur_cpu)
        metrics["extratrees"] = {**et_m, "DA": et_da["DA_significant"]}
        all_preds["extratrees"] = et_pred
        ckpt.save_pred(horizon_name, "extratrees", et_pred)
        elapsed_et = time.time() - stage_t
        ckpt.mark_done(horizon_name, "extratrees",
                       {"extratrees": metrics["extratrees"]}, elapsed_s=elapsed_et)
        log(f"  R2={et_m['R2']:.4f}  vs naive: "
            f"{(et_m['R2']-naive_cpu_m['R2'])*100:+.2f}pp  ({elapsed_et:.1f}s)")
        del et_mod

    del X_xgb_val, X_lgb_val; gc.collect()
    if check_interrupted(): return metrics

    # ---- 8/14  BiLSTM + Optuna + MC Dropout ----
    bilstm_params = {}
    bilstm_oof    = None
    bilstm_test   = None
    persisted_oof  = ckpt.load_pred(horizon_name, "bilstm_oof", expected_len=len(y_cres_tr))
    persisted_test = ckpt.load_pred(horizon_name, "bilstm_test")
    # check hetero_ensemble (not homo) because BiLSTM OOF is consumed by hetero stage
    need_bilstm    = not ckpt.is_done(horizon_name, "hetero_ensemble")

    if bilstm_enabled():
        if ckpt.is_done(horizon_name, "bilstm"):
            log("8/14  BiLSTM: cached")
            m_bi = ckpt.load_metrics(horizon_name, "bilstm")
            metrics["bilstm"] = m_bi.get("bilstm", {**naive_cpu_m, "DA": 0.0})
            bilstm_params = ckpt.load_json(horizon_name, "bilstm_params") or {}
            # load pred into all_preds so error correlation includes BiLSTM on resumes
            _bi_cached = ckpt.load_pred(horizon_name, "bilstm", expected_len=n_test)
            if _bi_cached is not None:
                all_preds["bilstm"] = _bi_cached
            if persisted_oof is not None and persisted_test is not None:
                bilstm_oof  = persisted_oof
                bilstm_test = persisted_test
            elif need_bilstm:
                log("    Recomputing BiLSTM OOF for ensemble...")
                bilstm_oof, bilstm_test = get_oof_bilstm(
                    X_all_tr, y_cres_tr, X_all_te, n_feat, bilstm_params)
                if bilstm_oof  is not None: ckpt.save_pred(horizon_name, "bilstm_oof",  bilstm_oof)
                if bilstm_test is not None: ckpt.save_pred(horizon_name, "bilstm_test", bilstm_test)
        else:
            stage_t = time.time()
            amp_tag = " [BF16]" if (cfg.USE_AMP and HAS_CUDA) else ""
            log(f"8/14  BiLSTM (Optuna HPO + fast MC Dropout){amp_tag}")
            try:
                bilstm_params = optuna_bilstm(
                    X_all_tr, y_cres_tr, X_all_val, y_cres_val, n_feat)
                ckpt.save_json(horizon_name, "bilstm_params", bilstm_params)

                bp = bilstm_params
                log("    Training final BiLSTM...")
                bilstm_model, bilstm_scaler, _ = train_bilstm(
                    X_all_tr, y_cres_tr, X_all_val, y_cres_val, n_feat,
                    hidden=bp.get("hidden"), layers=bp.get("layers"),
                    dropout=bp.get("dropout"), lr=bp.get("lr"),
                    batch_size=bp.get("batch_size"))

                if bilstm_model is not None:
                    bi_pred_res = predict_bilstm(bilstm_model, bilstm_scaler, X_all_te)
                    expected_offset = cfg.BILSTM_LOOKBACK - 1
                    # always derive offset from actual pred length to avoid shape mismatch
                    actual_offset = n_test - len(bi_pred_res)
                    if actual_offset != expected_offset:
                        log(f"  WARNING: BiLSTM offset {actual_offset} != expected "
                            f"{expected_offset} (lookback={cfg.BILSTM_LOOKBACK}, "
                            f"n_test={n_test}, n_pred={len(bi_pred_res)}). Using actual.")

                    bi_pred_abs = naive_cpu[actual_offset:] + bi_pred_res
                    bi_m  = calc_metrics(y_cpu_te[actual_offset:], bi_pred_abs)
                    bi_da = directional_accuracy(
                        y_cpu_te[actual_offset:], bi_pred_abs, y_cur_cpu[actual_offset:])
                    metrics["bilstm"] = {
                        **bi_m, "DA": bi_da["DA_significant"],
                        "n_aligned": len(bi_pred_res),
                        "lookback_offset": actual_offset,
                        "optuna_params": bilstm_params}
                    log(f"  R2={bi_m['R2']:.4f}  on {len(bi_pred_res):,} samples  "
                        f"vs naive: {(bi_m['R2']-naive_cpu_m['R2'])*100:+.2f}pp")

                    log("    MC Dropout uncertainty (fast path -- LSTM cached)...")
                    mc_mean, mc_std = mc_dropout_predict(
                        bilstm_model, bilstm_scaler, X_all_te)
                    mc_lower = mc_mean - 1.28 * mc_std
                    mc_upper = mc_mean + 1.28 * mc_std
                    mc_cov   = float(np.mean(
                        (y_cres_te[actual_offset:] >= mc_lower) &
                        (y_cres_te[actual_offset:] <= mc_upper)) * 100)
                    mc_width = float(np.mean(mc_upper - mc_lower))
                    metrics["bilstm"]["mc_dropout"] = {
                        "coverage_80": mc_cov, "mean_width": mc_width,
                        "mean_std": float(mc_std.mean()), "n_passes": cfg.MC_PASSES}
                    log(f"    MC coverage: {mc_cov:.1f}%  width: {mc_width:.3f}")

                    bi_full = np.full(n_test, np.nan)
                    bi_full[actual_offset:] = bi_pred_abs
                    ckpt.save_pred(horizon_name, "bilstm", bi_full)
                    all_preds["bilstm"] = bi_full
                    del bilstm_model, bilstm_scaler
                else:
                    metrics["bilstm"] = {**naive_cpu_m, "DA": 0.0,
                                         "note": "training_failed"}

                log("    BiLSTM OOF for ensemble...")
                bilstm_oof, bilstm_test = get_oof_bilstm(
                    X_all_tr, y_cres_tr, X_all_te, n_feat, bilstm_params)
                if bilstm_oof  is not None: ckpt.save_pred(horizon_name, "bilstm_oof",  bilstm_oof)
                if bilstm_test is not None: ckpt.save_pred(horizon_name, "bilstm_test", bilstm_test)

            except Exception as e:
                log(f"  BiLSTM ERROR: {e}")
                traceback.print_exc()
                metrics["bilstm"] = {**naive_cpu_m, "DA": 0.0, "note": str(e)}

            cleanup()
            ckpt.mark_done(horizon_name, "bilstm",
                           {"bilstm": metrics.get("bilstm", {})},
                           elapsed_s=time.time()-stage_t)
    else:
        metrics["bilstm"] = {**naive_cpu_m, "DA": 0.0, "note": "no_pytorch"}
        ckpt.mark_done(horizon_name, "bilstm", {"bilstm": metrics["bilstm"]})

    gc.collect()
    if check_interrupted(): return metrics

    # ---- 9/14  HOMO ENSEMBLE CPU ----
    homo_pred = None
    cached_homo, m_homo = _load_cached(ckpt, horizon_name, "homo_ensemble",
                                        expected_len=n_test)
    if cached_homo is not None:
        log("9/14  Homo Ensemble (CPU): cached")
        homo_pred = cached_homo
        all_preds["homo_ensemble"] = homo_pred
        metrics["homo_ensemble"]   = m_homo
    else:
        stage_t = time.time()
        sub_tag = f" [sub={cfg.OOF_SUBSAMPLE}]" if cfg.OOF_SUBSAMPLE < 1.0 else ""
        log(f"9/14  Homo Ensemble: XGB + LGB + ET -> NNLS{sub_tag}")
        xgb_oof, xgb_te_oof = get_oof_tree(
            xgb.XGBRegressor, cfg.XGB_PARAMS,
            X_xgb_tr, y_cres_tr, X_xgb_te, "XGB-oof",
            n_splits=cfg.N_CV_SPLITS, model_type="xgb")
        if HAS_CUDA: torch.cuda.empty_cache()

        lgbm_oof, lgbm_te_oof = get_oof_tree(
            lgb.LGBMRegressor, cfg.LGBM_PARAMS,
            X_lgb_tr, y_cres_tr, X_lgb_te, "LGB-oof",
            n_splits=cfg.N_CV_SPLITS, model_type="lgbm")
        et_oof, et_te_oof = get_oof_tree(
            ExtraTreesRegressor, cfg.ET_PARAMS,
            X_all_tr, y_cres_tr, X_all_te, "ET-oof",
            n_splits=cfg.N_CV_SPLITS, model_type="et")

        for nm, arr in [("oof_xgb", xgb_oof), ("oof_lgb", lgbm_oof),
                         ("oof_et", et_oof), ("oof_xgb_test", xgb_te_oof),
                         ("oof_lgb_test", lgbm_te_oof), ("oof_et_test", et_te_oof)]:
            ckpt.save_pred(horizon_name, nm, arr)

        valid_3 = ~np.isnan(xgb_oof) & ~np.isnan(lgbm_oof) & ~np.isnan(et_oof)
        nnls_homo = NNLSEnsemble()
        nnls_homo.fit(
            np.column_stack([xgb_oof[valid_3], lgbm_oof[valid_3], et_oof[valid_3]]),
            y_cres_tr[valid_3], names=["XGB", "LGB", "ET"])
        homo_res  = nnls_homo.predict(
            np.column_stack([xgb_te_oof, lgbm_te_oof, et_te_oof]))
        homo_pred = naive_cpu + homo_res
        homo_m    = calc_metrics(y_cpu_te, homo_pred)
        homo_da   = directional_accuracy(y_cpu_te, homo_pred, y_cur_cpu)
        metrics["homo_ensemble"] = {
            **homo_m, "DA": homo_da["DA_significant"],
            "nnls_weights": nnls_homo.get_weights_dict()}
        all_preds["homo_ensemble"] = homo_pred
        ckpt.save_pred(horizon_name, "homo_ensemble", homo_pred)
        elapsed_oof = time.time() - stage_t
        log(f"  R2={homo_m['R2']:.4f}  NNLS: {nnls_homo.describe()}  ({elapsed_oof:.1f}s)")
        ckpt.mark_done(horizon_name, "homo_ensemble",
                       {"homo_ensemble": metrics["homo_ensemble"]},
                       elapsed_s=elapsed_oof)
        del xgb_oof, lgbm_oof, et_oof
        del xgb_te_oof, lgbm_te_oof, et_te_oof

    if check_interrupted(): return metrics

    # ---- 10/14  HETERO ENSEMBLE CPU ----
    cached_het, m_het = _load_cached(ckpt, horizon_name, "hetero_ensemble",
                                      expected_len=n_test)
    if cached_het is not None:
        log("10/14  Hetero Ensemble (CPU): cached")
        all_preds["hetero_ensemble"] = cached_het
        metrics["hetero_ensemble"]   = m_het
    else:
        stage_t = time.time()
        log("10/14  Hetero Ensemble (CPU)")
        has_bilstm_oof = bilstm_oof is not None and bilstm_test is not None
        if not has_bilstm_oof:
            bilstm_oof  = ckpt.load_pred(horizon_name, "bilstm_oof",
                                          expected_len=len(y_cres_tr))
            bilstm_test = ckpt.load_pred(horizon_name, "bilstm_test")
            has_bilstm_oof = bilstm_oof is not None and bilstm_test is not None

        if has_bilstm_oof:
            xgb_oof_c = ckpt.load_pred(horizon_name, "oof_xgb")
            lgb_oof_c = ckpt.load_pred(horizon_name, "oof_lgb")
            et_oof_c  = ckpt.load_pred(horizon_name, "oof_et")
            xgb_te_c  = ckpt.load_pred(horizon_name, "oof_xgb_test")
            lgb_te_c  = ckpt.load_pred(horizon_name, "oof_lgb_test")
            et_te_c   = ckpt.load_pred(horizon_name, "oof_et_test")

            if all(x is not None for x in [xgb_oof_c, lgb_oof_c, et_oof_c]):
                valid_4 = (~np.isnan(xgb_oof_c) & ~np.isnan(lgb_oof_c) &
                           ~np.isnan(et_oof_c)  & ~np.isnan(bilstm_oof))
                n_valid = int(valid_4.sum())
                log(f"    4-model alignment: {n_valid:,} samples")

                if n_valid > 100:
                    meta_4    = np.column_stack([xgb_oof_c[valid_4], lgb_oof_c[valid_4],
                                                 et_oof_c[valid_4],  bilstm_oof[valid_4]])
                    nnls_het  = NNLSEnsemble()
                    nnls_het.fit(meta_4, y_cres_tr[valid_4],
                                 names=["XGB", "LGB", "ET", "BiLSTM"])

                    # fallback ensemble for the early rows where BiLSTM has no prediction
                    # (due to the lookback window at the start of the sequence)
                    valid_3f  = ~np.isnan(xgb_oof_c) & ~np.isnan(lgb_oof_c) & ~np.isnan(et_oof_c)
                    nnls_fb   = NNLSEnsemble()
                    nnls_fb.fit(
                        np.column_stack([xgb_oof_c[valid_3f], lgb_oof_c[valid_3f],
                                         et_oof_c[valid_3f]]),
                        y_cres_tr[valid_3f], names=["XGB", "LGB", "ET"])

                    n_bilstm  = len(bilstm_test)
                    n_pad     = n_test - n_bilstm
                    log(f"    BiLSTM covers {n_bilstm}/{n_test} samples (offset={n_pad})")
                    if n_pad < 0:
                        log(f"    WARNING: BiLSTM preds ({n_bilstm}) > n_test ({n_test}), clamping")
                        n_pad = 0

                    hetero_pred = np.zeros(n_test)
                    if n_pad > 0:
                        hetero_pred[:n_pad] = (
                            naive_cpu[:n_pad] + nnls_fb.predict(
                                np.column_stack([xgb_te_c[:n_pad],
                                                 lgb_te_c[:n_pad],
                                                 et_te_c[:n_pad]])))
                        hetero_pred[n_pad:] = (
                            naive_cpu[n_pad:] + nnls_het.predict(
                                np.column_stack([xgb_te_c[n_pad:], lgb_te_c[n_pad:],
                                                 et_te_c[n_pad:], bilstm_test])))
                    else:
                        hetero_pred = naive_cpu + nnls_het.predict(
                            np.column_stack([xgb_te_c, lgb_te_c, et_te_c,
                                             bilstm_test[:n_test]]))

                    het_m  = calc_metrics(y_cpu_te, hetero_pred)
                    het_da = directional_accuracy(y_cpu_te, hetero_pred, y_cur_cpu)
                    metrics["hetero_ensemble"] = {
                        **het_m, "DA": het_da["DA_significant"],
                        "nnls_weights": nnls_het.get_weights_dict(),
                        "n_hetero": n_test - n_pad, "n_fallback": n_pad}
                    all_preds["hetero_ensemble"] = hetero_pred
                    ckpt.save_pred(horizon_name, "hetero_ensemble", hetero_pred)
                    log(f"  R2={het_m['R2']:.4f}  NNLS: {nnls_het.describe()}")
                    log(f"    {n_test-n_pad:,} hetero, {n_pad} fallback")
                else:
                    log("    <100 aligned samples -- fallback to homo")
                    metrics["hetero_ensemble"] = {
                        **metrics.get("homo_ensemble", naive_cpu_m),
                        "note": "alignment_fail"}
                    all_preds["hetero_ensemble"] = homo_pred if homo_pred is not None else naive_cpu
                del xgb_oof_c, lgb_oof_c, et_oof_c
            else:
                log("    Tree OOF not available -- fallback to homo")
                metrics["hetero_ensemble"] = {
                    **metrics.get("homo_ensemble", naive_cpu_m), "note": "oof_unavailable"}
                all_preds["hetero_ensemble"] = homo_pred if homo_pred is not None else naive_cpu
        else:
            log("    BiLSTM OOF not available -- fallback to homo")
            metrics["hetero_ensemble"] = {
                **metrics.get("homo_ensemble", naive_cpu_m), "note": "bilstm_unavailable"}
            all_preds["hetero_ensemble"] = homo_pred if homo_pred is not None else naive_cpu

        if "hetero_ensemble" not in all_preds:
            all_preds["hetero_ensemble"] = homo_pred if homo_pred is not None else naive_cpu
        ckpt.save_pred(horizon_name, "hetero_ensemble", all_preds["hetero_ensemble"])
        ckpt.mark_done(horizon_name, "hetero_ensemble",
                       {"hetero_ensemble": metrics.get("hetero_ensemble", {})},
                       elapsed_s=time.time()-stage_t)
        # xgb_te_c etc. only assigned inside the has_bilstm_oof branch
        try:
            del xgb_te_c, lgb_te_c, et_te_c
        except NameError:
            pass

    # free bilstm arrays - not needed past this point
    if bilstm_oof is not None:
        del bilstm_oof
    if bilstm_test is not None:
        del bilstm_test
    bilstm_oof = bilstm_test = None
    gc.collect()

    if check_interrupted(): return metrics

    # ---- 11/14  HOMO ENSEMBLE MEM ----
    cached_mem_ens, m_mem_ens = _load_cached(ckpt, horizon_name,
                                              "homo_ensemble_mem", expected_len=n_test)
    if cached_mem_ens is not None:
        log("11/14  Homo Ensemble (MEM): cached")
        all_preds["homo_ensemble_mem"] = cached_mem_ens
        metrics["homo_ensemble_mem"]   = m_mem_ens
    else:
        stage_t = time.time()
        log("11/14  Homo Ensemble (MEM): XGB + LGB + ET -> NNLS")
        xgb_oof_m, xgb_te_m = get_oof_tree(
            xgb.XGBRegressor, cfg.XGB_PARAMS,
            X_xgb_tr, y_mres_tr, X_xgb_te, "XGB-MEM",
            n_splits=cfg.N_CV_SPLITS, model_type="xgb")
        if HAS_CUDA: torch.cuda.empty_cache()

        lgb_oof_m, lgb_te_m = get_oof_tree(
            lgb.LGBMRegressor, cfg.LGBM_PARAMS,
            X_lgb_tr, y_mres_tr, X_lgb_te, "LGB-MEM",
            n_splits=cfg.N_CV_SPLITS, model_type="lgbm")
        et_oof_m, et_te_m = get_oof_tree(
            ExtraTreesRegressor, cfg.ET_PARAMS,
            X_all_tr, y_mres_tr, X_all_te, "ET-MEM",
            n_splits=cfg.N_CV_SPLITS, model_type="et")

        valid_m  = ~np.isnan(xgb_oof_m) & ~np.isnan(lgb_oof_m) & ~np.isnan(et_oof_m)
        nnls_mem = NNLSEnsemble()
        nnls_mem.fit(
            np.column_stack([xgb_oof_m[valid_m], lgb_oof_m[valid_m],
                             et_oof_m[valid_m]]),
            y_mres_tr[valid_m], names=["XGB", "LGB", "ET"])
        mem_ens_pred = naive_mem + nnls_mem.predict(
            np.column_stack([xgb_te_m, lgb_te_m, et_te_m]))
        mem_ens_m  = calc_metrics(y_mem_te, mem_ens_pred)
        mem_ens_da = directional_accuracy(y_mem_te, mem_ens_pred, y_cur_mem)
        metrics["homo_ensemble_mem"] = {
            **mem_ens_m, "DA": mem_ens_da["DA_significant"],
            "nnls_weights": nnls_mem.get_weights_dict()}
        all_preds["homo_ensemble_mem"] = mem_ens_pred
        ckpt.save_pred(horizon_name, "homo_ensemble_mem", mem_ens_pred)
        log(f"  MEM R2={mem_ens_m['R2']:.4f}  NNLS: {nnls_mem.describe()}")
        ckpt.mark_done(horizon_name, "homo_ensemble_mem",
                       {"homo_ensemble_mem": metrics["homo_ensemble_mem"]},
                       elapsed_s=time.time()-stage_t)
        del xgb_oof_m, lgb_oof_m, et_oof_m
        del xgb_te_m, lgb_te_m, et_te_m

    if check_interrupted(): return metrics

    # ---- 12/14  ANALYSIS (AgACI) ----
    if not ckpt.is_done(horizon_name, "analysis"):
        stage_t = time.time()
        log("12/14  Post-processing + ACI uncertainty quantification...")

        cpu_preds = {k: v for k, v in all_preds.items() if not k.endswith("_mem")}
        corr = error_correlation_matrix(y_cpu_te, cpu_preds)

        for name, pred in all_preds.items():
            if name in ("naive", "naive_mem"): continue
            is_mem   = name.endswith("_mem")
            target   = y_mem_te  if is_mem else y_cpu_te
            baseline = naive_mem if is_mem else naive_cpu
            try:
                tt = paired_ttest(target, baseline, pred)
                dm = diebold_mariano(target, baseline, pred, h=horizon_steps)
                stat_tests[f"{name}_vs_naive"] = {**tt, **dm}
            except Exception:
                pass

        ci = {}
        for nm, (yt, yp) in [
            ("naive",             (y_cpu_te, naive_cpu)),
            ("homo_ensemble",     (y_cpu_te, all_preds.get("homo_ensemble", naive_cpu))),
            ("hetero_ensemble",   (y_cpu_te, all_preds.get("hetero_ensemble", naive_cpu))),
            ("naive_mem",         (y_mem_te, naive_mem)),
            ("homo_ensemble_mem", (y_mem_te, all_preds.get("homo_ensemble_mem", naive_mem))),
        ]:
            ci[nm] = calc_metrics_with_ci(yt, yp, cfg.N_BOOTSTRAP)

        cqr_result = None
        try:
            log("    Fitting ACI estimator (AgACI, same arch as homo_ensemble)...")
            cqr_est = NNLSEnsembleEstimator(
                xgb_params=cfg.XGB_PARAMS,
                lgbm_params=cfg.LGBM_PARAMS,
                et_params=cfg.ET_PARAMS,
                lgbm_early_stop=cfg.LGBM_EARLY_STOP)
            cqr_est.fit(X_all_tr, y_cres_tr)
            cqr_result = run_cqr(cqr_est, X_all_val, y_cres_val, X_all_te, y_cres_te)
            del cqr_est; gc.collect()
        except Exception as e:
            log(f"    WARNING: ACI failed ({e}), skipping UQ")

        cqr_summary = None
        if cqr_result is not None:
            np.savez_compressed(
                os.path.join(ckpt._hz_dir(horizon_name), "cqr_intervals.npz"),
                lower=cqr_result["lower"], upper=cqr_result["upper"],
                point_pred=cqr_result["point_pred"])
            cqr_summary = {k: v for k, v in cqr_result.items()
                           if k not in ("lower", "upper", "point_pred")}

        ckpt.mark_done(horizon_name, "analysis", {
            "error_correlation": corr,
            "statistical_tests": stat_tests,
            "ci": ci,
            "cqr": cqr_summary,
        }, elapsed_s=time.time()-stage_t)
    else:
        log("12/14  Analysis: cached")

    if check_interrupted(): return metrics

    # ---- 13/14  ABLATION ----
    if not ckpt.is_done(horizon_name, "ablation"):
        stage_t = time.time()
        log("13/14  Ablation -- CPU...")
        abl_cats = categorize_features(abl_cols, horizon_steps)
        ab_cpu = run_ablation(X_abl_tr, y_cres_tr, X_abl_val, y_cres_val,
                              X_abl_te, y_cres_te, abl_cols, abl_cats, "cpu")
        ckpt.save_df(horizon_name, "ablation_cpu", ab_cpu)

        log("Ablation -- MEM...")
        ab_mem = run_ablation(X_abl_tr, y_mres_tr, X_abl_val, y_mres_val,
                              X_abl_te, y_mres_te, abl_cols, abl_cats, "mem")
        ckpt.save_df(horizon_name, "ablation_mem", ab_mem)
        ckpt.mark_done(horizon_name, "ablation", elapsed_s=time.time()-stage_t)
        del X_abl_tr, X_abl_val, X_abl_te
    else:
        log("13/14  Ablation: cached")
        del X_abl_tr, X_abl_val, X_abl_te

    try: del X_all_val
    except NameError: pass
    gc.collect()

    if check_interrupted(): return metrics

    # ---- 14/14  CLUSTER ANALYSIS ----
    if has_cluster and not ckpt.is_done(horizon_name, "cluster_analysis"):
        stage_t = time.time()
        log("14/14  Per-cluster analysis...")
        cl_results = {}
        ens_pred   = all_preds.get("homo_ensemble", naive_cpu)
        for cid_val in sorted(np.unique(cluster_ids)):
            mask = cluster_ids == cid_val
            n    = int(mask.sum())
            if n < 50: continue
            y_c = y_cpu_te[mask]
            cl_results[f"cluster_{cid_val}"] = {
                "n": n,
                "ml_r2":          float(r2_score(y_c, ens_pred[mask])),
                "naive_r2":       float(r2_score(y_c, naive_cpu[mask])),
                "improvement_pp": float((r2_score(y_c, ens_pred[mask]) -
                                         r2_score(y_c, naive_cpu[mask])) * 100),
            }
        ckpt.mark_done(horizon_name, "cluster_analysis", cl_results,
                       elapsed_s=time.time()-stage_t)
    elif not has_cluster:
        log("14/14  Cluster analysis: no cluster_id column, skipping")
    else:
        log("14/14  Cluster analysis: cached")

    best    = metrics.get("hetero_ensemble", metrics.get("homo_ensemble", naive_cpu_m))
    best_r2 = best.get("R2", naive_cpu_m["R2"])
    be      = 1 - naive_cpu_m["R2"]
    me      = 1 - best_r2
    improvement = {
        "r2_improvement_pp":    float((best_r2 - naive_cpu_m["R2"]) * 100),
        "error_reduction_pct":  float((be - me) / be * 100 if be > 1e-6 else 0),
        "homo_vs_naive_pp":     float((metrics.get("homo_ensemble",     naive_cpu_m)["R2"] -
                                        naive_cpu_m["R2"]) * 100),
        "mem_ens_vs_naive_pp":  float((metrics.get("homo_ensemble_mem", naive_mem_m)["R2"] -
                                        naive_mem_m["R2"]) * 100),
    }
    ckpt.mark_done(horizon_name, "improvement", improvement)

    print(f"\n{'-' * 70}")
    print(f"  SUMMARY: {horizon_name} ({horizon_min}min, min_lag={min_lag})")
    print(f"{'-' * 70}")
    print(f"  {'Model':<25} {'R2':>8} {'MAE':>8} {'vs naive':>10}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10}")
    for name in ["naive", "ets", "arima", "linear_reg", "xgboost",
                 "lightgbm", "extratrees", "bilstm",
                 "homo_ensemble", "hetero_ensemble"]:
        if name in metrics and metrics[name].get("R2") is not None:
            m    = metrics[name]
            diff = (m["R2"] - naive_cpu_m["R2"]) * 100 if name != "naive" else 0
            print(f"  {name:<25} {m['R2']:>8.4f} {m.get('MAE', 0):>8.3f} {diff:>+9.2f}pp")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10}")
    for name in ["naive_mem", "homo_ensemble_mem"]:
        if name in metrics:
            m    = metrics[name]
            diff = (m["R2"] - naive_mem_m["R2"]) * 100 if name != "naive_mem" else 0
            print(f"  {name:<25} {m['R2']:>8.4f} {m.get('MAE', 0):>8.3f} {diff:>+9.2f}pp")

    elapsed = time.time() - t0
    log(f"{horizon_name} done in {elapsed/60:.1f}min  [{mem_gb():.1f}GB]")
    ckpt.mark_done(horizon_name, "COMPLETE")

    del all_preds
    del X_xgb_tr, X_lgb_tr, X_all_tr
    del X_xgb_te, X_lgb_te, X_all_te
    del y_cpu_te, y_cres_tr, y_cres_val, y_cres_te
    del y_mem_te, y_mres_tr, y_mres_val, y_mres_te
    cleanup()

    return metrics


def _horizon_worker_fn(horizon_name, cpu_start, n_cpus, output_dir, data_dir,
                       cfg_overrides=None):
    """Worker function that runs in a separate process for one horizon.
    Pins itself to a specific CPU slice so the 4 workers don't fight
    over cores when running gradient boosting in parallel."""
    try:
        os.sched_setaffinity(0, set(range(cpu_start, cpu_start + n_cpus)))
    except (AttributeError, OSError):
        pass

    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        os.environ[env_var] = str(n_cpus)

    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    cfg.OUTPUT_DIR = output_dir
    cfg.DATA_DIR   = data_dir

    cfg.XGB_PARAMS["n_jobs"]       = n_cpus
    cfg.ET_PARAMS["n_jobs"]        = n_cpus
    cfg.ABLATION_PARAMS["n_jobs"]  = n_cpus
    cfg.STAT_N_JOBS                = min(n_cpus, 10)
    # don't overwrite n_jobs=1 that was set for LightGBM GPU
    lgb_device = cfg.LGBM_PARAMS.get("device", "cpu")
    if lgb_device != "gpu":
        cfg.LGBM_PARAMS["n_jobs"] = n_cpus

    # per-worker log file so logs don't get garbled
    _setup_logging(output_dir, log_suffix=f"_{horizon_name}")
    log(f"[worker {horizon_name}] cpu={cpu_start}-{cpu_start+n_cpus-1}  "
        f"n_jobs={n_cpus}  pid={os.getpid()}")

    train_df, val_df, test_df = load_data()
    ckpt = Checkpoint(output_dir)

    try:
        result = train_horizon(horizon_name, train_df, val_df, test_df, ckpt)
        return horizon_name, result, None
    except Exception as e:
        tb = traceback.format_exc()
        log(f"[worker {horizon_name}] FAILED: {e}\n{tb}")
        return horizon_name, None, str(e)


def run_pipeline(horizons=None, rsync_dest=None):
    global _INTERRUPTED, _rolling_cache
    _INTERRUPTED = False
    _rolling_cache = {}

    ensure_dir(cfg.OUTPUT_DIR)
    ensure_dir(cfg.TMP_DIR)
    _setup_logging(cfg.OUTPUT_DIR)

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)
    ckpt        = Checkpoint(cfg.OUTPUT_DIR)

    if rsync_dest:
        start_rsync_backup(interval_minutes=15, rsync_dest=rsync_dest)

    print("\n" + "=" * 70)
    print("SPRINT 1 MAIN")
    print("=" * 70)
    print(f"Data dir: {cfg.DATA_DIR}")
    print(f"Output:   {cfg.OUTPUT_DIR}")
    print(f"Device:   {DEVICE if HAS_TORCH else 'CPU'}")
    print(f"Parallel: {cfg.PARALLEL_HORIZONS} "
          f"({cfg.N_CPU_WORKERS} workers x {cfg.CPUS_PER_WORKER} cores)")
    print(f"TF32:     {cfg.TF32 and HAS_CUDA}")
    print(f"XGBoost:  {cfg.XGB_PARAMS.get('device', 'cpu')}"
          f" / {cfg.XGB_PARAMS.get('tree_method', 'auto')}")
    print(f"LightGBM: device={cfg.LGBM_PARAMS.get('device', 'cpu')} "
          f"n_jobs={cfg.LGBM_PARAMS.get('n_jobs', -1)}")
    print(f"AMP:      {'BF16' if (cfg.USE_AMP and HAS_CUDA) else 'off'}")
    print(f"BiLSTM:   batch={cfg.BILSTM_BATCH}  stride={cfg.BILSTM_STRIDE}  "
          f"optuna_trials={cfg.OPTUNA_TRIALS}")
    print(f"ET:       max_depth={cfg.ET_PARAMS['max_depth']}  "
          f"max_samples={cfg.ET_PARAMS['max_samples']}  "
          f"n_estimators={cfg.ET_PARAMS['n_estimators']}")
    print(f"OOF sub:  {cfg.OOF_SUBSAMPLE}")
    print(f"ACI buf:  {cfg.ACI_RESIDUAL_BUFFER_SIZE} residuals")
    print(f"Stat n:   {cfg.STAT_MAX_CONTAINERS} containers (parallel)")
    print(f"DM test:  HLN-corrected, t(n-1) reference")
    print(f"CQR:      AgACI (Zaffran et al., ICML 2022)")
    print(f"Logs:     parallel -> run_<horizon>.log  sequential -> run.log")

    health_check()

    hz_list = horizons or list(cfg.HORIZONS.keys())
    for hz in hz_list:
        done = ckpt.get_completed(hz)
        if done:
            status = " (DONE)" if ckpt.horizon_complete(hz) else ""
            log(f"  {hz}: {len(done)} stages complete{status}")
    print()

    all_metrics = {}

    if cfg.PARALLEL_HORIZONS and len(hz_list) > 1:
        log(f"Running {len(hz_list)} horizons in PARALLEL "
            f"({cfg.N_CPU_WORKERS} workers x {cfg.CPUS_PER_WORKER} cores)")

        cfg_overrides = {
            "USE_AMP":                  cfg.USE_AMP,
            "NO_BILSTM":                cfg.NO_BILSTM,
            "OOF_SUBSAMPLE":            cfg.OOF_SUBSAMPLE,
            "OOF_SUBSAMPLE_SEED":       cfg.OOF_SUBSAMPLE_SEED,
            "BILSTM_BATCH":             cfg.BILSTM_BATCH,
            "ACI_RESIDUAL_BUFFER_SIZE": cfg.ACI_RESIDUAL_BUFFER_SIZE,
            "OPTUNA_TRIALS":            cfg.OPTUNA_TRIALS,
            "OPTUNA_TIMEOUT":           cfg.OPTUNA_TIMEOUT,
            "STAT_MAX_CONTAINERS":      cfg.STAT_MAX_CONTAINERS,
            "TF32":                     cfg.TF32,
        }

        ctx = mp.get_context("spawn")
        cpu_slots = [i * cfg.CPUS_PER_WORKER for i in range(cfg.N_CPU_WORKERS)]

        worker_args = [
            (hz, cpu_slots[i % len(cpu_slots)], cfg.CPUS_PER_WORKER,
             cfg.OUTPUT_DIR, cfg.DATA_DIR, cfg_overrides)
            for i, hz in enumerate(hz_list)
        ]

        failed = []
        with ProcessPoolExecutor(max_workers=cfg.N_CPU_WORKERS,
                                 mp_context=ctx) as pool:
            future_to_hz = {
                pool.submit(_horizon_worker_fn, *args): args[0]
                for args in worker_args
            }
            for future in as_completed(future_to_hz):
                hz = future_to_hz[future]
                try:
                    hz_name, result, err = future.result()
                    if err:
                        log(f"  {hz_name}: WORKER FAILED -- {err}")
                        failed.append(hz_name)
                    elif result:
                        all_metrics[hz_name] = result
                        log(f"  {hz_name}: complete")
                    else:
                        log(f"  {hz_name}: skipped (already done or no result)")
                except Exception as e:
                    log(f"  {hz}: FUTURE ERROR -- {e}")
                    failed.append(hz)

        if failed:
            log(f"WARNING: {len(failed)} horizon(s) failed: {failed}")
            log("Re-run to resume from checkpoint.")

    else:
        if not cfg.PARALLEL_HORIZONS:
            log("Running horizons SEQUENTIALLY (PARALLEL_HORIZONS=False)")
        train_df, val_df, test_df = load_data()

        for hz in hz_list:
            if check_interrupted():
                log(f"Skipping {hz} -- interrupt received"); break
            result = train_horizon(hz, train_df, val_df, test_df, ckpt)
            if result:
                all_metrics[hz] = result
            cleanup()

    log("Clearing rolling feature cache...")
    clear_rolling_cache()

    rows = []
    for hz in hz_list:
        hz_dir = os.path.join(cfg.OUTPUT_DIR, hz)
        if not os.path.isdir(hz_dir): continue
        m = {}
        for fname in os.listdir(hz_dir):
            if fname.endswith(".json") and fname != "_state.json":
                try:
                    with open(os.path.join(hz_dir, fname)) as f:
                        m.update(json.load(f))
                except (json.JSONDecodeError, IOError):
                    pass
        row = {"Horizon": hz}
        for name in ["naive", "ets", "arima", "linear_reg", "xgboost",
                     "lightgbm", "extratrees", "bilstm",
                     "homo_ensemble", "hetero_ensemble",
                     "naive_mem", "homo_ensemble_mem"]:
            if name in m:
                row[f"{name}_R2"]  = m[name].get("R2")
                row[f"{name}_MAE"] = m[name].get("MAE")
        if "r2_improvement_pp"   in m: row["cpu_improvement_pp"] = m["r2_improvement_pp"]
        if "mem_ens_vs_naive_pp" in m: row["mem_improvement_pp"] = m["mem_ens_vs_naive_pp"]
        rows.append(row)

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(f"{cfg.OUTPUT_DIR}/results_summary.csv", index=False)
        print(f"\n{'=' * 70}\nFINAL SUMMARY\n{'=' * 70}")
        print(summary.to_string(index=False))
    else:
        summary = pd.DataFrame()
        print("\nNo completed horizons yet.")

    shutil.rmtree(cfg.TMP_DIR, ignore_errors=True)
    signal.signal(signal.SIGINT, old_handler)

    if _INTERRUPTED:
        log("Ended early. Re-run to resume from checkpoint.")

    log(f"Results saved to {cfg.OUTPUT_DIR}")
    return summary


def print_timings():
    """Print a bar chart of how long each stage took per horizon."""
    print(f"\n{'=' * 70}\nSTAGE TIMINGS\n{'=' * 70}")
    for hz in cfg.HORIZONS:
        p = os.path.join(cfg.OUTPUT_DIR, hz, "_state.json")
        if not os.path.exists(p): continue
        with open(p) as f:
            state = json.load(f)
        timings = state.get("timings", {})
        if not timings: continue
        total = sum(timings.values())
        print(f"\n{hz} (total: {total:.0f}s = {total/60:.1f}min):")
        for stage, secs in sorted(timings.items(), key=lambda x: -x[1]):
            pct = secs / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 3)
            print(f"  {stage:<25} {secs:>7.1f}s  ({secs/60:.1f}min)  {pct:>4.1f}%  {bar}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Sprint 1 Main -- Kubernetes resource prediction")
    p.add_argument("--data-dir",   default="/workspace/thesis",
                   help="Directory containing train/val/test .parquet or .csv files")
    p.add_argument("--output-dir", default="/workspace/results/sprint1",
                   help="Where to write results and checkpoints")
    p.add_argument("--horizons",   nargs="+",
                   choices=list(Config.HORIZONS.keys()),
                   help="Which horizons to run (default: all)")
    p.add_argument("--rsync-dest", default=None,
                   help="Optional rsync destination for periodic backup")
    p.add_argument("--oof-subsample", type=float, default=None,
                   help="Override OOF subsample ratio (0 < x <= 1, default 1.0)")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable BF16 mixed precision")
    p.add_argument("--sequential", action="store_true",
                   help="Run horizons one at a time (easier to debug)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Number of parallel workers (default 4)")
    p.add_argument("--cpus-per-worker", type=int, default=None,
                   help="CPU cores per worker (default 20)")
    p.add_argument("--convert-parquet", action="store_true",
                   help="Convert CSVs to Parquet then exit")
    p.add_argument("--bilstm-batch", type=int, default=None,
                   help="Override BiLSTM batch size (default 4096)")
    p.add_argument("--no-bilstm", action="store_true",
                   help="Skip BiLSTM entirely")
    p.add_argument("--aci-buffer", type=int, default=None,
                   help="Override ACI residual buffer size (default 2000)")
    return p.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = _parse_args()

    cfg.DATA_DIR   = args.data_dir
    cfg.OUTPUT_DIR = args.output_dir

    if args.oof_subsample is not None:
        cfg.OOF_SUBSAMPLE = args.oof_subsample

    if args.no_amp:
        cfg.USE_AMP = False

    if args.sequential:
        cfg.PARALLEL_HORIZONS = False

    if args.n_workers is not None:
        cfg.N_CPU_WORKERS = args.n_workers

    if args.cpus_per_worker is not None:
        cfg.CPUS_PER_WORKER = args.cpus_per_worker

    if args.bilstm_batch is not None:
        cfg.BILSTM_BATCH = args.bilstm_batch

    if args.aci_buffer is not None:
        cfg.ACI_RESIDUAL_BUFFER_SIZE = args.aci_buffer

    if args.no_bilstm:
        cfg.NO_BILSTM = True
        print("BiLSTM disabled via --no-bilstm flag")

    if args.convert_parquet:
        convert_csv_to_parquet()
        print("Done. Re-run without --convert-parquet to train.")
        exit(0)

    results = run_pipeline(
        horizons=args.horizons,
        rsync_dest=args.rsync_dest)
