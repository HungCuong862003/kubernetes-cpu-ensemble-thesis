"""
chronos_benchmark.py — Zero-shot Chronos-Bolt foundation model benchmark.
 
Evaluates Amazon Chronos-Bolt (Small, 48M params) against the
domain-specific ensemble on all three datasets and four horizons.
No fine-tuning — pure zero-shot inference with 128-step context.
 
Datasets:
    Alibaba Cluster Trace 2018  (300 containers sampled, stride=3)
    Bitbrains GWA-T-12          (156 VMs, per-VM evaluation)
    ByteDance IaaS              (93 containers)
 
Inputs:
    Per-container .npy files     (ALI_PC_DIR/<hz>/)
    Raw Bitbrains CSVs           (BB_RAW_DIR/)
    ByteDance test.parquet       (BD_DATA)
 
Outputs:
    chronos_benchmark_results.csv              (OUTPUT_DIR)
    chronos_checkpoint.json                    (OUTPUT_DIR)
 
Infrastructure:
    Google Colab Pro+ (T4/A100 GPU).
"""
 
import os
import sys
import time
import json
import warnings
import traceback
import numpy as np
import pandas as pd
import torch
from chronos import ChronosBoltPipeline
 
warnings.filterwarnings("ignore")
 
# ── Paths (configured for Google Colab Pro+) ─────────────────
# Adjust these if running locally or on a different environment.
DRIVE = "/content/drive/MyDrive"
 
ALI_PC_DIR = f"{DRIVE}/thesis_upgrade/per_container"
BB_RAW_DIR = f"{DRIVE}/workspace/bitbrains/fastStorage/2013-8"
BB_RES_DIR = f"{DRIVE}/workspace/bitbrains/results"
BD_DATA    = f"{DRIVE}/workspace/thesis/bytedance/pipeline_data/test.parquet"
BD_RES_DIR = f"{DRIVE}/workspace/thesis/bytedance/results"
OUTPUT_DIR = f"{DRIVE}/thesis_upgrade/foundation_model_results"
CKPT_PATH  = f"{OUTPUT_DIR}/chronos_checkpoint.json"
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# ── Settings ───────────────────────────────────────────────────
ALI_STEPS = {"10min": 2,  "30min": 6,  "60min": 12, "120min": 24}
BD_STEPS  = {"10min": 1,  "30min": 3,  "60min": 6,  "120min": 12}
 
CONTEXT_LEN  = 128
BATCH_SIZE   = 64
ALI_SAMPLE_N = 300
STRIDE       = 3
MIN_LEN      = 30
SEED         = 42
 
# ── GPU auto-detection ─────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
# ── Load model ─────────────────────────────────────────────────
print("\nLoading Chronos-Bolt-Small...")
t0 = time.time()
pipeline = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map=DEVICE,
    dtype=torch.float32,
)
print(f"Ready in {time.time()-t0:.1f}s\n")
 
 
# ── Checkpoint helpers ─────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            ckpt = json.load(f)
        print(f"Resuming from checkpoint: {len(ckpt['results'])} results done")
        return ckpt
    return {"results": [], "done_keys": []}
 
def save_checkpoint(results, done_keys):
    with open(CKPT_PATH, "w") as f:
        json.dump({"results": results, "done_keys": done_keys}, f, indent=2)
 
 
# ── Metrics ────────────────────────────────────────────────────
def get_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return np.nan, np.nan, 0
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2  = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(yt - yp)))
    return float(r2), mae, len(yt)
 
 
# ── Chronos inference ──────────────────────────────────────────
def chronos_predict(context_list, horizon_steps):
    tensors = [torch.tensor(s.astype(np.float32)) for s in context_list]
    with torch.no_grad():
        forecast = pipeline.predict(tensors, prediction_length=horizon_steps)
    # quantile index 4 = median (0.5), last step of horizon
    result = forecast[:, 4, -1]
    if result.is_cuda:
        result = result.cpu()
    return result.numpy()
 
 
def rolling_eval(cpu, horizon_steps, stride=1):
    n = len(cpu)
    contexts, targets, naives = [], [], []
 
    for i in range(1, n - horizon_steps + 1, stride):
        target_idx = i + horizon_steps - 1
        if target_idx >= n:
            break
        ctx = cpu[max(0, i - CONTEXT_LEN):i]
        if len(ctx) < 3:
            continue
        contexts.append(ctx)
        targets.append(cpu[target_idx])
        naives.append(cpu[i - 1])
 
    if not contexts:
        return np.array([]), np.array([]), np.array([])
 
    preds = []
    for b in range(0, len(contexts), BATCH_SIZE):
        preds.extend(chronos_predict(contexts[b:b+BATCH_SIZE], horizon_steps))
 
    n_p = len(preds)
    return np.array(preds), np.array(naives[:n_p]), np.array(targets[:n_p])
 
 
# ══════════════════════════════════════════════════════════════
# ALIBABA
# ══════════════════════════════════════════════════════════════
def run_alibaba(hz_name):
    hz_steps = ALI_STEPS[hz_name]
    print(f"  [Alibaba {hz_name}]", end=" ", flush=True)
 
    hz_dir = f"{ALI_PC_DIR}/{hz_name}"
 
    # verify files exist before loading
    required = ["y_true.npy", "y_naive.npy", "pred_hetero_ensemble.npy",
                "container_index.csv", "raw_cpu_series.npz"]
    for fname in required:
        fpath = f"{hz_dir}/{fname}"
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing: {fpath}")
 
    y_true = np.load(f"{hz_dir}/y_true.npy")
    y_naive = np.load(f"{hz_dir}/y_naive.npy")
    y_ens   = np.load(f"{hz_dir}/pred_hetero_ensemble.npy")
    idx_df  = pd.read_csv(f"{hz_dir}/container_index.csv")
    raw     = np.load(f"{hz_dir}/raw_cpu_series.npz")
 
    naive_r2, naive_mae, _ = get_metrics(y_true, y_naive)
    ens_r2,   ens_mae,   _ = get_metrics(y_true, y_ens)
 
    np.random.seed(SEED)
    cids = np.random.choice(
        idx_df["container_id"].values,
        size=min(ALI_SAMPLE_N, len(idx_df)),
        replace=False)
 
    all_preds, all_naives, all_targets = [], [], []
    skipped = 0
    for i, cid in enumerate(cids):
        key = str(cid)
        if key not in raw:
            skipped += 1
            continue
        cpu = raw[key]
        if len(cpu) < MIN_LEN:
            skipped += 1
            continue
        cp, np_, ct = rolling_eval(cpu, hz_steps, stride=STRIDE)
        if len(cp) > 0:
            all_preds.extend(cp)
            all_naives.extend(np_)
            all_targets.extend(ct)
        if (i + 1) % 50 == 0:
            print(f"{i+1}", end=" ", flush=True)
 
    chron_r2, chron_mae, n = get_metrics(all_targets, all_preds)
    sub_naive_r2, _, _     = get_metrics(all_targets, all_naives)
 
    if skipped > 0:
        print(f"(skipped {skipped})", end=" ")
    print(f"| Naive={naive_r2:.4f} Ens={ens_r2:.4f} Chronos={chron_r2:.4f} (n={n})")
 
    return {
        "dataset": "Alibaba", "horizon": hz_name,
        "naive_r2": naive_r2,   "naive_mae": naive_mae,
        "ensemble_r2": ens_r2,  "ensemble_mae": ens_mae,
        "chronos_r2": chron_r2, "chronos_mae": chron_mae,
        "sub_naive_r2": sub_naive_r2, "chronos_n": n,
    }
 
 
# ══════════════════════════════════════════════════════════════
# BITBRAINS
# ══════════════════════════════════════════════════════════════
def run_bitbrains(hz_name):
    hz_steps = ALI_STEPS[hz_name]
    print(f"  [Bitbrains {hz_name}]", end=" ", flush=True)
 
    active_path = f"{BB_RES_DIR}/active_vms.csv"
    pvm_path    = f"{BB_RES_DIR}/bitbrains_per_vm.csv"
 
    if not os.path.exists(active_path):
        raise FileNotFoundError(f"Missing: {active_path}")
    if not os.path.exists(pvm_path):
        raise FileNotFoundError(f"Missing: {pvm_path}")
 
    active_ids = set(pd.read_csv(active_path)["vm_id"].astype(int))
    bb_pvm = pd.read_csv(pvm_path)
    bb_hz  = bb_pvm[bb_pvm["horizon"] == hz_name].set_index("vm_id")
 
    all_preds, all_naives, all_targets = [], [], []
    naive_r2_list, ens_r2_list = [], []
    vm_count, vm_errors = 0, 0
 
    csv_files = sorted([f for f in os.listdir(BB_RAW_DIR) if f.endswith(".csv")])
 
    for fname in csv_files:
        try:
            vm_id = int(fname.replace(".csv", ""))
        except ValueError:
            continue
        if vm_id not in active_ids or vm_id not in bb_hz.index:
            continue
 
        try:
            df = pd.read_csv(f"{BB_RAW_DIR}/{fname}", sep=';\t', engine='python')
            df.columns = df.columns.str.strip()
 
            if "CPU usage [%]" not in df.columns:
                # fallback: find any column with CPU and %
                cpu_col = None
                for c in df.columns:
                    if "CPU" in c and "%" in c:
                        cpu_col = c
                        break
                if cpu_col is None:
                    vm_errors += 1
                    continue
            else:
                cpu_col = "CPU usage [%]"
 
            cpu = df[cpu_col].values.astype(np.float32)
            cpu = cpu[np.isfinite(cpu)]
            if len(cpu) < MIN_LEN:
                continue
 
            split = int(len(cpu) * 0.8)
            n_test = len(cpu) - split
            if n_test < hz_steps + 2:
                continue
 
            contexts, targets, naives = [], [], []
            for i in range(1, n_test - hz_steps + 1, STRIDE):
                abs_i = split + i
                tgt_i = abs_i + hz_steps - 1
                if tgt_i >= len(cpu):
                    break
                ctx = cpu[max(0, abs_i - CONTEXT_LEN):abs_i]
                if len(ctx) < 3:
                    continue
                contexts.append(ctx)
                targets.append(cpu[tgt_i])
                naives.append(cpu[abs_i - 1])
 
            if len(contexts) < 2:
                continue
 
            preds = []
            for b in range(0, len(contexts), BATCH_SIZE):
                preds.extend(chronos_predict(contexts[b:b+BATCH_SIZE], hz_steps))
 
            n_p = len(preds)
            all_preds.extend(preds)
            all_naives.extend(naives[:n_p])
            all_targets.extend(targets[:n_p])
 
            row = bb_hz.loc[vm_id]
            if hasattr(row, "iloc") and len(row.shape) > 1:
                row = row.iloc[0]  # handle duplicate vm_id rows
            naive_r2_list.append(float(row["naive_r2"]))
            ens_r2_list.append(float(row["ml_r2"]))
 
            vm_count += 1
            if vm_count % 20 == 0:
                print(f"{vm_count}", end=" ", flush=True)
 
        except Exception as e:
            vm_errors += 1
            continue
 
    chron_r2, chron_mae, n = get_metrics(all_targets, all_preds)
    sub_naive_r2, _, _     = get_metrics(all_targets, all_naives)
    naive_r2 = float(np.nanmean(naive_r2_list)) if naive_r2_list else np.nan
    ens_r2   = float(np.nanmean(ens_r2_list))   if ens_r2_list   else np.nan
 
    if vm_errors > 0:
        print(f"(errors={vm_errors})", end=" ")
    print(f"| Naive={naive_r2:.4f} Ens={ens_r2:.4f} Chronos={chron_r2:.4f} ({vm_count} VMs, n={n})")
 
    return {
        "dataset": "Bitbrains", "horizon": hz_name,
        "naive_r2": naive_r2,   "naive_mae": np.nan,
        "ensemble_r2": ens_r2,  "ensemble_mae": np.nan,
        "chronos_r2": chron_r2, "chronos_mae": chron_mae,
        "sub_naive_r2": sub_naive_r2, "chronos_n": n,
    }
 
 
# ══════════════════════════════════════════════════════════════
# BYTEDANCE
# ══════════════════════════════════════════════════════════════
def run_bytedance(hz_name):
    hz_steps = BD_STEPS[hz_name]
    print(f"  [ByteDance {hz_name}]", end=" ", flush=True)
 
    # load pipeline metrics — confirmed nested structure
    naive_json_path = f"{BD_RES_DIR}/{hz_name}/naive.json"
    ens_json_path   = f"{BD_RES_DIR}/{hz_name}/hetero_ensemble.json"
 
    if not os.path.exists(naive_json_path):
        raise FileNotFoundError(f"Missing: {naive_json_path}")
    if not os.path.exists(ens_json_path):
        raise FileNotFoundError(f"Missing: {ens_json_path}")
 
    with open(naive_json_path) as f:
        naive_metrics = json.load(f)
    with open(ens_json_path) as f:
        ens_metrics = json.load(f)
 
    naive_r2  = float(naive_metrics["naive"]["R2"])
    naive_mae = float(naive_metrics["naive"]["MAE"])
    ens_r2    = float(ens_metrics["hetero_ensemble"]["R2"])
    ens_mae   = float(ens_metrics["hetero_ensemble"]["MAE"])
 
    if not os.path.exists(BD_DATA):
        raise FileNotFoundError(f"Missing: {BD_DATA}")
 
    df = pd.read_parquet(BD_DATA)
    df = df.sort_values(["container_id", "time_stamp"]).reset_index(drop=True)
 
    all_preds, all_naives, all_targets = [], [], []
    container_count = 0
 
    for cid, sub in df.groupby("container_id"):
        cpu = sub["cpu_util_percent"].values.astype(np.float32)
        cpu = cpu[np.isfinite(cpu)]
        if len(cpu) < MIN_LEN:
            continue
        cp, np_, ct = rolling_eval(cpu, hz_steps, stride=1)
        if len(cp) > 0:
            all_preds.extend(cp)
            all_naives.extend(np_)
            all_targets.extend(ct)
        container_count += 1
        if container_count % 20 == 0:
            print(f"{container_count}", end=" ", flush=True)
 
    chron_r2, chron_mae, n = get_metrics(all_targets, all_preds)
    sub_naive_r2, _, _     = get_metrics(all_targets, all_naives)
 
    print(f"| Naive={naive_r2:.4f} Ens={ens_r2:.4f} Chronos={chron_r2:.4f} ({container_count} containers, n={n})")
 
    return {
        "dataset": "ByteDance", "horizon": hz_name,
        "naive_r2": naive_r2,   "naive_mae": naive_mae,
        "ensemble_r2": ens_r2,  "ensemble_mae": ens_mae,
        "chronos_r2": chron_r2, "chronos_mae": chron_mae,
        "sub_naive_r2": sub_naive_r2, "chronos_n": n,
    }
 
 
# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("TASK 1: Chronos-Bolt Zero-Shot Benchmark")
    print("=" * 65)
 
    ckpt = load_checkpoint()
    results   = ckpt["results"]
    done_keys = set(ckpt["done_keys"])
    t_start   = time.time()
 
    ALL_JOBS = []
    for hz in ["10min", "30min", "60min", "120min"]:
        ALL_JOBS.append(("Alibaba",   hz, run_alibaba))
        ALL_JOBS.append(("Bitbrains", hz, run_bitbrains))
        ALL_JOBS.append(("ByteDance", hz, run_bytedance))
 
    for dataset_name, hz, run_fn in ALL_JOBS:
        key = f"{dataset_name}_{hz}"
        if key in done_keys:
            print(f"  [SKIP] {key} (already done)")
            continue
 
        print(f"\n--- {hz} ---")
        try:
            result = run_fn(hz)
            results.append(result)
            done_keys.add(key)
            save_checkpoint(results, list(done_keys))
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            print(f"  Skipping {key}.")
            continue
        except Exception as e:
            print(f"\n  UNEXPECTED ERROR in {key}: {e}")
            traceback.print_exc()
            print(f"  Saving checkpoint and continuing...")
            save_checkpoint(results, list(done_keys))
            continue
 
    # ── Save final CSV ─────────────────────────────────────────
    df_out = pd.DataFrame(results)
    out_path = f"{OUTPUT_DIR}/chronos_benchmark_results.csv"
    df_out.to_csv(out_path, index=False)
 
    elapsed = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"Done in {elapsed/60:.1f} min  |  Saved: {out_path}")
    print(f"{'='*65}")
 
    # ── Summary table ──────────────────────────────────────────
    print(f"\n{'Dataset':<12} {'Horizon':<8} {'Naive':>8} {'Ensemble':>10} {'Chronos':>9} {'SubNaive':>10} {'Winner':>8}")
    print("-" * 75)
    for _, row in df_out.iterrows():
        ens_str = f"{row.ensemble_r2:.4f}" if not np.isnan(row.ensemble_r2) else "     N/A"
        sn_str  = f"{row.sub_naive_r2:.4f}" if not np.isnan(row.sub_naive_r2) else "     N/A"
 
        # winner uses sub_naive (same sample) for fair comparison
        scores = {}
        if not np.isnan(row.sub_naive_r2):
            scores["Naive"] = row.sub_naive_r2
        if not np.isnan(row.ensemble_r2):
            scores["Ensemble"] = row.ensemble_r2
        if not np.isnan(row.chronos_r2):
            scores["Chronos"] = row.chronos_r2
        winner = max(scores, key=scores.get) if scores else "?"
 
        print(f"{row.dataset:<12} {row.horizon:<8} {row.naive_r2:>8.4f} "
              f"{ens_str:>10} {row.chronos_r2:>9.4f} {sn_str:>10} {winner:>8}")
 
    # ── Clean up checkpoint on success ─────────────────────────
    if len(results) == 12:
        if os.path.exists(CKPT_PATH):
            os.remove(CKPT_PATH)
        print("\nAll 12 jobs complete. Checkpoint removed.")
    else:
        print(f"\n{len(results)}/12 jobs complete. Re-run to finish remaining.")
 