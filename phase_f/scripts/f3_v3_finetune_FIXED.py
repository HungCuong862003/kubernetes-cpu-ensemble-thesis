"""
f3_v3_finetune.py — FIXED VERSION with adaptive val_frac

Key fix vs previous version:
  - split_train_val now uses ADAPTIVE val_frac per series
  - For short series (e.g. Alibaba median 2296 points), val_frac is increased
    so the val region is long enough to sample at least 1 window at h=60
  - Specifically: val region must be >= N_CONTEXT + max(HORIZONS_OBS) + buffer
  - This ensures Alibaba is INCLUDED in val computation (previously skipped silently)
  
Other behaviour identical to previous version.
"""

import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from peft import LoraConfig, get_peft_model
from chronos import BaseChronosPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=8, choices=[4, 8, 16])
parser.add_argument("--dora", action="store_true")
parser.add_argument("--curriculum", action="store_true")
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--history-csv", type=str, required=True)
args = parser.parse_args()

WORKSPACE  = Path("/workspace/kubernetes-cpu-ensemble-thesis")
DATA_DIR   = WORKSPACE / "data" / "processed"
PHASE_F    = WORKSPACE / "phase_f"
MODELS_DIR = Path(args.out_dir)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = Path(args.history_csv)

DATASETS     = ["alibaba", "bitbrains", "bytedance"]
INTERVAL_MIN = {"alibaba": 5, "bitbrains": 5, "bytedance": 10}
HORIZONS_MIN = [10, 30, 60, 120]
SKIP_CELLS   = {("bytedance", 10)}

N_CONTEXT     = 512
LORA_RANK     = args.rank
LORA_ALPHA    = 2 * LORA_RANK
LORA_DROPOUT  = 0.1
BATCH_SIZE    = 32
MAX_EPOCHS    = 20
PATIENCE      = 5
LR            = 1e-4
WEIGHT_DECAY  = 0.01
VAL_FRAC_FLOOR = 0.15        # minimum val_frac (was the fixed VAL_FRAC before)
PATCH_SIZE    = 16

PRIMARY_H_MIN = 60
PRIMARY_TAU   = 0.9
TAUS          = [0.5, 0.7, 0.8, 0.9, 0.95]

# Maximum h_obs we need to support during val sampling
MAX_H_OBS = max(h // INTERVAL_MIN[ds] for ds in DATASETS for h in HORIZONS_MIN if (ds, h) not in SKIP_CELLS)
# Minimum val region length to allow at least 1 sampling position with some headroom
VAL_MIN_LEN = N_CONTEXT + MAX_H_OBS
VAL_HEADROOM = 100  # extra points so val_start has range >0
VAL_REGION_MIN = VAL_MIN_LEN + VAL_HEADROOM   # 512 + 24 + 100 = 636

LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi", "wo", "output_layer", "residual_layer"]

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("=" * 70)
print(f"F3 v3 fine-tune — rank={LORA_RANK}, dora={args.dora}, curriculum={args.curriculum}")
print(f"FIX APPLIED: adaptive val_frac to include short-series datasets (e.g. Alibaba)")
print("=" * 70)
print(f"Out dir: {MODELS_DIR}")
print(f"History: {HISTORY_CSV}")
print(f"VAL_FRAC_FLOOR: {VAL_FRAC_FLOOR}, VAL_REGION_MIN: {VAL_REGION_MIN}")
print()

# -------------------------------------------------------------------------
# Load model + LoRA/DoRA
# -------------------------------------------------------------------------

print("Step 1: Loading Chronos-2 + adapter")
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2", device_map="cuda", dtype=torch.bfloat16
)
device = next(pipe.model.parameters()).device

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    use_dora=args.dora,
)
pipe.model = get_peft_model(pipe.model, lora_cfg)
trainable = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in pipe.model.parameters())
print(f"  Adapter: rank={LORA_RANK}, alpha={LORA_ALPHA}, dora={args.dora}")
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
print()

# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

def load_series(dataset):
    parts = []
    for split in ["train", "val", "test"]:
        p = DATA_DIR / dataset / f"{split}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    id_col  = next(c for c in ["container_id","vm_id","instance_id"] if c in df.columns)
    cpu_col = next(c for c in ["cpu_util_percent","cpu_target","cpu","cpu_percent"]
                   if c in df.columns)
    df = df.sort_values([id_col, "time_stamp"])
    out = {}
    for uid, grp in df.groupby(id_col):
        out[str(uid)] = grp[cpu_col].to_numpy(dtype=np.float32)
    return out

print("Step 2: Loading datasets")
all_series = {ds: load_series(ds) for ds in DATASETS}

def split_train_val_adaptive(series_dict):
    """
    FIXED: adaptive val_frac. Ensures val region has at least VAL_REGION_MIN points
    by increasing val_frac for short series. Series too short for both
    (train >= N_CONTEXT+2 AND val >= VAL_REGION_MIN) are dropped.
    """
    train, val = {}, {}
    n_skipped_short = 0
    n_adaptive = 0
    
    for sid, arr in series_dict.items():
        n = len(arr)
        # Required minimums
        train_min = N_CONTEXT + 2
        val_min = VAL_REGION_MIN
        
        # Compute val_frac that satisfies both constraints
        # train_len = n - val_len >= train_min  →  val_len <= n - train_min
        # val_len >= val_min  →  need n - train_min >= val_min  →  n >= train_min + val_min
        if n < train_min + val_min:
            n_skipped_short += 1
            continue
        
        # Default val_frac = floor; adapt if needed
        val_frac_default = VAL_FRAC_FLOOR
        val_len_default = int(n * val_frac_default)
        
        if val_len_default >= val_min:
            # Default works
            val_frac_use = val_frac_default
        else:
            # Need larger val_frac to meet val_min
            val_frac_use = val_min / n
            n_adaptive += 1
        
        cut = int(n * (1 - val_frac_use))
        train[sid] = arr[:cut]
        val[sid]   = arr[cut:]
    
    return train, val, n_skipped_short, n_adaptive

train_series = {}
val_series   = {}
for ds in DATASETS:
    t, v, n_skip, n_adapt = split_train_val_adaptive(all_series[ds])
    train_series[ds] = t
    val_series[ds]   = v
    avg_train_len = np.mean([len(arr) for arr in t.values()]) if t else 0
    avg_val_len = np.mean([len(arr) for arr in v.values()]) if v else 0
    print(f"  {ds}: {len(t)} train, {len(v)} val series "
          f"(adaptive={n_adapt}, skipped={n_skip}) | "
          f"avg train_len={avg_train_len:.0f}, val_len={avg_val_len:.0f}")
print()

# Sanity check: ALL datasets should produce val-eligible series at h=60
print("Step 2b: Val eligibility sanity check at h=60")
for ds in DATASETS:
    if (ds, PRIMARY_H_MIN) in SKIP_CELLS:
        print(f"  {ds}: SKIP (in SKIP_CELLS)")
        continue
    h_obs = PRIMARY_H_MIN // INTERVAL_MIN[ds]
    min_needed = N_CONTEXT + h_obs
    eligible = sum(1 for arr in val_series[ds].values() if len(arr) >= min_needed)
    total_v = len(val_series[ds])
    pct = 100 * eligible / total_v if total_v else 0
    print(f"  {ds}: {eligible}/{total_v} ({pct:.1f}%) val series eligible at h_obs={h_obs}")
    if eligible == 0:
        print(f"    *** ZERO eligible — CHECK FIX, this should not happen after the patch ***")
print()

# -------------------------------------------------------------------------
# Window sampling helpers
# -------------------------------------------------------------------------

def sample_window(arr, h_obs):
    min_len = N_CONTEXT + h_obs
    if len(arr) < min_len:
        return None
    start = random.randint(0, len(arr) - min_len)
    return arr[start:start+N_CONTEXT], arr[start+N_CONTEXT:start+N_CONTEXT+h_obs]

def pinball_np(y_true, y_pred, tau):
    diff = y_true - y_pred
    return float(np.where(diff >= 0, tau*diff, (tau-1)*diff).mean())

# -------------------------------------------------------------------------
# Validation metric
# -------------------------------------------------------------------------

def compute_val_metric(max_windows=500):
    pipe.model.eval()
    scores = []
    n_windows_per_ds = {}
    for ds in DATASETS:
        if (ds, PRIMARY_H_MIN) in SKIP_CELLS:
            continue
        h_obs = PRIMARY_H_MIN // INTERVAL_MIN[ds]
        ctxs, tgts = [], []
        for arr in list(val_series[ds].values())[:max_windows]:
            r = sample_window(arr, h_obs)
            if r is None:
                continue
            ctxs.append(r[0]); tgts.append(r[1])
        n_windows_per_ds[ds] = len(ctxs)
        if not ctxs:
            print(f"    *** WARNING: val skipped {ds} (0 windows produced)")
            continue
        ctxs = np.stack(ctxs); tgts = np.stack(tgts)
        ctx_t = torch.tensor(ctxs, dtype=torch.float32).unsqueeze(1)
        all_q90 = []
        with torch.no_grad():
            for i in range(0, len(ctx_t), 64):
                batch = ctx_t[i:i+64]
                q_list, _ = pipe.predict_quantiles(
                    batch, prediction_length=h_obs, quantile_levels=TAUS
                )
                q_stack = torch.stack(q_list, dim=0).squeeze(1)
                tau_idx = TAUS.index(PRIMARY_TAU)
                all_q90.append(q_stack[:, :, tau_idx].numpy())
        all_q90 = np.concatenate(all_q90, axis=0)
        score = pinball_np(tgts, all_q90, PRIMARY_TAU)
        scores.append(score)
        print(f"    val {ds} (n={len(ctxs)}): {score:.6f}")
    return float(np.mean(scores)) if scores else float("inf"), n_windows_per_ds

# -------------------------------------------------------------------------
# Build training windows (unchanged from previous version)
# -------------------------------------------------------------------------

def build_epoch_windows(epoch, curriculum):
    windows_by_h = defaultdict(list)
    
    if curriculum:
        for ds in DATASETS:
            available_h = [h for h in HORIZONS_MIN if (ds, h) not in SKIP_CELLS]
            for h_min in available_h:
                h_obs = h_min // INTERVAL_MIN[ds]
                sids = list(train_series[ds].keys())
                random.shuffle(sids)
                for sid in sids[:max(1, len(sids) // len(available_h))]:
                    arr = train_series[ds][sid]
                    r = sample_window(arr, h_obs)
                    if r is None: continue
                    windows_by_h[h_obs].append(r)
    else:
        for ds in DATASETS:
            available_h = [h for h in HORIZONS_MIN if (ds, h) not in SKIP_CELLS]
            h_min = available_h[epoch % len(available_h)]
            h_obs = h_min // INTERVAL_MIN[ds]
            for arr in train_series[ds].values():
                r = sample_window(arr, h_obs)
                if r is None: continue
                windows_by_h[h_obs].append(r)
    return windows_by_h

# -------------------------------------------------------------------------
# Optimizer
# -------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    [p for p in pipe.model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY
)

# -------------------------------------------------------------------------
# Pre-training val
# -------------------------------------------------------------------------

print("Step 3: Pre-training validation")
baseline_val, n_per_ds = compute_val_metric()
print(f"  Pre-training val: {baseline_val:.6f}")
print(f"  Windows per dataset: {n_per_ds}")
print()

# Assert that Alibaba is now in val (the whole point of the fix)
if "alibaba" not in n_per_ds or n_per_ds.get("alibaba", 0) == 0:
    print("ERROR: Alibaba still excluded from val after fix. Investigate.")
    raise SystemExit(1)

# -------------------------------------------------------------------------
# Training loop (unchanged from previous version)
# -------------------------------------------------------------------------

print(f"Step 4: Training (curriculum={args.curriculum})")
history = []
best_val   = float("inf")
best_epoch = -1
patience_count = 0

for epoch in range(1, MAX_EPOCHS + 1):
    pipe.model.train()
    epoch_losses = []
    
    windows_by_h = build_epoch_windows(epoch, args.curriculum)
    
    if args.curriculum:
        all_windows = []
        for h_obs, wlist in windows_by_h.items():
            for ctx, tgt in wlist:
                all_windows.append((h_obs, ctx, tgt))
        random.shuffle(all_windows)
        
        i = 0
        while i < len(all_windows):
            cur_h = all_windows[i][0]
            batch_items = []
            while i < len(all_windows) and len(batch_items) < BATCH_SIZE and all_windows[i][0] == cur_h:
                batch_items.append(all_windows[i])
                i += 1
            if not batch_items: continue
            
            h_obs = batch_items[0][0]
            ctxs = np.stack([b[1] for b in batch_items])
            tgts = np.stack([b[2] for b in batch_items])
            ctx_t = torch.tensor(ctxs, dtype=torch.float32).to(device)
            tgt_t = torch.tensor(tgts, dtype=torch.float32).to(device)
            n_out_patches = max(1, int(np.ceil(h_obs / PATCH_SIZE)))
            optimizer.zero_grad()
            output = pipe.model(context=ctx_t, future_target=tgt_t, num_output_patches=n_out_patches)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipe.model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            epoch_losses.append(float(loss.item()))
    else:
        for h_obs, windows in windows_by_h.items():
            random.shuffle(windows)
            for i in range(0, len(windows), BATCH_SIZE):
                batch = windows[i:i+BATCH_SIZE]
                if not batch: continue
                ctxs = np.stack([w[0] for w in batch])
                tgts = np.stack([w[1] for w in batch])
                ctx_t = torch.tensor(ctxs, dtype=torch.float32).to(device)
                tgt_t = torch.tensor(tgts, dtype=torch.float32).to(device)
                n_out_patches = max(1, int(np.ceil(h_obs / PATCH_SIZE)))
                optimizer.zero_grad()
                output = pipe.model(context=ctx_t, future_target=tgt_t, num_output_patches=n_out_patches)
                loss = output.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in pipe.model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                epoch_losses.append(float(loss.item()))
    
    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    print(f"Epoch {epoch}/{MAX_EPOCHS} | train_loss={mean_loss:.6f}")
    
    val_score, _ = compute_val_metric()
    
    improved = val_score < best_val
    if improved:
        best_val = val_score
        best_epoch = epoch
        patience_count = 0
        pipe.model.save_pretrained(str(MODELS_DIR))
        print(f"  NEW BEST val={val_score:.6f} → saved")
    else:
        patience_count += 1
        print(f"  val={val_score:.6f} | best={best_val:.6f} @ ep{best_epoch} | patience {patience_count}/{PATIENCE}")
    
    history.append({
        "epoch": epoch, "train_loss": mean_loss, "val_pinball": val_score,
        "is_best": improved, "patience": patience_count,
        "rank": LORA_RANK, "dora": args.dora, "curriculum": args.curriculum,
        "fix_version": "adaptive_val_frac_v1",
    })
    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
    
    if patience_count >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

print()
print("=" * 70)
print(f"Done. Best val: {best_val:.6f} at epoch {best_epoch}")
print(f"Adapter saved: {MODELS_DIR}")
print(f"History: {HISTORY_CSV}")
print("=" * 70)
