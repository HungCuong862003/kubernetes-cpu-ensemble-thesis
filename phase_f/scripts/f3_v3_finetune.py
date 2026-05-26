"""
f3_v3_finetune.py — Phase F3.6 + DoRA + curriculum follow-ups
Unified Chronos-2 LoRA fine-tune with CLI flags.

Supports:
  --rank {4, 8, 16}        LoRA rank
  --dora                   Enable DoRA (weight-decomposed LoRA)
  --curriculum             Mix horizons within each batch (vs rotate per epoch)
  --out-dir PATH           Where to save best adapter
  --history-csv PATH       Per-epoch training metrics

Run examples:
    # F3.6 rank sweep: rank=4
    python phase_f/scripts/f3_v3_finetune.py --rank 4 \
        --out-dir phase_f/models/f3_lora_rank4 \
        --history-csv phase_f/data/f3_training_rank4.csv

    # F3.6 rank sweep: rank=16
    python phase_f/scripts/f3_v3_finetune.py --rank 16 \
        --out-dir phase_f/models/f3_lora_rank16 \
        --history-csv phase_f/data/f3_training_rank16.csv

    # DoRA at rank=8
    python phase_f/scripts/f3_v3_finetune.py --rank 8 --dora \
        --out-dir phase_f/models/f3_dora_rank8 \
        --history-csv phase_f/data/f3_training_dora.csv

    # Curriculum learning (mixed horizons per batch)
    python phase_f/scripts/f3_v3_finetune.py --rank 8 --curriculum \
        --out-dir phase_f/models/f3_curriculum_rank8 \
        --history-csv phase_f/data/f3_training_curriculum.csv

Note: training objective is native Chronos-2 ForCausalLMLoss (same API constraint
as v2). Asymmetric pinball is applied post-hoc in evaluation, not training.
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

# -------------------------------------------------------------------------
# Parse CLI
# -------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=8, choices=[4, 8, 16])
parser.add_argument("--dora", action="store_true", help="Enable DoRA")
parser.add_argument("--curriculum", action="store_true", help="Mix horizons per batch")
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--history-csv", type=str, required=True)
args = parser.parse_args()

# -------------------------------------------------------------------------
# Config (mostly same as v2)
# -------------------------------------------------------------------------

WORKSPACE  = Path("/workspace/kubernetes-cpu-ensemble-thesis")
DATA_DIR   = WORKSPACE / "data" / "processed"
PHASE_F    = WORKSPACE / "phase_f"
MODELS_DIR = Path(args.out_dir)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = Path(args.history_csv)

BASELINE_JSON = PHASE_F / "data" / "f3_zero_shot_baseline.json"

DATASETS     = ["alibaba", "bitbrains", "bytedance"]
INTERVAL_MIN = {"alibaba": 5, "bitbrains": 5, "bytedance": 10}
HORIZONS_MIN = [10, 30, 60, 120]
SKIP_CELLS   = {("bytedance", 10)}

N_CONTEXT     = 512
LORA_RANK     = args.rank
LORA_ALPHA    = 2 * LORA_RANK   # scale alpha with rank
LORA_DROPOUT  = 0.1
BATCH_SIZE    = 32
MAX_EPOCHS    = 20
PATIENCE      = 5
LR            = 1e-4
WEIGHT_DECAY  = 0.01
VAL_FRAC      = 0.15
PATCH_SIZE    = 16

PRIMARY_H_MIN = 60
PRIMARY_TAU   = 0.9
TAUS          = [0.5, 0.7, 0.8, 0.9, 0.95]

LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi", "wo", "output_layer", "residual_layer"]

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("=" * 70)
print(f"F3 v3 fine-tune — rank={LORA_RANK}, dora={args.dora}, curriculum={args.curriculum}")
print("=" * 70)
print(f"Out dir: {MODELS_DIR}")
print(f"History: {HISTORY_CSV}")
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
# Load data (same as v2)
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

def split_train_val(series_dict, val_frac=VAL_FRAC):
    train, val = {}, {}
    for sid, arr in series_dict.items():
        cut = int(len(arr) * (1 - val_frac))
        if cut >= N_CONTEXT + 2:
            train[sid] = arr[:cut]
            val[sid]   = arr[cut:]
    return train, val

train_series = {}
val_series   = {}
for ds in DATASETS:
    train_series[ds], val_series[ds] = split_train_val(all_series[ds])
    print(f"  {ds}: {len(train_series[ds])} train, {len(val_series[ds])} val series")
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
        if not ctxs:
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
        print(f"    val {ds}: {score:.6f}")
    return float(np.mean(scores)) if scores else float("inf")

# -------------------------------------------------------------------------
# Build training windows
# -------------------------------------------------------------------------

def build_epoch_windows(epoch, curriculum):
    """
    Return dict: {h_obs: [(ctx, tgt), ...]}
    
    If curriculum=False: epoch-rotated, one horizon per dataset per epoch.
    If curriculum=True:  sample ALL horizons in every epoch.
    """
    windows_by_h = defaultdict(list)
    
    if curriculum:
        # Sample windows from all horizons in this epoch
        for ds in DATASETS:
            available_h = [h for h in HORIZONS_MIN if (ds, h) not in SKIP_CELLS]
            for h_min in available_h:
                h_obs = h_min // INTERVAL_MIN[ds]
                # Take ~25% of train series per (ds, horizon) to keep epoch size reasonable
                sids = list(train_series[ds].keys())
                random.shuffle(sids)
                for sid in sids[:max(1, len(sids) // len(available_h))]:
                    arr = train_series[ds][sid]
                    r = sample_window(arr, h_obs)
                    if r is None: continue
                    windows_by_h[h_obs].append(r)
    else:
        # Original v2 behaviour: rotate horizons per epoch
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
# Pre-training val baseline
# -------------------------------------------------------------------------

print("Step 3: Pre-training validation")
baseline_val = compute_val_metric()
print(f"  Pre-training val: {baseline_val:.6f}")
print()

# -------------------------------------------------------------------------
# Training loop
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
    
    # Curriculum mode: interleave horizons within epoch
    if args.curriculum:
        # Flatten and shuffle all (h_obs, ctx, tgt) tuples
        all_windows = []
        for h_obs, wlist in windows_by_h.items():
            for ctx, tgt in wlist:
                all_windows.append((h_obs, ctx, tgt))
        random.shuffle(all_windows)
        
        # Group consecutive same-h_obs items into batches
        i = 0
        while i < len(all_windows):
            # Find run of same h_obs
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
        # v2-style batching by horizon
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
    
    val_score = compute_val_metric()
    
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
