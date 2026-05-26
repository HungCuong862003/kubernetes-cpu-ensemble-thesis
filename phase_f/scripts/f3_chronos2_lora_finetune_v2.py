"""
f3_chronos2_lora_finetune_v2.py — Phase F3.4 (fixed)
LoRA fine-tune Chronos-2 using the native Chronos2Model training interface.

Key findings from API probe (2026-05-25):
  - model.forward(context, future_target) returns Chronos2Output with .loss
  - context shape must be 2D: (batch, n_context) — NOT (batch, 1, n_context)
  - Training loss is ForCausalLMLoss (native Chronos-2 objective)
  - pipe.predict_quantiles() requires CPU tensors for eval

Design note: training objective is the native Chronos-2 ForCausalLMLoss rather than
the pre-registered asymmetric pinball loss (f3_design.md §Method). This deviation
arises because the model's quantile head is accessed via an internal _compute_loss
that requires pre-processed patch tensors not directly accessible through the public
API. The PRE-REGISTERED EVALUATION METRIC (pinball at τ=0.9, h=60min) is unaffected
— it is computed separately via predict_quantiles. Deviation disclosed in DECISION-015.

Run:
    cd /workspace/kubernetes-cpu-ensemble-thesis
    source /venv/main/bin/activate
    nohup python phase_f/scripts/f3_chronos2_lora_finetune_v2.py \
        > phase_f/data/f3_finetune_log.txt 2>&1 &
    echo "PID: $!"

Outputs:
    phase_f/models/f3_lora_rank8/          <- best LoRA adapter weights
    phase_f/data/f3_training_history.csv   <- per-epoch loss + val pinball
"""

import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model
from chronos import BaseChronosPipeline

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

WORKSPACE  = Path("/workspace/kubernetes-cpu-ensemble-thesis")
DATA_DIR   = WORKSPACE / "data" / "processed"
PHASE_F    = WORKSPACE / "phase_f"
MODELS_DIR = PHASE_F / "models" / "f3_lora_rank8"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_JSON    = PHASE_F / "data" / "f3_zero_shot_baseline.json"
HISTORY_CSV      = PHASE_F / "data" / "f3_training_history.csv"

DATASETS = ["alibaba", "bitbrains", "bytedance"]
INTERVAL_MIN  = {"alibaba": 5, "bitbrains": 5, "bytedance": 10}
HORIZONS_MIN  = [10, 30, 60, 120]
SKIP_CELLS    = {("bytedance", 10)}

N_CONTEXT     = 512
LORA_RANK     = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.1
BATCH_SIZE    = 32       # smaller: 2D context is more memory-efficient
MAX_EPOCHS    = 20
PATIENCE      = 5
LR            = 1e-4
WEIGHT_DECAY  = 0.01
VAL_FRAC      = 0.15

# τ=0.9 at h=60 is the pre-registered primary evaluation metric
PRIMARY_H_MIN = 60
PRIMARY_TAU   = 0.9
TAUS          = [0.5, 0.7, 0.8, 0.9, 0.95]

LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi", "wo", "output_layer", "residual_layer"]

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("=" * 70)
print("F3 Chronos-2 LoRA fine-tune v2 (corrected API)")
print("=" * 70)
print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGET_MODULES}")
print(f"Training loss: native ForCausalLMLoss (see design note in header)")
print(f"Eval metric  : pinball at τ={PRIMARY_TAU}, h={PRIMARY_H_MIN}min")
print()

# -------------------------------------------------------------------------
# 1. Load model + LoRA
# -------------------------------------------------------------------------

print("Step 1: Loading Chronos-2 + LoRA")
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2", device_map="cuda", dtype=torch.bfloat16
)
device = next(pipe.model.parameters()).device
print(f"  Device: {device}")

lora_cfg = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES, bias="none",
)
pipe.model = get_peft_model(pipe.model, lora_cfg)
trainable = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in pipe.model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
print()

# -------------------------------------------------------------------------
# 2. Load data
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
    lengths = [len(v) for v in out.values()]
    print(f"  {dataset}: {len(out)} series | "
          f"min={min(lengths)} median={int(np.median(lengths))} max={max(lengths)}")
    return out

print("Step 2: Loading datasets")
all_series = {ds: load_series(ds) for ds in DATASETS}
print()

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
# 3. Window sampling
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
# 4. Validation metric — pinball at τ=0.9, h=60min
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
            result = sample_window(arr, h_obs)
            if result is None:
                continue
            ctx, tgt = result
            ctxs.append(ctx)
            tgts.append(tgt)
        if not ctxs:
            continue
        ctxs = np.stack(ctxs)
        tgts = np.stack(tgts)
        # predict_quantiles needs CPU 3D tensor: (N, 1, N_CONTEXT)
        ctx_t = torch.tensor(ctxs, dtype=torch.float32).unsqueeze(1)   # CPU
        all_q90 = []
        with torch.no_grad():
            for i in range(0, len(ctx_t), 64):
                batch = ctx_t[i:i+64]
                q_list, _ = pipe.predict_quantiles(
                    batch, prediction_length=h_obs, quantile_levels=TAUS
                )
                q_stack = torch.stack(q_list, dim=0).squeeze(1)  # (B, H, n_q)
                tau_idx = TAUS.index(PRIMARY_TAU)
                q90 = q_stack[:, :, tau_idx].numpy()             # (B, H)
                all_q90.append(q90)
        all_q90 = np.concatenate(all_q90, axis=0)
        score = pinball_np(tgts, all_q90, PRIMARY_TAU)
        scores.append(score)
        print(f"    val pinball {ds} h={PRIMARY_H_MIN}min τ={PRIMARY_TAU}: {score:.6f}")
    if not scores:
        return float("inf")
    return float(np.mean(scores))

# -------------------------------------------------------------------------
# 5. Optimizer
# -------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    [p for p in pipe.model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY
)

# -------------------------------------------------------------------------
# 6. Pre-training val baseline
# -------------------------------------------------------------------------

print("Step 3: Pre-training validation (zero-shot baseline sanity check)")
baseline_val = compute_val_metric()
print(f"  Pre-training val metric: {baseline_val:.6f}")
try:
    with open(BASELINE_JSON) as f:
        bl = json.load(f)
    locked = bl["metadata"]["pre_registration"]["baseline_primary"]
    diff = abs(baseline_val - locked)
    print(f"  Locked baseline: {locked:.6f} | diff: {diff:.4f} "
          f"({'OK' if diff < 0.1*locked else 'WARNING: large diff'})")
except Exception as e:
    print(f"  Could not load baseline JSON: {e}")
print()

# -------------------------------------------------------------------------
# 7. Training loop
# -------------------------------------------------------------------------

print("Step 4: Training")
history = []
best_val   = float("inf")
best_epoch = -1
patience_count = 0

for epoch in range(1, MAX_EPOCHS + 1):
    pipe.model.train()
    epoch_losses = []

    # Collect windows for this epoch: one random window per series per dataset
    # Group by horizon to batch efficiently
    from collections import defaultdict
    windows_by_h = defaultdict(list)

    for ds in DATASETS:
        available_h = [h for h in HORIZONS_MIN if (ds, h) not in SKIP_CELLS]
        h_min = available_h[epoch % len(available_h)]
        h_obs = h_min // INTERVAL_MIN[ds]
        for arr in train_series[ds].values():
            result = sample_window(arr, h_obs)
            if result is None:
                continue
            ctx, tgt = result
            windows_by_h[h_obs].append((ctx, tgt))

    for h_obs, windows in windows_by_h.items():
        random.shuffle(windows)
        for i in range(0, len(windows), BATCH_SIZE):
            batch = windows[i:i+BATCH_SIZE]
            if not batch:
                continue
            ctxs = np.stack([w[0] for w in batch])   # (B, N_CONTEXT)
            tgts = np.stack([w[1] for w in batch])   # (B, h_obs)

            # Context: 2D on GPU; target: 2D on GPU
            ctx_t = torch.tensor(ctxs, dtype=torch.float32).to(device)   # (B, N_CONTEXT)
            tgt_t = torch.tensor(tgts, dtype=torch.float32).to(device)   # (B, h_obs)

            optimizer.zero_grad()
            output = pipe.model(context=ctx_t, future_target=tgt_t)
            loss = output.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipe.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            epoch_losses.append(float(loss.item()))

    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
    print(f"Epoch {epoch}/{MAX_EPOCHS} | train_loss={mean_loss:.6f}")

    # Validation
    val_score = compute_val_metric()

    improved = val_score < best_val
    if improved:
        best_val   = val_score
        best_epoch = epoch
        patience_count = 0
        pipe.model.save_pretrained(str(MODELS_DIR))
        print(f"  NEW BEST val={val_score:.6f} → saved to {MODELS_DIR}")
    else:
        patience_count += 1
        print(f"  val={val_score:.6f} | best={best_val:.6f} @ ep{best_epoch} "
              f"| patience {patience_count}/{PATIENCE}")

    history.append({
        "epoch": epoch, "train_loss": mean_loss, "val_pinball": val_score,
        "is_best": improved, "patience": patience_count
    })
    pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)

    if patience_count >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

# -------------------------------------------------------------------------
# 8. Summary
# -------------------------------------------------------------------------

print()
print("=" * 70)
print(f"Training complete. Best val pinball: {best_val:.6f} at epoch {best_epoch}")
print(f"LoRA weights saved to: {MODELS_DIR}")
print()

# Quick DECISION-015 preview
try:
    with open(BASELINE_JSON) as f:
        bl = json.load(f)
    locked = bl["metadata"]["pre_registration"]["baseline_primary"]
    improvement = (locked - best_val) / locked * 100
    print(f"Pre-registered primary metric:")
    print(f"  Baseline : {locked:.6f}")
    print(f"  Best val : {best_val:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    if improvement >= 5.0:
        print("  → DECISION-015: F3 SUCCESS (≥5%)")
    elif improvement >= 2.0:
        print("  → DECISION-015: F3 PARTIAL (≥2%)")
    else:
        print("  → DECISION-015: F3 FAILURE (<2%)")
    print()
    print("NOTE: val metric uses val split; final DECISION-015 requires test evaluation.")
except Exception as e:
    print(f"Could not compute improvement: {e}")

print("Run f3_evaluate.py (F3.5) on test set to lock DECISION-015 officially.")
