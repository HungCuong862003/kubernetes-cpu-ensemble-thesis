"""
f3_chronos2_lora_finetune.py
Phase F3.3 — LoRA fine-tune Chronos-2 with asymmetric pinball loss.

Design decisions locked in f3_design.md + DECISION-016:
  - LoRA rank=8, alpha=16, dropout=0.1
  - target_modules: q/k/v/o (attention) + wi/wo (MLP/FFN) across all 12 encoder blocks
    + output_layer/residual_layer (output_patch_embedding = the quantile head)
  - Weighted pinball across tau in [0.5, 0.7, 0.8, 0.9, 0.95]; tau=0.9 gets 3x weight
  - AdamW lr=1e-4, weight_decay=0.01
  - 20 epochs max, early stopping patience=5 on val pinball at tau=0.9
  - 1 random window per container per epoch (efficient for large datasets)
  - Batch size 64

Run on Vast.ai:
    cd /workspace/kubernetes-cpu-ensemble-thesis
    source /venv/main/bin/activate
    python phase_f/scripts/f3_chronos2_lora_finetune.py 2>&1 | tee phase_f/data/f3_finetune_log.txt

Outputs:
    phase_f/models/f3_lora_rank8/          <- LoRA adapter weights (best val epoch)
    phase_f/data/f3_training_history.csv   <- per-epoch train loss + val pinball
    phase_f/data/f3_finetune_log.txt       <- full stdout via tee
"""

import os
import sys
import json
import math
import random
import inspect
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# -------------------------------------------------------------------------
# 0. Paths and configuration
# -------------------------------------------------------------------------

WORKSPACE = Path("/workspace/kubernetes-cpu-ensemble-thesis")
PHASE_F   = WORKSPACE / "phase_f"
DATA_DIR  = WORKSPACE / "data" / "processed"
MODELS_DIR = PHASE_F / "models" / "f3_lora_rank8"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_HISTORY_CSV = PHASE_F / "data" / "f3_training_history.csv"
BASELINE_JSON        = PHASE_F / "data" / "f3_zero_shot_baseline.json"

DATASETS   = ["alibaba", "bitbrains", "bytedance"]
HORIZONS   = [10, 30, 60, 120]        # minutes
INTERVALS  = {"alibaba": 5, "bitbrains": 5, "bytedance": 10}   # sampling interval in minutes
HORIZONS_OBS = {}                      # horizon in number of observations, filled below
for h in HORIZONS:
    for ds, interval in INTERVALS.items():
        HORIZONS_OBS[(ds, h)] = h // interval

# ByteDance has no h=10 (10-min sampling cannot predict 10-min ahead with this setup)
SKIP_CELLS = {("bytedance", 10)}

N_CONTEXT     = 512       # context window in observations
LORA_RANK     = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.1
BATCH_SIZE    = 64
MAX_EPOCHS    = 20
PATIENCE      = 5         # early stopping patience on val pinball at tau=0.9
LR            = 1e-4
WEIGHT_DECAY  = 0.01

TAUS          = [0.5, 0.7, 0.8, 0.9, 0.95]
TAU_WEIGHTS   = [1.0, 1.0, 1.5, 3.0, 1.5]   # tau=0.9 gets 3x weight (pre-reg primary tau)
assert len(TAUS) == len(TAU_WEIGHTS)

SEED          = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# target_modules: probe v5 recommendation + output head
# 'q','k','v','o','wi','wo' -> all 12 encoder blocks (suffix-matched by peft)
# 'output_layer','residual_layer' -> output_patch_embedding (quantile head)
#                                  + input_patch_embedding output/residual (acceptable)
LORA_TARGET_MODULES = ["q", "k", "v", "o", "wi", "wo", "output_layer", "residual_layer"]

print("=" * 70)
print("F3.3 Chronos-2 LoRA fine-tune — asymmetric pinball loss")
print("=" * 70)
print(f"WORKSPACE     : {WORKSPACE}")
print(f"Models output : {MODELS_DIR}")
print(f"LoRA rank     : {LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"Target modules: {LORA_TARGET_MODULES}")
print(f"Taus + weights: {list(zip(TAUS, TAU_WEIGHTS))}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Max epochs    : {MAX_EPOCHS}, patience={PATIENCE}")
print(f"LR            : {LR}, weight_decay={WEIGHT_DECAY}")
print()

# -------------------------------------------------------------------------
# 1. Load Chronos-2 and apply LoRA
# -------------------------------------------------------------------------

print("Step 1: Loading Chronos-2 + applying LoRA")
print("-" * 50)

from chronos import BaseChronosPipeline
from peft import LoraConfig, get_peft_model

pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    dtype=torch.bfloat16,
)
device = next(pipe.model.parameters()).device
print(f"Model loaded on {device}, dtype={next(pipe.model.parameters()).dtype}")

# Count params before LoRA
total_before = sum(p.numel() for p in pipe.model.parameters())
print(f"Params before LoRA: {total_before:,}")

# Apply LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
)
pipe.model = get_peft_model(pipe.model, lora_config)

total_after     = sum(p.numel() for p in pipe.model.parameters())
trainable_after = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
print(f"Params after LoRA  : {total_after:,}")
print(f"Trainable (LoRA)   : {trainable_after:,} ({100*trainable_after/total_after:.3f}%)")
print()

# -------------------------------------------------------------------------
# 2. Probe forward signature — determine training calling convention
# -------------------------------------------------------------------------

print("Step 2: Probing forward signature for gradient-capable training pass")
print("-" * 50)

fwd_sig = inspect.signature(pipe.model.forward)
print(f"pipe.model.forward signature: {fwd_sig}")

# Build a tiny test batch: 3 series, N_CONTEXT=64 (reduced for probe speed), H=12
PROBE_CONTEXT = 64
PROBE_H = 12
probe_ctx = torch.randn(3, 1, PROBE_CONTEXT, dtype=torch.float32, device=device)
probe_target = torch.randn(3, PROBE_H, dtype=torch.float32, device=device)

# Test 1: call pipe.predict_quantiles with gradients enabled
# This is the cleanest path if predict_quantiles is differentiable end-to-end.
pipe.model.train()
FORWARD_MODE = None

print("Testing predict_quantiles in train mode with enable_grad ...")
try:
    with torch.enable_grad():
        q_list_probe, _ = pipe.predict_quantiles(
            probe_ctx,
            prediction_length=PROBE_H,
            quantile_levels=TAUS,
        )
        # q_list_probe: list of tensors, each shape (1, PROBE_H, n_q)
        # stack -> (3, 1, PROBE_H, n_q), squeeze -> (3, PROBE_H, n_q)
        q_stack = torch.stack(q_list_probe, dim=0).squeeze(1)  # (3, PROBE_H, n_q)
        # test: does q_stack have a grad_fn?
        if q_stack.requires_grad or (q_stack.grad_fn is not None):
            print(f"  predict_quantiles IS differentiable. q_stack shape={q_stack.shape}")
            FORWARD_MODE = "predict_quantiles"
        else:
            print("  predict_quantiles output has NO grad_fn. Will try direct model forward.")
except Exception as e:
    print(f"  predict_quantiles in train mode raised: {e}")

# Test 2: direct model forward (try common signatures)
if FORWARD_MODE is None:
    print("Testing direct pipe.model() forward ...")
    for sig_attempt in [
        dict(context=probe_ctx, prediction_length=PROBE_H),
        dict(inputs=probe_ctx, prediction_length=PROBE_H),
        dict(context=probe_ctx.squeeze(1), prediction_length=PROBE_H),
    ]:
        try:
            with torch.enable_grad():
                out = pipe.model(**sig_attempt)
                # out may be a tensor or a dataclass
                if isinstance(out, torch.Tensor):
                    test_tensor = out
                elif hasattr(out, "quantile_preds"):
                    test_tensor = out.quantile_preds
                elif hasattr(out, "logits"):
                    test_tensor = out.logits
                else:
                    test_tensor = out[0] if isinstance(out, (tuple, list)) else None

                if test_tensor is not None and test_tensor.grad_fn is not None:
                    print(f"  Direct forward works with kwargs={list(sig_attempt.keys())}")
                    print(f"  Output shape: {test_tensor.shape}")
                    FORWARD_MODE = "direct"
                    FORWARD_KWARGS_KEYS = list(sig_attempt.keys())
                    break
                else:
                    print(f"  Tried {list(sig_attempt.keys())} -> output not differentiable or shape unexpected")
        except Exception as e:
            print(f"  Tried {list(sig_attempt.keys())} -> {e}")

if FORWARD_MODE is None:
    print()
    print("ERROR: Neither predict_quantiles nor direct model.forward produced a differentiable output.")
    print("Possible causes:")
    print("  1. The model internally uses operations that detach gradients (e.g. numpy, in-place ops).")
    print("  2. The normalisation step inside predict_quantiles uses stop-gradient.")
    print("  3. A different calling convention is needed.")
    print()
    print("Inspect the Chronos2Pipeline source:")
    print("  python -c \"import chronos; import inspect; print(inspect.getfile(chronos.BaseChronosPipeline))\"")
    print("Then grep for 'def predict_quantiles' and 'def forward' in that file.")
    print("Exiting — fix forward mode before re-running training.")
    sys.exit(1)

print(f"Forward mode locked: {FORWARD_MODE}")
print()

# -------------------------------------------------------------------------
# 3. Helper: pinball loss
# -------------------------------------------------------------------------

def pinball_loss_weighted(
    y_true: torch.Tensor,   # (batch, H)
    q_preds: torch.Tensor,  # (batch, H, n_q)  — quantiles along last axis
    taus: list,
    weights: list,
) -> torch.Tensor:
    """Weighted sum of pinball losses across tau values."""
    total_weight = sum(weights)
    loss = torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)
    for i, (tau, w) in enumerate(zip(taus, weights)):
        q_pred_i = q_preds[:, :, i]             # (batch, H)
        diff = y_true - q_pred_i
        pb = torch.where(diff >= 0, tau * diff, (tau - 1) * diff)
        loss = loss + (w / total_weight) * pb.mean()
    return loss


def pinball_at_tau(
    y_true: torch.Tensor,   # (batch, H) or (N,)
    q_pred: torch.Tensor,   # same shape as y_true
    tau: float,
) -> float:
    """Scalar pinball loss at a single tau. Used for val metric computation."""
    diff = y_true - q_pred
    pb = torch.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return float(pb.mean().item())

# -------------------------------------------------------------------------
# 4. Helper: run forward pass and extract q_preds (batch, H, n_q)
# -------------------------------------------------------------------------

def forward_get_qpreds(
    context_3d: torch.Tensor,      # (batch, 1, n_context)
    prediction_length: int,
    taus: list,
    mode: str,
) -> torch.Tensor:
    """
    Run one forward pass and return quantile predictions shaped (batch, H, n_q).
    Works in both train (gradient) and eval (no gradient) mode.
    """
    if mode == "predict_quantiles":
        q_list, _ = pipe.predict_quantiles(
            context_3d,
            prediction_length=prediction_length,
            quantile_levels=taus,
        )
        # q_list: list of length batch, each (1, H, n_q)
        q_stack = torch.stack(q_list, dim=0).squeeze(1)   # (batch, H, n_q)
        return q_stack

    elif mode == "direct":
        sig_kwargs = {}
        for key in FORWARD_KWARGS_KEYS:
            if key in ("context", "inputs"):
                sig_kwargs[key] = context_3d
            elif key == "prediction_length":
                sig_kwargs[key] = prediction_length
        out = pipe.model(**sig_kwargs)
        if isinstance(out, torch.Tensor):
            raw = out
        elif hasattr(out, "quantile_preds"):
            raw = out.quantile_preds
        elif hasattr(out, "logits"):
            raw = out.logits
        else:
            raw = out[0]
        # Expect raw: (batch, H, n_q) or (batch, n_q, H) — handle both
        if raw.shape[1] == len(taus):
            raw = raw.permute(0, 2, 1)     # (batch, H, n_q)
        return raw

    else:
        raise ValueError(f"Unknown FORWARD_MODE: {mode}")

# -------------------------------------------------------------------------
# 5. Load dataset — concat train+val+test per container, return list of arrays
# -------------------------------------------------------------------------

def load_series_list(dataset: str) -> list:
    """
    Load all train+val+test parquets for a dataset.
    Returns list of 1D numpy arrays (CPU float32), one per container/vm.
    Each array is the full concatenated CPU time series, sorted by timestamp.
    """
    print(f"  Loading {dataset} ...")
    ds_dir = DATA_DIR / dataset

    dfs = []
    for split in ["train", "val", "test"]:
        p = ds_dir / f"{split}.parquet"
        if not p.exists():
            print(f"    WARNING: {p} not found, skipping.")
            continue
        df = pd.read_parquet(p)
        dfs.append(df)
        print(f"    {split}: {len(df):,} rows")

    if not dfs:
        raise FileNotFoundError(f"No parquet files found in {ds_dir}")

    df_all = pd.concat(dfs, ignore_index=True)

    # detect id and cpu column names
    id_col = None
    for c in ["container_id", "vm_id", "instance_id"]:
        if c in df_all.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"No id column found in {dataset}. Columns: {df_all.columns.tolist()}")

    cpu_col = None
    for c in ["cpu_util_percent", "cpu_target", "cpu", "cpu_percent"]:
        if c in df_all.columns:
            cpu_col = c
            break
    if cpu_col is None:
        raise ValueError(f"No cpu column found in {dataset}. Columns: {df_all.columns.tolist()}")

    ts_col = "time_stamp"

    df_all = df_all.sort_values([id_col, ts_col])

    series_list = []
    ids = df_all[id_col].unique()
    for uid in ids:
        arr = df_all.loc[df_all[id_col] == uid, cpu_col].to_numpy(dtype=np.float32)
        series_list.append(arr)

    print(f"  {dataset}: {len(series_list)} series, "
          f"lengths min={min(len(s) for s in series_list)} "
          f"median={int(np.median([len(s) for s in series_list]))} "
          f"max={max(len(s) for s in series_list)}")
    return series_list


def split_series_train_val(series_list: list, val_frac: float = 0.15) -> tuple:
    """
    Split each series chronologically: first (1-val_frac) -> train windows,
    last val_frac -> val windows.
    Returns (train_series_list, val_series_list).
    """
    train_list, val_list = [], []
    for arr in series_list:
        cut = int(len(arr) * (1 - val_frac))
        if cut < N_CONTEXT + 2:
            # series too short to split — skip entirely
            continue
        train_list.append(arr[:cut])
        val_list.append(arr[cut:])
    return train_list, val_list


# -------------------------------------------------------------------------
# 6. Window sampling helpers
# -------------------------------------------------------------------------

def sample_window(arr: np.ndarray, horizon_obs: int) -> tuple:
    """
    Sample one random (context, target) window from a series.
    Returns (context_1d, target_1d) as float32 numpy arrays.
    Returns None if series is too short.
    """
    min_len = N_CONTEXT + horizon_obs
    if len(arr) < min_len:
        return None
    max_start = len(arr) - min_len
    start = random.randint(0, max_start)
    context = arr[start : start + N_CONTEXT]
    target  = arr[start + N_CONTEXT : start + N_CONTEXT + horizon_obs]
    return context, target


def collect_val_windows(val_series_list: list, horizon_obs: int, max_windows: int = 2000) -> tuple:
    """
    Collect validation windows: up to max_windows random windows from val series.
    Returns (context_arr, target_arr) as numpy arrays shaped (N, N_CONTEXT) and (N, H).
    """
    contexts, targets = [], []
    for arr in val_series_list:
        result = sample_window(arr, horizon_obs)
        if result is None:
            continue
        ctx, tgt = result
        contexts.append(ctx)
        targets.append(tgt)
        if len(contexts) >= max_windows:
            break
    if not contexts:
        return None, None
    return np.stack(contexts), np.stack(targets)


# -------------------------------------------------------------------------
# 7. Load all data
# -------------------------------------------------------------------------

print("Step 3: Loading all dataset series")
print("-" * 50)

all_train_series = {}   # dataset -> list of (partial) series for training
all_val_series   = {}   # dataset -> list of (partial) series for validation

for ds in DATASETS:
    raw = load_series_list(ds)
    train_s, val_s = split_series_train_val(raw, val_frac=0.15)
    all_train_series[ds] = train_s
    all_val_series[ds]   = val_s
    print(f"  {ds}: {len(train_s)} train series, {len(val_s)} val series after split")

print()

# -------------------------------------------------------------------------
# 8. Primary eval horizon and tau — for early stopping
#    Pre-reg primary: h=60min, tau=0.9
# -------------------------------------------------------------------------

PRIMARY_H_MIN    = 60
PRIMARY_TAU      = 0.9
PRIMARY_TAU_IDX  = TAUS.index(PRIMARY_TAU)

# Pre-reg primary metric = mean over datasets of pinball at (h=60, tau=0.9)
# Only datasets that have h=60 contribute.

def compute_primary_val_metric(lora_model_pipe) -> float:
    """
    Compute pre-reg primary metric: mean pinball at h=60, tau=0.9 across valid datasets.
    Uses at most 1000 val windows per dataset for speed.
    Runs in eval mode (no gradients).
    """
    lora_model_pipe.model.eval()
    dataset_scores = []

    for ds in DATASETS:
        if (ds, PRIMARY_H_MIN) in SKIP_CELLS:
            continue

        h_obs = HORIZONS_OBS[(ds, PRIMARY_H_MIN)]
        val_ctxs, val_tgts = collect_val_windows(
            all_val_series[ds], h_obs, max_windows=1000
        )
        if val_ctxs is None:
            print(f"    WARNING: no val windows for {ds} at h={PRIMARY_H_MIN}min — skipping.")
            continue

        all_preds = []
        with torch.no_grad():
            for start in range(0, len(val_ctxs), BATCH_SIZE):
                batch_ctx = torch.tensor(
                    val_ctxs[start:start+BATCH_SIZE], dtype=torch.float32, device=device
                ).unsqueeze(1)    # (B, 1, N_CONTEXT)

                q_preds = forward_get_qpreds(batch_ctx, h_obs, TAUS, FORWARD_MODE)
                # q_preds: (B, H, n_q)
                q90 = q_preds[:, :, PRIMARY_TAU_IDX]   # (B, H)
                all_preds.append(q90.cpu())

        all_preds_t  = torch.cat(all_preds, dim=0)      # (N_val, H)
        all_tgts_t   = torch.tensor(val_tgts, dtype=torch.float32)

        score = pinball_at_tau(all_tgts_t, all_preds_t, PRIMARY_TAU)
        dataset_scores.append(score)
        print(f"    val pinball {ds} h={PRIMARY_H_MIN}min tau={PRIMARY_TAU}: {score:.6f}")

    if not dataset_scores:
        print("    WARNING: no val scores computed. Returning inf.")
        return float("inf")

    mean_score = float(np.mean(dataset_scores))
    print(f"    Mean primary val metric: {mean_score:.6f}")
    return mean_score


# -------------------------------------------------------------------------
# 9. Optimizer
# -------------------------------------------------------------------------

print("Step 4: Setting up optimizer")
print("-" * 50)

optimizer = torch.optim.AdamW(
    [p for p in pipe.model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
print(f"AdamW optimizer: {sum(p.numel() for p in pipe.model.parameters() if p.requires_grad):,} trainable params")
print()

# -------------------------------------------------------------------------
# 10. Training loop
# -------------------------------------------------------------------------

print("Step 5: Training loop")
print("-" * 50)

history = []            # list of dicts for CSV export
best_val_score  = float("inf")
best_epoch      = -1
patience_count  = 0

# Compute val metric before any training (epoch 0 = zero-shot baseline sanity check)
print("Epoch 0 (zero-shot baseline sanity check) ...")
baseline_val = compute_primary_val_metric(pipe)
print(f"  Pre-training val metric: {baseline_val:.6f}")
print()

# Load baseline from JSON to cross-check (sanity only — won't stop training)
try:
    with open(BASELINE_JSON) as f:
        baseline_data = json.load(f)
    expected_baseline = baseline_data.get("primary_metric", None)
    if expected_baseline is not None:
        diff = abs(baseline_val - expected_baseline)
        if diff > 0.05 * expected_baseline:
            print(f"  WARNING: val metric {baseline_val:.4f} differs from "
                  f"locked baseline {expected_baseline:.4f} by >{5}%.")
            print(f"  This may be due to different val vs test split. Continuing.")
        else:
            print(f"  OK: val metric consistent with locked baseline (diff={diff:.4f}).")
except Exception as e:
    print(f"  Could not load baseline JSON for cross-check: {e}")

print()

# Training epochs
for epoch in range(1, MAX_EPOCHS + 1):
    pipe.model.train()
    epoch_losses = []

    # ---- build training batches for this epoch ----
    # Strategy: one random window per container per epoch, across ALL datasets.
    # This gives a different random sample each epoch.

    # Collect all training windows for this epoch (stored as lists then batched)
    train_windows = []    # list of (context_1d, target_1d, horizon_obs) tuples

    for ds in DATASETS:
        # Pick a horizon for this dataset-epoch: cycle through available horizons
        # to ensure all horizons see training signal. Use epoch index to rotate.
        available_horizons = [h for h in HORIZONS if (ds, h) not in SKIP_CELLS]
        epoch_h_min = available_horizons[epoch % len(available_horizons)]
        h_obs = HORIZONS_OBS[(ds, epoch_h_min)]

        for arr in all_train_series[ds]:
            result = sample_window(arr, h_obs)
            if result is None:
                continue
            ctx, tgt = result
            train_windows.append((ctx, tgt, h_obs))

    # Shuffle all windows
    random.shuffle(train_windows)

    print(f"Epoch {epoch}/{MAX_EPOCHS} | {len(train_windows)} training windows")

    # Process in batches — but only process horizons one at a time per batch
    # because each batch must have a single prediction_length for the forward pass.
    # Group windows by horizon_obs first.
    from collections import defaultdict
    windows_by_horizon = defaultdict(list)
    for ctx, tgt, h_obs in train_windows:
        windows_by_horizon[h_obs].append((ctx, tgt))

    for h_obs, windows_h in windows_by_horizon.items():
        for batch_start in range(0, len(windows_h), BATCH_SIZE):
            batch = windows_h[batch_start : batch_start + BATCH_SIZE]
            if not batch:
                continue

            ctxs = np.stack([w[0] for w in batch])     # (B, N_CONTEXT)
            tgts = np.stack([w[1] for w in batch])     # (B, H)

            ctx_t = torch.tensor(ctxs, dtype=torch.float32, device=device).unsqueeze(1)   # (B,1,N_CONTEXT)
            tgt_t = torch.tensor(tgts, dtype=torch.float32, device=device)                # (B, H)

            optimizer.zero_grad()

            with torch.enable_grad():
                q_preds = forward_get_qpreds(ctx_t, h_obs, TAUS, FORWARD_MODE)
                # q_preds: (B, H, n_q)

                loss = pinball_loss_weighted(tgt_t, q_preds, TAUS, TAU_WEIGHTS)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipe.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            epoch_losses.append(float(loss.item()))

    mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

    # ---- validation ----
    print(f"  Train loss: {mean_train_loss:.6f}")
    print(f"  Computing val metric ...")
    val_score = compute_primary_val_metric(pipe)

    # ---- early stopping ----
    improved = val_score < best_val_score
    if improved:
        best_val_score = val_score
        best_epoch     = epoch
        patience_count = 0
        # Save LoRA adapter weights
        pipe.model.save_pretrained(str(MODELS_DIR))
        print(f"  NEW BEST val={val_score:.6f} — LoRA weights saved to {MODELS_DIR}")
    else:
        patience_count += 1
        print(f"  No improvement. Patience {patience_count}/{PATIENCE} (best={best_val_score:.6f} at epoch {best_epoch})")

    history.append({
        "epoch"          : epoch,
        "train_loss_mean": mean_train_loss,
        "val_pinball_primary": val_score,
        "is_best"        : improved,
        "patience_count" : patience_count,
    })

    # Save training history after every epoch (so we have it even if interrupted)
    pd.DataFrame(history).to_csv(TRAINING_HISTORY_CSV, index=False)
    print(f"  History saved to {TRAINING_HISTORY_CSV}")
    print()

    if patience_count >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}. Best was epoch {best_epoch}.")
        break

# -------------------------------------------------------------------------
# 11. Final summary
# -------------------------------------------------------------------------

print("=" * 70)
print("Training complete")
print("=" * 70)
print(f"Best val metric (pinball h=60 tau=0.9): {best_val_score:.6f} at epoch {best_epoch}")
print(f"LoRA adapter saved to                 : {MODELS_DIR}")
print(f"Training history at                   : {TRAINING_HISTORY_CSV}")
print()
print("Next step: run f3_evaluate.py (F3.5) to compute primary + secondary")
print("metrics on the test set and lock DECISION-015.")
