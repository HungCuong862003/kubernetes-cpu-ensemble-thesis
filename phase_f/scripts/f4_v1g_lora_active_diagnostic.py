"""
f4_v1g_lora_active_diagnostic.py — Definitively determine whether the LoRA
                                    adapter is active in pipe.predict_quantiles().

Two complementary checks:

CHECK G1: parameter-count diagnostic. Walk the pipeline's submodules,
          report trainable param count per candidate root. PEFT LoRA-rank-8
          adapter on Chronos-2 should show ~2M trainable params on the
          correctly-wrapped submodule.

CHECK G2: reproduce F3 CSV pinball on F3's exact origin set. F3 used "one
          prediction per container at a single late-trace position". This
          script replicates that. Our resulting pinball should match the F3
          CSV per-cell values within 1% (allowing only for floating-point
          and seed variation).

If G1 shows ~2M trainable params AND G2 reproduces F3 CSV within 1%:
  LoRA is active. The CHECK D mismatch in v1f was origin-set divergence,
  not LoRA bypass. F4 can proceed with the cad30 parquet.

If G1 shows 0 trainable params OR G2 reproduces zero-shot pinball instead:
  LoRA was attached but bypassed. Debug load_finetuned_pipeline.

Run:
    python phase_f/scripts/f4_v1g_lora_active_diagnostic.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/workspace/kubernetes-cpu-ensemble-thesis",
))

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LORA_ADAPTER   = PROJECT_ROOT / "phase_f" / "models" / "f3_lora_rank8"
F3_CSV         = PROJECT_ROOT / "phase_f" / "data" / "f3_eval_lora_rank8.csv"

DATASETS = ["alibaba", "bitbrains", "bytedance"]
QUANTILE_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]
N_CONTEXT = 512
BATCH_SIZE = 256
H_STEPS = 12         # h=60min at 5-min resolution; h=60min/10-min for ByteDance handled below


# ── pinball ────────────────────────────────────────────────────────

def pinball(y, q, tau):
    diff = y - q
    return np.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()


# ── G1: parameter-count diagnostic ─────────────────────────────────

def diagnose_pipeline_params(pipe):
    print("\n" + "=" * 70)
    print("CHECK G1: LoRA parameter count diagnostic")
    print("=" * 70)

    # walk all common attribute names + actually iterate pipe's __dict__ to find
    # nn.Module instances
    candidates = []
    for cand in ("inner_model", "model", "_model", "base_model", "_inner_model"):
        if hasattr(pipe, cand):
            attr = getattr(pipe, cand)
            if isinstance(attr, torch.nn.Module):
                candidates.append((cand, attr))

    # also scan __dict__ for any other torch.nn.Module
    for k, v in vars(pipe).items():
        if isinstance(v, torch.nn.Module) and k not in [c[0] for c in candidates]:
            candidates.append((k, v))

    if not candidates:
        print("  WARNING: no torch.nn.Module found on pipeline. Inspecting pipe type:")
        print(f"    type(pipe) = {type(pipe)}")
        print(f"    dir(pipe)  = {[x for x in dir(pipe) if not x.startswith('_')][:30]}")
        return False

    print(f"  Found {len(candidates)} nn.Module attribute(s) on pipeline:")
    lora_evidence_found = False
    for name, module in candidates:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"    pipe.{name}:")
        print(f"      total params:     {total:>15,}")
        print(f"      trainable params: {trainable:>15,}")
        # PEFT LoRA-rank-8 on Chronos-2 should give ~1-5M trainable
        # (depends on which target_modules were used in the LoRA config)
        if 500_000 < trainable < 10_000_000:
            print(f"      EVIDENCE: trainable in [0.5M, 10M] range -- consistent with PEFT LoRA")
            lora_evidence_found = True
        elif trainable == 0:
            print(f"      WARN: 0 trainable -- no adapter active here, or eval mode")
        elif trainable == total:
            print(f"      WARN: trainable == total -- not LoRA-wrapped (full-model)")

        # also check submodule names: PEFT inserts "lora_A" / "lora_B" submodules
        lora_modules_found = []
        for sub_name, sub_mod in module.named_modules():
            if "lora_" in sub_name.lower():
                lora_modules_found.append(sub_name)
        if lora_modules_found:
            print(f"      PEFT signature: found {len(lora_modules_found)} lora_* submodules")
            print(f"      examples: {lora_modules_found[:3]}")
            lora_evidence_found = True
        else:
            print(f"      no 'lora_*' submodules found")

    print()
    if lora_evidence_found:
        print("  G1 VERDICT: LoRA evidence found in at least one submodule.")
        print("              Adapter is loaded onto the network. Proceed to G2.")
    else:
        print("  G1 VERDICT: NO LoRA evidence in any submodule.")
        print("              Adapter is NOT loaded. The PeftModel.from_pretrained")
        print("              call must have failed silently or attached to a")
        print("              detached copy of the inner module.")
    return lora_evidence_found


# ── G2: reproduce F3 CSV pinball on F3's exact origin set ──────────

def reproduce_f3_eval(pipe, dataset, horizon_obs=12, group="main"):
    """Run inference on F3's exact origin set for one (dataset, group, h=60)
    cell and compute pinball. Compare to F3 CSV.

    F3 used ONE prediction per container at a single position. Per F3 day-1
    doc the exact position was the last possible context+horizon window:
        origin_idx = (container_length - horizon_obs)
        context    = obs[origin_idx - N_CONTEXT : origin_idx]
        truth      = obs[origin_idx : origin_idx + horizon_obs]
    """
    print(f"\n  reproducing F3 eval on {dataset} h=60 {group} group ...")

    # load full data
    parts = []
    for split in ("train", "val", "test"):
        p = DATA_PROCESSED / dataset / f"{split}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    full = pd.concat(parts, ignore_index=True)
    id_col = "container_id" if "container_id" in full.columns else "vm_id"
    cpu_col = "cpu_util_percent"
    full = full.sort_values([id_col, "time_stamp"]).reset_index(drop=True)

    # alphabetical 70/30 split per ERRATA-016 / Jimmy's clarification
    all_ids = sorted(full[id_col].unique())
    split_at = int(len(all_ids) * 0.7)
    main_ids = set(all_ids[:split_at])
    holdout_ids = set(all_ids[split_at:])
    target_ids = main_ids if group == "main" else holdout_ids

    # one origin per container at the last-valid position
    sub = full[full[id_col].isin(target_ids)]
    print(f"    containers in {group}: {sub[id_col].nunique()}")

    contexts = []
    truths = []
    for cid, g in sub.groupby(id_col, sort=False):
        cv = g[cpu_col].to_numpy().astype(np.float32)
        if len(cv) < N_CONTEXT + horizon_obs:
            continue
        # F3 used the last valid window: end of context = len - horizon_obs
        end_ctx = len(cv) - horizon_obs
        ctx = cv[end_ctx - N_CONTEXT : end_ctx]
        truth = cv[end_ctx : end_ctx + horizon_obs]
        contexts.append(ctx)
        truths.append(truth)

    n_valid = len(contexts)
    print(f"    n_valid origins: {n_valid}")
    if n_valid == 0:
        return None

    # batched inference
    all_q = []
    t_start = time.time()
    for batch_start in range(0, n_valid, BATCH_SIZE):
        batch_ctx = np.stack(contexts[batch_start:batch_start + BATCH_SIZE])
        ctx_tensor = torch.from_numpy(batch_ctx).unsqueeze(1).float()
        q_list, _ = pipe.predict_quantiles(
            ctx_tensor,
            prediction_length=horizon_obs,
            quantile_levels=QUANTILE_LEVELS,
        )
        q_stack = torch.stack(q_list, dim=0).squeeze(1)
        q_arr = q_stack.detach().cpu().numpy().astype(np.float32)
        all_q.append(q_arr)
    elapsed = time.time() - t_start
    print(f"    inference: {elapsed:.1f}s, {n_valid/elapsed:.0f} origins/s")

    quantiles = np.concatenate(all_q, axis=0)              # (n_valid, H, n_q)
    truth_arr = np.stack(truths)                            # (n_valid, H)
    print(f"    quantiles shape: {quantiles.shape}, truth shape: {truth_arr.shape}")

    # per-tau pinball
    print(f"\n  pinball loss per tau (this run):")
    rows = []
    for i, tau in enumerate(QUANTILE_LEVELS):
        pred = quantiles[:, :, i].flatten()
        truth_flat = truth_arr.flatten()
        pb = pinball(truth_flat, pred, tau)
        rows.append({"tau": tau, "pinball": pb})
        print(f"    tau={tau}: {pb:.6f}")
    return rows, n_valid


def compare_to_f3_csv(rows_by_dataset, dataset, group):
    """Compare to F3 CSV per-cell at h=60."""
    if not F3_CSV.exists():
        print(f"  F3 CSV missing; skipping comparison")
        return None
    f3 = pd.read_csv(F3_CSV)
    f3_cell = f3[(f3["dataset"] == dataset) & (f3["horizon_min"] == 60) &
                 (f3["group"] == group)]
    if len(f3_cell) == 0:
        print(f"  no F3 CSV row for {dataset} h60 {group}")
        return None

    print(f"\n  comparison to F3 CSV ({dataset} h60 {group}):")
    print(f"    {'tau':>6}  {'ours':>10}  {'f3_csv':>10}  {'rel_err':>8}")
    max_err = 0.0
    for row in rows_by_dataset:
        tau = row["tau"]
        ours = row["pinball"]
        f3_row = f3_cell[f3_cell["tau"] == tau]
        if len(f3_row) == 0:
            continue
        f3_val = float(f3_row["sym_pinball"].iloc[0])
        rel_err = (ours - f3_val) / f3_val * 100.0 if f3_val > 0 else 0.0
        max_err = max(max_err, abs(rel_err))
        print(f"    {tau:>6.2f}  {ours:>10.6f}  {f3_val:>10.6f}  {rel_err:>+7.2f}%")
    return max_err


# ── load pipeline ──────────────────────────────────────────────────

def load_pipeline_with_lora():
    """Same as resaver, but with extra prints."""
    print("loading Chronos-2 base model ...")
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )

    if not LORA_ADAPTER.exists():
        print(f"  LoRA adapter dir missing: {LORA_ADAPTER}")
        return pipe

    print(f"loading LoRA adapter from {LORA_ADAPTER} ...")
    from peft import PeftModel

    # find attribute that holds the nn.Module
    inner_attr = None
    for cand in ("inner_model", "model", "_model", "base_model"):
        if hasattr(pipe, cand) and isinstance(getattr(pipe, cand), torch.nn.Module):
            inner_attr = cand
            break
    if inner_attr is None:
        print("  no inner nn.Module found on pipeline; cannot attach adapter")
        return pipe

    print(f"  attaching adapter to pipe.{inner_attr}")
    inner = getattr(pipe, inner_attr)
    wrapped = PeftModel.from_pretrained(inner, str(LORA_ADAPTER))
    setattr(pipe, inner_attr, wrapped)

    # set to eval to be safe
    wrapped.eval()

    return pipe


# ── main ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("F4 V1G: LoRA ACTIVE DIAGNOSTIC")
    print("=" * 70)

    pipe = load_pipeline_with_lora()

    # CHECK G1
    g1_pass = diagnose_pipeline_params(pipe)

    # CHECK G2: reproduce F3 eval. Just do Alibaba main first since it's
    # the largest n and gives the cleanest comparison.
    print("\n" + "=" * 70)
    print("CHECK G2: reproduce F3 CSV pinball on F3's exact origin set")
    print("=" * 70)
    print("\n(running Alibaba main only; if this matches F3 CSV, do the others)")

    result = reproduce_f3_eval(pipe, "alibaba", horizon_obs=12, group="main")
    if result is None:
        print("  G2 FAILED: no origins")
        return
    rows, n_valid = result
    max_err = compare_to_f3_csv(rows, "alibaba", "main")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  G1 (PEFT signature found): {'PASS' if g1_pass else 'FAIL'}")
    if max_err is not None:
        g2_pass = max_err < 5.0
        print(f"  G2 (Alibaba main pinball within 5%): "
              f"{'PASS' if g2_pass else 'FAIL'} (max_err={max_err:.2f}%)")
        if g1_pass and g2_pass:
            print("\n  CONCLUSION: LoRA is active end-to-end.")
            print("  CHECK D mismatch in v1f was origin-set divergence, not LoRA bypass.")
            print("  F4 cad30 parquet is usable.")
        elif g1_pass and not g2_pass:
            print("\n  CONCLUSION: PEFT loaded but inference diverges from F3 CSV.")
            print("  Possible: different N_CONTEXT, different origin definition,")
            print("  different evaluation script. Inspect F3 fine-tune eval code.")
        elif not g1_pass:
            print("\n  CONCLUSION: PEFT not loaded. Debug load_finetuned_pipeline.")
    else:
        print("  G2: could not compare (no F3 CSV row found)")


if __name__ == "__main__":
    main()
