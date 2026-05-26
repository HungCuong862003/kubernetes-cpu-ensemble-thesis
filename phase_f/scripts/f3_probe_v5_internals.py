"""
f3_probe_v5_internals.py
Phase F3.2 — inspect Chronos-2 model structure to identify LoRA target modules.

Run on Vast.ai:
    cd /workspace/kubernetes-cpu-ensemble-thesis
    source /venv/main/bin/activate
    python phase_f/scripts/f3_probe_v5_internals.py 2>&1 | tee phase_f/data/f3_probe_v5_output.txt

Expected output: module names, param counts, LoRA candidate list.
"""

import sys
import torch
from chronos import BaseChronosPipeline

# -------------------------------------------------------------------------
# 1. Load model
# -------------------------------------------------------------------------
print("=" * 70)
print("Step 1: Loading Chronos-2 pipeline")
print("=" * 70)

pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    dtype=torch.bfloat16,
)

print(f"Pipeline class: {type(pipe).__name__}")
print(f"Inner model class: {type(pipe.model).__name__}")

# -------------------------------------------------------------------------
# 2. Count total parameters
# -------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 2: Total parameter count")
print("=" * 70)

total_params = sum(p.numel() for p in pipe.model.parameters())
trainable_params = sum(p.numel() for p in pipe.model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params:    {total_params - trainable_params:,}")

# -------------------------------------------------------------------------
# 3. Print every named module with its class and param count
# -------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 3: Named modules (name | class | param_count)")
print("=" * 70)

rows = []
for name, module in pipe.model.named_modules():
    n_params = sum(p.numel() for p in module.parameters(recurse=False))
    cls = type(module).__name__
    rows.append((name, cls, n_params))

for name, cls, n_params in rows:
    if n_params > 0:
        print(f"  {name:<60}  {cls:<30}  {n_params:>10,}")

# -------------------------------------------------------------------------
# 4. Identify Linear layers (primary LoRA targets)
# -------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 4: Linear layers only (these are the LoRA candidates)")
print("=" * 70)

linear_names = []
for name, module in pipe.model.named_modules():
    if isinstance(module, torch.nn.Linear):
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        print(f"  {name:<60}  in={module.in_features:<6}  out={module.out_features:<6}  params={n_params:>8,}")
        linear_names.append(name)

print(f"\nTotal Linear layers: {len(linear_names)}")

# -------------------------------------------------------------------------
# 5. Group linear layers by function (attention vs ffn vs head)
# -------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 5: Grouping by function")
print("=" * 70)

attention_keywords = ["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value",
                      "q", "k", "v", "self_attn", "attention"]
ffn_keywords = ["fc1", "fc2", "gate_proj", "up_proj", "down_proj", "dense", "mlp",
                "intermediate", "output.dense", "ffn"]
head_keywords = ["head", "quantile", "lm_head", "output_projection"]

attention_layers = []
ffn_layers = []
head_layers = []
other_layers = []

for name in linear_names:
    name_lower = name.lower()
    if any(kw in name_lower for kw in head_keywords):
        head_layers.append(name)
    elif any(kw in name_lower for kw in attention_keywords):
        attention_layers.append(name)
    elif any(kw in name_lower for kw in ffn_keywords):
        ffn_layers.append(name)
    else:
        other_layers.append(name)

print(f"\nAttention QKV/O layers ({len(attention_layers)}):")
for n in attention_layers:
    print(f"  {n}")

print(f"\nFFN/MLP layers ({len(ffn_layers)}):")
for n in ffn_layers:
    print(f"  {n}")

print(f"\nHead layers ({len(head_layers)}):")
for n in head_layers:
    print(f"  {n}")

print(f"\nOther/unclassified layers ({len(other_layers)}):")
for n in other_layers:
    print(f"  {n}")

# -------------------------------------------------------------------------
# 6. Recommend LoRA target_modules for peft
# -------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 6: LoRA target_modules recommendation")
print("=" * 70)

# Strategy: attention QKV + output + last-two-transformer-block FFN + head.
# We derive last-two transformer blocks by looking at block numbering in names.

import re

block_numbers = set()
for name in linear_names:
    match = re.search(r"\.(\d+)\.", name)
    if match:
        block_numbers.add(int(match.group(1)))

if block_numbers:
    sorted_blocks = sorted(block_numbers)
    last_two = sorted_blocks[-2:] if len(sorted_blocks) >= 2 else sorted_blocks
    print(f"\nTransformer block indices found: {sorted_blocks}")
    print(f"Last two blocks: {last_two}")

    last_two_layers = []
    for name in linear_names:
        for b in last_two:
            if f".{b}." in name:
                last_two_layers.append(name)
                break

    print(f"\nLayers in last two blocks ({len(last_two_layers)}):")
    for n in last_two_layers:
        print(f"  {n}")
else:
    print("No block numbering found in layer names — inspect Step 4 output manually.")
    last_two_layers = []

# Recommended target_modules: unique suffixes (peft uses suffix matching)
candidate_names = attention_layers + head_layers + last_two_layers
unique_suffixes = sorted(set(n.split(".")[-1] for n in candidate_names))

print()
print("Recommended LoRA target_modules (suffix-matched by peft):")
print(f"  {unique_suffixes}")
print()
print("Paste this into f3_chronos2_lora_finetune.py:")
print(f'  target_modules = {unique_suffixes}')

# -------------------------------------------------------------------------
# 7. Verify peft is available for LoRA
# -------------------------------------------------------------------------
print()
print("=" * 70)
print("Step 7: peft availability check")
print("=" * 70)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    print("peft imported OK")

    # Dummy LoRA config to verify it applies cleanly
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=unique_suffixes if unique_suffixes else ["q_proj", "v_proj"],
        bias="none",
    )
    print(f"LoraConfig object created OK — rank={lora_config.r}, alpha={lora_config.lora_alpha}")

    peft_model = get_peft_model(pipe.model, lora_config)
    peft_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    peft_total = sum(p.numel() for p in peft_model.parameters())
    print(f"PEFT model created OK")
    print(f"Trainable after LoRA: {peft_trainable:,} / {peft_total:,} "
          f"({100*peft_trainable/peft_total:.3f}%)")

except ImportError:
    print("ERROR: peft not installed. Run: pip install peft")
    sys.exit(1)
except Exception as e:
    print(f"ERROR applying LoRA: {e}")
    print("Inspect Step 4/5 output and pick target_modules manually.")
    sys.exit(1)

print()
print("=" * 70)
print("PROBE COMPLETE — check Step 6 output for target_modules recommendation")
print("=" * 70)
