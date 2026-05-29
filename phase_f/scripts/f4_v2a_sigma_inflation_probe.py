"""
f4_v2a_sigma_inflation_probe.py -- Verify the sigma-inflation diagnosis for
                                    Bitbrains over-prediction.

Hypothesis: Chronos-2 standardises context with mu + sigma (mean + std), then
the network produces a standardised q_0.9 head value, then de-normalises via
mu + sigma * sinh(z_q). On Bitbrains spike-heavy windows, sigma is inflated,
the standardised head output is roughly constant, so back-transformed q_0.9
ends up huge at idle timesteps.

Per Ansari et al. (2025, Chronos-2 tech report, Eq. 1, 5).

Test: for 10 Bitbrains spike-windows, compute:
  - mu, sigma from the 512-step context
  - the model's back-transformed q_0.9 at the median forecast step
  - the implied STANDARDISED q_0.9 z_std = sinh^-1((q_0.9_pred - mu) / sigma)
  - the truth at that timestep

Three diagnostics:
  D1. range of z_std across windows (if near-constant: confirmed)
  D2. corr(sigma, back-transformed q_0.9) (if > 0.9: confirmed)
  D3. mechanical test: predict q_0.9 from mu + sigma * sinh(mean_z_std)
      and compare to the actual q_0.9 (if MAE < 1.0: confirmed)

Run on Vast (~2 min):
    python phase_f/scripts/f4_v2a_sigma_inflation_probe.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path


PROJECT_ROOT = Path("/workspace/kubernetes-cpu-ensemble-thesis")
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LORA_ADAPTER = PROJECT_ROOT / "phase_f" / "models" / "f3_lora_rank8"

N_CONTEXT = 512
H_STEPS = 12       # h=60 at 5-min resolution


def load_pipeline_with_lora():
    """Load Chronos-2 base + LoRA-FT adapter (same as f4_v1g)."""
    from chronos import BaseChronosPipeline
    from peft import PeftModel

    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,
    )
    for cand in ("inner_model", "model", "_model", "base_model"):
        if hasattr(pipe, cand):
            inner = getattr(pipe, cand)
            wrapped = PeftModel.from_pretrained(inner, str(LORA_ADAPTER))
            setattr(pipe, cand, wrapped)
            print(f"  attached LoRA to pipe.{cand}")
            break
    pipe.model.eval()
    return pipe


def find_spike_windows(bb, max_windows=10):
    """Find Bitbrains windows: 512-step context contains spike >50,
    but median is idle (<5), and forecast median-step is also idle."""
    spike_windows = []
    for cid, g in bb.groupby("container_id"):
        g = g.sort_values("time_stamp").reset_index(drop=True)
        cpu = g["cpu_util_percent"].to_numpy(dtype=np.float32)
        if len(cpu) < N_CONTEXT + H_STEPS:
            continue
        for i in range(N_CONTEXT, len(cpu) - H_STEPS):
            ctx = cpu[i - N_CONTEXT: i]
            fut = cpu[i: i + H_STEPS]
            if ctx.max() > 50 and np.median(ctx) < 5 and fut[H_STEPS // 2] < 5:
                spike_windows.append((cid, i, ctx, fut))
                break
        if len(spike_windows) >= max_windows:
            break
    return spike_windows


def main():
    print("=" * 70)
    print("F4 V2A: SIGMA INFLATION PROBE")
    print("=" * 70)

    print("\nloading Bitbrains test data ...")
    bb = pd.read_parquet(DATA_PROCESSED / "bitbrains" / "test.parquet")
    print(f"  rows: {len(bb)}, containers: {bb['container_id'].nunique()}")

    print("\nfinding spike windows (spike in context, idle median fut step) ...")
    spike_windows = find_spike_windows(bb, max_windows=10)
    print(f"  found {len(spike_windows)} windows")
    if not spike_windows:
        print("  no spike windows found; try lowering thresholds")
        return

    print("\nloading Chronos-2 + LoRA-FT pipeline ...")
    pipe = load_pipeline_with_lora()

    print(f"\nrunning inference on {len(spike_windows)} windows ...")
    rows = []
    for cid, i, ctx, fut in spike_windows:
        mu = float(ctx.mean())
        sigma = float(ctx.std())

        # Run inference
        ctx_tensor = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_list, _ = pipe.predict_quantiles(
                ctx_tensor,
                prediction_length=H_STEPS,
                quantile_levels=[0.5, 0.9],
            )
        q_stack = torch.stack(q_list, dim=0).squeeze(1)        # (1, H, 2)
        q_arr = q_stack.detach().cpu().numpy()[0]              # (H, 2)
        median_step = H_STEPS // 2
        q_50_final = float(q_arr[median_step, 0])
        q_90_final = float(q_arr[median_step, 1])

        # Back out the standardised q_0.9: z = sinh^-1((q_90 - mu) / sigma)
        if sigma > 1e-6:
            z_std = float(np.arcsinh((q_90_final - mu) / sigma))
        else:
            z_std = 0.0

        truth = float(fut[median_step])

        rows.append({
            "cid": cid,
            "i": i,
            "ctx_max": float(ctx.max()),
            "ctx_median": float(np.median(ctx)),
            "mu": mu,
            "sigma": sigma,
            "z_std": z_std,
            "q_50_pred": q_50_final,
            "q_90_pred": q_90_final,
            "truth": truth,
            "q_90_err": q_90_final - truth,
        })

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    print("\n--- D1: standardised z_std consistency across windows ---")
    z_min, z_max = df["z_std"].min(), df["z_std"].max()
    z_mean, z_std_of_z = df["z_std"].mean(), df["z_std"].std()
    print(f"  z_std range:  [{z_min:.3f}, {z_max:.3f}]")
    print(f"  z_std mean:   {z_mean:.3f}")
    print(f"  z_std stddev: {z_std_of_z:.3f}")
    if z_std_of_z < 0.5:
        print(f"  D1 PASS: z_std is near-constant. The network's standardised q_0.9")
        print(f"           output is roughly uniform across windows.")
    else:
        print(f"  D1 PARTIAL: z_std varies across windows.")

    print("\n--- D2: correlation between sigma and back-transformed q_0.9 ---")
    if len(df) >= 3:
        corr = df[["sigma", "q_90_pred"]].corr().iloc[0, 1]
        print(f"  corr(sigma, q_0.9_pred) = {corr:.3f}")
        if corr > 0.9:
            print(f"  D2 PASS: sigma is the dominant driver of q_0.9 over-prediction.")
        else:
            print(f"  D2 FAIL: sigma is not the dominant factor.")

    print("\n--- D3: mechanical model accuracy ---")
    mean_z = df["z_std"].mean()
    df["q_90_mechanical"] = df["mu"] + df["sigma"] * np.sinh(mean_z)
    mae = (df["q_90_mechanical"] - df["q_90_pred"]).abs().mean()
    print(f"  mechanical: q_0.9 ~ mu + sigma * sinh(mean_z={mean_z:.3f})")
    print(f"  MAE vs actual q_0.9: {mae:.3f}")
    if mae < 1.0:
        print(f"  D3 PASS: back-transform is the dominant factor (MAE < 1.0).")
    else:
        print(f"  D3 INCONCLUSIVE: MAE {mae:.3f} >= 1.0, other factors at play.")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    passed = 0
    if z_std_of_z < 0.5: passed += 1
    if 'corr' in dir() and corr > 0.9: passed += 1
    if mae < 1.0: passed += 1
    print(f"  diagnostics passed: {passed}/3")
    if passed >= 2:
        print("  SIGMA-INFLATION CONFIRMED. log1p input/output transform should fix.")
        print("  Proceed to Step 2: log1p retrain (~2h GPU).")
    else:
        print("  SIGMA-INFLATION NOT CONFIRMED. log1p may not be the right fix.")
        print("  Investigate alternative mechanisms (model overfitting bursts,")
        print("  encoder feature-space corruption, etc.).")


if __name__ == "__main__":
    main()
