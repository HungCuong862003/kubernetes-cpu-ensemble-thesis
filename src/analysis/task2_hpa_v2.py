"""
task2_hpa_v2.py — Fix reactive HPA safety-margin bug and add targetCPUUtilization sweep.

The original thesis_analysis.py simulate_hpa() never applied safety_margin
in reactive mode (line 1771), so all 10 margin values produced identical
reactive outputs. This script copies the simulation with the one-line fix,
adds a target_util sweep, and produces a fair Pareto comparison.

Inputs:
    pred_naive.npy                             (CKPT_DIR/{hz}/)
    pred_hetero_ensemble.npy                   (CKPT_DIR/{hz}/)

Outputs:
    hpa_simulation_v2.csv                      (OUTPUT_DIR)
    hpa_pareto_v2.pdf                          (OUTPUT_DIR)

Run on Colab:
    # adjust paths in CONFIG section, then:
    !python task2_hpa_v2.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque


# ── CONFIG (adjust these paths for your Drive) ──────────────────────

CKPT_DIR   = "/content/drive/MyDrive/sprint1_results"
OUTPUT_DIR = "."

HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}


# ── CONSTANTS ───────────────────────────────────────────────────────

# sweep parameters
SAFETY_MARGINS = np.linspace(1.0, 1.6, 10)
REACTION_LAGS  = [1, 2, 3, 4]
TARGET_UTILS   = [0.50, 0.60, 0.70, 0.80]   # K8s typical range


# ── HELPERS ─────────────────────────────────────────────────────────

def load_npy(ckpt_dir, hz, name):
    """Load a .npy file from checkpoint directory."""
    p = os.path.join(ckpt_dir, hz, f"pred_{name}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


# ── HPA SIMULATION (copied from thesis_analysis.py, with FIX marked) ──

class HPAConfig:
    """Kubernetes HPA configuration parameters."""
    def __init__(self):
        self.min_replicas    = 1
        self.max_replicas    = 100
        self.target_util     = 0.50   # 50% target CPU utilisation
        self.cpu_request     = 0.5    # vCPU per pod
        self.tolerance       = 0.10   # ±10% dead zone (K8s default)
        self.sync_steps      = 1      # evaluate every step
        # constants calibrated for 300s timesteps, not 15s sync cycles
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
      pred_cpu=None  → reactive (uses observation from reaction_lag steps ago)
      pred_cpu=array → ML-proactive (uses model predictions)

    FIX: reactive branch now applies cfg.safety_margin (was missing in original).
    """
    T     = len(cpu_series)
    state = HPAState(cfg.min_replicas)
    alloc = np.zeros(T, dtype=int)
    ready = np.zeros(T, dtype=int)

    for t in range(T):
        # process pending pods
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

        # ── demand estimation ───────────────────────────────────────
        if pred_cpu is not None:
            demand = pred_cpu[t] * cfg.safety_margin
        else:
            # FIX: multiply by safety_margin (was missing in original
            #      thesis_analysis.py line 1771 — all 10 margin values
            #      produced identical reactive outputs)
            demand = cpu_series[max(0, t - reaction_lag)] * cfg.safety_margin

        # ── scaling decision ────────────────────────────────────────
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
            # elapsed-time check matches K8s rolling-window semantics
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
        "violation_rate":     float(np.mean(viol_mask)),
        "waste_rate":         float(np.sum(waste) / max(np.sum(capacity), 1e-9)),
        "violation_severity": float(
            np.sum(np.maximum(0, cpu_series - capacity)) /
            max(np.sum(cpu_series), 1e-9)),
    }


def pareto_front(points):
    """Return indices of Pareto-optimal points when minimising both axes."""
    sort_ix = np.argsort(points[:, 0])
    pareto  = []
    min_y   = float("inf")
    for i in range(len(sort_ix)):
        idx = sort_ix[i]
        if points[idx, 1] < min_y:
            min_y = points[idx, 1]
            pareto.append(idx)
    return np.array(pareto)


# ── MAIN ────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── step 1: verify inputs ───────────────────────────────────────

    print("checking input files ...")
    missing = []
    for hz in HORIZONS:
        for name in ["naive", "hetero_ensemble"]:
            p = os.path.join(CKPT_DIR, hz, f"pred_{name}.npy")
            if not os.path.exists(p):
                missing.append(p)

    if missing:
        print(f"\n*** {len(missing)} file(s) missing: ***")
        for p in missing:
            print(f"  {p}")
        print("\nfix paths in CONFIG section and re-run.")
        raise SystemExit(1)

    print("  all files found.\n")

    # ── step 2: run HPA sweep per horizon ───────────────────────────

    all_rows = []

    for hz, steps in HORIZONS.items():
        print(f"\n{'='*60}")
        print(f"  [{hz}]")
        print(f"{'='*60}")

        # load predictions
        naive_arr = load_npy(CKPT_DIR, hz, "naive")
        ml_arr    = load_npy(CKPT_DIR, hz, "hetero_ensemble")

        if naive_arr is None or ml_arr is None:
            print(f"  missing predictions — skipping")
            continue

        # align: take common tail
        ref = min(len(naive_arr), len(ml_arr))
        cpu_series = naive_arr[-ref:].astype(np.float64)
        ml_preds   = ml_arr[-ref:].astype(np.float64)

        print(f"  {ref:,} test samples")

        # ── reactive sweep ──────────────────────────────────────────
        n_reactive = 0
        for lag in REACTION_LAGS:
            for tu in TARGET_UTILS:
                for sm in SAFETY_MARGINS:
                    cfg = HPAConfig()
                    cfg.target_util   = tu
                    cfg.safety_margin = sm
                    r = simulate_hpa(cpu_series, cfg,
                                     pred_cpu=None, reaction_lag=lag)
                    all_rows.append({
                        "Horizon":        hz,
                        "Strategy":       "Reactive",
                        "variant":        f"lag={lag}_tu={tu}",
                        "target_util":    tu,
                        "safety_margin":  round(float(sm), 3),
                        "violation_rate": r["violation_rate"],
                        "waste_rate":     r["waste_rate"],
                        "viol_severity":  r["violation_severity"],
                    })
                    n_reactive += 1

        # ── ML-proactive sweep ──────────────────────────────────────
        n_ml = 0
        for tu in TARGET_UTILS:
            for sm in SAFETY_MARGINS:
                cfg = HPAConfig()
                cfg.target_util   = tu
                cfg.safety_margin = sm
                r = simulate_hpa(cpu_series, cfg,
                                 pred_cpu=ml_preds, reaction_lag=2)
                all_rows.append({
                    "Horizon":        hz,
                    "Strategy":       "ML-Proactive",
                    "variant":        f"ML_tu={tu}",
                    "target_util":    tu,
                    "safety_margin":  round(float(sm), 3),
                    "violation_rate": r["violation_rate"],
                    "waste_rate":     r["waste_rate"],
                    "viol_severity":  r["violation_severity"],
                })
                n_ml += 1

        print(f"  {n_reactive} reactive + {n_ml} ML = {n_reactive + n_ml} points")

        # verify fix: reactive points are now distinct
        reactive_sub = [(r["violation_rate"], r["waste_rate"])
                        for r in all_rows
                        if r["Horizon"] == hz and r["Strategy"] == "Reactive"]
        n_unique = len(set(reactive_sub))
        expected_unique = len(REACTION_LAGS) * len(TARGET_UTILS) * len(SAFETY_MARGINS)
        # won't be fully unique if some margin/lag combos produce same result,
        # but should be >> 4 (the old buggy count)
        print(f"  reactive unique points: {n_unique} "
              f"(was 4 in buggy version, max possible {expected_unique})")

    if not all_rows:
        print("\nNo HPA rows produced. Check CKPT_DIR.")
        raise SystemExit(1)

    # ── step 3: save CSV ────────────────────────────────────────────

    hpa_df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, "hpa_simulation_v2.csv")
    hpa_df.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path} ({len(hpa_df)} rows)")

    # summary stats
    for hz in HORIZONS:
        sub = hpa_df[hpa_df["Horizon"] == hz]
        if sub.empty:
            continue
        react = sub[sub["Strategy"] == "Reactive"]
        ml    = sub[sub["Strategy"] == "ML-Proactive"]
        print(f"  {hz}: {len(react)} reactive, {len(ml)} ML-proactive")

    # ── step 4: Pareto frontier plot ────────────────────────────────

    print("\nplotting Pareto frontiers ...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for i, hz in enumerate(HORIZONS):
        ax  = axes_flat[i]
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

            ax.scatter(w, v, c=color, marker=marker, alpha=0.15, s=15)

            pidx = pareto_front(pts)
            pp   = pts[pidx][np.argsort(pts[pidx, 0])]
            ax.scatter(pp[:, 0], pp[:, 1], c=color, marker=marker,
                       s=90, edgecolors="black", linewidths=1.0, zorder=5,
                       label=f"{strat} (Pareto)")
            ax.step(pp[:, 0], pp[:, 1], where="post", color=color, lw=2)

        ax.set_xlabel("Waste rate (over-provisioning)")
        ax.set_ylabel("Violation rate (QoS loss)")
        ax.set_title(hz, fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig.suptitle("HPA simulation: Reactive vs ML-Proactive — Pareto frontier\n"
                 "(corrected: reactive safety_margin + target_util sweep)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, "hpa_pareto_v2.pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf_path}")

    # ── step 5: summary ────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nBug fix applied:")
    print(f"  OLD: demand = cpu_series[max(0, t - reaction_lag)]")
    print(f"  NEW: demand = cpu_series[max(0, t - reaction_lag)] * cfg.safety_margin")
    print(f"\nSweep parameters:")
    print(f"  safety_margins: {len(SAFETY_MARGINS)} values [{SAFETY_MARGINS[0]:.1f} .. {SAFETY_MARGINS[-1]:.1f}]")
    print(f"  reaction_lags:  {REACTION_LAGS}")
    print(f"  target_utils:   {TARGET_UTILS}")
    print(f"\nTotal rows: {len(hpa_df)}")
    print(f"  Reactive:     {len(hpa_df[hpa_df['Strategy'] == 'Reactive'])} "
          f"({len(REACTION_LAGS)} lags × {len(TARGET_UTILS)} targets × "
          f"{len(SAFETY_MARGINS)} margins × {len(HORIZONS)} horizons)")
    print(f"  ML-Proactive: {len(hpa_df[hpa_df['Strategy'] == 'ML-Proactive'])} "
          f"({len(TARGET_UTILS)} targets × {len(SAFETY_MARGINS)} margins × "
          f"{len(HORIZONS)} horizons)")

    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {pdf_path}")
    print(f"\ndone.")
