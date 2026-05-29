"""
simulate_hpa_v3.py — HPA simulation extended to accept full forecast-quantile path.

Extends task2_hpa_v2.simulate_hpa() with a `desired_path` callback that, at each
control sync step, returns a replica-count recommendation derived from the
forecast quantiles via f4_mpc.

Compatible with v2 reactive and ML-proactive paths. The new "chance-constrained"
mode takes a function `desired_path(t, state, cpu_history) -> int` and uses its
return value as the desired replica count directly.

Three modes:
    pred_cpu=None, desired_path=None      -> reactive (same as v2)
    pred_cpu=array, desired_path=None     -> ML-proactive (same as v2, scalar prediction)
    desired_path callable                  -> chance-constrained MPC

This script does NOT replace task2_hpa_v2.py. It is a sibling that imports
HPAConfig and HPAState from v2 to keep the simulation contracts compatible.
"""

import math
import numpy as np
from collections import deque


# Reuse the v2 classes; we import lazily to avoid module path issues when the
# user has v3 in a different directory.
def _import_v2():
    """
    Try to import task2_hpa_v2 from any reasonable location relative to this file
    OR from PROJECT_ROOT/src. Searches:
      - same directory as this script
      - parent directory
      - parent.parent/src  (e.g. .../phase_f/src — older layout)
      - parent.parent.parent/src  (e.g. .../src — actual current layout)
      - $PROJECT_ROOT/src
    """
    try:
        from task2_hpa_v2 import HPAConfig, HPAState
        return HPAConfig, HPAState
    except ImportError:
        pass

    import os
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here.parent,
        here.parent / "src",
        here.parent.parent / "src",
        here.parent.parent.parent / "src",
    ]
    project_root_env = os.environ.get("PROJECT_ROOT")
    if project_root_env:
        candidates.append(Path(project_root_env) / "src")

    for cand in candidates:
        if (cand / "task2_hpa_v2.py").exists():
            sys.path.insert(0, str(cand))
            from task2_hpa_v2 import HPAConfig, HPAState
            return HPAConfig, HPAState

    raise ModuleNotFoundError(
        "Cannot find task2_hpa_v2.py. Searched:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nEither copy task2_hpa_v2.py next to simulate_hpa_v3.py "
        "or set PROJECT_ROOT to a directory containing src/task2_hpa_v2.py."
    )


def simulate_hpa_with_desired_path(
    cpu_series,
    cfg,
    desired_path_fn,
    forecast_horizon=12,
    cadence_steps=6,
):
    """
    HPA simulation with externally-provided desired-replicas function.

    Args:
        cpu_series: 1D np.ndarray — ground-truth CPU demand per timestep
        cfg: HPAConfig
        desired_path_fn: callable (t, state, cpu_history_up_to_t) -> int
            Returns the desired replica count for the next sync interval.
            Called at every cadence_steps step starting at t=0.
        forecast_horizon: int — informational; passed to desired_path_fn so it
            knows how far ahead to look (the function reads its own forecasts
            from elsewhere; this is just metadata).
        cadence_steps: int — control interval in steps. Must match the MPC's
            re-decision cadence.

    Returns:
        dict with violation_rate, waste_rate, violation_severity, replica_trajectory,
        and detailed timestep-level arrays for downstream analysis.
    """
    HPAConfig, HPAState = _import_v2()

    T = len(cpu_series)
    state = HPAState(cfg.min_replicas)
    alloc = np.zeros(T, dtype=int)
    ready = np.zeros(T, dtype=int)
    desired_decisions = np.zeros(T, dtype=int)

    last_decision_t = -1

    for t in range(T):
        # Process pending pods
        still_pending = []
        new_ready = 0
        for (ready_at, count) in state.pending:
            if t >= ready_at:
                new_ready += count
            else:
                still_pending.append((ready_at, count))
        state.pending = still_pending
        state.ready = min(state.ready + new_ready, state.cur_replicas)

        alloc[t] = state.cur_replicas
        ready[t] = state.ready

        # Decision step?
        if t % cadence_steps != 0:
            continue

        # Call out to external decision function
        cpu_history = cpu_series[:t + 1]
        desired = desired_path_fn(t, state, cpu_history)
        desired = max(cfg.min_replicas, min(cfg.max_replicas, int(desired)))
        desired_decisions[t] = desired

        if desired > state.cur_replicas:
            add = desired - state.cur_replicas
            state.pending.append((t + cfg.pod_startup, add))
            state.cur_replicas = desired
            state.sd_recs.clear()
        elif desired < state.cur_replicas:
            state.sd_recs.append((t, desired))
            if (t - state.sd_recs[0][0]) >= cfg.scaledown_stab:
                stabilised = max(d for _, d in state.sd_recs)
                if stabilised < state.cur_replicas:
                    state.cur_replicas = stabilised
                    state.ready = min(state.ready, stabilised)
                    state.sd_recs.clear()

    capacity = ready * cfg.cpu_request
    viol_mask = cpu_series > capacity
    waste = np.maximum(0, capacity - cpu_series)

    return {
        "violation_rate":     float(np.mean(viol_mask)),
        "waste_rate":         float(np.sum(waste) / max(np.sum(capacity), 1e-9)),
        "violation_severity": float(
            np.sum(np.maximum(0, cpu_series - capacity)) /
            max(np.sum(cpu_series), 1e-9)),
        "replica_trajectory": alloc.copy(),
        "ready_trajectory":   ready.copy(),
        "capacity":           capacity.copy(),
        "viol_mask":          viol_mask.copy(),
        "desired_decisions":  desired_decisions.copy(),
    }


def make_mpc_desired_path_fn(quantile_path_lookup, alpha=0.95, mode="static_chance",
                              cpu_request=0.5, safety_margin=1.0,
                              min_replicas=1, max_replicas=100, churn_penalty=0.05):
    """
    Construct a desired_path_fn closure suitable for simulate_hpa_with_desired_path.

    Args:
        quantile_path_lookup: callable (t) -> np.ndarray shape (H,) of alpha-quantile
            forecasts for steps t+1..t+H. Returns None if no forecast available at t.
        alpha: float — chance constraint level
        mode: "static_chance" or "cvxpy_chance"
        cpu_request, safety_margin, min_replicas, max_replicas: HPA params
        churn_penalty: cvxpy only

    Returns:
        callable suitable for desired_path_fn argument of simulate_hpa_with_desired_path.
    """
    from f4_mpc import mpc_static_chance, mpc_cvxpy_chance

    def desired_path_fn(t, state, cpu_history):
        q_path = quantile_path_lookup(t)
        if q_path is None:
            # No forecast at this step; persist current replicas
            return state.cur_replicas
        if mode == "static_chance":
            return mpc_static_chance(
                q_path, cpu_request=cpu_request,
                min_replicas=min_replicas, max_replicas=max_replicas,
                safety_margin=safety_margin,
            )
        elif mode == "cvxpy_chance":
            return mpc_cvxpy_chance(
                q_path, current_replicas=state.cur_replicas,
                cpu_request=cpu_request, safety_margin=safety_margin,
                min_replicas=min_replicas, max_replicas=max_replicas,
                churn_penalty=churn_penalty,
            )
        else:
            raise ValueError(f"unknown mode {mode!r}")

    return desired_path_fn


if __name__ == "__main__":
    print("=" * 60)
    print("simulate_hpa_v3 smoke test")
    print("=" * 60)
    HPAConfig, HPAState = _import_v2()

    np.random.seed(42)
    T = 288   # 24 hours at 5-min cadence
    cpu = 30 + 20 * np.sin(np.linspace(0, 4*np.pi, T)) + np.random.normal(0, 3, T)
    cpu = np.clip(cpu, 0, 100)

    def fake_quantile_lookup(t):
        """Fake oracle: q_alpha = actual demand for the next 12 steps."""
        H = 12
        if t + H >= T:
            return None
        return cpu[t+1:t+1+H] * 1.05    # 5% over-prediction

    cfg = HPAConfig()
    cfg.safety_margin = 1.0
    cfg.max_replicas = 1000

    desired_path_fn = make_mpc_desired_path_fn(
        fake_quantile_lookup, alpha=0.95, mode="static_chance",
        cpu_request=0.5, safety_margin=1.0,
        min_replicas=cfg.min_replicas, max_replicas=cfg.max_replicas,
    )

    out = simulate_hpa_with_desired_path(
        cpu, cfg, desired_path_fn,
        forecast_horizon=12, cadence_steps=6,
    )

    print(f"violation_rate:     {out['violation_rate']:.4f}")
    print(f"waste_rate:         {out['waste_rate']:.4f}")
    print(f"violation_severity: {out['violation_severity']:.4f}")
    print(f"replica trajectory min={out['replica_trajectory'].min()}, "
          f"max={out['replica_trajectory'].max()}")
    print("PASS smoke test.")
