"""
f4_mpc.py — Chance-constrained Model Predictive Control for autoscaling.

Given a forecast quantile path q_alpha(t+1..t+H) for CPU demand and HPA config,
produce a target replica count for the next control interval that satisfies:

    Pr( demand(t+k) <= capacity(t+k) ) >= alpha   for all k in 1..H

The MPC formulation follows OptScaler (Zou et al. 2024, PVLDB 17:4090-4103)
with the chance constraint translated into a deterministic constraint via the
empirical quantile q_alpha:

    capacity(t+k) = replicas(t+k) * cpu_request >= q_alpha(t+k)

Two policies are supported:
    "static_chance":  replicas = ceil(max(q_alpha) * safety_margin / cpu_request),
                       clipped to [min_replicas, max_replicas].
    "cvxpy_chance":   minimise sum_k replicas(t+k) + churn_penalty * |Δreplicas|
                       subject to capacity and bounds.
"""

import math
import numpy as np


def mpc_static_chance(q_alpha_path, cpu_request=0.5, min_replicas=1, max_replicas=100,
                      safety_margin=1.0):
    """
    Provision enough replicas to cover the worst-case quantile across the
    horizon, multiplied by safety_margin, clipped to [min, max] replicas.

    Returns an int in [min_replicas, max_replicas].
    """
    if len(q_alpha_path) == 0:
        return min_replicas
    worst = float(np.max(q_alpha_path)) * safety_margin
    raw = worst / cpu_request
    desired = int(math.ceil(raw))
    return max(min_replicas, min(max_replicas, desired))


def mpc_cvxpy_chance(q_alpha_path, current_replicas, cpu_request=0.5, min_replicas=1,
                     max_replicas=100, safety_margin=1.0, churn_penalty=0.05):
    """
    Chance-constrained MPC with smoothness penalty. Returns the FIRST control action.
    Falls back to static_chance on solver failure.
    """
    import cvxpy as cp

    H = len(q_alpha_path)
    if H == 0:
        return current_replicas

    demand = q_alpha_path * safety_margin
    r = cp.Variable(H, nonneg=True)

    constraints = []
    for k in range(H):
        constraints.append(r[k] * cpu_request >= demand[k])
    constraints.append(r >= min_replicas)
    constraints.append(r <= max_replicas)

    cost_replica = cp.sum(r)
    churn = cp.abs(r[0] - current_replicas)
    if H > 1:
        churn = churn + cp.sum(cp.abs(cp.diff(r)))
    objective = cp.Minimize(cost_replica + churn_penalty * churn)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            prob.solve(verbose=False)
        except Exception:
            return mpc_static_chance(
                q_alpha_path, cpu_request=cpu_request,
                min_replicas=min_replicas, max_replicas=max_replicas,
                safety_margin=safety_margin,
            )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return mpc_static_chance(
            q_alpha_path, cpu_request=cpu_request,
            min_replicas=min_replicas, max_replicas=max_replicas,
            safety_margin=safety_margin,
        )

    r_val = np.array(r.value).flatten()
    desired = int(math.ceil(float(r_val[0])))
    return max(min_replicas, min(max_replicas, desired))


def select_alpha_quantile_path(quantile_row, alpha=0.95):
    """Pick the quantile column whose level is closest to alpha."""
    quantile_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    closest = min(quantile_levels, key=lambda q: abs(q - alpha))
    return quantile_row[f"q_{closest}"]


if __name__ == "__main__":
    print("=" * 60)
    print("F4 MPC smoke test")
    print("=" * 60)

    # Test 1: static_chance with raw demand below cap (no clipping)
    q_path_low = np.linspace(5, 25, 12)
    print(f"\nTest 1: low-demand path, max=25, no cap clipping")
    r1 = mpc_static_chance(q_path_low, cpu_request=0.5, min_replicas=1, max_replicas=100)
    expected_raw = math.ceil(25 / 0.5)   # 50
    expected = max(1, min(100, expected_raw))   # still 50, under cap
    print(f"   raw expected: {expected_raw}, after [{1},{100}] clip: {expected}")
    print(f"   result:       {r1}")
    assert r1 == expected, f"FAIL: {r1} != {expected}"
    print(f"   PASS")

    # Test 2: static_chance with demand ABOVE cap (clip activates)
    q_path_high = np.linspace(10, 80, 12)
    print(f"\nTest 2: high-demand path, max=80, cap clipping activates")
    r2 = mpc_static_chance(q_path_high, cpu_request=0.5, min_replicas=1, max_replicas=100)
    expected_raw = math.ceil(80 / 0.5)   # 160
    expected = max(1, min(100, expected_raw))   # clipped to 100
    print(f"   raw expected: {expected_raw}, after [{1},{100}] clip: {expected}")
    print(f"   result:       {r2}")
    assert r2 == expected, f"FAIL: {r2} != {expected}"
    print(f"   PASS")

    # Test 3: static_chance with max_replicas=1000 (matches v4 HPA grid)
    print(f"\nTest 3: max_replicas=1000 (matches v4 HPA grid)")
    r3 = mpc_static_chance(q_path_high, cpu_request=0.5, min_replicas=1, max_replicas=1000)
    expected_raw = math.ceil(80 / 0.5)   # 160
    expected = max(1, min(1000, expected_raw))   # 160
    print(f"   raw expected: {expected_raw}, after [{1},{1000}] clip: {expected}")
    print(f"   result:       {r3}")
    assert r3 == expected, f"FAIL: {r3} != {expected}"
    print(f"   PASS")

    # Test 4: cvxpy_chance (if available)
    try:
        import cvxpy
        r4 = mpc_cvxpy_chance(
            q_path_high, current_replicas=20, cpu_request=0.5,
            min_replicas=1, max_replicas=1000, safety_margin=1.0, churn_penalty=0.05
        )
        print(f"\nTest 4: cvxpy_chance with max=1000, current=20, churn_penalty=0.05")
        print(f"   result: {r4} replicas")
        # cvxpy should produce something >= the raw demand at step 0
        step0_min = math.ceil(10 / 0.5)   # = 20
        assert r4 >= step0_min, f"FAIL: cvxpy result {r4} below step-0 minimum {step0_min}"
        print(f"   PASS (>= step-0 minimum {step0_min})")
    except ImportError:
        print(f"\nTest 4: cvxpy not installed; static-only mode")

    print("\nAll smoke tests passed.")
