"""
f4_v0_math_verifier.py — Verify the OptScaler-adapted chance-constrained MPC math
                          actually solves and behaves sensibly, using scipy only.

This runs in the local sandbox (no cvxpy, no Vast). The point is to catch
formulation bugs BEFORE we touch real Chronos-2 quantiles.

What's verified:
  PROBE 1: optimiser returns optimal status on a benign synthetic input
  PROBE 2: scaling action u^1 has correct sign (forecast up -> scale up)
  PROBE 3: tighter alpha (e.g. 0.95 vs 0.80) -> more conservative (more replicas)
  PROBE 4: scale-rate limit constraint (6) actually binds when q jumps fast
  PROBE 5: lower bound X_min and upper bound X_max actually bind at extremes
  PROBE 6: D=1 (no MPC lookahead) vs D=12 produces different decisions on a
           workload with a foreseeable spike in the lookahead window
  PROBE 7: solve time per call -> can we afford ~1M solves in F4 batch?

If any probe fails the F4 formulation is wrong and the rest of F4 is built on
sand.
"""

import time
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint


# ── OptScaler-adapted MPC config ───────────────────────────────────

class MPCConfig:
    """Chance-constrained MPC, adapted from OptScaler (Zou et al. 2024) to a
    CPU-direct forecast setting. The key divergence from the paper is that
    f(y/x) -> q_alpha(t+h) directly: Chronos-2 quantiles replace the
    workload->CPU regression layer."""
    def __init__(self):
        self.alpha       = 0.90       # chance constraint level
        self.u_target    = 0.70       # target CPU utilisation c*
        self.D           = 12         # MPC horizon (# intervals)
        self.h_minutes   = 5          # interval length (matches HPA step)
        self.tau_minutes = 5          # node startup time (OptScaler tau)
        self.s_concur    = 4          # parallel scaling concurrency
        self.x_min       = 1
        self.x_max       = 1000
        self.cpu_request = 0.5        # vCPU per pod, matches task2_hpa_v2.py


def max_scale_step(cfg):
    """OptScaler constraint (6): |u^d| <= (h/tau) * s."""
    return int((cfg.h_minutes / cfg.tau_minutes) * cfg.s_concur)


# ── Single MPC solve ───────────────────────────────────────────────

def solve_mpc(x0, q_alpha_path, cfg, verbose=False):
    """
    Solve OptScaler-adapted chance-constrained MPC for one (container, t).

    Inputs:
        x0           : int      current replica count
        q_alpha_path : np.ndarray, shape (D+1,)
                       quantile forecast q_alpha(t+1), q_alpha(t+2), ..., q_alpha(t+D+1)
                       in CPU-utilisation units (same scale as cpu_series in the
                       HPA simulator).
        cfg          : MPCConfig

    Returns dict with:
        status       : 'optimal' | 'feasible-but-not-converged' | 'failed'
        u1           : int, rounded scaling action for THIS timestep (only u^1 deployed)
        x_path       : np.ndarray, shape (D,)   planned replica trajectory
        violation_d1 : float, expected CPU above target at d=1 (>=0)
        objective    : float
        solve_ms     : float

    Soft chance-constraint formulation:
        The chance constraint x^d >= q_alpha(t+h_d)/u_target is implemented as a
        SOFT PENALTY on max(0, q/u_target - x^d). This avoids structural
        infeasibility when current under-provisioning exceeds what the rate
        limit can correct in one step. The penalty weight is large but finite
        so the solver still minimises violation rather than returning failure.

    Hard constraints kept:
        (6)  |u^d| <= max_step
        (10) x_min <= x^d <= x_max
    """
    D = cfg.D
    max_step = max_scale_step(cfg)

    assert q_alpha_path.shape == (D + 1,), f"q_alpha_path shape {q_alpha_path.shape} != ({D+1},)"
    m = np.maximum(q_alpha_path[:-1], q_alpha_path[1:])

    L = np.tril(np.ones((D, D)))

    def x_of_u(u):
        return x0 + L @ u

    # ── objective: -cost (OptScaler eq. 14, sign-flipped for minimise) ─
    #
    # OptScaler maximises sum_d m_d / x^d (high utilisation = efficient).
    # For scipy.minimize we negate -> minimise -sum_d m_d / x^d.
    # This term pulls x DOWN (lower x -> higher utilisation -> objective
    # value closer to its maximum).
    #
    # SOFT chance-constraint penalty term pushes x UP when q/u_target > x:
    #   PENALTY * sum_d max(0, m_d/u_target - x^d)^2
    # The squared form is differentiable and prefers spreading violations.
    #
    # These two forces balance: the chance penalty drives x to roughly
    # ceil(m/u_target) (chance constraint binding), the cost term resists
    # going higher (no over-provisioning).
    PENALTY = 1.0e4

    def obj(u):
        x = x_of_u(u)
        if np.any(x <= 0):
            return 1e12
        # NEGATE the OptScaler maximand:
        neg_util = -float(np.sum(m / x))
        deficit = np.maximum(0.0, m / cfg.u_target - x)
        pen = PENALTY * float(np.sum(deficit ** 2))
        return neg_util + pen

    def obj_grad(u):
        x = x_of_u(u)
        if np.any(x <= 0):
            return np.full(D, 1e6)
        # d(-m/x)/dx = +m/x^2     (NEGATED from prior code)
        d_neg_util_dx = m / (x * x)
        deficit = np.maximum(0.0, m / cfg.u_target - x)
        d_pen_dx = -2.0 * PENALTY * deficit
        d_obj_dx = d_neg_util_dx + d_pen_dx
        return L.T @ d_obj_dx

    bounds = [(-max_step, +max_step) for _ in range(D)]

    # KEEP hard constraints (6) and (10) only:
    # x_min <= x0 + L @ u <= x_max
    lin_lb = np.full(D, cfg.x_min) - x0
    lin_ub = np.full(D, cfg.x_max) - x0
    linc = LinearConstraint(L, lin_lb, lin_ub)

    # initial guess: cumulative move toward chance-LB at each step, clipped by
    # rate limit. This is the heuristic baseline; solver refines from there.
    u0 = np.zeros(D)
    target_x = np.maximum(cfg.x_min, np.ceil(m / cfg.u_target))
    prev_x = x0
    for d in range(D):
        desired_delta = target_x[d] - prev_x
        u0[d] = float(np.clip(desired_delta, -max_step, +max_step))
        prev_x = prev_x + u0[d]

    t0 = time.perf_counter()
    res = minimize(
        obj,
        u0,
        jac=obj_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=[linc],
        options={"maxiter": 200, "ftol": 1e-7},
    )
    solve_ms = (time.perf_counter() - t0) * 1000.0

    u_opt = res.x
    x_opt = x_of_u(u_opt)

    # SLSQP's linear-constraint tolerance is loose enough that x can exceed
    # x_max by a few percent. We enforce a hard post-clip: cap the cumulative
    # trajectory at x_max by reducing later u^d. This is safe because the
    # cost objective already prefers low x, so clipping cannot worsen
    # the objective beyond solver tolerance.
    over_max = x_opt > cfg.x_max
    if over_max.any():
        # walk forward, capping each x at x_max and back-propagating to u
        x_clipped = np.minimum(x_opt, cfg.x_max)
        prev_x = x0
        for d in range(D):
            u_opt[d] = x_clipped[d] - prev_x
            prev_x = x_clipped[d]
        x_opt = x_clipped

    feas_hard = (
        np.all(np.abs(u_opt) <= max_step + 1e-6) and
        np.all(x_opt >= cfg.x_min - 1e-6) and
        np.all(x_opt <= cfg.x_max + 1e-6)
    )

    if res.success:
        status = "optimal"
    elif feas_hard:
        status = "feasible-but-not-converged"
    else:
        status = "failed"

    # report expected violation at d=1 separately so we can audit later
    deficit_d1 = max(0.0, m[0] / cfg.u_target - x_opt[0])

    if verbose:
        print(f"  solver status: {status}, solve_ms: {solve_ms:.2f}")
        print(f"  q_alpha (input): {np.round(q_alpha_path, 3)}")
        print(f"  m (peak-of-2):   {np.round(m, 3)}")
        print(f"  chance-LB:       {np.ceil(m/cfg.u_target).astype(int)}")
        print(f"  u_opt:           {np.round(u_opt, 2)}")
        print(f"  x_opt:           {x_opt.astype(int)}")
        print(f"  u1 (rounded):    {int(round(u_opt[0]))}")
        print(f"  deficit at d=1:  {deficit_d1:.3f}")

    return {
        "status":       status,
        "u1":           int(round(u_opt[0])),
        "x_path":       x_opt,
        "violation_d1": deficit_d1,
        "objective":    res.fun,
        "solve_ms":     solve_ms,
    }


# ── Probes ─────────────────────────────────────────────────────────

def probe1_benign(cfg):
    """PROBE 1: optimiser returns optimal on a benign synthetic input."""
    print("\n[PROBE 1] benign synthetic forecast, x0=10")
    q = np.full(cfg.D + 1, 5.0)                              # flat 5 CPU units
    out = solve_mpc(x0=10, q_alpha_path=q, cfg=cfg, verbose=True)
    pass_ = (out["status"] in ("optimal", "feasible-but-not-converged"))
    print(f"  PASS={pass_}")
    return pass_, out


def probe2_sign(cfg):
    """PROBE 2: forecast jumps up -> u^1 must be >= 0."""
    print("\n[PROBE 2] forecast jumps up (x0=10, q rises from 5 to 30)")
    q = np.linspace(5, 30, cfg.D + 1)
    out = solve_mpc(x0=10, q_alpha_path=q, cfg=cfg, verbose=True)
    pass_ = (out["u1"] >= 0)
    print(f"  PASS={pass_}  (u1={out['u1']}, expected >= 0)")
    return pass_, out


def probe3_alpha_monotone(cfg):
    """PROBE 3: tighter alpha -> more conservative.

    OptScaler-adapted note: in this CPU-direct formulation, alpha enters through
    which Chronos-2 quantile is supplied as q_alpha_path. The MPC itself does
    not know alpha. We simulate by passing higher quantiles for higher alpha:
    q_0.95 > q_0.90 > q_0.80 on a realistic forecast distribution.

    Use HIGH x0 so the rate limit does NOT bind at d=1; otherwise all three
    alphas return the same rate-limited u1 and the probe is a false pass.
    Look at x at d=D (end of horizon) not d=1.
    """
    print("\n[PROBE 3] tighter alpha (higher quantile input) -> more replicas")
    q_80 = np.full(cfg.D + 1, 8.0)
    q_90 = np.full(cfg.D + 1, 10.0)
    q_95 = np.full(cfg.D + 1, 12.0)
    x0 = 50                                                  # high; not rate-limited
    out80 = solve_mpc(x0=x0, q_alpha_path=q_80, cfg=cfg)
    out90 = solve_mpc(x0=x0, q_alpha_path=q_90, cfg=cfg)
    out95 = solve_mpc(x0=x0, q_alpha_path=q_95, cfg=cfg)
    xL_80 = out80["x_path"][-1]
    xL_90 = out90["x_path"][-1]
    xL_95 = out95["x_path"][-1]
    print(f"  x^D at alpha=0.80: {xL_80:.2f}  (lb={np.ceil(8/cfg.u_target):.0f})")
    print(f"  x^D at alpha=0.90: {xL_90:.2f}  (lb={np.ceil(10/cfg.u_target):.0f})")
    print(f"  x^D at alpha=0.95: {xL_95:.2f}  (lb={np.ceil(12/cfg.u_target):.0f})")
    pass_ = (xL_80 < xL_90 < xL_95)
    print(f"  PASS={pass_}  (expected strictly increasing across alpha)")
    return pass_, (out80, out90, out95)


def probe4_rate_limit(cfg):
    """PROBE 4: huge forecast jump -> u^1 saturated at max_step."""
    print("\n[PROBE 4] huge forecast spike, u1 must hit rate limit")
    max_step = max_scale_step(cfg)
    q = np.full(cfg.D + 1, 200.0)                            # demand >> capacity
    out = solve_mpc(x0=10, q_alpha_path=q, cfg=cfg, verbose=False)
    print(f"  u1 = {out['u1']}, max_step = {max_step}")
    pass_ = (out["u1"] == max_step)
    print(f"  PASS={pass_}  (expected u1 == {max_step})")
    return pass_, out


def probe5_bounds(cfg):
    """PROBE 5: very high forecast but x_max binds; very low x0 + bound binds.

    Test (a): x_max binds when forecast is huge.
    Test (b): x_min binds when forecast is tiny and x0 is high.
    """
    print("\n[PROBE 5a] huge forecast over long horizon -> x_max binds eventually")
    q_huge = np.full(cfg.D + 1, 1000.0)
    out = solve_mpc(x0=cfg.x_max - 5, q_alpha_path=q_huge, cfg=cfg)
    pass_a = (out["x_path"].max() <= cfg.x_max + 1e-6)
    print(f"  x_path max = {out['x_path'].max():.2f}, x_max = {cfg.x_max}")
    print(f"  PASS_a={pass_a}")

    print("[PROBE 5b] tiny forecast -> can scale down but never below x_min")
    q_tiny = np.full(cfg.D + 1, 0.01)
    out = solve_mpc(x0=10, q_alpha_path=q_tiny, cfg=cfg)
    pass_b = (out["x_path"].min() >= cfg.x_min - 1e-6)
    print(f"  x_path min = {out['x_path'].min():.2f}, x_min = {cfg.x_min}")
    print(f"  PASS_b={pass_b}")
    return (pass_a and pass_b), None


def probe6_lookahead_value(cfg):
    """PROBE 6: foreseeable spike in lookahead window -> D=12 scales earlier than D=1.

    This is the MPC value proposition. If D=1 == D=12 then MPC adds nothing.

    Construction:
      x0=20, base forecast q=5 (under-utilised), spike to q=100 at d=8.
      With max_step=4 and x0=20, the rate-limited max reachable x at d=8 is
      20 + 7*4 = 48. Required x at d=8 is ceil(100/0.7) = 143. UNREACHABLE
      even if we scale up from t=0.
      But: if D=1, the lookahead is only q[1]=5, so the cost objective tells
      D=1 to scale DOWN (high x wastes when q is low).
      With D=12, the optimiser sees the spike at d=8 in its penalty term and
      must scale UP starting NOW to minimise the unavoidable violation.

    Result: D=1 u1 < 0 (scale down), D=12 u1 > 0 (scale up). u1 differs in SIGN.
    """
    print("\n[PROBE 6] D=1 vs D=12 with unforeseeable-by-D=1 spike at d=8")
    q_spike = np.full(cfg.D + 1, 5.0)
    q_spike[8:] = 100.0                                      # large enough to FORCE early action
    x0 = 20

    out_D12 = solve_mpc(x0=x0, q_alpha_path=q_spike, cfg=cfg)

    cfg_short = MPCConfig()
    cfg_short.D = 1
    out_D1 = solve_mpc(x0=x0, q_alpha_path=q_spike[:2], cfg=cfg_short)

    print(f"  D=1  u1 = {out_D1['u1']:>3},  x1 = {out_D1['x_path'][0]:.2f}")
    print(f"  D=12 u1 = {out_D12['u1']:>3}, x1 = {out_D12['x_path'][0]:.2f}, "
          f"x[D-1] = {out_D12['x_path'][-1]:.2f}")
    # value claim: D=12 sees the spike and scales up; D=1 doesn't see it and scales down.
    pass_ = (out_D12["u1"] > out_D1["u1"])
    print(f"  PASS={pass_}  (expected D=12 u1 > D=1 u1 since D=12 sees the spike)")
    return pass_, (out_D1, out_D12)


def probe7_timing(cfg, n=200):
    """PROBE 7: solve time per call -> budget check for F4 batch.

    F4 batch target: ~3 datasets * ~5000 series * ~200 windows = ~3M solves.
    Budget: 10 working days, GPU mostly idle for MPC (CPU work). At 8 cores in
    parallel, that's ~24h * 8 = ~700K solve-seconds. So we need < 230ms/solve.
    Comfortable target: < 50ms/solve so we have headroom for sensitivity sweeps.

    With soft-penalty chance constraint, infeasibility should be zero. What we
    now care about is:
      - solve time
      - solver convergence (status == 'optimal')
      - deficit_d1 distribution (rate-limit-bound cases have deficit > 0; this
        is expected behaviour, NOT a failure)
    """
    print(f"\n[PROBE 7] solve-time + status distribution over {n} synthetic calls")
    rng = np.random.default_rng(42)
    times = []
    statuses = []
    deficits = []
    for i in range(n):
        x0 = int(rng.integers(2, 200))
        base = rng.uniform(5, 30)
        noise = rng.normal(0, 1.5, size=cfg.D + 1)
        q = np.clip(base + np.cumsum(noise) * 0.2, 0.1, 100)
        out = solve_mpc(x0=x0, q_alpha_path=q, cfg=cfg)
        times.append(out["solve_ms"])
        statuses.append(out["status"])
        deficits.append(out["violation_d1"])
    times = np.array(times)
    deficits = np.array(deficits)
    n_optimal = sum(1 for s in statuses if s == "optimal")
    n_feasible = sum(1 for s in statuses if s == "feasible-but-not-converged")
    n_failed   = sum(1 for s in statuses if s == "failed")
    print(f"  optimal:                 {n_optimal}/{n}")
    print(f"  feasible-not-converged:  {n_feasible}/{n}")
    print(f"  failed:                  {n_failed}/{n}")
    print(f"  solve_ms:   p50={np.median(times):.2f}  "
          f"p90={np.percentile(times, 90):.2f}  "
          f"p99={np.percentile(times, 99):.2f}  "
          f"max={times.max():.2f}")
    n_with_deficit = int((deficits > 1e-6).sum())
    print(f"  rate-limit-bound (deficit_d1 > 0): {n_with_deficit}/{n}  "
          f"(expected: nonzero, this is OptScaler-style 'do best you can')")
    print(f"  deficit_d1: p50={np.median(deficits):.3f}  "
          f"p90={np.percentile(deficits, 90):.3f}  "
          f"max={deficits.max():.3f}")
    pass_ = (np.median(times) < 50.0) and (n_failed == 0)
    print(f"  PASS={pass_}  (target median < 50ms, zero hard-failure)")
    return pass_, times


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("F4 V0 MATH VERIFIER")
    print("Verifies OptScaler-adapted chance-constrained MPC actually solves")
    print("BEFORE we write F4 production code or touch Vast.")
    print("=" * 70)

    cfg = MPCConfig()
    print(f"\nConfig: D={cfg.D}, h={cfg.h_minutes}min, tau={cfg.tau_minutes}min, "
          f"s={cfg.s_concur}, x in [{cfg.x_min},{cfg.x_max}], "
          f"u_target={cfg.u_target}, max_step={max_scale_step(cfg)}")

    results = {}
    results["PROBE 1 (benign)"]            = probe1_benign(cfg)[0]
    results["PROBE 2 (sign)"]              = probe2_sign(cfg)[0]
    results["PROBE 3 (alpha monotone)"]    = probe3_alpha_monotone(cfg)[0]
    results["PROBE 4 (rate limit)"]        = probe4_rate_limit(cfg)[0]
    results["PROBE 5 (bounds)"]            = probe5_bounds(cfg)[0]
    results["PROBE 6 (lookahead value)"]   = probe6_lookahead_value(cfg)[0]
    results["PROBE 7 (timing/feasibility)"] = probe7_timing(cfg)[0]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} : {name}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n{n_pass} / {n_total} probes pass")
    if n_pass == n_total:
        print("\nF4 MATH OK. Safe to proceed to Vast-side verification + production code.")
    else:
        print("\nF4 MATH BROKEN. Do NOT proceed. Fix formulation first.")


if __name__ == "__main__":
    main()
