"""
rederive_v4_per_horizon.py
==========================

Re-derive per-horizon ML-Proactive strict-dominance counts from the
canonical v4 HPA grids. Closes Q-004 and unblocks ERRATA-010.

Definition of strict ML dominance at a (horizon, target_util,
safety_margin) cell: the ML-Proactive row has BOTH
  - violation_rate < min(violation_rate of 4 Reactive lag variants)
  - waste_rate     < min(waste_rate    of 4 Reactive lag variants)

i.e. ML strictly Pareto-dominates the BEST reactive lag variant on
both axes simultaneously at the same (target_util, safety_margin).

Grid shape per (dataset, horizon): 40 cells = 4 target_util x 10
safety_margin. Each cell has 5 rows: 1 ML-Proactive + 4 Reactive
(lag = 1..4).

Alibaba and Bitbrains have 4 horizons (10/30/60/120 min) -> 160 cells.
Bytedance has 3 horizons (30/60/120 min only -> no h10) -> 120 cells.

Sanity check: memory recent_updates 2026-05-23 says Alibaba ML strict
per horizon = 40/35/15/1, Bitbrains = 40/0/0/0, Bytedance = -/33/32/29.
If outputs disagree, something is off and the operator stops before
updating ERRATA-010.

Run:
    cd /workspace/kubernetes-cpu-ensemble-thesis
    python phase_f/scripts/rederive_v4_per_horizon.py 2>&1 | \\
        tee phase_f/logs/q004_rederive_$(date +%Y%m%d_%H%M%S).log
"""

import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path("/workspace/kubernetes-cpu-ensemble-thesis")
GRID_DIR = ROOT / "results" / "bcf_v2"

FILES = {
    "alibaba":   GRID_DIR / "hpa_simulation_alibaba_v4.csv",
    "bitbrains": GRID_DIR / "hpa_simulation_bitbrains_v4.csv",
    "bytedance": GRID_DIR / "hpa_simulation_bytedance_v4.csv",
}

# Anchor values from memory recent_updates (2026-05-23 V4 HPA CANONICAL).
# Bytedance has no h10 in the v4 grid (it never had h10 in any version).
EXPECTED = {
    "alibaba":   {"10min": 40, "30min": 35, "60min": 15, "120min": 1},
    "bitbrains": {"10min": 40, "30min": 0,  "60min": 0,  "120min": 0},
    "bytedance": {                    "30min": 33, "60min": 32, "120min": 29},
}

# The CSV column names we need. If any are missing the script aborts loud,
# rather than silently mis-counting on a renamed column.
REQUIRED_COLS = ["Horizon", "Strategy", "target_util", "safety_margin",
                 "violation_rate", "waste_rate"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def strict_dominance_per_horizon(df, label):
    log(f"")
    log(f"=== {label} ===")
    log(f"  rows: {len(df):,}")
    log(f"  horizons present:   {sorted(df['Horizon'].unique())}")
    log(f"  strategies present: {sorted(df['Strategy'].unique())}")

    out = {}

    for h, g in df.groupby("Horizon", sort=False):
        n_cells = 0
        n_ml_wins = 0
        bad_cells = []

        for (tu, sm), cell in g.groupby(["target_util", "safety_margin"]):
            ml = cell[cell["Strategy"] == "ML-Proactive"]
            re = cell[cell["Strategy"] == "Reactive"]

            # Each cell must have exactly 1 ML row + 4 Reactive rows (lag=1..4).
            # Anything else is malformed and we skip it instead of guessing.
            if len(ml) != 1 or len(re) != 4:
                bad_cells.append((tu, sm, len(ml), len(re)))
                continue

            n_cells += 1

            ml_v = float(ml["violation_rate"].iloc[0])
            ml_w = float(ml["waste_rate"].iloc[0])

            # Best reactive on each axis -- this is the harshest comparison
            # for ML. If ML still wins strictly on both, it dominates the
            # entire reactive Pareto front in that cell.
            re_v_min = float(re["violation_rate"].min())
            re_w_min = float(re["waste_rate"].min())

            if ml_v < re_v_min and ml_w < re_w_min:
                n_ml_wins += 1

        out[h] = (n_ml_wins, n_cells)
        pct = 100.0 * n_ml_wins / n_cells if n_cells > 0 else 0.0
        log(f"  {h:<6}: ML strict-dominates {n_ml_wins}/{n_cells} ({pct:.1f}%)")
        if bad_cells:
            log(f"    WARN {len(bad_cells)} malformed cells "
                f"(first 3 shown as (tu, sm, n_ml, n_re)): {bad_cells[:3]}")

    return out


def main():
    log("Q-004 v4 per-horizon ML strict-dominance re-derivation")
    log("=" * 60)

    all_results = {}

    for ds, path in FILES.items():
        if not path.exists():
            log(f"FATAL: {path} not found")
            sys.exit(2)

        df = pd.read_csv(path)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            log(f"FATAL [{ds}]: missing columns {missing}. "
                f"Available: {list(df.columns)}")
            sys.exit(2)

        all_results[ds] = strict_dominance_per_horizon(df, ds)

    # -----------------------------------------------------------------
    # Sanity check vs the anchor values from memory recent_updates.
    # If anything fails, the operator must NOT update ERRATA-010 until
    # the discrepancy is understood.
    # -----------------------------------------------------------------
    log("")
    log("=== sanity check vs memory recent_updates (2026-05-23 anchor) ===")
    all_ok = True
    for ds, expected_per_h in EXPECTED.items():
        got_for_ds = all_results.get(ds, {})
        for h, expected_wins in expected_per_h.items():
            got_pair = got_for_ds.get(h)
            if got_pair is None:
                log(f"  {ds:<10} {h:<6} expected={expected_wins:>3}  "
                    f"got=MISSING_HORIZON  [MISMATCH]")
                all_ok = False
                continue
            got_wins, got_total = got_pair
            tag = "OK" if got_wins == expected_wins else "MISMATCH"
            if got_wins != expected_wins:
                all_ok = False
            log(f"  {ds:<10} {h:<6} expected={expected_wins:>3}  "
                f"got={got_wins:>3}/{got_total:<3}  [{tag}]")

    log("")
    log(f"Overall: {'PASS' if all_ok else 'FAIL -- investigate before updating ERRATA-010'}")

    # -----------------------------------------------------------------
    # ERRATA-010 substitution proposal
    # -----------------------------------------------------------------
    log("")
    log("=== ERRATA-010 substitution proposal ===")
    log("old (from submitted PDF, superseded hpa_simulation_v2.csv):")
    log("    128 / 156 / 157 / 92  (sum 533/640)")

    ali = all_results.get("alibaba", {})
    new_per_h = [ali.get(h, (None, None))[0] for h in ["10min", "30min", "60min", "120min"]]
    ali_sum = sum(x for x in new_per_h if x is not None)
    log(f"new (Alibaba v4, hpa_simulation_alibaba_v4.csv):")
    log(f"    {' / '.join(str(x) for x in new_per_h)}  (Alibaba sum = {ali_sum}/160)")

    # 11-cell total across all three datasets
    grand_total = 0
    for ds_name, ds_results in all_results.items():
        for _, (wins, _) in ds_results.items():
            grand_total += wins
    log(f"11-cell grand total: {grand_total} ML-strict (memory says 225)")

    log("")
    log("NOTE: the original ERRATA-010 row claimed 'total 533/640 sum unchanged'.")
    log("That claim is incorrect. The v4 sum is fundamentally different from the")
    log("v1-sprint sum because the underlying grids encode different replica caps")
    log("(v1: max_replicas=100, saturating 12-40% of cells; v4: max_replicas=1000).")
    log("ERRATA-010 must drop the 'sum unchanged' parenthetical and acknowledge")
    log("that Ch6 narrative 'rises to peak at h30, declines through h60/h120' is")
    log("reversed under v4: monotonic decline 40 -> 35 -> 15 -> 1.")


if __name__ == "__main__":
    main()
