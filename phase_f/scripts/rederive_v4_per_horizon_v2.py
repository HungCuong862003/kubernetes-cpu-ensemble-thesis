"""
rederive_v4_per_horizon_v2.py
=============================

CORRECTED v4 per-horizon ML-Proactive strict-dominance re-derivation.

What was wrong in v1
--------------------
The v1 script defined cell-level ML dominance as:

    ml_violation < min(reactive_violations) AND ml_waste < min(reactive_wastes)

i.e. ML must strict-beat the BEST reactive on EACH axis simultaneously.
This is the harshest reasonable definition; it returned all-zeros for
Alibaba where memory anchors say 40/35/15/1, so it's the wrong
definition for what hpa_v4_dominance_per_dataset.csv encodes.

What v2 does
------------
Standard per-lag Pareto-dominance check:

  For each lag k in 1..4 at the same (h, target_util, safety_margin):
      ML weakly dominates lag k iff
          ml_v <= lag_k_v  AND  ml_w <= lag_k_w
      ML strictly dominates lag k iff
          (ml_v <= lag_k_v AND ml_w <= lag_k_w)  AND
          (ml_v <  lag_k_v OR  ml_w <  lag_k_w)
      Reactive lag k strictly dominates ML iff
          (lag_k_v <= ml_v AND lag_k_w <= ml_w) AND
          (lag_k_v <  ml_v OR  lag_k_w <  ml_w)

  Cell-level ML strict dominance:
      ML strictly dominates EVERY lag k = 1..4 in the cell.
  Reactive-dominance count (per-pair, denominator 160 per horizon):
      Total number of (cell, lag) pairs where lag k strictly dominates ML.

Both counts are reported, so we can cross-check both "225 ML strict"
(per-cell sum across 11 groups) and "675 react dom" (per-pair sum)
against memory recent_updates.

Memory anchors (2026-05-23 V4 HPA CANONICAL):
  - per-cell ML strict:
      alibaba   40/35/15/1   bitbrains 40/0/0/0   bytedance -/33/32/29
  - per-pair react-dom %:
      alibaba   75 / 34.4 / 13.1 / 0.6
      bitbrains 73.1 / 0 / 0 / 0
      bytedance -    / 78.1 / 77.5 / 70

Run:
    cd /workspace/kubernetes-cpu-ensemble-thesis
    python phase_f/scripts/rederive_v4_per_horizon_v2.py 2>&1 | \\
        tee phase_f/logs/q004_rederive_v2_$(date +%Y%m%d_%H%M%S).log
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

# Per-cell ML strict-dominance anchors (denominator = 40 cells per horizon)
EXPECTED_CELL = {
    "alibaba":   {"10min": 40, "30min": 35, "60min": 15, "120min": 1},
    "bitbrains": {"10min": 40, "30min": 0,  "60min": 0,  "120min": 0},
    "bytedance": {                    "30min": 33, "60min": 32, "120min": 29},
}

# Per-pair reactive-dominance anchors (denominator = 160 per horizon).
# These are expected COUNTS, computed from memory's percentages * 160.
# Where percentages don't round cleanly to integers we use the closest.
EXPECTED_PAIR_REACT = {
    "alibaba":   {"10min": 120, "30min": 55, "60min": 21, "120min": 1},
    "bitbrains": {"10min": 117, "30min": 0,  "60min": 0,  "120min": 0},
    "bytedance": {                    "30min": 125, "60min": 124, "120min": 112},
}

REQUIRED_COLS = ["Horizon", "Strategy", "target_util", "safety_margin",
                 "violation_rate", "waste_rate"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ml_strict_dominates_lag(ml_v, ml_w, lag_v, lag_w):
    """True iff ML strictly Pareto-dominates this single reactive lag."""
    weakly = (ml_v <= lag_v) and (ml_w <= lag_w)
    strictly_one = (ml_v < lag_v) or (ml_w < lag_w)
    return weakly and strictly_one


def reactive_lag_strictly_dominates_ml(ml_v, ml_w, lag_v, lag_w):
    """True iff this reactive lag strictly Pareto-dominates ML."""
    weakly = (lag_v <= ml_v) and (lag_w <= ml_w)
    strictly_one = (lag_v < ml_v) or (lag_w < ml_w)
    return weakly and strictly_one


def per_horizon_counts(df, label):
    log("")
    log(f"=== {label} ===")
    log(f"  rows: {len(df):,}")
    log(f"  horizons present:   {sorted(df['Horizon'].unique())}")
    log(f"  strategies present: {sorted(df['Strategy'].unique())}")

    cell_counts = {}
    pair_react_counts = {}

    for h, g in df.groupby("Horizon", sort=False):
        n_cells = 0
        n_ml_strict_cells = 0
        n_pairs = 0
        n_react_strict_pairs = 0
        bad_cells = []

        for (tu, sm), cell in g.groupby(["target_util", "safety_margin"]):
            ml = cell[cell["Strategy"] == "ML-Proactive"]
            re = cell[cell["Strategy"] == "Reactive"]

            if len(ml) != 1 or len(re) != 4:
                bad_cells.append((tu, sm, len(ml), len(re)))
                continue

            n_cells += 1
            ml_v = float(ml["violation_rate"].iloc[0])
            ml_w = float(ml["waste_rate"].iloc[0])

            # Cell-level ML strict dominance: ML dominates EVERY one of the
            # 4 reactive lags. Counted per cell (denominator 40 per horizon).
            ml_dominates_all = True
            for _, lag_row in re.iterrows():
                lag_v = float(lag_row["violation_rate"])
                lag_w = float(lag_row["waste_rate"])
                if not ml_strict_dominates_lag(ml_v, ml_w, lag_v, lag_w):
                    ml_dominates_all = False
                # Pair-level reactive dominance count
                n_pairs += 1
                if reactive_lag_strictly_dominates_ml(ml_v, ml_w, lag_v, lag_w):
                    n_react_strict_pairs += 1

            if ml_dominates_all:
                n_ml_strict_cells += 1

        cell_counts[h] = (n_ml_strict_cells, n_cells)
        pair_react_counts[h] = (n_react_strict_pairs, n_pairs)

        cell_pct = 100.0 * n_ml_strict_cells / n_cells if n_cells > 0 else 0.0
        pair_pct = 100.0 * n_react_strict_pairs / n_pairs if n_pairs > 0 else 0.0
        log(f"  {h:<6}: ML strict-dom cells = {n_ml_strict_cells:>3}/{n_cells:<3} "
            f"({cell_pct:5.1f}%)   "
            f"Reactive-dom pairs = {n_react_strict_pairs:>3}/{n_pairs:<3} "
            f"({pair_pct:5.1f}%)")
        if bad_cells:
            log(f"    WARN {len(bad_cells)} malformed cells "
                f"(first 3 shown as (tu, sm, n_ml, n_re)): {bad_cells[:3]}")

    return cell_counts, pair_react_counts


def main():
    log("Q-004 v4 per-horizon ML strict-dominance re-derivation (v2, corrected definition)")
    log("=" * 80)

    all_cell = {}
    all_pair = {}

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

        cell, pair = per_horizon_counts(df, ds)
        all_cell[ds] = cell
        all_pair[ds] = pair

    # -----------------------------------------------------------------
    # Sanity check vs memory anchors -- both cell-level and pair-level
    # -----------------------------------------------------------------
    log("")
    log("=== sanity check vs memory recent_updates (cell-level ML strict, /40) ===")
    cell_ok = True
    for ds, expected_per_h in EXPECTED_CELL.items():
        got_for_ds = all_cell.get(ds, {})
        for h, expected_wins in expected_per_h.items():
            got_pair = got_for_ds.get(h)
            got_wins = got_pair[0] if got_pair else None
            tag = "OK" if got_wins == expected_wins else "MISMATCH"
            if got_wins != expected_wins:
                cell_ok = False
            log(f"  {ds:<10} {h:<6}  cell  expected={expected_wins:>3}  "
                f"got={got_wins!r:>5}  [{tag}]")

    log("")
    log("=== sanity check vs memory recent_updates (pair-level Reactive dom, /160) ===")
    pair_ok = True
    for ds, expected_per_h in EXPECTED_PAIR_REACT.items():
        got_for_ds = all_pair.get(ds, {})
        for h, expected_pairs in expected_per_h.items():
            got_pair = got_for_ds.get(h)
            got_n = got_pair[0] if got_pair else None
            # Allow +/-1 tolerance on pair anchors because memory's
            # percentages were rounded to 1 decimal place.
            tag = "OK"
            if got_n is None:
                tag = "MISSING"
                pair_ok = False
            elif abs(got_n - expected_pairs) > 1:
                tag = "MISMATCH"
                pair_ok = False
            log(f"  {ds:<10} {h:<6}  pair  expected~={expected_pairs:>3}  "
                f"got={got_n!r:>5}  [{tag}]")

    # -----------------------------------------------------------------
    # 11-cell grand totals
    # -----------------------------------------------------------------
    log("")
    log("=== 11-cell grand totals (memory says 225 ML strict / 675 react dom) ===")
    total_cell_strict = sum(
        wins
        for ds_counts in all_cell.values()
        for wins, _ in ds_counts.values()
    )
    total_cell_n = sum(
        n
        for ds_counts in all_cell.values()
        for _, n in ds_counts.values()
    )
    total_pair_react = sum(
        wins
        for ds_counts in all_pair.values()
        for wins, _ in ds_counts.values()
    )
    total_pair_n = sum(
        n
        for ds_counts in all_pair.values()
        for _, n in ds_counts.values()
    )
    log(f"  ML strict cells:  {total_cell_strict}/{total_cell_n}  (memory: 225)")
    log(f"  Reactive-dom pairs: {total_pair_react}/{total_pair_n}  (memory: 675)")

    log("")
    log(f"Cell anchors:  {'PASS' if cell_ok else 'FAIL'}")
    log(f"Pair anchors:  {'PASS' if pair_ok else 'FAIL'}")
    log("")
    if cell_ok:
        ali = all_cell["alibaba"]
        per_h = [ali[h][0] for h in ["10min", "30min", "60min", "120min"]]
        log("=== ERRATA-010 substitution proposal (Alibaba per-horizon) ===")
        log(f"old (submitted PDF):       128 / 156 / 157 / 92  (sum 533/640, v1-sprint scope)")
        log(f"new (Alibaba v4 cells):    {' / '.join(str(x) for x in per_h)}  "
            f"(sum {sum(per_h)}/160, v4 cell-level)")
        log(f"NOTE: prior ERRATA-010 claim 'total 533/640 sum unchanged' must be removed.")
        log(f"NOTE: Ch6 narrative 'rises to peak at h30' is FALSE under v4 -- monotonic decline.")
    else:
        log("Cell anchors did not pass. Do NOT update ERRATA-010 until investigated.")
        log("Likely next step: cat results/bcf_v2/hpa_v4_dominance_per_dataset.csv to")
        log("see what the canonical file's dominance definition actually is.")


if __name__ == "__main__":
    main()
