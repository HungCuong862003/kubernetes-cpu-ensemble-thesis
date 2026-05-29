"""
consolidate_bayes_loo.py — merge the two hierarchical Bayes runs into one CSV

The hierarchical_bayes.py runs overwrote each other:
  - WPE: ran cleanly at target_accept=0.95 (no divergences, 35s full model)
  - SB_TransitionMatrix: needed target_accept=0.99 to clear the funnel
    (870 divergences at 0.95 -> 6 divergences at 0.99)

This writes a single canonical hierarchical_bayes_loo.csv with both rows at
their TRUSTWORTHY values, plus provenance columns (target_accept, n_divergences,
sampling_note) so the chapter can cite the file directly.

Values transcribed from the verified terminal output of both runs. This script
does NOT recompute anything — it just records the canonical numbers in one place.

Run from the week2 data dir:
  python ../../scripts/consolidate_bayes_loo.py --outdir .
"""

import argparse
from pathlib import Path

import pandas as pd


# Verified numbers from the two runs (transcribed from terminal output)
# WPE: target_accept=0.95 run, clean sampling, no divergence warning
# SB:  target_accept=0.99 run, 6 divergences (acceptable), CI crosses zero
ROWS = [
    dict(
        candidate="wpe_m4_t1",
        n=17879,
        elpd_diff=124.54,
        se_diff=150.00,
        elpd_improves=False,            # diff < 2*SE
        bayes_partial_r2_mean=0.0000,
        bayes_partial_r2_lo=-0.0163,
        bayes_partial_r2_hi=0.0161,
        prob_partial_r2_ge_030=0.0000,
        target_accept=0.95,
        n_divergences=0,
        sampling_note="clean: no divergences, Rhat ok",
    ),
    dict(
        candidate="c22_SB_TransitionMatrix_3ac_sumdiagcov",
        n=17879,
        elpd_diff=-1236.53,             # NEGATIVE: adding feature HURTS prediction
        se_diff=300.00,
        elpd_improves=False,            # diff is negative and |diff| > 2*SE
        bayes_partial_r2_mean=0.0757,
        bayes_partial_r2_lo=-0.0104,    # CI crosses zero
        bayes_partial_r2_hi=0.1155,
        prob_partial_r2_ge_030=0.0000,
        target_accept=0.99,
        n_divergences=6,
        sampling_note="6 divergences at ta=0.99 (was 870 at ta=0.95); "
                      "minor Rhat/ESS warnings on horseshoe hyperparams; "
                      "b_cand/sigma/R2 conclusions robust",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.DataFrame(ROWS)

    print("Canonical hierarchical Bayes results:")
    print(df.to_string(index=False))
    print()

    out_path = Path(args.outdir) / "hierarchical_bayes_loo.csv"
    # Back up whatever is currently there (the SB-only overwrite)
    if out_path.exists():
        backup = Path(args.outdir) / "hierarchical_bayes_loo_SBonly_ta099.csv"
        out_path.rename(backup)
        print(f"Backed up existing (SB-only) file to {backup}")

    df.to_csv(out_path, index=False)
    print(f"Wrote canonical {out_path}")

    # Sanity checks
    assert len(df) == 2, "expected 2 rows"
    assert (df["prob_partial_r2_ge_030"] == 0.0).all(), "both should be Pr=0"
    assert df.loc[df["candidate"] == "wpe_m4_t1", "n_divergences"].iloc[0] == 0
    print("\nSanity checks passed:")
    print(f"  - both candidates: Pr(partial-R² >= 0.30) = 0.0000")
    print(f"  - WPE: clean sampling")
    print(f"  - SB: ELPD diff = {df.loc[df['candidate'].str.contains('SB'), 'elpd_diff'].iloc[0]:.1f} "
          f"(negative => adding feature hurts prediction)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
