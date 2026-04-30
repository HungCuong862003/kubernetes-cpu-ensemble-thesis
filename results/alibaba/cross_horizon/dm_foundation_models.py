"""
dm_foundation_models.py

Pairwise Diebold-Mariano tests (HLN-corrected, Harvey-Leybourne-Newbold 1997)
between foundation-model forecasts on the per-point evaluation schema used by
day1_chronos2_sanity.py / day2_timesfm_sanity.py.

Schema expected for every <model>_perpoint_<horizon>.npz:
    container_ids : (N,) int  - which container each row belongs to
    origins       : (N,) int  - origin step index inside that container
    y_true        : (N,) f    - actual future value
    y_pred_mean   : (N,) f    - point forecast (predictive mean)
    [y_naive, y_pred_p10/p50/p90, h_steps may also exist; ignored here]

Default scope: TimesFM vs Chronos-2 only.

Why nothing else by default:
    - granite_ttm has no benchmark output in this project yet.
    - hetero_ensemble lives as a ~1.7M-row dense flat array with no
      (cid, origin) keys, so it cannot be aligned against the per-point
      foundation-model arrays without first producing
      ensemble_perpoint_<horizon>.npz with the same schema. If/when you
      produce that file, pass --pairs "timesfm:hetero_ensemble" etc.

DM h parameter: in STEPS, not minutes. Compute as horizon_min / cadence_min.
    Alibaba / Bitbrains: cadence_min=5  -> 10/30/60/120min = 2/6/12/24 steps
    ByteDance:           cadence_min=10 -> 10/30/60/120min = 1/3/ 6/12 steps

Sign convention (don't get this wrong):
    d[i]   = |e1[i]|^p - |e2[i]|^p
    d_bar > 0  => model_1 has HIGHER loss  => model_2 wins
    d_bar < 0  => model_1 has LOWER  loss  => model_1 wins
    p_two_sided is the primary test for "is the gap signal or noise".
    p_one_sided (P(T > dm_hln)) tests H1: "model_1 has higher loss".
    Always read p-values together with the `winner` column.

Bonferroni: reported at two scopes. p_bonf_within multiplies by n_pairs
(family = the pairs tested at one horizon under one loss). p_bonf_global
multiplies by n_pairs * n_horizons * 2 (the *2 because MSE + MAE are
counted as separate tests). Pick whichever your committee prefers and
report only that one. Reporting both is fine; reporting whichever-passes
per row is p-hacking.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.stats import t as scipy_t


# -- DM test ------------------------------------------------------------------
# Same logic as sprint1_v9 / dm_ensemble_vs_chronos2: HLN factor sqrt(...),
# Newey-West autocov up to (h-1) lags, t(n-1) reference distribution.
def dm_test(y_true, pred1, pred2, h_steps, power):
    e1 = np.asarray(y_true - pred1, dtype=np.float64)
    e2 = np.asarray(y_true - pred2, dtype=np.float64)
    d  = np.abs(e1)**power - np.abs(e2)**power
    n  = len(d)

    if n < 3:
        return {"n": n, "dm_stat": 0.0, "dm_hln": 0.0,
                "p_one": 1.0, "p_two": 1.0, "max_lag": 0,
                "note": f"insufficient_n={n}"}

    d_bar = float(d.mean())
    d_dem = d - d_bar
    gamma_0 = float(np.mean(d_dem**2))

    # Autocov up to (h-1) lags. Cap at n//4 so we don't overfit small samples.
    max_lag = min(max(h_steps - 1, 1), n // 4)
    gamma_sum = 0.0
    for k in range(1, max_lag + 1):
        if k >= n:
            break
        gamma_sum += 2.0 * float(np.mean(d_dem[k:] * d_dem[:-k]))
    var_d = (gamma_0 + gamma_sum) / n

    dm_stat = d_bar / np.sqrt(max(var_d, 1e-12))
    hln_inner = (n + 1.0 - 2.0*h_steps + h_steps*(h_steps - 1.0)/n) / n
    hln_factor = np.sqrt(max(hln_inner, 0.0))
    dm_hln = float(hln_factor * dm_stat)

    p_one = float(scipy_t.sf(dm_hln, df=n - 1))
    p_two = float(2.0 * scipy_t.sf(abs(dm_hln), df=n - 1))

    return {"n": n, "dm_stat": float(dm_stat), "dm_hln": dm_hln,
            "p_one": p_one, "p_two": p_two,
            "max_lag": max_lag, "note": ""}


# -- I/O ----------------------------------------------------------------------
def load_perpoint(path):
    """Load a *_perpoint_*.npz file. Returns dict or None if file missing."""
    if not os.path.exists(path):
        return None
    # allow_pickle=True is REQUIRED. day2_timesfm_sanity.py and
    # day1_chronos2_sanity.py save container_ids with dtype=object (string
    # IDs like "c_2114"), which numpy stores via pickle. allow_pickle=False
    # would raise ValueError on the first load. dm_ensemble_vs_chronos2.py
    # uses True for the same reason — keep consistent.
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    required = {"container_ids", "origins", "y_true", "y_pred_mean"}
    missing = required - keys
    if missing:
        raise KeyError(
            f"{path} is missing keys {sorted(missing)}. "
            f"Found: {sorted(keys)}")

    # Cast cid/origin to a consistent dtype for merging across files. If the
    # source stored cid as an integer, force int64; otherwise fall back to
    # string. Mixed dtypes across the two files would silently produce zero
    # matches, so this matters.
    cid = np.asarray(z["container_ids"])
    if np.issubdtype(cid.dtype, np.integer):
        cid = cid.astype(np.int64)
    else:
        cid = cid.astype(str)

    return {
        "cid":    cid,
        "origin": np.asarray(z["origins"]).astype(np.int64),
        "y":      np.asarray(z["y_true"]).astype(np.float64),
        "pred":   np.asarray(z["y_pred_mean"]).astype(np.float64),
    }


def align_on_keys(a, b, name_a, name_b):
    """Inner-join on (cid, origin). Returns (y, pred_a, pred_b, n_aligned)."""
    df_a = pd.DataFrame({"cid": a["cid"], "origin": a["origin"],
                         "y_a": a["y"], "pred_a": a["pred"]})
    df_b = pd.DataFrame({"cid": b["cid"], "origin": b["origin"],
                         "y_b": b["y"], "pred_b": b["pred"]})

    # Drop intra-file duplicates first. Same (cid, origin) twice in one file
    # is a producer-side bug and would explode the merge size. Keep first.
    dup_a = df_a.duplicated(subset=["cid", "origin"]).sum()
    dup_b = df_b.duplicated(subset=["cid", "origin"]).sum()
    if dup_a or dup_b:
        sys.stderr.write(
            f"  WARN: duplicate (cid, origin) rows: "
            f"{name_a}={dup_a}, {name_b}={dup_b}. Keeping first occurrence.\n")
        df_a = df_a.drop_duplicates(subset=["cid", "origin"], keep="first")
        df_b = df_b.drop_duplicates(subset=["cid", "origin"], keep="first")

    merged = df_a.merge(df_b, on=["cid", "origin"], how="inner")
    if len(merged) == 0:
        raise ValueError(
            f"No overlapping (cid, origin) keys between {name_a} and {name_b}. "
            f"Check that both files come from the same test split and use the "
            f"same cid encoding.")

    # Sanity: y_true should match across files at the same (cid, origin).
    # Tolerate float32 -> float64 drift; warn on anything bigger.
    drift = np.abs(merged["y_a"].values - merged["y_b"].values)
    if drift.size and drift.max() > 1e-6:
        bad_n = int((drift > 1e-6).sum())
        sys.stderr.write(
            f"  WARN: y_true drift between {name_a} and {name_b}: "
            f"max={drift.max():.4e}, n_drift={bad_n}/{len(merged)}. "
            f"Likely a different test split — DM result may be invalid.\n")

    return (merged["y_a"].values.astype(np.float64),
            merged["pred_a"].values.astype(np.float64),
            merged["pred_b"].values.astype(np.float64),
            len(merged))


# -- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=".",
                    help="Folder containing <model>_perpoint_<horizon>.npz")
    ap.add_argument("--out", default="dm_foundation_models.csv")
    ap.add_argument("--horizons", nargs="+",
                    default=["10min", "30min", "60min", "120min"])
    ap.add_argument("--cadence_min", type=int, default=5,
                    help="Minutes per timestep. 5 for Alibaba/Bitbrains, "
                         "10 for ByteDance. Used to compute h_steps.")
    ap.add_argument("--pairs", nargs="+", default=["timesfm:chronos2"],
                    help="One or more 'model1:model2' tokens. Each model "
                         "name must match a *_perpoint_<hz>.npz prefix in "
                         "--results_dir.")
    args = ap.parse_args()

    pairs = []
    for tok in args.pairs:
        if ":" not in tok:
            print(f"Bad --pairs token {tok!r}, expected 'm1:m2'",
                  file=sys.stderr)
            sys.exit(2)
        m1, m2 = tok.split(":", 1)
        pairs.append((m1.strip(), m2.strip()))

    rows = []
    for hz in args.horizons:
        try:
            hz_min = int(str(hz).replace("min", "").strip())
        except ValueError:
            print(f"Cannot parse horizon {hz!r}, expected like '60min'",
                  file=sys.stderr)
            continue
        h_steps = max(hz_min // args.cadence_min, 1)

        for m1, m2 in pairs:
            p1 = os.path.join(args.results_dir, f"{m1}_perpoint_{hz}.npz")
            p2 = os.path.join(args.results_dir, f"{m2}_perpoint_{hz}.npz")

            try:
                a = load_perpoint(p1)
                b = load_perpoint(p2)
            except (KeyError, ValueError, OSError, EOFError,
                    pickle.UnpicklingError) as e:
                # KeyError              : missing required schema key.
                # ValueError            : object-array load issue, dtype mismatch.
                # OSError / EOFError    : truncated or unreadable npz.
                # pickle.UnpicklingError: corrupt-as-pickle (allow_pickle=True
                #                         path raises this on garbage bytes).
                # Skip this pair and move on; one bad file shouldn't crash
                # the whole run.
                print(f"[{hz}] SKIP {m1} vs {m2}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                continue

            if a is None:
                print(f"[{hz}] SKIP {m1} vs {m2}: missing {p1}",
                      file=sys.stderr)
                continue
            if b is None:
                print(f"[{hz}] SKIP {m1} vs {m2}: missing {p2}",
                      file=sys.stderr)
                continue

            try:
                y, pa, pb, n_aln = align_on_keys(a, b, m1, m2)
            except ValueError as e:
                print(f"[{hz}] SKIP {m1} vs {m2}: {e}", file=sys.stderr)
                continue

            mse_a = float(np.mean((y - pa)**2))
            mse_b = float(np.mean((y - pb)**2))
            mae_a = float(np.mean(np.abs(y - pa)))
            mae_b = float(np.mean(np.abs(y - pb)))

            for power, lname, la, lb in [(2, "MSE", mse_a, mse_b),
                                         (1, "MAE", mae_a, mae_b)]:
                res = dm_test(y, pa, pb, h_steps=h_steps, power=power)
                winner = m1 if la < lb else m2

                rows.append({
                    "horizon":     hz,
                    "cadence_min": args.cadence_min,
                    "h_steps":     h_steps,
                    "loss":        lname,
                    "model_1":     m1,
                    "model_2":     m2,
                    "n_aligned":   n_aln,
                    "loss_1":      la,
                    "loss_2":      lb,
                    "loss_diff":   la - lb,        # negative => m1 wins
                    "winner":      winner,
                    "dm_stat":     res["dm_stat"],
                    "dm_hln":      res["dm_hln"],
                    "p_one_sided": res["p_one"],
                    "p_two_sided": res["p_two"],
                    "max_lag":     res["max_lag"],
                    "note":        res["note"],
                })

                print(f"[{hz}] {m1:>12s} vs {m2:<12s} {lname}  "
                      f"n={n_aln:>6d}  h={h_steps:>2d}  "
                      f"DM_HLN={res['dm_hln']:+7.3f}  "
                      f"p2={res['p_two']:.2e}  "
                      f"L {la:.5f} vs {lb:.5f}  win={winner}")

    if not rows:
        print("\nNo DM results produced. Check --results_dir, --pairs, and "
              "that the *_perpoint_<horizon>.npz files exist.",
              file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    n_pairs = len(pairs)
    n_hz    = df["horizon"].nunique()
    n_loss  = df["loss"].nunique()
    df["p_bonf_within"] = (df["p_two_sided"] * max(n_pairs, 1)
                          ).clip(upper=1.0)
    df["p_bonf_global"] = (df["p_two_sided"] * max(n_pairs*n_hz*n_loss, 1)
                          ).clip(upper=1.0)
    df["sig_005"]              = df["p_two_sided"]   < 0.05
    df["sig_005_bonf_within"]  = df["p_bonf_within"] < 0.05
    df["sig_005_bonf_global"]  = df["p_bonf_global"] < 0.05

    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")
    print(f"  p_bonf_within  : alpha / {n_pairs}        "
          f"(family = pairs at one horizon, one loss)")
    print(f"  p_bonf_global  : alpha / {n_pairs*n_hz*n_loss}        "
          f"(family = all rows: {n_pairs} pairs * {n_hz} hz * "
          f"{n_loss} losses)")


if __name__ == "__main__":
    main()
