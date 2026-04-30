# same_sample_comparison.py
# ---------------------------------------------------------------------
# Recompute ensemble R² on the SAME (container_id, origin+h_steps)
# positions that the Chronos-2 zero-shot script evaluated at, so we can
# kill the "Chronos-2 on K=20 subsample vs ensemble on full test set"
# caveat.
#
# THIS IS THE CORRECTED VERSION. The previous version had two silent
# alignment bugs that made it NOT same-sample. Read the header below
# before trusting the numbers -- this is where the fix lives, and it's
# also where you'd want to look first if the sanity check ever trips
# again.
#
# ---------------------------------------------------------------------
# BUG 1 (target indexing off by h_steps)
# ---------------------------------------------------------------------
# sprint1's pipeline (see sprint1_Main.py:696 and loo_ablation_py.py:137)
# builds the target as:
#
#     cpu_target = g["cpu_util_percent"].shift(-horizon_steps)
#     # then dropna(subset=["cpu_target", "naive_cpu", ...])
#
# So y_true.npy is already the SHIFTED target. For a container with
# raw values [c_0, c_1, ..., c_{L-1}]:
#
#     y_true[start_idx + i]  ==  c_{i + h_steps}         for i in [0, n_points)
#
# The old script computed  global_pos = start_idx + origin + h_steps,
# which resolved to c_{origin + 2*h_steps}. That's the target for a
# LATER origin, not the one we want.
#
# Fix:  global_pos = start_idx + origin            # y_true is pre-shifted
#
# ---------------------------------------------------------------------
# BUG 2 (pick_origins got the wrong series_length)
# ---------------------------------------------------------------------
# day1_chronos2_sanity.py:286-292 builds per-container series directly
# from the raw test.parquet and calls pick_origins(len(series), ...).
# That means Chronos-2 used series_length = L_raw (the full raw length
# per container in test.parquet).
#
# The old script called pick_origins(n_points, ...) where n_points is
# the POST-dropna length = L_raw - h_steps (sprint1 drops the last
# h_steps rows when cpu_target's shift creates NaN). Since pick_origins
# uses np.linspace(lo, hi, k).round() with hi = series_length - h_steps
# - 1, different series_length produces different origin positions.
#
# Fix: reconstruct L_raw as n_points + h_steps (the algebraic identity
# given sprint1's dropna is tail-only). This sidesteps having to load
# the raw test.parquet, which isn't needed at all.
#
# An earlier version of this script tried to use raw_cpu_series.npz as
# the source of L_raw. That was wrong: the .npz actually contains the
# POST-dropna cpu_util_percent (verified by comparing against y_naive,
# which by construction is post-dropna cpu_util_percent and is byte-
# identical to raw_cpu_series.npz[cid]). Both have length n_points, not
# L_raw. The raw untruncated series lives only in test.parquet.
#
# ---------------------------------------------------------------------
# WHY THE OLD SANITY CHECK DIDN'T CATCH THIS
# ---------------------------------------------------------------------
# The old sanity check compared ensemble R² on the subsample against
# the hardcoded full-test R². Any representative subsample hits
# full-test R² within ~0.3pp regardless of whether the positions match
# Chronos-2's positions -- the ensemble is homogeneous enough across
# the test set that a random-ish subsample gets similar aggregate R².
# The old check only would have fired on catastrophic misalignment
# (random lookup), not on this quiet shift in positions.
#
# This version adds a harder check: for several real containers per
# horizon we verify the shift invariant y_true[s+i] == y_naive[s+i+h]
# at multiple positions. y_naive holds cpu_util_percent at post-dropna
# position i (the "current value" for the persistence baseline); y_true
# holds the h-step-ahead target. If sprint1's dropna was tail-only (no
# interior drops), these satisfy the shift invariant exactly. If some
# interior rows were dropped, the invariant fails at those positions
# and the lookup y_true[s+o] != cpu[o + h_steps] for origin o.
# ---------------------------------------------------------------------


# NOTE: to run on Colab, uncomment the Drive mount below. Guarded so
# that running the file outside Colab (pytest, local debug) doesn't
# crash at import time the way the previous version did.
try:
    from google.colab import drive        # type: ignore
    drive.mount('/content/drive')
except ImportError:
    pass

import os
import json
import numpy as np
import pandas as pd


# ========== CONFIG ==========================================================
# Adjust these if your Drive layout differs. The defaults match the paths
# used by chronos_benchmark.py and thesis_final_analyses_v3.py.

BASE = "/content/drive/MyDrive/k8s-ensemble-forecast"

# Where the per-container .npy files live for Alibaba. Same directory
# chronos_benchmark.py reads from (ALI_PC_DIR).
ALI_PC_DIR = os.path.join(
    BASE,
    "bitbrains-20260402T054032Z-3-001/thesis_upgrade/per_container",
)

# The Chronos-2 results JSON. If you saved it somewhere else, point
# this at the right path. We'll read num_origins and min_context from
# the JSON itself so the config can't drift.
C2_JSON_CANDIDATES = [
    os.path.join(BASE, "src/day2_chronos2/k20_alibaba_v2.json"),
    os.path.join(BASE, "day2_chronos2/k20_alibaba_v2.json"),
    os.path.join(BASE, "k20_alibaba_v2.json"),
    "/content/k20_alibaba_v2.json",
]

OUT_CSV = os.path.join(BASE, "same_sample_leaderboard.csv")

# Horizons -> prediction horizon in 5-min steps (Alibaba cadence).
HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}

# Full-test ensemble R² from comparison_table.csv. Used as sanity
# anchor only -- if our subsample R² is close to this, the subsample
# is representative; if it drifts by more than ~0.5pp, something is
# off (container subset used, origin scheme, etc.).
FULL_TEST_ENS_R2 = {
    "10min":  0.9213,
    "30min":  0.8404,
    "60min":  0.8011,
    "120min": 0.7642,
}

# ===========================================================================


def pick_origins(series_length, h_steps, k, min_context):
    """EXACT copy of the Chronos-2 origin picker
    (day1_chronos2_sanity.py:299). Do not edit without re-checking
    that script -- these have to stay in lock-step or the "same-sample"
    promise is broken.

    Returns K origin positions (0-indexed within the series) such that
    each origin has >= min_context history and enough room for an
    h_steps-ahead target. Dedup via np.unique in case linspace rounds
    to the same integer twice on short series.
    """
    lo = min_context - 1
    hi = series_length - h_steps - 1
    if hi < lo:
        return []
    if hi == lo:
        return [lo]
    origins = np.linspace(lo, hi, num=k).round().astype(int)
    return np.unique(origins).tolist()


def r2_score_pooled(y_true, y_pred):
    """Plain pooled R² -- same formula sprint1 uses everywhere.
    Written out so an examiner can verify it matches sklearn.r2_score."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def find_c2_json():
    """Try the candidate paths; return the first that exists."""
    for p in C2_JSON_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def load_c2_results():
    """Load the Chronos-2 JSON and pull K and MIN_CONTEXT from it.
    The JSON is produced by day1_chronos2_sanity.py and contains
    "num_origins" and "min_context" at the top level.

    Returns (zero_shot_dict, K, MIN_CONTEXT, json_path). Returns
    (None, None, None, None) if the JSON can't be found or parsed."""
    p = find_c2_json()
    if p is None:
        print("ERROR: could not find Chronos-2 JSON in any of:")
        for c in C2_JSON_CANDIDATES:
            print(f"   {c}")
        print("  -> edit C2_JSON_CANDIDATES at the top of the script.")
        return None, None, None, None
    print(f"Loading Chronos-2 results: {p}")
    with open(p) as f:
        blob = json.load(f)

    # Top-level should have "chronos2_zero_shot" per day1_chronos2_sanity.py.
    # Older versions may have just been the per-horizon dict directly.
    if "chronos2_zero_shot" in blob:
        zs = blob["chronos2_zero_shot"]
    else:
        print("  note: no 'chronos2_zero_shot' key, treating top level as the horizon dict")
        zs = blob
    print(f"  horizons in JSON: {list(zs.keys())}")

    # Pull K and MIN_CONTEXT directly from the JSON so they can't drift
    # out of sync with whatever run produced this file.
    K = blob.get("num_origins")
    MIN_CTX = blob.get("min_context")
    if K is None:
        print("  WARN: JSON has no 'num_origins' key -- defaulting K=20")
        K = 20
    if MIN_CTX is None:
        print("  WARN: JSON has no 'min_context' key -- defaulting MIN_CONTEXT=24")
        MIN_CTX = 24
    print(f"  K (num_origins) = {K}   MIN_CONTEXT = {MIN_CTX}")

    return zs, K, MIN_CTX, p


def pick_ensemble_file(hz_dir):
    """Prefer hetero; fall back to homo (at 10min they're identical
    because BiLSTM OOF crashed, so the homo fallback is already the
    right number there)."""
    hetero = os.path.join(hz_dir, "pred_hetero_ensemble.npy")
    homo   = os.path.join(hz_dir, "pred_homo_ensemble.npy")
    if os.path.exists(hetero):
        return hetero, "hetero_ensemble"
    if os.path.exists(homo):
        return homo, "homo_ensemble (fallback - 10min BiLSTM OOF crash)"
    return None, None


def diagnose_alignment(y_true, y_naive, idx_df, h_steps, n_check_containers=5):
    """Verify the shift invariant:

        y_true[s + i]  ==  y_naive[s + i + h_steps]    for i in [0, n_points - h_steps)

    y_naive[s + i] is cpu_util_percent at post-dropna position i (the
    "current value" used by the persistence baseline). y_true[s + i] is
    the h-step-ahead target, which should equal cpu at position i + h
    after dropna if, and only if, sprint1's dropna was tail-only (no
    interior rows dropped due to NaN in cpu_util_percent).

    For each of n_check_containers containers we verify the invariant
    at EVERY valid position (vectorised array comparison, cheap). If
    it holds on all tested positions, we have high confidence that
    global_pos = start_idx + origin resolves to Chronos-2's target.

    Returns True iff all tested positions passed.
    """
    if y_naive is None:
        print(f"  [DIAG] no y_naive.npy -- cannot verify shift invariant without it")
        return False

    # Take up to n_check_containers containers that are long enough.
    # Prefer longer ones (more positions per container = stronger test).
    candidates = []
    for _, row in idx_df.iterrows():
        n_p = int(row["n_points"])
        if n_p > h_steps + 5:
            candidates.append((n_p, row))
    if not candidates:
        print(f"  [DIAG] no containers long enough to test invariant")
        return False
    candidates.sort(key=lambda t: -t[0])
    tested = [row for _, row in candidates[:n_check_containers]]

    total_positions = 0
    total_matches   = 0
    first_failure   = None
    eps = 1e-4

    for row in tested:
        cid = str(row["container_id"])
        s   = int(row["start_idx"])
        n_p = int(row["n_points"])
        m   = n_p - h_steps         # number of valid test positions

        # Vectorized comparison: y_true[s:s+m] vs y_naive[s+h:s+h+m]
        lhs = y_true [s          : s + m]
        rhs = y_naive[s + h_steps: s + h_steps + m]
        matches = np.abs(lhs - rhs) < eps
        n_match = int(matches.sum())

        total_positions += m
        total_matches   += n_match

        if n_match < m and first_failure is None:
            bad_i = int(np.argmax(~matches))    # first failing index
            first_failure = (cid, s, n_p, bad_i, float(lhs[bad_i]), float(rhs[bad_i]))

    print(f"  [DIAG] shift invariant: {total_matches}/{total_positions} matches "
          f"across {len(tested)} containers "
          f"(n_points range {min(int(r['n_points']) for r in tested)}-"
          f"{max(int(r['n_points']) for r in tested)})")

    if total_matches == total_positions:
        print(f"  [DIAG] PASS -- y_true[s+i] == y_naive[s+i+h] holds at every tested position")
        return True

    cid, s, n_p, i, y_t, y_n = first_failure
    print(f"  [DIAG] FAIL -- first mismatch: container={cid}, s={s}, n_p={n_p}, i={i}")
    print(f"  [DIAG]   y_true[s+i]       = {y_t:.4f}")
    print(f"  [DIAG]   y_naive[s+i+h]    = {y_n:.4f}")
    print(f"  [DIAG]   -> sprint1 may have dropped interior rows for this container")
    print(f"  [DIAG]   -> origin-based lookups in y_true may be mis-aligned for some containers")
    return False


def run_one_horizon(hz, h_steps, c2_zs, K, MIN_CTX):
    """Return a row dict for the leaderboard, or None to skip."""
    print("\n" + "=" * 64)
    print(f"  {hz}  (h_steps={h_steps})")
    print("=" * 64)

    hz_dir = os.path.join(ALI_PC_DIR, hz)
    if not os.path.isdir(hz_dir):
        print(f"  SKIP: {hz_dir} does not exist")
        return None

    # ---- file existence checks ----
    ens_path, ens_name = pick_ensemble_file(hz_dir)
    if ens_path is None:
        print(f"  SKIP: no pred_{{hetero,homo}}_ensemble.npy in {hz_dir}")
        return None

    y_true_path  = os.path.join(hz_dir, "y_true.npy")
    ci_path      = os.path.join(hz_dir, "container_index.csv")
    y_naive_path = os.path.join(hz_dir, "y_naive.npy")  # strongly preferred

    for p in [y_true_path, ci_path]:
        if not os.path.exists(p):
            print(f"  SKIP: missing {p}")
            return None

    # ---- load ----
    y_ens   = np.load(ens_path)
    y_true  = np.load(y_true_path)
    idx_df  = pd.read_csv(ci_path)

    print(f"  ensemble:   {ens_name}  ({ens_path})")
    print(f"  y_ens len:  {len(y_ens):,}   y_true len: {len(y_true):,}   containers: {len(idx_df):,}")
    print(f"  idx_df cols: {list(idx_df.columns)}")

    # Length sanity: y_ens and y_true MUST be equal length.
    #
    # The previous version of this script tail-aligned them silently
    # (following residual_diagnostics_figure.py:127-129). That was wrong
    # here: residual_diagnostics uses pointwise y_true - ens_p which
    # doesn't care about absolute positions, but this script uses
    # y_ens[start_idx + origin] which does. Tail-aligning y_ens without
    # also shifting start_idx would silently misalign every lookup, and
    # the full-test R² sanity check would be too coarse to catch it.
    #
    # chronos_benchmark.py:172-173 calls get_metrics(y_true, y_ens) with
    # no alignment and doesn't crash, so in practice the arrays from
    # sprint1 have identical length. If this branch ever fires, the
    # saved files are inconsistent and need investigation, not a silent
    # patch.
    if len(y_ens) != len(y_true):
        print(f"  FATAL: y_ens/y_true length mismatch "
              f"(len(y_ens)={len(y_ens):,}, len(y_true)={len(y_true):,})")
        print(f"    container_index.csv's start_idx values are written against")
        print(f"    a specific length; silently tail-aligning would misalign")
        print(f"    every subsequent lookup. Re-save both arrays from the same")
        print(f"    post-dropna dataframe, or regenerate this horizon.")
        return None

    needed = {"container_id", "start_idx", "n_points"}
    if not needed.issubset(set(idx_df.columns)):
        print(f"  ERROR: container_index.csv missing columns {needed - set(idx_df.columns)}")
        return None

    # ---- load y_naive (strongly preferred) ----
    # Same length-mismatch rule as for y_ens: if y_naive exists but has
    # a different length from y_true, refuse to use it. We need y_naive
    # both for the shift-invariant diagnostic and for recomputing the
    # same-sample naive R². If missing, diagnostic is skipped and naive
    # R² falls back to the Chronos-2 JSON value.
    y_naive = None
    if os.path.exists(y_naive_path):
        y_naive_candidate = np.load(y_naive_path)
        if len(y_naive_candidate) == len(y_true):
            y_naive = y_naive_candidate
        else:
            print(f"  WARN: y_naive len ({len(y_naive_candidate):,}) != y_true len "
                  f"({len(y_true):,}); skipping local naive recomputation, "
                  f"will fall back to JSON")
    else:
        print(f"  WARN: no y_naive.npy -- diagnostic skipped, naive R² from JSON")

    # ---- shift-invariant diagnostic ----
    # Verify y_true[s+i] == y_naive[s+i+h] at several positions across
    # several containers. This is the only thing that validates our
    # assumption that global_pos = start_idx + origin correctly resolves
    # to Chronos-2's target.
    print(f"\n  --- alignment diagnostic (shift invariant) ---")
    diag_ok = diagnose_alignment(y_true, y_naive, idx_df, h_steps)
    if not diag_ok:
        print(f"  WARN: diagnostic did not pass cleanly. CHECK the numbers below.")

    # ---- main loop: collect same-sample predictions ----
    # L_raw = n_points + h_steps   (Chronos-2 picks origins on L_raw)
    # target at origin o = y_true[s + o]   (pre-shift cpu[o + h_steps])
    # naive  at origin o = y_naive[s + o]  (pre-shift cpu[o])
    ens_collected   = []
    true_collected  = []
    naive_collected = []

    n_used           = 0   # containers contributing >= 1 origin
    n_empty_origins  = 0   # raw series too short (n_points < min_ctx)
    n_skipped_oob    = 0   # defensive; should never trigger if lengths match

    for _, row in idx_df.iterrows():
        start_idx = int(row["start_idx"])
        n_points  = int(row["n_points"])

        # L_raw reconstruction (see header Bug 2 note).
        L_raw = n_points + h_steps

        # Bug 2 fix: use L_raw, not n_points, to match Chronos-2.
        origins = pick_origins(L_raw, h_steps, K, MIN_CTX)
        if not origins:
            n_empty_origins += 1
            continue
        n_used += 1

        for origin in origins:
            # pick_origins guarantees origin <= L_raw - h_steps - 1
            # = n_points - 1, so s + origin is always a valid index in
            # y_true/y_naive/y_ens within this container's slice.
            global_pos = start_idx + origin
            if global_pos < 0 or global_pos >= len(y_true):
                n_skipped_oob += 1
                continue

            ens_collected.append(y_ens[global_pos])
            true_collected.append(y_true[global_pos])
            if y_naive is not None:
                naive_collected.append(y_naive[global_pos])

    print(f"  containers used:          {n_used:,}")
    print(f"  empty origins (too short):{n_empty_origins:,}")
    print(f"  skipped OOB per-origin:   {n_skipped_oob:,}")
    print(f"  points collected:         {len(ens_collected):,}")

    if not ens_collected:
        print("  SKIP: no points collected")
        return None

    # ---- compare to Chronos-2's n_points in the JSON (same-sample sanity) ----
    c2_block = c2_zs.get(hz, {}) if isinstance(c2_zs, dict) else {}
    c2_n     = c2_block.get("n_points")
    if c2_n is not None:
        diff = len(ens_collected) - int(c2_n)
        # A tiny diff is fine (rounding / one container missing from
        # raw_cpu). A big diff means pick_origins disagrees somewhere.
        flag = "OK" if abs(diff) <= max(5, 0.001 * c2_n) else "MISMATCH"
        print(f"  Chronos-2 n_points (JSON): {c2_n:,}  -> diff: {diff:+d}  [{flag}]")

    # ---- metrics ----
    ens_arr  = np.asarray(ens_collected,  dtype=np.float64)
    true_arr = np.asarray(true_collected, dtype=np.float64)
    ens_r2   = r2_score_pooled(true_arr, ens_arr)

    # Naive R²: prefer recomputed from y_naive (on same positions).
    # Fall back to the Chronos-2 JSON's r2_naive_subsample only if we
    # couldn't recompute -- note this is a DIFFERENT subsample if Bug 2
    # shifted the positions, so recomputed is strictly better.
    if len(naive_collected) == len(true_collected):
        naive_arr    = np.asarray(naive_collected, dtype=np.float64)
        naive_sub_r2 = r2_score_pooled(true_arr, naive_arr)
        naive_source = "recomputed from y_naive.npy on same positions"
    else:
        naive_sub_r2 = float(c2_block.get("r2_naive_subsample", float("nan")))
        naive_source = "from Chronos-2 JSON (positions may differ)"

    try:
        c2_r2 = float(c2_block.get("r2_mean", float("nan")))
    except (TypeError, ValueError):
        c2_r2 = float("nan")

    delta_ens_c2   = (c2_r2 - ens_r2)       * 100.0 if not np.isnan(c2_r2) else float("nan")
    delta_naive_c2 = (c2_r2 - naive_sub_r2) * 100.0 if not (np.isnan(c2_r2) or np.isnan(naive_sub_r2)) else float("nan")

    full_ref      = FULL_TEST_ENS_R2.get(hz, float("nan"))
    delta_full_pp = (ens_r2 - full_ref) * 100.0 if not np.isnan(full_ref) else float("nan")

    print(f"  naive_sub_r2    = {naive_sub_r2:.4f}   ({naive_source})")
    print(f"  ensemble_sub_r2 = {ens_r2:.4f}")
    print(f"  chronos2_r2     = {c2_r2:.4f}")
    print(f"  delta_ens_vs_chronos2   = {delta_ens_c2:+.2f} pp   (positive = Chronos-2 better)")
    print(f"  delta_naive_vs_chronos2 = {delta_naive_c2:+.2f} pp")
    print(f"  sanity: ensemble_sub vs full-test R²={full_ref:.4f}  ->  diff = {delta_full_pp:+.2f} pp")
    if abs(delta_full_pp) > 0.5 and not np.isnan(delta_full_pp):
        print(f"  *** note: subsample R² is {delta_full_pp:+.2f} pp off from full-test.")
        print(f"      Expected ~0 pp after the fix. Investigate if diff stays large.")

    return {
        "dataset":              "alibaba",
        "horizon":              hz,
        "n_points":             len(true_arr),
        "naive_sub_r2":         round(naive_sub_r2, 6),
        "ensemble_sub_r2":      round(ens_r2, 6),
        "chronos2_r2":          round(c2_r2, 6),
        "delta_ens_c2_pp":      round(delta_ens_c2, 4),
        "delta_naive_c2_pp":    round(delta_naive_c2, 4),
        # Extras, useful for the thesis but not in the original spec.
        "delta_vs_full_test_pp": round(delta_full_pp, 4) if not np.isnan(delta_full_pp) else None,
        "full_test_ens_r2_ref":  full_ref,
        "c2_n_points_json":      int(c2_n) if c2_n is not None else None,
        "ensemble_source":       ens_name,
        "naive_source":          naive_source,
        "diag_passed":           bool(diag_ok),
    }


def main():
    print("=" * 64)
    print("SAME-SAMPLE ENSEMBLE vs CHRONOS-2 COMPARISON (Alibaba)")
    print("CORRECTED VERSION -- Bug 1 + Bug 2 fixed; see header.")
    print("=" * 64)
    print(f"ALI_PC_DIR = {ALI_PC_DIR}")

    if not os.path.isdir(ALI_PC_DIR):
        print(f"ERROR: ALI_PC_DIR does not exist: {ALI_PC_DIR}")
        print("Edit BASE / ALI_PC_DIR at the top of the script.")
        return

    zs, K, MIN_CTX, json_path = load_c2_results()
    if zs is None:
        print("ERROR: cannot proceed without Chronos-2 JSON (needed for K, MIN_CONTEXT).")
        return

    rows = []
    for hz, h_steps in HORIZONS.items():
        row = run_one_horizon(hz, h_steps, zs, K, MIN_CTX)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\nNo rows produced. Check the per-horizon messages above.")
        return

    df = pd.DataFrame(rows)
    # Put spec columns first, extras after.
    spec_cols = ["dataset", "horizon", "n_points",
                 "naive_sub_r2", "ensemble_sub_r2", "chronos2_r2",
                 "delta_ens_c2_pp", "delta_naive_c2_pp"]
    extra_cols = [c for c in df.columns if c not in spec_cols]
    df = df[spec_cols + extra_cols]

    out_dir = os.path.dirname(OUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 64)
    print("FINAL LEADERBOARD")
    print("=" * 64)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(df[spec_cols].to_string(index=False))
    print(f"\nSaved to: {OUT_CSV}")
    print(f"Source JSON: {json_path}   K={K}   MIN_CONTEXT={MIN_CTX}")

    # ---- sanity summary ----
    print("\n--- SANITY CHECK ---")
    for _, r in df.iterrows():
        diff_full = r.get("delta_vs_full_test_pp")
        c2_n      = r.get("c2_n_points_json")
        n_here    = r.get("n_points")
        if diff_full is None:
            continue
        flag_full = "OK" if abs(diff_full) <= 0.5 else "CHECK"
        np_diff   = (n_here - c2_n) if (c2_n is not None) else None
        flag_n    = "OK" if (np_diff is not None and abs(np_diff) <= max(5, 0.001 * c2_n)) else "CHECK"
        print(f"  {r['horizon']:>7s}: "
              f"ens_sub={r['ensemble_sub_r2']:.4f}  full-ref={r['full_test_ens_r2_ref']:.4f}  "
              f"diff={diff_full:+.2f} pp [{flag_full}]   "
              f"n_here={n_here:,}  c2_n={c2_n or 'NA'}  n_diff={np_diff if np_diff is not None else 'NA'} [{flag_n}]")
    print("\nIf any row says CHECK on diff: subsample R² drifted from full-test.")
    print("If any row says CHECK on n_diff: pick_origins is producing different")
    print("counts than Chronos-2 -- K or MIN_CONTEXT is wrong, or containers")
    print("are missing from raw_cpu_series.npz. Investigate before trusting.")


if __name__ == "__main__":
    main()