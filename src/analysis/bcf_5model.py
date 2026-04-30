"""
bcf_5model.py — Boundary Condition Framework AUC, extended to 5 models.

Reconstruction-and-extension of the original boundary_condition_stats.py.
Adds Granite-TTM (IBM Tiny Time Mixers) zero-shot as the 5th row in
bcf_per_model_auc.csv, alongside the four existing models (Naive, NNLS,
Chronos-2, TimesFM).

If the TTM JSON files are not on disk, the script gracefully skips TTM
and produces the same 4-row output as the original script. If the TTM
JSONs exist but are incomplete (partial dataset or horizon coverage
during a long-running zero-shot experiment) the script hard-fails with
a coverage report rather than silently producing a contaminated pooled
AUC — see assert_symmetric_coverage() and the partial-data note below.

Predicate:    (ACF@24h > 0.2) AND (horizon >= 30min)
Datasets:     Alibaba (ACF@24h=0.316), Bitbrains (0.116), ByteDance (0.489)
Horizons:     10min, 30min, 60min, 120min
Per-model:    12 cells (3 datasets x 4 horizons), Mann-Whitney AUC
Pooled:       3 or 4 ML models x 12 cells = 36 or 48 cells
              BCa 95% CI (n_boot=10000)
              Phipson-Smyth permutation p (n_perm=10000)

Inputs (all in --data-dir, defaults to '.'):
    bitbrains_per_vm.csv          # NNLS Bitbrains per-VM aggregation
    chronos_benchmark_results.csv # NNLS Alibaba/ByteDance R^2 (full-test)
    omega_summary.csv             # ACF@24h per dataset (sanity check)
    k20_alibaba.json              # Chronos-2 Alibaba
    k20_bitbrains.json            # Chronos-2 Bitbrains
    k20_bytedance_v2.json         # Chronos-2 ByteDance
    timesfm_k20_alibaba.json      # TimesFM Alibaba
    timesfm_k50_bitbrains.json    # TimesFM Bitbrains
    timesfm_k50_bytedance.json    # TimesFM ByteDance
    ttm_k20_alibaba.json          # Granite-TTM Alibaba (optional)
    ttm_k50_bitbrains.json        # Granite-TTM Bitbrains (optional)
    ttm_k50_bytedance.json        # Granite-TTM ByteDance (optional)

Outputs (in --output-dir):
    bcf_per_model_auc.csv         # 4 or 5 rows
    bcf_pairs.csv                 # 48 or 60 rows (model x dataset x horizon)
    bcf_pooled_results.json       # pooled AUC + CI + p-value

Run:
    python3 bcf_5model.py --data-dir . --output-dir .

Verification:
    With no TTM JSONs present, the script reproduces the 4-row
    bcf_per_model_auc.csv and 48-row bcf_pairs.csv exactly.
"""

import os
import sys
import csv
import json
import math
import random
import argparse
from collections import OrderedDict


# ── CONFIG ──────────────────────────────────────────────────────────────

# ACF@24h per dataset, used to evaluate the boundary predicate. These are
# the values from omega_summary.csv (and boundary_condition_table_corrected.csv);
# we hard-code them here as ground truth, then sanity-check against the CSV.
ACF_24H = {
    "Alibaba":   0.316,
    "Bitbrains": 0.116,
    "ByteDance": 0.489,
}

# Order in which datasets and horizons appear in the output CSVs.
DATASETS = ["Alibaba", "Bitbrains", "ByteDance"]
HORIZONS = ["10min", "30min", "60min", "120min"]
HORIZON_MIN = {"10min": 10, "30min": 30, "60min": 60, "120min": 120}

# Predicate: (ACF@24h > 0.2) AND (horizon >= 30min). Returns 0/1.
def predicate(dataset, horizon):
    return int(ACF_24H[dataset] > 0.2 and HORIZON_MIN[horizon] >= 30)

# Source-file map for each foundation model. Keep these as constants at the
# top so they're easy to override if the canonical K choice ever changes.
# These were verified against the existing bcf_pairs.csv values:
#   - Chronos-2 used k20 for all three datasets (k20_bytedance_v2 specifically)
#   - TimesFM used k20 for Alibaba, k50 for Bitbrains/ByteDance
#   - TTM file naming follows the TimesFM convention (best guess until
#     the JSONs actually exist on disk)
CHRONOS_FILES = {
    "Alibaba":   "k20_alibaba_v2.json",
    "Bitbrains": "k20_bitbrains_v2.json",
    "ByteDance": "k20_bytedance_v2.json",
}
TIMESFM_FILES = {
    "Alibaba":   "timesfm_k20_alibaba.json",
    "Bitbrains": "timesfm_k50_bitbrains.json",
    "ByteDance": "timesfm_k50_bytedance.json",
}
TTM_FILES = {
    "Alibaba":   "ttm_k20_alibaba.json",
    "Bitbrains": "ttm_k50_bitbrains.json",
    "ByteDance": "ttm_k50_bytedance.json",
}

# JSON top-level key per model, holding the per-horizon block.
CHRONOS_KEY = "chronos2_zero_shot"
TIMESFM_KEY = "timesfm_zero_shot"
# TTM key — we accept any of these to be tolerant of naming choices.
TTM_KEY_CANDIDATES = ["ttm_zero_shot", "granite_ttm_zero_shot",
                      "granite_tsfm_zero_shot", "tsfm_zero_shot"]

# Bootstrap and permutation knobs. Match the existing JSON exactly.
N_BOOT = 10000
N_PERM = 10000
ALPHA  = 0.05
RNG_SEED = 42  # deterministic CI/p-value


# ── HELPERS ─────────────────────────────────────────────────────────────

def load_json(path):
    """Read a JSON file, return None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    """Read a CSV into a list of dicts. Empty list if missing."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def median(xs):
    """Plain median, no numpy. Identical to numpy.median for our small lists."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def safe_float(x):
    """Cast a CSV string to float, treating empty strings as NaN."""
    if x is None or x == "":
        return float("nan")
    return float(x)


def find_ttm_block(payload):
    """TTM JSON might use any of several top-level keys. Try them in order."""
    if not isinstance(payload, dict):
        return None
    for k in TTM_KEY_CANDIDATES:
        if k in payload:
            return payload[k]
    return None


# ── AUC (Mann-Whitney form) ─────────────────────────────────────────────

def auc_score(predicate_vec, label_vec):
    """
    AUC of a binary "score" (predicate, 0/1) at predicting a binary label
    (ml_wins, 0/1). Implemented as the normalized Mann-Whitney U.

    For each (positive_label, negative_label) pair, count:
        +1 if score_pos >  score_neg
        +0.5 if score_pos == score_neg
        +0 otherwise
    Divide by the number of pairs.

    Returns float in [0, 1], or NaN if either class is empty.
    """
    pos = [p for p, y in zip(predicate_vec, label_vec) if y == 1]
    neg = [p for p, y in zip(predicate_vec, label_vec) if y == 0]
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                s += 1.0
            elif a == b:
                s += 0.5
    return s / (len(pos) * len(neg))


def predicate_label_match(predicate_vec, label_vec):
    """Plain accuracy: how often does predicate equal the label?"""
    if not predicate_vec:
        return float("nan")
    n = len(predicate_vec)
    matches = sum(1 for p, y in zip(predicate_vec, label_vec) if p == y)
    return matches / n


# ── BCa Bootstrap CI ────────────────────────────────────────────────────

def bca_ci(predicate_vec, label_vec, n_boot=N_BOOT, alpha=ALPHA, seed=RNG_SEED):
    """
    Bias-corrected and accelerated bootstrap CI for AUC.

    Returns (ci_lo, ci_hi, z0, a_hat, observed_auc).

    Standard Efron BCa:
      z0     = Phi^-1( fraction of bootstrapped AUCs < observed )
      a_hat  = sum(d^3) / (6 * sum(d^2)^1.5),  d = mean(jackknife) - jackknife_i
      alpha1 = Phi(z0 + (z0 + z_a/2) / (1 - a*(z0 + z_a/2)))
      alpha2 = Phi(z0 + (z0 + z_1-a/2) / (1 - a*(z0 + z_1-a/2)))
      CI     = [percentile(boot, alpha1), percentile(boot, alpha2)]

    Caveat already noted in bcf_pooled_results.json: with n_pairs=36 and
    only ~200 unique AUC values in the bootstrap distribution, BCa is
    precise to about 2 decimals, no more.
    """
    rng = random.Random(seed)
    n = len(predicate_vec)
    obs = auc_score(predicate_vec, label_vec)

    # bootstrap resamples (with replacement, indices)
    boot = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        p_sub = [predicate_vec[i] for i in idx]
        y_sub = [label_vec[i]     for i in idx]
        a = auc_score(p_sub, y_sub)
        if not math.isnan(a):
            boot.append(a)
    boot.sort()

    # z0: bias correction
    n_below = sum(1 for b in boot if b < obs)
    if len(boot) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), obs
    p0 = n_below / len(boot)
    # clip to avoid +/-inf when all bootstraps are above/below obs
    p0 = min(max(p0, 1e-6), 1 - 1e-6)
    z0 = inv_norm_cdf(p0)

    # a_hat: jackknife acceleration
    jk = []
    for i in range(n):
        p_sub = predicate_vec[:i] + predicate_vec[i+1:]
        y_sub = label_vec[:i]     + label_vec[i+1:]
        a = auc_score(p_sub, y_sub)
        if not math.isnan(a):
            jk.append(a)
    if not jk:
        a_hat = 0.0
    else:
        m = sum(jk) / len(jk)
        diffs = [m - x for x in jk]
        num = sum(d**3 for d in diffs)
        den = 6.0 * (sum(d**2 for d in diffs) ** 1.5)
        a_hat = num / den if den > 0 else 0.0

    # alpha-corrected percentiles
    z_lo = inv_norm_cdf(alpha / 2)
    z_hi = inv_norm_cdf(1 - alpha / 2)
    a1 = norm_cdf(z0 + (z0 + z_lo) / (1 - a_hat * (z0 + z_lo)))
    a2 = norm_cdf(z0 + (z0 + z_hi) / (1 - a_hat * (z0 + z_hi)))

    ci_lo = percentile_sorted(boot, a1)
    ci_hi = percentile_sorted(boot, a2)
    return ci_lo, ci_hi, z0, a_hat, obs


def percentile_sorted(sorted_xs, q):
    """q-th percentile (q in [0,1]) of an already-sorted list."""
    if not sorted_xs:
        return float("nan")
    if q <= 0:
        return sorted_xs[0]
    if q >= 1:
        return sorted_xs[-1]
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


# Standard normal CDF and inverse via erf (no scipy dependency).
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def inv_norm_cdf(p):
    """
    Inverse standard normal CDF (probit). Beasley-Springer-Moro
    rational approximation. Accurate to ~1e-9 in the central region,
    which is more than enough for BCa percentile lookup.
    """
    # boundary handling
    if p <= 0:
        return -float("inf")
    if p >= 1:
        return float("inf")

    # constants
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    p_low  = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


# ── Permutation test ────────────────────────────────────────────────────

def permutation_p(predicate_vec, label_vec, n_perm=N_PERM, seed=RNG_SEED):
    """
    One-sided permutation test for AUC > 0.5.

    Phipson-Smyth correction:
        p = (count_ge + 1) / (n_perm + 1)
    where count_ge = number of permutations with AUC >= observed.
    The +1 prevents p=0, which is never warranted in a finite simulation.
    """
    rng = random.Random(seed + 1)  # different stream from BCa
    obs = auc_score(predicate_vec, label_vec)
    if math.isnan(obs):
        return float("nan")

    label_copy = list(label_vec)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(label_copy)
        a = auc_score(predicate_vec, label_copy)
        if not math.isnan(a) and a >= obs:
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)


# ── Build per-model pairs ───────────────────────────────────────────────

def build_naive_pairs():
    """
    Naive baseline: r2_model and r2_naive are blank (it IS the naive),
    delta_pp = 0, ml_wins = 0 by construction. Predicate still computed.
    Order: dataset-major, matching existing CSV.
    """
    rows = []
    for dset in DATASETS:
        for hz in HORIZONS:
            rows.append({
                "model": "Naive",
                "dataset": dset,
                "horizon": hz,
                "r2_model": "",
                "r2_naive": "",
                "delta_pp": 0.0,
                "predicate": predicate(dset, hz),
                "ml_wins": 0,
            })
    return rows


def build_nnls_pairs(data_dir):
    """
    NNLS rows. Source files:
      - chronos_benchmark_results.csv for Alibaba and ByteDance
        (full-test pooled R^2; ensemble_r2 = NNLS, naive_r2 = naive)
      - bitbrains_per_vm.csv for Bitbrains, aggregated as per-VM medians
        (NNLS_BITBRAINS_AGG = 'per_vm_median' in the original script).

    Order: horizon-major (10min for all 3 datasets, then 30min, etc.)
    Matches the existing bcf_pairs.csv exactly.
    """
    chronos_csv = load_csv(os.path.join(data_dir, "chronos_benchmark_results.csv"))
    bb_csv      = load_csv(os.path.join(data_dir, "bitbrains_per_vm.csv"))

    # index chronos benchmark CSV by (dataset, horizon)
    chronos_idx = {}
    for r in chronos_csv:
        chronos_idx[(r["dataset"], r["horizon"])] = r

    # group bitbrains_per_vm.csv by horizon for aggregation
    bb_by_hz = {hz: [] for hz in HORIZONS}
    for r in bb_csv:
        h = r.get("horizon")
        if h in bb_by_hz:
            bb_by_hz[h].append(r)

    rows = []
    for hz in HORIZONS:
        for dset in DATASETS:
            if dset == "Bitbrains":
                # per-VM median aggregation (canonical)
                sub = bb_by_hz[hz]
                if not sub:
                    print(f"  WARN: no bitbrains_per_vm rows for hz={hz}")
                    continue
                med_naive = median([float(r["naive_r2"]) for r in sub])
                med_ml    = median([float(r["ml_r2"])    for r in sub])
                med_delta = median([float(r["r2_delta"]) for r in sub])  # in fraction
                delta_pp  = med_delta * 100.0
                ml_wins   = int(med_delta > 0)
                r2_model  = med_ml
                r2_naive  = med_naive
            else:
                # full-test pooled R^2 from chronos_benchmark_results.csv
                row = chronos_idx.get((dset, hz))
                if row is None:
                    print(f"  WARN: no chronos_benchmark row for {dset}/{hz}")
                    continue
                r2_model = float(row["ensemble_r2"])
                r2_naive = float(row["naive_r2"])
                delta_pp = (r2_model - r2_naive) * 100.0
                ml_wins  = int(delta_pp > 0)

            rows.append({
                "model": "NNLS",
                "dataset": dset,
                "horizon": hz,
                "r2_model": r2_model,
                "r2_naive": r2_naive,
                "delta_pp": delta_pp,
                "predicate": predicate(dset, hz),
                "ml_wins": ml_wins,
            })
    return rows


def build_foundation_pairs(model_name, file_map, json_key, data_dir,
                           is_optional=False):
    """
    Generic builder for foundation model rows (Chronos-2, TimesFM, TTM).
    file_map[dataset] -> JSON filename in data_dir.
    json_key may be a string or a list of candidate keys (TTM case).

    Returns ([] , True) and a 'skipped' flag if the JSONs are missing AND
    is_optional=True. If some datasets were loaded successfully before
    a missing one is encountered, a WARN is printed listing what was
    discarded — partial coverage is never returned, but it is logged.
    Otherwise, missing JSONs are a fatal error.

    Order: dataset-major.
    """
    rows = []
    for dset in DATASETS:
        path = os.path.join(data_dir, file_map[dset])
        payload = load_json(path)
        if payload is None:
            if is_optional:
                # graceful skip. If we already loaded data for earlier
                # datasets, log the discard so the user knows the input
                # state was partial.
                if rows:
                    loaded_dsets = sorted(set(r["dataset"] for r in rows))
                    print(f"  WARN: {model_name}: discarding "
                          f"{len(rows)} rows already loaded for "
                          f"{loaded_dsets} because {file_map[dset]} "
                          f"is missing — pooled AUC needs all "
                          f"{len(DATASETS)} datasets")
                return [], True
            else:
                raise FileNotFoundError(
                    f"{model_name}: required file missing: {path}")

        # extract the per-horizon block
        if isinstance(json_key, list):
            block = None
            for k in json_key:
                if k in payload:
                    block = payload[k]
                    break
            if block is None:
                # also try the find_ttm_block helper as a last resort
                block = find_ttm_block(payload)
            if block is None:
                raise KeyError(
                    f"{model_name}: no recognized top-level key in {path}. "
                    f"Tried: {json_key}")
        else:
            block = payload.get(json_key)
            if block is None:
                raise KeyError(
                    f"{model_name}: missing key '{json_key}' in {path}")

        for hz in HORIZONS:
            hb = block.get(hz)
            if hb is None:
                print(f"  WARN: {model_name}/{dset}/{hz} not in JSON, skipping")
                continue
            r2_model = hb.get("r2_mean")
            r2_naive = hb.get("r2_naive_subsample")
            if r2_model is None or r2_naive is None:
                print(f"  WARN: {model_name}/{dset}/{hz} missing r2 fields")
                continue
            delta_pp = (r2_model - r2_naive) * 100.0
            ml_wins  = int(delta_pp > 0)

            rows.append({
                "model": model_name,
                "dataset": dset,
                "horizon": hz,
                "r2_model": r2_model,
                "r2_naive": r2_naive,
                "delta_pp": delta_pp,
                "predicate": predicate(dset, hz),
                "ml_wins": ml_wins,
            })
    return rows, False


def assert_symmetric_coverage(model_name, rows):
    """
    Each ML model must contribute exactly len(DATASETS) * len(HORIZONS)
    cells (12) to be eligible for pooled AUC. Partial coverage means a
    model would contribute fewer pairs to the pool than its peers, which
    biases pooled statistics in whichever direction the missing cells
    happen to point. The pooled JSON reader has no way to tell from
    n_pooled_pairs alone whether the asymmetry exists.

    This applies to NNLS, Chronos-2, TimesFM, and (when present) TTM —
    not just TTM. The original boundary_condition_stats.py implicitly
    assumed full coverage by virtue of NNLS reading from a hand-curated
    chronos_benchmark_results.csv that was always 12 rows. We make
    that assumption explicit.

    Raises ValueError with a coverage report if the row count is wrong.
    """
    expected = len(DATASETS) * len(HORIZONS)
    if len(rows) == expected:
        return
    seen = {(r["dataset"], r["horizon"]) for r in rows}
    missing = []
    for d in DATASETS:
        for h in HORIZONS:
            if (d, h) not in seen:
                missing.append(f"{d}/{h}")
    raise ValueError(
        f"{model_name} has {len(rows)} rows, need {expected}. "
        f"Missing cells: {missing}. Pooled AUC requires symmetric "
        f"coverage across all {expected} (dataset, horizon) cells. "
        f"Refusing to score a partial model. Either complete the run "
        f"or remove the {model_name} input(s) to skip it entirely.")


# ── Output writers ──────────────────────────────────────────────────────

def write_pairs_csv(path, all_rows):
    """Write bcf_pairs.csv."""
    fields = ["model", "dataset", "horizon",
              "r2_model", "r2_naive", "delta_pp",
              "predicate", "ml_wins"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            # leave blank values blank, not NaN
            out = {}
            for k in fields:
                v = r.get(k, "")
                if isinstance(v, float) and math.isnan(v):
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def write_per_model_csv(path, per_model_rows):
    """Write bcf_per_model_auc.csv."""
    fields = ["model", "n_pairs", "n_ml_wins", "n_predicate_pos",
              "auc", "predicate_label_match"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_model_rows:
            out = dict(r)
            for k in ("auc", "predicate_label_match"):
                if isinstance(out.get(k), float) and math.isnan(out[k]):
                    out[k] = ""
            w.writerow(out)


# ── MAIN ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BCF AUC across 4 or 5 models. Adds Granite-TTM as a "
                    "5th row when the TTM JSONs are present.")
    parser.add_argument("--data-dir", default=".",
                        help="Directory containing input CSVs and JSONs")
    parser.add_argument("--output-dir", default=".",
                        help="Where to write bcf_pairs.csv etc.")
    parser.add_argument("--n-boot", type=int, default=N_BOOT,
                        help=f"Bootstrap iterations (default {N_BOOT})")
    parser.add_argument("--n-perm", type=int, default=N_PERM,
                        help=f"Permutations (default {N_PERM})")
    parser.add_argument("--seed", type=int, default=RNG_SEED,
                        help=f"RNG seed (default {RNG_SEED})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("bcf_5model.py — Boundary Condition Framework AUC, 4 or 5 models")
    print("=" * 70)
    print(f"data_dir:   {args.data_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"predicate:  (ACF@24h > 0.2) AND (horizon >= 30min)")
    print(f"datasets:   {DATASETS}")
    print(f"horizons:   {HORIZONS}")
    print(f"ACF@24h:    {ACF_24H}")
    print()

    # ── sanity check ACF@24h against omega_summary.csv if present ──────
    omega = load_csv(os.path.join(args.data_dir, "omega_summary.csv"))
    if omega:
        for r in omega:
            ds = r.get("Dataset", "").strip()
            # 'ByteDance IaaS' in omega_summary -> 'ByteDance' in our keys
            ds_key = ds.split()[0] if ds else ""
            if ds_key in ACF_24H:
                csv_acf = safe_float(r.get("ACF@24h"))
                hard_acf = ACF_24H[ds_key]
                if abs(csv_acf - hard_acf) > 0.005:
                    print(f"  WARN: ACF@24h mismatch for {ds_key}: "
                          f"hardcoded={hard_acf}, csv={csv_acf}")
                else:
                    print(f"  OK ACF@24h({ds_key:9s}) = {csv_acf:.3f}")
        print()

    # ── 1. build per-model pairs ───────────────────────────────────────
    print("-" * 70)
    print("1. Building pairs for each model")
    print("-" * 70)

    naive_rows = build_naive_pairs()
    print(f"  Naive:     {len(naive_rows)} rows (constant: ml_wins=0)")
    assert_symmetric_coverage("Naive", naive_rows)

    nnls_rows = build_nnls_pairs(args.data_dir)
    print(f"  NNLS:      {len(nnls_rows)} rows")
    print(f"             (Bitbrains uses per-VM median aggregation)")
    assert_symmetric_coverage("NNLS", nnls_rows)

    chronos_rows, _ = build_foundation_pairs(
        "Chronos-2", CHRONOS_FILES, CHRONOS_KEY,
        args.data_dir, is_optional=False)
    print(f"  Chronos-2: {len(chronos_rows)} rows")
    assert_symmetric_coverage("Chronos-2", chronos_rows)

    timesfm_rows, _ = build_foundation_pairs(
        "TimesFM", TIMESFM_FILES, TIMESFM_KEY,
        args.data_dir, is_optional=False)
    print(f"  TimesFM:   {len(timesfm_rows)} rows")
    assert_symmetric_coverage("TimesFM", timesfm_rows)

    ttm_rows, ttm_skipped = build_foundation_pairs(
        "Granite-TTM", TTM_FILES, TTM_KEY_CANDIDATES,
        args.data_dir, is_optional=True)
    if ttm_skipped:
        print(f"  Granite-TTM: SKIPPED (JSONs not on disk)")
        print(f"             expected files: {list(TTM_FILES.values())}")
    else:
        print(f"  Granite-TTM: {len(ttm_rows)} rows")
        # Only apply the symmetric-coverage check if TTM was actually
        # scored. If is_optional skipped it cleanly, ttm_rows is [] by
        # design and that is not the same as a partial model.
        assert_symmetric_coverage("Granite-TTM", ttm_rows)
    print()

    all_pairs = naive_rows + nnls_rows + chronos_rows + timesfm_rows + ttm_rows

    # ── 2. write bcf_pairs.csv ─────────────────────────────────────────
    pairs_path = os.path.join(args.output_dir, "bcf_pairs.csv")
    write_pairs_csv(pairs_path, all_pairs)
    print(f"  Wrote {pairs_path} ({len(all_pairs)} rows)")
    print()

    # ── 3. per-model AUC ────────────────────────────────────────────────
    print("-" * 70)
    print("2. Per-model AUC (12 cells each)")
    print("-" * 70)

    per_model = []
    model_blocks = [
        ("Naive",       naive_rows),
        ("NNLS",        nnls_rows),
        ("Chronos-2",   chronos_rows),
        ("TimesFM",     timesfm_rows),
    ]
    if not ttm_skipped:
        model_blocks.append(("Granite-TTM", ttm_rows))

    for name, rows in model_blocks:
        pred = [r["predicate"] for r in rows]
        wins = [r["ml_wins"]   for r in rows]
        n = len(rows)
        n_wins = sum(wins)
        n_pred = sum(pred)
        if name == "Naive":
            # Naive is excluded from AUC by construction (ml_wins always 0)
            auc = float("nan")
        else:
            auc = auc_score(pred, wins)
        plm = predicate_label_match(pred, wins)

        per_model.append({
            "model": name,
            "n_pairs": n,
            "n_ml_wins": n_wins,
            "n_predicate_pos": n_pred,
            "auc": auc,
            "predicate_label_match": plm,
        })
        auc_str = f"{auc:.4f}" if not math.isnan(auc) else "n/a"
        print(f"  {name:13s}  n={n:2d}  wins={n_wins:2d}  "
              f"pred_pos={n_pred:2d}  AUC={auc_str}  "
              f"label_match={plm:.4f}")

    per_model_path = os.path.join(args.output_dir, "bcf_per_model_auc.csv")
    write_per_model_csv(per_model_path, per_model)
    print(f"\n  Wrote {per_model_path}")
    print()

    # ── 4. pooled AUC across all ML models (excluding Naive) ───────────
    print("-" * 70)
    print("3. Pooled AUC (ML models only)")
    print("-" * 70)

    pooled = nnls_rows + chronos_rows + timesfm_rows
    if not ttm_skipped:
        pooled = pooled + ttm_rows

    pooled_pred = [r["predicate"] for r in pooled]
    pooled_wins = [r["ml_wins"]   for r in pooled]

    pooled_auc = auc_score(pooled_pred, pooled_wins)
    n_pool     = len(pooled)
    n_wins     = sum(pooled_wins)
    n_pred     = sum(pooled_pred)
    print(f"  n_pooled_pairs:   {n_pool}")
    print(f"  n_ml_wins:        {n_wins}")
    print(f"  n_predicate_pos:  {n_pred}")
    print(f"  pooled_auc:       {pooled_auc:.4f}")
    print()

    # ── 5. BCa CI ──────────────────────────────────────────────────────
    print(f"  Running BCa bootstrap (n_boot={args.n_boot}, seed={args.seed})...")
    ci_lo, ci_hi, z0, a_hat, _ = bca_ci(
        pooled_pred, pooled_wins,
        n_boot=args.n_boot, alpha=ALPHA, seed=args.seed)
    print(f"  bca_95_ci_lo:     {ci_lo:.4f}")
    print(f"  bca_95_ci_hi:     {ci_hi:.4f}")
    print(f"  bca_z0:           {z0:.6f}")
    print(f"  bca_a_hat:        {a_hat:.6f}")
    print()

    # ── 6. Permutation p ───────────────────────────────────────────────
    print(f"  Running permutation test (n_perm={args.n_perm}, seed={args.seed})...")
    p_perm = permutation_p(
        pooled_pred, pooled_wins,
        n_perm=args.n_perm, seed=args.seed)
    print(f"  permutation_p:    {p_perm:.6f}")
    print(f"  (Phipson-Smyth (count_ge + 1) / (n_perm + 1))")
    print()

    # ── 7. write pooled JSON ───────────────────────────────────────────
    excluded = ["Naive (all ml_wins=0 by construction)"]
    if ttm_skipped:
        excluded.append("Granite-TTM (JSONs not on disk)")

    # provenance — record the resolved paths and mtimes of every input
    # that contributed numbers to this pooled result. The thesis grader
    # can reproduce by checking these mtimes against checkpoint logs.
    def src_entry(rel_path):
        full = os.path.join(args.data_dir, rel_path)
        if os.path.exists(full):
            return {
                "path":  os.path.abspath(full),
                "mtime": os.path.getmtime(full),
                "bytes": os.path.getsize(full),
            }
        return {"path": os.path.abspath(full), "mtime": None, "bytes": None}

    sources = OrderedDict()
    sources["nnls_alibaba_bytedance"] = src_entry("chronos_benchmark_results.csv")
    sources["nnls_bitbrains"]         = src_entry("bitbrains_per_vm.csv")
    for dset, fn in CHRONOS_FILES.items():
        sources[f"chronos2_{dset.lower()}"] = src_entry(fn)
    for dset, fn in TIMESFM_FILES.items():
        sources[f"timesfm_{dset.lower()}"]  = src_entry(fn)
    if not ttm_skipped:
        for dset, fn in TTM_FILES.items():
            sources[f"ttm_{dset.lower()}"]  = src_entry(fn)

    caveats = [
        ("NNLS and foundation models use different naive references "
         "(NNLS: full-test pooled naive from chronos_benchmark_results.csv; "
         "foundation models: k20/k50 same-subsample naive from sanity JSONs). "
         "Each delta is internally consistent (model vs naive on identical "
         "points) but reference sets differ across models."),
        ("NNLS Bitbrains uses median of per-VM deltas "
         "(NNLS_BITBRAINS_AGG='per_vm_median') to match "
         "boundary_condition_table_corrected.csv. Pooled scoring flips "
         "Bitbrains-120min from win to loss and shifts pooled AUC by ~0.03."),
        (f"{n_pool} pooled pairs = 12 (dataset, horizon) cells under "
         f"{n_pool // 12} models. Pairs are not iid; CI and p-value are "
         f"pair-level and overstate precision relative to cluster-aware "
         f"inference at n=12."),
        ("BCa CI precision is illusory beyond 2 decimals; ~200 unique AUC "
         "values in the bootstrap distribution. Report CI rounded."),
    ]

    pooled_json = OrderedDict([
        ("predicate",         "(ACF@24h > 0.2) AND (horizon >= 30min)"),
        ("nnls_bitbrains_agg", "per_vm_median"),
        ("acf_24h_used",       ACF_24H),
        ("n_pooled_pairs",     n_pool),
        ("n_ml_wins",          n_wins),
        ("n_predicate_pos",    n_pred),
        ("pooled_auc",         pooled_auc),
        ("bca_95_ci_lo",       ci_lo),
        ("bca_95_ci_hi",       ci_hi),
        ("bca_z0",             z0),
        ("bca_a_hat",          a_hat),
        ("permutation_p",      p_perm),
        ("permutation_method", "Phipson-Smyth (count_ge + 1) / (n_perm + 1)"),
        ("n_boot",             args.n_boot),
        ("n_perm",             args.n_perm),
        ("excluded",           excluded),
        ("sources",            sources),
        ("caveats",            caveats),
    ])

    pooled_path = os.path.join(args.output_dir, "bcf_pooled_results.json")
    with open(pooled_path, "w") as f:
        json.dump(pooled_json, f, indent=2)
    print(f"  Wrote {pooled_path}")
    print()

    # ── done ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  models scored:    {len(per_model)} "
          f"({'with' if not ttm_skipped else 'without'} Granite-TTM)")
    print(f"  pooled AUC:       {pooled_auc:.4f}  "
          f"(BCa 95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")
    print(f"  permutation p:    {p_perm:.4f}")
    print()
    if ttm_skipped:
        print("  To add Granite-TTM as a 5th row, place these JSONs in "
              "data-dir:")
        for dset, fn in TTM_FILES.items():
            print(f"    {fn}  ({dset})")
        print("  with a top-level key named ttm_zero_shot, "
              "granite_ttm_zero_shot,")
        print("  granite_tsfm_zero_shot, or tsfm_zero_shot, holding the same")
        print("  per-horizon schema as timesfm_*.json (r2_mean, "
              "r2_naive_subsample).")


if __name__ == "__main__":
    main()
