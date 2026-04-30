"""
bitbrains_xval.py — Bitbrains cross-dataset validation pipeline.

Trains XGBoost per-VM across 156 active VMs at four horizons to
test whether the Alibaba-trained ensemble generalises to a
different infrastructure environment. Bitbrains has high short-lag
autocorrelation but no diurnal cycle (ACF@24h = 0.116), providing
a natural test of the boundary condition framework.

Stages (run sequentially or individually):
    1  scan_vms    — profile all 1250 VMs, filter to active subset
    2  train       — per-VM XGBoost with checkpoint resume
    3  summarise   — aggregate per-VM results, comparison table
    4  sanity_fig  — quick 3-panel diagnostic (PNG)
    5  cv_strat    — CV-stratified win-rate analysis (Wilcoxon + Cliff's δ)
    6  pub_fig     — publication-quality 3-panel figure (PDF)

Inputs:
    Bitbrains fastStorage CSVs                 (--raw-dir)

Outputs:
    bitbrains_summary.csv, bitbrains_per_vm_results.csv,
    bitbrains_cv_stratified.csv, active_vms.csv,
    bitbrains_cross_validation.pdf             (--output-dir)

Bugs fixed vs original:
    1. Timestamp unit: Bitbrains uses Unix seconds, not ms — auto-detected
       via _parse_timestamps() median heuristic (>1e11 → ms, else → s)
    2. dayofweek .values: pd.to_datetime returns DatetimeIndex, so .dayofweek
       is already ndarray — removed spurious .values call
    3. vm_id float suffix: pandas reads numeric filenames as float64 (1005.0),
       breaking file path construction — normalised to int-string before open
    4. cv_strat merge: same float/int mismatch on both sides of merge — both
       normalised to int-string via int(float(x)) before joining
"""

import argparse
import json
import os
import sys
import traceback
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Alibaba reference values (sprint1_v9 confirmed results) ──────────────────
ALI_DELTA_HETERO = {"10min": +0.25, "30min": +0.43, "60min": +1.33, "120min": +4.64}
ALI_XGB_DELTA    = [+0.15, +0.14, +0.39, +0.96]
ALI_NAIVE        = [0.9188, 0.8361, 0.7878, 0.7178]
ALI_SKILL        = {"10min": -0.1194, "30min": -0.0194, "60min": +0.0330, "120min": +0.1111}

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = "/content/drive/MyDrive/Thesis/bitbrains/fastStorage/2013-8"
DEFAULT_OUTPUT_DIR = "/content/drive/MyDrive/Thesis/bitbrains/results"

HORIZONS = {"10min": 2, "30min": 6, "60min": 12, "120min": 24}

MEAN_THRESH = 15.0
STD_THRESH  =  3.0

BLUE   = "#2166ac"
ORANGE = "#d6604d"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_vm_id(x):
    """Normalise any vm_id representation to plain int-string, e.g. '1005.0' → '1005'."""
    try:
        return str(int(float(x)))
    except (ValueError, OverflowError):
        return str(x)


def _parse_timestamps(raw_values):
    """
    Auto-detect whether timestamps are Unix-ms or Unix-s and return a
    DatetimeIndex in UTC.

    Bitbrains GWA-T-12 uses Unix seconds (~1.37e9 for 2013).
    Passing seconds as unit='ms' places all rows in Jan 1970 making
    hour_sin/cos constant and destroying temporal features.

    Heuristic: median > 1e11  →  milliseconds (year 2001+ in ms = 9.8e11)
               median ≤ 1e11  →  seconds
    """
    raw = pd.Series(raw_values).dropna()
    if len(raw) == 0:
        return None
    unit = "ms" if float(raw.median()) > 1e11 else "s"
    try:
        ts = pd.to_datetime(raw_values, unit=unit, utc=True, errors="coerce")
        if ts.isna().mean() > 0.1:
            return None
        return ts
    except Exception:
        return None


def _is_notebook():
    try:
        ip  = get_ipython()
        cls = ip.__class__
        if cls.__name__ == "ZMQInteractiveShell":
            return True
        if "google.colab" in cls.__module__:
            return True
        return False
    except NameError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — VM scan
# ═══════════════════════════════════════════════════════════════════════════════

def scan_vms(data_dir: str, output_dir: str) -> pd.DataFrame:
    try:
        from hurst import compute_Hc
        hurst_available = True
    except ImportError:
        print("WARNING: hurst not installed — Hurst will be NaN. pip install hurst")
        hurst_available = False

    all_csv = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    print(f"found {len(all_csv)} CSV files in {data_dir}")
    print("scanning — approx 3 min …")

    vm_stats = []
    for i, fname in enumerate(all_csv):
        fpath = os.path.join(data_dir, fname)
        try:
            df      = pd.read_csv(fpath, sep=";", header=0)
            cpu_col = [c for c in df.columns if "CPU usage [%]" in c]
            if not cpu_col:
                continue
            cpu = df[cpu_col[0]].dropna()

            H = float("nan")
            if hurst_available:
                try:
                    H, _, _ = compute_Hc(cpu.values, kind="price", simplified=True)
                except Exception:
                    pass

            cv   = float(cpu.std() / cpu.mean()) if cpu.mean() > 0 else float("nan")
            acf1 = float(pd.Series(cpu.values).autocorr(lag=1))

            vm_stats.append({
                "vm_id":  fname.replace(".csv", ""),
                "n_rows": len(cpu),
                "mean":   float(cpu.mean()),
                "std":    float(cpu.std()),
                "max":    float(cpu.max()),
                "cv":     round(cv,   4),
                "hurst":  round(float(H), 4) if not np.isnan(H) else float("nan"),
                "acf1":   round(acf1, 4),
            })
        except Exception:
            pass

        if (i + 1) % 250 == 0:
            print(f"  {i + 1} / {len(all_csv)}")

    stats_df = pd.DataFrame(vm_stats)
    print(f"\nscanned {len(stats_df)} VMs total")
    print(f"overall mean CPU : {stats_df['mean'].mean():.1f}%")
    print(f"overall median CPU: {stats_df['mean'].median():.1f}%\n")

    for t in [5, 10, 15, 20]:
        pct = (stats_df["mean"] < t).mean() * 100
        print(f"  VMs with mean < {t}%: {pct:.1f}%")

    active_vms = stats_df[
        (stats_df["mean"] > MEAN_THRESH) &
        (stats_df["std"]  > STD_THRESH)
    ].copy().reset_index(drop=True)

    n_total  = len(stats_df)
    n_active = len(active_vms)
    print(f"\nactive VMs (mean>{MEAN_THRESH}%, std>{STD_THRESH}%): "
          f"{n_active} / {n_total}  ({n_active / n_total * 100:.1f}%)")

    hurst_vals = active_vms["hurst"].dropna()
    print(f"\nworkload characterisation (active VMs, n={n_active}):")
    print(f"  CV    median={active_vms['cv'].median():.3f}  mean={active_vms['cv'].mean():.3f}")
    print(f"  Hurst median={hurst_vals.median():.3f}  mean={hurst_vals.mean():.3f}"
          + ("  (NaN = hurst not installed)" if len(hurst_vals) == 0 else ""))
    print(f"  ACF1  median={active_vms['acf1'].median():.3f}  mean={active_vms['acf1'].mean():.3f}")

    out_path = os.path.join(output_dir, "active_vms.csv")
    active_vms.to_csv(out_path, index=False)
    print(f"\nactive_vms saved → {out_path}")
    return active_vms


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — feature engineering + training
# ═══════════════════════════════════════════════════════════════════════════════

def make_features(series_vals, horizon_steps: int, timestamps=None) -> tuple:
    """
    Build supervised learning table from a 1-D CPU series.
    timestamps : raw array of timestamps in any unit (auto-detected).
    """
    df = pd.DataFrame({"y": series_vals})

    # lag features
    for lag in range(1, 13):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # rolling stats
    for w in [3, 6, 12]:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["y"].shift(1).rolling(w).std().fillna(0)
        df[f"roll_max_{w}"]  = df["y"].shift(1).rolling(w).max()

    # trend / momentum
    df["trend_short"]  = df["y"].shift(1) - df["y"].shift(2)
    df["trend_medium"] = df["y"].shift(1) - df["y"].shift(6)
    df["trend_long"]   = df["y"].shift(1) - df["y"].shift(12)
    df["momentum"]     = df["roll_mean_3"] - df["roll_mean_12"]

    # same-time-yesterday
    std_lag = max(1, 288 - horizon_steps)
    df["same_time_1d"] = df["y"].shift(std_lag).fillna(df["y"].shift(1))
    df["diff_1d"]      = (df["y"].shift(1) - df["same_time_1d"]).fillna(0)

    # temporal features — FIX: use _parse_timestamps (auto-detects ms vs s)
    if timestamps is not None:
        ts = _parse_timestamps(timestamps[:len(series_vals)])
        if ts is not None:
            # FIX: ts.hour / ts.dayofweek return numpy arrays on DatetimeIndex
            # — no .values call needed or valid
            hour = np.array(ts.hour) + np.array(ts.minute) / 60.0
            dow  = np.array(ts.dayofweek, dtype=float)
            df["hour_sin"]       = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"]       = np.cos(2 * np.pi * hour / 24)
            df["dow_sin"]        = np.sin(2 * np.pi * dow  / 7)
            df["dow_cos"]        = np.cos(2 * np.pi * dow  / 7)
            df["business_hours"] = ((hour >= 8) & (hour < 18) & (dow < 5)).astype(float)

    df["y_target"] = df["y"].shift(-horizon_steps)
    df["y_naive"]  = df["y"]
    df = df.dropna(subset=["y_target", "lag_1", "lag_12"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("y", "y_target", "y_naive")]
    return df[feature_cols], df["y_target"], df["y_naive"]


def train_and_eval(series_vals, horizon_steps: int, timestamps=None):
    X, y_target, y_naive = make_features(series_vals, horizon_steps, timestamps)

    if len(X) < 80:
        return None

    split      = int(len(X) * 0.8)
    X_train, X_test     = X.iloc[:split], X.iloc[split:]
    y_train, y_test     = y_target.iloc[:split], y_target.iloc[split:]
    naive_test          = y_naive.iloc[split:]

    naive_r2  = r2_score(y_test, naive_test)
    naive_mae = mean_absolute_error(y_test, naive_test)

    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ml_r2_raw     = r2_score(y_test, y_pred)
    ml_mae        = mean_absolute_error(y_test, y_pred)
    ml_r2_clipped = max(ml_r2_raw, -10.0)
    skill         = float(1.0 - ml_mae / (naive_mae + 1e-9))

    return {
        "naive_r2"  : round(naive_r2,      6),
        "naive_mae" : round(naive_mae,     6),
        "ml_r2"     : round(ml_r2_clipped, 6),
        "ml_r2_raw" : round(ml_r2_raw,     6),
        "ml_mae"    : round(ml_mae,        6),
        "r2_delta"  : round(ml_r2_clipped - naive_r2, 6),
        "skill"     : round(skill,         6),
        "ml_wins"   : int(ml_r2_raw > naive_r2),
    }


def train_all(active_vms, data_dir, output_dir, reset_checkpoint=False):
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    per_vm_csv      = os.path.join(output_dir, "bitbrains_per_vm.csv")

    if reset_checkpoint and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("old checkpoint deleted (--reset_checkpoint)")

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        results     = ckpt["results"]
        done_vm_ids = set(ckpt["done_vm_ids"])
        print(f"resuming from checkpoint  ({len(done_vm_ids)} VMs done, {len(results)} results)")
    else:
        results     = []
        done_vm_ids = set()
        print("starting fresh (no checkpoint found)")

    total     = len(active_vms)
    errors    = 0
    error_log = []
    print(f"\ntraining on {total} active VMs × 4 horizons")
    print("checkpoint saved every 10 VMs")
    print("=" * 52)

    for i, row in active_vms.iterrows():
        vm_id = str(row["vm_id"])
        if vm_id in done_vm_ids:
            continue

        # FIX: normalise float vm_id (e.g. '1005.0') to int-string ('1005')
        vm_id_file = _norm_vm_id(vm_id)
        fpath = os.path.join(data_dir, f"{vm_id_file}.csv")

        try:
            df      = pd.read_csv(fpath, sep=";", header=0)
            cpu_col = [c for c in df.columns if "CPU usage [%]" in c][0]
            ts_col  = [c for c in df.columns if "Timestamp" in c]

            cpu_raw = df[cpu_col].dropna()
            series  = (cpu_raw / 100.0).values

            ts_raw  = df.loc[cpu_raw.index, ts_col[0]].values if ts_col else None

            if len(series) < 150:
                done_vm_ids.add(vm_id)
                continue

            for h_name, h_steps in HORIZONS.items():
                metrics = train_and_eval(series, h_steps, timestamps=ts_raw)
                if metrics is None:
                    continue
                results.append({"vm_id": vm_id, "horizon": h_name, **metrics})

            done_vm_ids.add(vm_id)

        except Exception as e:
            errors += 1
            msg = f"vm_id={vm_id}: {type(e).__name__}: {e}"
            error_log.append(msg)
            if errors <= 5:
                print(f"\n[ERROR #{errors}] {msg}")
                traceback.print_exc()
                print()
            done_vm_ids.add(vm_id)
            continue

        if (i + 1) % 10 == 0:
            _save_checkpoint(checkpoint_file, results, done_vm_ids)
            pct = len(done_vm_ids) / total * 100
            print(f"  [{len(done_vm_ids):3d}/{total}] {pct:.0f}%  "
                  f"results: {len(results)}  errors: {errors}  ✅")

    _save_checkpoint(checkpoint_file, results, done_vm_ids)

    if not results:
        print(f"\n[FATAL] 0 results — all {errors} VMs failed.")
        if error_log:
            for msg in error_log:
                print(f"  {msg}")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df.to_csv(per_vm_csv, index=False)
    print(f"\ndone!  total results: {len(results_df)}  "
          f"unique VMs: {results_df['vm_id'].nunique()}  errors: {errors}")
    return results_df


def _save_checkpoint(path, results, done_vm_ids):
    with open(path, "w") as f:
        json.dump({"results": results, "done_vm_ids": list(done_vm_ids)}, f)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — summarise
# ═══════════════════════════════════════════════════════════════════════════════

def summarise(output_dir):
    per_vm_csv  = os.path.join(output_dir, "bitbrains_per_vm.csv")
    summary_csv = os.path.join(output_dir, "bitbrains_summary.csv")
    results_df  = pd.read_csv(per_vm_csv)

    print("=" * 65)
    print("BITBRAINS RESULTS — active VMs only")
    print("metrics = MEDIAN across VMs")
    print("=" * 65)
    print(f"{'Horizon':<10} {'Naive R²':>10} {'ML R²':>10} "
          f"{'Δ pp':>8} {'Skill':>8} {'% VMs win':>11}")
    print("-" * 65)

    summary_rows = []
    for h in ["10min", "30min", "60min", "120min"]:
        hdf      = results_df[results_df["horizon"] == h]
        naive_r2 = hdf["naive_r2"].median()
        ml_r2    = hdf["ml_r2"].median()
        delta_pp = hdf["r2_delta"].median() * 100
        win_pct  = hdf["ml_wins"].mean() * 100
        skill_med = hdf["skill"].median() if "skill" in hdf.columns \
                    else (1.0 - hdf["ml_mae"] / hdf["naive_mae"]).median()

        print(f"{h:<10} {naive_r2:>10.3f} {ml_r2:>10.3f} "
              f"{delta_pp:>8.2f} {skill_med:>8.4f} {win_pct:>10.1f}%")

        summary_rows.append({
            "Horizon":            h,
            "BB_Naive_R2":        round(naive_r2,  3),
            "BB_ML_R2_median":    round(ml_r2,     3),
            "BB_Delta_pp":        round(delta_pp,  2),
            "BB_Skill_median":    round(skill_med, 4),
            "BB_Pct_VMs_ML_Wins": round(win_pct,   1),
        })

    print("=" * 65)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nsummary saved → {summary_csv}")

    print("\ncomparison with Alibaba hetero-ensemble results:")
    print(f"\n{'Horizon':<10} {'Alibaba Δpp':>13} {'Bitbrains Δpp':>15} {'Same direction?':>17}")
    print("-" * 58)
    for row in summary_rows:
        h    = row["Horizon"]
        ali  = ALI_DELTA_HETERO[h]
        bb   = row["BB_Delta_pp"]
        same = "✅ yes" if (ali > 0) == (bb > 0) else "⚠️  no"
        print(f"{h:<10} {ali:>13.2f} {bb:>15.2f}  {same:>17}")

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — sanity figure
# ═══════════════════════════════════════════════════════════════════════════════

def sanity_fig(output_dir):
    summary_df = pd.read_csv(os.path.join(output_dir, "bitbrains_summary.csv"))
    figure_png = os.path.join(output_dir, "bitbrains_sanity.png")
    horizons   = ["10min", "30min", "60min", "120min"]

    bb_naive = [summary_df[summary_df.Horizon == h]["BB_Naive_R2"].values[0]        for h in horizons]
    bb_delta = [summary_df[summary_df.Horizon == h]["BB_Delta_pp"].values[0]        for h in horizons]
    bb_wins  = [summary_df[summary_df.Horizon == h]["BB_Pct_VMs_ML_Wins"].values[0] for h in horizons]

    xi  = range(len(horizons))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Bitbrains Sanity-Check — XGBoost on active VMs", fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(xi, ALI_NAIVE, "o-", color=BLUE,   lw=2, ms=7, label="Alibaba")
    ax.plot(xi, bb_naive,  "s-", color=ORANGE, lw=2, ms=7, label="Bitbrains")
    ax.set_xticks(xi); ax.set_xticklabels(horizons)
    ax.set_title("Naive R² Decay", fontweight="bold")
    ax.set_ylabel("Median Naive R²")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.bar([x - 0.18 for x in xi], ALI_XGB_DELTA, 0.34, color=BLUE,   alpha=0.85, label="Alibaba (XGB)")
    ax.bar([x + 0.18 for x in xi], bb_delta,      0.34, color=ORANGE, alpha=0.85, label="Bitbrains (XGB)")
    ax.axhline(0, color="black", lw=1.4, ls="--")
    ax.set_xticks(xi); ax.set_xticklabels(horizons)
    ax.set_title("XGB Δpp vs Naive\n(same model both sides)", fontweight="bold")
    ax.set_ylabel("Median ΔR² (pp)"); ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")

    ax = axes[2]
    bar_colors = ["#d73027", "#fc8d59", "#4dac26", "#1a9641"]
    bars = ax.bar(horizons, bb_wins, color=bar_colors, width=0.5, edgecolor="white")
    ax.axhline(50, color="black", lw=1.5, ls="--", label="50% line")
    for bar, v in zip(bars, bb_wins):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title("% VMs Where ML Wins", fontweight="bold")
    ax.set_ylabel("% of Active VMs")
    ax.set_ylim(0, max(bb_wins) * 1.18 + 5)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(figure_png, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved → {figure_png}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 — CV-stratified analysis
# ═══════════════════════════════════════════════════════════════════════════════

def cv_strat(output_dir):
    per_vm_csv  = os.path.join(output_dir, "bitbrains_per_vm.csv")
    active_csv  = os.path.join(output_dir, "active_vms.csv")
    strat_csv   = os.path.join(output_dir, "bitbrains_cv_stratified.csv")

    results_df = pd.read_csv(per_vm_csv)
    active_vms = pd.read_csv(active_csv)

    # FIX: normalise vm_id on BOTH sides to int-string before merging.
    # active_vms.csv stores filenames like '1005' but pandas may have
    # read them as float64 → '1005.0', breaking the join silently.
    results_df["vm_id"] = results_df["vm_id"].apply(_norm_vm_id)
    active_vms["vm_id"] = active_vms["vm_id"].apply(_norm_vm_id)

    cv_lookup = active_vms.set_index("vm_id")[["cv", "hurst", "acf1"]]
    merged    = results_df.merge(cv_lookup, on="vm_id", how="left")

    if "skill" not in merged.columns:
        merged["skill"] = 1.0 - merged["ml_mae"] / merged["naive_mae"].replace(0, float("nan"))

    bins   = [0, 0.3, 0.7, float("inf")]
    labels = ["Low  CV (<0.3)", "Med  CV (0.3-0.7)", "High CV (>0.7)"]
    merged["cv_bin"] = pd.cut(merged["cv"], bins=bins, labels=labels)

    # Diagnostic: show merge quality
    n_missing_cv = merged["cv"].isna().sum()
    if n_missing_cv > 0:
        print(f"[WARN] {n_missing_cv} rows have missing CV after merge — "
              f"vm_id type mismatch? Unique results vm_ids (sample): "
              f"{results_df['vm_id'].head(3).tolist()}, "
              f"active vm_ids (sample): {active_vms.index[:3].tolist()}")
    else:
        print(f"merge OK — CV populated for all {len(merged)} rows")

    print("\nCV-STRATIFIED ANALYSIS — Bitbrains active VMs")
    print("=" * 74)
    print(f"{'Horizon':<10} {'CV bin':<22} {'N':>5} {'Win%':>8} {'Skill':>8}  p-val  Cliff δ")
    print("-" * 74)

    strat_rows = []
    for h in ["10min", "30min", "60min", "120min"]:
        hdf = merged[merged["horizon"] == h]
        for bin_label in labels:
            g = hdf[hdf["cv_bin"] == bin_label].dropna(subset=["skill"])
            n = len(g)
            if n < 5:
                print(f"{h:<10} {bin_label:<22} {n:>5}  (too few VMs)")
                continue

            win_pct    = g["ml_wins"].mean() * 100
            skill_med  = g["skill"].median()
            skills_arr = g["skill"].dropna().values

            try:
                if len(skills_arr) >= 10 and np.std(skills_arr) > 1e-9:
                    _, pval = wilcoxon(skills_arr)
                else:
                    pval = float("nan")
            except Exception:
                pval = float("nan")

            n_pos    = (skills_arr > 0).sum()
            n_neg    = (skills_arr < 0).sum()
            cliffs_d = (n_pos - n_neg) / len(skills_arr)

            sig = "—" if np.isnan(pval) else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))

            print(f"{h:<10} {bin_label:<22} {n:>5} {win_pct:>7.1f}% "
                  f"{skill_med:>8.4f}  {sig:<4} {cliffs_d:>+.3f}")

            strat_rows.append({
                "Horizon":      h,
                "CV_bin":       bin_label,
                "N":            n,
                "Win_pct":      round(win_pct,    1),
                "Skill_median": round(skill_med,  4),
                "p_value":      round(float(pval), 4) if not np.isnan(pval) else float("nan"),
                "Cliffs_delta": round(cliffs_d,   3),
            })
        print()

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(strat_csv, index=False)
    print(f"saved → {strat_csv}")

    print("\nAlibaba skill scores for reference:")
    for h, s in ALI_SKILL.items():
        print(f"  {h}: {s:+.4f}")

    return strat_df


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6 — publication figure
# ═══════════════════════════════════════════════════════════════════════════════

def pub_fig(output_dir):
    summary_df = pd.read_csv(os.path.join(output_dir, "bitbrains_summary.csv"))
    strat_df   = pd.read_csv(os.path.join(output_dir, "bitbrains_cv_stratified.csv"))
    fig_path   = os.path.join(output_dir, "bitbrains_cross_validation.pdf")

    horizons = ["10min", "30min", "60min", "120min"]
    bb_naive = [summary_df[summary_df.Horizon == h]["BB_Naive_R2"].values[0]        for h in horizons]
    bb_delta = [summary_df[summary_df.Horizon == h]["BB_Delta_pp"].values[0]        for h in horizons]

    x     = np.arange(len(horizons))
    width = 0.32
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "Bitbrains Cross-Dataset Validation\n"
        "XGBoost baseline on active VMs vs. Alibaba (hetero ensemble · XGB-only reference)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Panel 1: Naive R² decay
    ax = axes[0]
    xi = range(len(horizons))
    ax.plot(xi, ALI_NAIVE, "o-", color=BLUE,   lw=2.2, ms=8, label="Alibaba (containers)")
    ax.plot(xi, bb_naive,  "s-", color=ORANGE, lw=2.2, ms=8, label="Bitbrains (active VMs)")
    ax.fill_between(xi, ALI_NAIVE, bb_naive, alpha=0.08, color="grey")
    ax.set_xticks(range(len(horizons))); ax.set_xticklabels(horizons)
    ax.set_title("Naive Baseline Degradation\nwith Prediction Horizon", fontweight="bold")
    ax.set_ylabel("Median Naive Persistence R²", fontsize=10)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.legend(fontsize=9); ax.grid(alpha=0.25); ax.tick_params(labelsize=9)

    # Panel 2: XGBoost Δpp
    ax = axes[1]
    bars1 = ax.bar(x - width / 2, ALI_XGB_DELTA, width, color=BLUE,   alpha=0.85,
                   label="Alibaba (XGBoost)", edgecolor="white", lw=0.8)
    bars2 = ax.bar(x + width / 2, bb_delta,      width, color=ORANGE, alpha=0.85,
                   label="Bitbrains (XGBoost)", edgecolor="white", lw=0.8)
    ax.axhline(0, color="black", lw=1.5, ls="--", label="No improvement")
    ax.set_title("XGBoost ML Improvement over Naive\n(R² pp, same model both sides)", fontweight="bold")
    ax.set_ylabel("Median ΔR² (pp)", fontsize=10)
    ax.set_xlabel("Prediction Horizon")
    ax.set_xticks(x); ax.set_xticklabels(horizons)
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y"); ax.tick_params(labelsize=9)

    all_vals = list(ALI_XGB_DELTA) + list(bb_delta)
    y_range  = max(all_vals) - min(all_vals) if all_vals else 1.0
    offset   = max(0.05 * y_range, 0.05)
    for bar in list(bars1) + list(bars2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (offset if v >= 0 else -offset * 3),
                f"{v:+.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Panel 3: CV × Horizon heatmap
    ax        = axes[2]
    cv_labels = ["Low CV\n(<0.3)", "Med CV\n(0.3-0.7)", "High CV\n(>0.7)"]
    bin_key   = {
        "Low  CV (<0.3)":    0,
        "Med  CV (0.3-0.7)": 1,
        "High CV (>0.7)":    2,
    }
    heat = np.full((len(cv_labels), len(horizons)), np.nan)
    for row in strat_df.itertuples():
        h_idx = horizons.index(row.Horizon)
        b_idx = bin_key.get(str(row.CV_bin).strip(), -1)
        if b_idx >= 0:
            heat[b_idx, h_idx] = row.Win_pct

    im = ax.imshow(heat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="% VMs where ML wins", shrink=0.8)
    ax.set_xticks(range(len(horizons)));  ax.set_xticklabels(horizons, fontsize=9)
    ax.set_yticks(range(len(cv_labels))); ax.set_yticklabels(cv_labels, fontsize=9)
    ax.set_title("ML Win Rate by CV Bin × Horizon\n(Bitbrains active VMs)", fontweight="bold")
    for i in range(len(cv_labels)):
        for j in range(len(horizons)):
            v = heat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v < 30 or v > 70 else "black")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"saved → {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

STAGES = ["scan_vms", "train", "summarise", "sanity_fig", "cv_strat", "pub_fig", "all"]


def main():
    parser = argparse.ArgumentParser(description="Bitbrains cross-dataset validation pipeline")
    parser.add_argument("--stage",            default="all", choices=STAGES)
    parser.add_argument("--data_dir",         default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir",       default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reset_checkpoint", action="store_true")
    args = parser.parse_args()

    if not _is_notebook():
        matplotlib.use("Agg")

    os.makedirs(args.output_dir, exist_ok=True)

    run_all    = args.stage == "all"
    active_vms = None

    if run_all or args.stage == "scan_vms":
        active_vms = scan_vms(args.data_dir, args.output_dir)

    if run_all or args.stage == "train":
        if active_vms is None:
            avcsv = os.path.join(args.output_dir, "active_vms.csv")
            if not os.path.exists(avcsv):
                sys.exit("active_vms.csv not found — run --stage scan_vms first")
            active_vms = pd.read_csv(avcsv)
            print(f"active_vms loaded from CSV  ({len(active_vms)} VMs)")
        train_all(active_vms, args.data_dir, args.output_dir,
                  reset_checkpoint=args.reset_checkpoint)

    if run_all or args.stage == "summarise":
        summarise(args.output_dir)

    if run_all or args.stage == "sanity_fig":
        sanity_fig(args.output_dir)

    if run_all or args.stage == "cv_strat":
        cv_strat(args.output_dir)

    if run_all or args.stage == "pub_fig":
        pub_fig(args.output_dir)


if __name__ == "__main__":
    main()