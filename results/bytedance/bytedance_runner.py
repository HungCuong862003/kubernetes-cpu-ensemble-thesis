#!/usr/bin/env python3
"""
bytedance_runner.py — Patch sprint1_Main.py for ByteDance IaaS dataset.

ByteDance IaaS differs from Alibaba in 4 ways:
  1. Sampling interval is 10 min (not 5 min)
  2. Points-per-day is 144 (not 288)
  3. Only CPU metric exists (no mem_util_percent)
  4. Column names differ: "cols" → container_id, "data" → cpu_util_percent

Generates a patched copy (sprint1_bytedance.py) and optionally runs it.

Inputs:
    sprint1_Main.py                            (--sprint-src)

Outputs:
    sprint1_bytedance.py                       (--output-script)
"""

import argparse
import os
import re
import subprocess
import sys


def patch_script(src_path, dst_path):
    """Read sprint1_Main.py, apply ByteDance patches, write sprint1_bytedance.py."""

    with open(src_path, "r") as f:
        code = f.read()

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 1: SAMPLING_INTERVAL 300 → 600
    # ByteDance samples every 10 minutes, not 5.
    # ──────────────────────────────────────────────────────────────────────
    code = re.sub(
        r'SAMPLING_INTERVAL\s*=\s*300.*',
        'SAMPLING_INTERVAL = 600   # ByteDance: 10-min intervals',
        code, count=1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 2: Horizon steps
    # Original (5-min steps): {"10min": 2, "30min": 6, "60min": 12, "120min": 24}
    # ByteDance (10-min steps): {"10min": 1, "30min": 3, "60min": 6, "120min": 12}
    # ──────────────────────────────────────────────────────────────────────
    code = re.sub(
        r'HORIZONS\s*=\s*\{[^}]+\}',
        'HORIZONS = {"10min": 1, "30min": 3, "60min": 6, "120min": 12}  # ByteDance: 10-min steps',
        code, count=1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 3: ppd (points per day) 288 → 144
    # At 10-min intervals: 24*60/10 = 144 points/day.
    # ──────────────────────────────────────────────────────────────────────
    code = re.sub(
        r'ppd\s*=\s*288',
        'ppd = 144  # ByteDance: 144 points/day at 10-min intervals',
        code, count=1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 4: Add SKIP_MEM_ENSEMBLE flag to Config
    # ──────────────────────────────────────────────────────────────────────
    code = code.replace(
        "NO_BILSTM = False",
        "NO_BILSTM = False\n    SKIP_MEM_ENSEMBLE = True   # ByteDance: CPU-only dataset",
        1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 5: Column remapping + synthetic mem in load_data()
    # Inject right before "return train, val, test" in load_data()
    # ──────────────────────────────────────────────────────────────────────
    load_data_return = '    return train, val, test\n\n\ndef convert_csv_to_parquet():'
    patched_return = '''    # --- ByteDance patches: column remapping + synthetic mem ---
    for _df in [train, val, test]:
        if "cols" in _df.columns and "container_id" not in _df.columns:
            _df.rename(columns={"cols": "container_id"}, inplace=True)
        if "data" in _df.columns and "cpu_util_percent" not in _df.columns:
            _df.rename(columns={"data": "cpu_util_percent"}, inplace=True)
        if "mem_util_percent" not in _df.columns:
            _df["mem_util_percent"] = 0.0
    # --- end ByteDance patches ---

    return train, val, test


def convert_csv_to_parquet():'''
    code = code.replace(load_data_return, patched_return, 1)

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 6: Make naive MEM evaluation conditional
    # ──────────────────────────────────────────────────────────────────────
    code = code.replace(
        '    naive_mem_m  = calc_metrics(y_mem_te, naive_mem)\n'
        '    naive_mem_da = directional_accuracy(y_mem_te, naive_mem, y_cur_mem)\n'
        '    metrics["naive_mem"] = {**naive_mem_m, "DA": naive_mem_da["DA_significant"]}\n'
        '    all_preds["naive_mem"] = naive_mem\n'
        '    ckpt.save_pred(horizon_name, "naive_mem", naive_mem)',

        '    if not cfg.SKIP_MEM_ENSEMBLE:\n'
        '        naive_mem_m  = calc_metrics(y_mem_te, naive_mem)\n'
        '        naive_mem_da = directional_accuracy(y_mem_te, naive_mem, y_cur_mem)\n'
        '        metrics["naive_mem"] = {**naive_mem_m, "DA": naive_mem_da["DA_significant"]}\n'
        '        all_preds["naive_mem"] = naive_mem\n'
        '        ckpt.save_pred(horizon_name, "naive_mem", naive_mem)\n'
        '    else:\n'
        '        naive_mem_m = {"R2": 0.0, "MAE": 0.0, "RMSE": 0.0}',
        1
    )

    # Fix the mark_done to not include naive_mem when skipping
    code = code.replace(
        '    ckpt.mark_done(horizon_name, "naive",\n'
        '                   {"naive": metrics["naive"], "naive_mem": metrics["naive_mem"]},',
        '    ckpt.mark_done(horizon_name, "naive",\n'
        '                   {"naive": metrics["naive"]} | ({"naive_mem": metrics["naive_mem"]} if not cfg.SKIP_MEM_ENSEMBLE else {}),',
        1
    )

    # Skip MEM log line
    code = code.replace(
        '    log(f"  MEM  R2={naive_mem_m[\'R2\']:.4f}  MAE={naive_mem_m[\'MAE\']:.3f}")',
        '    if not cfg.SKIP_MEM_ENSEMBLE:\n'
        '        log(f"  MEM  R2={naive_mem_m[\'R2\']:.4f}  MAE={naive_mem_m[\'MAE\']:.3f}")',
        1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 7: Skip stage 11/14 (Homo Ensemble MEM)
    # ──────────────────────────────────────────────────────────────────────
    code = code.replace(
        '    # ---- 11/14  HOMO ENSEMBLE MEM ----\n'
        '    cached_mem_ens, m_mem_ens = _load_cached(ckpt, horizon_name,\n'
        '                                              "homo_ensemble_mem", expected_len=n_test)',

        '    # ---- 11/14  HOMO ENSEMBLE MEM ----\n'
        '    if cfg.SKIP_MEM_ENSEMBLE:\n'
        '        log("11/14  Homo Ensemble (MEM): SKIPPED (CPU-only dataset)")\n'
        '        ckpt.mark_done(horizon_name, "homo_ensemble_mem", {}, elapsed_s=0)\n'
        '        cached_mem_ens, m_mem_ens = None, None  # skip block below\n'
        '    else:\n'
        '        cached_mem_ens, m_mem_ens = _load_cached(ckpt, horizon_name,\n'
        '                                                  "homo_ensemble_mem", expected_len=n_test)',
        1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 8: Skip mem references in improvement section
    # ──────────────────────────────────────────────────────────────────────
    code = re.sub(
        r'("mem_ens_vs_naive_pp":\s*float\(.+?\* 100\))',
        r'\1 if not cfg.SKIP_MEM_ENSEMBLE else 0.0',
        code, count=1, flags=re.DOTALL
    )

    # Skip mem rows in the summary print
    code = code.replace(
        '    for name in ["naive_mem", "homo_ensemble_mem"]:\n'
        '        if name in metrics:',
        '    for name in ["naive_mem", "homo_ensemble_mem"]:\n'
        '        if name in metrics and not cfg.SKIP_MEM_ENSEMBLE:',
        1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 9: Skip mem in CI calculation (analysis section)
    # ──────────────────────────────────────────────────────────────────────
    code = code.replace(
        '            ("naive_mem",         (y_mem_te, naive_mem)),\n'
        '            ("homo_ensemble_mem", (y_mem_te, all_preds.get("homo_ensemble_mem", naive_mem))),\n'
        '        ]:',

        '        ] + ([\n'
        '            ("naive_mem",         (y_mem_te, naive_mem)),\n'
        '            ("homo_ensemble_mem", (y_mem_te, all_preds.get("homo_ensemble_mem", naive_mem))),\n'
        '        ] if not cfg.SKIP_MEM_ENSEMBLE else []):',
        1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 10: dropna — don't require mem columns when SKIP_MEM_ENSEMBLE
    # ──────────────────────────────────────────────────────────────────────
    code = code.replace(
        '    required = ["cpu_target", "cpu_residual", "naive_cpu",\n'
        '                "mem_target", "mem_residual", "naive_mem"]',
        '    required = ["cpu_target", "cpu_residual", "naive_cpu"]\n'
        '    if not cfg.SKIP_MEM_ENSEMBLE:\n'
        '        required += ["mem_target", "mem_residual", "naive_mem"]',
        1
    )

    # ──────────────────────────────────────────────────────────────────────
    # PATCH 11: Title
    # ──────────────────────────────────────────────────────────────────────
    code = code.replace(
        'print("SPRINT 1 MAIN")',
        'print("SPRINT 1 — BYTEDANCE IaaS CROSS-VALIDATION")',
        1
    )

    with open(dst_path, "w") as f:
        f.write(code)

    print(f"Patched script written to: {dst_path}")

    # Verify key patches
    checks = [
        (r"SAMPLING_INTERVAL\s*=\s*600", "SAMPLING_INTERVAL = 600"),
        (r"ppd = 144", "ppd = 144"),
        (r"SKIP_MEM_ENSEMBLE\s*=\s*True", "SKIP_MEM_ENSEMBLE = True"),
        (r'"10min": 1', "Horizon steps (10min: 1)"),
        (r'"30min": 3', "Horizon steps (30min: 3)"),
        (r'"cols".*"container_id"', "Column rename: cols → container_id"),
        (r'"data".*"cpu_util_percent"', "Column rename: data → cpu_util_percent"),
        (r'"mem_util_percent" not in _df\.columns', "Synthetic mem_util_percent"),
        (r'mem_ens_vs_naive_pp.*SKIP_MEM_ENSEMBLE', "mem_ens_vs_naive guarded"),
        (r"BYTEDANCE", "Title"),
    ]
    with open(dst_path) as f:
        patched = f.read()
    all_ok = True
    for pattern, label in checks:
        if re.search(pattern, patched, re.DOTALL):
            print(f"  ✓ {label}")
        else:
            print(f"  ✗ {label} — PATCH FAILED")
            all_ok = False

    if all_ok:
        print("\nAll patches applied successfully.")
    else:
        print("\nWARNING: Some patches failed — check output above.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Patch sprint1_Main.py for ByteDance IaaS dataset")
    parser.add_argument("--sprint-src", required=True,
                        help="Path to sprint1_Main.py")
    parser.add_argument("--output-script", default=None,
                        help="Output path for patched script (default: sprint1_bytedance.py "
                             "in same directory as sprint-src)")
    parser.add_argument("--no-run", action="store_true",
                        help="Only generate the patched script, don't run it")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (passed to patched script)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (passed to patched script)")
    args = parser.parse_args()

    src = os.path.abspath(args.sprint_src)
    if not os.path.exists(src):
        print(f"ERROR: {src} not found")
        sys.exit(1)

    dst = args.output_script or os.path.join(
        os.path.dirname(src), "sprint1_bytedance.py")

    patch_script(src, dst)

    if args.no_run:
        print("\n--no-run specified. Patched script ready at:", dst)
        return

    # Run the patched script
    cmd = [sys.executable, dst, "--sequential"]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()