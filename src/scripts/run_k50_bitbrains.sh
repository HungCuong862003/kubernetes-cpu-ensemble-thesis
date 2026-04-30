#!/usr/bin/env bash
# run_k50_bitbrains.sh
#
# K=50 sensitivity check for Bitbrains Chronos-2 zero-shot. The existing
# day1_chronos2_sanity.py already takes --num_origins, so this is just the
# right env + args to point it at the bitbrains dataset with K=50 origins
# per VM instead of the K=20 we used on Day 1.
#
# Why K=50: K=20 gave us ~2840 eval points per horizon on bitbrains (156
# active VMs * ~18 reachable origins/VM after the MIN_CONTEXT=24 filter).
# That ~18 is already close to K=20's ceiling, so K=50 won't 2.5x the
# sample count - pick_origins saturates at (hi-lo+1) unique integers and
# many bitbrains VMs are short. Realistic uplift: maybe +10-30% more eval
# points. Still a meaningful sensitivity check: if the K=20 numbers were
# noise-driven, shifting the origin positions to the K=50 grid will move
# the reported R^2 by more than tolerance. If they were stable, deltas
# will be small. The sibling compare script makes that call.
#
# Run this on the Vast.ai 4090 box. Weight cache is already warm from the
# Day 1 run so there is no ~500MB download this time.
#
# Expected wall time: ~25-35 min (roughly 2x the K=20 run - more predict_df
# passes, but many origin positions saturate on short VMs so the factor
# isn't a clean 50/20).

set -euo pipefail

# Point at the bitbrains data root. Override with env if your layout differs.
DATA_ROOT="${DATA_ROOT:-/workspace/k8s-ensemble-forecast/data/bitbrains}"
OUTPUT_JSON="${OUTPUT_JSON:-k50_bitbrains_check.json}"

# 512 was stable on the day 1 bitbrains run so we reuse it. If you see OOM
# (shouldn't happen - bitbrains has fewer VMs than alibaba) drop to 256.
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "[run_k50_bitbrains] DATA_ROOT    = ${DATA_ROOT}"
echo "[run_k50_bitbrains] OUTPUT_JSON  = ${OUTPUT_JSON}"
echo "[run_k50_bitbrains] BATCH_SIZE   = ${BATCH_SIZE}"
echo "[run_k50_bitbrains] num_origins  = 50"
echo

# Sanity: refuse to run if the things we need aren't there. These checks
# are cheap and save 30 min of wasted GPU time if something's wrong.
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "FATAL: DATA_ROOT does not exist: ${DATA_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${DATA_ROOT}/test.parquet" ]]; then
    echo "FATAL: ${DATA_ROOT}/test.parquet not found. The sanity script" \
         "needs a raw test.parquet at the top of DATA_ROOT - not a per-horizon" \
         "split. If your file is elsewhere set FORCE_TEST_FILE." >&2
    exit 1
fi
if [[ ! -f "day1_chronos2_sanity.py" ]]; then
    echo "FATAL: day1_chronos2_sanity.py not in current dir. cd to the" \
         "project root before running this wrapper." >&2
    exit 1
fi
if [[ ! "${DATA_ROOT}" == *"bitbrains"* ]]; then
    # Not fatal - we pass --dataset bitbrains explicitly below so auto-detect
    # can't go wrong. But if the path doesn't mention bitbrains someone may
    # have pointed this at the wrong data entirely.
    echo "WARN: DATA_ROOT does not contain the word 'bitbrains'." \
         "The --dataset flag below will force bitbrains regardless, but" \
         "double-check you haven't pointed at alibaba/bytedance by mistake." >&2
fi

# Explicit --dataset flag here as belt-and-braces; auto-detect from path
# would work but being explicit means one less thing to misread in the log.
DATA_ROOT="${DATA_ROOT}" \
python day1_chronos2_sanity.py \
    --dataset bitbrains \
    --num_origins 50 \
    --batch_size "${BATCH_SIZE}" \
    --output "${OUTPUT_JSON}"

# Verify the output was actually written. set -e catches a non-zero exit
# from the python call, but not a silent "ran to completion but wrote
# nothing" which has happened before on filesystem quota issues.
if [[ ! -s "${OUTPUT_JSON}" ]]; then
    echo "FATAL: python call exited 0 but ${OUTPUT_JSON} is missing or" \
         "empty. Check disk space and permissions." >&2
    exit 1
fi

echo
echo "[run_k50_bitbrains] done. Output: ${OUTPUT_JSON}"
echo "[run_k50_bitbrains] next: run compare_k20_k50_bitbrains.py against"
echo "                    k20_bitbrains_v2.json to check |deltaR2|. The"
echo "                    comparator is pure-stdlib Python so it runs"
echo "                    anywhere - on this Vast box or on your laptop."
