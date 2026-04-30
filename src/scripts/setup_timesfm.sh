#!/bin/bash
# setup_timesfm.sh — install TimesFM 2.5 from git on Vast.ai.
# PyPI ships stale 2.0; we need git HEAD for the 2.5 API.

set -e

echo "=== TimesFM 2.5 setup ==="

python - <<'PY'
import torch
print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
PY

pip install -q git+https://github.com/google-research/timesfm.git

python - <<'PY'
import timesfm
print(f"timesfm module: {timesfm.__file__}")
assert hasattr(timesfm, "TimesFM_2p5_200M_torch"), \
    "TimesFM_2p5_200M_torch not found — pip pulled an older version"
print("TimesFM_2p5_200M_torch class present")
PY

# Warm up: download weights (~925 MB) so first real run doesn't wait
python - <<'PY'
import timesfm
print("downloading weights (~925 MB)...")
m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
print("weights cached")
PY

echo "=== setup done ==="