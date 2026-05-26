#!/bin/bash
# Inspect Chronos-2's actual loss function from the installed source
# Run this on Vast to get the definitive answer

set -e
cd /venv/main/lib/python3.12/site-packages/chronos/chronos2

echo "=========================================="
echo "INSPECTION 1: The loss function definition"
echo "=========================================="
sed -n '480,570p' model.py
echo ""

echo "=========================================="
echo "INSPECTION 2: Where _compute_loss is called from"
echo "=========================================="
grep -n "_compute_loss" model.py

echo ""
echo "=========================================="
echo "INSPECTION 3: The forward() signature and loss assembly"
echo "=========================================="
grep -n "def forward" model.py
sed -n '/def forward/,/return.*Chronos2Output/p' model.py | head -80

echo ""
echo "=========================================="
echo "INSPECTION 4: What does Chronos2Output contain?"
echo "=========================================="
grep -n "class Chronos2Output\|dataclass\|loss\s*[:=]" model.py | head -20

echo ""
echo "=========================================="
echo "INSPECTION 5: chronos-forecasting version"
echo "=========================================="
python3 -c "import chronos; print('chronos:', chronos.__version__ if hasattr(chronos, '__version__') else 'no __version__ attr')"
pip show chronos-forecasting | grep -E "Name|Version|Location"
