#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
source .venv-m2d/bin/activate

python scripts/patch_m2d_portable.py

M2D_WEIGHT=$(find weights/m2d -name "*.pth" | head -n 1)
if [ -z "$M2D_WEIGHT" ]; then
  echo "[ERROR] No M2D checkpoint found under weights/m2d"
  exit 1
fi

python -m hs.eval \
  --manifest artifacts/circor/records_test.json \
  --checkpoint outputs/circor_outcome_m2d/best.pt \
  --m2d_weight_path "$M2D_WEIGHT" \
  --batch_size 8 \
  --num_workers 4 \
  --max_sec 10.0 \
  --output_json outputs/circor_outcome_m2d/test_metrics.json
