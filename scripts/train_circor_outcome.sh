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

CUDA_VISIBLE_DEVICES=3 python -m hs.train \
  --train_manifest artifacts/circor/records_train.json \
  --valid_manifest artifacts/circor/records_valid.json \
  --m2d_weight_path "$M2D_WEIGHT" \
  --output_dir outputs/circor_outcome_m2d \
  --epochs 10 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 1e-4 \
  --max_sec 10.0
