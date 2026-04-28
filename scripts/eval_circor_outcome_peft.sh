#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
source .venv-m2d/bin/activate

python scripts/patch_m2d_portable.py

M2D_WEIGHT=${M2D_WEIGHT:-$(find weights/m2d -name "*.pth" | head -n 1)}
if [ -z "$M2D_WEIGHT" ]; then
  echo "[ERROR] No M2D checkpoint found under weights/m2d"
  exit 1
fi

PEFT_METHOD=${PEFT_METHOD:-norm}
TRAINABLE_LAST_BLOCKS=${TRAINABLE_LAST_BLOCKS:-1}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16.0}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_KEYWORDS=${LORA_TARGET_KEYWORDS:-qkv,proj}

OUTPUT_DIR=${OUTPUT_DIR:-outputs/circor_outcome_m2d_${PEFT_METHOD}}
RUN_DIR=${RUN_DIR:-$OUTPUT_DIR}
PATIENT_AGGREGATION=${PATIENT_AGGREGATION:-mean}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_SEC=${MAX_SEC:-10.0}

python -m hs.eval \
  --manifest artifacts/circor/records_test.json \
  --checkpoint "$RUN_DIR/best.pt" \
  --m2d_weight_path "$M2D_WEIGHT" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --max_sec "$MAX_SEC" \
  --patient_aggregation "$PATIENT_AGGREGATION" \
  --peft_method "$PEFT_METHOD" \
  --trainable_last_blocks "$TRAINABLE_LAST_BLOCKS" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_keywords "$LORA_TARGET_KEYWORDS" \
  --output_json "$RUN_DIR/test_metrics_${PATIENT_AGGREGATION}.json"
