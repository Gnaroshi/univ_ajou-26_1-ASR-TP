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

EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-1e-4}
ENCODER_LR=${ENCODER_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
MAX_SEC=${MAX_SEC:-10.0}

OUTPUT_DIR=${OUTPUT_DIR:-outputs/circor_outcome_m2d_${PEFT_METHOD}}

python -m hs.train \
  --train_manifest artifacts/circor/records_train.json \
  --valid_manifest artifacts/circor/records_valid.json \
  --m2d_weight_path "$M2D_WEIGHT" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --encoder_lr "$ENCODER_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --max_sec "$MAX_SEC" \
  --peft_method "$PEFT_METHOD" \
  --trainable_last_blocks "$TRAINABLE_LAST_BLOCKS" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_keywords "$LORA_TARGET_KEYWORDS"