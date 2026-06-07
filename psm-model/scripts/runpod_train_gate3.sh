#!/bin/bash
# Gate 3: full StorageDecision from Gate-2 best checkpoint (step 9800).
# NOTE: --steps is ABSOLUTE target, not additional steps → 9800 + 3000 = 12800.
set -euo pipefail
cd /workspace/PSM
export PYTHONPATH=psm-model/src

RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.tokenizer.json}"
TARGET_STEPS="${TARGET_STEPS:-12800}"

python3 -m psm_model.train \
  psm-model/data/curriculum/psm-50m-full-storage-v1-filtered.jsonl \
  --out psm-model/checkpoints/real-v3-50m-full-v2.pt \
  --resume "$RESUME" \
  --tokenizer "$TOK" \
  --steps "$TARGET_STEPS" \
  --batch-size 1 \
  --preset 50m \
  --output-format tagged \
  --sampling action_balanced \
  --device cuda \
  --save-every 200 \
  --metrics-out psm-model/checkpoints/real-v3-50m-full-v2.metrics.jsonl
