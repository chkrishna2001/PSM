#!/bin/bash
# Gate 2 repair: resume from mixed-v1 best checkpoint on mixed-v2 curriculum.
# Expects HF bootstrap: step-007600.pt + tokenizer under psm-model/checkpoints/
set -euo pipefail
cd /workspace/PSM
export PYTHONPATH=psm-model/src

RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-action-mixed-v1-step-007600.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-action-mixed-v1-step-007600.tokenizer.json}"

python3 -m psm_model.train \
  psm-model/data/curriculum/psm-50m-action-mixed-v2-ctx2048.jsonl \
  --out psm-model/checkpoints/real-v3-50m-action-mixed-v2.pt \
  --tokenizer "$TOK" \
  --resume "$RESUME" \
  --steps 12000 \
  --batch-size 1 \
  --preset 50m \
  --learning-rate 0.00015 \
  --min-learning-rate 0.00005 \
  --warmup-steps 100 \
  --device cuda \
  --save-every 200 \
  --metrics-out psm-model/checkpoints/real-v3-50m-action-mixed-v2.metrics.jsonl \
  --output-format action \
  --sampling action_balanced \
  --action-span-loss-weight 1 \
  --structural-loss-weight 1 \
  --action-span-weight promote_semantic=6 \
  --action-span-weight store_episodic=2.5 \
  --action-span-weight ignore=4 \
  --action-span-weight update_existing=2 \
  --eval-every 200 \
  --probe psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
  --manual-probe psm-model/data/direct-behavior-v1/manual-probe.jsonl \
  --abort-after-step 8000 \
  --collapse-threshold 0.85
