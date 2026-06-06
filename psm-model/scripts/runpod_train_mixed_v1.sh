#!/bin/bash
set -euo pipefail
cd /workspace/PSM
export PYTHONPATH=psm-model/src

python3 -m psm_model.train \
  psm-model/data/curriculum/psm-50m-action-mixed-v1-ctx2048.jsonl \
  --out psm-model/checkpoints/real-v3-50m-action-mixed-v1.pt \
  --tokenizer psm-model/checkpoints/real-v3-50m-action-mixed-v1-step-000400.tokenizer.json \
  --resume auto \
  --steps 8000 \
  --batch-size 1 \
  --preset 50m \
  --learning-rate 0.0003 \
  --min-learning-rate 0.0001 \
  --warmup-steps 50 \
  --device cuda \
  --save-every 200 \
  --metrics-out psm-model/checkpoints/real-v3-50m-action-mixed-v1.metrics.jsonl \
  --output-format action \
  --sampling action_balanced \
  --action-span-loss-weight 1 \
  --structural-loss-weight 1 \
  --action-span-weight promote_semantic=2 \
  --action-span-weight ignore=3 \
  --action-span-weight update_existing=2 \
  --eval-every 400 \
  --abort-after-step 500 \
  --collapse-threshold 0.85 \
  --probe psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl
