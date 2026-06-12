#!/bin/bash
# Warm pod: build gate5 mixed curriculum, start training tmux, exit.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PYTHONPATH=psm-model/src
export PSM_RUNPOD=1

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
DEVICE="${PSM_TRAIN_DEVICE:-cuda}"
RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2-step-048000.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-full-v2-step-048000.tokenizer.json}"
TARGET_STEPS="${TARGET_STEPS:-51000}"
CURRICULUM="${GATE5_CURRICULUM:-psm-model/data/curriculum/psm-50m-gate5-train-v1.jsonl}"
SAVE_EVERY="${SAVE_EVERY:-200}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-120}"
UPLOAD_ALL="${UPLOAD_ALL:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-5}"
EVAL_EVERY="${EVAL_EVERY:-400}"
RESUME_STEP="$(basename "$RESUME" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"

echo "=== start gate5 train-only $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$RESUME_STEP target=$TARGET_STEPS ==="

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
chmod +x psm-model/scripts/runpod_upload_gate4.sh 2>/dev/null || true

if [[ "${SKIP_CURRICULUM_BUILD:-0}" != "1" ]]; then
  RECALL_PROBE="${GATE5_RECALL_PROBE:-psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl}"
  if [[ ! -f "$RECALL_PROBE" ]]; then
    python3 -m psm_model.generate_recall_curriculum "$RECALL_PROBE"
  fi
  python3 -m psm_model.build_gate5_train_v1 "$CURRICULUM" \
    --expanded-probes psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    --direct-probes psm-model/data/probes/direct_probes.jsonl \
    --expanded-copies "${EXPANDED_COPIES:-25}" \
    --direct-copies "${DIRECT_COPIES:-100}" \
    --recall-copies "${RECALL_COPIES:-20}"
fi

export GATE4_PINNED_STEPS="${RESUME_STEP}"
SYNC_CMD="cd '$ROOT' && export HF_TOKEN=\"\${HF_TOKEN:-}\" UPLOAD_ALL=${UPLOAD_ALL} KEEP_LOCAL=$KEEP_LOCAL GATE4_PINNED_STEPS='$RESUME_STEP' PSM_HF_MODEL_REPO='$MODEL_REPO' PSM_HF_DATASET_REPO='$DATASET_REPO' && while true; do sleep $SYNC_INTERVAL_SEC; bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate5-sync.log; done"
tmux kill-session -t psm-gate5-sync 2>/dev/null || true
tmux new-session -d -s psm-gate5-sync bash -lc "$(printf '%q' "$SYNC_CMD")"
echo "HF sync: tmux psm-gate5-sync every ${SYNC_INTERVAL_SEC}s"

TRAIN_LOG="/tmp/psm-gate5-train.log"
TRAIN_DONE="/tmp/psm-gate5.done"
rm -f "$TRAIN_DONE"
tmux kill-session -t psm-gate5 2>/dev/null || true
tmux new-session -d -s psm-gate5 bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src
  python3 -m psm_model.train \
    '$CURRICULUM' \
    --out psm-model/checkpoints/real-v3-50m-full-v2.pt \
    --resume '$RESUME' \
    --tokenizer '$TOK' \
    --steps '$TARGET_STEPS' \
    --batch-size '$BATCH_SIZE' \
    --learning-rate '$LEARNING_RATE' \
    --min-learning-rate '$MIN_LEARNING_RATE' \
    --warmup-steps '${WARMUP_STEPS:-50}' \
    --preset 50m \
    --output-format tagged \
    --sampling random \
    --device '$DEVICE' \
    --save-every '$SAVE_EVERY' \
    --metrics-out psm-model/checkpoints/real-v3-50m-full-v2-gate5.metrics.jsonl \
    --structural-loss-weight '${STRUCTURAL_LOSS_WEIGHT:-1}' \
    --eval-every '$EVAL_EVERY' \
    --probe psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    --manual-probe psm-model/data/probes/direct_probes.jsonl \
    --abort-after-step '${ABORT_AFTER_STEP:-60000}' \
    --collapse-threshold 0.90 \
    2>&1 | tee '$TRAIN_LOG'
  echo done > '$TRAIN_DONE'
"

sleep 8
tmux ls 2>/dev/null || true
head -5 "$TRAIN_LOG" 2>/dev/null || true
tail -8 "$TRAIN_LOG" 2>/dev/null || true
pgrep -af 'psm_model.train' || echo "WARN: no train process yet"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo "=== gate5 train tmux started (detached) ==="
