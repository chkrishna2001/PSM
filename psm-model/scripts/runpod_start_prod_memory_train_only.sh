#!/bin/bash
# Warm pod: prod-memory v2 smoke train from 058000, exit after tmux start.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
DEVICE="${PSM_TRAIN_DEVICE:-cuda}"
RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2-step-058000.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-full-v2-step-058000.tokenizer.json}"
CURRICULUM="${PROD_CURRICULUM:-psm-model/prod-memory/data/prod-extraction-v2.jsonl}"
TARGET_STEPS="${TARGET_STEPS:-60000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
SAVE_EVERY="${SAVE_EVERY:-200}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-120}"
UPLOAD_ALL="${UPLOAD_ALL:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
RESUME_STEP="$(basename "$RESUME" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"
OUT_STEM="${OUT_STEM:-real-v3-50m-full-v2-prod-memory}"

echo "=== start prod-memory train-only $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$RESUME_STEP target=$TARGET_STEPS ctx=$CONTEXT_LENGTH ==="

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
chmod +x psm-model/scripts/runpod_upload_gate4.sh 2>/dev/null || true
mkdir -p psm-model/checkpoints psm-model/prod-memory/data psm-model/prod-memory/results

download_ckpt() {
  local rel="$1"
  if [[ ! -f "$rel" ]]; then
    echo "Downloading $rel from $MODEL_REPO..."
    # ponytail: pod RUNPOD_SECRET HF_TOKEN is often wrong; always pass --token explicitly
    hf download "$MODEL_REPO" "$rel" --local-dir . --token "${HF_TOKEN:-}"
  fi
}
for rel in "$RESUME" "${RESUME%.pt}.tokenizer.json" "${RESUME%.pt}.meta.json"; do
  download_ckpt "$rel"
done
if [[ ! -f "$RESUME" ]]; then
  echo "Prod-memory resume checkpoint unavailable on HF: $RESUME" >&2
  exit 1
fi

if [[ ! -f "$CURRICULUM" ]]; then
  echo "Downloading prod curriculum from dataset repo $DATASET_REPO..."
  hf download "$DATASET_REPO" \
    prod-memory/prod-extraction-v2.jsonl \
    prod-memory/prod-extraction-v2.manifest.json \
    --repo-type dataset --local-dir . \
    --token "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}"
  if [[ ! -f "$CURRICULUM" && -f prod-memory/prod-extraction-v2.jsonl ]]; then
    cp -f prod-memory/prod-extraction-v2.jsonl "$CURRICULUM"
  fi
fi
if [[ ! -f "$CURRICULUM" ]]; then
  echo "Fallback: downloading prod curriculum from model repo $MODEL_REPO..."
  hf download "$MODEL_REPO" \
    prod-memory/prod-extraction-v2.jsonl \
    prod-memory/prod-extraction-v2.manifest.json \
    --local-dir . \
    --token "${HF_TOKEN:-}"
  if [[ ! -f "$CURRICULUM" && -f prod-memory/prod-extraction-v2.jsonl ]]; then
    cp -f prod-memory/prod-extraction-v2.jsonl "$CURRICULUM"
  fi
fi
if [[ ! -f "$CURRICULUM" ]]; then
  echo "Prod curriculum missing on pod: $CURRICULUM" >&2
  exit 1
fi

export GATE4_PINNED_STEPS="${RESUME_STEP}"
SYNC_CMD="cd '$ROOT' && export HF_TOKEN=\"\${HF_TOKEN:-}\" UPLOAD_ALL=${UPLOAD_ALL} KEEP_LOCAL=$KEEP_LOCAL GATE4_PINNED_STEPS='$RESUME_STEP' PSM_HF_MODEL_REPO='$MODEL_REPO' && while true; do sleep $SYNC_INTERVAL_SEC; bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/prod-memory-sync.log; done"
tmux kill-session -t psm-prod-memory-sync 2>/dev/null || true
tmux new-session -d -s psm-prod-memory-sync bash -lc "$(printf '%q' "$SYNC_CMD")"
echo "HF sync: tmux psm-prod-memory-sync every ${SYNC_INTERVAL_SEC}s"

TRAIN_LOG="/tmp/psm-prod-memory-train.log"
TRAIN_DONE="/tmp/psm-prod-memory.done"
rm -f "$TRAIN_DONE"
tmux kill-session -t psm-prod-memory 2>/dev/null || true
tmux new-session -d -s psm-prod-memory bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src:psm-model/prod-memory
  python3 -m psm_model.train \
    '$CURRICULUM' \
    --out psm-model/checkpoints/${OUT_STEM}.pt \
    --resume '$RESUME' \
    --tokenizer '$TOK' \
    --steps '$TARGET_STEPS' \
    --context-length '$CONTEXT_LENGTH' \
    --batch-size '$BATCH_SIZE' \
    --learning-rate '$LEARNING_RATE' \
    --min-learning-rate '$MIN_LEARNING_RATE' \
    --warmup-steps '$WARMUP_STEPS' \
    --preset 50m \
    --output-format tagged \
    --sampling random \
    --device '$DEVICE' \
    --save-every '$SAVE_EVERY' \
    --metrics-out psm-model/checkpoints/${OUT_STEM}.metrics.jsonl \
    --structural-loss-weight '${STRUCTURAL_LOSS_WEIGHT:-1}' \
    --abort-after-step '${ABORT_AFTER_STEP:-65000}' \
    2>&1 | tee '$TRAIN_LOG'
  echo done > '$TRAIN_DONE'
"

sleep 8
tmux ls 2>/dev/null || true
head -5 "$TRAIN_LOG" 2>/dev/null || true
tail -8 "$TRAIN_LOG" 2>/dev/null || true
pgrep -af 'psm_model.train' || echo "WARN: no train process yet"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo "=== prod-memory train tmux started (detached) ==="
