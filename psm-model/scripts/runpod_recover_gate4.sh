#!/bin/bash
# Recover crashed Gate 4 run: prune corrupt saves, sync all checkpoints to HF, resume training.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
RUN_STEM="${RUN_STEM:-real-v3-50m-full-v2}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
MIN_CKPT_BYTES="${MIN_CKPT_BYTES:-500000000}"
TARGET_STEPS="${TARGET_STEPS:-36000}"
SAVE_EVERY="${SAVE_EVERY:-400}"

cd "$ROOT"
export PYTHONPATH=psm-model/src

echo "=== Gate 4 recover $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
df -h /workspace || true

CKPT_DIR="$ROOT/psm-model/checkpoints"
mapfile -t STEP_FILES < <(ls -1 "$CKPT_DIR"/${RUN_STEM}-step-*.pt 2>/dev/null | sort -t- -k3 -n || true)
if [[ ${#STEP_FILES[@]} -eq 0 ]]; then
  echo "No step checkpoints in $CKPT_DIR" >&2
  exit 1
fi

for path in "${STEP_FILES[@]}"; do
  size=$(stat -c%s "$path" 2>/dev/null || echo 0)
  if [[ "$size" -lt "$MIN_CKPT_BYTES" ]]; then
    echo "Removing corrupt checkpoint ($size bytes): $path"
    rm -f "$path" "${path%.pt}.meta.json" "${path%.pt}.tokenizer.json"
  fi
done

mapfile -t STEP_FILES < <(ls -1 "$CKPT_DIR"/${RUN_STEM}-step-*.pt 2>/dev/null | sort -t- -k3 -n || true)
RESUME_PATH="${STEP_FILES[-1]}"
export RESUME_CHECKPOINT="${RESUME_PATH#$ROOT/}"
export TOKENIZER="${RESUME_PATH%.pt}.tokenizer.json"
export TOKENIZER="${TOKENIZER#$ROOT/}"
export TARGET_STEPS SAVE_EVERY KEEP_LOCAL

echo "resume=$RESUME_CHECKPOINT target=$TARGET_STEPS save_every=$SAVE_EVERY"

bash "$ROOT/psm-model/scripts/runpod_upload_gate4.sh"

export GATE4_CURRICULUM_BUILDER="${GATE4_CURRICULUM_BUILDER:-v1}"
bash "$ROOT/psm-model/scripts/runpod_train_gate4.sh"
