#!/bin/bash
# Prune local Gate 4 checkpoints only (no HF). Use when HF storage is full.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
RUN_STEM="${RUN_STEM:-real-v3-50m-full-v2}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
MIN_CKPT_BYTES="${MIN_CKPT_BYTES:-500000000}"

CKPT_DIR="$ROOT/psm-model/checkpoints"
for path in "$CKPT_DIR"/${RUN_STEM}-step-*.pt; do
  [[ -f "$path" ]] || continue
  size=$(stat -c%s "$path" 2>/dev/null || echo 0)
  if [[ "$size" -lt "$MIN_CKPT_BYTES" ]]; then
    echo "Removing corrupt checkpoint ($size bytes): $path"
    rm -f "$path" "${path%.pt}.meta.json" "${path%.pt}.tokenizer.json"
  fi
done

mapfile -t STEP_FILES < <(ls -1 "$CKPT_DIR"/${RUN_STEM}-step-*.pt 2>/dev/null | sort -t- -k3 -n || true)
if [[ ${#STEP_FILES[@]} -le "$KEEP_LOCAL" ]]; then
  echo "Nothing to prune (${#STEP_FILES[@]} <= keep-local=$KEEP_LOCAL)"
else
  delete_count=$((${#STEP_FILES[@]} - KEEP_LOCAL))
  for ((i = 0; i < delete_count; i++)); do
    path="${STEP_FILES[$i]}"
    echo "Pruning local: $path"
    rm -f "$path" "${path%.pt}.meta.json" "${path%.pt}.tokenizer.json"
  done
fi

df -h /workspace || true
LATEST=$(ls -1 "$CKPT_DIR"/${RUN_STEM}-step-*.pt 2>/dev/null | sort -t- -k3 -n | tail -1 || true)
if [[ -n "$LATEST" ]]; then
  echo "RESUME_CHECKPOINT=${LATEST#$ROOT/}"
  echo "RESUME_STEP=$(basename "$LATEST" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"
fi
