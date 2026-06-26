#!/bin/bash
# HF LoRA LoCoMo ingest smoke (n=25 default).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1

MODEL_REPO="${PSM_HF_MODEL_REPO:-krishnach7262/psm-prod-memory-hf}"
MODEL_KEY="${HF_MODEL_KEY:-qwen0.5b}"
ADAPTER_DIR="${HF_ADAPTER_DIR:-psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter}"
ADAPTER_PREFIX="${HF_ADAPTER_PREFIX:-hf-prod-v5c-qwen0.5b}"
LIMIT="${HF_LOCOMO_LIMIT:-25}"
OUT="${HF_LOCOMO_OUT:-psm-model/prod-memory/results/hf-prod-v5c-locomo-n25.json}"
OUTPUT_FORMAT="${HF_OUTPUT_FORMAT:-minimal}"
HF_TOKEN="${HF_TOKEN:-}"

pip install -q torch transformers peft accelerate huggingface_hub sentencepiece 2>/dev/null \
  || pip install -q torch transformers peft accelerate huggingface_hub sentencepiece

BINARY_DIR="${HF_BINARY_ADAPTER_DIR:-}"
BINARY_PREFIX="${HF_BINARY_ADAPTER_PREFIX:-}"

mkdir -p benchmark/locomo/data "$(dirname "$ADAPTER_DIR")" psm-model/prod-memory/results

if [[ -n "$BINARY_DIR" && ! -f "$BINARY_DIR/adapter_model.safetensors" ]]; then
  echo "Downloading binary adapter from $MODEL_REPO/$BINARY_PREFIX..."
  hf download "$MODEL_REPO" \
    --repo-type model \
    --include "${BINARY_PREFIX}/adapter/*" \
    --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" \
    --token "$HF_TOKEN"
  mkdir -p "$BINARY_DIR"
  shopt -s nullglob
  for f in "psm-model/prod-memory/checkpoints/_hf_dl/${BINARY_PREFIX}/adapter"/*; do
    cp -a "$f" "$BINARY_DIR/"
  done
  shopt -u nullglob
fi

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "Downloading adapter from $MODEL_REPO/$ADAPTER_PREFIX..."
  hf download "$MODEL_REPO" \
    --repo-type model \
    --include "${ADAPTER_PREFIX}/adapter/*" \
    --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" \
    --token "$HF_TOKEN"
  mkdir -p "$ADAPTER_DIR"
  shopt -s nullglob
  for f in "psm-model/prod-memory/checkpoints/_hf_dl/${ADAPTER_PREFIX}/adapter"/*; do
    cp -a "$f" "$ADAPTER_DIR/"
  done
  shopt -u nullglob
fi

python -m prod_memory.eval_hf_locomo \
  --adapter-dir "$ADAPTER_DIR" \
  ${HF_BINARY_ADAPTER_DIR:+--binary-adapter "$HF_BINARY_ADAPTER_DIR"} \
  --model "$MODEL_KEY" \
  --limit "$LIMIT" \
  --output-format "$OUTPUT_FORMAT" \
  --device cuda \
  --out "$OUT" \
  --label "${HF_CHECKPOINT_LABEL:-hf-prod-v5c-qwen0.5b}-locomo-n${LIMIT}"
LOCOMO_RC=$?

test -f "$OUT"
echo "locomo smoke written: $OUT ($(wc -c < "$OUT") bytes)"
exit "$LOCOMO_RC"
