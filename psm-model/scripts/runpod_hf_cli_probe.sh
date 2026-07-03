#!/bin/bash
# CLI-shape LoCoMo probe on pod GPU (adapter already on volume).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1

ADAPTER_DIR="${HF_ADAPTER_DIR:-psm-model/prod-memory/checkpoints/hf-prod-v5n-dpo-qwen0.5b/adapter}"
OUT="${HF_PROBE_OUT:-benchmark/locomo/results/probe-locomo-hf-storage-v5n-dpo-cli-shape-json-cuda.jsonl}"
OUTPUT_FORMAT="${HF_OUTPUT_FORMAT:-json}"

pip install -q torch transformers peft accelerate huggingface_hub sentencepiece 2>/dev/null \
  || pip install -q torch transformers peft accelerate huggingface_hub sentencepiece

mkdir -p benchmark/locomo/data "$(dirname "$OUT")"
DATA="benchmark/locomo/data/locomo10.json"
if [[ ! -f "$DATA" ]]; then
  curl -fsSL -o "$DATA" "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
fi

test -f "$ADAPTER_DIR/adapter_model.safetensors"

python psm-model/scripts/probe_locomo_hf_storage_cli_shape.py \
  --adapter-dir "$ADAPTER_DIR" \
  --device cuda \
  --output-format "$OUTPUT_FORMAT" \
  --out "$OUT"

test -f "$OUT"
echo "probe written: $OUT ($(wc -c < "$OUT") bytes)"
