#!/bin/bash
# Prod fixture eval for HF LoRA adapter (download from HF, no retrain).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"

MODEL_REPO="${PSM_HF_MODEL_REPO:-krishnach7262/psm-prod-memory-hf}"
MODEL_KEY="${HF_MODEL_KEY:-qwen0.5b}"
ADAPTER_DIR="${HF_ADAPTER_DIR:-psm-model/prod-memory/checkpoints/hf-prod-v1-${MODEL_KEY}/adapter}"
EVAL_OUT="${HF_EVAL_OUT:-psm-model/prod-memory/results/hf-prod-v1-qwen0.5b-prod-grounding.json}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== hf lora eval-only $(date -u +%Y-%m-%dT%H:%M:%SZ) repo=$MODEL_REPO adapter=$ADAPTER_DIR ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 || true

GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
if [[ "${HF_SKIP_CLONE:-0}" != "1" ]]; then
  if [[ ! -d "$ROOT/.git" ]]; then
    if [[ -d "$ROOT" ]]; then
      rm -rf "$ROOT"
    fi
    mkdir -p "$(dirname "$ROOT")"
    git clone --depth 1 "$GIT_URL" "$ROOT"
  fi
fi
cd "$ROOT"
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1

pip install -q torch transformers peft accelerate huggingface_hub 2>/dev/null \
  || pip install -q torch transformers peft accelerate huggingface_hub

mkdir -p "$(dirname "$ADAPTER_DIR")" psm-model/prod-memory/results

if [[ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]]; then
  echo "Downloading adapter from $MODEL_REPO..."
  hf download "$MODEL_REPO" \
    --repo-type model \
    --include "hf-prod-v1-${MODEL_KEY}/*" \
    --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" \
    --token "$HF_TOKEN"
  mkdir -p "$ADAPTER_DIR"
  shopt -s nullglob
  for f in "psm-model/prod-memory/checkpoints/_hf_dl/hf-prod-v1-${MODEL_KEY}"/*; do
    cp -a "$f" "$ADAPTER_DIR/"
  done
  shopt -u nullglob
fi

python -m prod_memory.eval_hf_grounding \
  --adapter-dir "$ADAPTER_DIR" \
  --model "$MODEL_KEY" \
  --device cuda \
  --output-format tagged \
  --checkpoint-label "hf-prod-v1-${MODEL_KEY}" \
  --out "$EVAL_OUT"
export HF_EVAL_OUT="$EVAL_OUT"

test -f "$EVAL_OUT"
echo "eval written: $EVAL_OUT ($(wc -c < "$EVAL_OUT") bytes)"

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<'PY'
import os
from huggingface_hub import HfApi
from pathlib import Path

repo = os.environ.get("PSM_HF_MODEL_REPO", "krishnach7262/psm-prod-memory-hf")
path = Path(os.environ["HF_EVAL_OUT"])
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(path),
    path_in_repo="eval/hf-prod-v1-qwen0.5b-prod-grounding.json",
    repo_id=repo,
    repo_type="model",
    commit_message="upload prod grounding eval report",
)
print("uploaded eval to", repo, "eval/hf-prod-v1-qwen0.5b-prod-grounding.json")
PY
fi
