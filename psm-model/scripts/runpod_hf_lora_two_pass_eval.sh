#!/bin/bash
# Two-pass prod eval: binary gate adapter + minimal_extract adapter.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-krishnach7262/psm-prod-memory-hf}"
MODEL_KEY="${HF_MODEL_KEY:-qwen0.5b}"
BINARY_DIR="${HF_BINARY_ADAPTER_DIR:-psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-${MODEL_KEY}/adapter}"
EXTRACT_DIR="${HF_EXTRACT_ADAPTER_DIR:-psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-${MODEL_KEY}/adapter}"
EVAL_OUT="${HF_EVAL_OUT:-psm-model/prod-memory/results/hf-prod-v5k-two-pass-prod-grounding.json}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== hf two-pass eval $(date -u +%Y-%m-%dT%H:%M:%SZ) binary=$BINARY_DIR extract=$EXTRACT_DIR ==="

cd "$ROOT"
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1

pip install -q torch transformers peft accelerate huggingface_hub sentencepiece 2>/dev/null \
  || pip install -q torch transformers peft accelerate huggingface_hub sentencepiece

_fetch_adapter() {
  local subpath="$1"
  local dest="$2"
  if [[ -f "$dest/adapter_model.safetensors" ]]; then
    return 0
  fi
  hf download "$MODEL_REPO" \
    --repo-type model \
    --include "${subpath}/*" \
    --local-dir "psm-model/prod-memory/checkpoints/_hf_dl" \
    --token "$HF_TOKEN"
  mkdir -p "$dest"
  shopt -s nullglob
  for f in "psm-model/prod-memory/checkpoints/_hf_dl/${subpath}"/*; do
    cp -a "$f" "$dest/"
  done
  shopt -u nullglob
}

_fetch_adapter "${HF_BINARY_ADAPTER_PREFIX:-hf-prod-v5k-gate-${MODEL_KEY}}/adapter" "$BINARY_DIR"
_fetch_adapter "${HF_EXTRACT_ADAPTER_PREFIX:-hf-prod-v5k-extract-${MODEL_KEY}}/adapter" "$EXTRACT_DIR"

python -m prod_memory.eval_hf_two_pass \
  --binary-adapter "$BINARY_DIR" \
  --extract-adapter "$EXTRACT_DIR" \
  --model "$MODEL_KEY" \
  --device cuda \
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
eval_repo_path = os.environ.get("HF_EVAL_REPO_PATH", "eval/hf-prod-v5k-two-pass-prod-grounding.json")
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(path),
    path_in_repo=eval_repo_path,
    repo_id=repo,
    repo_type="model",
    commit_message="upload two-pass prod grounding eval report",
)
print("uploaded eval to", repo, eval_repo_path)
PY
fi
