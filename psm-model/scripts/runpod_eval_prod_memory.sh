#!/bin/bash
# Warm pod: prod-memory grounding eval (fixtures + remember path on CUDA).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
DEVICE="${PSM_EVAL_DEVICE:-cuda}"
EVAL_STEP="${EVAL_STEP:-060000}"
RUN_STEM="${RUN_STEM:-real-v3-50m-full-v2-prod-memory}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
COMPARE_BASELINE_STEP="${COMPARE_BASELINE_STEP:-}"

echo "=== prod-memory grounding eval $(date -u +%Y-%m-%dT%H:%M:%SZ) step=$EVAL_STEP device=$DEVICE ==="

if ! command -v hf >/dev/null 2>&1; then
  pip install -q huggingface_hub hf_transfer 2>/dev/null || pip install -q huggingface_hub
fi

mkdir -p psm-model/checkpoints psm-model/prod-memory/results psm-model/prod-memory/fixtures

download_ckpt() {
  local rel="$1"
  if [[ ! -f "$rel" ]]; then
    echo "Downloading $rel from $MODEL_REPO..."
    hf download "$MODEL_REPO" "$rel" --local-dir . --token "${HF_TOKEN:-}"
  fi
}

run_eval() {
  local step="$1"
  local stem="$2"
  local ckpt="psm-model/checkpoints/${stem}-step-${step}.pt"
  for rel in "$ckpt" "${ckpt%.pt}.tokenizer.json" "${ckpt%.pt}.meta.json"; do
    download_ckpt "$rel"
  done
  if [[ ! -f "$ckpt" ]]; then
    echo "Checkpoint missing: $ckpt" >&2
    return 1
  fi
  local out="psm-model/prod-memory/results/prod-grounding-${step}.json"
  local log="/tmp/psm-prod-memory-eval-${step}.log"
  python3 -m prod_memory.eval_grounding \
    --checkpoint "$ckpt" \
    --checkpoint-label "$step" \
    --fixtures psm-model/prod-memory/fixtures/cases.json \
    --out "$out" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    2>&1 | tee "$log"
  echo "PSM_PROD_EVAL_OUT=$out"
  python3 - <<PY
import json
from pathlib import Path
report = json.loads(Path("$out").read_text(encoding="utf-8"))
print(json.dumps({"step": "$step", "aggregate": report["aggregate"], "suites": report["suites"]}, indent=2))
PY
}

if [[ ! -f psm-model/prod-memory/fixtures/cases.json ]]; then
  echo "Downloading prod-memory fixtures from dataset repo..."
  hf download "$DATASET_REPO" \
    prod-memory/fixtures/cases.json \
    --repo-type dataset --local-dir . \
    --token "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" 2>/dev/null || true
  if [[ -f prod-memory/fixtures/cases.json && ! -f psm-model/prod-memory/fixtures/cases.json ]]; then
    cp -f prod-memory/fixtures/cases.json psm-model/prod-memory/fixtures/cases.json
  fi
fi
if [[ ! -f psm-model/prod-memory/fixtures/cases.json ]]; then
  echo "Prod fixtures missing on pod" >&2
  exit 1
fi

python3 - <<'PY'
import sys
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FATAL: CUDA not available", file=sys.stderr)
    sys.exit(1)
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

run_eval "$EVAL_STEP" "$RUN_STEM"

if [[ -n "$COMPARE_BASELINE_STEP" ]]; then
  run_eval "$COMPARE_BASELINE_STEP" "real-v3-50m-full-v2"
fi

python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi

repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi()
results = Path("psm-model/prod-memory/results")
for path in sorted(results.glob("prod-grounding-*.json")):
    remote = path.as_posix()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo,
        repo_type="model",
        commit_message=f"prod-memory grounding eval {path.name}",
    )
    print(f"uploaded_hf {remote}")
PY

echo "=== prod-memory grounding eval done ==="
