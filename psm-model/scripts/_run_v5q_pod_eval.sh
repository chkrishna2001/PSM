#!/bin/bash
# One-off: prod fixture eval for v5q on the train pod, upload report to HF.
set -euo pipefail
cd /workspace/PSM
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN missing}"
mkdir -p psm-model/prod-memory/results
python3 -m prod_memory.eval_hf_grounding \
  --adapter-dir psm-model/prod-memory/checkpoints/hf-prod-v5q-qwen0.5b/adapter \
  --checkpoint-label hf-prod-v5q-qwen0.5b \
  --output-format json \
  --max-new-tokens 768 \
  --out psm-model/prod-memory/results/hf-prod-v5q-qwen0.5b-prod-grounding.json
python3 - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj="psm-model/prod-memory/results/hf-prod-v5q-qwen0.5b-prod-grounding.json",
    path_in_repo="eval/hf-prod-v5q-qwen0.5b-prod-grounding.json",
    repo_id="krishnach7262/psm-prod-memory-hf",
    repo_type="model",
)
print("uploaded eval report to HF")
PY
