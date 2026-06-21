#!/bin/bash
# Eval prod grounding on legacy chkrishna2001 donor (022800 not on HF; default 032000).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

LEGACY_REPO="${PSM_HF_LEGACY_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
STEM="${DONOR_STEM:-real-v3-50m-full-v2}"
STEP="${LEGACY_DONOR_STEP:-032000}"
TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}"
DEVICE="${PSM_EVAL_DEVICE:-cuda}"

echo "=== legacy donor eval step=$STEP repo=$LEGACY_REPO ==="

for ext in pt tokenizer.json meta.json; do
  rel="psm-model/checkpoints/${STEM}-step-${STEP}.${ext}"
  if [[ ! -f "$rel" ]]; then
    hf download "$LEGACY_REPO" "$rel" --local-dir . --token "$TOKEN"
  fi
done

OUT="psm-model/prod-memory/results/prod-grounding-${STEP}.json"
python3 -m prod_memory.eval_grounding \
  --checkpoint "psm-model/checkpoints/${STEM}-step-${STEP}.pt" \
  --checkpoint-label "$STEP" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --out "$OUT" \
  --device "$DEVICE" \
  --max-new-tokens 384

python3 - <<PY
import json
from pathlib import Path
r = json.loads(Path("$OUT").read_text(encoding="utf-8"))
print(json.dumps({"step": "$STEP", "aggregate": r["aggregate"]}, indent=2))
PY
