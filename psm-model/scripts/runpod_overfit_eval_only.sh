#!/bin/bash
# Re-eval overfit checkpoint after train (058000 + 800 steps).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

FINAL_STEP="${OVERFIT_FINAL_STEP:-058800}"
CKPT="psm-model/checkpoints/real-v3-50m-overfit-fixture-step-${FINAL_STEP}.pt"
if [[ ! -f "$CKPT" ]]; then
  CKPT="psm-model/checkpoints/real-v3-50m-overfit-fixture.pt"
fi
echo "=== overfit eval ckpt=$CKPT ==="

OUT="psm-model/prod-memory/results/prod-grounding-overfit-${FINAL_STEP}.json"
python3 -m prod_memory.eval_grounding \
  --checkpoint "$CKPT" \
  --checkpoint-label "overfit-${FINAL_STEP}" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --out "$OUT" \
  --device cuda \
  --max-new-tokens 384

python3 - <<PY
import json
from pathlib import Path
r = json.loads(Path("$OUT").read_text(encoding="utf-8"))
a = r["aggregate"]
print("=== OVERFIT_EVAL ===")
print(json.dumps({"step": "$FINAL_STEP", "aggregate": a}, indent=2))
want = {"cursor-01-summary", "cursor-02-debug", "plan-01-handoff"}
for c in r.get("cases", []):
    if c.get("id") in want:
        print(json.dumps({"fixture": c["id"], "effective_stored": c["effective_stored"],
                          "parse_valid": c.get("parse_valid"), "action": c.get("action")}))
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi(token=os.environ["HF_TOKEN"])
path = Path("$OUT")
if path.exists():
    api.upload_file(str(path), path.as_posix(), repo_id=repo, repo_type="model",
                    commit_message=f"overfit eval {path.name}")
    print("uploaded", path.name)
PY
fi
