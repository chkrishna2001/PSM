#!/bin/bash
# Experiment C: two-stage overfit — binary classify then minimal extraction (3 fixtures).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
RESUME_STEP="${EXP_C_RESUME_STEP:-058000}"
STEM="${EXP_C_STEM:-real-v3-50m-full-v2}"
STAGE1_STEM="${EXP_C_STAGE1_STEM:-real-v3-50m-exp-c-binary}"
STAGE2_STEM="${EXP_C_STAGE2_STEM:-real-v3-50m-exp-c-minimal}"
FIXTURE_IDS="${EXP_C_FIXTURE_IDS:-cursor-01-summary,cursor-02-debug,plan-01-handoff}"
STAGE_STEPS="${EXP_C_STAGE_STEPS:-800}"
STAGE1_LR="${EXP_C_STAGE1_LR:-3e-5}"
STAGE2_LR="${EXP_C_STAGE2_LR:-1e-5}"
CTX="${EXP_C_CONTEXT_LENGTH:-2048}"

echo "=== exp-c two-stage $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$RESUME_STEP stages=${STAGE_STEPS}+${STAGE_STEPS} ctx=$CTX ==="

pip install -q huggingface_hub 2>/dev/null || true
mkdir -p psm-model/checkpoints psm-model/prod-memory/data psm-model/prod-memory/results psm-model/prod-memory/fixtures

for ext in pt tokenizer.json meta.json; do
  rel="psm-model/checkpoints/${STEM}-step-${RESUME_STEP}.${ext}"
  if [[ ! -f "$rel" ]]; then
    hf download "$MODEL_REPO" "$rel" --local-dir . --token "${HF_TOKEN:-}"
  fi
done

if [[ ! -f psm-model/prod-memory/fixtures/cases.json ]]; then
  hf download "$DATASET_REPO" prod-memory/fixtures/cases.json \
    --repo-type dataset --local-dir . --token "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" 2>/dev/null || true
  cp -f prod-memory/fixtures/cases.json psm-model/prod-memory/fixtures/cases.json 2>/dev/null || true
fi

python3 - <<PY
from pathlib import Path
from prod_memory.build_binary_fixture_rows import write_binary_fixture_jsonl
from prod_memory.build_minimal_fixture_rows import write_minimal_fixture_jsonl
ids = [x.strip() for x in "${FIXTURE_IDS}".split(",") if x.strip()]
b = Path("psm-model/prod-memory/data/exp-c-binary-fixtures.jsonl")
m = Path("psm-model/prod-memory/data/exp-c-minimal-fixtures.jsonl")
print(f"WROTE binary rows={write_binary_fixture_jsonl(b, ids)}")
print(f"WROTE minimal rows={write_minimal_fixture_jsonl(m, ids)}")
PY

RESUME_BASE="psm-model/checkpoints/${STEM}-step-${RESUME_STEP}.pt"
TOK_BASE="psm-model/checkpoints/${STEM}-step-${RESUME_STEP}.tokenizer.json"
BASE_NUM=$((10#$RESUME_STEP))
STAGE1_END=$((BASE_NUM + 10#$STAGE_STEPS))

echo "=== stage 1 binary classify -> step $(printf '%06d' "$STAGE1_END") ==="
python3 -m psm_model.train \
  psm-model/prod-memory/data/exp-c-binary-fixtures.jsonl \
  --out "psm-model/checkpoints/${STAGE1_STEM}.pt" \
  --resume "$RESUME_BASE" \
  --tokenizer "$TOK_BASE" \
  --steps "$STAGE1_END" \
  --context-length "$CTX" \
  --batch-size 1 \
  --learning-rate "$STAGE1_LR" \
  --min-learning-rate "$(python3 -c "print(float('$STAGE1_LR')/5)")" \
  --warmup-steps 20 \
  --preset 50m \
  --output-format binary \
  --sampling random \
  --action-loss-weight 1.0 \
  --device cuda \
  --save-every 400 \
  --metrics-out "psm-model/checkpoints/${STAGE1_STEM}.metrics.jsonl" \
  --structural-loss-weight 0.25

STAGE1_STEP=$(printf '%06d' "$STAGE1_END")
STAGE1_CKPT="psm-model/checkpoints/${STAGE1_STEM}-step-${STAGE1_STEP}.pt"
if [[ ! -f "$STAGE1_CKPT" ]]; then
  STAGE1_CKPT="psm-model/checkpoints/${STAGE1_STEM}.pt"
fi
STAGE1_TOK="${STAGE1_CKPT%.pt}.tokenizer.json"

STAGE2_END=$((STAGE1_END + 10#$STAGE_STEPS))
echo "=== stage 2 minimal extraction -> step $(printf '%06d' "$STAGE2_END") ==="
python3 -m psm_model.train \
  psm-model/prod-memory/data/exp-c-minimal-fixtures.jsonl \
  --out "psm-model/checkpoints/${STAGE2_STEM}.pt" \
  --resume "$STAGE1_CKPT" \
  --tokenizer "$STAGE1_TOK" \
  --steps "$STAGE2_END" \
  --context-length "$CTX" \
  --batch-size 1 \
  --learning-rate "$STAGE2_LR" \
  --min-learning-rate "$(python3 -c "print(float('$STAGE2_LR')/5)")" \
  --warmup-steps 20 \
  --preset 50m \
  --output-format minimal \
  --sampling random \
  --action-loss-weight 1.0 \
  --device cuda \
  --save-every 400 \
  --metrics-out "psm-model/checkpoints/${STAGE2_STEM}.metrics.jsonl" \
  --structural-loss-weight 0.25

FINAL_STEP=$(printf '%06d' "$STAGE2_END")
FINAL_CKPT="psm-model/checkpoints/${STAGE2_STEM}-step-${FINAL_STEP}.pt"
if [[ ! -f "$FINAL_CKPT" ]]; then
  FINAL_CKPT="psm-model/checkpoints/${STAGE2_STEM}.pt"
fi

CLASSIFY_OUT="psm-model/prod-memory/results/exp-c-classify-${FINAL_STEP}.json"
GROUND_OUT="psm-model/prod-memory/results/exp-c-grounding-${FINAL_STEP}.json"
GROUND_ALL="psm-model/prod-memory/results/exp-c-grounding-all-${FINAL_STEP}.json"

python3 -m prod_memory.eval_classify \
  --checkpoint "$FINAL_CKPT" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --fixture-ids "$FIXTURE_IDS" \
  --out "$CLASSIFY_OUT" \
  --output-format minimal \
  --raw-input \
  --device cuda \
  --max-new-tokens 128

python3 -m prod_memory.eval_grounding \
  --checkpoint "$FINAL_CKPT" \
  --checkpoint-label "exp-c-${FINAL_STEP}" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --fixture-ids "$FIXTURE_IDS" \
  --out "$GROUND_OUT" \
  --output-format minimal \
  --raw-input \
  --device cuda \
  --max-new-tokens 128

python3 -m prod_memory.eval_grounding \
  --checkpoint "$FINAL_CKPT" \
  --checkpoint-label "exp-c-all-${FINAL_STEP}" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --out "$GROUND_ALL" \
  --output-format minimal \
  --raw-input \
  --device cuda \
  --max-new-tokens 128

python3 - <<PY
import json
from pathlib import Path
want = {x.strip() for x in "${FIXTURE_IDS}".split(",") if x.strip()}
cls = json.loads(Path("$CLASSIFY_OUT").read_text(encoding="utf-8"))
grd = json.loads(Path("$GROUND_OUT").read_text(encoding="utf-8"))
allg = json.loads(Path("$GROUND_ALL").read_text(encoding="utf-8"))
classify_hits = sum(1 for c in cls["cases"] if c.get("classify_match"))
ground_hits = sum(1 for c in grd["cases"] if c.get("effective_stored"))
print("=== EXP_C_EVAL ===")
print(json.dumps({
    "step": "$FINAL_STEP",
    "classify_match": f"{classify_hits}/{len(want)}",
    "effective_stored_trained": f"{ground_hits}/{len(want)}",
    "effective_stored_all": allg["aggregate"]["effective_stored"],
    "pass": classify_hits == len(want) and ground_hits == len(want),
}, indent=2))
for c in grd["cases"]:
    print(json.dumps({
        "fixture": c["id"],
        "classify_match": next((x["classify_match"] for x in cls["cases"] if x["id"] == c["id"]), None),
        "effective_stored": c.get("effective_stored"),
        "memory_content": c.get("memory_content"),
    }))
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi(token=os.environ["HF_TOKEN"])
for path in Path("psm-model/prod-memory/results").glob("exp-c-*-${FINAL_STEP}.json"):
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path.as_posix(), repo_id=repo, repo_type="model",
                    commit_message=f"exp-c {path.name}")
    print("uploaded", path.name)
PY
fi

echo "=== exp-c two-stage done ==="
