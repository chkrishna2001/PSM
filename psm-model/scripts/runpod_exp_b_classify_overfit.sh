#!/bin/bash
# Experiment B: binary store/ignore classify-only overfit (3 fixtures, 800 steps).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
RESUME_STEP="${EXP_B_RESUME_STEP:-058000}"
STEM="${EXP_B_STEM:-real-v3-50m-full-v2}"
OUT_STEM="${EXP_B_OUT_STEM:-real-v3-50m-exp-b-binary}"
FIXTURE_IDS="${EXP_B_FIXTURE_IDS:-cursor-01-summary,cursor-02-debug,plan-01-handoff}"
TARGET_STEPS="${EXP_B_TARGET_STEPS:-800}"
LR="${EXP_B_LR:-3e-5}"
CTX="${EXP_B_CONTEXT_LENGTH:-2048}"

echo "=== exp-b binary classify $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$RESUME_STEP ctx=$CTX fixtures=$FIXTURE_IDS ==="

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
ids = [x.strip() for x in "${FIXTURE_IDS}".split(",") if x.strip()]
out = Path("psm-model/prod-memory/data/exp-b-binary-fixtures.jsonl")
n = write_binary_fixture_jsonl(out, ids)
print(f"WROTE {out} rows={n}")
PY

RESUME="psm-model/checkpoints/${STEM}-step-${RESUME_STEP}.pt"
TOK="psm-model/checkpoints/${STEM}-step-${RESUME_STEP}.tokenizer.json"
RESUME_NUM=$((10#$RESUME_STEP))
TARGET_NUM=$((RESUME_NUM + 10#$TARGET_STEPS))

python3 -m psm_model.train \
  psm-model/prod-memory/data/exp-b-binary-fixtures.jsonl \
  --out "psm-model/checkpoints/${OUT_STEM}.pt" \
  --resume "$RESUME" \
  --tokenizer "$TOK" \
  --steps "$TARGET_NUM" \
  --context-length "$CTX" \
  --batch-size 1 \
  --learning-rate "$LR" \
  --min-learning-rate "$(python3 -c "print(float('$LR')/5)")" \
  --warmup-steps 20 \
  --preset 50m \
  --output-format binary \
  --sampling random \
  --action-loss-weight 1.0 \
  --device cuda \
  --save-every 400 \
  --metrics-out "psm-model/checkpoints/${OUT_STEM}.metrics.jsonl" \
  --structural-loss-weight 0.25

FINAL_STEP=$(printf '%06d' "$TARGET_NUM")
CKPT="psm-model/checkpoints/${OUT_STEM}-step-${FINAL_STEP}.pt"
if [[ ! -f "$CKPT" ]]; then
  cp -f "psm-model/checkpoints/${OUT_STEM}.pt" "$CKPT"
  cp -f "psm-model/checkpoints/${OUT_STEM}.tokenizer.json" "${CKPT%.pt}.tokenizer.json"
  cp -f "psm-model/checkpoints/${OUT_STEM}.meta.json" "${CKPT%.pt}.meta.json" 2>/dev/null || true
fi

EVAL_OUT="psm-model/prod-memory/results/exp-b-binary-${FINAL_STEP}.json"
python3 -m prod_memory.eval_classify \
  --checkpoint "$CKPT" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --fixture-ids "$FIXTURE_IDS" \
  --out "$EVAL_OUT" \
  --output-format binary \
  --raw-input \
  --device cuda \
  --max-new-tokens 32

python3 - <<PY
import json
from pathlib import Path
r = json.loads(Path("$EVAL_OUT").read_text(encoding="utf-8"))
a = r["aggregate"]
want = [x.strip() for x in "${FIXTURE_IDS}".split(",") if x.strip()]
print("=== EXP_B_EVAL ===")
print(json.dumps({"step": "$FINAL_STEP", "aggregate": a, "pass_criterion": f"{len(want)}/{len(want)} classify_match"}, indent=2))
for c in r.get("cases", []):
    print(json.dumps(c))
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi(token=os.environ["HF_TOKEN"])
for path in [Path("$EVAL_OUT"), Path("psm-model/prod-memory/data/exp-b-binary-fixtures.jsonl")]:
    if path.exists():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.as_posix(), repo_id=repo, repo_type="model",
                        commit_message=f"exp-b binary {path.name}")
        print("uploaded", path.name)
PY
fi

echo "=== exp-b binary done ==="
