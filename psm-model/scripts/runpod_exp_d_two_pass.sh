#!/bin/bash
# Experiment D: separate binary + extract checkpoints, two-pass inference at eval.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

SUBBU_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
LEGACY_REPO="${PSM_HF_LEGACY_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
BINARY_RESUME="${EXP_D_BINARY_RESUME:-058000}"
EXTRACT_RESUME="${EXP_D_EXTRACT_RESUME:-032000}"
BINARY_STEM="${EXP_D_BINARY_STEM:-real-v3-50m-full-v2}"
EXTRACT_STEM="${EXP_D_EXTRACT_STEM:-real-v3-50m-full-v2}"
BINARY_OUT="${EXP_D_BINARY_OUT:-real-v3-50m-exp-d-binary}"
EXTRACT_OUT="${EXP_D_EXTRACT_OUT:-real-v3-50m-exp-d-extract}"
FIXTURE_IDS="${EXP_D_FIXTURE_IDS:-cursor-01-summary,cursor-02-debug,plan-01-handoff}"
BINARY_STEPS="${EXP_D_BINARY_STEPS:-800}"
EXTRACT_STEPS="${EXP_D_EXTRACT_STEPS:-1600}"
CTX="${EXP_D_CONTEXT_LENGTH:-2048}"

echo "=== exp-d two-pass $(date -u +%Y-%m-%dT%H:%M:%SZ) binary=${BINARY_RESUME}+${BINARY_STEPS} extract=${EXTRACT_RESUME}+${EXTRACT_STEPS} ==="

pip install -q huggingface_hub 2>/dev/null || true
mkdir -p psm-model/checkpoints psm-model/prod-memory/data psm-model/prod-memory/results psm-model/prod-memory/fixtures

download_ckpt() {
  local repo="$1" stem="$2" step="$3" token="$4"
  for ext in pt tokenizer.json meta.json; do
    rel="psm-model/checkpoints/${stem}-step-${step}.${ext}"
    if [[ ! -f "$rel" ]]; then
      hf download "$repo" "$rel" --local-dir . --token "$token"
    fi
  done
}

download_ckpt "$SUBBU_REPO" "$BINARY_STEM" "$BINARY_RESUME" "${HF_TOKEN:-}"
download_ckpt "$LEGACY_REPO" "$EXTRACT_STEM" "$EXTRACT_RESUME" "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}"

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
print(f"WROTE binary rows={write_binary_fixture_jsonl(Path('psm-model/prod-memory/data/exp-d-binary-fixtures.jsonl'), ids)}")
print(f"WROTE extract rows={write_minimal_fixture_jsonl(Path('psm-model/prod-memory/data/exp-d-extract-fixtures.jsonl'), ids)}")
PY

# Train binary classifier (independent checkpoint)
BIN_BASE=$((10#$BINARY_RESUME))
BIN_END=$((BIN_BASE + 10#$BINARY_STEPS))
echo "=== train binary -> $(printf '%06d' "$BIN_END") ==="
python3 -m psm_model.train \
  psm-model/prod-memory/data/exp-d-binary-fixtures.jsonl \
  --out "psm-model/checkpoints/${BINARY_OUT}.pt" \
  --resume "psm-model/checkpoints/${BINARY_STEM}-step-${BINARY_RESUME}.pt" \
  --tokenizer "psm-model/checkpoints/${BINARY_STEM}-step-${BINARY_RESUME}.tokenizer.json" \
  --steps "$BIN_END" \
  --context-length "$CTX" \
  --batch-size 1 \
  --learning-rate 3e-5 \
  --min-learning-rate 6e-6 \
  --warmup-steps 20 \
  --preset 50m \
  --output-format binary \
  --sampling random \
  --action-loss-weight 1.0 \
  --device cuda \
  --save-every 400 \
  --structural-loss-weight 0.25

BINARY_STEP=$(printf '%06d' "$BIN_END")
BINARY_CKPT="psm-model/checkpoints/${BINARY_OUT}-step-${BINARY_STEP}.pt"
[[ -f "$BINARY_CKPT" ]] || BINARY_CKPT="psm-model/checkpoints/${BINARY_OUT}.pt"

# Train extract model from legacy donor (never touches binary weights file)
EXT_BASE=$((10#$EXTRACT_RESUME))
EXT_END=$((EXT_BASE + 10#$EXTRACT_STEPS))
echo "=== train extract -> $(printf '%06d' "$EXT_END") ==="
python3 -m psm_model.train \
  psm-model/prod-memory/data/exp-d-extract-fixtures.jsonl \
  --out "psm-model/checkpoints/${EXTRACT_OUT}.pt" \
  --resume "psm-model/checkpoints/${EXTRACT_STEM}-step-${EXTRACT_RESUME}.pt" \
  --tokenizer "psm-model/checkpoints/${EXTRACT_STEM}-step-${EXTRACT_RESUME}.tokenizer.json" \
  --steps "$EXT_END" \
  --context-length "$CTX" \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --min-learning-rate 4e-6 \
  --warmup-steps 40 \
  --preset 50m \
  --output-format minimal_extract \
  --sampling random \
  --action-loss-weight 1.0 \
  --device cuda \
  --save-every 400 \
  --structural-loss-weight 0.25

EXTRACT_STEP=$(printf '%06d' "$EXT_END")
EXTRACT_CKPT="psm-model/checkpoints/${EXTRACT_OUT}-step-${EXTRACT_STEP}.pt"
[[ -f "$EXTRACT_CKPT" ]] || EXTRACT_CKPT="psm-model/checkpoints/${EXTRACT_OUT}.pt"

EVAL_TRAINED="psm-model/prod-memory/results/exp-d-two-pass-${BINARY_STEP}-${EXTRACT_STEP}.json"
EVAL_ALL="psm-model/prod-memory/results/exp-d-two-pass-all-${BINARY_STEP}-${EXTRACT_STEP}.json"

python3 -m prod_memory.eval_two_pass \
  --binary-checkpoint "$BINARY_CKPT" \
  --extract-checkpoint "$EXTRACT_CKPT" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --fixture-ids "$FIXTURE_IDS" \
  --out "$EVAL_TRAINED" \
  --raw-input \
  --device cuda

python3 -m prod_memory.eval_two_pass \
  --binary-checkpoint "$BINARY_CKPT" \
  --extract-checkpoint "$EXTRACT_CKPT" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --out "$EVAL_ALL" \
  --raw-input \
  --device cuda

python3 - <<PY
import json
from pathlib import Path
want = {x.strip() for x in "${FIXTURE_IDS}".split(",") if x.strip()}
trained = json.loads(Path("$EVAL_TRAINED").read_text(encoding="utf-8"))
allr = json.loads(Path("$EVAL_ALL").read_text(encoding="utf-8"))
a = trained["aggregate"]
print("=== EXP_D_EVAL ===")
print(json.dumps({
    "binary_step": "$BINARY_STEP",
    "extract_step": "$EXTRACT_STEP",
    "classify_match": f"{a['classify_match']}/{len(want)}",
    "effective_stored_trained": f"{a['effective_stored']}/{len(want)}",
    "effective_stored_all": allr["aggregate"]["effective_stored"],
    "pass": a["classify_match"] == len(want) and a["effective_stored"] == len(want),
}, indent=2))
for c in trained["cases"]:
    print(json.dumps(c))
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi(token=os.environ["HF_TOKEN"])
for path in [Path("$EVAL_TRAINED"), Path("$EVAL_ALL")]:
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path.as_posix(), repo_id=repo, repo_type="model",
                    commit_message=f"exp-d {path.name}")
    print("uploaded", path.name)
PY
fi

echo "=== exp-d two-pass done ==="
