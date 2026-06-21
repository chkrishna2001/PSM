#!/bin/bash
# Experiment E: scratch 50M extract-only (no gate resume), v3 storage curriculum.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

SUBBU_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
BINARY_RESUME="${EXP_E_BINARY_RESUME:-058000}"
FIXTURE_IDS="${EXP_E_FIXTURE_IDS:-cursor-01-summary,cursor-02-debug,plan-01-handoff}"
SCRATCH_STEPS="${EXP_E_SCRATCH_STEPS:-2500}"
BINARY_STEPS="${EXP_E_BINARY_STEPS:-800}"
CTX="${EXP_E_CONTEXT_LENGTH:-2048}"
OUT_SCRATCH="${EXP_E_OUT:-real-v3-50m-exp-e-scratch}"
OUT_BINARY="${EXP_E_BINARY_OUT:-real-v3-50m-exp-e-binary}"

echo "=== exp-e scratch extract $(date -u +%Y-%m-%dT%H:%M:%SZ) steps=$SCRATCH_STEPS ctx=$CTX ==="

pip install -q huggingface_hub 2>/dev/null || true
mkdir -p psm-model/checkpoints psm-model/prod-memory/data psm-model/prod-memory/results psm-model/prod-memory/fixtures

TOK="psm-model/checkpoints/real-v3-50m-full-v2-step-${BINARY_RESUME}.tokenizer.json"
if [[ ! -f "$TOK" ]]; then
  hf download "$SUBBU_REPO" "$TOK" --local-dir . --token "${HF_TOKEN:-}"
fi

if [[ ! -f psm-model/prod-memory/data/prod-extraction-v3.jsonl ]]; then
  hf download "$DATASET_REPO" prod-memory/data/prod-extraction-v3.jsonl \
    --repo-type dataset --local-dir psm-model/prod-memory --token "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" || \
  hf download "$DATASET_REPO" data/prod-extraction-v3.jsonl \
    --repo-type dataset --local-dir . --token "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" 2>/dev/null || true
fi

export EXP_E_FIXTURE_IDS="$FIXTURE_IDS"
python3 - <<PY
import os
from pathlib import Path
from prod_memory.build_prod_extraction_v6_storage_only import build_prod_extraction_v6_storage_only
from prod_memory.build_binary_fixture_rows import write_binary_fixture_jsonl

root = Path("psm-model/prod-memory/data")
sources = [
    root / "prod-extraction-v3.jsonl",
    Path("data/prod-extraction-v3.jsonl"),
    Path("prod-memory/data/prod-extraction-v3.jsonl"),
]
source = next((p for p in sources if p.exists()), None)
if source is None:
    raise SystemExit("prod-extraction-v3.jsonl not found on pod")
out = root / "exp-e-storage-curriculum.jsonl"
manifest = build_prod_extraction_v6_storage_only(out, source=source)
print("CURRICULUM", out, "rows=", manifest.get("rows"), "storage_p50=", manifest.get("storage_p50_chars"))
ids = [x.strip() for x in os.environ["EXP_E_FIXTURE_IDS"].split(",") if x.strip()]
write_binary_fixture_jsonl(root / "exp-e-binary-fixtures.jsonl", ids)
PY

echo "=== scratch train (no --resume) -> step $SCRATCH_STEPS ==="
python3 -m psm_model.train \
  psm-model/prod-memory/data/exp-e-storage-curriculum.jsonl \
  --out "psm-model/checkpoints/${OUT_SCRATCH}.pt" \
  --tokenizer "$TOK" \
  --steps "$SCRATCH_STEPS" \
  --context-length "$CTX" \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 100 \
  --preset 50m \
  --output-format minimal_extract \
  --sampling action_balanced \
  --action-loss-weight 1.0 \
  --device cuda \
  --save-every 500 \
  --metrics-out "psm-model/checkpoints/${OUT_SCRATCH}.metrics.jsonl" \
  --structural-loss-weight 0.25

SCRATCH_CKPT="psm-model/checkpoints/${OUT_SCRATCH}-step-$(printf '%06d' "$SCRATCH_STEPS").pt"
if [[ ! -f "$SCRATCH_CKPT" ]]; then
  SCRATCH_CKPT="psm-model/checkpoints/${OUT_SCRATCH}.pt"
fi

if [[ ! -f psm-model/prod-memory/fixtures/cases.json ]]; then
  hf download "$DATASET_REPO" prod-memory/fixtures/cases.json \
    --repo-type dataset --local-dir . --token "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" 2>/dev/null || true
  cp -f prod-memory/fixtures/cases.json psm-model/prod-memory/fixtures/cases.json 2>/dev/null || true
fi

EVAL_EXTRACT="psm-model/prod-memory/results/exp-e-extract-${SCRATCH_STEPS}.json"
python3 -m prod_memory.eval_grounding \
  --checkpoint "$SCRATCH_CKPT" \
  --checkpoint-label "exp-e-scratch" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --fixture-ids "$FIXTURE_IDS" \
  --out "$EVAL_EXTRACT" \
  --output-format minimal_extract \
  --raw-input \
  --device cuda \
  --max-new-tokens 128

echo "=== binary train for two-pass eval -> step $((10#$BINARY_RESUME + 10#$BINARY_STEPS)) ==="
BIN_END=$((10#$BINARY_RESUME + 10#$BINARY_STEPS))
python3 -m psm_model.train \
  psm-model/prod-memory/data/exp-e-binary-fixtures.jsonl \
  --out "psm-model/checkpoints/${OUT_BINARY}.pt" \
  --resume "psm-model/checkpoints/real-v3-50m-full-v2-step-${BINARY_RESUME}.pt" \
  --tokenizer "$TOK" \
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

BINARY_CKPT="psm-model/checkpoints/${OUT_BINARY}-step-$(printf '%06d' "$BIN_END").pt"
[[ -f "$BINARY_CKPT" ]] || BINARY_CKPT="psm-model/checkpoints/${OUT_BINARY}.pt"

EVAL_TWOPASS="psm-model/prod-memory/results/exp-e-two-pass-${SCRATCH_STEPS}.json"
python3 -m prod_memory.eval_two_pass \
  --binary-checkpoint "$BINARY_CKPT" \
  --extract-checkpoint "$SCRATCH_CKPT" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --fixture-ids "$FIXTURE_IDS" \
  --out "$EVAL_TWOPASS" \
  --raw-input \
  --device cuda

python3 - <<PY
import json
from pathlib import Path
want = {x.strip() for x in "${FIXTURE_IDS}".split(",") if x.strip()}
ext = json.loads(Path("$EVAL_EXTRACT").read_text(encoding="utf-8"))
tp = json.loads(Path("$EVAL_TWOPASS").read_text(encoding="utf-8"))
ext_hits = sum(1 for c in ext["cases"] if c.get("effective_stored"))
tp_cls = tp["aggregate"]["classify_match"]
tp_grd = tp["aggregate"]["effective_stored"]
print("=== EXP_E_EVAL ===")
print(json.dumps({
    "scratch_steps": $SCRATCH_STEPS,
    "extract_only_trained": f"{ext_hits}/{len(want)}",
    "two_pass_classify": f"{tp_cls}/{len(want)}",
    "two_pass_grounded": f"{tp_grd}/{len(want)}",
    "pass": ext_hits == len(want) or tp_grd == len(want),
}, indent=2))
for c in ext["cases"]:
    print("extract_only", json.dumps({"id": c["id"], "effective_stored": c.get("effective_stored"), "memory_content": c.get("memory_content"), "action": c.get("action")}))
for c in tp["cases"]:
    print("two_pass", json.dumps({"id": c["id"], "classify_match": c.get("classify_match"), "effective_stored": c.get("effective_stored"), "extract_output": c.get("extract_output")}))
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi(token=os.environ["HF_TOKEN"])
for path in [Path("$EVAL_EXTRACT"), Path("$EVAL_TWOPASS")]:
    if path.exists():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.as_posix(), repo_id=repo, repo_type="model",
                        commit_message=f"exp-e {path.name}")
        print("uploaded", path.name)
PY
fi

echo "=== exp-e scratch extract done ==="
