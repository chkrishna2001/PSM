#!/bin/bash
# Step 2: 50M capacity — overfit prod fixture row(s), then re-eval all fixtures.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
RESUME_STEP="${OVERFIT_RESUME_STEP:-058000}"
STEM="${OVERFIT_STEM:-real-v3-50m-full-v2}"
OUT_STEM="${OVERFIT_OUT_STEM:-real-v3-50m-overfit-fixture}"
FIXTURE_IDS="${OVERFIT_FIXTURE_IDS:-cursor-01-summary,cursor-02-debug,plan-01-handoff}"
TARGET_STEPS="${OVERFIT_TARGET_STEPS:-800}"
LR="${OVERFIT_LR:-3e-5}"
ACTION_W="${ACTION_LOSS_WEIGHT:-1.0}"

echo "=== overfit fixture $(date -u +%Y-%m-%dT%H:%M:%SZ) resume=$RESUME_STEP fixtures=$FIXTURE_IDS steps=$TARGET_STEPS ==="

pip install -q huggingface_hub hf_transfer 2>/dev/null || pip install -q huggingface_hub
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
import json
from pathlib import Path
from prod_memory.curriculum_sources import build_fixture_rows, load_fixture_cases

ids = [x.strip() for x in "${FIXTURE_IDS}".split(",") if x.strip()]
rows = [r for r in build_fixture_rows() if r["id"].replace("fixture-", "") in ids or any(r["id"].endswith(i) for i in ids)]
if not rows:
    # match by case id prefix fixture-{id}
    want = set(ids)
    rows = [r for r in build_fixture_rows() if r["id"] in {f"fixture-{i}" for i in want}]
if not rows:
    raise SystemExit(f"no fixture rows for ids={ids}")

# ponytail: add minimal grounded facts so labels match ship guards
cases = {c["id"]: c for c in load_fixture_cases()}
for row in rows:
    cid = row["id"].replace("fixture-", "")
    case = cases.get(cid)
    if not case or row["expected"].get("action") == "ignore":
        continue
    text = str(case.get("llmResponse") or "")
    keys = [str(k) for k in (case.get("keyTokens") or [])[:3]]
    facts = []
    for i, key in enumerate(keys):
        idx = text.lower().find(key.lower())
        snippet = text[max(0, idx - 20): idx + len(key) + 40].strip() if idx >= 0 else text[:80]
        facts.append({
            "subject": "assistant_text",
            "predicate": f"mentions_{key}",
            "value": key,
            "value_text": key,
            "confidence": 0.9,
            "inference_kind": "explicit",
            "evidence_text": snippet[:200],
        })
    if facts:
        row["expected"]["facts"] = facts

out = Path("psm-model/prod-memory/data/overfit-fixtures.jsonl")
with out.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"WROTE {out} rows={len(rows)}")
PY

RESUME="psm-model/checkpoints/${STEM}-step-${RESUME_STEP}.pt"
TOK="psm-model/checkpoints/${STEM}-step-${RESUME_STEP}.tokenizer.json"
RESUME_NUM=$((10#$RESUME_STEP))
TARGET_NUM=$((RESUME_NUM + 10#$TARGET_STEPS))

python3 -m psm_model.train \
  psm-model/prod-memory/data/overfit-fixtures.jsonl \
  --out "psm-model/checkpoints/${OUT_STEM}.pt" \
  --resume "$RESUME" \
  --tokenizer "$TOK" \
  --steps "$TARGET_NUM" \
  --context-length 4096 \
  --batch-size 1 \
  --learning-rate "$LR" \
  --min-learning-rate "$(python3 -c "print(float('$LR')/5)")" \
  --warmup-steps 20 \
  --preset 50m \
  --output-format tagged \
  --sampling random \
  --action-loss-weight "$ACTION_W" \
  --device cuda \
  --save-every 400 \
  --metrics-out "psm-model/checkpoints/${OUT_STEM}.metrics.jsonl" \
  --structural-loss-weight 1

FINAL_STEP=$(printf '%06d' "$TARGET_NUM")
CKPT="psm-model/checkpoints/${OUT_STEM}-step-${FINAL_STEP}.pt"
if [[ ! -f "$CKPT" ]]; then
  cp -f "psm-model/checkpoints/${OUT_STEM}.pt" "$CKPT"
  cp -f "psm-model/checkpoints/${OUT_STEM}.tokenizer.json" "${CKPT%.pt}.tokenizer.json"
  cp -f "psm-model/checkpoints/${OUT_STEM}.meta.json" "${CKPT%.pt}.meta.json" 2>/dev/null || true
fi

EVAL_OUT="psm-model/prod-memory/results/prod-grounding-overfit-${FINAL_STEP}.json"
python3 -m prod_memory.eval_grounding \
  --checkpoint "$CKPT" \
  --checkpoint-label "overfit-${FINAL_STEP}" \
  --fixtures psm-model/prod-memory/fixtures/cases.json \
  --out "$EVAL_OUT" \
  --device cuda \
  --max-new-tokens 384

python3 - <<PY
import json
from pathlib import Path
r = json.loads(Path("$EVAL_OUT").read_text(encoding="utf-8"))
a = r["aggregate"]
print("=== OVERFIT_EVAL ===")
print(json.dumps({"step": "$FINAL_STEP", "aggregate": a}, indent=2))
for c in r.get("cases", []):
    if c.get("id") in {x.strip() for x in "${FIXTURE_IDS}".split(",")}:
        print(json.dumps({"fixture": c["id"], "effective_stored": c["effective_stored"],
                          "action": c.get("action"), "content_grounded": c.get("content_grounded")}))
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi(token=os.environ["HF_TOKEN"])
for path in [Path("$EVAL_OUT"), Path("psm-model/prod-memory/data/overfit-fixtures.jsonl")]:
    if path.exists():
        api.upload_file(str(path), path.as_posix(), repo_id=repo, repo_type="model",
                        commit_message=f"overfit fixture {path.name}")
        print("uploaded", path)
PY
fi

echo "=== overfit fixture done ==="
