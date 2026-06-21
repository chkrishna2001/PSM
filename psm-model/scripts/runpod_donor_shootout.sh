#!/bin/bash
# Step 1: prod grounding eval on gate donors (no training).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1
export PYTHONPATH=psm-model/src:psm-model/prod-memory

SUBBU_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
LEGACY_REPO="${PSM_HF_LEGACY_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
STEM="${DONOR_STEM:-real-v3-50m-full-v2}"
DEVICE="${PSM_EVAL_DEVICE:-cuda}"
MAX_NEW="${MAX_NEW_TOKENS:-384}"

echo "=== donor shootout $(date -u +%Y-%m-%dT%H:%M:%SZ) subbu=$SUBBU_REPO legacy=$LEGACY_REPO ==="

pip install -q huggingface_hub hf_transfer 2>/dev/null || pip install -q huggingface_hub

mkdir -p psm-model/checkpoints psm-model/prod-memory/results psm-model/prod-memory/fixtures

if [[ ! -f psm-model/prod-memory/fixtures/cases.json ]]; then
  hf download "$DATASET_REPO" prod-memory/fixtures/cases.json \
    --repo-type dataset --local-dir . --token "${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" 2>/dev/null || true
  if [[ -f prod-memory/fixtures/cases.json ]]; then
    cp -f prod-memory/fixtures/cases.json psm-model/prod-memory/fixtures/cases.json
  fi
fi

download_ckpt() {
  local repo="$1" step="$2" token="$3"
  local ckpt="psm-model/checkpoints/${STEM}-step-${step}.pt"
  for rel in "$ckpt" "${ckpt%.pt}.tokenizer.json" "${ckpt%.pt}.meta.json"; do
    if [[ ! -f "$rel" ]]; then
      echo "Downloading $rel from $repo..."
      hf download "$repo" "$rel" --local-dir . --token "$token"
    fi
  done
}

eval_step() {
  local step="$1"
  local ckpt="psm-model/checkpoints/${STEM}-step-${step}.pt"
  local out="psm-model/prod-memory/results/prod-grounding-${step}.json"
  python3 -m prod_memory.eval_grounding \
    --checkpoint "$ckpt" \
    --checkpoint-label "$step" \
    --fixtures psm-model/prod-memory/fixtures/cases.json \
    --out "$out" \
    --device "$DEVICE" \
    --max-new-tokens "$MAX_NEW"
  python3 - <<PY
import json
from pathlib import Path
r = json.loads(Path("$out").read_text(encoding="utf-8"))
a = r["aggregate"]
print(json.dumps({"step": "$step", "effective_stored": a["effective_stored"], "cases": a["cases"],
  "parse_valid_rate": a["parse_valid_rate"], "fail_safe_ignore_rate": a["fail_safe_ignore_rate"]}, indent=2))
PY
}

python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

# subbu83 donors
for step in 058000 048000; do
  download_ckpt "$SUBBU_REPO" "$step" "${HF_TOKEN:-}"
  eval_step "$step"
done

# chkrishna2001 legacy donor (022800 not on HF; earliest .pt is 032000 + gate3 real-v3-50m-full-v2.pt)
LEGACY_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}"
LEGACY_STEP="${LEGACY_DONOR_STEP:-032000}"
download_ckpt "$LEGACY_REPO" "$LEGACY_STEP" "$LEGACY_TOKEN"
eval_step "$LEGACY_STEP"

python3 - <<PY
import json, os
from pathlib import Path
legacy = os.environ.get("LEGACY_DONOR_STEP", "032000")
rows = []
for step in ("058000", "048000", legacy):
    p = Path(f"psm-model/prod-memory/results/prod-grounding-{step}.json")
    if not p.exists():
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    a = r["aggregate"]
    rows.append({"step": step, "effective_stored": f"{a['effective_stored']}/{a['cases']}",
                 "parse": a["parse_valid_rate"], "ignore": a["fail_safe_ignore_rate"]})
print("=== DONOR_SHOOTOUT_SUMMARY ===")
print(json.dumps(rows, indent=2))
Path("psm-model/prod-memory/results/donor-shootout-summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY

if [[ -n "${HF_TOKEN:-}" ]]; then
  python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi
repo = os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
api = HfApi(token=os.environ.get("HF_TOKEN"))
for path in Path("psm-model/prod-memory/results").glob("prod-grounding-*.json"):
    api.upload_file(str(path), path.as_posix(), repo_id=repo, repo_type="model",
                    commit_message=f"donor shootout {path.name}")
    print("uploaded", path.name)
summary = Path("psm-model/prod-memory/results/donor-shootout-summary.json")
if summary.exists():
    api.upload_file(str(summary), summary.as_posix(), repo_id=repo, repo_type="model",
                    commit_message="donor shootout summary")
PY
fi

echo "=== donor shootout done ==="
