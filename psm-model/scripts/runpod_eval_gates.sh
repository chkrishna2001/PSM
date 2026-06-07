#!/bin/bash
# Bootstrap + Gate 2/3 (and optional expanded) eval on GPU. Pipe via SSH on a stock PyTorch pod.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
DEVICE="${PSM_EVAL_DEVICE:-cuda}"
OUT_DIR="${PSM_EVAL_OUT:-psm-model/checkpoints/gate-eval}"
RUN_EXPANDED="${PSM_EVAL_EXPANDED:-0}"

FULL_CKPT="${PSM_EVAL_FULL_CKPT:-psm-model/checkpoints/real-v3-50m-full-v2.pt}"
ACTION_CKPT="psm-model/checkpoints/real-v3-50m-action-mixed-v2-step-009800.pt"

echo "=== PSM gate eval $(date -u +%Y-%m-%dT%H:%M:%SZ) device=$DEVICE ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git >/dev/null 2>&1 || true
pip install -q huggingface_hub hf_transfer numpy

mkdir -p "$ROOT"
if [[ ! -d "$ROOT/psm-model/src" ]]; then
  echo "Cloning PSM repo..."
  git clone --depth 1 "$GIT_URL" "$ROOT"
fi
cd "$ROOT"
export PYTHONPATH=psm-model/src

mkdir -p psm-model/checkpoints psm-model/data/probes psm-model/data/direct-behavior-v1 "$OUT_DIR"

download_ckpt() {
  local rel="$1"
  if [[ ! -f "$rel" ]]; then
    echo "Downloading $rel from $MODEL_REPO..."
    hf download "$MODEL_REPO" "$rel" --local-dir .
  fi
}

for rel in \
  "$FULL_CKPT" \
  "${FULL_CKPT%.pt}.tokenizer.json" \
  "$ACTION_CKPT" \
  "${ACTION_CKPT%.pt}.tokenizer.json"; do
  download_ckpt "$rel"
done

if [[ ! -f psm-model/data/probes/direct_probes.jsonl ]]; then
  hf download "$DATASET_REPO" \
    data/probes/direct_probes.jsonl \
    data/direct-behavior-v1/manual-probe.jsonl \
    data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    --repo-type dataset --local-dir . || true
fi
for rel in \
  psm-model/data/probes/direct_probes.jsonl \
  psm-model/data/direct-behavior-v1/manual-probe.jsonl \
  psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl; do
  if [[ ! -f "$rel" ]]; then
    hf download "$DATASET_REPO" \
      probes/direct_probes.jsonl \
      probes/manual-probe.jsonl \
      probes/expanded-probe-v1-filtered.jsonl \
      --repo-type dataset --local-dir psm-model/data || true
    cp -f psm-model/data/probes/manual-probe.jsonl psm-model/data/direct-behavior-v1/ 2>/dev/null || true
    cp -f psm-model/data/probes/expanded-probe-v1-filtered.jsonl psm-model/data/direct-behavior-v1/ 2>/dev/null || true
  fi
done

python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

run_eval() {
  local name="$1"
  shift
  local out="$OUT_DIR/${name}.json"
  echo "--- $name ---"
  set +e
  python3 "$@" | tee "$out"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "wrote $out (exit $rc)"
  return $rc
}

GATE3_RC=0
GATE2_RC=0
SMOKE_RC=0
EXPANDED_RC=0

run_eval gate3-full-direct \
  -m psm_model.eval_checkpoint \
  "$FULL_CKPT" psm-model/data/probes/direct_probes.jsonl \
  --device "$DEVICE" --output-format tagged || GATE3_RC=$?

run_eval gate2-phase1-action \
  -m psm_model.gate_checkpoint \
  "$ACTION_CKPT" \
  --mode phase1-action --device "$DEVICE" --output-format action || GATE2_RC=$?

run_eval gate2-action-smoke \
  -m psm_model.action_smoke \
  "$ACTION_CKPT" psm-model/data/direct-behavior-v1/manual-probe.jsonl \
  --device "$DEVICE" --output-format action --prefix-eval || SMOKE_RC=$?

if [[ "$RUN_EXPANDED" == "1" ]]; then
  EXPANDED_PROBE="psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl"
  EXPANDED_BUDGET="psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl"
  if [[ -f "$EXPANDED_PROBE" ]]; then
    python3 -m psm_model.filter_by_token_budget \
      "$EXPANDED_PROBE" "$EXPANDED_BUDGET" \
      --tokenizer "${FULL_CKPT%.pt}.tokenizer.json" \
      --max-tokens 1536 --output-format tagged || true
    if [[ -f "$EXPANDED_BUDGET" ]]; then
      EXPANDED_PROBE="$EXPANDED_BUDGET"
    fi
  fi
  run_eval gate4-full-expanded \
    -m psm_model.eval_checkpoint \
    "$FULL_CKPT" "$EXPANDED_PROBE" \
    --device "$DEVICE" --output-format tagged --gate-mode expanded || EXPANDED_RC=$?
  if [[ -f "$OUT_DIR/gate4-full-expanded.json" ]]; then
    run_eval gate4-failure-analysis \
      -m psm_model.analyze_eval_report \
      "$OUT_DIR/gate4-full-expanded.json" \
      --gate-mode expanded || true
  fi
fi

python3 - <<PY
import json
from pathlib import Path

out_dir = Path("$OUT_DIR")
summary = {"device": "$DEVICE", "reports": {}}
for path in sorted(out_dir.glob("*.json")):
    try:
        summary["reports"][path.stem] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        summary["reports"][path.stem] = {"error": "invalid json", "path": str(path)}
summary["exit_codes"] = {
    "gate3": int("$GATE3_RC"),
    "gate2": int("$GATE2_RC"),
    "smoke": int("$SMOKE_RC"),
    "expanded": int("$EXPANDED_RC"),
}
summary["passed"] = all(
    code == 0 for key, code in summary["exit_codes"].items()
    if key != "expanded" or "$RUN_EXPANDED" == "1"
)
path = out_dir / "summary.json"
path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "=== gate eval done $(date -u +%H:%M:%SZ) reports in $OUT_DIR ==="
if [[ "$GATE3_RC" -ne 0 || "$GATE2_RC" -ne 0 || "$SMOKE_RC" -ne 0 ]]; then
  exit 1
fi
if [[ "$RUN_EXPANDED" == "1" && "$EXPANDED_RC" -ne 0 ]]; then
  exit 1
fi
