#!/bin/bash
# Gate 5: dual eval (Gate 4 storage expanded + Gate 5 recall) on one checkpoint.
# Env: EVAL_STEP (required, e.g. 051000), BOOTSTRAP_ONLY=1 to stop after downloads.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
DEVICE="${PSM_EVAL_DEVICE:-cuda}"
EVAL_STEP="${EVAL_STEP:?set EVAL_STEP, e.g. 051000}"

echo "=== PSM Gate 5 dual eval $(date -u +%Y-%m-%dT%H:%M:%SZ) step=$EVAL_STEP device=$DEVICE ==="

export PSM_RUNPOD=1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux >/dev/null 2>&1 || true
pip install -q huggingface_hub hf_transfer numpy

if [[ -d "$ROOT/psm-model/src" ]]; then
  cd "$ROOT"
  git pull --ff-only || true
else
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 "$GIT_URL" "$ROOT"
  cd "$ROOT"
fi

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
export PYTHONPATH=psm-model/src
mkdir -p psm-model/checkpoints/gate-eval psm-model/data/curriculum psm-model/data/direct-behavior-v1

fetch_psm_code() {
  local name="$1"
  if python3 -c "import psm_model.${name//-/_}" 2>/dev/null; then
    return 0
  fi
  HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${name}.py" \
    --repo-type dataset --local-dir /tmp/psm-gate5-code 2>/dev/null || true
  if [[ -f "/tmp/psm-gate5-code/psm-code/${name}.py" ]]; then
    cp "/tmp/psm-gate5-code/psm-code/${name}.py" "psm-model/src/psm_model/${name}.py"
  fi
}

for mod in recall_schema generate_recall_curriculum build_gate5_train_v1 eval_recall eval_dual_gate prompts gates; do
  fetch_psm_code "$mod" || true
done

CKPT="psm-model/checkpoints/real-v3-50m-full-v2-step-${EVAL_STEP}.pt"
for rel in "$CKPT" "${CKPT%.pt}.tokenizer.json" "${CKPT%.pt}.meta.json"; do
  if [[ ! -f "$rel" ]]; then
    echo "Downloading $rel from $MODEL_REPO..."
    hf download "$MODEL_REPO" "$rel" --local-dir .
  fi
done

STORAGE_PROBE="${GATE5_STORAGE_PROBE:-psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl}"
RECALL_PROBE="${GATE5_RECALL_PROBE:-psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl}"

for rel in \
  "$STORAGE_PROBE" \
  "$RECALL_PROBE" \
  psm-model/data/probes/direct_probes.jsonl; do
  if [[ ! -f "$rel" ]]; then
    HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" \
      "data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl" \
      "curriculum/psm-50m-recall-plan-v1.jsonl" \
      --repo-type dataset --local-dir . 2>/dev/null || true
  fi
done

if [[ ! -f "$RECALL_PROBE" ]]; then
  python3 -m psm_model.generate_recall_curriculum "$RECALL_PROBE"
fi

python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ "${BOOTSTRAP_ONLY:-0}" == "1" ]]; then
  echo "PSM_BOOTSTRAP_OK=1"
  exit 0
fi

EVAL_OUT="psm-model/checkpoints/gate-eval/gate5-dual-step-${EVAL_STEP}.json"
EVAL_LOG="/tmp/psm-gate5-dual-eval.log"
EVAL_DONE="/tmp/psm-gate5-dual-eval.done"
rm -f "$EVAL_DONE"
tmux kill-session -t psm-gate5-eval 2>/dev/null || true
tmux new-session -d -s psm-gate5-eval bash -lc "
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src
  python3 -m psm_model.eval_dual_gate \
    '$CKPT' \
    --storage-probe '$STORAGE_PROBE' \
    --recall-probe '$RECALL_PROBE' \
    --device '$DEVICE' \
    > '$EVAL_OUT' 2> '$EVAL_LOG'
  echo \$? > '$EVAL_DONE'
"

sleep 20
if ! tmux has-session -t psm-gate5-eval 2>/dev/null && [[ ! -f "$EVAL_DONE" ]]; then
  echo "FATAL: dual eval tmux died at startup" >&2
  tail -n 40 "$EVAL_LOG" >&2 || true
  exit 1
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
echo "PSM_GATE5_EVAL_STARTED=1 out=$EVAL_OUT log=$EVAL_LOG done_sentinel=$EVAL_DONE"
