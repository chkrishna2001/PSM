#!/bin/bash
# Gate 4: expanded eval of one checkpoint on an eval-only pod (no training).
# Env knobs:
#   EVAL_STEP          zero-padded step, e.g. 045000 (required)
#   BOOTSTRAP_ONLY=1   stop after clone + downloads (so the caller can push
#                      uncommitted src patches before the eval starts)
#   DATASET_HF_TOKEN   chkrishna2001 token for the dataset repo (probes);
#                      HF_TOKEN (chinna) only sees the model repo
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
DEVICE="${PSM_EVAL_DEVICE:-cuda}"
EVAL_STEP="${EVAL_STEP:?set EVAL_STEP, e.g. 045000}"

echo "=== PSM Gate 4 expanded eval $(date -u +%Y-%m-%dT%H:%M:%SZ) step=$EVAL_STEP device=$DEVICE ==="

export PSM_RUNPOD=1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux >/dev/null 2>&1 || true
pip install -q huggingface_hub hf_transfer numpy

if [[ -d "$ROOT/psm-model/src" ]]; then
  echo "PSM repo present at $ROOT"
  cd "$ROOT"
  git pull --ff-only || true
else
  echo "PSM repo missing; fresh clone into $ROOT"
  if [[ -d "$ROOT" ]]; then
    mv "$ROOT" "${ROOT}.stale.$(date +%s)" 2>/dev/null || rm -rf "$ROOT"
  fi
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 "$GIT_URL" "$ROOT"
  cd "$ROOT"
fi

export PYTHONPATH=psm-model/src
mkdir -p psm-model/checkpoints/gate-eval psm-model/data/probes

CKPT="psm-model/checkpoints/real-v3-50m-full-v2-step-${EVAL_STEP}.pt"
for rel in "$CKPT" "${CKPT%.pt}.tokenizer.json" "${CKPT%.pt}.meta.json"; do
  if [[ ! -f "$rel" ]]; then
    echo "Downloading $rel from $MODEL_REPO..."
    hf download "$MODEL_REPO" "$rel" --local-dir .
  fi
done

PROBES="psm-model/data/probes/expanded-probe-v1-budget.jsonl"
if [[ ! -f "$PROBES" ]]; then
  echo "Downloading probes from $DATASET_REPO (dataset token)..."
  HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" \
    probes/expanded-probe-v1-budget.jsonl --repo-type dataset --local-dir psm-model/data
fi
echo "probe rows: $(wc -l < "$PROBES")"

python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ "${BOOTSTRAP_ONLY:-0}" == "1" ]]; then
  echo "PSM_BOOTSTRAP_OK=1 (stopping before eval; push src patches now)"
  exit 0
fi

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true

EVAL_OUT="psm-model/checkpoints/gate-eval/gate4-full-expanded-step-${EVAL_STEP}.json"
EVAL_LOG="/tmp/psm-gate4-eval.log"
EVAL_DONE="/tmp/psm-gate4-eval.done"
rm -f "$EVAL_DONE"
tmux kill-session -t psm-gate4-eval 2>/dev/null || true
tmux new-session -d -s psm-gate4-eval bash -lc "
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src
  python3 -m psm_model.eval_checkpoint \
    '$CKPT' '$PROBES' \
    --device '$DEVICE' --output-format tagged --gate-mode expanded \
    > '$EVAL_OUT' 2> '$EVAL_LOG'
  echo \$? > '$EVAL_DONE'
"

sleep 20
if ! tmux has-session -t psm-gate4-eval 2>/dev/null && [[ ! -f "$EVAL_DONE" ]]; then
  echo "FATAL: eval tmux died at startup" >&2
  tail -n 40 "$EVAL_LOG" >&2 || true
  exit 1
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
echo "PSM_EVAL_STARTED=1 out=$EVAL_OUT log=$EVAL_LOG done_sentinel=$EVAL_DONE"
