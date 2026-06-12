#!/bin/bash
# Gate 5: mixed storage+recall training from Gate 4 best (default step-048000).
# Post-train: dual gate eval (storage expanded + recall probes). Reuses gate4 HF upload/registry.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"
DEVICE="${PSM_TRAIN_DEVICE:-cuda}"

RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2-step-048000.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-full-v2-step-048000.tokenizer.json}"
TARGET_STEPS="${TARGET_STEPS:-51000}"
CURRICULUM="${GATE5_CURRICULUM:-psm-model/data/curriculum/psm-50m-gate5-train-v1.jsonl}"
SAVE_EVERY="${SAVE_EVERY:-200}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-120}"
UPLOAD_ALL="${UPLOAD_ALL:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-1e-5}"
EVAL_EVERY="${EVAL_EVERY:-400}"
RESUME_STEP="$(basename "$RESUME" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"

echo "=== PSM Gate 5 train $(date -u +%Y-%m-%dT%H:%M:%SZ) device=$DEVICE target=$TARGET_STEPS resume_step=$RESUME_STEP ==="

export PSM_RUNPOD=1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux >/dev/null 2>&1 || true
pip install -q huggingface_hub hf_transfer numpy

if [[ -d "$ROOT/psm-model/src" ]]; then
  cd "$ROOT"
  git pull --ff-only || true
else
  if [[ -d "$ROOT" ]]; then
    mv "$ROOT" "${ROOT}.stale.$(date +%s)" 2>/dev/null || rm -rf "$ROOT"
  fi
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 "$GIT_URL" "$ROOT"
  cd "$ROOT"
fi

for gate_script in runpod_upload_gate4.sh runpod_eval_gate5_dual.sh; do
  HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${gate_script}" \
    --repo-type dataset --local-dir /tmp/psm-gate5-scripts 2>/dev/null || true
  if [[ -f "/tmp/psm-gate5-scripts/psm-code/${gate_script}" ]]; then
    mkdir -p psm-model/scripts
    cp "/tmp/psm-gate5-scripts/psm-code/${gate_script}" "psm-model/scripts/${gate_script}"
  fi
done

for mod in recall_schema generate_recall_curriculum build_gate5_train_v1 eval_recall eval_dual_gate gate4_checkpoint_registry eval_generation prompts gates; do
  if ! python3 -c "import psm_model.${mod}" 2>/dev/null; then
    HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" "psm-code/${mod}.py" \
      --repo-type dataset --local-dir /tmp/psm-gate5-code 2>/dev/null || true
    if [[ -f "/tmp/psm-gate5-code/psm-code/${mod}.py" ]]; then
      mkdir -p psm-model/src/psm_model
      cp "/tmp/psm-gate5-code/psm-code/${mod}.py" "psm-model/src/psm_model/${mod}.py"
    fi
  fi
done

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
chmod +x psm-model/scripts/runpod_upload_gate4.sh psm-model/scripts/runpod_eval_gate5_dual.sh 2>/dev/null || true

export PYTHONPATH=psm-model/src
mkdir -p psm-model/checkpoints psm-model/checkpoints/gate-eval psm-model/data/curriculum psm-model/data/probes psm-model/data/direct-behavior-v1

download_ckpt() {
  local rel="$1"
  if [[ ! -f "$rel" ]]; then
    echo "Downloading $rel from $MODEL_REPO..."
    hf download "$MODEL_REPO" "$rel" --local-dir .
  fi
}

for rel in "$RESUME" "${RESUME%.pt}.tokenizer.json" "${RESUME%.pt}.meta.json"; do
  download_ckpt "$rel"
done
if [[ ! -f "$RESUME" ]]; then
  echo "Gate 5 resume checkpoint unavailable on HF: $RESUME" >&2
  exit 1
fi

for rel in \
  psm-model/data/probes/direct_probes.jsonl \
  psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl; do
  if [[ ! -f "$rel" ]]; then
    HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" \
      data/probes/direct_probes.jsonl \
      data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
      --repo-type dataset --local-dir . 2>/dev/null || true
  fi
done

RECALL_PROBE="${GATE5_RECALL_PROBE:-psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl}"
if [[ ! -f "$RECALL_PROBE" ]]; then
  HF_TOKEN="${DATASET_HF_TOKEN:-${HF_TOKEN:-}}" hf download "$DATASET_REPO" curriculum/psm-50m-recall-plan-v1.jsonl \
    --repo-type dataset --local-dir psm-model/data 2>/dev/null || true
  if [[ ! -f "$RECALL_PROBE" ]]; then
    python3 -m psm_model.generate_recall_curriculum "$RECALL_PROBE"
  fi
fi

if [[ "${SKIP_CURRICULUM_BUILD:-0}" != "1" ]]; then
  if [[ -f "$CURRICULUM" && "${GATE5_FORCE_REBUILD:-0}" != "1" ]]; then
    echo "Using existing gate5 curriculum: $CURRICULUM"
  else
    python3 -m psm_model.build_gate5_train_v1 "$CURRICULUM" \
      --expanded-probes psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
      --direct-probes psm-model/data/probes/direct_probes.jsonl \
      --expanded-copies "${EXPANDED_COPIES:-25}" \
      --direct-copies "${DIRECT_COPIES:-100}" \
      --recall-copies "${RECALL_COPIES:-20}"
  fi
fi

python3 - <<'PY'
import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

export GATE4_PINNED_STEPS="${RESUME_STEP}"
SYNC_CMD="cd '$ROOT' && export UPLOAD_ALL=${UPLOAD_ALL} KEEP_LOCAL=$KEEP_LOCAL GATE4_PINNED_STEPS='$RESUME_STEP' PSM_HF_MODEL_REPO='$MODEL_REPO' PSM_HF_DATASET_REPO='$DATASET_REPO' && while true; do sleep $SYNC_INTERVAL_SEC; bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate5-sync.log; done"
tmux kill-session -t psm-gate5-sync 2>/dev/null || true
tmux new-session -d -s psm-gate5-sync bash -lc "$(printf '%q' "$SYNC_CMD")"

TRAIN_LOG="/tmp/psm-gate5-train.log"
TRAIN_DONE="/tmp/psm-gate5.done"
rm -f "$TRAIN_DONE"
tmux kill-session -t psm-gate5 2>/dev/null || true
tmux new-session -d -s psm-gate5 bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src
  python3 -m psm_model.train \
    '$CURRICULUM' \
    --out psm-model/checkpoints/real-v3-50m-full-v2.pt \
    --resume '$RESUME' \
    --tokenizer '$TOK' \
    --steps '$TARGET_STEPS' \
    --batch-size '$BATCH_SIZE' \
    --learning-rate '$LEARNING_RATE' \
    --min-learning-rate '$MIN_LEARNING_RATE' \
    --warmup-steps '${WARMUP_STEPS:-50}' \
    --preset 50m \
    --output-format tagged \
    --sampling random \
    --device '$DEVICE' \
    --save-every '$SAVE_EVERY' \
    --metrics-out psm-model/checkpoints/real-v3-50m-full-v2-gate5.metrics.jsonl \
    --structural-loss-weight '${STRUCTURAL_LOSS_WEIGHT:-1}' \
    --eval-every '$EVAL_EVERY' \
    --probe psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    --manual-probe psm-model/data/probes/direct_probes.jsonl \
    --abort-after-step '${ABORT_AFTER_STEP:-60000}' \
    --collapse-threshold 0.90 \
    2>&1 | tee '$TRAIN_LOG'
  echo done > '$TRAIN_DONE'
"

sleep 2
tail -n 20 -f "$TRAIN_LOG" &
TAIL_PID=$!
while [[ ! -f "$TRAIN_DONE" ]]; do
  if ! tmux has-session -t psm-gate5 2>/dev/null; then
    echo "tmux session psm-gate5 ended unexpectedly" >&2
    break
  fi
  sleep 30
done
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true

echo "=== Gate 5 train done $(date -u +%H:%M:%SZ) ==="
tmux kill-session -t psm-gate5-sync 2>/dev/null || true
export UPLOAD_ALL=1
bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate5-sync.log

TARGET_STEP_PADDED="$(printf '%06d' "$TARGET_STEPS")"
if [[ "${GATE5_EVAL_AFTER:-1}" == "1" ]]; then
  export EVAL_STEP="$TARGET_STEP_PADDED"
  export GATE5_STORAGE_PROBE="psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl"
  export GATE5_RECALL_PROBE="$RECALL_PROBE"
  echo "--- post-train dual gate eval step=$EVAL_STEP ---"
  bash psm-model/scripts/runpod_eval_gate5_dual.sh
  while [[ ! -f /tmp/psm-gate5-dual-eval.done ]]; do
    if ! tmux has-session -t psm-gate5-eval 2>/dev/null; then
      break
    fi
    sleep 30
  done
  if [[ -f /tmp/psm-gate5-dual-eval.done ]]; then
    DUAL_RC="$(cat /tmp/psm-gate5-dual-eval.done)"
    echo "dual eval exit=$DUAL_RC"
    if [[ -f "psm-model/checkpoints/gate-eval/gate5-dual-step-${TARGET_STEP_PADDED}.json" ]]; then
      python3 -m psm_model.gate4_checkpoint_registry update-eval \
        "psm-model/checkpoints/gate-eval/gate5-dual-step-${TARGET_STEP_PADDED}.json" || true
    fi
  fi
fi

TARGET_STEP_NUM="$(printf '%06d' "$TARGET_STEPS" | sed 's/^0*//')"
export GATE4_FINAL_SYNC=1
export UPLOAD_ALL=1
export GATE4_KEEP_BEST_ONLY=1
export GATE4_PINNED_STEPS="${RESUME_STEP},${TARGET_STEP_NUM}"
bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate5-sync.log
python3 -m psm_model.gate4_checkpoint_registry verify-hf \
  --repo-id "$MODEL_REPO" \
  --run-stem real-v3-50m-full-v2 \
  --checkpoint-dir psm-model/checkpoints
echo "GATE5_FINAL_SYNC_OK=1"
