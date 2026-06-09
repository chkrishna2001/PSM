#!/bin/bash
# Fast path: warm pod already has repo + checkpoints. Build curriculum, start tmux, exit.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PYTHONPATH=psm-model/src

MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
DEVICE="${PSM_TRAIN_DEVICE:-cuda}"
RESUME="${RESUME_CHECKPOINT:-psm-model/checkpoints/real-v3-50m-full-v2-step-042000.pt}"
TOK="${TOKENIZER:-psm-model/checkpoints/real-v3-50m-full-v2-step-042000.tokenizer.json}"
TARGET_STEPS="${TARGET_STEPS:-43500}"
CURRICULUM="${GATE4_CURRICULUM:-psm-model/data/curriculum/psm-50m-gate4-train-micro.jsonl}"
CURRICULUM_BUILDER="${GATE4_CURRICULUM_BUILDER:-micro}"
SAVE_EVERY="${SAVE_EVERY:-200}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
SYNC_INTERVAL_SEC="${SYNC_INTERVAL_SEC:-120}"
UPLOAD_ALL="${UPLOAD_ALL:-1}"
STRUCTURAL_LOSS_WEIGHT="${STRUCTURAL_LOSS_WEIGHT:-8}"
PROMOTE_SPAN_WEIGHT="${PROMOTE_SPAN_WEIGHT:-4}"
EVAL_EVERY="${EVAL_EVERY:-200}"
RESUME_STEP="$(basename "$RESUME" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"

echo "=== start gate4 train-only $(date -u +%Y-%m-%dT%H:%M:%SZ) builder=$CURRICULUM_BUILDER resume=$RESUME_STEP target=$TARGET_STEPS ==="

if [[ "$CURRICULUM_BUILDER" == "micro" ]]; then
  EVAL_REPORT="${GATE4_EVAL_REPORT:-psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042000.json}"
  if [[ ! -f "$EVAL_REPORT" && -f psm-model/checkpoints/gate-eval/gate4-full-expanded.json ]]; then
    EVAL_REPORT="psm-model/checkpoints/gate-eval/gate4-full-expanded.json"
  fi
  REPAIR_SOURCE="${GATE4_REPAIR_SOURCE:-psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl}"
  PARSE_REPAIR="${GATE4_PARSE_REPAIR:-psm-model/data/curriculum/gate4-parse-repair-step-042000.jsonl}"
  if [[ ! -f "$EVAL_REPORT" ]]; then
    echo "missing eval report: $EVAL_REPORT" >&2
    exit 1
  fi
  python3 -m psm_model.build_gate4_parse_repair_micro "$CURRICULUM" \
    --direct-probes psm-model/data/probes/direct_probes.jsonl \
    --eval-report "$EVAL_REPORT" \
    --repair-source "$REPAIR_SOURCE" \
    --parse-repair "$PARSE_REPAIR" \
    --direct-copies "${DIRECT_COPIES:-20}" \
    --drill-rows-per-action "${DRILL_ROWS_PER_ACTION:-120}" \
    --drill-copies "${DRILL_COPIES:-5}" \
    --repair-copies "${REPAIR_COPIES:-12}"
fi

export GATE4_PINNED_STEPS="${RESUME_STEP}"
SYNC_CMD="cd '$ROOT' && export HF_TOKEN=\"\${HF_TOKEN:-}\" UPLOAD_ALL=${UPLOAD_ALL} KEEP_LOCAL=$KEEP_LOCAL GATE4_PINNED_STEPS='$RESUME_STEP' PSM_HF_MODEL_REPO='$MODEL_REPO' PSM_HF_DATASET_REPO='$DATASET_REPO' && while true; do sleep $SYNC_INTERVAL_SEC; bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tee -a psm-model/checkpoints/gate4-sync.log; done"
tmux kill-session -t psm-gate4-sync 2>/dev/null || true
tmux new-session -d -s psm-gate4-sync bash -lc "$(printf '%q' "$SYNC_CMD")"
echo "HF sync: tmux psm-gate4-sync every ${SYNC_INTERVAL_SEC}s"

TRAIN_LOG="/tmp/psm-gate4-train.log"
TRAIN_DONE="/tmp/psm-gate4.done"
rm -f "$TRAIN_DONE"
tmux kill-session -t psm-gate4 2>/dev/null || true
tmux new-session -d -s psm-gate4 bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src
  python3 -m psm_model.train \
    '$CURRICULUM' \
    --out psm-model/checkpoints/real-v3-50m-full-v2.pt \
    --resume '$RESUME' \
    --tokenizer '$TOK' \
    --steps '$TARGET_STEPS' \
    --batch-size 1 \
    --preset 50m \
    --output-format tagged \
    --sampling action_balanced \
    --device '$DEVICE' \
    --save-every '$SAVE_EVERY' \
    --metrics-out psm-model/checkpoints/real-v3-50m-full-v2-gate4.metrics.jsonl \
    --action-span-weight ignore=4 \
    --action-span-weight promote_semantic='$PROMOTE_SPAN_WEIGHT' \
    --action-span-weight store_episodic=8 \
    --action-span-weight flag_conflict=3 \
    --structural-loss-weight '$STRUCTURAL_LOSS_WEIGHT' \
    --eval-every '$EVAL_EVERY' \
    --probe psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    --manual-probe psm-model/data/probes/direct_probes.jsonl \
    --abort-after-step '${ABORT_AFTER_STEP:-60000}' \
    --collapse-threshold 0.90 \
    2>&1 | tee '$TRAIN_LOG'
  echo done > '$TRAIN_DONE'
"

sleep 3
tmux ls
echo "--- train log head ---"
head -5 "$TRAIN_LOG" 2>/dev/null || true
echo "--- train log tail ---"
tail -8 "$TRAIN_LOG" 2>/dev/null || true
pgrep -af 'psm_model.train' || echo "WARN: no train process yet"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo "=== train tmux started (detached) ==="
