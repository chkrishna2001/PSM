#!/bin/bash
# Fast path: warm pod — expanded Gate 4 eval only, tmux detached.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1 PYTHONPATH=psm-model/src

DEVICE="${PSM_EVAL_DEVICE:-cuda}"
FULL_CKPT="${PSM_EVAL_FULL_CKPT:-psm-model/checkpoints/real-v3-50m-full-v2-step-043400.pt}"
OUT_DIR="${PSM_EVAL_OUT:-psm-model/checkpoints/gate-eval}"
STEP="$(basename "$FULL_CKPT" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"
EVAL_OUT="${OUT_DIR}/gate4-full-expanded-step-${STEP}.json"

echo "=== start gate4 eval-only $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpt=$FULL_CKPT ==="

if [[ ! -f "$FULL_CKPT" ]]; then
  echo "missing checkpoint: $FULL_CKPT" >&2
  exit 1
fi

EXPANDED_PROBE="psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl"
if [[ ! -f "$EXPANDED_PROBE" && -f psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl ]]; then
  python3 -m psm_model.filter_by_token_budget \
    psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl \
    "$EXPANDED_PROBE" \
    --tokenizer "${FULL_CKPT%.pt}.tokenizer.json" \
    --max-tokens 1536 \
    --output-format tagged
fi
if [[ ! -f "$EXPANDED_PROBE" ]]; then
  EXPANDED_PROBE="psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl"
fi

mkdir -p "$OUT_DIR"
EVAL_LOG="/tmp/psm-gate4-eval.log"
EVAL_DONE="/tmp/psm-gate4-eval.done"
rm -f "$EVAL_DONE"

tmux kill-session -t psm-gate4-eval 2>/dev/null || true
tmux new-session -d -s psm-gate4-eval bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PSM_RUNPOD=1 PYTHONPATH=psm-model/src
  python3 -m psm_model.eval_checkpoint \
    '$FULL_CKPT' '$EXPANDED_PROBE' \
    --device '$DEVICE' \
    --output-format tagged \
    --gate-mode expanded \
    2>&1 | tee '$EVAL_LOG' | tee '$EVAL_OUT'
  python3 -m psm_model.analyze_eval_report '$EVAL_OUT' --gate-mode expanded \
    > '${OUT_DIR}/gate4-failure-analysis-step-${STEP}.json' 2>&1 || true
  echo done > '$EVAL_DONE'
"

sleep 4
tmux ls 2>/dev/null || true
pgrep -af 'psm_model.eval_checkpoint' | head -2 || echo "WARN: no eval process"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
tail -5 "$EVAL_LOG" 2>/dev/null || true
echo "=== eval tmux started -> $EVAL_OUT ==="
