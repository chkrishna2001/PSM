#!/bin/bash
# Quick status of the gate4 expanded eval running on this pod.
cd "${PSM_REPO_ROOT:-/workspace/PSM}" 2>/dev/null || true
if [[ -f /tmp/psm-gate4-eval.done ]]; then
  echo "PSM_EVAL_DONE rc=$(cat /tmp/psm-gate4-eval.done)"
else
  tmux has-session -t psm-gate4-eval 2>/dev/null && echo "PSM_EVAL_RUNNING" || echo "PSM_EVAL_TMUX_MISSING"
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
echo "--- last log lines ---"
tail -n 6 /tmp/psm-gate4-eval.log 2>/dev/null || echo "(no log yet)"
OUT="psm-model/checkpoints/gate-eval/gate4-full-expanded-step-${EVAL_STEP:-045000}.json"
if [[ -f "$OUT" ]]; then
  echo "out_size_bytes=$(stat -c%s "$OUT")"
fi
