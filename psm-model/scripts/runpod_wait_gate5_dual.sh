#!/bin/bash
# Wait for dual eval tmux started by runpod_eval_gate5_dual.sh (WAIT_EVAL_DONE=0).
set -euo pipefail
EVAL_DONE="/tmp/psm-gate5-dual-eval.done"
while [[ ! -f "$EVAL_DONE" ]]; do
  if ! tmux has-session -t psm-gate5-eval 2>/dev/null; then
    echo "FATAL: eval tmux ended before completion" >&2
    tail -n 40 /tmp/psm-gate5-dual-eval.log >&2 || true
    exit 1
  fi
  sleep 30
done
cat "$EVAL_DONE"
