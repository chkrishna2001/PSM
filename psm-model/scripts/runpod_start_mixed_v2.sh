#!/bin/bash
# Run on RunPod after bootstrap: mixed-v2 Gate 2 repair training in tmux.
set -euo pipefail
ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
bash psm-model/scripts/runpod_bootstrap.sh
export PYTHONPATH=psm-model/src

SESSION="${TMUX_SESSION:-psm-mixed-v2}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION already exists; attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash psm-model/scripts/runpod_train_mixed_v2.sh 2>&1 | tee psm-model/checkpoints/mixed-v2-train.log"
echo "Started training in tmux session: $SESSION"
echo "  tmux attach -t $SESSION"
echo "  tail -f psm-model/checkpoints/mixed-v2-train.log"
