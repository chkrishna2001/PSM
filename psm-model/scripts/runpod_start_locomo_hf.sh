#!/bin/bash
# Warm pod: start LoCoMo HF ingest+eval in tmux (same pattern as prod-memory train).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
cd "$ROOT"
export PSM_RUNPOD=1

LOG="/tmp/psm-locomo.log"
DONE="/tmp/psm-locomo.done"

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
chmod +x psm-model/scripts/runpod_locomo.sh 2>/dev/null || true

if ! command -v tmux >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1 || true
fi

rm -f "$DONE"
: > "$LOG"
tmux kill-session -t psm-locomo 2>/dev/null || true
tmux new-session -d -s psm-locomo bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export PSM_RUNPOD=1
  bash psm-model/scripts/runpod_locomo.sh 2>&1 | tee -a '$LOG'
  echo \$? > '$DONE'
"

sleep 5
tmux has-session -t psm-locomo 2>/dev/null && echo TMUX_OK || echo TMUX_MISSING
pgrep -af 'runpod_locomo|ingest-cli' | head -3 || echo PROC_MISSING
head -5 '$LOG' 2>/dev/null || true
echo "=== LoCoMo tmux started (detached) log=$LOG ==="
