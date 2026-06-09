#!/bin/bash
tmux has-session -t psm-gate4 2>/dev/null && echo TMUX_OK || echo TMUX_MISSING
pgrep -af psm_model.train | grep -v tmux | head -2 || echo PROC_MISSING
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- log tail ---"
tail -5 /tmp/psm-gate4-train.log 2>/dev/null || true
grep -oE 'step [0-9]+' /tmp/psm-gate4-train.log 2>/dev/null | tail -1 || true
