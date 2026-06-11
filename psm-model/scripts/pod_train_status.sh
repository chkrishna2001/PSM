#!/bin/bash
echo "=== $(date -u +%H:%M:%SZ) ==="
tmux ls 2>/dev/null || echo "no tmux sessions"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
if [[ -f /tmp/psm-gate4-train.log ]]; then
  echo "--- train log tail ---"
  tail -n 8 /tmp/psm-gate4-train.log
fi
if [[ -f /workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-gate4.metrics.jsonl ]]; then
  echo "--- metrics tail ---"
  tail -n 2 /workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-gate4.metrics.jsonl
fi
