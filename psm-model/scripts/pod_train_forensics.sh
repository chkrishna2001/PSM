#!/bin/bash
set -euo pipefail
cd /workspace/PSM
echo "=== tmux ==="
tmux ls 2>/dev/null || echo NO_TMUX
echo "=== done flags ==="
ls -la /tmp/psm-gate4.done /tmp/psm-gate4-eval.done 2>/dev/null || true
echo "=== checkpoints ==="
ls -la psm-model/checkpoints/real-v3-50m-full-v2-step-*.pt 2>/dev/null | tail -8 || true
echo "=== metrics tail ==="
tail -5 psm-model/checkpoints/real-v3-50m-full-v2-gate4.metrics.jsonl 2>/dev/null || true
echo "=== train log lines ==="
wc -l /tmp/psm-gate4-train.log 2>/dev/null || true
echo "=== train log head ==="
head -30 /tmp/psm-gate4-train.log 2>/dev/null || true
echo "=== train log tail ==="
tail -80 /tmp/psm-gate4-train.log 2>/dev/null || true
echo "=== errors ==="
grep -iE 'error|traceback|exception|abort|collapse|killed|oom|cuda' /tmp/psm-gate4-train.log 2>/dev/null | tail -20 || true
echo "=== step lines ==="
grep -oE 'step[= ][0-9]+|\"step\": [0-9]+' /tmp/psm-gate4-train.log 2>/dev/null | tail -10 || true
