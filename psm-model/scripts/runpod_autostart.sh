#!/bin/bash
# Runs automatically on pod start (no SSH). Bootstrap from HF + train mixed-v2 + periodic HF sync.
set -euo pipefail
exec > >(tee -a /workspace/autostart.log) 2>&1
echo "=== PSM autostart $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

export PSM_REPO_ROOT=/workspace/PSM
export PSM_SYNC_GIT=1
export PSM_HF_MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
export PSM_HF_DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
export PYTHONPATH=psm-model/src

pip install -q huggingface_hub hf_transfer numpy tmux git

mkdir -p "$PSM_REPO_ROOT"
cd "$PSM_REPO_ROOT"

hf download "$PSM_HF_DATASET_REPO" runpod/ --repo-type dataset --local-dir /tmp/psm-hf
chmod +x /tmp/psm-hf/runpod/*.sh
mkdir -p psm-model/scripts psm-model/src/psm_model psm-model/checkpoints
cp /tmp/psm-hf/runpod/*.sh psm-model/scripts/
cp /tmp/psm-hf/runpod/train.py psm-model/src/psm_model/train.py

bash psm-model/scripts/runpod_bootstrap.sh

SYNC_CMD='cd /workspace/PSM && export PYTHONPATH=psm-model/src && while true; do sleep 600; python3 psm-model/scripts/sync_training_to_hf.py --repo '"$PSM_HF_MODEL_REPO"' --checkpoint-dir psm-model/checkpoints --run-stem real-v3-50m-action-mixed-v2 --metrics-out psm-model/checkpoints/real-v3-50m-action-mixed-v2.metrics.jsonl --only-new --keep-local 3 2>&1 | tee -a psm-model/checkpoints/sync.log; done'

if ! tmux has-session -t psm-sync 2>/dev/null; then
  tmux new-session -d -s psm-sync "bash -lc $(printf '%q' "$SYNC_CMD")"
  echo "Started HF sync loop in tmux: psm-sync"
fi

bash psm-model/scripts/runpod_start_mixed_v2.sh
echo "=== autostart complete $(date -u +%H:%M:%SZ) — training in tmux psm-mixed-v2 ==="
exec sleep infinity
