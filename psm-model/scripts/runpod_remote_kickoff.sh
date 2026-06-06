#!/bin/bash
# Paste on RunPod after SSH (stock PyTorch pod). Bootstraps from HF + starts mixed-v2 training.
set -euo pipefail
export PSM_REPO_ROOT=/workspace/PSM
export PSM_SYNC_GIT=1
mkdir -p "$PSM_REPO_ROOT"
cd "$PSM_REPO_ROOT"
pip install -q huggingface_hub hf_transfer numpy tmux 2>/dev/null || true
hf download chkrishna2001/psm-50m-action-mixed-v1 runpod/ --repo-type dataset --local-dir /tmp/psm-hf
chmod +x /tmp/psm-hf/runpod/*.sh
bash /tmp/psm-hf/runpod/runpod_bootstrap.sh
bash /tmp/psm-hf/runpod/runpod_start_mixed_v2.sh
