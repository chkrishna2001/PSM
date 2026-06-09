#!/bin/bash
set -euo pipefail
cd /workspace/PSM
export HF_TOKEN="${HF_TOKEN:-}"
export PSM_HF_MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
export UPLOAD_ALL=1
export GATE4_PINNED_STEPS="${GATE4_PINNED_STEPS:-42000,42400}"
export KEEP_LOCAL=2
bash psm-model/scripts/runpod_upload_gate4.sh
