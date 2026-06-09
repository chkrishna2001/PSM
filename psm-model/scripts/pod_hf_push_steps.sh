#!/bin/bash
set -euo pipefail
cd /workspace/PSM
export PSM_HF_MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
export STEPS="${STEPS:-42000,42400}"
export PYTHONPATH=psm-model/src
python3 psm-model/scripts/pod_hf_push_steps.py
