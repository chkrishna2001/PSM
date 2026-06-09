#!/bin/bash
# Push latest Gate 4 upload/registry to pod and run a sync (pinned best + resume).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"

cd "$ROOT"
export PYTHONPATH=psm-model/src
pip install -q huggingface_hub hf_transfer

hf download "$DATASET_REPO" \
  psm-code/runpod_upload_gate4.sh \
  psm-code/gate4_checkpoint_registry.py \
  --repo-type dataset --local-dir /tmp/psm-gate4-hotfix

mkdir -p psm-model/scripts psm-model/src/psm_model psm-model/checkpoints
cp /tmp/psm-gate4-hotfix/psm-code/runpod_upload_gate4.sh psm-model/scripts/
cp /tmp/psm-gate4-hotfix/psm-code/gate4_checkpoint_registry.py psm-model/src/psm_model/

hf download "$MODEL_REPO" psm-model/checkpoints/gate4-checkpoint-registry.json --local-dir . || true

export KEEP_LOCAL=2
export GATE4_PINNED_STEPS="41600,42000"
bash psm-model/scripts/runpod_upload_gate4.sh
