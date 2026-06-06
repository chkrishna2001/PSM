#!/bin/bash
# Idempotent: pull training artifacts from HF into /workspace/PSM.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"

cd "$ROOT"
export PYTHONPATH=psm-model/src

mkdir -p psm-model/checkpoints psm-model/data/curriculum psm-model/data/direct-behavior-v1

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN not set (expect RunPod secret: HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}); skipping HF download."
  exit 0
fi

need_ckpt=1
if ls psm-model/checkpoints/real-v3-50m-action-mixed-v1-step-*.pt >/dev/null 2>&1; then
  need_ckpt=0
fi

if [[ "$need_ckpt" -eq 1 ]]; then
  echo "Downloading latest checkpoint from HF model repo..."
  hf download "$MODEL_REPO" checkpoints/ --repo-type model --local-dir psm-model
fi

if [[ ! -f psm-model/data/curriculum/psm-50m-action-mixed-v1-ctx2048.jsonl ]]; then
  hf download "$DATASET_REPO" curriculum/psm-50m-action-mixed-v1-ctx2048.jsonl \
    --repo-type dataset --local-dir psm-model/data
fi

if [[ ! -f psm-model/data/direct-behavior-v1/manual-probe.jsonl ]]; then
  hf download "$DATASET_REPO" probes/expanded-probe-v1-filtered.jsonl probes/manual-probe.jsonl \
    --repo-type dataset --local-dir psm-model/data
  cp psm-model/data/probes/expanded-probe-v1-filtered.jsonl psm-model/data/probes/manual-probe.jsonl \
    psm-model/data/direct-behavior-v1/
fi

echo "Bootstrap complete."
ls -lh psm-model/checkpoints/*.pt 2>/dev/null | tail -3 || true
