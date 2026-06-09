#!/bin/bash
set -euo pipefail
cd /workspace/PSM
export PSM_HF_MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
for f in \
  psm-model/checkpoints/real-v3-50m-full-v2-step-042000.pt \
  psm-model/checkpoints/real-v3-50m-full-v2-step-042000.tokenizer.json \
  psm-model/checkpoints/real-v3-50m-full-v2-step-042000.meta.json; do
  if [[ ! -f "$f" ]]; then
    echo "downloading $f"
    hf download "$PSM_HF_MODEL_REPO" "$f" --local-dir .
  fi
done
ls -la psm-model/checkpoints/real-v3-50m-full-v2-step-042000.*
