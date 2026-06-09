#!/bin/bash
# Idempotent: pull training artifacts from HF into /workspace/PSM.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"

if [[ "${PSM_SYNC_GIT:-1}" == "1" ]] && command -v git >/dev/null 2>&1; then
  if [[ ! -d "$ROOT/.git" ]]; then
    echo "Cloning latest PSM repo from $GIT_URL ..."
    rm -rf "${ROOT}.image-bak"
    if [[ -d "$ROOT/psm-model/src" ]]; then
      mv "$ROOT" "${ROOT}.image-bak"
    fi
    git clone --depth 1 "$GIT_URL" "$ROOT"
  else
    echo "Pulling latest PSM repo ..."
    git -C "$ROOT" pull --ff-only || true
  fi
fi

cd "$ROOT"
export PYTHONPATH=psm-model/src

if ! command -v hf >/dev/null 2>&1; then
  pip install -q huggingface_hub hf_transfer numpy
fi

mkdir -p psm-model/checkpoints psm-model/data/curriculum psm-model/data/direct-behavior-v1

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf download "$DATASET_REPO" runpod/ --repo-type dataset --local-dir /tmp/psm-hf-runpod || true
  if [[ -d /tmp/psm-hf-runpod/runpod ]]; then
    cp /tmp/psm-hf-runpod/runpod/*.sh psm-model/scripts/ 2>/dev/null || true
    chmod +x psm-model/scripts/*.sh 2>/dev/null || true
    if [[ -f /tmp/psm-hf-runpod/runpod/train.py ]]; then
      cp /tmp/psm-hf-runpod/runpod/train.py psm-model/src/psm_model/train.py
    fi
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN not set (expect RunPod secret: HF_TOKEN_C={{ RUNPOD_SECRET_HF_TOKEN_C }}); skipping HF download."
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

for curriculum in \
  psm-50m-action-mixed-v1-ctx2048.jsonl \
  psm-50m-action-mixed-v2-ctx2048.jsonl; do
  if [[ ! -f "psm-model/data/curriculum/$curriculum" ]]; then
    hf download "$DATASET_REPO" "curriculum/$curriculum" \
      --repo-type dataset --local-dir psm-model/data || true
  fi
done

if [[ ! -f psm-model/checkpoints/real-v3-50m-action-mixed-v1-step-007600.pt ]]; then
  hf download "$MODEL_REPO" \
    psm-model/checkpoints/real-v3-50m-action-mixed-v1-step-007600.pt \
    psm-model/checkpoints/real-v3-50m-action-mixed-v1-step-007600.tokenizer.json \
    --local-dir . || true
fi

if [[ ! -f psm-model/data/direct-behavior-v1/manual-probe.jsonl ]]; then
  hf download "$DATASET_REPO" probes/expanded-probe-v1-filtered.jsonl probes/manual-probe.jsonl \
    --repo-type dataset --local-dir psm-model/data
  cp psm-model/data/probes/expanded-probe-v1-filtered.jsonl psm-model/data/probes/manual-probe.jsonl \
    psm-model/data/direct-behavior-v1/
fi

echo "Bootstrap complete."
ls -lh psm-model/checkpoints/*.pt 2>/dev/null | tail -3 || true
