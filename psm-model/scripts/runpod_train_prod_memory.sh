#!/bin/bash
# Cold pod: bootstrap repo + prod-memory v2 smoke train.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"

echo "=== PSM prod-memory cold bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

export PSM_RUNPOD=1
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux >/dev/null 2>&1 || true
pip install -q huggingface_hub hf_transfer numpy torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || pip install -q huggingface_hub hf_transfer numpy torch

if [[ ! -f "$ROOT/psm-model/src/psm_model/train.py" ]]; then
  if [[ -d "$ROOT" ]]; then
    mv "$ROOT" "${ROOT}.stale.$(date +%s)" 2>/dev/null || rm -rf "$ROOT"
  fi
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 "$GIT_URL" "$ROOT"
fi
cd "$ROOT"
git pull --ff-only || true

sed -i 's/\r$//' psm-model/scripts/*.sh 2>/dev/null || true
chmod +x psm-model/scripts/runpod_upload_gate4.sh psm-model/scripts/runpod_start_prod_memory_train_only.sh 2>/dev/null || true

exec bash psm-model/scripts/runpod_start_prod_memory_train_only.sh
