#!/bin/bash
# Git-sync then HF LoRA prod eval (adapter from HF, no train).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
GIT_URL="${PSM_GIT_URL:-https://github.com/chkrishna2001/PSM.git}"

echo "=== hf lora eval bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) git=$GIT_URL ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 || true

if [[ ! -d "$ROOT/.git" ]]; then
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 "$GIT_URL" "$ROOT"
else
  git -C "$ROOT" pull --ff-only
fi

cd "$ROOT"
exec bash psm-model/scripts/runpod_hf_lora_eval_only.sh
