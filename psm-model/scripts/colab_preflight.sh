#!/usr/bin/env bash
# Preflight before Colab LoCoMo launch (WSL).
set -euo pipefail

REPO_WIN="/mnt/c/Users/chkri/source/repos/PSM"
CHECKPOINT="${1:-$REPO_WIN/benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-n2960-checkpoint.db}"
export PATH="$HOME/.local/bin:$PATH"

err() { echo "preflight FAIL: $*" >&2; exit 1; }

command -v colab >/dev/null || err "colab not found — run colab_wsl_setup.sh"

[[ -n "${HF_TOKEN:-}" ]] || err "HF_TOKEN unset — use: o krishnachhftoken then WSLENV=HF_TOKEN/u from PowerShell"

[[ -f "$CHECKPOINT" ]] || err "checkpoint missing: $CHECKPOINT"

if ! colab sessions >/dev/null 2>&1; then
  err "colab OAuth required — run: bash $REPO_WIN/psm-model/scripts/colab_auth_wsl.sh"
fi

echo "preflight OK (colab, HF_TOKEN, checkpoint)"
