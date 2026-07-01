#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
SCRIPT_DIR="/mnt/c/Users/chkri/source/repos/PSM/psm-model/scripts"
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN missing — from PowerShell: o krishnachhftoken; \$env:WSLENV='HF_TOKEN/u'" >&2
  exit 1
fi
bash "$SCRIPT_DIR/colab_preflight.sh" "${1:-}"
exec bash "$SCRIPT_DIR/colab_locomo_hf_automate.sh" "$@"
