#!/usr/bin/env bash
# One-time Google OAuth for colab CLI (must run in an interactive WSL terminal).
set -euo pipefail
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
if ! command -v colab >/dev/null; then
  echo "Run colab_wsl_setup.sh first" >&2
  exit 1
fi
echo "Open the URL below, approve, paste the code when prompted."
colab sessions
