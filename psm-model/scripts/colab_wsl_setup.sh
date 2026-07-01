#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv pipx curl ca-certificates
pipx ensurepath
export PATH="$HOME/.local/bin:$PATH"
pipx install google-colab-cli
colab version
