#!/bin/bash
set -euo pipefail

if [[ -f /workspace/PSM/psm-model/scripts/runpod_bootstrap.sh ]]; then
  bash /workspace/PSM/psm-model/scripts/runpod_bootstrap.sh || true
fi

exec "$@"
