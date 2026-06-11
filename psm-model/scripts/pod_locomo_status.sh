#!/bin/bash
cd "${PSM_REPO_ROOT:-/workspace/PSM}" 2>/dev/null || true
echo "node=$(node --version 2>/dev/null || echo missing)"
ls -la benchmark/locomo/results/locomo-psm-model-step-*-n25.log 2>/dev/null | tail -3 || echo "no locomo logs"
for f in benchmark/locomo/results/locomo-psm-model-step-*-n25.log; do
  [[ -f "$f" ]] && echo "--- tail $f ---" && tail -n 12 "$f"
done
pgrep -af 'runpod_locomo|ingest-psm|npm' | head -8 || echo "no locomo/npm procs"
