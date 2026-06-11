#!/bin/bash
head -n 4 /workspace/PSM/psm-model/scripts/runpod_upload_gate4.sh | cat -A
tmux ls 2>/dev/null || echo "no tmux"
bash -n /workspace/PSM/psm-model/scripts/runpod_upload_gate4.sh && echo "upload_sh_syntax_ok"
