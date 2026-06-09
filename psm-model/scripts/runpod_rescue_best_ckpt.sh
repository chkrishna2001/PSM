#!/bin/bash
# Rescue: upload best Gate 4 checkpoint (.pt) to HF in one commit; do not delete locals.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
RUN_STEM="${RUN_STEM:-real-v3-50m-full-v2}"
BEST_STEP="${GATE4_BEST_STEP:-42000}"

cd "$ROOT"
export PYTHONPATH=psm-model/src
pip install -q huggingface_hub hf_transfer 2>/dev/null || true

CKPT="psm-model/checkpoints/${RUN_STEM}-step-$(printf '%06d' "$BEST_STEP").pt"
TOK="${CKPT%.pt}.tokenizer.json"
META="${CKPT%.pt}.meta.json"

for f in "$CKPT" "$TOK"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING required file: $f" >&2
    ls -la psm-model/checkpoints/${RUN_STEM}-step-*.pt 2>/dev/null | tail -20 || true
    exit 1
  fi
done

echo "=== Rescue upload step $BEST_STEP $(date -u +%H:%M:%SZ) ==="
ls -lh "$CKPT" "$TOK" ${META:+$META} 2>/dev/null || true

python3 - <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = os.environ["PSM_HF_MODEL_REPO"]
root = Path("$ROOT")
paths = [root / "$CKPT", root / "$TOK"]
meta = root / "$META"
if meta.exists():
    paths.append(meta)

api = HfApi()
for path in paths:
    remote = path.relative_to(root).as_posix()
    print(f"uploading {path} -> {remote}")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"rescue best gate4 checkpoint step $BEST_STEP ({remote})",
    )
    print(f"OK {remote} bytes={path.stat().st_size}")

print("RESCUE_OK step=$BEST_STEP")
PY

echo "=== Rescue done ==="
