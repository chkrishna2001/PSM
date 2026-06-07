#!/bin/bash
# Upload latest Gate 4 full-model checkpoints + metrics to HF (not full history).
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
RUN_STEM="${RUN_STEM:-real-v3-50m-full-v2}"
KEEP_LOCAL="${KEEP_LOCAL:-3}"

cd "$ROOT"
pip install -q huggingface_hub hf_transfer 2>/dev/null || true

echo "=== PSM Gate 4 HF upload $(date -u +%Y-%m-%dT%H:%M:%SZ) repo=$MODEL_REPO ==="

export HF_UPLOAD_REPO="$MODEL_REPO"
export HF_UPLOAD_ROOT="$ROOT"
export HF_UPLOAD_STEM="$RUN_STEM"
export HF_KEEP_LOCAL="$KEEP_LOCAL"

python3 - <<'PY'
import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi

# Quiet tqdm progress bars — raw ANSI breaks some Windows SSH decoders.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

repo_id = os.environ["HF_UPLOAD_REPO"]
root = Path(os.environ["HF_UPLOAD_ROOT"])
run_stem = os.environ["HF_UPLOAD_STEM"]
keep_local = int(os.environ["HF_KEEP_LOCAL"])
api = HfApi()

step_re = re.compile(r"-step-(\d+)\.pt$")
ckpt_dir = root / "psm-model/checkpoints"
steps = sorted(
    (p for p in ckpt_dir.glob(f"{run_stem}-step-*.pt") if step_re.search(p.name)),
    key=lambda p: int(step_re.search(p.name).group(1)),
)
if not steps:
    raise SystemExit(f"no checkpoints for {run_stem} in {ckpt_dir}")

upload_paths: list[Path] = []
for step_path in steps[-keep_local:]:
    for suffix in ("", ".tokenizer.json", ".meta.json"):
        candidate = step_path.with_name(step_path.stem + suffix) if suffix else step_path
        if candidate.exists():
            upload_paths.append(candidate)

for name in (
    "real-v3-50m-full-v2.pt",
    "real-v3-50m-full-v2.tokenizer.json",
    "real-v3-50m-full-v2.meta.json",
    "real-v3-50m-full-v2-gate4.metrics.jsonl",
):
    path = ckpt_dir / name
    if path.exists() and path not in upload_paths:
        upload_paths.append(path)

uploaded = []
for path in upload_paths:
    remote = path.relative_to(root).as_posix()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"sync {remote}",
    )
    uploaded.append({"remote": remote, "bytes": path.stat().st_size})
    print(json.dumps({"event": "uploaded", "remote": remote, "bytes": path.stat().st_size}))

print(
    json.dumps(
        {
            "event": "upload_complete",
            "repo_id": repo_id,
            "latest_step": int(step_re.search(steps[-1].name).group(1)),
            "uploaded_count": len(uploaded),
            "files": uploaded,
        },
        sort_keys=True,
    )
)
PY

echo "=== Gate 4 HF upload done $(date -u +%H:%M:%SZ) ==="
