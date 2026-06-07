#!/bin/bash
# Upload Gate 4 checkpoints to HF and prune old local step saves.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-chkrishna2001/psm-50m-mixed-v1-run}"
RUN_STEM="${RUN_STEM:-real-v3-50m-full-v2}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"

cd "$ROOT"
pip install -q huggingface_hub hf_transfer 2>/dev/null || true

echo "=== PSM Gate 4 HF upload $(date -u +%Y-%m-%dT%H:%M:%SZ) repo=$MODEL_REPO keep-local=$KEEP_LOCAL ==="
df -h /workspace || true

MIN_CKPT_BYTES="${MIN_CKPT_BYTES:-500000000}"
CKPT_DIR="$ROOT/psm-model/checkpoints"
for path in "$CKPT_DIR"/${RUN_STEM}-step-*.pt; do
  [[ -f "$path" ]] || continue
  size=$(stat -c%s "$path" 2>/dev/null || echo 0)
  if [[ "$size" -lt "$MIN_CKPT_BYTES" ]]; then
    echo "Removing corrupt checkpoint ($size bytes): $path"
    rm -f "$path" "${path%.pt}.meta.json" "${path%.pt}.tokenizer.json"
  fi
done

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

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

repo_id = os.environ["HF_UPLOAD_REPO"]
root = Path(os.environ["HF_UPLOAD_ROOT"])
run_stem = os.environ["HF_UPLOAD_STEM"]
keep_local = int(os.environ["HF_KEEP_LOCAL"])
api = HfApi()
manifest_path = root / "psm-model/checkpoints/.hf_sync_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

step_re = re.compile(r"-step-(\d+)\.pt$")
ckpt_dir = root / "psm-model/checkpoints"
steps = sorted(
    (p for p in ckpt_dir.glob(f"{run_stem}-step-*.pt") if step_re.search(p.name)),
    key=lambda p: int(step_re.search(p.name).group(1)),
)
if not steps:
    raise SystemExit(f"no checkpoints for {run_stem} in {ckpt_dir}")

def related(step_path: Path) -> list[Path]:
    return [p for p in (step_path, step_path.with_suffix(".tokenizer.json"), step_path.with_name(step_path.stem + ".meta.json")) if p.exists()]

def sig(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"

uploaded = []
for step_path in steps:
    for path in related(step_path):
        remote = path.relative_to(root).as_posix()
        if manifest.get(remote) == sig(path):
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"sync {remote}",
        )
        manifest[remote] = sig(path)
        uploaded.append({"remote": remote, "bytes": path.stat().st_size})
        print(json.dumps({"event": "uploaded", "remote": remote, "bytes": path.stat().st_size}))

for name in (
    f"{run_stem}.pt",
    f"{run_stem}.tokenizer.json",
    f"{run_stem}.meta.json",
    "real-v3-50m-full-v2-gate4.metrics.jsonl",
):
    path = ckpt_dir / name
    if not path.exists():
        continue
    remote = path.relative_to(root).as_posix()
    if manifest.get(remote) == sig(path):
        continue
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"sync {remote}",
    )
    manifest[remote] = sig(path)
    uploaded.append({"remote": remote, "bytes": path.stat().st_size})
    print(json.dumps({"event": "uploaded", "remote": remote, "bytes": path.stat().st_size}))

manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

keep_steps = {int(step_re.search(p.name).group(1)) for p in steps[-keep_local:]}
deleted = []
for step_path in steps:
    step = int(step_re.search(step_path.name).group(1))
    if step in keep_steps:
        continue
    for path in related(step_path):
        path.unlink()
        deleted.append(str(path))
        print(json.dumps({"event": "deleted_local", "path": str(path)}))

print(
    json.dumps(
        {
            "event": "upload_complete",
            "repo_id": repo_id,
            "latest_step": int(step_re.search(steps[-1].name).group(1)),
            "kept_local_steps": sorted(keep_steps),
            "uploaded_count": len(uploaded),
            "deleted_count": len(deleted),
        },
        sort_keys=True,
    )
)
PY

df -h /workspace || true
LATEST=$(ls -1 "$CKPT_DIR"/${RUN_STEM}-step-*.pt 2>/dev/null | sort -t- -k3 -n | tail -1 || true)
if [[ -n "$LATEST" ]]; then
  echo "RESUME_CHECKPOINT=${LATEST#$ROOT/}"
  echo "RESUME_STEP=$(basename "$LATEST" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"
fi
echo "=== Gate 4 HF upload done $(date -u +%H:%M:%SZ) ==="
