#!/bin/bash
# Upload Gate 4 checkpoints to HF. Default: upload every step file; never delete local until HF manifest match.
# Final sync (GATE4_FINAL_SYNC=1): keep registry best only locally + on HF, exit non-zero if upload incomplete.
set -euo pipefail

ROOT="${PSM_REPO_ROOT:-/workspace/PSM}"
MODEL_REPO="${PSM_HF_MODEL_REPO:-subbu83/psm-50m-mixed-v1-run}"
DATASET_REPO="${PSM_HF_DATASET_REPO:-chkrishna2001/psm-50m-action-mixed-v1}"
RUN_STEM="${RUN_STEM:-real-v3-50m-full-v2}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"
UPLOAD_ALL="${UPLOAD_ALL:-1}"
GATE4_FINAL_SYNC="${GATE4_FINAL_SYNC:-0}"
GATE4_KEEP_BEST_ONLY="${GATE4_KEEP_BEST_ONLY:-0}"

cd "$ROOT"
pip install -q huggingface_hub hf_transfer 2>/dev/null || true

echo "=== PSM Gate 4 HF upload $(date -u +%Y-%m-%dT%H:%M:%SZ) repo=$MODEL_REPO upload_all=$UPLOAD_ALL final=$GATE4_FINAL_SYNC keep_best=$GATE4_KEEP_BEST_ONLY ==="
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
export HF_UPLOAD_DATASET_REPO="$DATASET_REPO"
export HF_UPLOAD_ROOT="$ROOT"
export HF_UPLOAD_STEM="$RUN_STEM"
export HF_KEEP_LOCAL="$KEEP_LOCAL"
export HF_UPLOAD_ALL="$UPLOAD_ALL"
export HF_GATE4_FINAL_SYNC="$GATE4_FINAL_SYNC"
export HF_GATE4_KEEP_BEST_ONLY="$GATE4_KEEP_BEST_ONLY"
export HF_GATE4_PINNED_STEPS="${GATE4_PINNED_STEPS:-}"

python3 - <<'PY'
import json
import os
import re
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, create_commit

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

repo_id = os.environ["HF_UPLOAD_REPO"]
dataset_repo = os.environ["HF_UPLOAD_DATASET_REPO"]
root = Path(os.environ["HF_UPLOAD_ROOT"])
run_stem = os.environ["HF_UPLOAD_STEM"]
keep_local = int(os.environ["HF_KEEP_LOCAL"])
upload_all = os.environ.get("HF_UPLOAD_ALL", "1") == "1"
final_sync = os.environ.get("HF_GATE4_FINAL_SYNC", "0") == "1"
keep_best_only = os.environ.get("HF_GATE4_KEEP_BEST_ONLY", "0") == "1"
pinned_env = os.environ.get("HF_GATE4_PINNED_STEPS", "")
pinned_from_env = {int(x) for x in pinned_env.split(",") if x.strip().isdigit()}

api = HfApi()
manifest_path = root / "psm-model/checkpoints/.hf_sync_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

registry_path = root / "psm-model/checkpoints/gate4-checkpoint-registry.json"
registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
best = registry.get("best") or {}
best_step = int(best["step"]) if best.get("step") is not None else None
if best_step is not None:
    pinned_from_env.add(best_step)

step_re = re.compile(r"-step-(\d+)\.pt$")
ckpt_dir = root / "psm-model/checkpoints"
steps = sorted(
    (p for p in ckpt_dir.glob(f"{run_stem}-step-*.pt") if step_re.search(p.name)),
    key=lambda p: int(step_re.search(p.name).group(1)),
)
if not steps:
    raise SystemExit(f"no checkpoints for {run_stem} in {ckpt_dir}")


def related(step_path: Path) -> list[Path]:
    return [
        p
        for p in (
            step_path,
            step_path.with_suffix(".tokenizer.json"),
            step_path.with_name(step_path.stem + ".meta.json"),
        )
        if p.exists()
    ]


def sig(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def fully_synced(step_path: Path) -> bool:
    return all(manifest.get(p.relative_to(root).as_posix()) == sig(p) for p in related(step_path))


def batch_upload(repo: str, repo_type: str, pending: list[tuple[str, Path]]) -> list[str]:
    if not pending:
        return []
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)) for remote, path in pending]
    create_commit(
        repo_id=repo,
        repo_type=repo_type,
        operations=operations,
        commit_message=f"Gate4 sync {len(operations)} files",
    )
    uploaded: list[str] = []
    for remote, path in pending:
        manifest[remote] = sig(path)
        uploaded.append(remote)
        print(json.dumps({"event": "uploaded", "remote": remote, "bytes": path.stat().st_size}))
    return uploaded


latest_step = int(step_re.search(steps[-1].name).group(1))
if keep_best_only and best_step is not None:
    keep_steps = {best_step}
else:
    keep_steps = {int(step_re.search(p.name).group(1)) for p in steps[-keep_local:]} | pinned_from_env

upload_targets = steps if upload_all or final_sync else [p for p in steps if int(step_re.search(p.name).group(1)) in keep_steps]

uploaded: list[str] = []
errors: list[dict] = []
model_pending: list[tuple[str, Path]] = []
for step_path in upload_targets:
    step = int(step_re.search(step_path.name).group(1))
    for path in related(step_path):
        remote = path.relative_to(root).as_posix()
        if manifest.get(remote) == sig(path):
            continue
        model_pending.append((remote, path))

for name in (
    f"{run_stem}.pt",
    f"{run_stem}.tokenizer.json",
    f"{run_stem}.meta.json",
    f"{run_stem}-gate4.metrics.jsonl",
    "gate4-checkpoint-registry.json",
):
    path = ckpt_dir / name
    if not path.exists():
        continue
    remote = path.relative_to(root).as_posix()
    if manifest.get(remote) != sig(path):
        model_pending.append((remote, path))

try:
    uploaded.extend(batch_upload(repo_id, "model", model_pending))
except Exception as exc:  # noqa: BLE001
    for remote, path in model_pending:
        errors.append({"remote": remote, "error": str(exc)})
        print(json.dumps({"event": "upload_error", "remote": remote, "error": str(exc)}))

for step_path in upload_targets:
    if fully_synced(step_path):
        step = int(step_re.search(step_path.name).group(1))
        print(json.dumps({"event": "step_synced", "step": step}))

dataset_pending: list[tuple[str, Path]] = []
eval_dir = ckpt_dir / "gate-eval"
if final_sync and eval_dir.is_dir():
    for eval_path in sorted(eval_dir.glob("gate4-full-expanded-step-*.json")):
        remote = f"curriculum/{eval_path.name}"
        dataset_pending.append((remote, eval_path))

if dataset_pending:
    try:
        batch_upload(dataset_repo, "dataset", dataset_pending)
        for remote, _ in dataset_pending:
            print(json.dumps({"event": "uploaded_eval", "remote": remote, "dataset": dataset_repo}))
    except Exception as exc:  # noqa: BLE001
        for remote, _ in dataset_pending:
            errors.append({"remote": remote, "error": str(exc)})
            print(json.dumps({"event": "upload_eval_error", "remote": remote, "error": str(exc)}))

manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

deleted = []
for step_path in steps:
    step = int(step_re.search(step_path.name).group(1))
    if step in keep_steps:
        continue
    if not fully_synced(step_path):
        print(json.dumps({"event": "skip_local_delete", "step": step, "reason": "not_fully_synced_to_hf"}))
        continue
    for path in related(step_path):
        path.unlink()
        deleted.append(str(path))
        print(json.dumps({"event": "deleted_local", "path": str(path)}))

verify_steps = set(keep_steps)
if best_step is not None:
    verify_steps.add(best_step)
missing_local = [step for step in sorted(verify_steps) if not any(int(step_re.search(p.name).group(1)) == step for p in steps if fully_synced(p))]

print(
    json.dumps(
        {
            "event": "upload_complete",
            "repo_id": repo_id,
            "latest_step": latest_step,
            "best_step": best_step,
            "kept_local_steps": sorted(keep_steps),
            "uploaded_count": len(uploaded),
            "deleted_count": len(deleted),
            "error_count": len(errors),
            "final_sync": final_sync,
            "keep_best_only": keep_best_only,
            "missing_local_sync": missing_local,
        },
        sort_keys=True,
    )
)

if errors and (final_sync or os.environ.get("GATE4_REQUIRE_SYNC") == "1"):
    sys.exit(1)
if final_sync and best_step is not None:
    best_paths = [p for p in steps if int(step_re.search(p.name).group(1)) == best_step]
    if not best_paths or not fully_synced(best_paths[0]):
        print(json.dumps({"event": "final_sync_failed", "reason": "best_step_not_synced", "best_step": best_step}))
        sys.exit(1)
if final_sync and missing_local:
    print(json.dumps({"event": "final_sync_failed", "reason": "required_steps_not_synced", "steps": missing_local}))
    sys.exit(1)
PY
UPLOAD_RC=$?

if [[ "$GATE4_FINAL_SYNC" == "1" && -f "$CKPT_DIR/gate4-checkpoint-registry.json" ]]; then
  if [[ "$GATE4_KEEP_BEST_ONLY" == "1" ]]; then
    python3 -m psm_model.gate4_checkpoint_registry prune-hf-keep-best \
      --repo-id "$MODEL_REPO" \
      --run-stem "$RUN_STEM" \
      --checkpoint-dir "$CKPT_DIR" || true
  fi
  python3 -m psm_model.gate4_checkpoint_registry prune-hf \
    --repo-id "$MODEL_REPO" \
    --run-stem "$RUN_STEM" \
    --checkpoint-dir "$CKPT_DIR" || true
fi

df -h /workspace || true
LATEST=$(ls -1 "$CKPT_DIR"/${RUN_STEM}-step-*.pt 2>/dev/null | sort -t- -k3 -n | tail -1 || true)
if [[ -n "$LATEST" ]]; then
  echo "RESUME_CHECKPOINT=${LATEST#$ROOT/}"
  echo "RESUME_STEP=$(basename "$LATEST" | sed -n 's/.*-step-\([0-9]*\)\.pt/\1/p')"
fi
if [[ "$UPLOAD_RC" -ne 0 ]]; then
  echo "GATE4_UPLOAD_FAILED=1" >&2
  exit "$UPLOAD_RC"
fi
echo "GATE4_UPLOAD_OK=1"
echo "=== Gate 4 HF upload done $(date -u +%H:%M:%SZ) ==="
