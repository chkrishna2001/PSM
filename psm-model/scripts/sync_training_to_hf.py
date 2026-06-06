#!/usr/bin/env python3
"""Upload PSM training artifacts to a private HF repo and prune old local checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STEP_SUFFIX = re.compile(r"-step-(\d+)\.pt$")
MANIFEST_NAME = ".hf_sync_manifest.json"


def _step_from_path(path: Path) -> int | None:
    match = STEP_SUFFIX.search(path.name)
    return int(match.group(1)) if match else None


def _related_checkpoint_files(step_path: Path) -> list[Path]:
    stem = step_path.stem
    suffix = step_path.suffix
    parent = step_path.parent
    return [
        path
        for path in (
            step_path,
            parent / f"{stem}.meta.json",
            parent / f"{stem}.tokenizer.json",
        )
        if path.exists()
    ]


def _hf_path(local_path: Path, *, repo_root: Path) -> str:
    return local_path.relative_to(repo_root).as_posix()


def _upload_files(api: object, repo_id: str, files: list[Path], *, repo_root: Path, dry_run: bool) -> None:
    from huggingface_hub import HfApi

    hf_api = api if isinstance(api, HfApi) else HfApi()
    for path in files:
        remote_path = _hf_path(path, repo_root=repo_root)
        if dry_run:
            print(json.dumps({"event": "upload_dry_run", "local": str(path), "remote": remote_path}))
            continue
        hf_api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"sync {remote_path}",
        )
        print(json.dumps({"event": "uploaded", "local": str(path), "remote": remote_path}))


def _load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def sync_training_artifacts(
    *,
    repo_id: str,
    checkpoint_dir: Path,
    run_stem: str,
    metrics_path: Path | None,
    repo_root: Path,
    keep_local: int,
    dry_run: bool,
    only_new: bool,
) -> dict[str, object]:
    if keep_local < 1:
        raise ValueError("--keep-local must be at least 1")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("install huggingface_hub: pip install huggingface_hub") from exc

    api = HfApi()
    uploaded: list[str] = []
    deleted: list[str] = []

    step_checkpoints = sorted(
        (path for path in checkpoint_dir.glob(f"{run_stem}-step-*.pt") if _step_from_path(path) is not None),
        key=lambda path: _step_from_path(path) or 0,
    )
    if not step_checkpoints:
        raise SystemExit(f"no step checkpoints found for {run_stem} in {checkpoint_dir}")

    manifest_path = checkpoint_dir / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)

    files_to_upload: list[Path] = []
    for step_path in step_checkpoints:
        files_to_upload.extend(_related_checkpoint_files(step_path))
    if metrics_path is not None and metrics_path.exists():
        files_to_upload.append(metrics_path)

    if only_new:
        pending: list[Path] = []
        for path in files_to_upload:
            rel = _hf_path(path, repo_root=repo_root)
            sig = _file_signature(path)
            if manifest.get(rel) == sig:
                continue
            pending.append(path)
        files_to_upload = pending

    if files_to_upload:
        _upload_files(api, repo_id, files_to_upload, repo_root=repo_root, dry_run=dry_run)
        if not dry_run:
            for path in files_to_upload:
                manifest[_hf_path(path, repo_root=repo_root)] = _file_signature(path)
            _save_manifest(manifest_path, manifest)
    uploaded.extend(str(path) for path in files_to_upload)

    keep_steps = {_step_from_path(path) for path in step_checkpoints[-keep_local:]}
    for step_path in step_checkpoints:
        step = _step_from_path(step_path)
        if step in keep_steps:
            continue
        for path in _related_checkpoint_files(step_path):
            if dry_run:
                print(json.dumps({"event": "delete_dry_run", "local": str(path)}))
            else:
                path.unlink()
            deleted.append(str(path))

    report = {
        "repo_id": repo_id,
        "latest_local_step": _step_from_path(step_checkpoints[-1]),
        "kept_local_steps": sorted(step for step in keep_steps if step is not None),
        "uploaded_count": len(uploaded),
        "deleted_count": len(deleted),
        "dry_run": dry_run,
    }
    print(json.dumps({"event": "sync_complete", **report}, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default="chkrishna2001/psm-50m-mixed-v1-run",
        help="Private HF model repo id (default: chkrishna2001/psm-50m-mixed-v1-run)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("psm-model/checkpoints"),
        help="Directory containing step checkpoints",
    )
    parser.add_argument(
        "--run-stem",
        default="real-v3-50m-action-mixed-v1",
        help="Checkpoint stem before -step-NNNNNN",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("psm-model/checkpoints/real-v3-50m-action-mixed-v1.metrics.jsonl"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Local repo root used to compute HF paths (default: current directory)",
    )
    parser.add_argument(
        "--keep-local",
        type=int,
        default=1,
        help="How many newest step checkpoints to retain locally after upload",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Skip files already uploaded with the same size/mtime (uses .hf_sync_manifest.json)",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    sync_training_artifacts(
        repo_id=args.repo,
        checkpoint_dir=(repo_root / args.checkpoint_dir).resolve(),
        run_stem=args.run_stem,
        metrics_path=(repo_root / args.metrics).resolve() if args.metrics else None,
        repo_root=repo_root,
        keep_local=args.keep_local,
        dry_run=args.dry_run,
        only_new=args.only_new,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
