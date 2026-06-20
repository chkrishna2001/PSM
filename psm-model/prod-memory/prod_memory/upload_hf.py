from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from prod_memory.hf_assets import (
    CURRICULUM_REL,
    DEFAULT_CURRICULUM_REPO,
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    LOCAL_CURRICULUM,
    LOCAL_MANIFEST,
    MANIFEST_REL,
)


def upload_prod_curriculum(
    *,
    repo_id: str,
    repo_type: str,
    curriculum: Path,
    manifest: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not curriculum.exists():
        raise FileNotFoundError(f"curriculum not found: {curriculum}")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    uploads = [
        (CURRICULUM_REL, curriculum),
        (MANIFEST_REL, manifest),
    ]

    if dry_run:
        return {
            "repo_id": repo_id,
            "repo_type": repo_type,
            "dry_run": True,
            "files": [{"remote": remote, "local": str(path)} for remote, path in uploads],
        }

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("install huggingface_hub: pip install huggingface_hub") from exc

    token = (
        os.environ.get("DATASET_HF_TOKEN") or os.environ.get("HF_TOKEN")
        if repo_type == "dataset"
        else os.environ.get("HF_TOKEN") or os.environ.get("DATASET_HF_TOKEN")
    )
    api = HfApi(token=token)
    uploaded: list[str] = []
    for remote, path in uploads:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"prod-memory: sync {remote}",
        )
        uploaded.append(remote)
        print(json.dumps({"event": "uploaded", "remote": remote, "bytes": path.stat().st_size, "repo": repo_id}))

    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "uploaded": uploaded,
        "curriculum_rows": sum(1 for line in curriculum.read_text(encoding="utf-8").splitlines() if line.strip()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload prod-extraction curriculum to Hugging Face.")
    parser.add_argument(
        "--repo",
        default=DEFAULT_CURRICULUM_REPO,
        help="Default: dataset repo (chkrishna2001/psm-50m-action-mixed-v1)",
    )
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="dataset")
    parser.add_argument("--curriculum", type=Path, default=LOCAL_CURRICULUM)
    parser.add_argument("--manifest", type=Path, default=LOCAL_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = upload_prod_curriculum(
        repo_id=args.repo,
        repo_type=args.repo_type,
        curriculum=args.curriculum,
        manifest=args.manifest,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
