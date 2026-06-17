from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prod_memory.hf_assets import (
    CURRICULUM_REL,
    DEFAULT_DATASET_REPO,
    LOCAL_CURRICULUM,
    LOCAL_MANIFEST,
    MANIFEST_REL,
)


def upload_prod_dataset(
    *,
    repo_id: str,
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
            "repo_type": "dataset",
            "dry_run": True,
            "files": [{"remote": remote, "local": str(path)} for remote, path in uploads],
        }

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("install huggingface_hub: pip install huggingface_hub") from exc

    import os

    token = os.environ.get("DATASET_HF_TOKEN") or os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    uploaded: list[str] = []
    for remote, path in uploads:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"prod-memory: sync {remote}",
        )
        uploaded.append(remote)
        print(json.dumps({"event": "uploaded", "remote": remote, "bytes": path.stat().st_size}))

    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "uploaded": uploaded,
        "curriculum_rows": sum(1 for line in curriculum.read_text(encoding="utf-8").splitlines() if line.strip()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload prod-extraction-v1 curriculum to Hugging Face dataset repo.")
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--curriculum", type=Path, default=LOCAL_CURRICULUM)
    parser.add_argument("--manifest", type=Path, default=LOCAL_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = upload_prod_dataset(
        repo_id=args.repo,
        curriculum=args.curriculum,
        manifest=args.manifest,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
