#!/usr/bin/env python3
"""Upload clean prod-memory HF curriculum + teacher v3 to krishnach7262 dataset repo."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_DATASET_REPO = "krishnach7262/psm-prod-memory-data"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("PSM_HF_DATASET_REPO", DEFAULT_DATASET_REPO))
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=Path("psm-model/prod-memory/data/hf-prod-v1.jsonl"),
    )
    parser.add_argument(
        "--source-v3",
        type=Path,
        default=Path("psm-model/prod-memory/data/prod-extraction-v3.jsonl"),
    )
    parser.add_argument(
        "--source-v5",
        type=Path,
        default=Path("psm-model/prod-memory/data/prod-extraction-v5.jsonl"),
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN required (o krishnachhftoken)", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    try:
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=True)
    except Exception as exc:
        print(f"create_repo: {exc}", file=sys.stderr)

    uploads: list[tuple[Path, str]] = []
    for local, remote in (
        (args.curriculum, f"prod-memory/{args.curriculum.name}"),
        (args.curriculum.with_suffix(".manifest.json"), f"prod-memory/{args.curriculum.stem}.manifest.json"),
        (args.source_v3, f"prod-memory/{args.source_v3.name}"),
        (args.source_v3.with_suffix(".manifest.json"), f"prod-memory/{args.source_v3.stem}.manifest.json"),
        (args.source_v5, f"prod-memory/{args.source_v5.name}"),
    ):
        if local.exists():
            uploads.append((local, remote))

    for local, remote in uploads:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"upload {remote}",
        )
        print(json.dumps({"uploaded": remote, "bytes": local.stat().st_size}))

    print(json.dumps({"repo": args.repo, "files": len(uploads)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
