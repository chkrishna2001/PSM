from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prod_memory.hf_assets import (
    CHECKPOINT_REL,
    CURRICULUM_REL,
    DEFAULT_DATASET_REPO,
    DEFAULT_MODEL_REPO,
    DIRECT_PROBE_REL,
    EXPANDED_PROBE_REL,
    MANIFEST_REL,
    META_REL,
    TOKENIZER_REL,
)


def _hf_download(
    repo_id: str,
    filenames: list[str],
    *,
    repo_type: str,
    local_dir: Path,
    token: str | None = None,
) -> list[str]:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for filename in filenames:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(local_dir),
            token=token,
        )
        downloaded.append(path)
        print(json.dumps({"event": "downloaded", "repo": repo_id, "file": filename, "local": path}))
    return downloaded


def _download_first_available(
    repo_id: str,
    candidates: list[str],
    *,
    repo_type: str,
    local_dir: Path,
    token: str | None = None,
) -> str | None:
    from huggingface_hub.utils import EntryNotFoundError

    errors: tuple[type[Exception], ...] = (EntryNotFoundError,)
    try:
        from huggingface_hub.errors import RemoteEntryNotFoundError

        errors = (*errors, RemoteEntryNotFoundError)
    except ImportError:
        pass

    for filename in candidates:
        try:
            _hf_download(repo_id, [filename], repo_type=repo_type, local_dir=local_dir, token=token)
            return filename
        except errors:
            continue
    return None


def download_colab_assets(root: Path) -> dict[str, Any]:
    import os

    model_token = os.environ.get("HF_TOKEN") or None
    dataset_token = os.environ.get("DATASET_HF_TOKEN") or model_token

    root.mkdir(parents=True, exist_ok=True)
    model_files = _hf_download(
        DEFAULT_MODEL_REPO,
        [CHECKPOINT_REL, TOKENIZER_REL, META_REL],
        repo_type="model",
        local_dir=root,
        token=model_token,
    )
    dataset_files = _hf_download(
        DEFAULT_DATASET_REPO,
        [CURRICULUM_REL, MANIFEST_REL],
        repo_type="dataset",
        local_dir=root,
        token=dataset_token,
    )
    expanded = _download_first_available(
        DEFAULT_DATASET_REPO,
        [
            EXPANDED_PROBE_REL,
            "probes/expanded-probe-v1-filtered.jsonl",
            "data/probes/expanded-probe-v1-filtered.jsonl",
        ],
        repo_type="dataset",
        local_dir=root,
        token=dataset_token,
    )
    direct = _download_first_available(
        DEFAULT_DATASET_REPO,
        [DIRECT_PROBE_REL, "probes/direct_probes.jsonl", "data/probes/direct_probes.jsonl"],
        repo_type="dataset",
        local_dir=root,
        token=dataset_token,
    )
    return {
        "root": str(root),
        "model_repo": DEFAULT_MODEL_REPO,
        "dataset_repo": DEFAULT_DATASET_REPO,
        "model_files": model_files,
        "dataset_files": dataset_files,
        "expanded_probe": expanded,
        "direct_probe": direct,
        "checkpoint": str(root / CHECKPOINT_REL),
        "curriculum": str(root / CURRICULUM_REL),
    }


def upload_step_checkpoints(root: Path, *, run_stem: str, steps: list[int], repo_id: str = DEFAULT_MODEL_REPO) -> dict[str, Any]:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    uploaded: list[str] = []
    ckpt_dir = root / "psm-model" / "checkpoints"
    for step in steps:
        stem = f"{run_stem}-step-{step:06d}"
        for suffix in (".pt", ".tokenizer.json", ".meta.json"):
            path = ckpt_dir / f"{stem}{suffix}"
            if not path.exists():
                continue
            remote = path.relative_to(root).as_posix()
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"prod-extraction colab: {remote}",
            )
            uploaded.append(remote)
            print(json.dumps({"event": "uploaded", "remote": remote}))
    return {"repo_id": repo_id, "uploaded": uploaded}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/upload HF assets for prod-extraction Colab runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download resume checkpoint, curriculum, and probes.")
    download.add_argument("--root", type=Path, default=Path("."))

    upload = sub.add_parser("upload-steps", help="Upload completed step checkpoints to HF model repo.")
    upload.add_argument("--root", type=Path, default=Path("."))
    upload.add_argument("--run-stem", default="real-v3-50m-full-v2")
    upload.add_argument("--steps", type=int, nargs="+", required=True)
    upload.add_argument("--repo", default=DEFAULT_MODEL_REPO)

    args = parser.parse_args(argv)
    if args.command == "download":
        report = download_colab_assets(args.root)
    else:
        report = upload_step_checkpoints(args.root, run_stem=args.run_stem, steps=args.steps, repo_id=args.repo)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
