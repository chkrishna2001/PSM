#!/usr/bin/env python3
"""Delete nano-psm Hugging Face repos (datasets + model checkpoints). Keeps psm-50m* repos."""

from __future__ import annotations

import argparse
import json
import os
import sys

from huggingface_hub import HfApi

NANO_PSM_REPOS = [
    # datasets
    "chkrishna2001/nano-psm",
    "chkrishna2001/nano-psm-raw-sources",
    "chkrishna2001/nano-psm-fast-mixed-10k",
    "chkrishna2001/nano-psm-fast-mixed-reviewed-5k",
    "chkrishna2001/nano-psm-fast-mixed-reviewed-v2-5k",
    "chkrishna2001/nano-psm-fast-mixed-reviewed-incremental-5k",
    "chkrishna2001/nano-psm-retention-decay-5k",
    "chkrishna2001/nano-psm-retention-blend-7k",
    "chkrishna2001/nano-psm-retention-blend-codex-84k",
    "chkrishna2001/nano-psm-retention-dominant-codex-9k",
    "chkrishna2001/nano-psm-codex-sessions-gpt41-mini-200",
    # model / checkpoint repos
    "chkrishna2001/nano-psm-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-fast-mixed-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-reviewed-5k-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-reviewed-v2-from-prev-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-incremental-5k-from-reviewed-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-retention-decay-from-reviewed-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-retention-blend-from-reviewed-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-retention-blend-codex-multival-from-blend-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-retention-dominant-codex-from-retention-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-retention-dominant-codex-selector-v2-checkpoints",
    "chkrishna2001/nano-psm-primary-10m-retention-dominant-codex-selector-v3-checkpoints",
]

KEEP_PREFIXES = (
    "chkrishna2001/psm-50m",
    "chkrishna2001/psm-memory",
)


def _resolve_repo_type(api: HfApi, repo_id: str) -> str | None:
    for repo_type in ("dataset", "model", "space"):
        try:
            api.repo_info(repo_id, repo_type=repo_type)
            return repo_type
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set HF_TOKEN (e.g. `o hftoken` → Get-Clipboard).")
    api = HfApi(token=token)

    # Also pick up any author repos matching nano-psm not in static list.
    discovered: set[str] = set(NANO_PSM_REPOS)
    for list_fn, _ in ((api.list_models, "model"), (api.list_datasets, "dataset"), (api.list_spaces, "space")):
        for row in list_fn(author="chkrishna2001", limit=500):
            rid = row.id
            if "/nano-psm" in rid and not any(rid.startswith(prefix) for prefix in KEEP_PREFIXES):
                discovered.add(rid)

    to_delete: list[tuple[str, str]] = []
    missing: list[str] = []
    for repo_id in sorted(discovered):
        if any(repo_id.startswith(prefix) for prefix in KEEP_PREFIXES):
            continue
        repo_type = _resolve_repo_type(api, repo_id)
        if repo_type is None:
            missing.append(repo_id)
            continue
        to_delete.append((repo_type, repo_id))

    print(json.dumps({"to_delete": to_delete, "missing": missing}, indent=2))
    if not to_delete:
        print("No nano-psm repos found on HF.")
        return 0
    if not args.yes and not args.dry_run:
        answer = input(f"Delete {len(to_delete)} HF repos? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    for repo_type, repo_id in to_delete:
        if args.dry_run:
            print(json.dumps({"event": "dry_run_delete", "repo_type": repo_type, "repo_id": repo_id}))
            continue
        try:
            api.delete_repo(repo_id=repo_id, repo_type=repo_type)
            deleted.append(repo_id)
            print(json.dumps({"event": "deleted", "repo_type": repo_type, "repo_id": repo_id}))
        except Exception as exc:  # noqa: BLE001
            errors.append({"repo_id": repo_id, "error": str(exc)})
            print(json.dumps({"event": "delete_error", "repo_id": repo_id, "error": str(exc)}))

    print(json.dumps({"deleted_count": len(deleted), "error_count": len(errors), "deleted": deleted}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
