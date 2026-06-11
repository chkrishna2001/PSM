"""Tar-push specific repo files to a RunPod pod (proxy-safe; never SCP).

Usage:
    python psm-model/scripts/push_pod_files.py --proxy-user <pod>-<suffix> file1 [file2 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as ctl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-user", required=True)
    parser.add_argument("--host", default="ssh.runpod.io")
    parser.add_argument("--port", default="22")
    parser.add_argument("--remote-root", default="/workspace/PSM")
    parser.add_argument("files", nargs="+", help="Repo-relative file paths.")
    args = parser.parse_args()

    for rel in args.files:
        if not (REPO / rel).is_file():
            raise SystemExit(f"missing local file: {rel}")

    rc = ctl._push_repo_files_via_tar(
        ctl.SSH_CONFIG_HOST,
        REPO,
        args.files,
        args.remote_root,
        host=args.host,
        port=args.port,
        user=args.proxy_user,
    )
    print(f"push exit: {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
