"""Pull a remote pod directory to a local path via tar+base64 (proxy-safe; never SCP).

Usage:
    python psm-model/scripts/pull_pod_dir.py --proxy-user <pod>-<suffix> REMOTE_DIR LOCAL_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as ctl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-user", required=True)
    parser.add_argument("--host", default="ssh.runpod.io")
    parser.add_argument("--port", default="22")
    parser.add_argument("remote_dir")
    parser.add_argument("local_dir")
    args = parser.parse_args()

    rc = ctl._ssh_pull_dir(
        ctl.SSH_CONFIG_HOST,
        args.remote_dir,
        Path(args.local_dir),
        host=args.host,
        port=args.port,
        user=args.proxy_user,
    )
    print(f"pull exit: {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
