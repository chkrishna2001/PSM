"""Run a local bash script on a RunPod pod over proxy SSH (proxy-safe; never SCP).

Usage:
    python psm-model/scripts/run_pod_script.py --proxy-user <pod>-<suffix> \
        [--env KEY=VALUE ...] [--timeout-sec N] path/to/script.sh

HF_TOKEN, DATASET_HF_TOKEN, PSM_HF_MODEL_REPO, PSM_HF_DATASET_REPO are
forwarded from the local environment automatically when set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as ctl  # noqa: E402

PASSTHROUGH_ENV = ("HF_TOKEN", "DATASET_HF_TOKEN", "PSM_HF_MODEL_REPO", "PSM_HF_DATASET_REPO")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-user", required=True)
    parser.add_argument("--host", default="ssh.runpod.io")
    parser.add_argument("--port", default="22")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("script")
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.is_file():
        raise SystemExit(f"missing script: {script_path}")

    extra_env: dict[str, str] = {}
    for name in PASSTHROUGH_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            extra_env[name] = value
    for kv in args.env:
        key, _, value = kv.partition("=")
        if not key or not value:
            raise SystemExit(f"bad --env (want KEY=VALUE): {kv}")
        extra_env[key] = value

    rc = ctl._ssh_run_script(
        ctl.SSH_CONFIG_HOST,
        script_path,
        host=args.host,
        port=args.port,
        user=args.proxy_user,
        timeout_sec=args.timeout_sec,
        extra_env=extra_env,
        skip_ssh_wait=True,
    )
    print(f"script exit: {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
