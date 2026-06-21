#!/usr/bin/env python3
"""Deploy pod and run Experiment E (scratch extract-only 50M)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"


def _deploy() -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "deploy", "--auto-gpu", "--name", "psm-exp-e", "--wait-ssh", "300"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    pod_id = ""
    proxy_user = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        pod_id = payload.get("pod_id") or payload.get("id") or pod_id
        proxy_user = payload.get("pod_host_id") or proxy_user
        for target in payload.get("targets") or []:
            if target.get("user"):
                proxy_user = target["user"]
    return pod_id, proxy_user


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if args.deploy and not pod_id:
        pod_id, proxy_user = _deploy()
    if not pod_id or not proxy_user:
        print("pod_id and proxy_user required", file=sys.stderr)
        return 1

    ns = argparse.Namespace(
        pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, ssh_host, ssh_port, ssh_user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)

    for local, remote in [
        (REPO / "psm-model" / "src", "/workspace/PSM/psm-model/src"),
        (REPO / "psm-model" / "prod-memory", "/workspace/PSM/psm-model/prod-memory"),
        (SCRIPTS, "/workspace/PSM/psm-model/scripts"),
    ]:
        rc._ssh_push_dir("runpod-psm-proxy", local, remote, host=ssh_host, port=ssh_port, user=ssh_user)

    extra = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "DATASET_HF_TOKEN": os.environ.get("DATASET_HF_TOKEN", ""),
        "PSM_REPO_ROOT": "/workspace/PSM",
    }
    rc_ = rc._ssh_run_script(
        "runpod-psm-proxy",
        SCRIPTS / "runpod_exp_e_scratch_extract.sh",
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=10800,
        extra_env=extra,
        skip_ssh_wait=True,
    )
    print(f"exp-e exit {rc_}")
    return rc_ if isinstance(rc_, int) else rc_[0]


if __name__ == "__main__":
    raise SystemExit(main())
