#!/usr/bin/env python3
"""Deploy pod and run Experiment A (minimal format overfit)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"


def _ssh_args(pod_id: str, proxy_user: str) -> tuple[str, str | None, str | None, str]:
    args = argparse.Namespace(
        pod_id=pod_id,
        proxy_user=proxy_user,
        deploy=False,
        host_alias="runpod-psm-proxy",
        name="",
        image="",
        template="",
        gpu="",
        volume_gb=0,
        container_disk_gb=0,
        autostart=False,
        wait_ssh=0,
        ssh_ready_timeout_sec=300,
        auto_gpu=False,
    )
    return rc._resolve_train_pod_ssh(args, proxy_user=proxy_user)


def _push(host_alias: str, ssh_host, ssh_port, ssh_user) -> None:
    for local, remote in [
        (REPO / "psm-model" / "src", "/workspace/PSM/psm-model/src"),
        (REPO / "psm-model" / "prod-memory", "/workspace/PSM/psm-model/prod-memory"),
        (SCRIPTS, "/workspace/PSM/psm-model/scripts"),
    ]:
        rc._ssh_push_dir(host_alias, local, remote, host=ssh_host, port=ssh_port, user=ssh_user)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if args.deploy or not pod_id:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "deploy", "--auto-gpu", "--name", "psm-exp-a", "--wait-ssh", "300"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode
        import json as _json

        for line in proc.stdout.splitlines():
            if line.strip().startswith("{"):
                try:
                    payload = _json.loads(line)
                    if payload.get("pod_id"):
                        pod_id = payload["pod_id"]
                    if payload.get("targets"):
                        proxy_user = payload["targets"][0].get("user", proxy_user)
                except _json.JSONDecodeError:
                    pass
    if not pod_id:
        print("pod_id required", file=sys.stderr)
        return 1
    if not proxy_user:
        print("proxy_user required", file=sys.stderr)
        return 1

    _, ssh_host, ssh_port, ssh_user = _ssh_args(pod_id, proxy_user)
    _push("runpod-psm-proxy", ssh_host, ssh_port, ssh_user)

    extra = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "DATASET_HF_TOKEN": os.environ.get("DATASET_HF_TOKEN", ""),
        "PSM_REPO_ROOT": "/workspace/PSM",
        "EXP_A_CONTEXT_LENGTH": "2048",
    }
    script = SCRIPTS / "runpod_exp_a_minimal_overfit.sh"
    rc_ = rc._ssh_run_script(
        "runpod-psm-proxy",
        script,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=5400,
        extra_env=extra,
        skip_ssh_wait=True,
    )
    print(f"exp-a exit {rc_}")
    return rc_


if __name__ == "__main__":
    raise SystemExit(main())
