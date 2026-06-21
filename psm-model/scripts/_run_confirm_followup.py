#!/usr/bin/env python3
"""Push fixes and re-run legacy donor + overfit eval on confirm pod."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

POD_ID = "gpnaavate1i0i4"
PROXY_USER = "gpnaavate1i0i4-64410f3d"
HOST_ALIAS = "runpod-psm-proxy"


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    scripts = repo / "psm-model" / "scripts"
    args = argparse.Namespace(
        pod_id=POD_ID,
        proxy_user=PROXY_USER,
        deploy=False,
        host_alias=HOST_ALIAS,
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
    _, ssh_host, ssh_port, ssh_user = rc._resolve_train_pod_ssh(args, proxy_user=PROXY_USER)

    for local, remote in [
        (repo / "psm-model" / "src", "/workspace/PSM/psm-model/src"),
        (scripts, "/workspace/PSM/psm-model/scripts"),
    ]:
        rc_ = rc._ssh_push_dir(HOST_ALIAS, local, remote, host=ssh_host, port=ssh_port, user=ssh_user)
        print(f"push {local.name}: {rc_}")

    extra = {
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "DATASET_HF_TOKEN": os.environ.get("DATASET_HF_TOKEN", ""),
        "PSM_REPO_ROOT": "/workspace/PSM",
        "LEGACY_DONOR_STEP": "032000",
    }

    for name in ("runpod_donor_legacy_only.sh", "runpod_overfit_eval_only.sh"):
        script = scripts / name
        print(f"=== {name} ===")
        rc_ = rc._ssh_run_script(
            HOST_ALIAS,
            script,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            timeout_sec=1200,
            extra_env=extra,
            skip_ssh_wait=True,
        )
        print(f"exit {rc_}")
        if rc_ != 0:
            return rc_
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
