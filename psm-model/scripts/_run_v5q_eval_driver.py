#!/usr/bin/env python3
"""One-off driver: push full psm_model + prod_memory packages to pod, run v5q fixture eval."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
POD_ID = "897tbvxqu23xig"
PROXY_USER = "897tbvxqu23xig-64410f3c"


def main() -> int:
    files = [
        str(p.relative_to(REPO)).replace("\\", "/")
        for pattern in ("psm-model/src/psm_model/**/*.py", "psm-model/prod-memory/prod_memory/**/*.py")
        for p in REPO.glob(pattern)
    ]
    files.append("psm-model/prod-memory/fixtures/cases.json")
    print(f"pushing {len(files)} files")

    ns = rc.argparse.Namespace(
        pod_id=POD_ID, proxy_user=PROXY_USER, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=PROXY_USER)
    rc._push_repo_files_via_tar(
        "runpod-psm-proxy", REPO, files, "/workspace/PSM",
        host=host, port=port, user=user,
    )
    return int(
        rc._ssh_run_script(
            "runpod-psm-proxy",
            REPO / "psm-model/scripts/_run_v5q_pod_eval.sh",
            host=host, port=port, user=user,
            timeout_sec=900,
            extra_env={"HF_TOKEN": os.environ["HF_TOKEN"]},
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
