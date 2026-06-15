#!/usr/bin/env python3
"""Restart one pod until container is healthy, then run Gate 5 dual eval @ 058000."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as rp  # noqa: E402

POD_ID = "v65au7oae5e1q7"
PROXY_USER = "v65au7oae5e1q7-64411549@ssh.runpod.io"
SSH_USER = "v65au7oae5e1q7-64411549"


def container_healthy() -> bool:
    probe = subprocess.run(
        [
            rp.SSH_BIN,
            "-tt",
            "-i",
            rp.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=25",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{SSH_USER}@ssh.runpod.io",
            "bash",
            "-s",
        ],
        input="echo PSM_CONTAINER_OK\nnvidia-smi -L 2>&1 | head -1\nexit\n",
        capture_output=True,
        text=True,
        timeout=90,
    )
    combined = f"{probe.stdout}\n{probe.stderr}".lower()
    return (
        probe.returncode == 0
        and "psm_container_ok" in combined
        and "container not found" not in combined
        and "gpu" in combined
    )


def main() -> int:
    print(f"Stopping pod {POD_ID}...", flush=True)
    rp._rest("POST", f"/pods/{POD_ID}/stop")
    time.sleep(12)
    print(f"Starting pod {POD_ID}...", flush=True)
    ok, body = rp._rest_try("POST", f"/pods/{POD_ID}/start")
    if not ok:
        print(f"start failed: {body}", file=sys.stderr)
        return 1

    print("Waiting for healthy container (up to 10 min)...", flush=True)
    for attempt in range(40):
        time.sleep(15)
        if container_healthy():
            print(f"Container healthy after {(attempt + 1) * 15}s", flush=True)
            break
        print(f"  not ready yet ({(attempt + 1) * 15}s)...", flush=True)
    else:
        print("FATAL: container never became healthy — check RunPod dashboard", file=sys.stderr)
        return 1

    ctl = REPO / "psm-model" / "scripts" / "runpod_ctl.py"
    env = os.environ.copy()
    env.setdefault("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run")
    cmd = [
        sys.executable,
        str(ctl),
        "eval-gate5-dual",
        "--pod-id",
        POD_ID,
        "--proxy-user",
        PROXY_USER,
        "--eval-step",
        "58000",
        "--keep-pod",
        "--pull-reports",
        "psm-model/checkpoints/gate-eval",
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
