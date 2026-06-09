#!/usr/bin/env python3
"""Push eval script and start expanded Gate 4 eval on warm pod; verify GPU."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as ctl  # noqa: E402

PROXY_USER = "6c9efizq1aoocf-64411022"
SSH_HOST = "ssh.runpod.io"
SSH_PORT = "22"
CKPT = "psm-model/checkpoints/real-v3-50m-full-v2-step-043400.pt"


def _ssh_status() -> str:
    stdin = (
        "tmux ls 2>/dev/null || echo NO_TMUX\n"
        "pgrep -af psm_model.eval_checkpoint | head -1 || echo NO_EVAL\n"
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null\n"
        "test -f /tmp/psm-gate4-eval.done && echo EVAL_DONE || echo EVAL_RUNNING\n"
        "exit\n"
    )
    proc = subprocess.run(
        [
            ctl.SSH_BIN,
            "-tt",
            "-i",
            ctl.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            f"{PROXY_USER}@{SSH_HOST}",
            "bash",
            "-s",
        ],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=45,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout


def main() -> int:
    scripts = REPO / "psm-model" / "scripts"
    print("Pushing scripts...", flush=True)
    rc = ctl._ssh_push_dir(
        ctl.SSH_CONFIG_HOST,
        scripts,
        "/workspace/PSM/psm-model/scripts",
        host=SSH_HOST,
        port=SSH_PORT,
        user=PROXY_USER,
    )
    if rc != 0:
        return rc

    script = REPO / "psm-model" / "scripts" / "runpod_start_gate4_eval_only.sh"
    extra_env = {
        "PSM_EVAL_DEVICE": "cuda",
        "PSM_EVAL_FULL_CKPT": CKPT,
    }
    print("Starting eval tmux...", flush=True)
    rc = ctl._ssh_run_script(
        ctl.SSH_CONFIG_HOST,
        script,
        host=SSH_HOST,
        port=SSH_PORT,
        user=PROXY_USER,
        timeout_sec=120,
        extra_env=extra_env,
        skip_ssh_wait=True,
    )
    if rc != 0:
        print(f"start failed exit {rc}", file=sys.stderr)
        return rc if isinstance(rc, int) else rc[0]

    print("Verifying pod activity...", flush=True)
    out = _ssh_status()
    for line in out.splitlines():
        if any(k in line for k in ("psm-gate4-eval", "eval_checkpoint", "%", "MiB", "EVAL_", "NO_")):
            print(line)
    if "NO_EVAL" in out and "psm-gate4-eval" not in out:
        print("ERROR: eval not running on pod", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
