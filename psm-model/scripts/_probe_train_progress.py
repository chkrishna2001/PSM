#!/usr/bin/env python3
"""Quick train progress probe."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

PROBE = r"""
stat /tmp/psm-hf-lora-train.log 2>/dev/null | grep Modify || true
tail -3 /tmp/psm-hf-lora-train.log 2>/dev/null | while IFS= read -r line; do echo "TAIL=$line"; done
grep -c "'epoch'" /tmp/psm-hf-lora-train.log 2>/dev/null | while read n; do echo "LOG_LINES=$n"; done
test -f /tmp/psm-hf-lora.done && echo DONE=1 || echo DONE=0
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | while read g; do echo "GPU=$g"; done
date -u | while read d; do echo "UTC=$d"; done
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="jf5j5htrfqkyc1")
    parser.add_argument("--proxy-user", default="jf5j5htrfqkyc1-6441145a")
    args = parser.parse_args()
    ns = argparse.Namespace(
        pod_id=args.pod_id,
        proxy_user=args.proxy_user,
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
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=args.proxy_user)
    proc = subprocess.run(
        [
            rc.SSH_BIN,
            "-tt",
            "-i",
            rc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            *rc._ssh_endpoint("runpod-psm-proxy", host=host, port=port, user=user),
            "bash",
            "-s",
        ],
        input=f"{PROBE}exit\n",
        capture_output=True,
        text=True,
        timeout=90,
        encoding="utf-8",
        errors="replace",
    )
    epoch = None
    for line in (proc.stdout or "").splitlines():
        print(line)
        if line.startswith("TAIL="):
            m = re.search(r"'epoch': '([0-9.]+)'", line)
            if m:
                epoch = float(m.group(1))
    if epoch is not None:
        steps_per_epoch = 2289 / 8  # batch=1, grad_accum=8
        total_epochs = 2400 / steps_per_epoch
        done_steps = epoch * steps_per_epoch
        remain_steps = max(0, 2400 - done_steps)
        pct = 100 * done_steps / 2400
        # ~2h from user; use observed rate if we know start was ~15:04
        print(f"PROGRESS epoch={epoch:.3f}/{total_epochs:.2f} steps~{done_steps:.0f}/2400 ({pct:.1f}%) remain~{remain_steps:.0f}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
