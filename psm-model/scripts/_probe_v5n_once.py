#!/usr/bin/env python3
"""One-shot v5n pod probe."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

PROBE = r"""
for ROOT in /workspace/PSM /root/PSM; do [[ -d "$ROOT" ]] && break; done
CKPT="$ROOT/psm-model/prod-memory/checkpoints/hf-prod-v5n-qwen0.5b"
test -f /tmp/psm-hf-lora.done && echo DONE=1 || echo DONE=0
tmux ls 2>/dev/null || echo NO_TMUX
pgrep -af hf_lora 2>/dev/null || echo NO_PROC
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo CKPT_LIST
ls -la "$CKPT" 2>/dev/null | head -15
test -f "$CKPT/train.metrics.json" && echo METRICS_OK || echo METRICS_MISSING
test -f "$CKPT/adapter/adapter_model.safetensors" && echo ADAPTER_OK || echo ADAPTER_MISSING
echo LOG_LINES
grep -c "'epoch'" /tmp/psm-hf-lora-train.log 2>/dev/null || echo 0
echo LOG_TAIL
tail -40 /tmp/psm-hf-lora-train.log 2>/dev/null | sed 's/[^[:print:]\t]//g'
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="rsxzc6nnc4urlu")
    parser.add_argument("--proxy-user", default="rsxzc6nnc4urlu-6441127b")
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
    sys.stdout.buffer.write((proc.stdout or "").encode("utf-8", errors="replace"))
    if proc.stderr:
        sys.stderr.buffer.write(proc.stderr[-2000:].encode("utf-8", errors="replace"))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
