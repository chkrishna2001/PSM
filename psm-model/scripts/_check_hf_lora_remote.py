#!/usr/bin/env python3
"""One-shot remote status check for HF LoRA train pod."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

SSH_BIN = rc.SSH_BIN
SSH_KEY_PATH = rc.SSH_KEY_PATH

PROBE = r"""
for ROOT in /workspace/PSM /root/PSM; do
  if [[ -d "$ROOT" ]]; then break; fi
done
CKPT="$ROOT/psm-model/prod-memory/checkpoints/hf-prod-v1-qwen0.5b"
echo PSM_ROOT="$ROOT"
test -f /tmp/psm-hf-lora.done && echo PSM_HF_DONE=1 || echo PSM_HF_DONE=0
test -f "$CKPT/train.metrics.json" && echo PSM_METRICS=1 || echo PSM_METRICS=0
test -d "$CKPT/adapter" && echo PSM_ADAPTER=1 || echo PSM_ADAPTER=0
ls "$CKPT/adapter/" 2>/dev/null | head -5 | while IFS= read -r f; do echo "PSM_ADPT=$f"; done
if [[ -f "$CKPT/train.metrics.json" ]]; then cat "$CKPT/train.metrics.json" | head -c 2000 | while IFS= read -r line; do echo "PSM_M=$line"; done; fi
EVAL="$ROOT/psm-model/prod-memory/results/hf-prod-v1-qwen0.5b-prod-grounding.json"
if [[ -f "$EVAL" ]]; then python3 -c "import json;d=json.load(open('$EVAL'));c=d['cases'][0];print('PSM_SAMPLE',c.get('case_id'),c.get('parse_valid'),repr(c.get('raw_output','')[:200]))"; fi
if [[ -f /tmp/psm-hf-lora-train.log ]]; then tail -30 /tmp/psm-hf-lora-train.log | while IFS= read -r line; do echo "PSM_LOG=$line"; done; fi
find /workspace/PSM /root/PSM -name "hf-prod-v1-qwen0.5b-prod-grounding.json" 2>/dev/null | while IFS= read -r f; do echo "PSM_EVAL=$f"; done
ls -la /workspace/PSM/psm-model/prod-memory/results/ 2>/dev/null | while IFS= read -r line; do echo "PSM_LS=$line"; done
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="ymb1lfyvf5kgoz")
    parser.add_argument("--proxy-user", default="ymb1lfyvf5kgoz-64410f25")
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
    endpoint = rc._ssh_endpoint("runpod-psm-proxy", host=host, port=port, user=user)
    proc = subprocess.run(
        [
            SSH_BIN,
            "-tt",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            *endpoint,
            "bash",
            "-s",
        ],
        input=f"{PROBE}exit\n",
        capture_output=True,
        text=True,
        timeout=75,
        encoding="utf-8",
        errors="replace",
    )
    for line in (proc.stdout or "").splitlines():
        if line.startswith("PSM_"):
            print(line)
    if proc.returncode != 0 and proc.stderr:
        print(proc.stderr[-800:], file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
