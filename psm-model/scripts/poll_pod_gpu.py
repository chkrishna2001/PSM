#!/usr/bin/env python3
"""Poll RunPod pod GPU/eval/train activity until idle or done."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

PROXY_USER = "6c9efizq1aoocf-64411022"
SSH_HOST = "ssh.runpod.io"
SSH_KEY = r"C:\Users\chkri\.ssh\id_ed25519"


def pod_status() -> dict[str, str]:
    stdin = (
        "tmux ls 2>/dev/null || echo NO_TMUX\n"
        "pgrep -af 'psm_model.eval_checkpoint|psm_model.train' | grep -v tmux | head -2 || echo NO_JOB\n"
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo NO_GPU\n"
        "test -f /tmp/psm-gate4-eval.done && echo EVAL_DONE || echo EVAL_NOT_DONE\n"
        "test -f /tmp/psm-gate4.done && echo TRAIN_DONE || echo TRAIN_NOT_DONE\n"
        "ls -la /workspace/PSM/psm-model/checkpoints/gate-eval/gate4-full-expanded-step-043400.json 2>/dev/null | awk '{print $5}' || echo NO_EVAL_JSON\n"
        "exit\n"
    )
    proc = subprocess.run(
        [
            "ssh.exe",
            "-tt",
            "-i",
            SSH_KEY,
            "-o",
            "ConnectTimeout=20",
            f"{PROXY_USER}@{SSH_HOST}",
            "bash",
            "-s",
        ],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="replace",
    )
    raw = proc.stdout
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("\x1b")]
    out: dict[str, str] = {"ssh_rc": str(proc.returncode)}
    for line in lines:
        if "psm-gate4" in line and ":" in line and "windows" in line:
            out["tmux"] = line
        elif "psm_model." in line and "NO_JOB" not in line:
            out.setdefault("process", line[:120])
        elif "%" in line and "MiB" in line:
            out["gpu"] = line
        elif line in {
            "EVAL_DONE",
            "EVAL_NOT_DONE",
            "TRAIN_DONE",
            "TRAIN_NOT_DONE",
            "NO_TMUX",
            "NO_JOB",
            "NO_GPU",
            "NO_EVAL_JSON",
        }:
            out[line.lower().replace("_", "-")] = "yes"
        elif line.isdigit() and int(line) > 10000:
            out["eval_json_bytes"] = line
    util = 0
    if "gpu" in out:
        try:
            util = int(out["gpu"].split("%", 1)[0].strip())
        except ValueError:
            util = -1
    out["gpu_util_pct"] = str(util)
    out["state"] = "active" if util > 5 or "process" in out else "idle"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--max-polls", type=int, default=30)
    args = parser.parse_args()

    for i in range(1, args.max_polls + 1):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            st = pod_status()
        except subprocess.TimeoutExpired:
            st = {"state": "ssh_timeout", "error": "ssh timed out"}
        except Exception as exc:  # noqa: BLE001
            st = {"state": "error", "error": str(exc)}
        st["poll"] = str(i)
        st["ts"] = ts
        print(json.dumps(st, sort_keys=True), flush=True)

        if st.get("eval-done") == "yes":
            print("EVAL_FINISHED", flush=True)
            return 0
        if st.get("train-done") == "yes" and st.get("state") == "idle" and "process" not in st:
            print("TRAIN_FINISHED_IDLE", flush=True)
            return 0
        if st.get("state") == "idle" and st.get("eval-not-done") == "yes" and i > 1:
            print("GPU_IDLE_EVAL_INCOMPLETE", flush=True)
            return 2
        if i < args.max_polls:
            time.sleep(args.interval_sec)
    print("POLL_TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
