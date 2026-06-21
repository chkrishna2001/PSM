#!/usr/bin/env python3
"""Poll HF LoRA train; run prod fixture eval when done."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"

DEFAULT_POD = "ymb1lfyvf5kgoz"
DEFAULT_PROXY = "ymb1lfyvf5kgoz-64410f25"
ADAPTER = "psm-model/prod-memory/checkpoints/hf-prod-v1-qwen0.5b/adapter"
EVAL_OUT = "psm-model/prod-memory/results/hf-prod-v1-qwen0.5b-prod-grounding.json"


def _verify(pod_id: str, proxy_user: str) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "runpod_ctl.py"),
            "verify-pod",
            "--pod-id",
            pod_id,
            "--proxy-user",
            proxy_user,
            "--train-log",
            "/tmp/psm-hf-lora-train.log",
            "--tmux-session",
            "psm-hf-lora",
            "--process-pattern",
            "hf_lora_train",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] = {}
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if line.startswith("{") and "pod_id" in line:
            try:
                block = json.loads(line)
                if "job_state" in block or "passed" in block:
                    payload = block
            except json.JSONDecodeError:
                continue
    return proc.returncode, payload


def _push_eval_files(pod_id: str, proxy_user: str) -> None:
    ns = argparse.Namespace(
        pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, ssh_host, ssh_port, ssh_user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)
    rc._push_repo_files_via_tar(
        "runpod-psm-proxy",
        REPO,
        [
            "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
            "psm-model/prod-memory/prod_memory/hf_prompts.py",
        ],
        "/workspace/PSM",
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
    )


def _run_eval(pod_id: str, proxy_user: str) -> int:
    _push_eval_files(pod_id, proxy_user)
    ns = argparse.Namespace(
        pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, ssh_host, ssh_port, ssh_user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)
    remote = f"""set -euo pipefail
cd /workspace/PSM
export PYTHONPATH=psm-model/src:psm-model/prod-memory
export PSM_RUNPOD=1
python -m prod_memory.eval_hf_grounding \\
  --adapter-dir {ADAPTER} \\
  --model qwen0.5b \\
  --device cuda \\
  --output-format tagged \\
  --checkpoint-label hf-prod-v1-qwen0.5b \\
  --out {EVAL_OUT}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
        tmp.write(remote)
        script_path = Path(tmp.name)
    try:
        return int(
            rc._ssh_run_script(
                "runpod-psm-proxy",
                script_path,
                host=ssh_host,
                port=ssh_port,
                user=ssh_user,
                timeout_sec=900,
            )
        )
    finally:
        script_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default=DEFAULT_POD)
    parser.add_argument("--proxy-user", default=DEFAULT_PROXY)
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        code = _run_eval(args.pod_id, args.proxy_user)
        print(f"eval exit {code}")
        return 0 if code == 0 else 1

    print(f"watching pod {args.pod_id} every {args.interval_sec}s", flush=True)
    while True:
        code, status = _verify(args.pod_id, args.proxy_user)
        state = status.get("job_state", "?")
        gpu = status.get("gpu_util_pct", "?")
        tail = status.get("train_log_tail") or []
        print(json.dumps({"code": code, "job_state": state, "gpu_util_pct": gpu, "tail": tail[-3:]}, indent=2), flush=True)

        if status.get("train_done") or state in {"train_finished", "idle_billing", "stopped"} or code == 2:
            print("train finished — running prod fixture eval", flush=True)
            eval_code = _run_eval(args.pod_id, args.proxy_user)
            print(f"eval exit {eval_code}", flush=True)
            return 0 if eval_code == 0 else 1

        if state == "training" or status.get("passed"):
            time.sleep(args.interval_sec)
            continue

        print("unexpected state; retrying", flush=True)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
