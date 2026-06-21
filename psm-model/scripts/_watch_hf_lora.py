#!/usr/bin/env python3
"""Poll HF LoRA train; run prod fixture eval + pull results when done."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"

PROFILES: dict[str, dict[str, str]] = {
    "v1": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v1-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v1-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v1-qwen0.5b",
        "hf_eval": "eval/hf-prod-v1-qwen0.5b-prod-grounding.json",
    },
    "v2": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v2-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v2-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v2-qwen0.5b",
        "hf_eval": "eval/hf-prod-v2-qwen0.5b-prod-grounding.json",
    },
}

DEFAULT_POD = "jf5j5htrfqkyc1"
DEFAULT_PROXY = "jf5j5htrfqkyc1-6441145a"


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
            "--timeout-sec",
            "90",
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


def _probe_train(pod_id: str, proxy_user: str, profile: dict[str, str]) -> dict[str, str]:
    ns = argparse.Namespace(
        pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)
    adapter = profile["adapter"]
    probe = f"""
test -f /tmp/psm-hf-lora.done && echo PSM_DONE=1 || echo PSM_DONE=0
test -f /workspace/PSM/{adapter}/adapter_model.safetensors && echo PSM_ADPT=1 || echo PSM_ADPT=0
if [[ -f /tmp/psm-hf-lora-train.log ]]; then
  tail -8 /tmp/psm-hf-lora-train.log | while IFS= read -r line; do echo "PSM_LOG=$line"; done
fi
"""
    proc = subprocess.run(
        [
            rc.SSH_BIN, "-tt", "-i", rc.SSH_KEY_PATH, "-o", "ConnectTimeout=20",
            *rc._ssh_endpoint("runpod-psm-proxy", host=host, port=port, user=user),
            "bash", "-s",
        ],
        input=f"{probe}exit\n",
        capture_output=True,
        text=True,
        timeout=90,
        encoding="utf-8",
        errors="replace",
    )
    out: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if line.startswith("PSM_"):
            key, _, value = line.partition("=")
            if key == "PSM_LOG":
                out.setdefault("log_tail", "")
                out["log_tail"] = (out.get("log_tail", "") + value + "\n").strip()
            else:
                out[key.replace("PSM_", "").lower()] = value.strip()
    return out


def _train_finished(status: dict[str, Any], probe: dict[str, str], verify_code: int) -> bool:
    if probe.get("done") == "1":
        return True
    log = probe.get("log_tail") or ""
    if probe.get("adpt") == "1" and ("train_runtime" in log or "uploaded adapter" in log):
        return True
    state = status.get("job_state", "")
    gpu = int(status.get("gpu_util_pct") or 0)
    proc = status.get("process", "MISSING")
    if state in {"stopped", "idle_billing"} and proc == "MISSING" and gpu < 5:
        return probe.get("adpt") == "1"
    if verify_code == 2 and probe.get("adpt") == "1":
        return True
    return False


def _finish(pod_id: str, proxy_user: str, profile_key: str) -> int:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "_run_hf_lora_eval.py"),
            "--pod-id",
            pod_id,
            "--proxy-user",
            proxy_user,
            "--profile",
            profile_key,
        ],
        cwd=REPO,
    )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default=DEFAULT_POD)
    parser.add_argument("--proxy-user", default=DEFAULT_PROXY)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v2")
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    if args.eval_only:
        return _finish(args.pod_id, args.proxy_user, args.profile)

    print(
        f"watching pod {args.pod_id} profile={args.profile} every {args.interval_sec}s",
        flush=True,
    )
    while True:
        code, status = _verify(args.pod_id, args.proxy_user)
        probe = _probe_train(args.pod_id, args.proxy_user, profile)
        state = status.get("job_state", "?")
        gpu = status.get("gpu_util_pct", "?")
        print(
            json.dumps(
                {
                    "code": code,
                    "job_state": state,
                    "gpu_util_pct": gpu,
                    "done": probe.get("done"),
                    "adapter": probe.get("adpt"),
                    "log_tail": (probe.get("log_tail") or "").splitlines()[-3:],
                },
                indent=2,
            ),
            flush=True,
        )

        if _train_finished(status, probe, code):
            print("train finished — eval + pull + HF upload", flush=True)
            finish_rc = _finish(args.pod_id, args.proxy_user, args.profile)
            print(f"finish exit {finish_rc}", flush=True)
            return 0 if finish_rc == 0 else 1

        if state == "training" or status.get("passed"):
            time.sleep(args.interval_sec)
            continue

        print("unexpected state; retrying", flush=True)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
