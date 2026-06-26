#!/usr/bin/env python3
"""Poll HF LoRA train; run prod fixture eval + pull results when done."""
from __future__ import annotations

import argparse
import json
import os
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
    "v4": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v4-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v4-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v4-qwen0.5b",
        "hf_eval": "eval/hf-prod-v4-qwen0.5b-prod-grounding.json",
    },
    "v5b": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5b-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5b-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5b-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5b-qwen0.5b-prod-grounding.json",
        "output_format": "tagged",
    },
    "v5c": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5c-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5c-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5c-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5d": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5d-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5d-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5d-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5d-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5e": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5e-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5e-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5e-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5e-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5f": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5f-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5f-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5f-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5f-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5f-b": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5f-b-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5f-b-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5f-b-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5f-b-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5g": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5g-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5g-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5g-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5g-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5h": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5h-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5h-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5h-qwen0.5b-prod-grounding.json",
        "output_format": "json",
    },
    "v5i": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5i-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5i-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5i-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5i-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5j": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5j-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5j-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5j-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5j-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5k-gate": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-gate-fix": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-fix-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-fix-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-fix-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-fix-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-gate-distill": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-distill-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-distill-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-distill-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-gate-dpo": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-dpo-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-dpo-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-dpo-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-dpo-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-extract": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-extract-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5k-extract-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-extract-qwen0.5b-prod-grounding.json",
        "output_format": "minimal_extract",
    },
    "v5k-two-pass": {
        "binary_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter",
        "extract_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-two-pass-prod-grounding.json",
        "label": "hf-prod-v5k-two-pass",
        "hf_eval": "eval/hf-prod-v5k-two-pass-prod-grounding.json",
        "two_pass": "1",
    },
}

DEFAULT_POD = "bkbe17mgff9f0q"
DEFAULT_PROXY = "bkbe17mgff9f0q-64411536"


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token.startswith("hf_"):
        return token
    subprocess.run(["o", "krishnachhftoken"], check=False, capture_output=True)
    if os.name == "nt":
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-Clipboard -Raw).Trim()"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip().startswith("hf_"):
            return proc.stdout.strip()
    return token


def _parse_verify_json(text: str) -> dict[str, Any]:
    """Last verify-pod JSON block (single-line events or indent=2 report)."""
    payload: dict[str, Any] = {}
    combined = text
    for line in combined.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            block = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(block, dict) and ("job_state" in block or "passed" in block):
            payload = block
    if payload:
        return payload
    for start, ch in enumerate(combined):
        if ch != "{":
            continue
        depth = 0
        for end in range(start, len(combined)):
            if combined[end] == "{":
                depth += 1
            elif combined[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        block = json.loads(combined[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(block, dict) and ("job_state" in block or "passed" in block):
                        payload = block
                    break
    return payload


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
    payload = _parse_verify_json(proc.stdout + proc.stderr)
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


FAIL_LOG_MARKERS = (
    "RuntimeError:",
    "CUDA error",
    "Traceback (most recent call last)",
    "OutOfMemoryError",
    "CUDA out of memory",
)


def _log_text(probe: dict[str, str], status: dict[str, Any]) -> str:
    parts = [probe.get("log_tail") or ""]
    tail = status.get("train_log_tail")
    if isinstance(tail, list):
        parts.extend(str(line) for line in tail)
    return "\n".join(parts)


def _train_failed(status: dict[str, Any], probe: dict[str, str]) -> bool:
    """Train died without adapter — don't spin on 'unexpected state' forever."""
    if probe.get("adpt") == "1" or probe.get("done") == "1":
        return False
    log = _log_text(probe, status)
    if any(marker in log for marker in FAIL_LOG_MARKERS):
        proc = str(status.get("process") or "")
        if proc == "MISSING" or "hf_lora_train" not in proc:
            return True
    state = str(status.get("job_state") or "")
    gpu = int(status.get("gpu_util_pct") or 0)
    proc = str(status.get("process") or "MISSING")
    tmux = str(status.get("tmux") or "")
    if state in {"stopped", "idle_billing"} and proc == "MISSING" and gpu < 5 and tmux == "MISSING":
        return True
    return False


def _sync_pod(profile_key: str, pod_id: str, proxy_user: str) -> None:
    env = os.environ.copy()
    token = _hf_token()
    if token:
        env["HF_TOKEN"] = token
    sync_proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "_sync_hf_lora.py"),
            "--profile",
            profile_key,
            "--pod-id",
            pod_id,
            "--proxy-user",
            proxy_user,
            "--upload-from-pod",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    if sync_proc.stdout.strip():
        print(sync_proc.stdout.strip(), flush=True)
    if sync_proc.returncode != 0 and sync_proc.stderr.strip():
        print(f"sync warn: {sync_proc.stderr.strip()}", flush=True)


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


def _finish(pod_id: str, proxy_user: str, profile_key: str, *, locomo: bool = False) -> int:
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
    if proc.returncode != 0:
        pull = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "_run_hf_lora_eval.py"),
                "--profile",
                profile_key,
                "--pull-only",
            ],
            cwd=REPO,
            check=False,
        )
        if pull.returncode != 0:
            return proc.returncode
    if not locomo:
        return 0
    locomo_proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "_run_hf_locomo.py"),
            "--pod-id",
            pod_id,
            "--proxy-user",
            proxy_user,
            "--profile",
            profile_key,
        ],
        cwd=REPO,
    )
    return locomo_proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default=DEFAULT_POD)
    parser.add_argument("--proxy-user", default=DEFAULT_PROXY)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v5f")
    parser.add_argument("--interval-sec", type=int, default=600, help="Poll interval (default 10m; use 300 for 2h trains)")
    parser.add_argument("--stop-pod-on-done", action="store_true", help="Stop pod after eval (not delete)")
    parser.add_argument("--locomo-on-done", action="store_true", help="Run LoCoMo n=25 smoke after fixture eval")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    if args.eval_only:
        return _finish(args.pod_id, args.proxy_user, args.profile, locomo=args.locomo_on_done)

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

        if _train_failed(status, probe):
            print("train FAILED (crash or idle without adapter) — syncing + exiting", flush=True)
            _sync_pod(args.profile, args.pod_id, args.proxy_user)
            if args.stop_pod_on_done:
                subprocess.run(
                    [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "stop-pod", args.pod_id],
                    cwd=REPO,
                    check=False,
                )
                print(f"stopped pod {args.pod_id}", flush=True)
            return 2

        if _train_finished(status, probe, code):
            print("train finished — eval + pull + HF upload", flush=True)
            finish_rc = _finish(args.pod_id, args.proxy_user, args.profile, locomo=args.locomo_on_done)
            print(f"finish exit {finish_rc}", flush=True)
            if args.stop_pod_on_done:
                eval_local = REPO / profile["eval_out"]
                if finish_rc != 0 or not eval_local.is_file():
                    print(
                        f"skip stop-pod: eval missing or failed (rc={finish_rc}, local={eval_local})",
                        flush=True,
                    )
                else:
                    subprocess.run(
                        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "stop-pod", args.pod_id],
                        cwd=REPO,
                        check=False,
                    )
                    print(f"stopped pod {args.pod_id}", flush=True)
            return 0 if finish_rc == 0 else 1

        gpu_n = int(gpu) if str(gpu).isdigit() else 0
        _sync_pod(args.profile, args.pod_id, args.proxy_user)
        if state == "training" or status.get("passed") or gpu_n >= 50:
            time.sleep(args.interval_sec)
            continue

        print(f"unexpected state (job_state={state} gpu={gpu}); retrying", flush=True)
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    # ponytail: self-check indented verify-pod JSON
    _sample = '{\n  "pod_id": "abc",\n  "job_state": "training",\n  "passed": true\n}'
    assert _parse_verify_json(_sample).get("job_state") == "training"
    raise SystemExit(main())
