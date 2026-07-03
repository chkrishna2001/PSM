#!/usr/bin/env python3
"""Single-pod holdout gate matrix: parallel first, sequential fallback."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"
MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
GIT_URL = "https://github.com/chkrishna2001/PSM.git"
DEFAULT_SAMPLES = "conv-30,conv-41"
MATRIX_OUT = REPO / "benchmark/locomo/results/holdout-gate-matrix.json"

PROFILES: dict[str, dict[str, str]] = {
    "v5n-dpo": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5n-dpo-qwen0.5b/adapter",
        "prefix": "hf-prod-v5n-dpo-qwen0.5b",
        "gate_out": "benchmark/locomo/results/holdout-gate-v5n-dpo-conv-30-conv-41.json",
    },
    "v5n": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5n-qwen0.5b/adapter",
        "prefix": "hf-prod-v5n-qwen0.5b",
        "gate_out": "benchmark/locomo/results/holdout-gate-v5n-conv-30-conv-41.json",
    },
    "v5h": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter",
        "prefix": "hf-prod-v5h-qwen0.5b",
        "gate_out": "benchmark/locomo/results/holdout-gate-v5h-conv-30-conv-41.json",
    },
}

PUSH_FILES = [
    "psm-model/scripts/runpod_holdout_gate.sh",
    "psm-model/scripts/runpod_holdout_gate_matrix.sh",
    "psm-model/src/psm_model/hf_single_remember_server.py",
    "psm-model/src/psm_model/hf_remember_server.py",
    "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
    "psm-model/prod-memory/prod_memory/hf_prompts.py",
    "psm-model/prod-memory/prod_memory/grounding.py",
    "psm-model/src/psm_model/remember_cli.py",
    "benchmark/locomo/src/ingest-psm-model.ts",
    "benchmark/locomo/src/evaluate.ts",
    "benchmark/locomo/src/answer-evaluate.ts",
    "benchmark/locomo/src/cloudflare-ai.ts",
    "benchmark/locomo/src/common.ts",
    "src/psm-core/src/remember-server.ts",
    "src/psm-core/src/psm-model-runtime.ts",
    "src/psm-core/src/deterministic-plan-runtime.ts",
]


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    subprocess.run(["o", "krishnachhftoken"], check=False, capture_output=True)
    if os.name == "nt":
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-Clipboard -Raw).Trim()"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return ""


def _cf_env() -> dict[str, str]:
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "") or os.environ.get("CF_API_TOKEN", "")
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "") or os.environ.get("CF_ACCOUNT_ID", "")
    if not token or not account:
        raise SystemExit("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID required")
    return {
        "CLOUDFLARE_API_TOKEN": token,
        "CLOUDFLARE_ACCOUNT_ID": account,
        "LOCOMO_LLM_PROVIDER": "cloudflare",
    }


def _deploy(name: str) -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "deploy", "--auto-gpu", "--name", name, "--wait-ssh", "300"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"deploy failed: {combined[-2000:]}")
    pod_id = ""
    proxy_user = ""
    for line in reversed(combined.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not pod_id:
            pod_id = str(payload.get("pod_id") or payload.get("id") or "")
        if not proxy_user:
            proxy_user = str(payload.get("pod_host_id") or "")
        for target in payload.get("targets") or []:
            if isinstance(target, dict) and target.get("user"):
                proxy_user = str(target["user"])
        if pod_id and proxy_user:
            break
    if not pod_id:
        for line in combined.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") == "pod_created" and payload.get("id"):
                pod_id = str(payload["id"])
                break
    if pod_id and not proxy_user:
        info = subprocess.run(
            [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "ssh-info", pod_id],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        for line in (info.stdout + info.stderr).splitlines():
            if not line.strip().startswith("{"):
                continue
            try:
                payload = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            for target in payload.get("targets") or []:
                if isinstance(target, dict) and target.get("user"):
                    proxy_user = str(target["user"])
    if not pod_id or not proxy_user:
        raise RuntimeError(f"deploy missing pod_id/proxy_user: {combined[-2000:]}")
    return pod_id, proxy_user


def _ssh(pod_id: str, proxy_user: str) -> tuple[str, str, str, str]:
    ns = argparse.Namespace(
        pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)
    return "runpod-psm-proxy", host, port, user


def _bootstrap(pod_id: str, proxy_user: str) -> None:
    alias, host, port, user = _ssh(pod_id, proxy_user)
    clone_script = SCRIPTS / "_runpod_holdout_clone.sh"
    clone_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/PSM
if [[ ! -f "$ROOT/package.json" ]]; then
  rm -rf "$ROOT"
  git clone --depth 1 "{GIT_URL}" "$ROOT"
fi
echo clone_ok
""",
        encoding="utf-8",
    )
    if rc._ssh_run_script(alias, clone_script, host=host, port=port, user=user, timeout_sec=300, skip_ssh_wait=True) != 0:
        raise RuntimeError("clone failed")
    rc._push_repo_files_via_tar(alias, REPO, PUSH_FILES, "/workspace/PSM", host=host, port=port, user=user)


def _pull_file(pod_id: str, proxy_user: str, remote: str, local: Path) -> bool:
    alias, host, port, user = _ssh(pod_id, proxy_user)
    local.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            rc.SSH_BIN, "-tt", "-i", rc.SSH_KEY_PATH, "-o", "ConnectTimeout=20",
            *rc._ssh_endpoint(alias, host=host, port=port, user=user),
            f"cat /workspace/PSM/{remote}",
        ],
        capture_output=True,
        timeout=180,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return False
    local.write_bytes(proc.stdout)
    return True


def _pull_results(pod_id: str, proxy_user: str, profiles: list[str]) -> list[dict]:
    rows: list[dict] = []
    _pull_file(pod_id, proxy_user, "benchmark/locomo/results/holdout-gate-matrix.json", MATRIX_OUT)
    for key in profiles:
        profile = PROFILES[key]
        local = REPO / profile["gate_out"]
        ok = _pull_file(pod_id, proxy_user, profile["gate_out"], local)
        row: dict = {"profile": key, "gate_out": str(local), "exit_code": 0 if ok else 1}
        if ok and local.is_file():
            data = json.loads(local.read_text(encoding="utf-8"))
            row["retrieval_hit_at_1"] = (data.get("retrieval") or {}).get("hit_at_1")
            row["answer_accuracy"] = (data.get("answer") or {}).get("answer_accuracy")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-pod holdout gate matrix on RunPod.")
    parser.add_argument("--profiles", default="v5n-dpo,v5n,v5h")
    parser.add_argument("--sample-ids", default=DEFAULT_SAMPLES)
    parser.add_argument("--answer-limit", type=int, default=30)
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    args = parser.parse_args()

    if not _hf_token():
        raise SystemExit("HF_TOKEN missing")

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for key in profiles:
        if key not in PROFILES:
            raise SystemExit(f"unknown profile: {key}")

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if not pod_id:
        pod_id, proxy_user = _deploy("psm-holdout-gate")
        print(json.dumps({"event": "pod_ready", "pod_id": pod_id, "proxy_user": proxy_user}), flush=True)

    _bootstrap(pod_id, proxy_user)
    alias, host, port, user = _ssh(pod_id, proxy_user)
    extra = {
        "HF_TOKEN": _hf_token(),
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": MODEL_REPO,
        "PSM_RUNPOD": "1",
        "GATE_PROFILES": args.profiles,
        "HOLDOUT_SAMPLE_IDS": args.sample_ids,
        "GATE_ANSWER_LIMIT": str(args.answer_limit),
        **_cf_env(),
    }
    code = int(
        rc._ssh_run_script(
            alias,
            SCRIPTS / "runpod_holdout_gate_matrix.sh",
            host=host,
            port=port,
            user=user,
            timeout_sec=10800,
            extra_env=extra,
        )
    )
    rows = _pull_results(pod_id, proxy_user, profiles)
    if MATRIX_OUT.is_file():
        matrix = json.loads(MATRIX_OUT.read_text(encoding="utf-8"))
    else:
        matrix = {"mode": "single_pod", "profiles": rows}
        MATRIX_OUT.parent.mkdir(parents=True, exist_ok=True)
        MATRIX_OUT.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    if not args.keep_pod:
        subprocess.run([sys.executable, str(SCRIPTS / "runpod_ctl.py"), "stop-pod", pod_id], cwd=REPO, check=False)

    print(json.dumps({"matrix_out": str(MATRIX_OUT), "remote_exit": code, "profiles": rows}, indent=2))
    ok = code == 0 and all(r.get("exit_code") == 0 for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
