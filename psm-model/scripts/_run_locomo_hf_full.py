#!/usr/bin/env python3
"""Deploy/start pod and run full LoCoMo benchmark with HF v5k two-pass adapters."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"
MODEL_REPO = "krishnach7262/psm-prod-memory-hf"

BINARY_ADAPTER = "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter"
EXTRACT_ADAPTER = "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter"
BINARY_PREFIX = "hf-prod-v5k-gate-distill-qwen0.5b"
EXTRACT_PREFIX = "hf-prod-v5k-extract-qwen0.5b"

PUSH_FILES = [
    "psm-model/src/psm_model/hf_remember_server.py",
    "psm-model/src/psm_model/remember_cli.py",
    "psm-model/src/psm_model/lean_format.py",
    "psm-model/prod-memory/prod_memory/hf_prompts.py",
    "psm-model/prod-memory/prod_memory/eval_classify.py",
    "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
    "psm-model/scripts/runpod_locomo.sh",
    "src/psm-core/src/psm-model-runtime.ts",
    "src/psm-core/src/remember-server.ts",
    "benchmark/locomo/src/ingest-psm-model.ts",
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


def _ssh_info(pod_id: str) -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "ssh-info", pod_id],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    proxy_user = ""
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        proxy_user = str(payload.get("pod_host_id") or proxy_user)
        for target in payload.get("targets") or []:
            if isinstance(target, dict) and target.get("user"):
                proxy_user = str(target["user"])
    return pod_id, proxy_user


def _deploy_pod() -> tuple[str, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "runpod_ctl.py"),
            "deploy",
            "--auto-gpu",
            "--name",
            "psm-locomo-hf",
            "--wait-ssh",
            "300",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    pod_id = ""
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "pod_created" and payload.get("id"):
            pod_id = str(payload["id"])
        if payload.get("pod_id"):
            pod_id = str(payload["pod_id"])
        if payload.get("pod_host_id"):
            return pod_id or str(payload.get("pod_id") or ""), str(payload["pod_host_id"])
    if not pod_id:
        raise SystemExit("deploy succeeded but no pod_id in output")
    return _ssh_info(pod_id)


GIT_URL = "https://github.com/chkrishna2001/PSM.git"


def _ensure_pod_repo(
    alias: str,
    *,
    host: str | None,
    port: str | None,
    user: str,
) -> int:
    """Clone full monorepo before tar-push so runpod_locomo.sh does not rm -rf our files."""
    cmd = (
        "ROOT=/workspace/PSM; "
        f'GIT_URL="{GIT_URL}"; '
        'if [[ ! -f "$ROOT/package.json" ]]; then '
        'echo "Ensuring PSM repo on pod..."; '
        'mkdir -p "$(dirname "$ROOT")"; rm -rf "$ROOT"; '
        'git clone --depth 1 "$GIT_URL" "$ROOT"; '
        'else echo "PSM repo already present"; fi'
    )
    return rc._ssh_run_bash(alias, cmd, host=host, port=port, user=user, timeout_sec=600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--deploy", action="store_true", help="Deploy new GPU pod")
    parser.add_argument("--limit", type=int, default=0, help="0 = all LoCoMo turns")
    args = parser.parse_args()

    token = _hf_token()
    if not token:
        print("HF_TOKEN required — run: o krishnachhftoken", file=sys.stderr)
        return 1
    os.environ["HF_TOKEN"] = token

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if args.deploy or not pod_id:
        pod_id, proxy_user = _deploy_pod()
    elif not proxy_user and pod_id:
        _, proxy_user = _ssh_info(pod_id)

    if not pod_id or not proxy_user:
        print("--pod-id + --proxy-user required (or --deploy)", file=sys.stderr)
        return 1

    ns = argparse.Namespace(pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy")
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)
    alias = "runpod-psm-proxy"

    ensure_rc = _ensure_pod_repo(alias, host=host, port=port, user=user)
    if ensure_rc != 0:
        return ensure_rc

    rc._push_repo_files_via_tar(alias, REPO, PUSH_FILES, "/workspace/PSM", host=host, port=port, user=user)

    limit_tag = "full" if args.limit == 0 else str(args.limit)
    env = {
        "HF_TOKEN": token,
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": MODEL_REPO,
        "PSM_RUNPOD": "1",
        "LOCOMO_WAIT_FOR_EVAL": "0",
        "LOCOMO_DEVICE": "cuda",
        "LOCOMO_LIMIT": str(args.limit),
        "LOCOMO_HF_BINARY_ADAPTER": BINARY_ADAPTER,
        "LOCOMO_HF_EXTRACT_ADAPTER": EXTRACT_ADAPTER,
        "LOCOMO_HF_BINARY_PREFIX": BINARY_PREFIX,
        "LOCOMO_HF_EXTRACT_PREFIX": EXTRACT_PREFIX,
        "LOCOMO_HF_MODEL_KEY": "qwen0.5b",
        "LOCOMO_HF_LABEL": "hf-prod-v5k-two-pass",
    }
    print(json.dumps({"pod_id": pod_id, "proxy_user": proxy_user, "limit": limit_tag}), flush=True)
    code = int(
        rc._ssh_run_script(
            alias,
            SCRIPTS / "runpod_locomo.sh",
            host=host,
            port=port,
            user=user,
            timeout_sec=14400,
            extra_env=env,
        )
    )
    if code != 0:
        return code

    remote_results = "/workspace/PSM/benchmark/locomo/results"
    local_results = REPO / "benchmark/locomo/results"
    local_results.mkdir(parents=True, exist_ok=True)
    tmp = local_results.parent / ".locomo_hf_pull_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    pull_code = rc._ssh_pull_dir(
        alias, remote_results, tmp, host=host, port=port, user=user
    )
    if pull_code == 0:
        for name in [
            f"locomo-hf-prod-v5k-two-pass-n{limit_tag}.db",
            f"locomo-hf-prod-v5k-two-pass-n{limit_tag}-results.json",
            f"locomo-hf-prod-v5k-two-pass-n{limit_tag}.log",
            "ingest-psm-model-summary.json",
        ]:
            src = tmp / name
            if src.is_file():
                shutil.copy2(src, local_results / name)
                print(f"pulled {name} ({src.stat().st_size} bytes)")
    shutil.rmtree(tmp, ignore_errors=True)
    return pull_code if pull_code != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
