#!/usr/bin/env python3
"""Deploy pod and start HF LoRA train via git pull on pod (no tar-push)."""
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

DEFAULT_DATASET_REPO = "krishnach7262/psm-prod-memory-data"
DEFAULT_MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
DEFAULT_GIT_URL = "https://github.com/chkrishna2001/PSM.git"


def _load_hf_token() -> str:
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
        if proc.returncode == 0:
            return proc.stdout.strip()
    return ""


def _deploy() -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "deploy", "--auto-gpu", "--name", "psm-hf-lora", "--wait-ssh", "300"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    print(combined)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    pod_id = ""
    proxy_user = ""
    for line in combined.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        pod_id = payload.get("pod_id") or payload.get("id") or pod_id
        if payload.get("event") == "pod_created" and payload.get("id"):
            pod_id = payload.get("id") or pod_id
        proxy_user = payload.get("pod_host_id") or proxy_user
        for target in payload.get("targets") or []:
            if target.get("user"):
                proxy_user = target["user"]
    if not pod_id:
        return "", ""
    if not proxy_user:
        info = subprocess.run(
            [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "ssh-info", pod_id],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        for line in (info.stdout + info.stderr).splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            proxy_user = payload.get("pod_host_id") or proxy_user
            for target in payload.get("targets") or []:
                if target.get("user"):
                    proxy_user = target["user"]
    return pod_id, proxy_user


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--delete-pod-id", default="", help="Stop/delete idle pod before launch.")
    parser.add_argument("--model", default="qwen0.5b", choices=["qwen0.5b", "smol360m"])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--git-url", default=os.environ.get("PSM_GIT_URL", DEFAULT_GIT_URL))
    parser.add_argument("--sync-code", action="store_true", help="Tar-push train fixes before launch (ahead of git remote).")
    args = parser.parse_args()

    hf_token = _load_hf_token()
    if not hf_token:
        print("HF_TOKEN required — run: o krishnachhftoken", file=sys.stderr)
        return 1
    os.environ["HF_TOKEN"] = hf_token
    os.environ.setdefault("PSM_HF_DATASET_REPO", DEFAULT_DATASET_REPO)
    os.environ.setdefault("PSM_HF_MODEL_REPO", DEFAULT_MODEL_REPO)

    if args.upload_only:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "_upload_hf_prod_assets.py")],
            cwd=REPO,
            env=os.environ.copy(),
        ).returncode

    if args.delete_pod_id.strip():
        subprocess.run(
            [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "delete-pod", args.delete_pod_id.strip(), "--force-delete-pod"],
            cwd=REPO,
            check=False,
        )

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if args.deploy and not pod_id:
        pod_id, proxy_user = _deploy()
    if not pod_id or not proxy_user:
        print("pod_id and proxy_user required (or use --deploy)", file=sys.stderr)
        return 1

    subprocess.run(
        [sys.executable, str(SCRIPTS / "_upload_hf_prod_assets.py")],
        cwd=REPO,
        env=os.environ.copy(),
        check=False,
    )

    ns = argparse.Namespace(
        pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, ssh_host, ssh_port, ssh_user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)

    extra = {
        "HF_TOKEN": hf_token,
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_GIT_URL": args.git_url,
        "PSM_HF_DATASET_REPO": os.environ["PSM_HF_DATASET_REPO"],
        "PSM_HF_MODEL_REPO": os.environ["PSM_HF_MODEL_REPO"],
        "HF_MODEL_KEY": args.model,
        "HF_TRAIN_STEPS": str(args.steps),
    }

    if args.sync_code:
        rc._push_repo_files_via_tar(
            "runpod-psm-proxy",
            REPO,
            [
                "psm-model/scripts/runpod_hf_lora_train.sh",
                "psm-model/scripts/runpod_hf_lora_bootstrap.sh",
                "psm-model/src/psm_model/hf_lora_train.py",
                "psm-model/src/psm_model/lean_format.py",
                "psm-model/src/psm_model/prompts.py",
                "psm-model/prod-memory/prod_memory/build_hf_curriculum.py",
                "psm-model/prod-memory/prod_memory/hf_prompts.py",
                "psm-model/prod-memory/prod_memory/row_validation.py",
            ],
            "/workspace/PSM",
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
        rc._ssh_run_script(
            "runpod-psm-proxy",
            SCRIPTS / "runpod_hf_lora_train.sh",
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            timeout_sec=120,
            extra_env=extra,
            skip_ssh_wait=True,
        )
    else:
        rc._ssh_run_script(
            "runpod-psm-proxy",
            SCRIPTS / "runpod_hf_lora_bootstrap.sh",
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            timeout_sec=120,
            extra_env=extra,
            skip_ssh_wait=True,
        )

    verify = subprocess.run(
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
    )
    return verify.returncode


if __name__ == "__main__":
    raise SystemExit(main())
