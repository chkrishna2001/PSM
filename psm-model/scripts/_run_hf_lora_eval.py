#!/usr/bin/env python3
"""Deploy pod, run HF LoRA prod eval from HF adapter, pull results locally."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"
MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
GIT_URL = "https://github.com/chkrishna2001/PSM.git"

PROFILES: dict[str, dict[str, str]] = {
    "v1": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v1-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v1-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v1-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v1-qwen0.5b",
        "hf_eval": "eval/hf-prod-v1-qwen0.5b-prod-grounding.json",
    },
    "v2": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v2-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v2-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v2-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v2-qwen0.5b",
        "hf_eval": "eval/hf-prod-v2-qwen0.5b-prod-grounding.json",
    },
    "v4": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v4-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v4-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v4-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v4-qwen0.5b",
        "hf_eval": "eval/hf-prod-v4-qwen0.5b-prod-grounding.json",
    },
    "v5b": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5b-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5b-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5b-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5b-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5b-qwen0.5b-prod-grounding.json",
        "output_format": "tagged",
    },
    "v5c": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5c-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5c-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5c-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5c-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5d": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5d-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5d-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5d-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5d-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5d-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5e": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5e-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5e-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5e-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5e-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5e-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5f": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5f-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5f-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5f-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5f-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5f-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5f-b": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5f-b-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5f-b-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5f-b-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5f-b-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5f-b-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5g": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5g-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5g-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5g-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5g-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5g-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5h": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5h-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5h-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5h-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5h-qwen0.5b-prod-grounding.json",
        "output_format": "json",
    },
    "v5i": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5i-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5i-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5i-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5i-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5i-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5j": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5j-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5j-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5j-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5j-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5j-qwen0.5b-prod-grounding.json",
        "output_format": "minimal",
    },
    "v5k-gate": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5k-gate-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-gate-fix": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-fix-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-fix-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-fix-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5k-gate-fix-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-fix-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-gate-distill": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-distill-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-distill-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5k-gate-distill-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-distill-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-gate-dpo": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-dpo-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-gate-dpo-qwen0.5b-classify.json",
        "label": "hf-prod-v5k-gate-dpo-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5k-gate-dpo-qwen0.5b",
        "hf_eval": "eval/hf-prod-v5k-gate-dpo-qwen0.5b-classify.json",
        "output_format": "binary",
    },
    "v5k-extract": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter",
        "eval_out": "psm-model/prod-memory/results/hf-prod-v5k-extract-qwen0.5b-prod-grounding.json",
        "label": "hf-prod-v5k-extract-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5k-extract-qwen0.5b",
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
    return os.environ.get("HF_TOKEN", "").strip()


def _deploy() -> tuple[str, str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "runpod_ctl.py"),
            "deploy",
            "--auto-gpu",
            "--name",
            "psm-hf-lora-eval",
            "--wait-ssh",
            "300",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    print(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    pod_id = ""
    proxy_user = ""
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        pod_id = payload.get("pod_id") or payload.get("id") or pod_id
        proxy_user = payload.get("pod_host_id") or proxy_user
        if payload.get("event") == "pod_created" and payload.get("id"):
            pod_id = payload.get("id") or pod_id
        for target in payload.get("targets") or []:
            if target.get("user"):
                proxy_user = target["user"]
    if pod_id and not proxy_user:
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
            pod_id = payload.get("pod_id") or payload.get("id") or pod_id
            proxy_user = payload.get("pod_host_id") or proxy_user
            for target in payload.get("targets") or []:
                if target.get("user"):
                    proxy_user = target["user"]
    return pod_id, proxy_user


def _ns(pod_id: str, proxy_user: str) -> argparse.Namespace:
    return argparse.Namespace(
        pod_id=pod_id,
        proxy_user=proxy_user,
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


def _ssh(pod_id: str, proxy_user: str) -> tuple[str, str, str, str]:
    _, host, port, user = rc._resolve_train_pod_ssh(_ns(pod_id, proxy_user), proxy_user=proxy_user)
    return "runpod-psm-proxy", host, port, user


def _push_eval_files(pod_id: str, proxy_user: str) -> None:
    alias, host, port, user = _ssh(pod_id, proxy_user)
    rc._push_repo_files_via_tar(
        alias,
        REPO,
        [
            "psm-model/scripts/runpod_hf_lora_eval_only.sh",
            "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
            "psm-model/prod-memory/prod_memory/eval_classify.py",
            "psm-model/prod-memory/prod_memory/hf_prompts.py",
            "psm-model/prod-memory/prod_memory/eval_grounding.py",
            "psm-model/prod-memory/prod_memory/grounding.py",
            "psm-model/prod-memory/fixtures/cases.json",
            "psm-model/src/psm_model/hf_lora_train.py",
            "psm-model/src/psm_model/lean_format.py",
            "psm-model/src/psm_model/prompts.py",
            "psm-model/src/psm_model/remember_cli.py",
            "psm-model/src/psm_model/schema.py",
            "psm-model/src/psm_model/storage_decision_repair.py",
        ],
        "/workspace/PSM",
        host=host,
        port=port,
        user=user,
    )


def cmd_pull_eval(profile: dict[str, str]) -> int:
    eval_out = profile["eval_out"]
    local = REPO / eval_out
    local.parent.mkdir(parents=True, exist_ok=True)
    token = _hf_token()
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=profile["hf_eval"],
        repo_type="model",
        token=token,
        local_dir=str(local.parent / "_hf_eval_dl"),
    )
    data = Path(path).read_text(encoding="utf-8")
    local.write_text(data if data.endswith("\n") else data + "\n", encoding="utf-8")
    report = json.loads(data)
    print(f"pulled {local} ({local.stat().st_size} bytes) from HF", flush=True)
    print(json.dumps({"checkpoint": report.get("checkpoint"), "aggregate": report.get("aggregate", {})}, indent=2))
    return 0


def cmd_upload_eval_hf(profile: dict[str, str]) -> int:
    token = _hf_token()
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1
    local = REPO / profile["eval_out"]
    if not local.is_file():
        print(f"missing local eval: {local}", file=sys.stderr)
        return 1
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=profile["hf_eval"],
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="upload prod grounding eval report",
    )
    print(f"uploaded eval to {MODEL_REPO}/{profile['hf_eval']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v5b")
    parser.add_argument("--deploy", action="store_true", help="Deploy new eval pod")
    parser.add_argument("--pull-only", action="store_true")
    parser.add_argument("--upload-eval-hf", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]

    if args.upload_eval_hf:
        return cmd_upload_eval_hf(profile)

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if args.deploy and not pod_id:
        pod_id, proxy_user = _deploy()
        if not pod_id or not proxy_user:
            print("deploy failed: no pod_id/proxy_user", file=sys.stderr)
            return 1
        print(json.dumps({"pod_id": pod_id, "proxy_user": proxy_user}), flush=True)

    if args.pull_only:
        return cmd_pull_eval(profile)

    if not pod_id or not proxy_user:
        print("--deploy or --pod-id + --proxy-user required", file=sys.stderr)
        return 1

    token = _hf_token()
    if not token:
        print("HF_TOKEN missing — run: o krishnachhftoken", file=sys.stderr)
        return 1

    alias, host, port, user = _ssh(pod_id, proxy_user)
    extra = {
        "HF_TOKEN": token,
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_GIT_URL": GIT_URL,
        "PSM_HF_MODEL_REPO": MODEL_REPO,
        "HF_MODEL_KEY": "qwen0.5b",
        "HF_EVAL_OUT": profile["eval_out"],
        "HF_EVAL_REPO_PATH": profile["hf_eval"],
        "HF_CHECKPOINT_LABEL": profile["label"],
    }
    if profile.get("two_pass"):
        extra["HF_BINARY_ADAPTER_DIR"] = profile["binary_adapter"]
        extra["HF_EXTRACT_ADAPTER_DIR"] = profile["extract_adapter"]
        extra["HF_BINARY_ADAPTER_PREFIX"] = "hf-prod-v5k-gate-distill-qwen0.5b"
        extra["HF_EXTRACT_ADAPTER_PREFIX"] = "hf-prod-v5k-extract-qwen0.5b"
        eval_script = SCRIPTS / "runpod_hf_lora_two_pass_eval.sh"
    else:
        extra["HF_ADAPTER_DIR"] = profile["adapter"]
        extra["HF_ADAPTER_PREFIX"] = profile["hf_adapter_prefix"]
        if profile.get("output_format"):
            extra["HF_OUTPUT_FORMAT"] = profile["output_format"]
        eval_script = SCRIPTS / "runpod_hf_lora_eval_only.sh"
    _push_eval_files(pod_id, proxy_user)
    if profile.get("two_pass"):
        alias, host, port, user = _ssh(pod_id, proxy_user)
        rc._push_repo_files_via_tar(
            alias,
            REPO,
            [
                "psm-model/scripts/runpod_hf_lora_two_pass_eval.sh",
                "psm-model/prod-memory/prod_memory/eval_hf_two_pass.py",
                "psm-model/prod-memory/prod_memory/eval_classify.py",
            ],
            "/workspace/PSM",
            host=host,
            port=port,
            user=user,
        )
    extra_run = {**extra, "HF_SKIP_CLONE": "1"}
    code = int(
        rc._ssh_run_script(
            alias,
            eval_script,
            host=host,
            port=port,
            user=user,
            timeout_sec=900,
            extra_env=extra_run,
        )
    )

    if code != 0:
        print(f"eval on pod failed exit={code}", file=sys.stderr)
        return code
    local_eval = REPO / profile["eval_out"]
    if local_eval.is_file():
        return cmd_upload_eval_hf(profile)
    if cmd_pull_eval(profile) != 0:
        return 1
    return cmd_upload_eval_hf(profile)


if __name__ == "__main__":
    raise SystemExit(main())
