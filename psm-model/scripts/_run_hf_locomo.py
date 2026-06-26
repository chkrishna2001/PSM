#!/usr/bin/env python3
"""Run HF LoRA LoCoMo ingest smoke on pod; pull results locally + HF."""
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

PROFILES: dict[str, dict[str, str]] = {
    "v5c": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5c-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5c-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5c-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5c-qwen0.5b-locomo-n25.json",
        "output_format": "minimal",
    },
    "v5e": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5e-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5e-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5e-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5e-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5e-qwen0.5b-locomo-n25.json",
        "output_format": "minimal",
    },
    "v5f": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5f-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5f-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5f-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5f-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5f-qwen0.5b-locomo-n25.json",
        "output_format": "minimal",
    },
    "v5f-b": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5f-b-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5f-b-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5f-b-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5f-b-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5f-b-qwen0.5b-locomo-n25.json",
        "output_format": "minimal",
    },
    "v5g": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5g-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5g-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5g-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5g-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5g-qwen0.5b-locomo-n25.json",
        "output_format": "minimal",
    },
    "v5h": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5h-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5h-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5h-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5h-qwen0.5b-locomo-n25.json",
        "output_format": "json",
    },
    "v5i": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5i-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5i-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5i-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5i-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5i-qwen0.5b-locomo-n25.json",
        "output_format": "minimal",
    },
    "v5j": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5j-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5j-qwen0.5b-locomo-n25.json",
        "label": "hf-prod-v5j-qwen0.5b",
        "hf_adapter_prefix": "hf-prod-v5j-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5j-qwen0.5b-locomo-n25.json",
        "output_format": "minimal",
    },
    "v5k-two-pass": {
        "adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter",
        "binary_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter",
        "locomo_out": "psm-model/prod-memory/results/hf-prod-v5k-two-pass-locomo-n25.json",
        "label": "hf-prod-v5k-two-pass",
        "hf_adapter_prefix": "hf-prod-v5k-extract-qwen0.5b",
        "hf_binary_prefix": "hf-prod-v5k-gate-distill-qwen0.5b",
        "hf_locomo": "eval/hf-prod-v5k-two-pass-locomo-n25.json",
        "output_format": "minimal_extract",
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
    return ""


def _upload_locomo_hf(profile: dict[str, str], token: str) -> int:
    local = REPO / profile["locomo_out"]
    if not local.is_file():
        print(f"locomo missing: {local}", file=sys.stderr)
        return 1
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; from huggingface_hub import HfApi; "
                f"api=HfApi(token=os.environ['HF_TOKEN']); "
                f"api.upload_file(path_in_repo={profile['hf_locomo']!r}, "
                f"path_or_fileobj={str(local)!r}, repo_id={MODEL_REPO!r}, repo_type='model')"
            ),
        ],
        cwd=REPO,
        env={**os.environ, "HF_TOKEN": token},
    )
    return proc.returncode


def _pull_locomo(pod_id: str, proxy_user: str, profile: dict[str, str]) -> int:
    alias, host, port, user = _resolve(pod_id, proxy_user)
    local = REPO / profile["locomo_out"]
    local.parent.mkdir(parents=True, exist_ok=True)
    return rc._scp_from_pod(
        alias,
        f"/workspace/PSM/{profile['locomo_out']}",
        local,
        host=host,
        port=port,
        user=user,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--proxy-user", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v5e")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--pull-only", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    token = _hf_token()
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1

    if args.pull_only:
        return _pull_locomo(args.pod_id, args.proxy_user, profile)

    alias, host, port, user = _resolve(args.pod_id, args.proxy_user)
    extra = {
        "HF_TOKEN": token,
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": MODEL_REPO,
        "HF_MODEL_KEY": "qwen0.5b",
        "HF_ADAPTER_DIR": profile["adapter"],
        "HF_ADAPTER_PREFIX": profile["hf_adapter_prefix"],
        "HF_CHECKPOINT_LABEL": profile["label"],
        "HF_LOCOMO_OUT": profile["locomo_out"],
        "HF_LOCOMO_LIMIT": str(args.limit),
        "HF_OUTPUT_FORMAT": profile.get("output_format", "minimal"),
    }
    if profile.get("binary_adapter"):
        extra["HF_BINARY_ADAPTER_DIR"] = profile["binary_adapter"]
        extra["HF_BINARY_ADAPTER_PREFIX"] = profile.get("hf_binary_prefix", "")
    rc._push_repo_files_via_tar(
        alias,
        REPO,
        [
            "psm-model/scripts/runpod_hf_locomo_smoke.sh",
            "psm-model/prod-memory/prod_memory/eval_hf_locomo.py",
            "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
            "psm-model/prod-memory/prod_memory/eval_classify.py",
            "psm-model/prod-memory/prod_memory/hf_prompts.py",
            "psm-model/prod-memory/prod_memory/grounding.py",
            "psm-model/src/psm_model/remember_cli.py",
            "psm-model/src/psm_model/lean_format.py",
        ],
        "/workspace/PSM",
        host=host,
        port=port,
        user=user,
    )
    code = int(
        rc._ssh_run_script(
            alias,
            SCRIPTS / "runpod_hf_locomo_smoke.sh",
            host=host,
            port=port,
            user=user,
            timeout_sec=900,
            extra_env=extra,
        )
    )
    if code != 0:
        return code
    if _pull_locomo(args.pod_id, args.proxy_user, profile) != 0:
        return 1
    summary = json.loads((REPO / profile["locomo_out"]).read_text(encoding="utf-8"))["summary"]
    print(json.dumps(summary, indent=2))
    if _upload_locomo_hf(profile, token) != 0:
        return 1
    return 0 if summary.get("passed") else 1


def _resolve(pod_id: str, proxy_user: str) -> tuple[str, str, int, str]:
    ns = argparse.Namespace(
        pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    return rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)


if __name__ == "__main__":
    raise SystemExit(main())
