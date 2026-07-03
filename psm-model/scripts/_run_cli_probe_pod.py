#!/usr/bin/env python3
"""Run CLI-shape LoCoMo probe on pod GPU; pull results locally."""
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

PROBE_FILES = [
    "psm-model/scripts/runpod_hf_cli_probe.sh",
    "psm-model/scripts/probe_locomo_hf_storage_cli_shape.py",
    "psm-model/scripts/probe_locomo_hf_storage_cpu.py",
    "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
    "psm-model/prod-memory/prod_memory/hf_prompts.py",
    "psm-model/prod-memory/prod_memory/grounding.py",
    "psm-model/prod-memory/prod_memory/eval_grounding.py",
    "psm-model/src/psm_model/remember_cli.py",
    "psm-model/src/psm_model/lean_format.py",
    "psm-model/src/psm_model/hf_lora_train.py",
    "psm-model/src/psm_model/schema.py",
    "psm-model/src/psm_model/storage_decision_repair.py",
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


def _resolve(pod_id: str, proxy_user: str) -> tuple[str, str, str, str]:
    ns = argparse.Namespace(
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
    return rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)


def _pull_remote_file(alias: str, remote: str, local: Path, *, host: str, port: str, user: str) -> int:
    local.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            rc.SSH_BIN,
            "-tt",
            "-i",
            rc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            *rc._ssh_endpoint(alias, host=host, port=port, user=user),
            "bash",
            "-s",
        ],
        input=f"cat {remote}\nexit\n",
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        return proc.returncode
    local.write_bytes(proc.stdout)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--proxy-user", required=True)
    parser.add_argument("--adapter-dir", default="psm-model/prod-memory/checkpoints/hf-prod-v5n-dpo-qwen0.5b/adapter")
    parser.add_argument(
        "--out",
        default="benchmark/locomo/results/probe-locomo-hf-storage-v5n-dpo-cli-shape-json-cuda.jsonl",
    )
    parser.add_argument("--pull-only", action="store_true")
    parser.add_argument("--output-format", default="json")
    args = parser.parse_args()

    alias, host, port, user = _resolve(args.pod_id, args.proxy_user)
    local_out = REPO / args.out
    summary_remote = args.out.replace(".jsonl", ".summary.json")
    summary_path = local_out.with_suffix(".summary.json")
    if args.pull_only:
        if _pull_remote_file(alias, f"/workspace/PSM/{args.out}", local_out, host=host, port=port, user=user) != 0:
            return 1
        _pull_remote_file(alias, f"/workspace/PSM/{summary_remote}", summary_path, host=host, port=port, user=user)
        if summary_path.is_file():
            print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))
        print(f"pulled {local_out}")
        return 0

    rc._push_repo_files_via_tar(alias, REPO, PROBE_FILES, "/workspace/PSM", host=host, port=port, user=user)
    extra = {
        "PSM_REPO_ROOT": "/workspace/PSM",
        "HF_ADAPTER_DIR": args.adapter_dir,
        "HF_PROBE_OUT": args.out,
        "HF_OUTPUT_FORMAT": args.output_format,
    }
    code = int(
        rc._ssh_run_script(
            alias,
            SCRIPTS / "runpod_hf_cli_probe.sh",
            host=host,
            port=port,
            user=user,
            timeout_sec=900,
            extra_env=extra,
        )
    )
    if code != 0:
        return code

    if _pull_remote_file(alias, f"/workspace/PSM/{args.out}", local_out, host=host, port=port, user=user) != 0:
        return 1
    _pull_remote_file(alias, f"/workspace/PSM/{summary_remote}", summary_path, host=host, port=port, user=user)
    if summary_path.is_file():
        print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))
    print(f"pulled {local_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
