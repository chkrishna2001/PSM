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

HF_PROFILES: dict[str, dict[str, str | int]] = {
    "v2": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v2.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v2-qwen0.5b",
        "run_prefix": "hf-prod-v2-qwen0.5b",
        "curriculum_profile": "hf-prod-v2",
        "steps": 2400,
    },
    "v4": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v4.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v4-qwen0.5b",
        "run_prefix": "hf-prod-v4-qwen0.5b",
        "curriculum_profile": "hf-prod-v4",
        "steps": 2400,
        "recall_fraction": "0",
    },
    "v5b": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5b.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5b-qwen0.5b",
        "run_prefix": "hf-prod-v5b-qwen0.5b",
        "curriculum_profile": "hf-prod-v5b",
        "steps": 1000,
        "recall_fraction": "0",
        "output_format": "tagged",
    },
    "v5c": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5c.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b",
        "run_prefix": "hf-prod-v5c-qwen0.5b",
        "curriculum_profile": "hf-prod-v5c",
        "steps": 800,
        "recall_fraction": "0",
        "output_format": "minimal",
        "resume_prefix": "hf-prod-v5-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5-qwen0.5b/adapter",
    },
    "v5d": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5d.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5d-qwen0.5b",
        "run_prefix": "hf-prod-v5d-qwen0.5b",
        "curriculum_profile": "hf-prod-v5d",
        "steps": 1000,
        "recall_fraction": "0",
        "output_format": "minimal",
        "learning_rate": "1e-4",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5e": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5e.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5e-qwen0.5b",
        "run_prefix": "hf-prod-v5e-qwen0.5b",
        "curriculum_profile": "hf-prod-v5e",
        "steps": 500,
        "recall_fraction": "0",
        "output_format": "minimal",
        "learning_rate": "5e-5",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5f": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5f.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5f-qwen0.5b",
        "run_prefix": "hf-prod-v5f-qwen0.5b",
        "curriculum_profile": "hf-prod-v5f",
        "steps": 300,
        "recall_fraction": "0",
        "output_format": "minimal",
        "learning_rate": "2e-5",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5f-b": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5f.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5f-b-qwen0.5b",
        "run_prefix": "hf-prod-v5f-b-qwen0.5b",
        "curriculum_profile": "hf-prod-v5f",
        "steps": 200,
        "recall_fraction": "0",
        "output_format": "minimal",
        "learning_rate": "1e-5",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5g": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5g.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5g-qwen0.5b",
        "run_prefix": "hf-prod-v5g-qwen0.5b",
        "curriculum_profile": "hf-prod-v5g",
        "steps": 240,
        "recall_fraction": "0",
        "output_format": "minimal",
        "learning_rate": "1e-5",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5h": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5h.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b",
        "run_prefix": "hf-prod-v5h-qwen0.5b",
        "curriculum_profile": "hf-prod-v5h",
        "steps": 400,
        "recall_fraction": "0",
        "output_format": "json",
        "learning_rate": "1e-5",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5i": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5i.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5i-qwen0.5b",
        "run_prefix": "hf-prod-v5i-qwen0.5b",
        "curriculum_profile": "hf-prod-v5i",
        "steps": 280,
        "recall_fraction": "0",
        "output_format": "minimal",
        "learning_rate": "1e-5",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5j": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5j.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5j-qwen0.5b",
        "run_prefix": "hf-prod-v5j-qwen0.5b",
        "curriculum_profile": "hf-prod-v5j",
        "steps": 200,
        "recall_fraction": "0",
        "output_format": "minimal",
        "learning_rate": "5e-6",
        "resume_prefix": "hf-prod-v5c-qwen0.5b",
        "resume_adapter": "psm-model/prod-memory/checkpoints/hf-prod-v5c-qwen0.5b/adapter",
    },
    "v5k-gate": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5k-gate.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-qwen0.5b",
        "run_prefix": "hf-prod-v5k-gate-qwen0.5b",
        "curriculum_profile": "hf-prod-v5k-gate",
        "steps": 80,
        "save_steps": 40,
        "recall_fraction": "0",
        "output_format": "binary",
        "learning_rate": "3e-5",
        "max_length": "2048",
    },
    "v5k-gate-fix": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5k-gate-fix.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-fix-qwen0.5b",
        "run_prefix": "hf-prod-v5k-gate-fix-qwen0.5b",
        "curriculum_profile": "hf-prod-v5k-gate-fix",
        "steps": 150,
        "save_steps": 50,
        "recall_fraction": "0",
        "output_format": "binary",
        "learning_rate": "5e-5",
        "max_length": "2048",
    },
    "v5k-gate-distill": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5k-gate-distill.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b",
        "run_prefix": "hf-prod-v5k-gate-distill-qwen0.5b",
        "curriculum_profile": "hf-prod-v5k-gate-distill",
        "steps": 120,
        "save_steps": 40,
        "recall_fraction": "0",
        "output_format": "binary",
        "learning_rate": "5e-5",
        "max_length": "2048",
    },
    "v5k-gate-dpo": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5k-gate-dpo.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-dpo-qwen0.5b",
        "run_prefix": "hf-prod-v5k-gate-dpo-qwen0.5b",
        "curriculum_profile": "hf-prod-v5k-gate-dpo",
        "steps": 80,
        "save_steps": 40,
        "recall_fraction": "0",
        "output_format": "binary",
        "learning_rate": "5e-6",
        "max_length": "2048",
        "train_mode": "dpo",
        "dpo_beta": "0.2",
    },
    "v5k-extract": {
        "curriculum": "psm-model/prod-memory/data/hf-prod-v5k-extract.jsonl",
        "out_dir": "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b",
        "run_prefix": "hf-prod-v5k-extract-qwen0.5b",
        "curriculum_profile": "hf-prod-v5k-extract",
        "steps": 120,
        "save_steps": 40,
        "recall_fraction": "0",
        "output_format": "minimal_extract",
        "learning_rate": "5e-6",
        "max_length": "2048",
    },
}


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


def _parse_json_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            block = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(block, dict):
            events.append(block)
    if events:
        return events
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        block = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(block, dict):
                        events.append(block)
                    break
    return events


def _deploy(gpu_preferences: str = "") -> tuple[str, str]:
    deploy_cmd = [
        sys.executable,
        str(SCRIPTS / "runpod_ctl.py"),
        "deploy",
        "--auto-gpu",
        "--name",
        "psm-hf-lora",
        "--wait-ssh",
        "300",
    ]
    if gpu_preferences:
        deploy_cmd.extend(["--gpu-preferences", gpu_preferences])
    proc = subprocess.run(deploy_cmd, cwd=REPO, capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    print(combined)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    pod_id = ""
    proxy_user = ""
    for payload in _parse_json_events(combined):
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
        for payload in _parse_json_events(info.stdout + info.stderr):
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
    parser.add_argument("--profile", choices=sorted(HF_PROFILES), default="v5e")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--git-url", default=os.environ.get("PSM_GIT_URL", DEFAULT_GIT_URL))
    parser.add_argument("--sync-code", action="store_true", help="Tar-push train fixes before launch (ahead of git remote).")
    parser.add_argument(
        "--gpu-preferences",
        default="",
        help="Comma-separated GPU order for --deploy (default: L4 first for v5e, else runpod_ctl default).",
    )
    args = parser.parse_args()

    profile = HF_PROFILES[args.profile]
    steps = args.steps or int(profile["steps"])

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
        gpu_prefs = args.gpu_preferences
        if not gpu_prefs and args.profile in ("v5e", "v5f", "v5g", "v5h", "v5i", "v5j", "v5k-gate", "v5k-gate-fix", "v5k-gate-distill", "v5k-gate-dpo", "v5k-extract"):
            gpu_prefs = "NVIDIA L4,NVIDIA RTX A5000,NVIDIA GeForce RTX 3090"
        pod_id, proxy_user = _deploy(gpu_preferences=gpu_prefs)
    if not pod_id or not proxy_user:
        print("pod_id and proxy_user required (or use --deploy)", file=sys.stderr)
        return 1

    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "_upload_hf_prod_assets.py"),
            "--curriculum",
            str(REPO / str(profile["curriculum"])),
        ],
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
        "HF_TRAIN_STEPS": str(steps),
        "HF_OUTPUT_DIR": str(profile["out_dir"]),
        "HF_CURRICULUM": str(profile["curriculum"]),
        "HF_RUN_PREFIX": str(profile["run_prefix"]),
        "HF_CURRICULUM_PROFILE": str(profile["curriculum_profile"]),
        "HF_RECALL_FRACTION": str(profile.get("recall_fraction", "0.12")),
    }
    if profile.get("output_format"):
        extra["HF_OUTPUT_FORMAT"] = str(profile["output_format"])
    if profile.get("resume_adapter"):
        extra["HF_RESUME_ADAPTER"] = str(profile["resume_adapter"])
    if profile.get("resume_prefix"):
        extra["HF_RESUME_PREFIX"] = str(profile["resume_prefix"])
    if profile.get("learning_rate"):
        extra["HF_LEARNING_RATE"] = str(profile["learning_rate"])
    if profile.get("save_steps"):
        extra["HF_SAVE_STEPS"] = str(profile["save_steps"])
    if profile.get("max_length"):
        extra["HF_MAX_LENGTH"] = str(profile["max_length"])
    if profile.get("train_mode"):
        extra["HF_TRAIN_MODE"] = str(profile["train_mode"])
    if profile.get("dpo_beta"):
        extra["HF_DPO_BETA"] = str(profile["dpo_beta"])

    if args.sync_code:
        rc._push_repo_files_via_tar(
            "runpod-psm-proxy",
            REPO,
            [
                "psm-model/scripts/runpod_hf_lora_train.sh",
                "psm-model/scripts/_sync_hf_lora.py",
                "psm-model/scripts/runpod_hf_lora_bootstrap.sh",
                "psm-model/src/psm_model/hf_lora_train.py",
                "psm-model/src/psm_model/lean_format.py",
                "psm-model/src/psm_model/prompts.py",
                "psm-model/prod-memory/prod_memory/build_hf_curriculum.py",
                "psm-model/prod-memory/prod_memory/build_binary_fixture_rows.py",
                "psm-model/prod-memory/prod_memory/build_minimal_fixture_rows.py",
                "psm-model/prod-memory/prod_memory/grounding.py",
                "psm-model/prod-memory/prod_memory/hf_prompts.py",
                "psm-model/prod-memory/prod_memory/curriculum_sources.py",
                "psm-model/prod-memory/prod_memory/indexable_labels.py",
                "psm-model/prod-memory/prod_memory/row_validation.py",
                str(profile["curriculum"]),
                str(Path(str(profile["curriculum"])).with_suffix(".manifest.json")),
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
