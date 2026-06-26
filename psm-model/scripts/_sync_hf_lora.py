#!/usr/bin/env python3
"""Upload HF LoRA artifacts pod→HF and HF→local (periodic sync during train)."""
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
MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
TRAIN_LOG = "/tmp/psm-hf-lora-train.log"

PROFILE_PREFIX: dict[str, str] = {
    "v1": "hf-prod-v1-qwen0.5b",
    "v2": "hf-prod-v2-qwen0.5b",
    "v4": "hf-prod-v4-qwen0.5b",
    "v5b": "hf-prod-v5b-qwen0.5b",
    "v5c": "hf-prod-v5c-qwen0.5b",
    "v5d": "hf-prod-v5d-qwen0.5b",
    "v5e": "hf-prod-v5e-qwen0.5b",
    "v5f": "hf-prod-v5f-qwen0.5b",
    "v5f-b": "hf-prod-v5f-b-qwen0.5b",
    "v5g": "hf-prod-v5g-qwen0.5b",
    "v5h": "hf-prod-v5h-qwen0.5b",
    "v5i": "hf-prod-v5i-qwen0.5b",
    "v5j": "hf-prod-v5j-qwen0.5b",
    "v5k-gate": "hf-prod-v5k-gate-qwen0.5b",
    "v5k-gate-fix": "hf-prod-v5k-gate-fix-qwen0.5b",
    "v5k-gate-distill": "hf-prod-v5k-gate-distill-qwen0.5b",
    "v5k-gate-dpo": "hf-prod-v5k-gate-dpo-qwen0.5b",
    "v5k-extract": "hf-prod-v5k-extract-qwen0.5b",
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


def _run_remote(pod_id: str, proxy_user: str, body: str, *, timeout_sec: int = 600) -> int:
    _, host, port, user = rc._resolve_train_pod_ssh(_ns(pod_id, proxy_user), proxy_user=proxy_user)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
        tmp.write(body)
        path = Path(tmp.name)
    try:
        token = _hf_token()
        if not token:
            print("HF_TOKEN missing", file=sys.stderr)
            return 1
        return int(
            rc._ssh_run_script(
                "runpod-psm-proxy",
                path,
                host=host,
                port=port,
                user=user,
                timeout_sec=timeout_sec,
                extra_env={"HF_TOKEN": token, "PSM_HF_MODEL_REPO": MODEL_REPO},
            )
        )
    finally:
        path.unlink(missing_ok=True)


def cmd_upload_from_pod(pod_id: str, proxy_user: str, prefix: str, out_dir: str) -> int:
    script = f"""set -euo pipefail
cd /workspace/PSM
export HF_TOKEN="${{HF_TOKEN:?HF_TOKEN missing}}"
export PSM_HF_MODEL_REPO="{MODEL_REPO}"
python3 - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

repo = os.environ["PSM_HF_MODEL_REPO"]
prefix = "{prefix}"
out = Path("{out_dir}")
log = Path("{TRAIN_LOG}")
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=True)
uploaded = []
for path in sorted(out.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(out).as_posix()
    dest = f"{{prefix}}/{{rel}}"
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=dest,
        repo_id=repo,
        repo_type="model",
        commit_message=f"sync {{dest}}",
    )
    uploaded.append(dest)
if log.is_file():
    dest = f"logs/{{prefix}}-train.log"
    api.upload_file(
        path_or_fileobj=str(log),
        path_in_repo=dest,
        repo_id=repo,
        repo_type="model",
        commit_message="sync train log",
    )
    uploaded.append(dest)
print(json.dumps({{"uploaded": len(uploaded), "prefix": prefix}}))
PY
"""
    return _run_remote(pod_id, proxy_user, script)


def cmd_pull_local(prefix: str, out_dir: str) -> int:
    token = _hf_token()
    if not token:
        return 1
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    remote_files = api.list_repo_files(MODEL_REPO, repo_type="model")
    prefix_slash = f"{prefix}/"
    pulled = 0
    local_root = REPO / out_dir
    local_root.mkdir(parents=True, exist_ok=True)
    for remote in remote_files:
        if not remote.startswith(prefix_slash):
            continue
        rel = remote[len(prefix_slash) :]
        if not rel or rel.endswith("/"):
            continue
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=remote,
            repo_type="model",
            token=token,
            local_dir=str(REPO / "psm-model/prod-memory/checkpoints/_hf_dl"),
        )
        dest = local_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
        pulled += 1
    log_remote = f"logs/{prefix}-train.log"
    if log_remote in remote_files:
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=log_remote,
            repo_type="model",
            token=token,
            local_dir=str(REPO / "psm-model/prod-memory/checkpoints/_hf_dl"),
        )
        dest = REPO / "psm-model/prod-memory/results" / f"{prefix}-train.log"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        pulled += 1
    print(json.dumps({"pulled": pulled, "prefix": prefix, "local": str(local_root)}, indent=2))
    return 0


def cmd_verify(prefix: str) -> int:
    token = _hf_token()
    if not token:
        return 1
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = set(api.list_repo_files(MODEL_REPO, repo_type="model"))
    want = [
        f"{prefix}/adapter/adapter_model.safetensors",
        f"{prefix}/adapter/adapter_config.json",
        f"logs/{prefix}-train.log",
    ]
    present = [p for p in want if p in files]
    missing = [p for p in want if p not in files]
    ckpts = sorted(p for p in files if p.startswith(f"{prefix}/checkpoint-"))
    # ponytail: adapter may not exist until train ends; checkpoints count as partial sync
    has_artifacts = bool(present) or bool(ckpts)
    print(json.dumps({"present": present, "missing": missing, "checkpoints_on_hf": ckpts}, indent=2))
    return 0 if has_artifacts else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync HF LoRA train artifacts pod↔HF↔local.")
    parser.add_argument("--profile", choices=sorted(PROFILE_PREFIX), default="v5d")
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--upload-from-pod", action="store_true")
    parser.add_argument("--pull-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    prefix = PROFILE_PREFIX[args.profile]
    out_dir = f"psm-model/prod-memory/checkpoints/{prefix}"

    if args.verify_only:
        return cmd_verify(prefix)

    if args.pull_only:
        return cmd_pull_local(prefix, out_dir)

    if args.upload_from_pod:
        if not args.pod_id or not args.proxy_user:
            print("--pod-id and --proxy-user required for upload", file=sys.stderr)
            return 1
        rc = cmd_upload_from_pod(args.pod_id, args.proxy_user, prefix, out_dir)
        if rc != 0:
            return rc
        return cmd_pull_local(prefix, out_dir)

    print("use --upload-from-pod (needs pod) or --pull-only", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
