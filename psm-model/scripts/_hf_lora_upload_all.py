#!/usr/bin/env python3
"""Upload all HF LoRA v2 artifacts from pod to HF; verify; pull metrics locally."""
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
MODEL_KEY = "qwen0.5b"
PREFIX = f"hf-prod-v2-{MODEL_KEY}"
OUT_DIR = f"psm-model/prod-memory/checkpoints/{PREFIX}"
TRAIN_LOG = "/tmp/psm-hf-lora-train.log"

REQUIRED_HF = [
    f"{PREFIX}/adapter/adapter_config.json",
    f"{PREFIX}/adapter/adapter_model.safetensors",
    f"{PREFIX}/train.metrics.json",
    f"logs/{PREFIX}-train.log",
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


def _run_remote(pod_id: str, proxy_user: str, body: str, *, timeout_sec: int = 900) -> int:
    _, host, port, user = rc._resolve_train_pod_ssh(_ns(pod_id, proxy_user), proxy_user=proxy_user)
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
        tmp.write(body)
        path = Path(tmp.name)
    try:
        token = _hf_token()
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


def cmd_upload(pod_id: str, proxy_user: str) -> int:
    script = f"""set -euo pipefail
cd /workspace/PSM
export HF_TOKEN="${{HF_TOKEN:?HF_TOKEN missing}}"
export PSM_HF_MODEL_REPO="{MODEL_REPO}"
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo = os.environ["PSM_HF_MODEL_REPO"]
prefix = "{PREFIX}"
out = Path("{OUT_DIR}")
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
        commit_message=f"upload {{dest}}",
    )
    uploaded.append(dest)
    print("uploaded", dest, path.stat().st_size)

if log.is_file():
    dest = f"logs/{{prefix}}-train.log"
    api.upload_file(
        path_or_fileobj=str(log),
        path_in_repo=dest,
        repo_id=repo,
        repo_type="model",
        commit_message="upload train log",
    )
    uploaded.append(dest)
    print("uploaded", dest, log.stat().st_size)

print("UPLOAD_COUNT", len(uploaded))
PY
"""
    return _run_remote(pod_id, proxy_user, script)


def cmd_verify() -> tuple[int, list[str]]:
    token = _hf_token()
    if not token:
        print("HF_TOKEN missing", file=sys.stderr)
        return 1, REQUIRED_HF
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = set(api.list_repo_files(MODEL_REPO, repo_type="model"))
    missing = [p for p in REQUIRED_HF if p not in files]
    present = [p for p in REQUIRED_HF if p in files]
    print(json.dumps({"present": present, "missing": missing, "total_hf_files": len(files)}, indent=2))
    return (0 if not missing else 1), missing


def cmd_pull_metrics() -> int:
    token = _hf_token()
    if not token:
        return 1
    from huggingface_hub import hf_hub_download

    local_dir = REPO / OUT_DIR
    local_dir.mkdir(parents=True, exist_ok=True)
    for remote in (f"{PREFIX}/train.metrics.json", f"logs/{PREFIX}-train.log"):
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=remote,
            repo_type="model",
            token=token,
            local_dir=str(REPO / "psm-model/prod-memory/checkpoints/_hf_dl"),
        )
        dest = REPO / OUT_DIR / "train.metrics.json" if remote.endswith("metrics.json") else REPO / "psm-model/prod-memory/results" / f"{PREFIX}-train.log"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"pulled {dest} ({dest.stat().st_size} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="jf5j5htrfqkyc1")
    parser.add_argument("--proxy-user", default="jf5j5htrfqkyc1-6441145a")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--pull-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        code, _ = cmd_verify()
        return code
    if args.pull_only:
        return cmd_pull_metrics()

    print("uploading all v2 artifacts from pod...", flush=True)
    rc_upload = cmd_upload(args.pod_id, args.proxy_user)
    if rc_upload != 0:
        print(f"upload exit={rc_upload}", file=sys.stderr)
        return rc_upload

    code, missing = cmd_verify()
    if code != 0:
        print(f"verify failed missing={missing}", file=sys.stderr)
        return code
    return cmd_pull_metrics()


if __name__ == "__main__":
    raise SystemExit(main())
