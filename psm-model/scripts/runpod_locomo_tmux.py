#!/usr/bin/env python3
"""Fast LoCoMo launch: git clone on pod + small tar-push of local fixes + tmux."""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
TMUX = "psm-locomo"
LOG = "/tmp/psm-locomo.log"

# Only local fixes not guaranteed on GitHub + prebuilt dist
PUSH_FILES = [
    "psm-model/src/psm_model/hf_remember_server.py",
    "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
    "psm-model/scripts/runpod_locomo.sh",
    "psm-model/scripts/ingest-cli.mjs",
    "src/psm-core/src/remember-server.ts",
    "src/psm-core/dist/remember-server.js",
    "benchmark/locomo/src/ingest-psm-model.ts",
    "dist/benchmark/locomo/src/ingest-psm-model.js",
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


def _proxy_user(pod_id: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "ssh-info", pod_id],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        if not line.strip().startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        for target in payload.get("targets") or []:
            if isinstance(target, dict) and target.get("user"):
                return str(target["user"])
    raise SystemExit(f"no proxy user for pod {pod_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--offset", type=int, default=2963)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--checkpoint-db",
        default=str(REPO / "benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db"),
    )
    args = parser.parse_args()

    token = _hf_token()
    if not token:
        print("HF_TOKEN required", file=sys.stderr)
        return 1

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip() or _proxy_user(pod_id)
    host, port, user = "ssh.runpod.io", "22", proxy_user
    alias = "runpod-psm-proxy"

    print("step 1/4: git clone on pod...", flush=True)
    clone_script = SCRIPTS / "_runpod_locomo_clone.sh"
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
        return 1

    print("step 2/4: push local fixes...", flush=True)
    if rc._push_repo_files_via_tar(alias, REPO, PUSH_FILES, "/workspace/PSM", host=host, port=port, user=user) != 0:
        return 1

    limit_tag = "full" if args.limit == 0 else str(args.limit)
    remote_db = f"/workspace/PSM/benchmark/locomo/results/locomo-hf-prod-v5k-two-pass-n{limit_tag}.db"
    ckpt = Path(args.checkpoint_db)
    if ckpt.is_file():
        print("step 3/4: push checkpoint db...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            bundle_root = Path(tmp) / "bundle"
            dest = bundle_root / "benchmark/locomo/results" / ckpt.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt, dest)
            if rc._ssh_push_dir(alias, bundle_root, "/workspace/PSM", host=host, port=port, user=user) != 0:
                return 1

    print("step 4/4: start tmux ingest...", flush=True)
    env_lines = "\n".join(
        f"export {k}='{v}'"
        for k, v in {
            "HF_TOKEN": token,
            "PSM_REPO_ROOT": "/workspace/PSM",
            "PSM_HF_MODEL_REPO": MODEL_REPO,
            "PSM_RUNPOD": "1",
            "PSM_SKIP_GIT_PULL": "1",
            "LOCOMO_WAIT_FOR_EVAL": "0",
            "LOCOMO_DEVICE": "cuda",
            "LOCOMO_LIMIT": str(args.limit),
            "LOCOMO_OFFSET": str(args.offset),
            "LOCOMO_RESUME_DB": remote_db,
            "LOCOMO_HF_BINARY_ADAPTER": "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter",
            "LOCOMO_HF_EXTRACT_ADAPTER": "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter",
            "LOCOMO_HF_BINARY_PREFIX": "hf-prod-v5k-gate-distill-qwen0.5b",
            "LOCOMO_HF_EXTRACT_PREFIX": "hf-prod-v5k-extract-qwen0.5b",
            "LOCOMO_HF_MODEL_KEY": "qwen0.5b",
            "LOCOMO_HF_LABEL": "hf-prod-v5k-two-pass",
            "LOCOMO_SKIP_BUILD": "1",
        }.items()
    )
    start_body = f"""#!/usr/bin/env bash
set -euo pipefail
{env_lines}
pkill -f 'runpod_locomo.sh' 2>/dev/null || true
nohup bash -lc 'cd /workspace/PSM && bash psm-model/scripts/runpod_locomo.sh >>{LOG} 2>&1; echo $? > /tmp/psm-locomo.done' >/dev/null 2>&1 &
echo $! > /tmp/psm-locomo.pid
sleep 2
pgrep -af runpod_locomo && echo ingest_started
"""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8") as tmp:
        tmp.write(start_body)
        start_script = Path(tmp.name)
    try:
        if rc._ssh_run_script(alias, start_script, host=host, port=port, user=user, timeout_sec=120, skip_ssh_wait=True) != 0:
            return 1
    finally:
        start_script.unlink(missing_ok=True)

    print(json.dumps({"pod_id": pod_id, "proxy_user": proxy_user, "tmux": TMUX, "log": LOG}), flush=True)
    verify = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "runpod_ctl.py"),
            "verify-pod",
            "--pod-id",
            pod_id,
            "--proxy-user",
            proxy_user,
            "--tmux-session",
            TMUX,
            "--process-pattern",
            "runpod_locomo|ingest-cli|ingest-psm|npm",
            "--train-log",
            LOG,
            "--timeout-sec",
            "45",
            "--no-require-gpu",
        ],
        cwd=REPO,
    )
    return verify.returncode


if __name__ == "__main__":
    raise SystemExit(main())
