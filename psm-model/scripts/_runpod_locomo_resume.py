#!/usr/bin/env python3
"""Resume LoCoMo on an already-provisioned pod (clone done)."""
from __future__ import annotations

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
PUSH_FILES = [
    "psm-model/src/psm_model/hf_remember_server.py",
    "psm-model/prod-memory/prod_memory/eval_hf_grounding.py",
    "psm-model/scripts/runpod_locomo.sh",
    "psm-model/scripts/ingest-cli.mjs",
    "src/psm-core/dist/remember-server.js",
    "dist/benchmark/locomo/src/ingest-psm-model.js",
]
USER = "qa61vj3nkpriqx-64411108"
HOST, PORT, ALIAS = "ssh.runpod.io", "22", "runpod-psm-proxy"
TMUX, LOG = "psm-locomo", "/tmp/psm-locomo.log"


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN required", file=sys.stderr)
        return 1

    print("push fixes...", flush=True)
    if rc._push_repo_files_via_tar(ALIAS, REPO, PUSH_FILES, "/workspace/PSM", host=HOST, port=PORT, user=USER) != 0:
        return 1

    ckpt = REPO / "benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db"
    print("push checkpoint db...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        bundle_root = Path(tmp) / "bundle"
        dest = bundle_root / "benchmark/locomo/results" / ckpt.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, dest)
        if rc._ssh_push_dir(ALIAS, bundle_root, "/workspace/PSM", host=HOST, port=PORT, user=USER) != 0:
            return 1

    env_lines = "\n".join(
        f"export {k}='{v}'"
        for k, v in {
            "HF_TOKEN": token,
            "PSM_REPO_ROOT": "/workspace/PSM",
            "PSM_HF_MODEL_REPO": "krishnach7262/psm-prod-memory-hf",
            "PSM_RUNPOD": "1",
            "PSM_SKIP_GIT_PULL": "1",
            "LOCOMO_WAIT_FOR_EVAL": "0",
            "LOCOMO_DEVICE": "cuda",
            "LOCOMO_LIMIT": "0",
            "LOCOMO_OFFSET": "2963",
            "LOCOMO_RESUME_DB": "/workspace/PSM/benchmark/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.db",
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
        start = Path(tmp.name)
    try:
        print("start tmux...", flush=True)
        if rc._ssh_run_script(ALIAS, start, host=HOST, port=PORT, user=USER, timeout_sec=120, skip_ssh_wait=True) != 0:
            return 1
    finally:
        start.unlink(missing_ok=True)

    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "runpod_ctl.py"),
            "verify-pod",
            "--pod-id",
            "qa61vj3nkpriqx",
            "--proxy-user",
            USER,
            "--tmux-session",
            "psm-locomo-nohup",
            "--process-pattern",
            "runpod_locomo|ingest-cli|npm",
            "--train-log",
            LOG,
            "--timeout-sec",
            "60",
            "--no-require-gpu",
        ],
        cwd=REPO,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
