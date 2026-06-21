#!/usr/bin/env python3
"""Upload prod-memory-v5 checkpoints + eval results from pod to HF."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as rp  # noqa: E402

PROXY = os.environ.get("RUNPOD_PROXY_USER", "m79372pnu8ci7z-644112f2@ssh.runpod.io")
TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not TOKEN:
    print("HF_TOKEN required", file=sys.stderr)
    raise SystemExit(1)

REPO_ID = os.environ.get("PSM_HF_MODEL_REPO", rp.DEFAULT_HF_MODEL_REPO)
STEM = "real-v3-50m-full-v2-prod-memory-v5"

cmd = f"""
set -euo pipefail
cd /workspace/PSM
pip install -q huggingface_hub hf_transfer 2>/dev/null || true
export HF_TOKEN='{TOKEN}'
python3 - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "{REPO_ID}"
root = Path("/workspace/PSM/psm-model/checkpoints")
uploaded = []
for path in sorted(root.glob("{STEM}*")):
    remote = f"psm-model/checkpoints/{{path.name}}"
    print(f"upload {{remote}} {{path.stat().st_size}}", flush=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo,
        repo_type="model",
        commit_message=f"prod-memory v5 {{path.name}}",
    )
    uploaded.append(remote)
print("PSM_UPLOAD_DONE count=" + str(len(uploaded)))
PY
"""

user = PROXY.split("@")[0] if "@" in PROXY else PROXY

proc = subprocess.run(
    [
        rp.SSH_BIN,
        "-tt",
        "-i",
        rp.SSH_KEY_PATH,
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        "22",
        f"{user}@ssh.runpod.io",
        "bash",
        "-s",
    ],
    input=f"{cmd}\nexit\n",
    capture_output=True,
    text=True,
    timeout=7200,
    encoding="utf-8",
    errors="replace",
)
sys.stdout.buffer.write((proc.stdout[-6000:] if len(proc.stdout) > 6000 else proc.stdout).encode("utf-8", errors="replace"))
if proc.stderr:
    sys.stderr.buffer.write(proc.stderr[-2000:].encode("utf-8", errors="replace"))
if "PSM_UPLOAD_DONE" not in proc.stdout:
    raise SystemExit(proc.returncode or 1)
raise SystemExit(0)
