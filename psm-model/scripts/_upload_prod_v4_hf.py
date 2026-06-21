#!/usr/bin/env python3
"""Direct HF upload of prod-memory-v4 artifacts from pod."""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as rp  # noqa: E402

TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not TOKEN:
    print("HF_TOKEN required", file=sys.stderr)
    raise SystemExit(1)

REPO_ID = os.environ.get("PSM_HF_MODEL_REPO", rp.DEFAULT_HF_MODEL_REPO)
PROXY = "st0luf214e32c5-64411541"

cmd = f"""
set -euo pipefail
cd /workspace/PSM
pip install -q huggingface_hub hf_transfer 2>/dev/null || true
export HF_TOKEN='{TOKEN}'
python3 - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "{REPO_ID}"
root = Path("/workspace/PSM/psm-model/checkpoints")
uploaded = []
for path in sorted(root.glob("real-v3-50m-full-v2-prod-memory-v4*")):
    remote = f"psm-model/checkpoints/{{path.name}}"
    print(f"upload {{remote}} {{path.stat().st_size}}", flush=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo,
        repo_type="model",
        commit_message=f"recover {{path.name}}",
    )
    uploaded.append(remote)
results = Path("/workspace/PSM/psm-model/prod-memory/results")
for path in sorted(results.glob("prod-grounding-*.json")):
    remote = path.as_posix()
    print(f"upload {{remote}}", flush=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote,
        repo_id=repo,
        repo_type="model",
        commit_message=f"recover {{path.name}}",
    )
    uploaded.append(remote)
print("PSM_UPLOAD_DONE count=" + str(len(uploaded)))
PY
"""

import subprocess

proc = subprocess.run(
    [
        rp.SSH_BIN,
        "-tt",
        "-i",
        rp.SSH_KEY_PATH,
        "-o",
        "ConnectTimeout=20",
        *rp._ssh_endpoint(rp.SSH_CONFIG_HOST, host="ssh.runpod.io", port="22", user=PROXY),
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
if proc.stdout:
    sys.stdout.buffer.write(proc.stdout.encode("utf-8", errors="replace"))
if proc.stderr:
    print(proc.stderr[-3000:], file=sys.stderr)
if "PSM_UPLOAD_DONE" not in proc.stdout:
    print("upload may have failed", file=sys.stderr)
    raise SystemExit(proc.returncode or 1)
count = [l for l in proc.stdout.splitlines() if l.startswith("PSM_UPLOAD_DONE")]
print(count[-1] if count else "done")
raise SystemExit(0)
