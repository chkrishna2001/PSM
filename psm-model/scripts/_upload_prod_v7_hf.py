"""Upload prod-memory-v7 checkpoints from RunPod pod to HF."""
import os
import re
import subprocess
import sys

PROXY = "u5qdh40qhkh45a-64411544@ssh.runpod.io"
KEY = os.path.expanduser("~/.ssh/id_ed25519")
HF = os.environ.get("HF_TOKEN", "")

probe = f"""
cd /workspace/PSM
export HF_TOKEN='{HF}'
export RUN_STEM=real-v3-50m-full-v2-prod-memory-v7
export UPLOAD_ALL=1
export KEEP_LOCAL=2
bash psm-model/scripts/runpod_upload_gate4.sh 2>&1 | tail -20
"""

proc = subprocess.run(
    ["ssh.exe", "-tt", "-i", KEY, "-o", "ConnectTimeout=20", "-p", "22", PROXY, "bash", "-s"],
    input=probe + "exit\n",
    capture_output=True,
    text=True,
    timeout=900,
    encoding="utf-8",
    errors="replace",
)
out = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", proc.stdout or "")
print(out[-3000:])
sys.exit(proc.returncode)
