import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as r  # noqa: E402

USER = "1244fekd6g914j-64410fb2"
env = {
    "LOCOMO_CHECKPOINT": "psm-model/checkpoints/real-v3-50m-full-v2-step-058000.pt",
    "LOCOMO_LIMIT": "25",
    "LOCOMO_WAIT_FOR_EVAL": "0",
    "LOCOMO_DEVICE": "cuda",
    "PSM_HF_MODEL_REPO": os.environ.get("PSM_HF_MODEL_REPO", "subbu83/psm-50m-mixed-v1-run"),
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "DATASET_HF_TOKEN": os.environ.get("DATASET_HF_TOKEN", ""),
}
# Pod already has repo + checkpoint + synced src from eval — skip tar-push.
rc = r._ssh_run_script(
    "runpod-psm",
    Path(__file__).resolve().parent / "runpod_locomo.sh",
    host="ssh.runpod.io",
    port="22",
    user=USER,
    timeout_sec=7200,
    extra_env=env,
    skip_ssh_wait=True,
)
raise SystemExit(rc if isinstance(rc, int) else rc[0])
