import subprocess
import json
from pathlib import Path

USER = "1244fekd6g914j-64410fb2@ssh.runpod.io"
script = """
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
echo PSM_LOCOMO_DONE=$(test -f /tmp/psm-locomo.done && cat /tmp/psm-locomo.done || echo running)
pgrep -af 'ingest-psm-model|remember_server|evaluate.js' 2>/dev/null | grep -v pgrep | head -3 || echo PSM_PROC=none
echo '---log---'
tail -12 benchmark/locomo/results/locomo-psm-model-step-058000-n25.log 2>/dev/null || tail -12 /tmp/psm-locomo.log 2>/dev/null || echo no_log
echo '---results---'
test -f benchmark/locomo/results/locomo-psm-model-step-058000-n25-results.json && tail -c 1500 benchmark/locomo/results/locomo-psm-model-step-058000-n25-results.json || echo no_results_yet
"""
r = subprocess.run(
    ["ssh.exe", "-tt", "-i", r"C:\Users\chkri\.ssh\id_ed25519", "-o", "ConnectTimeout=25", USER, "bash", "-s"],
    input=script + "\nexit\n",
    capture_output=True,
    text=True,
    timeout=50,
)
out = r.stdout
# strip ansi
import re
out = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", out)
lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "RUNPOD" not in ln and "root@" not in ln and ln.strip() != "exit"]
print("\n".join(lines[-25:]))
