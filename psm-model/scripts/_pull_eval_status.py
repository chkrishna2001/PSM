import subprocess
import sys

user = "1244fekd6g914j-64410fb2@ssh.runpod.io"
script = r"""
ls -la /workspace/PSM/psm-model/checkpoints/gate-eval/ 2>/dev/null || echo no_dir
echo '---DONE---'
test -f /tmp/psm-gate5-dual-eval.done && cat /tmp/psm-gate5-dual-eval.done || echo no_done
echo '---LOG---'
tail -30 /tmp/psm-gate5-dual-eval.log 2>/dev/null || echo no_log
echo '---REPORT---'
test -f /workspace/PSM/psm-model/checkpoints/gate-eval/gate5-dual-step-058000.json && head -c 2000 /workspace/PSM/psm-model/checkpoints/gate-eval/gate5-dual-step-058000.json || echo no_report
"""
r = subprocess.run(
    ["ssh.exe", "-tt", "-i", r"C:\Users\chkri\.ssh\id_ed25519", "-o", "ConnectTimeout=30", user, "bash", "-s"],
    input=script + "\nexit\n",
    capture_output=True,
    text=True,
    timeout=120,
)
sys.stdout.write(r.stdout[-4000:])
sys.stderr.write(r.stderr[-1000:])
sys.exit(r.returncode)
