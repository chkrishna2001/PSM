#!/usr/bin/env python3
import subprocess

PROXY = "6c9efizq1aoocf-64411022@ssh.runpod.io"
KEY = r"C:\Users\chkri\.ssh\id_ed25519"
stdin2 = """cd /workspace/PSM
python3 - <<'PY'
import json
from pathlib import Path

def load_report(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            return json.loads(raw[i:])
        except json.JSONDecodeError:
            continue
    return None

for path in [
    Path("psm-model/checkpoints/gate-eval/gate4-full-expanded-step-043400.json"),
    Path("/tmp/psm-gate4-eval.log"),
]:
    d = load_report(path)
    if d:
        m = d.get("aggregate_metrics", d)
        g = d.get("gate", {})
        print(json.dumps({
            "source": str(path),
            "parse": m.get("parse_valid_rate"),
            "schema": m.get("schema_valid_rate"),
            "action": m.get("action_accuracy"),
            "passed": g.get("passed"),
            "failures": g.get("failures"),
        }))
        break
else:
    print(json.dumps({"error": "no_valid_report"}))
PY
exit
"""
proc = subprocess.run(
    ["ssh.exe", "-tt", "-i", KEY, PROXY, "bash", "-s"],
    input=stdin2,
    capture_output=True,
    text=True,
    timeout=60,
)
for line in proc.stdout.splitlines():
    s = line.strip()
    if s.startswith("{"):
        print(s)
        break
else:
    print(proc.stdout[-1500:], file=__import__("sys").stderr)
    raise SystemExit(proc.returncode or 1)
