#!/usr/bin/env python3
"""Pull 43400 eval from pod (via SSH JSON extract) and analyze failures."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "src"))

PROXY = "6c9efizq1aoocf-64411022@ssh.runpod.io"
KEY = r"C:\Users\chkri\.ssh\id_ed25519"
REMOTE = "psm-model/checkpoints/gate-eval/gate4-full-expanded-step-043400.json"
LOCAL = REPO / REMOTE.replace("/", "\\") if "\\" in str(REPO) else REPO / REMOTE


def pull_report() -> None:
    stdin = f"""cd /workspace/PSM
python3 - <<'PY'
import base64, json
from pathlib import Path
raw = Path("{REMOTE}").read_text(encoding="utf-8", errors="replace")
for i, ch in enumerate(raw):
    if ch != "{{":
        continue
    try:
        d = json.loads(raw[i:])
        break
    except json.JSONDecodeError:
        continue
else:
    raise SystemExit("no json")
print(base64.b64encode(json.dumps(d).encode()).decode())
PY
exit
"""
    proc = subprocess.run(
        ["ssh.exe", "-tt", "-i", KEY, PROXY, "bash", "-s"],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    import base64

    for line in proc.stdout.splitlines():
        line = line.strip()
        if len(line) > 200 and line.replace("+", "").replace("/", "").replace("=", "").isalnum():
            data = json.loads(base64.b64decode(line))
            LOCAL.parent.mkdir(parents=True, exist_ok=True)
            LOCAL.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(f"pulled {LOCAL} ({LOCAL.stat().st_size} bytes)")
            return
    raise SystemExit("pull failed")


def main() -> int:
    pull_report()
    from psm_model.analyze_eval_report import main as analyze_main

    sys.argv = [
        "analyze",
        str(LOCAL),
        "--gate-mode",
        "expanded",
    ]
    return analyze_main()


if __name__ == "__main__":
    raise SystemExit(main())
