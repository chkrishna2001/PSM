#!/usr/bin/env python3
"""Quick timing check for warm remember server vs cold CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
CHECKPOINT = ROOT / "psm-model" / "checkpoints" / "real-v3-50m-full-v2-step-048000.pt"
PAYLOAD = {
    "operation": "remember_llm_response",
    "conversation": [
        {"role": "user", "content": "Caroline said: Hey Mel! Good to see you!"},
        {"role": "assistant", "content": "Melanie said: Hey Caroline! Great to see you too."},
    ],
}


def main() -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "psm-model" / "src")
    env["PSM_FORCE_CPU"] = "1"
    proc = subprocess.Popen(
        [str(PYTHON), "-m", "psm_model.remember_server", str(CHECKPOINT), "--device", "cpu"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None
    ready = json.loads(proc.stdout.readline())
    print("server ready:", ready, file=sys.stderr)
    times: list[float] = []
    for index in range(3):
        start = time.perf_counter()
        proc.stdin.write(json.dumps({"payload": PAYLOAD}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        report = json.loads(line)
        print(f"turn {index + 1}: {elapsed:.1f}s status={report.get('repair_status')} action={report.get('parsed', {}).get('action')}")
    proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=30)
    print(f"avg after warm load: {sum(times) / len(times):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
