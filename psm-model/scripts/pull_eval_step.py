#!/usr/bin/env python3
"""Pull expanded eval report for a step and print metrics."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as ctl  # noqa: E402

PROXY = "6c9efizq1aoocf-64411022"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=int, help="Checkpoint step, e.g. 42400")
    args = parser.parse_args()
    step = f"{args.step:06d}"
    remote = f"/workspace/PSM/psm-model/checkpoints/gate-eval/gate4-full-expanded-step-{step}.json"
    local = REPO / "psm-model" / "checkpoints" / "gate-eval" / f"gate4-full-expanded-step-{step}.json"

    proc = subprocess.run(
        [
            ctl.SSH_BIN,
            "-tt",
            "-i",
            ctl.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            f"{PROXY}@ssh.runpod.io",
            "bash",
            "-s",
        ],
        input=f"cat '{remote}'\nexit\n",
        capture_output=True,
        text=True,
        timeout=180,
        encoding="utf-8",
        errors="replace",
    )
    raw = proc.stdout
    idx = raw.find("{")
    if idx < 0:
        print("no report", file=sys.stderr)
        return 1
    payload = raw[idx:]
    data, end = json.JSONDecoder().raw_decode(payload)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(payload[:end], encoding="utf-8")

    reports = data.get("reports", [])
    parse_fails = sum(1 for r in reports if not r.get("parse_valid"))
    g = data.get("gate", {})
    out = {
        "step": args.step,
        "parse_valid_rate": data.get("parse_valid_rate"),
        "schema_valid_rate": data.get("schema_valid_rate"),
        "action_accuracy": data.get("action_accuracy"),
        "gate_passed": g.get("passed"),
        "parse_fails": parse_fails,
        "total": len(reports),
        "local_path": str(local),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
