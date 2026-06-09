#!/usr/bin/env python3
"""Pull step-042800 expanded eval and print metrics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as ctl  # noqa: E402

STEP = "042800"
REMOTE = f"/workspace/PSM/psm-model/checkpoints/gate-eval/gate4-full-expanded-step-{STEP}.json"
LOCAL = REPO / "psm-model" / "checkpoints" / "gate-eval" / f"gate4-full-expanded-step-{STEP}.json"
PROXY = "6c9efizq1aoocf-64411022"


def main() -> int:
    stdin = f"cat '{REMOTE}'\nexit\n"
    proc = subprocess.run(
        [
            ctl.SSH_BIN,
            "-tt",
            "-i",
            ctl.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            f"{PROXY}@{ctl.SSH_HOST if hasattr(ctl, 'SSH_HOST') else 'ssh.runpod.io'}",
            "bash",
            "-s",
        ],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    raw = proc.stdout
    idx = raw.find("{")
    if idx < 0:
        print("no JSON in ssh output", file=sys.stderr)
        return 1
    payload = raw[idx:]
    decoder = json.JSONDecoder()
    data, end = decoder.raw_decode(payload)
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    LOCAL.write_text(payload[:end], encoding="utf-8")
    m = data.get("aggregate_metrics", data)
    g = data.get("gate", {})
    results = data.get("results", [])
    parse_fails = sum(1 for r in results if not r.get("parse_valid"))
    promote_parse = sum(
        1
        for r in results
        if r.get("expected_action") == "promote_semantic" and not r.get("parse_valid")
    )
    out = {
        "checkpoint": data.get("checkpoint"),
        "parse_valid_rate": m.get("parse_valid_rate"),
        "schema_valid_rate": m.get("schema_valid_rate"),
        "action_accuracy": m.get("action_accuracy"),
        "gate_passed": g.get("passed"),
        "gate_failures": g.get("failures"),
        "total_probes": len(results),
        "parse_fails": parse_fails,
        "promote_semantic_parse_fails": promote_parse,
        "local_path": str(LOCAL),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
