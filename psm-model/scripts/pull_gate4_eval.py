#!/usr/bin/env python3
"""Pull gate4 eval report from pod and print key metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as ctl  # noqa: E402

REMOTE = "/workspace/PSM/psm-model/checkpoints/gate-eval/gate4-full-expanded-step-043400.json"
LOCAL = REPO / "psm-model/checkpoints/gate-eval/gate4-full-expanded-step-043400.json"
PROXY = "6c9efizq1aoocf-64411022"


def main() -> int:
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    rc = ctl._scp_from_pod(
        ctl.SSH_CONFIG_HOST,
        REMOTE,
        LOCAL.parent,
        host="ssh.runpod.io",
        port="22",
        user=PROXY,
    )
    if not LOCAL.is_file():
        print(f"pull failed rc={rc}", file=sys.stderr)
        return 1
    data = json.loads(LOCAL.read_text(encoding="utf-8"))
    metrics = data.get("aggregate_metrics", data)
    gate = data.get("gate", {})
    print(json.dumps({
        "checkpoint": data.get("checkpoint"),
        "parse_valid_rate": metrics.get("parse_valid_rate"),
        "schema_valid_rate": metrics.get("schema_valid_rate"),
        "action_accuracy": metrics.get("action_accuracy"),
        "gate_passed": gate.get("passed"),
        "gate_failures": gate.get("failures"),
        "local_path": str(LOCAL),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
