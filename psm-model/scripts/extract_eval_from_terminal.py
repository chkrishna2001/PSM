#!/usr/bin/env python3
"""Extract eval_checkpoint JSON from a runpod_ctl terminal log."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def extract(path: Path, *, step: int) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    needle = f"step-{step:06d}.pt"
    best: dict | None = None
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[i : j + 1]
                    if needle in chunk and '"gate_mode": "expanded"' in chunk:
                        try:
                            obj = json.loads(chunk)
                        except json.JSONDecodeError:
                            obj = None
                        if isinstance(obj, dict) and obj.get("examples"):
                            best = obj
                    i = j + 1
                    break
        else:
            i += 1
    if best is None:
        raise SystemExit(f"expanded eval JSON for step {step} not found in {path}")
    return best


def main() -> int:
    log = Path(sys.argv[1])
    step = int(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        f"psm-model/checkpoints/gate-eval/gate4-full-expanded-step-{step:06d}.json"
    )
    report = extract(log, step=step)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(out), "parse": report["parse_valid_rate"], "action": report["action_accuracy"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
