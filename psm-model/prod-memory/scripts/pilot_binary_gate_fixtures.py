#!/usr/bin/env python3
"""Binary gate classify eval on fixtures for one OpenRouter model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.binary_gate_teacher import DEFAULT_BINARY_MODEL, classify_binary  # noqa: E402
from prod_memory.curriculum_sources import load_fixture_cases  # noqa: E402
from prod_memory.openrouter_teacher import TeacherConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_BINARY_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = TeacherConfig.from_env(model=args.model)
    if not cfg.api_key:
        print("OPENROUTER_API_KEY required", file=sys.stderr)
        return 1

    hits = 0
    cases: list[dict] = []
    for case in load_fixture_cases():
        label = classify_binary(case["llmResponse"], config=cfg, model=args.model)
        exp = case["expectAction"]
        ok = label.label == exp
        hits += int(ok)
        row = {
            "id": case["id"],
            "expect": exp,
            "got": label.label,
            "raw": label.raw,
            "ok": ok,
        }
        cases.append(row)
        print(f"{case['id']}: expect={exp} got={label.label} raw={label.raw!r} ok={ok}")

    report = {"model": args.model, "match": f"{hits}/10", "cases": cases}
    print(f"{args.model} binary gate: {hits}/10")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
