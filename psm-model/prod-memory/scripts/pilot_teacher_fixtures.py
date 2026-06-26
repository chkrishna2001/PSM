#!/usr/bin/env python3
"""Quick fixture action match for one OpenRouter teacher model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.curriculum_sources import load_fixture_cases  # noqa: E402
from prod_memory.openrouter_teacher import TeacherConfig, label_assistant_with_teacher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anthropic/claude-sonnet-4")
    args = parser.parse_args()
    cfg = TeacherConfig.from_env(model=args.model)
    if not cfg.api_key:
        print("OPENROUTER_API_KEY required", file=sys.stderr)
        return 1
    hits = 0
    for case in load_fixture_cases():
        expected, meta = label_assistant_with_teacher(case["llmResponse"], config=cfg)
        exp = case["expectAction"]
        action = str((expected or {}).get("action") or "ignore")
        got = "ignore" if action == "ignore" else "store"
        ok = got == exp
        hits += int(ok)
        print(f"{case['id']}: expect={exp} got={got} action={action} ok={ok}")
    print(f"{args.model} JSON teacher: {hits}/10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
