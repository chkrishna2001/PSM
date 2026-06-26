#!/usr/bin/env python3
"""Gate-only classify eval on fixtures — run before v5k-gate train to justify GPU spend."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.eval_classify import classify_match  # noqa: E402
from prod_memory.eval_grounding import DEFAULT_FIXTURES  # noqa: E402
from prod_memory.eval_hf_grounding import open_hf_session  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=None, help="LoRA adapter; omit for base Qwen only")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cases = json.loads(args.fixtures.read_text(encoding="utf-8")).get("cases", [])
    session = open_hf_session(args.adapter_dir, model_key=args.model, device=args.device)

    hits = 0
    for case in cases:
        if not isinstance(case, dict):
            continue
        raw = session.generate(str(case["llmResponse"]), output_format="binary", max_new_tokens=16)
        ok = classify_match(str(case.get("expectAction") or "store"), raw, output_format="binary")
        hits += int(ok)
        print(f"{case['id']}: expect={case.get('expectAction')} raw={raw.strip()[:40]!r} ok={ok}")
    print(json.dumps({"classify_match": hits, "cases": len(cases)}, indent=2))
    return 0 if hits >= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
