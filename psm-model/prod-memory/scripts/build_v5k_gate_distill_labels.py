#!/usr/bin/env python3
"""Label fixtures + generate Claude noise variants for v5k-gate-distill curriculum."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.binary_gate_teacher import (  # noqa: E402
    DEFAULT_BINARY_MODEL,
    classify_binary,
    generate_noise_variants,
)
from prod_memory.curriculum_sources import load_fixture_cases  # noqa: E402
from prod_memory.openrouter_teacher import TeacherConfig  # noqa: E402

DEFAULT_CACHE = PACKAGE_ROOT / "data" / "v5k-gate-distill-cache.json"

FALLBACK_NOISE = [
    "Sure, no problem. Happy to help anytime.",
    "Got it — I'll wait for your next message.",
    "Thanks! Let me know when you're ready to continue.",
    "Okay, understood. Nothing else to add from my side.",
    "I don't have any new durable information in this reply.",
    "This is just an acknowledgment — no preferences or procedures here.",
    "Sounds good. I have no concrete facts to remember from that.",
    "Noted. There's nothing worth storing from this short reply.",
    "All set on my end. No handoff content to capture.",
    "Right — that's conversational filler without durable memory.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_BINARY_MODEL)
    parser.add_argument("--noise-count", type=int, default=30)
    parser.add_argument("--out", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--skip-noise-gen", action="store_true", help="Reuse existing noise_variants in cache")
    args = parser.parse_args()

    cfg = TeacherConfig.from_env(model=args.model)
    if not cfg.api_key:
        print("OPENROUTER_API_KEY required", file=sys.stderr)
        return 1

    existing: dict = {}
    if args.out.is_file():
        existing = json.loads(args.out.read_text(encoding="utf-8"))

    fixture_audit: list[dict] = []
    mismatches = 0
    for case in load_fixture_cases():
        label = classify_binary(case["llmResponse"], config=cfg, model=args.model)
        expect = case["expectAction"]
        ok = label.label == expect
        mismatches += int(not ok)
        fixture_audit.append(
            {
                "id": case["id"],
                "expect": expect,
                "teacher": label.label,
                "ok": ok,
                "raw": label.raw,
            }
        )
        print(f"{case['id']}: expect={expect} teacher={label.label} ok={ok}")

    noise_variants = list(existing.get("noise_variants") or [])
    if not args.skip_noise_gen or not noise_variants:
        print(f"Generating {args.noise_count} noise variants via {args.model}...")
        noise_variants = generate_noise_variants(count=args.noise_count, config=cfg, model=args.model)
        for idx, text in enumerate(noise_variants[:5]):
            print(f"  noise[{idx}]: {text[:80]}")

    payload = {
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture_audit": fixture_audit,
        "fixture_mismatches": mismatches,
        "noise_variants": noise_variants,
        "fallback_noise": FALLBACK_NOISE,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} ({len(noise_variants)} noise, {mismatches} fixture mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
