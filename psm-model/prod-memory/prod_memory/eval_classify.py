from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.generate import generate_storage_json, open_generation_session

from prod_memory.grounding import would_model_store

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = PACKAGE_ROOT / "fixtures" / "cases.json"


def _first_line(raw: str) -> str:
    for line in raw.splitlines():
        stripped = line.strip().lower()
        if stripped:
            return stripped
    return ""


def binary_predicts_store(raw: str) -> bool:
    line = _first_line(raw)
    if not line or line in {"ignore", "ignore_noise"}:
        return False
    return line == "store" or line.startswith("store") or line.startswith("store_")


def _predicts_store(raw: str, *, output_format: str) -> bool:
    if output_format == "binary":
        return binary_predicts_store(raw)
    if output_format == "minimal":
        line = _first_line(raw)
        return line.startswith("store:")
    if output_format == "action":
        line = _first_line(raw)
        if line.startswith("a:"):
            action = line.split(":", 1)[1].strip()
            return action not in {"ignore", "ignore_noise"}
        return False
    return would_model_store({"action": "store_episodic", "memory": {"content": raw}})


def classify_match(expect_action: str, raw: str, *, output_format: str) -> bool:
    predicts_store = _predicts_store(raw, output_format=output_format)
    if expect_action == "ignore":
        return not predicts_store
    return predicts_store


def run_case(
    case: dict[str, Any],
    *,
    session: Any,
    checkpoint: Path,
    output_format: str,
    device: str,
    max_new_tokens: int,
    raw_input: bool,
) -> dict[str, Any]:
    from psm_model.remember_cli import to_model_input

    payload = {
        "operation": "remember_llm_response",
        "conversation": [{"role": "assistant", "content": str(case["llmResponse"])}],
    }
    model_input = payload if raw_input else to_model_input(payload)
    raw = generate_storage_json(
        checkpoint,
        model_input,
        max_new_tokens=max_new_tokens,
        output_format=output_format,
        device=device,
        session=session,
    )
    expect = str(case.get("expectAction") or "store")
    matched = classify_match(expect, raw, output_format=output_format)
    return {
        "id": case["id"],
        "expectAction": expect,
        "raw_output": raw.strip()[:120],
        "predicts_store": _predicts_store(raw, output_format=output_format),
        "classify_match": matched,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binary store/ignore classification eval on prod fixtures.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-format", default="binary", choices=["binary", "minimal", "action"])
    parser.add_argument("--raw-input", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--fixture-ids", default="", help="Comma-separated subset; default all fixtures")
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = [c for c in fixture.get("cases", []) if isinstance(c, dict)]
    if args.fixture_ids.strip():
        want = {x.strip() for x in args.fixture_ids.split(",") if x.strip()}
        cases = [c for c in cases if c.get("id") in want]

    session = open_generation_session(args.checkpoint, output_format=args.output_format, device=args.device)
    results = [
        run_case(
            case,
            session=session,
            checkpoint=args.checkpoint,
            output_format=args.output_format,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            raw_input=args.raw_input,
        )
        for case in cases
    ]
    matched = sum(1 for row in results if row["classify_match"])
    report = {
        "checkpoint": str(args.checkpoint),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_format": args.output_format,
        "aggregate": {"cases": len(results), "classify_match": matched},
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
