from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.generate import generate_storage_json, open_generation_session
from psm_model.remember_cli import apply_product_boundary, to_model_input

from prod_memory.grounding import (
    apply_storage_guards,
    grounding_overlap_score,
    has_curriculum_bleed,
    is_fail_safe_report,
    key_tokens_grounded,
    stored_text_from_decision,
    would_model_store,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = PACKAGE_ROOT / "fixtures" / "cases.json"
DEFAULT_OUT = PACKAGE_ROOT / "results" / "prod-grounding-baseline.json"


def build_remember_payload(llm_response: str) -> dict[str, Any]:
    return {
        "operation": "remember_llm_response",
        "conversation": [{"role": "assistant", "content": llm_response}],
    }


def run_case(
    checkpoint: Path,
    case: dict[str, Any],
    *,
    session: Any,
    output_format: str,
    device: str,
    max_new_tokens: int,
    raw_input: bool = False,
) -> dict[str, Any]:
    llm_response = str(case["llmResponse"])
    payload = build_remember_payload(llm_response)
    model_input = payload if raw_input else to_model_input(payload)
    raw = generate_storage_json(
        checkpoint,
        model_input,
        max_new_tokens=max_new_tokens,
        output_format=output_format,
        device=device,
        session=session,
    )
    report = apply_product_boundary(raw, output_format=output_format)
    decision = report.get("parsed")
    if not isinstance(decision, dict):
        decision = {}
    stored_text = stored_text_from_decision(decision)
    model_store = would_model_store(decision)
    guarded = apply_storage_guards(llm_response, decision)
    effective_stored = model_store and not guarded["rejected"]
    overlap = grounding_overlap_score(llm_response, stored_text)
    key_tokens = case.get("keyTokens") if isinstance(case.get("keyTokens"), list) else []
    content_grounded = effective_stored and (
        key_tokens_grounded([str(token) for token in key_tokens], stored_text) or bool(overlap["grounded"])
    )
    return {
        "id": case["id"],
        "suite": case["suite"],
        "expectAction": case.get("expectAction"),
        "action": decision.get("action"),
        "repair_status": report.get("repair_status"),
        "model_would_store": model_store,
        "effective_stored": effective_stored,
        "guard_rejected": guarded["rejected"],
        "guard_route": guarded["route"],
        "fail_safe": is_fail_safe_report(report),
        "curriculum_bleed": effective_stored and has_curriculum_bleed(stored_text),
        "content_grounded": content_grounded,
        "grounding_overlap": overlap["overlap"],
        "grounding_required": overlap["required"],
        "memory_content": stored_text[:240] if stored_text else None,
        "issues": report.get("issues"),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = len(rows)
    model_stored = [row for row in rows if row.get("model_would_store")]
    effective = [row for row in rows if row.get("effective_stored")]
    return {
        "cases": cases,
        "model_stored": len(model_stored),
        "effective_stored": len(effective),
        "content_grounding_rate": _rate(effective, lambda row: bool(row.get("content_grounded"))),
        "curriculum_bleed_rate": _rate(effective, lambda row: bool(row.get("curriculum_bleed"))),
        "fail_safe_ignore_rate": round(sum(1 for row in rows if row.get("fail_safe")) / max(1, cases), 4),
        "parse_valid_rate": round(sum(1 for row in rows if row.get("repair_status") != "failed_safe") / max(1, cases), 4),
        "guard_reject_rate": round(sum(1 for row in rows if row.get("guard_rejected")) / max(1, cases), 4),
        "action_match_rate": _action_match_rate(rows),
    }


def aggregate_by_suite(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["suite"]), []).append(row)
    return {suite: aggregate_metrics(items) for suite, items in grouped.items()}


def _rate(rows: list[dict[str, Any]], predicate: Any) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4)


def _action_match_rate(rows: list[dict[str, Any]]) -> float:
    expected_rows = [row for row in rows if row.get("expectAction")]
    if not expected_rows:
        return 1.0
    hits = 0
    for row in expected_rows:
        expect = str(row["expectAction"])
        if expect == "ignore":
            if not row.get("effective_stored"):
                hits += 1
        elif row.get("effective_stored"):
            hits += 1
    return round(hits / len(expected_rows), 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prod-shaped remember_target grounding eval (isolated from gate evals).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-format", default="tagged", choices=["json", "tagged", "at_tag", "minimal", "minimal_extract", "binary"])
    parser.add_argument("--raw-input", action="store_true", help="Skip to_model_input rewrite (match training JSON input).")
    parser.add_argument("--fixture-ids", default="", help="Comma-separated case ids; default all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)
    if args.output_format in {"minimal", "binary"} and args.max_new_tokens == 384:
        args.max_new_tokens = 128

    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise SystemExit(f"Invalid fixtures file: {args.fixtures}")
    if args.fixture_ids.strip():
        want = {x.strip() for x in args.fixture_ids.split(",") if x.strip()}
        cases = [c for c in cases if isinstance(c, dict) and c.get("id") in want]

    label = args.checkpoint_label or args.checkpoint.stem
    args.out.parent.mkdir(parents=True, exist_ok=True)

    session = open_generation_session(args.checkpoint, output_format=args.output_format, device=args.device)
    results = [
        run_case(
            args.checkpoint,
            case,
            session=session,
            output_format=args.output_format,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            raw_input=args.raw_input,
        )
        for case in cases
        if isinstance(case, dict)
    ]

    aggregate = aggregate_metrics(results)
    report = {
        "checkpoint": label,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixtures": str(args.fixtures.resolve()),
        "max_new_tokens": args.max_new_tokens,
        "suites": aggregate_by_suite(results),
        "aggregate": aggregate,
        "cases": results,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": label, "aggregate": aggregate, "suites": report["suites"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
