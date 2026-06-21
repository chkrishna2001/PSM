from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.generate import generate_storage_json, open_generation_session
from psm_model.remember_cli import apply_product_boundary, to_model_input

from prod_memory.eval_classify import _predicts_store, classify_match
from prod_memory.eval_grounding import build_remember_payload
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


def run_two_pass_case(
    case: dict[str, Any],
    *,
    binary_checkpoint: Path,
    extract_checkpoint: Path,
    binary_session: Any,
    extract_session: Any,
    device: str,
    raw_input: bool,
    binary_max_tokens: int,
    extract_max_tokens: int,
) -> dict[str, Any]:
    llm_response = str(case["llmResponse"])
    payload = build_remember_payload(llm_response)
    model_input = payload if raw_input else to_model_input(payload)

    raw_binary = generate_storage_json(
        binary_checkpoint,
        model_input,
        max_new_tokens=binary_max_tokens,
        output_format="binary",
        device=device,
        session=binary_session,
    )
    expect = str(case.get("expectAction") or "store")
    classify_ok = classify_match(expect, raw_binary, output_format="binary")

    raw_extract = ""
    if _predicts_store(raw_binary, output_format="binary"):
        raw_extract = generate_storage_json(
            extract_checkpoint,
            model_input,
            max_new_tokens=extract_max_tokens,
            output_format="minimal_extract",
            device=device,
            session=extract_session,
        )
        report = apply_product_boundary(raw_extract, output_format="minimal")
    else:
        report = apply_product_boundary("ignore", output_format="minimal")

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
        "suite": case.get("suite"),
        "expectAction": expect,
        "classify_match": classify_ok,
        "binary_output": raw_binary.strip()[:80],
        "extract_output": raw_extract.strip()[:200] if raw_extract else None,
        "action": decision.get("action"),
        "repair_status": report.get("repair_status"),
        "model_would_store": model_store,
        "effective_stored": effective_stored,
        "content_grounded": content_grounded,
        "guard_rejected": guarded["rejected"],
        "fail_safe": is_fail_safe_report(report),
        "curriculum_bleed": effective_stored and has_curriculum_bleed(stored_text),
        "memory_content": stored_text[:240] if stored_text else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-pass prod eval: binary classify ckpt + extract ckpt.")
    parser.add_argument("--binary-checkpoint", type=Path, required=True)
    parser.add_argument("--extract-checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture-ids", default="")
    parser.add_argument("--raw-input", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--binary-max-tokens", type=int, default=32)
    parser.add_argument("--extract-max-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = [c for c in fixture.get("cases", []) if isinstance(c, dict)]
    if args.fixture_ids.strip():
        want = {x.strip() for x in args.fixture_ids.split(",") if x.strip()}
        cases = [c for c in cases if c.get("id") in want]

    binary_session = open_generation_session(args.binary_checkpoint, output_format="binary", device=args.device)
    extract_session = open_generation_session(args.extract_checkpoint, output_format="minimal_extract", device=args.device)

    results = [
        run_two_pass_case(
            case,
            binary_checkpoint=args.binary_checkpoint,
            extract_checkpoint=args.extract_checkpoint,
            binary_session=binary_session,
            extract_session=extract_session,
            device=args.device,
            raw_input=args.raw_input,
            binary_max_tokens=args.binary_max_tokens,
            extract_max_tokens=args.extract_max_tokens,
        )
        for case in cases
    ]

    classify_hits = sum(1 for row in results if row["classify_match"])
    ground_hits = sum(1 for row in results if row["effective_stored"])
    report = {
        "binary_checkpoint": str(args.binary_checkpoint),
        "extract_checkpoint": str(args.extract_checkpoint),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "cases": len(results),
            "classify_match": classify_hits,
            "effective_stored": ground_hits,
        },
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
