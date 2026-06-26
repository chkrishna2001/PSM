from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.remember_cli import apply_product_boundary

from prod_memory.eval_classify import _predicts_store, classify_match
from prod_memory.eval_grounding import DEFAULT_FIXTURES, aggregate_by_suite, aggregate_metrics
from prod_memory.eval_hf_grounding import HfGenerationSession, open_hf_session
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


def template_collapse_prefixes(cases: list[dict[str, Any]], *, min_shared: int = 3, prefix_len: int = 40) -> dict[str, Any]:
    """Reject promotion when many cases share identical output prefix (memorized template)."""
    prefixes = [
        str(row.get("extract_output") or row.get("binary_output") or row.get("raw_output") or "")[:prefix_len]
        for row in cases
        if str(row.get("extract_output") or row.get("binary_output") or row.get("raw_output") or "").strip()
    ]
    if not prefixes:
        return {"collapsed": False, "top_prefix": "", "top_count": 0}
    top_prefix, top_count = Counter(prefixes).most_common(1)[0]
    return {"collapsed": top_count >= min_shared, "top_prefix": top_prefix, "top_count": top_count}


def run_hf_two_pass_case(
    case: dict[str, Any],
    *,
    binary_session: HfGenerationSession,
    extract_session: HfGenerationSession,
    binary_max_tokens: int,
    extract_max_tokens: int,
) -> dict[str, Any]:
    llm_response = str(case["llmResponse"])
    expect = str(case.get("expectAction") or "store")

    raw_binary = binary_session.generate(llm_response, output_format="binary", max_new_tokens=binary_max_tokens)
    classify_ok = classify_match(expect, raw_binary, output_format="binary")

    raw_extract = ""
    if _predicts_store(raw_binary, output_format="binary"):
        raw_extract = extract_session.generate(
            llm_response,
            output_format="minimal_extract",
            max_new_tokens=extract_max_tokens,
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
    action_match = classify_ok and (
        (expect == "ignore" and not effective_stored) or (expect != "ignore" and effective_stored and content_grounded)
    )
    return {
        "id": case["id"],
        "suite": case.get("suite"),
        "expectAction": expect,
        "classify_match": classify_ok,
        "action_match": action_match,
        "binary_output": raw_binary.strip()[:80],
        "extract_output": raw_extract.strip()[:240] if raw_extract else None,
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
    parser = argparse.ArgumentParser(description="Two-pass HF LoRA eval: binary gate + minimal_extract.")
    parser.add_argument("--binary-adapter", type=Path, required=True)
    parser.add_argument("--extract-adapter", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--binary-max-tokens", type=int, default=16)
    parser.add_argument("--extract-max-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = [c for c in fixture.get("cases", []) if isinstance(c, dict)]

    binary_session = open_hf_session(args.binary_adapter, model_key=args.model, device=args.device)
    extract_session = open_hf_session(args.extract_adapter, model_key=args.model, device=args.device)

    results = [
        run_hf_two_pass_case(
            case,
            binary_session=binary_session,
            extract_session=extract_session,
            binary_max_tokens=args.binary_max_tokens,
            extract_max_tokens=args.extract_max_tokens,
        )
        for case in cases
    ]
    collapse = template_collapse_prefixes(results)
    classify_hits = sum(1 for row in results if row["classify_match"])
    action_hits = sum(1 for row in results if row["action_match"])
    ground_hits = sum(1 for row in results if row["effective_stored"])
    aggregate = aggregate_metrics(results)
    report = {
        "binary_adapter": str(args.binary_adapter),
        "extract_adapter": str(args.extract_adapter),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_type": "prod_fixtures_hf_two_pass",
        "template_collapse": collapse,
        "aggregate": {
            **aggregate,
            "classify_match": classify_hits,
            "action_match": action_hits,
            "effective_stored": ground_hits,
        },
        "suites": aggregate_by_suite(results),
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    if collapse["collapsed"]:
        print(f"TEMPLATE_COLLAPSE prefix={collapse['top_prefix']!r} count={collapse['top_count']}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
