from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from psm_model.eval_generation import _parse_output
from psm_model.generate import generate_storage_json
from psm_model.schema import validate_storage_decision
from psm_model.storage_decision_repair import FAILSAFE_DECISION, RepairResult, repair_storage_decision

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def to_model_input(payload: dict[str, Any]) -> dict[str, Any]:
    conversation = payload.get("conversation")
    if isinstance(conversation, list):
        lines: list[str] = []
        for message in conversation:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        conversation_text = "\n".join(lines)
        # remember() without userMessage sends assistant-only text; probes train on User: lines.
        if lines and not any(line.startswith("User:") for line in lines):
            if len(lines) == 1 and lines[0].startswith("Assistant:"):
                conversation_text = f"User: {lines[0].split(':', 1)[1].strip()}"
    elif isinstance(conversation, str):
        conversation_text = conversation
    else:
        conversation_text = ""
    model_input: dict[str, Any] = {"conversation": conversation_text}
    temporal_markers = ("yesterday", "last week", "last month", "today", "tomorrow", "next week", "next month")
    needs_timestamp = any(marker in conversation_text.lower() for marker in temporal_markers)
    if needs_timestamp:
        for key in ("source_id", "source_timestamp"):
            value = payload.get(key)
            if value:
                model_input[key] = value
        source = payload.get("source")
        if isinstance(source, dict):
            for key in ("source_id", "source_timestamp"):
                value = source.get(key)
                if value and key not in model_input:
                    model_input[key] = value
    return model_input


def _looks_tagged(raw: str) -> bool:
    return any(
        line.strip().startswith(("A:", "M:", "T:", "C:", "Q:", "G:", "F:", "R:", "END"))
        for line in raw.splitlines()
    )


def _extract_json_object(raw: str) -> str | None:
    match = _JSON_OBJECT_RE.search(raw)
    return match.group(0) if match else None


def _repair_json_decision(raw: str) -> RepairResult:
    blob = _extract_json_object(raw) or raw.strip()
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        return RepairResult(
            status="failed_safe",
            decision=dict(FAILSAFE_DECISION),
            issues=[str(exc)],
        )
    if not isinstance(parsed, dict):
        return RepairResult(
            status="failed_safe",
            decision=dict(FAILSAFE_DECISION),
            issues=["invalid_json_shape"],
        )
    repairs: list[str] = []
    decision = dict(parsed)
    if decision.get("action") == "ignore":
        if decision.get("memory") is not None:
            decision["memory"] = None
            repairs.append("cleared_ignore_memory")
        if decision.get("facts") is None:
            decision["facts"] = []
            repairs.append("filled_ignore_facts")
        if not str(decision.get("reasoning") or "").strip():
            decision["reasoning"] = "Nothing durable to store."
            repairs.append("synthesized_ignore_reasoning")
    if not str(decision.get("reasoning") or "").strip():
        memory = decision.get("memory")
        if isinstance(memory, dict) and memory.get("content"):
            decision["reasoning"] = f"Stored: {memory['content']}"
            repairs.append("synthesized_reasoning_from_content")
    result = validate_storage_decision(decision)
    if result.ok:
        return RepairResult(
            status="repaired" if repairs else "parsed",
            decision=decision,
            repairs=repairs,
        )
    return RepairResult(
        status="failed_safe",
        decision=dict(FAILSAFE_DECISION),
        issues=[f"{issue.path}: {issue.message}" for issue in result.issues],
    )


def apply_product_boundary(raw: str, *, output_format: str = "tagged") -> dict[str, Any]:
    """Product boundary: strict model parse + deterministic repair + fail-safe ignore."""
    if raw.lstrip().startswith("{"):
        result = _repair_json_decision(raw)
    elif output_format == "tagged" or _looks_tagged(raw):
        result = repair_storage_decision(raw)
    else:
        parsed, parse_issues = _parse_output(raw, output_format)
        validation = validate_storage_decision(parsed) if parsed is not None else None
        if validation and validation.ok and not parse_issues:
            result = RepairResult(status="parsed", decision=parsed)
        else:
            result = RepairResult(
                status="failed_safe",
                decision=dict(FAILSAFE_DECISION),
                issues=[f"{issue.path}: {issue.message}" for issue in (parse_issues or ())],
            )
    return {
        "parsed": result.decision,
        "valid": True,
        "model_parse_valid": result.status == "parsed",
        "repair_status": result.status,
        "repairs": result.repairs,
        "issues": [{"path": "", "message": issue} for issue in result.issues],
    }


def remember_from_repair_payload(payload: dict[str, Any], *, output_format: str = "tagged") -> dict[str, Any]:
    invalid = str(payload.get("invalid_model_output") or "")
    boundary = apply_product_boundary(invalid, output_format=output_format)
    return {
        "raw": invalid,
        "output_format": output_format,
        **boundary,
        "model_input": {"operation": "repair_remember_json"},
    }


def remember_storage_decision(
    checkpoint: Path,
    payload: dict[str, Any],
    *,
    output_format: str = "tagged",
    device: str = "auto",
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    if payload.get("operation") == "repair_remember_json":
        return remember_from_repair_payload(payload, output_format=output_format)

    conversation = payload.get("conversation")
    if isinstance(conversation, list) or payload.get("operation") == "remember_llm_response":
        model_input = to_model_input(payload)
    elif isinstance(conversation, str):
        model_input = payload
    else:
        model_input = to_model_input(payload)
    raw = generate_storage_json(
        checkpoint,
        model_input,
        max_new_tokens=max_new_tokens,
        output_format=output_format,
        device=device,
    )
    boundary = apply_product_boundary(raw, output_format=output_format)
    return {
        "raw": raw,
        "output_format": output_format,
        "model_input": model_input,
        **boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PSM StorageDecision JSON for psm-core remember().")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--input", help="JSON model input object (default: read remember payload JSON from stdin)")
    parser.add_argument("--output-format", default="tagged", choices=["json", "tagged", "at_tag"])
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    if args.input:
        payload = json.loads(args.input)
    else:
        payload = json.load(sys.stdin)
    report = remember_storage_decision(
        args.checkpoint,
        payload,
        output_format=args.output_format,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("parsed") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
