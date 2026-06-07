from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from psm_model.eval_generation import _parse_output
from psm_model.generate import generate_storage_json
from psm_model.schema import validate_storage_decision


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


def remember_storage_decision(
    checkpoint: Path,
    payload: dict[str, Any],
    *,
    output_format: str = "tagged",
    device: str = "cpu",
    max_new_tokens: int = 1200,
) -> dict[str, Any]:
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
    parsed, parse_issues = _parse_output(raw, output_format)
    validation = validate_storage_decision(parsed) if parsed is not None else None
    return {
        "raw": raw,
        "parsed": parsed,
        "output_format": output_format,
        "valid": bool(validation and validation.ok and not parse_issues),
        "issues": [{"path": issue.path, "message": issue.message} for issue in (parse_issues or ())],
        "model_input": model_input,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a PSM StorageDecision JSON for psm-core remember().")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--input", help="JSON model input object (default: read remember payload JSON from stdin)")
    parser.add_argument("--output-format", default="tagged", choices=["json", "tagged", "at_tag"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
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
    return 0 if report["valid"] and report["parsed"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
