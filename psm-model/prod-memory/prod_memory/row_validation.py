from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psm_model.data import validate_training_row
from psm_model.schema import validate_storage_decision

from prod_memory.grounding import has_curriculum_bleed, is_grounded_in_source


def remember_target_from_input(input_payload: dict[str, Any]) -> str:
    conversation = input_payload.get("conversation")
    if isinstance(conversation, list):
        parts: list[str] = []
        for message in conversation:
            if isinstance(message, dict) and message.get("content"):
                parts.append(str(message["content"]))
        return "\n".join(parts).strip()
    if isinstance(conversation, str):
        return conversation.strip()
    return ""


def label_text_from_expected(expected: dict[str, Any]) -> str:
    parts: list[str] = []
    memory = expected.get("memory")
    if isinstance(memory, dict) and memory.get("content"):
        parts.append(str(memory["content"]))
    for fact in expected.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        for key in ("subject", "predicate", "value_text", "value", "evidence_text"):
            if fact.get(key):
                parts.append(str(fact[key]))
    for indexable in expected.get("indexables") or []:
        if not isinstance(indexable, dict):
            continue
        for key in ("key", "reconstructive_hint", "evidence_text"):
            if indexable.get(key):
                parts.append(str(indexable[key]))
        for step in indexable.get("steps") or []:
            parts.append(str(step))
    if expected.get("reasoning"):
        parts.append(str(expected["reasoning"]))
    return " ".join(parts)


def bleed_check_text_from_expected(expected: dict[str, Any]) -> str:
    """Match psm-core applyStorageGuards: memory.content + fact fields only."""
    parts: list[str] = []
    memory = expected.get("memory")
    if isinstance(memory, dict) and memory.get("content"):
        parts.append(str(memory["content"]))
    for fact in expected.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        for key in ("subject", "predicate", "value_text", "value", "evidence_text"):
            if fact.get(key):
                parts.append(str(fact[key]))
    return " ".join(parts)


def validate_prod_row(row: dict[str, Any]) -> None:
    row_id = str(row.get("id") or "row")
    _, issues = validate_training_row(row)
    if issues:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise ValueError(f"{row_id}: {formatted}")

    expected = row.get("expected")
    if not isinstance(expected, dict):
        raise ValueError(f"{row_id}: missing expected object")

    result = validate_storage_decision(expected)
    if not result.ok:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        raise ValueError(f"{row_id}: invalid storage decision: {formatted}")

    input_payload = row.get("input")
    if not isinstance(input_payload, dict):
        raise ValueError(f"{row_id}: missing input object")

    label_text = bleed_check_text_from_expected(expected)
    if has_curriculum_bleed(label_text):
        raise ValueError(f"{row_id}: curriculum bleed in labels")

    action = str(expected.get("action") or "")
    if action == "ignore":
        return

    remember_target = remember_target_from_input(input_payload)
    grounding_text = label_text_from_expected(expected)
    if remember_target and not is_grounded_in_source(remember_target, grounding_text):
        raise ValueError(f"{row_id}: labels not grounded in remember_target")


def validate_prod_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("id") or "row")
        try:
            validate_prod_row(row)
        except ValueError as exc:
            failures.append({"id": row_id, "error": str(exc)})
    return {
        "total": len(rows),
        "valid": len(rows) - len(failures),
        "failures": failures,
        "ok": not failures,
    }


def write_jsonl(path: Any, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
