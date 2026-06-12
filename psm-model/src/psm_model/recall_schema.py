from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

MEMORY_TABLES = frozenset({"episodic", "semantic", "archival"})


@dataclass(frozen=True)
class RecallPlan:
    intent: str
    target_tables: tuple[str, ...]
    filters: dict[str, Any]
    ranking_hints: tuple[str, ...]
    temporal_intent: str | None
    top_k: int


@dataclass(frozen=True)
class RecallValidationResult:
    ok: bool
    issues: tuple[str, ...]
    plan: RecallPlan | None = None


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return tuple(items)


def _normalize_tables(value: Any) -> tuple[str, ...]:
    tables = [table for table in _string_list(value) if table in MEMORY_TABLES]
    if tables:
        return tuple(dict.fromkeys(tables))
    return ("semantic", "episodic")


def validate_recall_plan(value: Any) -> RecallValidationResult:
    issues: list[str] = []
    if not isinstance(value, dict):
        return RecallValidationResult(False, ("expected object",), None)

    intent = value.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        issues.append("intent must be a non-empty string")

    target_tables = _normalize_tables(value.get("target_tables"))
    if not target_tables:
        issues.append("target_tables must include episodic, semantic, or archival")

    filters = value.get("filters")
    if filters is None:
        filters = {}
    elif not isinstance(filters, dict):
        issues.append("filters must be an object")
        filters = {}

    ranking_hints = _string_list(value.get("ranking_hints"))
    temporal_intent = value.get("temporal_intent")
    if temporal_intent is not None and (not isinstance(temporal_intent, str) or not temporal_intent.strip()):
        issues.append("temporal_intent must be a non-empty string when present")
        temporal_intent = None
    elif isinstance(temporal_intent, str):
        temporal_intent = temporal_intent.strip()

    top_k_raw = value.get("top_k", 5)
    top_k = int(top_k_raw) if isinstance(top_k_raw, int) or isinstance(top_k_raw, float) else 5
    if top_k < 1 or top_k > 20:
        issues.append("top_k must be between 1 and 20")

    if issues:
        return RecallValidationResult(False, tuple(issues), None)

    return RecallValidationResult(
        True,
        (),
        RecallPlan(
            intent=str(intent).strip(),
            target_tables=target_tables,
            filters=filters,
            ranking_hints=ranking_hints,
            temporal_intent=temporal_intent,
            top_k=top_k,
        ),
    )


def parse_recall_plan_json(raw_text: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, ("no JSON object found",)
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, (f"invalid JSON: {exc.msg}",)
    if not isinstance(parsed, dict):
        return None, ("parsed value must be an object",)
    result = validate_recall_plan(parsed)
    if not result.ok or result.plan is None:
        return parsed, result.issues
    return parsed, ()


def tables_match(expected: tuple[str, ...], predicted: tuple[str, ...]) -> bool:
    return set(expected) == set(predicted)


def primary_table_match(expected: tuple[str, ...], predicted: tuple[str, ...]) -> bool:
    if not expected or not predicted:
        return False
    return expected[0] in predicted


def ranking_hints_score(expected: tuple[str, ...], predicted: tuple[str, ...]) -> float:
    if not expected:
        return 1.0
    expected_tokens = {token.lower() for hint in expected for token in hint.lower().split() if token}
    predicted_tokens = {token.lower() for hint in predicted for token in hint.lower().split() if token}
    if not expected_tokens:
        return 1.0
    overlap = len(expected_tokens & predicted_tokens)
    return overlap / len(expected_tokens)


def score_recall_plan(expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, Any]:
    expected_result = validate_recall_plan(expected)
    predicted_result = validate_recall_plan(predicted)
    if not expected_result.ok or expected_result.plan is None:
        raise ValueError("expected recall plan is invalid")
    exp = expected_result.plan
    if not predicted_result.ok or predicted_result.plan is None:
        return {
            "parse_valid": False,
            "schema_valid": False,
            "target_tables_exact": False,
            "target_tables_primary": False,
            "ranking_hints_score": 0.0,
            "top_k_exact": False,
            "temporal_intent_exact": False,
        }
    pred = predicted_result.plan
    return {
        "parse_valid": True,
        "schema_valid": True,
        "target_tables_exact": tables_match(exp.target_tables, pred.target_tables),
        "target_tables_primary": primary_table_match(exp.target_tables, pred.target_tables),
        "ranking_hints_score": ranking_hints_score(exp.ranking_hints, pred.ranking_hints),
        "top_k_exact": exp.top_k == pred.top_k,
        "temporal_intent_exact": (exp.temporal_intent or "").lower() == (pred.temporal_intent or "").lower(),
    }
