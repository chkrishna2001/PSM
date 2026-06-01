from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


ACTIONS = frozenset(
    {
        "ignore",
        "store_episodic",
        "promote_semantic",
        "update_existing",
        "flag_conflict",
        "flag_and_store",
    }
)

MEMORY_TYPES = frozenset({"episodic", "semantic"})
INFERENCE_KINDS = frozenset({"explicit"})
_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_INVALID_EVIDENCE_SUBSTRINGS = (
    "benchmark dataset",
    "conversation-memory input",
    "extraction guidance",
    "current turn to remember",
    "previous context",
    "source id:",
)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: tuple[ValidationIssue, ...]
    decision: "StorageDecision | None" = None


@dataclass(frozen=True)
class Memory:
    content: str
    type: str
    strength: float | None = None
    decay_rate: float | None = None
    emotional_weight: float | None = None
    confidence: float | None = None
    tags: tuple[str, ...] = ()
    temporal_expression: str | None = None
    resolved_time: str | None = None
    resolved_time_confidence: float | None = None


@dataclass(frozen=True)
class Fact:
    subject: str
    predicate: str
    value: Any
    confidence: float | None
    inference_kind: str
    evidence_text: str
    object: str | None = None
    value_text: str | None = None
    value_json: Any = None
    fact_type: str | None = None
    temporal_expression: str | None = None
    resolved_time: str | None = None
    resolved_time_confidence: float | None = None


@dataclass(frozen=True)
class StorageDecision:
    action: str
    memory: Memory | None
    facts: tuple[Fact, ...]
    reasoning: str
    confidence: float | None = None
    emotional_weight: float | None = None
    contradiction_score: float | None = None


def parse_and_validate_storage_decision(raw_text: str) -> ValidationResult:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return _fail("$", f"invalid JSON: {exc.msg}")
    return validate_storage_decision(parsed)


def validate_storage_decision(value: Any) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if not isinstance(value, dict):
        return _fail("$", "storage decision must be a JSON object")

    action = _required_string(value, "action", "$.action", issues)
    if action and action not in ACTIONS:
        issues.append(ValidationIssue("$.action", f"unsupported action: {action}"))

    memory = _validate_memory(value.get("memory"), "$.memory", issues)
    if action == "ignore" and memory is not None:
        issues.append(ValidationIssue("$.memory", "ignore decisions must use null memory"))
    if action and action != "ignore" and memory is None:
        issues.append(ValidationIssue("$.memory", f"{action} decisions require memory"))

    facts = _validate_facts(value.get("facts", []), "$.facts", issues)
    reasoning = _required_string(value, "reasoning", "$.reasoning", issues)
    confidence = _optional_number(value, "confidence", "$.confidence", issues)
    emotional_weight = _optional_number(value, "emotional_weight", "$.emotional_weight", issues)
    contradiction_score = _optional_number(value, "contradiction_score", "$.contradiction_score", issues)

    if issues:
        return ValidationResult(False, tuple(issues))

    decision = StorageDecision(
        action=action,
        memory=memory,
        facts=tuple(facts),
        reasoning=reasoning,
        confidence=confidence,
        emotional_weight=emotional_weight,
        contradiction_score=contradiction_score,
    )
    return ValidationResult(True, (), decision)


def _validate_memory(value: Any, path: str, issues: list[ValidationIssue]) -> Memory | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        issues.append(ValidationIssue(path, "memory must be an object or null"))
        return None

    content = _required_string(value, "content", f"{path}.content", issues)
    memory_type = _required_string(value, "type", f"{path}.type", issues)
    if memory_type and memory_type not in MEMORY_TYPES:
        issues.append(ValidationIssue(f"{path}.type", f"unsupported memory type: {memory_type}"))

    tags_value = value.get("tags", [])
    tags: tuple[str, ...] = ()
    if not isinstance(tags_value, list) or not all(isinstance(item, str) and item.strip() for item in tags_value):
        issues.append(ValidationIssue(f"{path}.tags", "tags must be a list of non-empty strings"))
    else:
        tags = tuple(item for item in tags_value)

    return Memory(
        content=content,
        type=memory_type,
        strength=_optional_number(value, "strength", f"{path}.strength", issues),
        decay_rate=_optional_number(value, "decay_rate", f"{path}.decay_rate", issues),
        emotional_weight=_optional_number(value, "emotional_weight", f"{path}.emotional_weight", issues),
        confidence=_optional_number(value, "confidence", f"{path}.confidence", issues),
        tags=tags,
        temporal_expression=_optional_string(value, "temporal_expression", f"{path}.temporal_expression", issues),
        resolved_time=_optional_string(value, "resolved_time", f"{path}.resolved_time", issues),
        resolved_time_confidence=_optional_number(
            value, "resolved_time_confidence", f"{path}.resolved_time_confidence", issues
        ),
    )


def _validate_facts(value: Any, path: str, issues: list[ValidationIssue]) -> list[Fact]:
    if not isinstance(value, list):
        issues.append(ValidationIssue(path, "facts must be a list"))
        return []

    facts: list[Fact] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict):
            issues.append(ValidationIssue(item_path, "fact must be an object"))
            continue

        subject = _required_string(item, "subject", f"{item_path}.subject", issues)
        predicate = _required_string(item, "predicate", f"{item_path}.predicate", issues)
        if predicate and not _PREDICATE_RE.fullmatch(predicate):
            issues.append(ValidationIssue(f"{item_path}.predicate", "predicate must be snake_case"))

        has_value = "value" in item
        if not has_value:
            issues.append(ValidationIssue(f"{item_path}.value", "fact requires value"))

        inference_kind = _required_string(item, "inference_kind", f"{item_path}.inference_kind", issues)
        if inference_kind and inference_kind not in INFERENCE_KINDS:
            issues.append(ValidationIssue(f"{item_path}.inference_kind", "only explicit facts are accepted"))

        evidence_text = _required_string(item, "evidence_text", f"{item_path}.evidence_text", issues)
        if evidence_text and _is_invalid_evidence(evidence_text):
            issues.append(ValidationIssue(f"{item_path}.evidence_text", "evidence text is not from the current turn"))

        facts.append(
            Fact(
                subject=subject,
                predicate=predicate,
                value=item.get("value"),
                confidence=_optional_number(item, "confidence", f"{item_path}.confidence", issues),
                inference_kind=inference_kind,
                evidence_text=evidence_text,
                object=_optional_string(item, "object", f"{item_path}.object", issues),
                value_text=_optional_string(item, "value_text", f"{item_path}.value_text", issues),
                value_json=item.get("value_json"),
                fact_type=_optional_string(item, "fact_type", f"{item_path}.fact_type", issues),
                temporal_expression=_optional_string(item, "temporal_expression", f"{item_path}.temporal_expression", issues),
                resolved_time=_optional_string(item, "resolved_time", f"{item_path}.resolved_time", issues),
                resolved_time_confidence=_optional_number(
                    item, "resolved_time_confidence", f"{item_path}.resolved_time_confidence", issues
                ),
            )
        )
    return facts


def _required_string(value: dict[str, Any], key: str, path: str, issues: list[ValidationIssue]) -> str:
    raw = value.get(key)
    if isinstance(raw, str) and raw.strip():
        return raw
    issues.append(ValidationIssue(path, "required non-empty string"))
    return ""


def _optional_string(value: dict[str, Any], key: str, path: str, issues: list[ValidationIssue]) -> str | None:
    if key not in value or value[key] is None:
        return None
    raw = value[key]
    if isinstance(raw, str) and raw.strip():
        return raw
    issues.append(ValidationIssue(path, "must be a non-empty string when present"))
    return None


def _optional_number(value: dict[str, Any], key: str, path: str, issues: list[ValidationIssue]) -> float | None:
    if key not in value or value[key] is None:
        return None
    raw = value[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        issues.append(ValidationIssue(path, "must be a finite number when present"))
        return None
    number = float(raw)
    if number != number or number in (float("inf"), float("-inf")):
        issues.append(ValidationIssue(path, "must be a finite number when present"))
        return None
    return number


def _is_invalid_evidence(value: str) -> bool:
    normalized = value.lower()
    return any(fragment in normalized for fragment in _INVALID_EVIDENCE_SUBSTRINGS)


def _fail(path: str, message: str) -> ValidationResult:
    return ValidationResult(False, (ValidationIssue(path, message),))

