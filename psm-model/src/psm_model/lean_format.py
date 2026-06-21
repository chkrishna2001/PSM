from __future__ import annotations

import json
from typing import Any

from psm_model.schema import ACTIONS, MEMORY_TYPES, ValidationIssue, validate_storage_decision


def encode_at_tag_decision(decision: dict[str, Any]) -> str:
    result = validate_storage_decision(decision)
    if not result.ok:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        raise ValueError(f"cannot encode invalid decision: {formatted}")

    lines = [f"@a {decision['action']}"]
    memory = decision.get("memory")
    if memory is None:
        lines.append("@m none")
    else:
        lines.append(f"@t {memory['type']}")
        lines.append(f"@c {memory['content']}")
        _append_optional(lines, "@s", memory.get("strength"))
        _append_optional(lines, "@d", memory.get("decay_rate"))
        _append_optional(lines, "@e", memory.get("emotional_weight"))
        _append_optional(lines, "@p", memory.get("confidence"))
        tags = memory.get("tags") or []
        if tags:
            lines.append("@g " + " ".join(tags))
        if memory.get("temporal_expression"):
            lines.append(f"@te {memory['temporal_expression']}")
        if memory.get("resolved_time"):
            lines.append(f"@rt {memory['resolved_time']}")

    for fact in decision.get("facts") or []:
        lines.extend(
            [
                "@f",
                f"sub={fact['subject']}",
                f"pred={fact['predicate']}",
                f"val={fact['value']}",
                f"conf={_number_text(fact.get('confidence'))}",
                f"kind={fact.get('inference_kind') or 'explicit'}",
                f"ev={fact['evidence_text']}",
            ]
        )
        if fact.get("temporal_expression"):
            lines.append(f"fte={fact['temporal_expression']}")
        if fact.get("resolved_time"):
            lines.append(f"frt={fact['resolved_time']}")
        lines.append("@ef")
    lines.append(f"@r {decision['reasoning']}")
    lines.append("@end")
    return "\n".join(lines)


def parse_at_tag_decision(text: str) -> tuple[dict[str, Any] | None, tuple[ValidationIssue, ...]]:
    issues: list[ValidationIssue] = []
    action: str | None = None
    memory: dict[str, Any] | None = {}
    facts: list[dict[str, Any]] = []
    current_fact: dict[str, Any] | None = None
    reasoning: str | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line == "@end":
            break
        if line == "@f":
            if current_fact is not None:
                issues.append(ValidationIssue(f"$.line[{line_number}]", "nested fact block"))
            current_fact = {}
            continue
        if line == "@ef":
            if current_fact is None:
                issues.append(ValidationIssue(f"$.line[{line_number}]", "fact block close without open"))
            else:
                facts.append(current_fact)
                current_fact = None
            continue

        if current_fact is not None:
            key, sep, value = line.partition("=")
            if not sep:
                issues.append(ValidationIssue(f"$.line[{line_number}]", "fact field missing '=' separator"))
                continue
            if key == "sub":
                current_fact["subject"] = value
            elif key == "pred":
                current_fact["predicate"] = value
            elif key == "val":
                current_fact["value"] = value
            elif key == "conf":
                current_fact["confidence"] = _parse_float(value, f"$.facts[{len(facts)}].confidence", issues) if value else None
            elif key == "kind":
                current_fact["inference_kind"] = value
            elif key == "ev":
                current_fact["evidence_text"] = value
            elif key == "fte":
                current_fact["temporal_expression"] = value
            elif key == "frt":
                current_fact["resolved_time"] = value
            else:
                issues.append(ValidationIssue(f"$.line[{line_number}]", f"unknown fact field: {key}"))
            continue

        tag, _, value = line.partition(" ")
        if tag == "@a":
            action = value
            if action not in ACTIONS:
                issues.append(ValidationIssue("$.action", f"unsupported action: {action}"))
        elif tag == "@m" and value == "none":
            memory = None
        elif tag == "@t":
            memory = _ensure_memory(memory)
            memory["type"] = value
        elif tag == "@c":
            memory = _ensure_memory(memory)
            memory["content"] = value
        elif tag == "@s":
            memory = _ensure_memory(memory)
            memory["strength"] = _parse_float(value, "$.memory.strength", issues)
        elif tag == "@d":
            memory = _ensure_memory(memory)
            memory["decay_rate"] = _parse_float(value, "$.memory.decay_rate", issues)
        elif tag == "@e":
            memory = _ensure_memory(memory)
            memory["emotional_weight"] = _parse_float(value, "$.memory.emotional_weight", issues)
        elif tag == "@p":
            memory = _ensure_memory(memory)
            memory["confidence"] = _parse_float(value, "$.memory.confidence", issues)
        elif tag == "@g":
            memory = _ensure_memory(memory)
            memory["tags"] = [item for item in value.split(" ") if item]
        elif tag == "@te":
            memory = _ensure_memory(memory)
            memory["temporal_expression"] = value
        elif tag == "@rt":
            memory = _ensure_memory(memory)
            memory["resolved_time"] = value
        elif tag == "@r":
            reasoning = value
        else:
            issues.append(ValidationIssue(f"$.line[{line_number}]", f"unknown tag: {tag}"))

    if current_fact is not None:
        issues.append(ValidationIssue("$.facts", "unclosed fact block"))

    decision = {"action": action, "memory": memory, "facts": facts, "reasoning": reasoning}
    result = validate_storage_decision(decision)
    issues.extend(result.issues)
    return (decision if not issues else None), tuple(issues)


def encode_tagged_decision(decision: dict[str, Any]) -> str:
    result = validate_storage_decision(decision)
    if not result.ok:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        raise ValueError(f"cannot encode invalid decision: {formatted}")

    lines = [f"A:{decision['action']}"]
    memory = decision.get("memory")
    if memory is None:
        lines.append("M:-")
    else:
        lines.append(f"T:{memory['type']}")
        lines.append(f"C:{_escape(memory['content'])}")
        lines.append(
            "Q:"
            + ",".join(
                _number_text(memory.get(key))
                for key in ("strength", "decay_rate", "emotional_weight", "confidence")
            )
        )
        tags = memory.get("tags") or []
        if tags:
            lines.append("G:" + ",".join(_escape(tag) for tag in tags))
        if memory.get("temporal_expression"):
            lines.append(f"TE:{_escape(memory['temporal_expression'])}")
        if memory.get("resolved_time"):
            lines.append(f"RT:{_escape(memory['resolved_time'])}")

    for fact in decision.get("facts") or []:
        lines.append(
            "F:"
            + "|".join(
                [
                    _escape(str(fact["subject"])),
                    _escape(str(fact["predicate"])),
                    _escape(str(fact["value"])),
                    _number_text(fact.get("confidence")),
                    _escape(str(fact.get("inference_kind") or "explicit")),
                    _escape(str(fact["evidence_text"])),
                ]
            )
        )
    for indexable in decision.get("indexables") or []:
        lines.append(_encode_indexable_line(indexable))
    lines.append(f"R:{_escape(decision['reasoning'])}")
    lines.append("END")
    return "\n".join(lines)


def parse_tagged_decision(text: str) -> tuple[dict[str, Any] | None, tuple[ValidationIssue, ...]]:
    issues: list[ValidationIssue] = []
    action: str | None = None
    memory: dict[str, Any] | None = {}
    facts: list[dict[str, Any]] = []
    indexables: list[dict[str, Any]] = []
    reasoning: str | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line == "END":
            break
        key, sep, value = line.partition(":")
        if not sep:
            issues.append(ValidationIssue(f"$.line[{line_number}]", "missing ':' separator"))
            continue
        if key == "A":
            action = value.strip()
            if action not in ACTIONS:
                issues.append(ValidationIssue("$.action", f"unsupported action: {action}"))
        elif key == "M" and value.strip() == "-":
            memory = None
        elif key == "T":
            mem_type = value.strip()
            if mem_type not in MEMORY_TYPES:
                issues.append(ValidationIssue("$.memory.type", f"unsupported memory type: {mem_type}"))
            memory = _ensure_memory(memory)
            memory["type"] = mem_type
        elif key == "C":
            memory = _ensure_memory(memory)
            memory["content"] = _unescape(value)
        elif key == "Q":
            memory = _ensure_memory(memory)
            parts = value.split(",")
            for field, raw in zip(("strength", "decay_rate", "emotional_weight", "confidence"), parts):
                if raw:
                    memory[field] = _parse_float(raw, f"$.memory.{field}", issues)
        elif key == "G":
            memory = _ensure_memory(memory)
            memory["tags"] = [_unescape(item) for item in value.split(",") if item]
        elif key == "TE":
            memory = _ensure_memory(memory)
            memory["temporal_expression"] = _unescape(value)
        elif key == "RT":
            memory = _ensure_memory(memory)
            memory["resolved_time"] = _unescape(value)
        elif key == "F":
            facts.append(_parse_fact(value, len(facts), issues))
        elif key == "X":
            indexables.append(_parse_indexable(value, len(indexables), issues))
        elif key == "R":
            reasoning = _unescape(value)
        else:
            issues.append(ValidationIssue(f"$.line[{line_number}]", f"unknown tag: {key}"))

    decision = {"action": action, "memory": memory, "facts": facts, "indexables": indexables, "reasoning": reasoning}
    result = validate_storage_decision(decision)
    issues.extend(result.issues)
    return (decision if not issues else None), tuple(issues)


def compact_json_array(decision: dict[str, Any]) -> str:
    result = validate_storage_decision(decision)
    if not result.ok:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        raise ValueError(f"cannot encode invalid decision: {formatted}")
    memory = decision.get("memory")
    memory_array = None
    if memory is not None:
        memory_array = [
            memory["type"],
            memory["content"],
            memory.get("strength"),
            memory.get("decay_rate"),
            memory.get("emotional_weight"),
            memory.get("confidence"),
            memory.get("tags", []),
            memory.get("temporal_expression"),
            memory.get("resolved_time"),
        ]
    facts = [
        [
            fact["subject"],
            fact["predicate"],
            fact["value"],
            fact.get("confidence"),
            fact.get("inference_kind", "explicit"),
            fact["evidence_text"],
            fact.get("temporal_expression"),
            fact.get("resolved_time"),
        ]
        for fact in decision.get("facts", [])
    ]
    return json.dumps([decision["action"], memory_array, facts, decision["reasoning"]], ensure_ascii=False, separators=(",", ":"))


def _encode_indexable_line(indexable: dict[str, Any]) -> str:
    parts = [
        _escape(str(indexable["kind"])),
        _escape(str(indexable["key"])),
        _number_text(indexable.get("salience")),
        _escape(str(indexable.get("reconstructive_hint") or "")),
        _escape(str(indexable.get("evidence_text") or "")),
    ]
    steps = indexable.get("steps") or []
    if steps:
        parts.append(",".join(_escape(str(step)) for step in steps))
    return "X:" + "|".join(parts)


def _parse_indexable(value: str, index: int, issues: list[ValidationIssue]) -> dict[str, Any]:
    parts = value.split("|")
    if len(parts) < 5:
        issues.append(ValidationIssue(f"$.indexables[{index}]", "indexable line must have at least 5 pipe-delimited fields"))
        return {}
    if len(parts) > 6:
        parts = parts[:5] + ["|".join(parts[5:])]
    kind, key, salience, hint, evidence = (_unescape(part) for part in parts[:5])
    row: dict[str, Any] = {
        "kind": kind,
        "key": key,
        "salience": _parse_float(salience, f"$.indexables[{index}].salience", issues) if salience else None,
        "reconstructive_hint": hint or None,
        "evidence_text": evidence or None,
    }
    if len(parts) == 6 and parts[5]:
        row["steps"] = [_unescape(step) for step in parts[5].split(",") if step]
    return row


def _ensure_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    if memory is None:
        return {}
    return memory


def _parse_fact(value: str, index: int, issues: list[ValidationIssue]) -> dict[str, Any]:
    parts = value.split("|")
    if len(parts) < 6:
        issues.append(ValidationIssue(f"$.facts[{index}]", "fact line must have 6 pipe-delimited fields"))
        return {}
    if len(parts) > 6:
        parts = parts[:5] + ["|".join(parts[5:])]
    subject, predicate, fact_value, confidence, inference_kind, evidence_text = (_unescape(part) for part in parts)
    return {
        "subject": subject,
        "predicate": predicate,
        "value": fact_value,
        "confidence": _parse_float(confidence, f"$.facts[{index}].confidence", issues) if confidence else None,
        "inference_kind": inference_kind,
        "evidence_text": evidence_text,
    }


def _parse_float(value: str, path: str, issues: list[ValidationIssue]) -> float | None:
    try:
        return float(value)
    except ValueError:
        issues.append(ValidationIssue(path, "must be a number"))
        return None


def _number_text(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):g}"


def _append_optional(lines: list[str], tag: str, value: Any) -> None:
    if value is not None:
        lines.append(f"{tag} {_number_text(value)}")


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("|", "\\p").replace(",", "\\c")


def _unescape(value: str) -> str:
    output: list[str] = []
    escaping = False
    for char in value:
        if escaping:
            output.append({"n": "\n", "p": "|", "c": ","}.get(char, char))
            escaping = False
        elif char == "\\":
            escaping = True
        else:
            output.append(char)
    if escaping:
        output.append("\\")
    return "".join(output)


def encode_minimal_decision(decision: dict[str, Any]) -> str:
    action = str(decision.get("action") or "ignore").strip().lower()
    if action in {"ignore", "ignore_noise"}:
        return "ignore"
    memory = decision.get("memory")
    content = ""
    if isinstance(memory, dict):
        content = str(memory.get("content") or "").strip()
    if not content:
        raise ValueError("minimal store decision requires memory.content")
    return f"store: {content}"


def encode_binary_decision(decision: dict[str, Any]) -> str:
    action = str(decision.get("action") or "ignore").strip().lower()
    if action in {"ignore", "ignore_noise"}:
        return "ignore"
    return "store"


def parse_binary_decision(text: str) -> tuple[dict[str, Any] | None, tuple[ValidationIssue, ...]]:
    issues: list[ValidationIssue] = []
    line = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped:
            line = stripped.lower()
            break
    if not line:
        issues.append(ValidationIssue("$.output", "empty binary output"))
        return None, tuple(issues)
    if line == "ignore":
        return {
            "action": "ignore",
            "memory": None,
            "facts": [],
            "indexables": [],
            "reasoning": "No durable memory.",
        }, ()
    if line == "store":
        return {
            "action": "store_episodic",
            "memory": None,
            "facts": [],
            "indexables": [],
            "reasoning": "Classify store.",
        }, ()
    issues.append(ValidationIssue("$.output", f"unsupported binary line: {line[:80]}"))
    return None, tuple(issues)


def parse_minimal_decision(text: str) -> tuple[dict[str, Any] | None, tuple[ValidationIssue, ...]]:
    issues: list[ValidationIssue] = []
    line = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped:
            line = stripped
            break
    if not line:
        issues.append(ValidationIssue("$.output", "empty minimal output"))
        return None, tuple(issues)
    if line == "ignore":
        return {
            "action": "ignore",
            "memory": None,
            "facts": [],
            "indexables": [],
            "reasoning": "No durable memory.",
        }, ()
    if line.lower().startswith("store:"):
        content = line.split(":", 1)[1].strip()
        if not content:
            issues.append(ValidationIssue("$.memory.content", "store line missing content"))
            return None, tuple(issues)
        return {
            "action": "store_episodic",
            "memory": {
                "content": content,
                "type": "episodic",
                "strength": 0.86,
                "decay_rate": 0.02,
                "emotional_weight": 0.22,
                "confidence": 0.92,
                "tags": [],
            },
            "facts": [],
            "indexables": [],
            "reasoning": content,
        }, ()
    issues.append(ValidationIssue("$.output", f"unsupported minimal line: {line[:80]}"))
    return None, tuple(issues)
