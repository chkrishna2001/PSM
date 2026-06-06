from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psm_model.schema import StorageDecision, ValidationIssue, validate_storage_decision


@dataclass(frozen=True)
class TrainingRow:
    id: str
    input: dict[str, Any]
    expected: StorageDecision
    source: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class DatasetGateReport:
    total: int
    valid: int
    failures: tuple[dict[str, Any], ...]
    action_counts: dict[str, int]
    memory_type_counts: dict[str, int]
    duplicate_ids: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures and not self.duplicate_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total": self.total,
            "valid": self.valid,
            "valid_rate": self.valid / self.total if self.total else 0.0,
            "failures": list(self.failures),
            "action_counts": self.action_counts,
            "memory_type_counts": self.memory_type_counts,
            "duplicate_ids": list(self.duplicate_ids),
        }


def validate_training_row(value: Any) -> tuple[TrainingRow | None, tuple[ValidationIssue, ...]]:
    issues: list[ValidationIssue] = []
    if not isinstance(value, dict):
        return None, (ValidationIssue("$", "training row must be a JSON object"),)

    row_id = _required_string(value, "id", "$.id", issues)
    row_input = value.get("input")
    if not isinstance(row_input, dict):
        issues.append(ValidationIssue("$.input", "input must be an object"))
        row_input = {}
    elif not _has_prompt_payload(row_input):
        issues.append(
            ValidationIssue(
                "$.input",
                "input must include at least one of conversation, source, context, or prompt",
            )
        )

    expected_result = validate_storage_decision(value.get("expected"))
    issues.extend(ValidationIssue(f"$.expected{issue.path[1:]}", issue.message) for issue in expected_result.issues)

    source = _optional_string(value, "source", "$.source", issues)
    split = _optional_string(value, "split", "$.split", issues)
    if split and split not in {"train", "validation", "test", "probe"}:
        issues.append(ValidationIssue("$.split", "split must be train, validation, test, or probe"))

    if issues:
        return None, tuple(issues)

    assert expected_result.decision is not None
    return TrainingRow(
        id=row_id,
        input=row_input,
        expected=expected_result.decision,
        source=source,
        split=split,
    ), ()


def load_jsonl_rows(path: Path) -> DatasetGateReport:
    total = 0
    valid = 0
    failures: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    memory_type_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {
                        "line": line_number,
                        "id": None,
                        "issues": [{"path": "$", "message": f"invalid JSON: {exc.msg}"}],
                    }
                )
                continue

            row, issues = validate_training_row(raw)
            row_id = raw.get("id") if isinstance(raw, dict) else None
            if isinstance(row_id, str):
                if row_id in seen_ids:
                    duplicate_ids.add(row_id)
                seen_ids.add(row_id)

            if issues:
                failures.append(
                    {
                        "line": line_number,
                        "id": row_id,
                        "issues": [{"path": issue.path, "message": issue.message} for issue in issues],
                    }
                )
                continue

            assert row is not None
            valid += 1
            action_counts[row.expected.action] += 1
            memory_type_counts[row.expected.memory.type if row.expected.memory else "none"] += 1

    return DatasetGateReport(
        total=total,
        valid=valid,
        failures=tuple(failures),
        action_counts=dict(sorted(action_counts.items())),
        memory_type_counts=dict(sorted(memory_type_counts.items())),
        duplicate_ids=tuple(sorted(duplicate_ids)),
    )


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


def _has_prompt_payload(value: dict[str, Any]) -> bool:
    return any(isinstance(value.get(key), str) and value[key].strip() for key in ("conversation", "source", "context", "prompt"))

