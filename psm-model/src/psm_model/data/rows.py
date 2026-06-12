from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psm_model.prompts import row_task
from psm_model.recall_schema import validate_recall_plan
from psm_model.schema import ValidationIssue, validate_storage_decision


@dataclass(frozen=True)
class TrainingRow:
    id: str
    task: str
    input: dict[str, Any]
    expected: dict[str, Any]
    source: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class DatasetGateReport:
    total: int
    valid: int
    failures: tuple[dict[str, Any], ...]
    action_counts: dict[str, int]
    memory_type_counts: dict[str, int]
    task_counts: dict[str, int]
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
            "task_counts": self.task_counts,
            "duplicate_ids": list(self.duplicate_ids),
        }


def infer_row_task(value: dict[str, Any]) -> str:
    explicit = value.get("task")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    row_input = value.get("input")
    if isinstance(row_input, dict):
        return row_task(row_input)
    return "storage"


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
                "input must include conversation, question, user_prompt, source, context, or prompt",
            )
        )

    task = infer_row_task(value)
    expected_raw = value.get("expected")
    expected: dict[str, Any] | None = None
    if task == "storage":
        expected_result = validate_storage_decision(expected_raw)
        issues.extend(ValidationIssue(f"$.expected{issue.path[1:]}", issue.message) for issue in expected_result.issues)
        if expected_result.ok and isinstance(expected_raw, dict):
            expected = expected_raw
    else:
        recall_result = validate_recall_plan(expected_raw)
        issues.extend(ValidationIssue(f"$.expected", issue) for issue in recall_result.issues)
        if isinstance(expected_raw, dict) and recall_result.ok:
            expected = expected_raw

    source = _optional_string(value, "source", "$.source", issues)
    split = _optional_string(value, "split", "$.split", issues)
    if split and split not in {"train", "validation", "test", "probe"}:
        issues.append(ValidationIssue("$.split", "split must be train, validation, test, or probe"))

    if issues or expected is None:
        return None, tuple(issues)

    return TrainingRow(
        id=row_id,
        task=task,
        input=row_input,
        expected=expected,
        source=source,
        split=split,
    ), ()


def load_jsonl_rows(path: Path) -> DatasetGateReport:
    total = 0
    valid = 0
    failures: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    memory_type_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
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
            task_counts[row.task] += 1
            if row.task == "storage":
                action_counts[str(row.expected.get("action", "unknown"))] += 1
                memory = row.expected.get("memory")
                memory_type = memory.get("type") if isinstance(memory, dict) else "none"
                memory_type_counts[str(memory_type)] += 1

    return DatasetGateReport(
        total=total,
        valid=valid,
        failures=tuple(failures),
        action_counts=dict(sorted(action_counts.items())),
        memory_type_counts=dict(sorted(memory_type_counts.items())),
        task_counts=dict(sorted(task_counts.items())),
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
    if isinstance(value.get("conversation"), list) and value["conversation"]:
        return True
    return any(
        isinstance(value.get(key), str) and value[key].strip()
        for key in ("conversation", "source", "context", "prompt", "question", "user_prompt")
    )
