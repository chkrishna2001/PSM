from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DIRECT_PROBE_THRESHOLDS = {
    "parse_valid_rate": 1.0,
    "schema_valid_rate": 1.0,
    "action_accuracy": 1.0,
    "memory_type_accuracy": 1.0,
    "memory_content_exact_rate": 1.0,
    "fact_count_accuracy": 1.0,
    "facts_exact_rate": 1.0,
}


@dataclass(frozen=True)
class GateFailure:
    metric: str
    actual: float
    required: float


def evaluate_thresholds(report: dict[str, Any], thresholds: dict[str, float] | None = None) -> list[GateFailure]:
    active_thresholds = thresholds or DIRECT_PROBE_THRESHOLDS
    failures: list[GateFailure] = []
    for metric, required in active_thresholds.items():
        actual = float(report.get(metric, 0.0))
        if actual < required:
            failures.append(GateFailure(metric=metric, actual=actual, required=required))
    return failures


def gate_report(report: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    failures = evaluate_thresholds(report, thresholds)
    return {
        "passed": not failures,
        "thresholds": dict(thresholds or DIRECT_PROBE_THRESHOLDS),
        "failures": [
            {"metric": failure.metric, "actual": failure.actual, "required": failure.required}
            for failure in failures
        ],
    }
