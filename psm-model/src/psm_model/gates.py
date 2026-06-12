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

# Gate 4 — full StorageDecision on expanded probe (budget-filtered). Aligned with Gate 2 action bar.
EXPANDED_PROBE_THRESHOLDS = {
    "parse_valid_rate": 0.95,
    "schema_valid_rate": 0.95,
    "action_accuracy": 0.85,
    "memory_type_accuracy": 0.70,
    "memory_content_exact_rate": 0.50,
    "fact_count_accuracy": 0.70,
    "facts_exact_rate": 0.50,
}

# Gate 5 — recall / context planning (JSON RecallPlan on recall probes).
RECALL_PROBE_THRESHOLDS = {
    "parse_valid_rate": 0.95,
    "schema_valid_rate": 0.95,
    "target_tables_exact_rate": 0.90,
    "target_tables_primary_rate": 0.95,
    "ranking_hints_score": 0.50,
    "top_k_exact_rate": 0.90,
}

GATE_MODE_THRESHOLDS = {
    "direct": DIRECT_PROBE_THRESHOLDS,
    "expanded": EXPANDED_PROBE_THRESHOLDS,
    "recall": RECALL_PROBE_THRESHOLDS,
}


def thresholds_for_gate_mode(mode: str) -> dict[str, float]:
    try:
        return dict(GATE_MODE_THRESHOLDS[mode])
    except KeyError as exc:
        supported = ", ".join(sorted(GATE_MODE_THRESHOLDS))
        raise ValueError(f"unsupported gate mode {mode!r}; expected one of: {supported}") from exc


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
