from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from psm_model.gates import gate_report, thresholds_for_gate_mode


def classify_row(row: dict[str, Any]) -> str:
    if row.get("skipped"):
        return "context_overflow"
    if not row.get("parse_valid"):
        return "parse_fail"
    if not row.get("schema_valid"):
        return "schema_fail"
    if row.get("predicted_action") != row.get("expected_action"):
        return "wrong_action"
    if row.get("predicted_memory_type") != row.get("expected_memory_type"):
        return "wrong_memory_type"
    if not row.get("memory_content_exact"):
        return "wrong_memory_content"
    if row.get("predicted_fact_count") != row.get("expected_fact_count"):
        return "wrong_fact_count"
    if not row.get("facts_exact"):
        return "wrong_facts"
    return "pass"


def analyze_eval_report(
    report: dict[str, Any],
    *,
    gate_mode: str = "expanded",
) -> dict[str, Any]:
    rows = report.get("reports")
    if not isinstance(rows, list):
        raise ValueError("report missing 'reports' list — pass full eval_checkpoint JSON output")

    thresholds = thresholds_for_gate_mode(gate_mode)
    bucket_counts: Counter[str] = Counter()
    by_expected_action: dict[str, Counter[str]] = defaultdict(Counter)
    wrong_action_pairs: Counter[tuple[str, str]] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if not isinstance(row, dict):
            continue
        bucket = classify_row(row)
        bucket_counts[bucket] += 1
        expected_action = str(row.get("expected_action") or "unknown")
        by_expected_action[expected_action][bucket] += 1
        if bucket == "wrong_action":
            predicted = str(row.get("predicted_action") or "null")
            wrong_action_pairs[(expected_action, predicted)] += 1
        if bucket != "pass" and len(examples[bucket]) < 8:
            examples[bucket].append(
                {
                    "id": row.get("id"),
                    "expected_action": row.get("expected_action"),
                    "predicted_action": row.get("predicted_action"),
                    "action_head_prediction": row.get("action_head_prediction"),
                    "issues": row.get("issues", [])[:3],
                }
            )

    aggregate = {key: report.get(key) for key in thresholds if key in report}
    gate = gate_report(aggregate, thresholds)

    evaluated = sum(1 for row in rows if isinstance(row, dict) and not row.get("skipped"))
    passed_rows = bucket_counts.get("pass", 0)

    return {
        "gate_mode": gate_mode,
        "checkpoint": report.get("checkpoint"),
        "data": report.get("data"),
        "device": report.get("device"),
        "examples_total": len(rows),
        "examples_evaluated": evaluated,
        "examples_skipped": bucket_counts.get("context_overflow", 0),
        "row_pass_rate": passed_rows / evaluated if evaluated else 0.0,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "by_expected_action": {
            action: dict(sorted(counts.items())) for action, counts in sorted(by_expected_action.items())
        },
        "top_wrong_action_pairs": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in wrong_action_pairs.most_common(15)
        ],
        "sample_failures": dict(examples),
        "aggregate_metrics": aggregate,
        "gate": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bucket failures from a psm_model.eval_checkpoint JSON report (Gate 4 diagnostics)."
    )
    parser.add_argument("report", type=Path, help="Path to eval_checkpoint JSON output")
    parser.add_argument(
        "--gate-mode",
        default="expanded",
        choices=["direct", "expanded"],
        help="Threshold set for aggregate gate (default: expanded)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write analysis JSON",
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    analysis = analyze_eval_report(report, gate_mode=args.gate_mode)
    payload = json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    print(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    return 0 if analysis["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
