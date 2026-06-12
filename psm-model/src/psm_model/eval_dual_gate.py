from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.eval_checkpoint import evaluate_checkpoint
from psm_model.eval_recall import evaluate_recall_checkpoint
from psm_model.gates import EXPANDED_PROBE_THRESHOLDS, RECALL_PROBE_THRESHOLDS, gate_report


def evaluate_dual_gate(
    checkpoint: Path,
    *,
    storage_probe: Path,
    recall_probe: Path,
    output_format: str = "tagged",
    storage_gate_mode: str = "expanded",
    device: str = "auto",
) -> dict[str, object]:
    storage_report = evaluate_checkpoint(
        checkpoint,
        storage_probe,
        output_format=output_format,
        device=device,
        gate_mode=storage_gate_mode,
    )
    recall_report = evaluate_recall_checkpoint(
        checkpoint,
        recall_probe,
        device=device,
    )
    storage_gate = gate_report(storage_report, EXPANDED_PROBE_THRESHOLDS)
    recall_gate = gate_report(recall_report, RECALL_PROBE_THRESHOLDS)
    return {
        "checkpoint": str(checkpoint),
        "passed": storage_gate["passed"] and recall_gate["passed"],
        "storage": {
            "data": str(storage_probe),
            "gate_mode": storage_gate_mode,
            "metrics": {key: storage_report.get(key) for key in EXPANDED_PROBE_THRESHOLDS},
            "gate": storage_gate,
        },
        "recall": {
            "data": str(recall_probe),
            "metrics": {key: recall_report.get(key) for key in RECALL_PROBE_THRESHOLDS},
            "gate": recall_gate,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Gate 4 storage + Gate 5 recall eval; fail if either regresses."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--storage-probe",
        type=Path,
        default=Path("psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl"),
    )
    parser.add_argument(
        "--recall-probe",
        type=Path,
        default=Path("psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl"),
    )
    parser.add_argument("--output-format", default="tagged", choices=["json", "tagged", "at_tag"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate_dual_gate(
        args.checkpoint,
        storage_probe=args.storage_probe,
        recall_probe=args.recall_probe,
        output_format=args.output_format,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
