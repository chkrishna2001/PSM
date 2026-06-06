from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psm_model.action_diagnostics import evaluate_action_head, evaluate_action_prefixes
from psm_model.probe_checkpoint import _load_probe_file, probe_checkpoint


DEFAULT_FOUNDATION_PROBE = Path("psm-model/data/action-foundation-v1/action-probe.jsonl")
DEFAULT_CONCEPT_PROBE = Path("psm-model/data/concept-curriculum-v1/action-probe.jsonl")
DEFAULT_FAST_MIXED_PROBE = Path("psm-model/data/fast-mixed-10k-ctx2048/action-probe-20.jsonl")
DEFAULT_MANUAL_PROBE = Path("psm-model/data/direct-behavior-v1/manual-probe.jsonl")
DEFAULT_EXPANDED_PROBE = Path("psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl")

PHASE1_ACTION_THRESHOLDS = {
    "expanded_macro_action_prefix_accuracy": 0.85,
    "manual_macro_action_prefix_accuracy": 0.80,
}
PHASE1_MIN_DISTINCT_ACTIONS = 4


DEFAULT_THRESHOLDS = {
    "foundation_macro_action_prefix_accuracy": 0.90,
    "concept_macro_action_prefix_accuracy": 0.85,
    "fast_mixed_macro_action_prefix_accuracy": 0.70,
    "manual_safe_expected_action_accuracy": 1.00,
    "manual_safe_valid_rate": 1.00,
    "manual_model_action_accuracy": 0.70,
}

PRODUCT_SAFE_THRESHOLDS = {
    "manual_safe_expected_action_accuracy": 1.00,
    "manual_safe_valid_rate": 1.00,
}


@dataclass(frozen=True)
class GateCheck:
    metric: str
    actual: float | None
    required: float

    @property
    def passed(self) -> bool:
        return self.actual is not None and self.actual >= self.required


def evaluate_checkpoint_gate(
    checkpoint: Path,
    *,
    foundation_probe: Path = DEFAULT_FOUNDATION_PROBE,
    concept_probe: Path = DEFAULT_CONCEPT_PROBE,
    fast_mixed_probe: Path = DEFAULT_FAST_MIXED_PROBE,
    manual_probe: Path = DEFAULT_MANUAL_PROBE,
    thresholds: dict[str, float] | None = None,
    mode: str = "diagnostic",
    output_format: str | None = None,
    device: str = "cpu",
    action_classifier: Path | None = None,
) -> dict[str, Any]:
    active_thresholds = thresholds or thresholds_for_mode(mode)
    foundation = evaluate_action_prefixes(checkpoint, foundation_probe, output_format=output_format, device=device)
    concept = evaluate_action_prefixes(checkpoint, concept_probe, output_format=output_format, device=device)
    fast_mixed = evaluate_action_prefixes(checkpoint, fast_mixed_probe, output_format=output_format, device=device)
    manual_head = evaluate_action_head(checkpoint, manual_probe, output_format=output_format, device=device)
    manual_safe = probe_checkpoint(
        checkpoint,
        _load_probe_file(manual_probe),
        output_format=output_format,
        device=device,
        safe=True,
        action_classifier=action_classifier,
    )
    metrics = {
        "foundation_macro_action_prefix_accuracy": foundation["macro_action_prefix_accuracy"],
        "concept_macro_action_prefix_accuracy": concept["macro_action_prefix_accuracy"],
        "fast_mixed_macro_action_prefix_accuracy": fast_mixed["macro_action_prefix_accuracy"],
        "manual_safe_expected_action_accuracy": manual_safe["expected_action_accuracy"],
        "manual_safe_valid_rate": manual_safe["valid_rate"],
        "manual_model_action_accuracy": manual_safe["model_action_accuracy"],
        "manual_action_head_accuracy": manual_head["action_head_accuracy"],
        "manual_macro_action_head_accuracy": manual_head["macro_action_head_accuracy"],
    }
    checks = evaluate_gate_metrics(metrics, active_thresholds)
    return {
        "checkpoint": str(checkpoint),
        "action_classifier": str(action_classifier) if action_classifier is not None else None,
        "device": device,
        "mode": mode,
        "passed": all(check.passed for check in checks),
        "thresholds": active_thresholds,
        "metrics": metrics,
        "failures": [
            {"metric": check.metric, "actual": check.actual, "required": check.required}
            for check in checks
            if not check.passed
        ],
        "details": {
            "foundation": compact_action_report(foundation),
            "concept": compact_action_report(concept),
            "fast_mixed": compact_action_report(fast_mixed),
            "manual_action_head": compact_head_report(manual_head),
            "manual_safe": compact_manual_report(manual_safe),
        },
    }


def evaluate_gate_metrics(metrics: dict[str, float | None], thresholds: dict[str, float]) -> list[GateCheck]:
    return [GateCheck(metric=metric, actual=metrics.get(metric), required=required) for metric, required in thresholds.items()]


def thresholds_for_mode(mode: str) -> dict[str, float]:
    if mode == "diagnostic":
        return dict(DEFAULT_THRESHOLDS)
    if mode == "product-safe":
        return dict(PRODUCT_SAFE_THRESHOLDS)
    if mode == "phase1-action":
        return dict(PHASE1_ACTION_THRESHOLDS)
    raise ValueError(f"unsupported gate mode: {mode}")


def evaluate_phase1_action_gate(
    checkpoint: Path,
    *,
    expanded_probe: Path = DEFAULT_EXPANDED_PROBE,
    manual_probe: Path = DEFAULT_MANUAL_PROBE,
    output_format: str | None = "action",
    device: str = "cpu",
    thresholds: dict[str, float] | None = None,
    min_distinct_actions: int = PHASE1_MIN_DISTINCT_ACTIONS,
) -> dict[str, Any]:
    active_thresholds = thresholds or thresholds_for_mode("phase1-action")
    expanded = evaluate_action_prefixes(checkpoint, expanded_probe, output_format=output_format, device=device)
    manual = evaluate_action_prefixes(checkpoint, manual_probe, output_format=output_format, device=device)
    distinct_predicted = sum(1 for count in expanded["predicted_action_counts"].values() if count > 0)
    metrics = {
        "expanded_macro_action_prefix_accuracy": expanded["macro_action_prefix_accuracy"],
        "manual_macro_action_prefix_accuracy": manual["macro_action_prefix_accuracy"],
        "distinct_predicted_actions": float(distinct_predicted),
        "collapse_fraction": expanded["collapse_fraction"],
    }
    checks = evaluate_gate_metrics(metrics, active_thresholds)
    distinct_ok = distinct_predicted >= min_distinct_actions
    return {
        "checkpoint": str(checkpoint),
        "device": device,
        "mode": "phase1-action",
        "passed": all(check.passed for check in checks) and distinct_ok,
        "thresholds": active_thresholds,
        "min_distinct_actions": min_distinct_actions,
        "metrics": metrics,
        "failures": [
            {"metric": check.metric, "actual": check.actual, "required": check.required}
            for check in checks
            if not check.passed
        ]
        + (
            [{"metric": "distinct_predicted_actions", "actual": distinct_predicted, "required": min_distinct_actions}]
            if not distinct_ok
            else []
        ),
        "details": {
            "expanded": compact_action_report(expanded),
            "manual": compact_action_report(manual),
        },
    }


def compact_action_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": report["data"],
        "examples": report["examples"],
        "macro_action_prefix_accuracy": report["macro_action_prefix_accuracy"],
        "per_action_accuracy": report["per_action_accuracy"],
        "predicted_action_counts": report["predicted_action_counts"],
        "failures": [
            {
                "id": row["id"],
                "expected_action": row["expected_action"],
                "predicted_action": row["predicted_action"],
                "gold_rank": row["gold_rank"],
            }
            for row in report["reports"]
            if row["expected_action"] != row["predicted_action"]
        ][:12],
    }


def compact_head_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "data": report["data"],
        "examples": report["examples"],
        "action_head_accuracy": report["action_head_accuracy"],
        "macro_action_head_accuracy": report["macro_action_head_accuracy"],
        "per_action_accuracy": report["per_action_accuracy"],
        "predicted_action_counts": report["predicted_action_counts"],
        "failures": [
            {
                "id": row["id"],
                "expected_action": row["expected_action"],
                "predicted_action": row["predicted_action"],
            }
            for row in report["reports"]
            if row["expected_action"] != row["predicted_action"]
        ][:12],
    }


def compact_manual_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "examples": report["examples"],
        "expected_action_accuracy": report["expected_action_accuracy"],
        "model_action_accuracy": report["model_action_accuracy"],
        "valid_rate": report["valid_rate"],
        "failures": [
            {
                "case": row["case"],
                "expected_action": row["expected_action"],
                "model_action": row.get("model_action"),
                "decoder_action": row.get("decoder_action"),
                "parsed_action": row["parsed_action"],
                "valid": row["valid"],
            }
            for row in report["reports"]
            if row.get("expected_action") != row["parsed_action"] or not row["valid"]
        ],
        "model_action_misses": [
            {
                "case": row["case"],
                "expected_action": row["expected_action"],
                "model_action": row.get("model_action"),
                "decoder_action": row.get("decoder_action"),
                "calibrated_action": row.get("calibrated_action"),
            }
            for row in report["reports"]
            if row.get("expected_action") != row.get("model_action")
        ][:12],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the current PSM 50M checkpoint quality gate.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--foundation-probe", type=Path, default=DEFAULT_FOUNDATION_PROBE)
    parser.add_argument("--concept-probe", type=Path, default=DEFAULT_CONCEPT_PROBE)
    parser.add_argument("--fast-mixed-probe", type=Path, default=DEFAULT_FAST_MIXED_PROBE)
    parser.add_argument("--manual-probe", type=Path, default=DEFAULT_MANUAL_PROBE)
    parser.add_argument("--expanded-probe", type=Path, default=DEFAULT_EXPANDED_PROBE)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"])
    parser.add_argument("--device", default="cpu", help="Evaluation device: cpu, cuda, or auto.")
    parser.add_argument("--action-classifier", type=Path, help="Optional standalone action classifier checkpoint for product-safe manual action selection.")
    parser.add_argument("--foundation-threshold", type=float, default=DEFAULT_THRESHOLDS["foundation_macro_action_prefix_accuracy"])
    parser.add_argument("--concept-threshold", type=float, default=DEFAULT_THRESHOLDS["concept_macro_action_prefix_accuracy"])
    parser.add_argument("--fast-mixed-threshold", type=float, default=DEFAULT_THRESHOLDS["fast_mixed_macro_action_prefix_accuracy"])
    parser.add_argument("--manual-safe-threshold", type=float, default=DEFAULT_THRESHOLDS["manual_safe_expected_action_accuracy"])
    parser.add_argument("--manual-model-threshold", type=float, default=DEFAULT_THRESHOLDS["manual_model_action_accuracy"])
    parser.add_argument(
        "--mode",
        choices=["diagnostic", "product-safe", "phase1-action"],
        default="diagnostic",
        help="diagnostic keeps model-only action thresholds; product-safe gates safe_generate; phase1-action gates action-only Phase 1 promotion.",
    )
    parser.add_argument("--expanded-threshold", type=float, default=PHASE1_ACTION_THRESHOLDS["expanded_macro_action_prefix_accuracy"])
    parser.add_argument("--manual-macro-threshold", type=float, default=PHASE1_ACTION_THRESHOLDS["manual_macro_action_prefix_accuracy"])
    parser.add_argument("--min-distinct-actions", type=int, default=PHASE1_MIN_DISTINCT_ACTIONS)
    args = parser.parse_args()

    if args.mode == "phase1-action":
        report = evaluate_phase1_action_gate(
            args.checkpoint,
            expanded_probe=args.expanded_probe,
            manual_probe=args.manual_probe,
            output_format=args.output_format or "action",
            device=args.device,
            thresholds={
                "expanded_macro_action_prefix_accuracy": args.expanded_threshold,
                "manual_macro_action_prefix_accuracy": args.manual_macro_threshold,
            },
            min_distinct_actions=args.min_distinct_actions,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1

    if args.mode == "diagnostic":
        thresholds = {
            "foundation_macro_action_prefix_accuracy": args.foundation_threshold,
            "concept_macro_action_prefix_accuracy": args.concept_threshold,
            "fast_mixed_macro_action_prefix_accuracy": args.fast_mixed_threshold,
            "manual_safe_expected_action_accuracy": args.manual_safe_threshold,
            "manual_safe_valid_rate": 1.0,
            "manual_model_action_accuracy": args.manual_model_threshold,
        }
    else:
        thresholds = {
            "manual_safe_expected_action_accuracy": args.manual_safe_threshold,
            "manual_safe_valid_rate": 1.0,
        }
    report = evaluate_checkpoint_gate(
        args.checkpoint,
        foundation_probe=args.foundation_probe,
        concept_probe=args.concept_probe,
        fast_mixed_probe=args.fast_mixed_probe,
        manual_probe=args.manual_probe,
        thresholds=thresholds,
        mode=args.mode,
        output_format=args.output_format,
        device=args.device,
        action_classifier=args.action_classifier,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
