from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psm_model.eval_generation import _parse_output
from psm_model.generate import generate_storage_json, load_checkpoint_metadata
from psm_model.safe_generate import safe_storage_decision
from psm_model.schema import validate_storage_decision


DEFAULT_PROBES: tuple[tuple[str, str], ...] = (
    (
        "preference",
        "User says: I prefer concise technical answers and want future assistant replies to avoid marketing language.",
    ),
    (
        "event",
        "Today I met Dana at 3pm to review the PSM roadmap and she asked me to send the revised memory gate by Friday.",
    ),
    (
        "noise",
        "The user says okay thanks haha and the weather outside is cloudy.",
    ),
    (
        "rule",
        "For future coding tasks, always ask before deleting generated checkpoints.",
    ),
)


Probe = tuple[str, dict[str, Any], str | None]


def probe_checkpoint(
    checkpoint: Path,
    probes: list[Probe],
    *,
    output_format: str | None = None,
    max_new_tokens: int = 220,
    device: str = "cpu",
    safe: bool = False,
    action_classifier: Path | None = None,
) -> dict[str, Any]:
    output_format = output_format or str(load_checkpoint_metadata(checkpoint).get("output_format", "json"))
    reports: list[dict[str, Any]] = []
    valid = 0
    expected_action_total = 0
    expected_action_correct = 0
    model_action_correct = 0
    for name, payload, expected_action in probes:
        if safe:
            safe_report = safe_storage_decision(checkpoint, payload, output_format=output_format, device=device, action_classifier=action_classifier)
            decision = safe_report["decision"]
            validation = validate_storage_decision(decision)
            ok = validation.ok
            valid += int(ok)
            if expected_action:
                expected_action_total += 1
                expected_action_correct += int(decision["action"] == expected_action)
                model_action_correct += int(safe_report["model_action"] == expected_action)
            reports.append(
                {
                    "case": name,
                    "action_scores": safe_report["action_scores"],
                    "calibrated_action": safe_report["calibrated_action"],
                    "decoder_action": safe_report.get("decoder_action"),
                    "expected_action": expected_action,
                    "model_action": safe_report["model_action"],
                    "parsed_action": decision["action"],
                    "parsed_memory_type": decision.get("memory", {}).get("type") if decision.get("memory") else None,
                    "raw": None,
                    "valid": ok,
                    "issues": [{"path": issue.path, "message": issue.message} for issue in validation.issues],
                }
            )
            continue
        raw = generate_storage_json(
            checkpoint,
            payload,
            max_new_tokens=max_new_tokens,
            output_format=output_format,
            device=device,
        )
        parsed, parse_issues = _parse_output(raw, output_format)
        validation = validate_storage_decision(parsed) if parsed is not None else None
        issues = parse_issues if parse_issues else (validation.issues if validation else ())
        ok = bool(validation and validation.ok and not parse_issues)
        valid += int(ok)
        if expected_action and parsed:
            expected_action_total += 1
            expected_action_correct += int(parsed.get("action") == expected_action)
        reports.append(
            {
                "case": name,
                "expected_action": expected_action,
                "raw": raw,
                "parsed_action": parsed.get("action") if parsed else None,
                "parsed_memory_type": parsed.get("memory", {}).get("type") if parsed and parsed.get("memory") else None,
                "valid": ok,
                "issues": [{"path": issue.path, "message": issue.message} for issue in issues],
            }
        )
    return {
        "checkpoint": str(checkpoint),
        "device": device,
        "examples": len(probes),
        "expected_action_accuracy": expected_action_correct / expected_action_total if expected_action_total else None,
        "expected_action_examples": expected_action_total,
        "model_action_accuracy": model_action_correct / expected_action_total if safe and expected_action_total else None,
        "output_format": output_format,
        "valid_rate": valid / len(probes) if probes else 0.0,
        "reports": reports,
    }


def _load_probe_file(path: Path) -> list[Probe]:
    probes: list[Probe] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row.get("case") or row.get("id") or f"probe-{idx}")
        payload = row.get("input") or {"text": row["text"]}
        expected = row.get("expected") or {}
        expected_action = expected.get("action") if isinstance(expected, dict) else row.get("expected_action")
        probes.append((name, payload, expected_action))
    return probes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run direct text probes against a PSM checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--probe-file", type=Path, help="JSONL rows with case/id and input or text.")
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"])
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--safe", action="store_true", help="Use action scoring plus constrained extractive fields instead of raw free generation.")
    parser.add_argument("--action-classifier", type=Path, help="Optional standalone action classifier checkpoint for --safe action selection.")
    args = parser.parse_args()

    if args.probe_file:
        probes = _load_probe_file(args.probe_file)
    else:
        probes = [
            (
                name,
                {
                    "conversation": f"User: {text}",
                    "operation": "remember",
                    "source_id": f"manual-probe-{name}",
                    "source_kind": "manual_probe",
                    "source_timestamp": "2026-06-03T00:00:00Z",
                },
                {
                    "preference": "promote_semantic",
                    "event": "store_episodic",
                    "noise": "ignore",
                    "rule": "promote_semantic",
                }[name],
            )
            for name, text in DEFAULT_PROBES
        ]
    print(
        json.dumps(
            probe_checkpoint(
                args.checkpoint,
                probes,
                output_format=args.output_format,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
                safe=args.safe,
                action_classifier=args.action_classifier,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
