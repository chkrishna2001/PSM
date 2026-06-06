from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from psm_model.data import validate_training_row
from psm_model.generate_action_foundation_curriculum import expected_ignore, expected_memory


PREFERENCES = [
    ("concise technical answers", "answer_style", "concise_technical_answers"),
    ("one-line PowerShell commands", "command_style", "powershell_one_line"),
    ("small focused diffs", "coding_style", "small_focused_diffs"),
    ("checkpoint gates before long training", "training_process", "checkpoint_gates_first"),
    ("direct evidence in memory facts", "memory_quality", "direct_evidence"),
    ("quiet, factual status updates", "status_style", "quiet_factual_updates"),
]

RULES = [
    ("ask before deleting generated checkpoints", "checkpoint_deletion", "requires_confirmation"),
    ("gate datasets before training", "dataset_training", "gate_first"),
    ("run direct probes after every checkpoint", "checkpoint_evaluation", "direct_probe_required"),
    ("avoid CPU torch for GPU training", "training_runtime", "cuda_torch_required"),
    ("keep generated checkpoints out of git", "source_control", "exclude_checkpoints"),
    ("use action diagnostics before full validation", "validation_order", "action_diagnostics_first"),
]

EVENTS = [
    ("met Dana at 3pm to review the PSM roadmap", "roadmap_review", "met_dana"),
    ("ran the foundation action-prefix gate", "action_prefix_gate", "completed"),
    ("uploaded the dataset snapshot", "dataset_snapshot", "uploaded"),
    ("fixed the malformed fact parser", "fact_parser", "fixed"),
    ("created the mixed repair checkpoint", "mixed_repair_checkpoint", "created"),
    ("validated the concept probe", "concept_probe", "validated"),
]

NOISE = [
    "okay thanks haha and the weather outside is cloudy",
    "please continue, I will paste the JSON when the terminal finishes",
    "sounds good",
    "that worked",
    "the terminal is still running",
    "I will check nvidia-smi",
]


def build_rows(rows_per_action: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(rows_per_action):
        preference, preference_predicate, preference_value = PREFERENCES[index % len(PREFERENCES)]
        rule, rule_predicate, rule_value = RULES[index % len(RULES)]
        event, event_predicate, event_value = EVENTS[index % len(EVENTS)]
        noise = NOISE[index % len(NOISE)]
        date = f"2026-06-{(index % 20) + 1:02d}"

        rows.append(
            row(
                f"direct-promote-preference-{index}",
                base_input(f"User: I prefer {preference}.", index),
                expected_memory(
                    "promote_semantic",
                    "semantic",
                    f"The user prefers {preference}.",
                    ["preference", preference_predicate],
                    "User",
                    "prefers",
                    preference_value,
                    f"I prefer {preference}.",
                    "The user stated a durable preference.",
                ),
            )
        )
        rows.append(
            row(
                f"direct-promote-rule-{index}",
                base_input(f"User: For future coding tasks, always {rule}.", index),
                expected_memory(
                    "promote_semantic",
                    "semantic",
                    f"For future coding tasks, always {rule}.",
                    ["rule", rule_predicate],
                    "future coding tasks",
                    rule_predicate,
                    rule_value,
                    f"For future coding tasks, always {rule}.",
                    "The user stated a durable rule.",
                ),
            )
        )
        rows.append(
            row(
                f"direct-store-event-{index}",
                base_input(f"User: On {date}, I {event}.", index, source_timestamp=f"{date}T12:00:00Z"),
                expected_memory(
                    "store_episodic",
                    "episodic",
                    f"On {date}, the user {event}.",
                    ["event", event_predicate],
                    "User",
                    event_predicate,
                    event_value,
                    f"On {date}, I {event}.",
                    "The user described a completed dated event.",
                    temporal_expression=date,
                    resolved_time=date,
                    decay_rate=0.05,
                ),
            )
        )
        rows.append(
            row(
                f"direct-store-today-{index}",
                base_input(f"User: Today I {event}.", index, source_timestamp=f"{date}T12:00:00Z"),
                expected_memory(
                    "store_episodic",
                    "episodic",
                    f"Today the user {event}.",
                    ["event", event_predicate],
                    "User",
                    event_predicate,
                    event_value,
                    f"Today I {event}.",
                    "The user described a completed event in the current turn.",
                    temporal_expression=date,
                    resolved_time=date,
                    decay_rate=0.05,
                ),
            )
        )
        rows.append(row(f"direct-ignore-{index}", base_input(f"User: {noise}.", index), expected_ignore("No durable memory value.")))
        rows.append(
            row(
                f"direct-update-{index}",
                base_input(
                    f"User: Correction: use {preference} instead.",
                    index,
                    context=json.dumps({"memory_store": [{"id": "m1", "content": f"User previously preferred {rule}."}]}),
                ),
                expected_memory(
                    "update_existing",
                    "semantic",
                    f"Update the stored preference: the user now prefers {preference}.",
                    ["update", preference_predicate],
                    "User",
                    "updated_preference",
                    preference_value,
                    f"Correction: use {preference} instead.",
                    "The current message updates an existing memory.",
                ),
            )
        )
        rows.append(
            row(
                f"direct-conflict-{index}",
                base_input(
                    f"User: Actually, do not {rule}; keep the previous behavior.",
                    index,
                    context=json.dumps({"memory_store": [{"id": "m1", "content": f"For future coding tasks, always {rule}."}]}),
                ),
                expected_memory(
                    "flag_conflict",
                    "semantic",
                    f"The user contradicted the prior rule: {rule}.",
                    ["conflict", rule_predicate],
                    "future coding tasks",
                    "conflicts_on",
                    rule_value,
                    f"Actually, do not {rule}; keep the previous behavior.",
                    "The message conflicts with an existing stored rule.",
                ),
            )
        )
        rows.append(
            row(
                f"direct-flag-store-{index}",
                base_input(f"User: Correction: do not {rule}; instead, always {PREFERENCES[(index + 1) % len(PREFERENCES)][0]}.", index),
                expected_memory(
                    "flag_and_store",
                    "semantic",
                    f"Correction: do not {rule}; instead, always {PREFERENCES[(index + 1) % len(PREFERENCES)][0]}.",
                    ["correction", rule_predicate],
                    "future coding tasks",
                    "corrected_rule",
                    rule_value,
                    f"Correction: do not {rule}; instead, always {PREFERENCES[(index + 1) % len(PREFERENCES)][0]}.",
                    "The user provided a correction that should be stored.",
                ),
            )
        )
    return rows


def base_input(conversation: str, index: int, *, context: str | None = None, source_timestamp: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "conversation": conversation,
        "operation": "remember",
        "source_id": f"direct-behavior-{index}",
        "source_kind": "manual_probe",
        "source_timestamp": source_timestamp or "2026-06-03T12:00:00Z",
    }
    if context:
        payload["context"] = context
    return payload


def row(row_id: str, input_payload: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    result = {"id": row_id, "input": input_payload, "expected": expected, "source": "direct_behavior_curriculum"}
    _, issues = validate_training_row(result)
    if issues:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise ValueError(f"{row_id}: {formatted}")
    return result


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "validation": [], "test": []}
    for item in rows:
        bucket = int(hashlib.sha256(item["id"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < 0.1:
            splits["test"].append(item)
        elif bucket < 0.25:
            splits["validation"].append(item)
        else:
            splits["train"].append(item)
    return splits


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate direct manual-style PSM behavior curriculum rows.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rows-per-action", type=int, default=450)
    args = parser.parse_args()

    rows = build_rows(args.rows_per_action)
    splits = split_rows(rows)
    for split, split_rows_value in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_rows_value)
    write_jsonl(args.output_dir / "all.jsonl", rows)
    report = {
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "rows_per_action": args.rows_per_action,
        "splits": {split: len(split_rows_value) for split, split_rows_value in splits.items()},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
