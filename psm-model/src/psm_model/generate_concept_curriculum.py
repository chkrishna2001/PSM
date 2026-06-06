from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from psm_model.data import validate_training_row


TOOLS = [
    ("Postgres", "production services", "migrations and constraints are easier to audit", "postgres"),
    ("SQLite", "local prototypes", "it is easy to inspect", "sqlite"),
    ("Redis", "short-lived cache state", "expiration behavior is simple", "redis"),
    ("DuckDB", "local analytics", "it handles columnar queries without a server", "duckdb"),
    ("TypeScript", "frontend projects", "compile-time checks catch integration mistakes", "typescript"),
    ("PowerShell", "Windows automation", "it works consistently on this machine", "powershell"),
    ("Playwright", "browser smoke tests", "screenshots catch layout regressions", "playwright"),
    ("pytest", "Python regression tests", "fixtures keep cases readable", "pytest"),
]

EVENTS = [
    ("moved the staging API from port 8080 to 8090", "staging_api", "port_8090"),
    ("migrated the cache store to Redis", "cache_store", "redis"),
    ("uploaded the reviewed PSM dataset to Hugging Face", "psm_dataset", "hugging_face"),
    ("fixed the tokenizer prompt mismatch", "tokenizer_prompt", "prompt_mismatch"),
    ("ran the direct-probe gate on the new checkpoint", "direct_probe_gate", "checkpoint_gate"),
    ("created the action-prefix diagnostic report", "action_prefix_diagnostic", "diagnostic"),
    ("reduced the GPU memory fraction for laptop training", "gpu_memory_fraction", "thermal_training"),
    ("converted the fast-mixed nano dataset into storage rows", "fast_mixed_dataset", "dataset_conversion"),
]

RULES = [
    ("always run unit tests before packaging a release", "release_process", "run_tests_before_packaging"),
    ("use checkpoint-best.pt for evaluation reports", "evaluation", "checkpoint_best"),
    ("keep generated checkpoints out of git", "source_control", "ignore_checkpoints"),
    ("gate datasets before training", "data_quality", "gate_before_training"),
    ("prefer action-prefix diagnostics before long validation sweeps", "training_diagnostics", "action_prefix_first"),
    ("use CUDA only when thermals remain stable", "training_runtime", "thermal_cuda"),
]

NOISE = [
    "thanks, that worked",
    "okay sounds good",
    "please continue",
    "that output is fine",
    "show me the next command",
    "I will paste the JSON after it finishes",
    "the terminal is still running",
    "never mind, I found it",
]


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(80):
        tool, use_case, reason, tag = TOOLS[index % len(TOOLS)]
        rows.append(
            row(
                f"concept-preference-{index}",
                {"conversation": f"User: I prefer {tool} for {use_case} because {reason}."},
                expected_semantic(
                    "promote_semantic",
                    f"The user prefers {tool} for {use_case} because {reason}.",
                    ["preference", tag, slug(use_case)],
                    "user",
                    "prefers",
                    f"{tool} for {use_case}",
                    f"I prefer {tool} for {use_case} because {reason}.",
                    "The user stated a durable preference.",
                ),
            )
        )
        rows.append(
            row(
                f"concept-preference-ignore-{index}",
                {"conversation": f"User: {tool} can be used for {use_case}, and {reason}."},
                expected_ignore("The message describes a tool generally but does not state the user's durable preference."),
            )
        )

    for index in range(80):
        rule, subject, value = RULES[index % len(RULES)]
        evidence = f"For this repo, {rule}."
        rows.append(
            row(
                f"concept-rule-{index}",
                {"conversation": f"User: {evidence}"},
                expected_semantic(
                    "promote_semantic",
                    f"For this repo, {rule}.",
                    ["repo_rule", subject, value],
                    "this repo",
                    "requires",
                    value,
                    evidence,
                    "The user gave a durable project rule.",
                ),
            )
        )
        rows.append(
            row(
                f"concept-rule-ignore-{index}",
                {"conversation": f"User: I might {rule} later if there is time."},
                expected_ignore("The message is tentative and does not establish a durable rule."),
            )
        )

    for index in range(80):
        event, subject, value = EVENTS[index % len(EVENTS)]
        evidence = f"This morning I {event} after the validation failed."
        rows.append(
            row(
                f"concept-event-{index}",
                {
                    "conversation": f"User: {evidence}",
                    "source_timestamp": "2026-06-03T09:15:00-04:00",
                },
                expected_episodic(
                    f"The user {event} this morning after validation failed.",
                    ["event", subject, value],
                    "user",
                    "completed",
                    value,
                    evidence,
                    "This morning",
                    "2026-06-03",
                ),
            )
        )
        rows.append(
            row(
                f"concept-event-ignore-{index}",
                {"conversation": f"User: I may {event} sometime this week if needed."},
                expected_ignore("The message is tentative future planning, not a completed event."),
            )
        )

    for index in range(120):
        message = NOISE[index % len(NOISE)]
        rows.append(row(f"concept-noise-{index}", {"conversation": f"User: {message}"}, expected_ignore("No durable memory value.")))

    for index in range(60):
        old = RULES[index % len(RULES)]
        new = RULES[(index + 1) % len(RULES)]
        evidence = f"Correction: do not {old[0]}; instead, {new[0]}."
        rows.append(
            row(
                f"concept-correction-{index}",
                {"conversation": f"User: {evidence}"},
                expected_semantic(
                    "flag_and_store",
                    f"Do not {old[0]}; instead, {new[0]}.",
                    ["correction", new[1], new[2]],
                    "this repo",
                    "corrected_rule",
                    new[2],
                    evidence,
                    "The user corrected a prior durable rule.",
                ),
            )
        )
    return rows


def row(row_id: str, input_payload: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    result = {"id": row_id, "input": input_payload, "expected": expected, "source": "concept_curriculum"}
    _, issues = validate_training_row(result)
    if issues:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise ValueError(f"{row_id}: {formatted}")
    return result


def expected_ignore(reasoning: str) -> dict[str, Any]:
    return {"action": "ignore", "memory": None, "facts": [], "reasoning": reasoning}


def expected_semantic(
    action: str,
    content: str,
    tags: list[str],
    subject: str,
    predicate: str,
    value: str,
    evidence: str,
    reasoning: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "memory": {
            "content": content,
            "type": "semantic",
            "strength": 0.86,
            "decay_rate": 0.02,
            "emotional_weight": 0.1,
            "confidence": 0.92,
            "tags": tags,
        },
        "facts": [fact(subject, predicate, value, evidence)],
        "reasoning": reasoning,
    }


def expected_episodic(
    content: str,
    tags: list[str],
    subject: str,
    predicate: str,
    value: str,
    evidence: str,
    temporal_expression: str,
    resolved_time: str,
) -> dict[str, Any]:
    return {
        "action": "store_episodic",
        "memory": {
            "content": content,
            "type": "episodic",
            "strength": 0.74,
            "decay_rate": 0.05,
            "emotional_weight": 0.1,
            "confidence": 0.9,
            "tags": tags,
            "temporal_expression": temporal_expression,
            "resolved_time": resolved_time,
        },
        "facts": [
            fact(
                subject,
                predicate,
                value,
                evidence,
                temporal_expression=temporal_expression,
                resolved_time=resolved_time,
            )
        ],
        "reasoning": "The user described a completed dated event.",
    }


def fact(
    subject: str,
    predicate: str,
    value: str,
    evidence: str,
    *,
    temporal_expression: str | None = None,
    resolved_time: str | None = None,
) -> dict[str, Any]:
    result = {
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "confidence": 0.92,
        "inference_kind": "explicit",
        "evidence_text": evidence,
    }
    if temporal_expression is not None:
        result["temporal_expression"] = temporal_expression
    if resolved_time is not None:
        result["resolved_time"] = resolved_time
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


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic PSM concept/minimal-pair curriculum rows.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows = build_rows()
    splits = split_rows(rows)
    for split, split_rows_value in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_rows_value)
    write_jsonl(args.output_dir / "all.jsonl", rows)
    report = {
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "splits": {split: len(split_rows_value) for split, split_rows_value in splits.items()},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
