from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psm_model.data.rows import validate_training_row


def generate_seed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_semantic_preferences())
    rows.extend(_episodic_events())
    rows.extend(_project_instructions())
    rows.extend(_ignore_noise())
    rows.extend(_conflicts())
    rows.extend(_updates())
    _validate_rows(rows)
    return rows


def write_seed_jsonl(path: Path) -> int:
    rows = generate_seed_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return len(rows)


def split_rows(rows: list[dict[str, Any]], *, validation_every: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        target = validation if index % validation_every == validation_every - 1 else train
        target.append(row)
    return train, validation


def _semantic_preferences() -> list[dict[str, Any]]:
    items = [
        ("sqlite", "SQLite", "local prototypes", "easy to inspect"),
        ("duckdb", "DuckDB", "local analytics", "fast columnar queries"),
        ("csharp", "CSharp", "backend services", "strong typing"),
        ("markdown", "Markdown", "planning notes", "simple diffs"),
        ("powershell", "PowerShell", "Windows automation", "native shell support"),
        ("pytest", "pytest", "Python tests", "clear fixtures"),
        ("typescript", "TypeScript", "SDK code", "type safety"),
        ("onnx", "ONNX", "CPU deployment", "portable runtime"),
        ("jsonl", "JSONL", "dataset records", "streaming friendly"),
        ("small_prs", "small focused PRs", "repo changes", "reviewability"),
    ]
    rows = []
    for index, (key, value, scope, reason) in enumerate(items, start=1):
        evidence = f"I prefer {value} for {scope} because it is {reason}."
        rows.append(
            _row(
                f"semantic_pref_{index:03d}",
                evidence,
                {
                    "action": "promote_semantic",
                    "memory": _memory(
                        f"The user prefers {value} for {scope} because it is {reason}.",
                        "semantic",
                        ["preference", key],
                        confidence=0.92,
                        strength=0.82,
                        decay_rate=0.02,
                    ),
                    "facts": [_fact("user", "prefers", f"{value} for {scope}", 0.92, evidence)],
                    "reasoning": "The user stated a durable preference.",
                },
            )
        )
    return rows


def _episodic_events() -> list[dict[str, Any]]:
    items = [
        ("Redis", "cache store", "staging deploy", "2026-05-31"),
        ("SQLite", "local memory store", "smoke test", "2026-05-30"),
        ("ONNX", "export path", "CPU benchmark", "2026-05-29"),
        ("Hugging Face", "checkpoint sync", "Colab run", "2026-05-28"),
        ("LoCoMo", "answer evaluation", "benchmark pass", "2026-05-27"),
        ("TypeScript", "package build", "release prep", "2026-05-26"),
        ("DuckDB", "report query", "analysis session", "2026-05-25"),
        ("pytest", "schema tests", "validation pass", "2026-05-24"),
        ("PowerShell", "install script", "company VM setup", "2026-05-23"),
        ("JSONL", "seed dataset", "model smoke test", "2026-05-22"),
    ]
    rows = []
    for index, (tool, target, event, date) in enumerate(items, start=1):
        evidence = f"On {date}, I migrated the {target} to {tool} during the {event}."
        rows.append(
            _row(
                f"episodic_event_{index:03d}",
                evidence,
                {
                    "action": "store_episodic",
                    "memory": _memory(
                        f"The user migrated the {target} to {tool} during the {event} on {date}.",
                        "episodic",
                        ["event", tool.lower().replace(" ", "_")],
                        confidence=0.9,
                        strength=0.74,
                        decay_rate=0.05,
                        temporal_expression=f"On {date}",
                        resolved_time=date,
                    ),
                    "facts": [_fact("user", "migrated", f"{target} to {tool}", 0.9, evidence, temporal_expression=f"On {date}", resolved_time=date)],
                    "reasoning": "The message describes a dated project event.",
                },
            )
        )
    return rows


def _project_instructions() -> list[dict[str, Any]]:
    items = [
        ("release", "run version-packages before npm run build"),
        ("model eval", "check generated JSON validity before action accuracy"),
        ("dataset gate", "reject facts without evidence_text"),
        ("runtime write", "validate schema before writing memory"),
        ("company VM", "prefer CPU-first dependencies"),
        ("benchmark", "run direct probes before LoCoMo"),
        ("training", "save resumable checkpoints"),
        ("review", "inspect generated-agent diffs before merging"),
        ("recall", "dedupe retrieved memories before prompt packing"),
        ("fallback", "use Qwen when local generation is invalid"),
    ]
    rows = []
    for index, (scope, rule) in enumerate(items, start=1):
        evidence = f"For {scope}, always {rule}."
        rows.append(
            _row(
                f"project_instruction_{index:03d}",
                evidence,
                {
                    "action": "promote_semantic",
                    "memory": _memory(
                        f"For {scope}, always {rule}.",
                        "semantic",
                        ["workflow", scope.replace(" ", "_")],
                        confidence=0.94,
                        strength=0.88,
                        decay_rate=0.01,
                    ),
                    "facts": [_fact(scope, "requires", rule, 0.94, evidence)],
                    "reasoning": "The user gave a durable workflow instruction.",
                },
            )
        )
    return rows


def _ignore_noise() -> list[dict[str, Any]]:
    messages = [
        "okay thanks",
        "sounds good",
        "nice",
        "continue",
        "go on",
        "yes",
        "no worries",
        "that works",
        "cool",
        "thanks for checking",
    ]
    return [
        _row(
            f"ignore_noise_{index:03d}",
            message,
            {"action": "ignore", "memory": None, "facts": [], "reasoning": "The message has no durable memory value."},
        )
        for index, message in enumerate(messages, start=1)
    ]


def _conflicts() -> list[dict[str, Any]]:
    items = [
        ("do not use the JSON fallback parser for model readiness", "it hides invalid outputs"),
        ("do not treat label accuracy as production readiness", "we need generated storage decisions"),
        ("do not upload raw private transcripts", "they may contain secrets"),
        ("do not let generated agents edit the main worktree", "diff ownership becomes unclear"),
        ("do not train on raw transcripts directly", "examples need exact storage outputs"),
        ("do not claim LoCoMo readiness before schema gates", "invalid writes corrupt evaluation"),
        ("do not use all recalled memories in the prompt", "context must be budgeted"),
        ("do not rely on fact counts alone", "fact content and evidence must match"),
        ("do not use hidden background agent windows", "visibility helps diagnose stalls"),
        ("do not silently truncate training rows", "the model cannot learn missing output"),
    ]
    rows = []
    for index, (rule, reason) in enumerate(items, start=1):
        evidence = f"Correction: {rule}; {reason}."
        rows.append(
            _row(
                f"conflict_{index:03d}",
                evidence,
                {
                    "action": "flag_and_store",
                    "memory": _memory(
                        f"{rule.capitalize()} because {reason}.",
                        "semantic",
                        ["correction", "safety"],
                        confidence=0.93,
                        strength=0.86,
                        decay_rate=0.01,
                        emotional_weight=0.2,
                    ),
                    "facts": [_fact("project workflow", "should_not", rule, 0.93, evidence)],
                    "reasoning": "The user corrected prior behavior and the new rule should be stored.",
                },
            )
        )
    return rows


def _updates() -> list[dict[str, Any]]:
    items = [
        ("preferred local database", "SQLite", "DuckDB"),
        ("release checklist", "npm run build first", "version-packages before npm run build"),
        ("model output format", "full JSON", "pipe-tagged DSL"),
        ("training context", "1024 tokens", "2048 tokens"),
        ("agent workflow", "hidden background workers", "visible isolated worktrees"),
        ("readiness gate", "loss only", "parse and schema validation"),
        ("fact validation", "fact count only", "subject predicate value and evidence exact match"),
        ("runtime fallback", "store parser fallback content", "reject or Qwen fallback"),
        ("dataset source", "raw transcript rows", "canonical input and expected output rows"),
        ("deployment target", "GPU-only model", "CPU-first model"),
    ]
    rows = []
    for index, (subject, old, new) in enumerate(items, start=1):
        evidence = f"Update the {subject}: use {new}, not {old}."
        rows.append(
            _row(
                f"update_existing_{index:03d}",
                evidence,
                {
                    "action": "update_existing",
                    "memory": _memory(
                        f"Update the {subject}: use {new} instead of {old}.",
                        "semantic",
                        ["update", subject.replace(" ", "_")],
                        confidence=0.91,
                        strength=0.84,
                        decay_rate=0.015,
                    ),
                    "facts": [_fact(subject, "updated_to", new, 0.91, evidence)],
                    "reasoning": "The message updates a previously stored project preference or rule.",
                },
            )
        )
    return rows


def _row(row_id: str, conversation: str, expected: dict[str, Any]) -> dict[str, Any]:
    return {"id": row_id, "input": {"conversation": f"User: {conversation}"}, "expected": expected, "source": "deterministic_seed"}


def _memory(
    content: str,
    memory_type: str,
    tags: list[str],
    *,
    confidence: float,
    strength: float,
    decay_rate: float,
    emotional_weight: float = 0.1,
    temporal_expression: str | None = None,
    resolved_time: str | None = None,
) -> dict[str, Any]:
    memory = {
        "content": content,
        "type": memory_type,
        "strength": strength,
        "decay_rate": decay_rate,
        "emotional_weight": emotional_weight,
        "confidence": confidence,
        "tags": tags,
    }
    if temporal_expression:
        memory["temporal_expression"] = temporal_expression
    if resolved_time:
        memory["resolved_time"] = resolved_time
    return memory


def _fact(
    subject: str,
    predicate: str,
    value: str,
    confidence: float,
    evidence_text: str,
    *,
    temporal_expression: str | None = None,
    resolved_time: str | None = None,
) -> dict[str, Any]:
    fact = {
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "confidence": confidence,
        "inference_kind": "explicit",
        "evidence_text": evidence_text,
    }
    if temporal_expression:
        fact["temporal_expression"] = temporal_expression
    if resolved_time:
        fact["resolved_time"] = resolved_time
    return fact


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"duplicate seed id: {row['id']}")
        ids.add(row["id"])
        _, issues = validate_training_row(row)
        if issues:
            formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
            raise ValueError(f"invalid seed row {row['id']}: {formatted}")

