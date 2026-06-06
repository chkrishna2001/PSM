from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from psm_model.data import validate_training_row
from psm_model.generate_action_foundation_curriculum import expected_ignore, expected_memory


LOCAL_SEMANTIC = [
    ("incremental version upgrades with staged npm run version-packages, npm run build, then git commit", "version_workflow", "staged_version_build_commit"),
    ("semantic versioning over patch-level changes", "versioning_preference", "semantic_versioning"),
    ("direct evidence in memory facts before storing extracted claims", "memory_quality", "direct_evidence_required"),
    ("small focused diffs for repair work", "coding_style", "small_focused_diffs"),
]

LOCAL_EPISODIC = [
    ("validated hybrid retrieval ranking in ranking.ts during the production-readiness pass", "hybrid_retrieval_validation", "validated"),
    ("created the Colab restart-safe checkpoint workflow for the 50M storage model", "colab_checkpoint_workflow", "created"),
    ("ran the consolidated gate after the concept repair checkpoint", "concept_repair_gate", "ran"),
    ("uploaded a checkpoint artifact snapshot after the repair segment", "checkpoint_artifact_snapshot", "uploaded"),
]

LONGMEM_QUESTIONS = [
    ("Who gave me a new stand mixer as a birthday gift?", "my sister"),
    ("Which event did I attend first, the Effective Time Management workshop or the Data Analysis using Python webinar?", "Data Analysis using Python webinar"),
    ("How many days ago did I harvest my first batch of fresh herbs from the herb garden kit?", "3 days ago"),
    ("Which city did I visit before the product planning workshop?", "Boston"),
]

PERSONAL_EVENTS = [
    ("met Dana at 3pm to review the PSM roadmap", "roadmap_review", "met_dana"),
    ("attended a festival featuring traditional Pacific music in 2012", "festival_attendance", "traditional_pacific_music_festival"),
    ("continued education planning and explored career options", "career_planning", "education_and_career_options"),
    ("ran the foundation action-prefix gate and checked the promote failures", "action_gate_review", "reviewed_promote_failures"),
]

NOISE = [
    "okay thanks haha and the weather outside is cloudy",
    "please continue, I will paste the JSON when the terminal finishes",
    "sounds good, I will check nvidia-smi",
    "the terminal is still running and I will paste the result",
]

RULES = [
    ("always ask before deleting generated checkpoints", "checkpoint_deletion", "requires_confirmation"),
    ("always gate datasets before starting a long training run", "dataset_training", "gate_first"),
    ("use the consolidated checkpoint gate after every segment", "checkpoint_evaluation", "gate_after_segment"),
    ("prefer restart-safe Colab commands with resume auto and a fallback checkpoint", "colab_training", "restart_safe_resume"),
]


def build_rows(rows_per_pattern: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(rows_per_pattern):
        semantic, semantic_predicate, semantic_value = LOCAL_SEMANTIC[index % len(LOCAL_SEMANTIC)]
        episodic, episodic_predicate, episodic_value = LOCAL_EPISODIC[index % len(LOCAL_EPISODIC)]
        question, answer = LONGMEM_QUESTIONS[index % len(LONGMEM_QUESTIONS)]
        event, event_predicate, event_value = PERSONAL_EVENTS[index % len(PERSONAL_EVENTS)]
        noise = NOISE[index % len(NOISE)]
        rule, rule_predicate, rule_value = RULES[index % len(RULES)]
        date = f"2026-06-{(index % 20) + 1:02d}"

        rows.append(
            row(
                f"hard-local-semantic-{index}",
                base_input(
                    f"User: User prefers {semantic}.",
                    index,
                    source_kind="local_psm_db",
                    source_id=f"semantic:hard-{index}",
                    source_timestamp=f"{date}T13:10:52Z",
                ),
                expected_memory(
                    "promote_semantic",
                    "semantic",
                    f"User prefers {semantic}.",
                    ["local_psm", "semantic", semantic_predicate],
                    "User",
                    semantic_predicate,
                    semantic_value,
                    f"User prefers {semantic}.",
                    "Existing semantic PSM memory row should remain semantic.",
                ),
            )
        )
        rows.append(
            row(
                f"hard-local-episodic-{index}",
                base_input(
                    f"User: User {episodic}.",
                    index,
                    source_kind="local_psm_db",
                    source_id=f"episodic:hard-{index}",
                    source_timestamp=f"{date}T17:09:36Z",
                ),
                expected_memory(
                    "store_episodic",
                    "episodic",
                    f"User {episodic}.",
                    ["local_psm", "episodic", episodic_predicate],
                    "User",
                    episodic_predicate,
                    episodic_value,
                    f"User {episodic}.",
                    "Existing episodic PSM memory row should remain episodic.",
                    temporal_expression=date,
                    resolved_time=date,
                    decay_rate=0.05,
                ),
            )
        )
        rows.append(
            row(
                f"hard-longmem-update-{index}",
                base_input(
                    f"User: {question} {answer}",
                    index,
                    source_kind="longmemeval_update",
                    source_id=f"longmemeval-update-hard-{index}",
                    source_timestamp=f"2023/05/{(index % 20) + 1:02d} (Mon) 06:47",
                    context=json.dumps({"memory_store": [{"id": f"longmem-old-{index}", "content": f"Prior user information may be outdated for: {question}"}]}),
                ),
                expected_memory(
                    "update_existing",
                    "semantic",
                    f"User's current answer for \"{question}\" is: {answer}.",
                    ["longmemeval", "update"],
                    "User",
                    "current_answer_for",
                    f"{question}: {answer}",
                    f"{question} {answer}",
                    "Question-answer wording with outdated prior context should update existing memory.",
                ),
            )
        )
        rows.append(
            row(
                f"hard-personal-event-{index}",
                base_input(
                    f"User: Today I {event}.",
                    index,
                    source_kind="personamem",
                    source_id=f"personamem-hard-{index}",
                    source_timestamp=f"{date}T12:00:00Z",
                ),
                expected_memory(
                    "store_episodic",
                    "episodic",
                    f"Today the user {event}.",
                    ["event", event_predicate],
                    "User",
                    event_predicate,
                    event_value,
                    f"Today I {event}.",
                    "The user described a specific personal event.",
                    temporal_expression=date,
                    resolved_time=date,
                    decay_rate=0.05,
                ),
            )
        )
        rows.append(row(f"hard-ignore-noise-{index}", base_input(f"User: {noise}.", index, source_kind="manual_probe"), expected_ignore("No durable memory value.")))
        rows.append(
            row(
                f"hard-rule-promote-{index}",
                base_input(f"User: For future coding tasks, {rule}.", index, source_kind="manual_probe"),
                expected_memory(
                    "promote_semantic",
                    "semantic",
                    f"For future coding tasks, {rule}.",
                    ["rule", rule_predicate],
                    "future coding tasks",
                    rule_predicate,
                    rule_value,
                    f"For future coding tasks, {rule}.",
                    "The user stated a durable future rule.",
                ),
            )
        )
    return rows


def base_input(
    conversation: str,
    index: int,
    *,
    source_kind: str,
    source_id: str | None = None,
    source_timestamp: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "conversation": conversation,
        "operation": "remember",
        "source_id": source_id or f"hard-behavior-{index}",
        "source_kind": source_kind,
        "source_timestamp": source_timestamp or "2026-06-03T12:00:00Z",
    }
    if context:
        payload["context"] = context
    return payload


def row(row_id: str, input_payload: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    result = {"id": row_id, "input": input_payload, "expected": expected, "source": "hard_behavior_curriculum"}
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
    parser = argparse.ArgumentParser(description="Generate hard behavior repair rows for PSM action scoring.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rows-per-pattern", type=int, default=300)
    args = parser.parse_args()

    rows = build_rows(args.rows_per_pattern)
    splits = split_rows(rows)
    for split, split_rows_value in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_rows_value)
    write_jsonl(args.output_dir / "all.jsonl", rows)
    report = {
        "output_dir": str(args.output_dir),
        "rows": len(rows),
        "rows_per_pattern": args.rows_per_pattern,
        "splits": {split: len(split_rows_value) for split, split_rows_value in splits.items()},
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
