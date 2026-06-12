from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PSM_SYSTEM = (
    "You are the Personal Small Model (PSM), a specialized AI trained exclusively to perform memory "
    "management operations for LLM agents. You do not answer user questions. You plan memory retrieval "
    "and storage. Always respond with a valid JSON object."
)

AVAILABLE_TABLES = ["episodic", "semantic", "archival"]


def _recall_row(
    row_id: str,
    *,
    operation: str,
    question: str | None = None,
    user_prompt: str | None = None,
    expected: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": operation,
        "available_tables": AVAILABLE_TABLES,
        "requested_top_k": top_k,
    }
    if operation == "recall_plan":
        payload["question"] = question
    else:
        payload["user_prompt"] = user_prompt or question
    return {
        "id": row_id,
        "task": operation,
        "input": payload,
        "expected": expected,
        "source": "generate_recall_curriculum",
        "split": "train",
    }


def build_recall_probe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_recall(row_id: str, question: str, expected_tables: list[str], hints: list[str], temporal: str | None = None) -> None:
        rows.append(
            _recall_row(
                row_id,
                operation="recall_plan",
                question=question,
                expected={
                    "intent": "recall",
                    "target_tables": expected_tables,
                    "filters": {},
                    "ranking_hints": hints,
                    "temporal_intent": temporal,
                    "top_k": 5,
                },
            )
        )

    def add_context(row_id: str, prompt: str, expected_tables: list[str], hints: list[str], temporal: str | None = None) -> None:
        rows.append(
            _recall_row(
                row_id,
                operation="context_plan",
                user_prompt=prompt,
                expected={
                    "intent": "context",
                    "target_tables": expected_tables,
                    "filters": {},
                    "ranking_hints": hints,
                    "temporal_intent": temporal,
                    "top_k": 5,
                },
            )
        )

    # Episodic / event questions
    add_recall("recall-episodic-painting-2022", "What painting did Melanie share from 2022?", ["episodic"], ["Melanie", "painting", "2022"], "2022")
    add_recall("recall-episodic-camping-trip", "When did we go camping last summer?", ["episodic"], ["camping", "summer", "trip"], "last summer")
    add_recall("recall-episodic-yesterday-meeting", "What happened in yesterday's standup?", ["episodic"], ["standup", "yesterday", "meeting"], "yesterday")
    add_recall("recall-episodic-recent-error", "What bug did I fix earlier today?", ["episodic"], ["bug", "fix", "today"], "today")

    # Semantic / profile / preference
    add_recall("recall-semantic-dark-mode", "What are my UI theme preferences?", ["semantic"], ["dark mode", "theme", "preference"])
    add_recall("recall-semantic-diet", "Do I have any dietary restrictions?", ["semantic"], ["vegetarian", "diet", "food", "allergy"])
    add_recall("recall-semantic-work-role", "What is my current job title?", ["semantic"], ["job", "title", "role", "work"])
    add_recall("recall-semantic-tooling", "Which database do I prefer for prototypes?", ["semantic"], ["SQLite", "database", "prototype"])

    # Archival / old history
    add_recall("recall-archival-old-project", "What did we decide about the 2019 migration project?", ["archival"], ["2019", "migration", "project"], "2019")
    add_recall("recall-archival-former-address", "Where did I live before moving to Seattle?", ["archival", "semantic"], ["address", "Seattle", "move"])

    # Mixed tables
    add_recall("recall-mixed-relationship", "Tell me about Melanie's hobbies and recent activities.", ["semantic", "episodic"], ["Melanie", "hobbies", "activities"])
    add_recall("recall-mixed-project-status", "What's the status of the PSM Gate 4 work and recent training runs?", ["semantic", "episodic"], ["PSM", "Gate 4", "training"])

    # Temporal-heavy
    add_recall("recall-temporal-may-2023", "What happened on 7 May 2023?", ["episodic"], ["May", "2023", "7"], "2023-05-07")
    add_recall("recall-temporal-last-year", "What major trips did I take last year?", ["episodic", "archival"], ["trip", "travel", "last year"], "last year")

    # Context-plan mirrors (agent prompt, not question)
    add_context("context-episodic-followup", "Continue the story about Melanie's pottery class.", ["episodic"], ["Melanie", "pottery", "class"])
    add_context("context-semantic-coding-style", "Write code in the style I usually prefer.", ["semantic"], ["coding style", "preference", "lint"])
    add_context("context-mixed-trip-plan", "Help me plan a trip using what you know about my past travel.", ["semantic", "episodic"], ["travel", "trip", "preference"])
    add_context("context-archival-history", "Summarize early project history without inventing details.", ["archival", "semantic"], ["project", "history", "decision"])
    add_context("context-temporal-deadline", "Am I free next Friday based on what I said about deadlines?", ["episodic", "semantic"], ["Friday", "deadline", "schedule"], "next Friday")

    # Edge routing
    add_recall("recall-entity-name", "Who is Caroline and what do I know about her?", ["semantic", "episodic"], ["Caroline"])
    add_recall("recall-source-id", "What was stored from conv-26 session D1?", ["episodic", "semantic"], ["conv-26", "D1", "source"])
    add_recall("recall-fact-predicate", "What is Melanie's relationship status?", ["semantic"], ["Melanie", "relationship"])
    add_recall("recall-workflow-preference", "How do I usually run local eval for PSM checkpoints?", ["semantic", "episodic"], ["eval", "checkpoint", "local", "PSM"])

    return rows


def build_recall_curriculum(output: Path) -> dict[str, Any]:
    rows = build_recall_probe_rows()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {
        "output": str(output),
        "rows": len(rows),
        "recall_plan_rows": sum(1 for row in rows if row["task"] == "recall_plan"),
        "context_plan_rows": sum(1 for row in rows if row["task"] == "context_plan"),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl"),
    )
    args = parser.parse_args()
    print(json.dumps(build_recall_curriculum(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
