from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from psm_model.data import validate_training_row
from psm_model.generate_action_foundation_curriculum import expected_memory, row

PREFERENCE_BITS = [
    "JSON over XML",
    "SQLite for local prototypes",
    "small focused diffs",
    "checkpoint gates before long training",
    "PowerShell one-line commands",
    "direct evidence in memory facts",
]
RULE_BITS = [
    "gate datasets before training",
    "ask before deleting checkpoints",
    "run direct probes after every checkpoint",
    "keep generated checkpoints out of git",
    "avoid CPU torch for GPU training",
]
EVENT_BITS = [
    "migrated the auth service to OIDC",
    "fixed the malformed fact parser",
    "uploaded the dataset snapshot",
    "validated the concept probe",
    "created the mixed repair checkpoint",
]


def _base_input(conversation: str, index: int, *, source_timestamp: str | None = None) -> dict[str, Any]:
    return {
        "conversation": conversation,
        "operation": "remember",
        "source_id": f"fact-format-drill-{index}",
        "source_kind": "parse_drill",
        "source_timestamp": source_timestamp or "2026-06-03T12:00:00Z",
    }


def _templates(index: int) -> list[dict[str, Any]]:
    pref = PREFERENCE_BITS[index % len(PREFERENCE_BITS)]
    rule = RULE_BITS[index % len(RULE_BITS)]
    event = EVENT_BITS[index % len(EVENT_BITS)]
    date = f"2026-06-{(index % 20) + 1:02d}"
    tag_suffix = f"v{index % 9}.{index % 7}.{index % 5}|stable"

    return [
        row(
            f"fact-drill-pref-{index}",
            _base_input(f"User: I prefer {pref}.", index),
            expected_memory(
                "promote_semantic",
                "semantic",
                f"The user prefers {pref}.",
                ["preference", f"pref_{index}"],
                "User",
                "prefers",
                pref.replace(" ", "_").lower()[:32],
                f"I prefer {pref}.",
                "Explicit durable preference.",
            ),
        ),
        row(
            f"fact-drill-rule-{index}",
            _base_input(f"User: For deployments, always use tag {tag_suffix}.", index),
            expected_memory(
                "promote_semantic",
                "semantic",
                f"For deployments, always use tag {tag_suffix}.",
                ["rule", f"rule_{index}"],
                "deployments",
                "requires_tag",
                tag_suffix.replace("|", "_"),
                f"For deployments, always use tag {tag_suffix}.",
                "Evidence contains pipe characters requiring escape in tagged output.",
            ),
        ),
        row(
            f"fact-drill-rule-always-{index}",
            _base_input(f"User: For future coding tasks, always {rule}.", index),
            expected_memory(
                "promote_semantic",
                "semantic",
                f"For future coding tasks, always {rule}.",
                ["rule", f"coding_{index}"],
                "future coding tasks",
                "requires",
                rule.replace(" ", "_")[:32],
                f"For future coding tasks, always {rule}.",
                "Durable coding rule.",
            ),
        ),
        row(
            f"fact-drill-event-{index}",
            _base_input(f"User: On {date}, I {event}.", index, source_timestamp=f"{date}T09:30:00Z"),
            expected_memory(
                "store_episodic",
                "episodic",
                f"On {date}, the user {event}.",
                ["event", f"event_{index}"],
                "User",
                "completed_event",
                str(index),
                f"On {date}, I {event}.",
                "Dated episodic event.",
                temporal_expression=date,
                resolved_time=date,
                decay_rate=0.05,
            ),
        ),
        row(
            f"fact-drill-today-{index}",
            _base_input(f"User: Today I {event}.", index, source_timestamp=f"{date}T14:00:00Z"),
            expected_memory(
                "store_episodic",
                "episodic",
                f"Today the user {event}.",
                ["event", f"today_{index}"],
                "User",
                "completed_event",
                str(index),
                f"Today I {event}.",
                "Same-day episodic memory.",
                temporal_expression=date,
                resolved_time=date,
                decay_rate=0.05,
            ),
        ),
    ]


def build_fact_format_drills(*, variants: int = 200) -> list[dict[str, Any]]:
    drills: list[dict[str, Any]] = []
    for index in range(variants):
        drills.extend(_templates(index))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in drills:
        key = hashlib.sha256(json.dumps(item["expected"], sort_keys=True).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build fact-format parse drill JSONL for Gate 4.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", type=int, default=200)
    args = parser.parse_args()
    rows = build_fact_format_drills(variants=args.variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
