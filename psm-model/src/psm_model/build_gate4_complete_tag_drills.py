from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from psm_model.build_gate4_fact_format_drills import build_fact_format_drills
from psm_model.data import validate_training_row
from psm_model.generate_action_foundation_curriculum import expected_memory, row


def _base_input(conversation: str, index: int, *, source_timestamp: str | None = None) -> dict[str, Any]:
    return {
        "conversation": conversation,
        "operation": "remember",
        "source_id": f"complete-tag-drill-{index}",
        "source_kind": "parse_drill",
        "source_timestamp": source_timestamp or "2026-06-03T12:00:00Z",
    }


def _complete_promote(index: int, *, conversation: str, content: str, tags: list[str], predicate: str, value: str, evidence: str, reasoning: str) -> dict[str, Any]:
    return row(
        f"complete-tag-promote-{index}",
        _base_input(conversation, index),
        expected_memory(
            "promote_semantic",
            "semantic",
            content,
            tags,
            "User",
            predicate,
            value,
            evidence,
            reasoning,
        ),
    )


def _complete_store(index: int, *, conversation: str, content: str, tags: list[str], predicate: str, value: str, evidence: str, reasoning: str, date: str) -> dict[str, Any]:
    return row(
        f"complete-tag-store-{index}",
        _base_input(conversation, index, source_timestamp=f"{date}T12:00:00Z"),
        expected_memory(
            "store_episodic",
            "episodic",
            content,
            tags,
            "User",
            predicate,
            value,
            evidence,
            reasoning,
            temporal_expression=date,
            resolved_time=date,
            decay_rate=0.05,
        ),
    )


def build_complete_tag_drills(*, variants: int = 120) -> list[dict[str, Any]]:
    """Drills for full tagged output: F lines + Q numeric line + non-empty R + END."""
    templates = [
        lambda i: _complete_promote(
            i,
            conversation=f"User: I prefer checkpoint gates before long training runs.",
            content="The user prefers checkpoint gates before long training runs.",
            tags=["preference", "training_process"],
            predicate="prefers",
            value="checkpoint_gates_first",
            evidence="I prefer checkpoint gates before long training runs.",
            reasoning="The user stated a durable training preference with explicit evidence.",
        ),
        lambda i: _complete_promote(
            i,
            conversation=f"User: For API responses, always return tagged StorageDecision output, never markdown.",
            content="For API responses, always return tagged StorageDecision output, never markdown.",
            tags=["rule", "output_format"],
            predicate="requires",
            value="tagged_storage_decision",
            evidence="For API responses, always return tagged StorageDecision output, never markdown.",
            reasoning="The user stated a durable output-format rule.",
        ),
        lambda i: _complete_store(
            i,
            conversation=f"User: On 2026-06-08, I finished the Gate 4 v3 training run.",
            content="On 2026-06-08, the user finished the Gate 4 v3 training run.",
            tags=["event", "gate4_train"],
            predicate="completed",
            value="gate4_v3_train",
            evidence="On 2026-06-08, I finished the Gate 4 v3 training run.",
            reasoning="The user reported a dated episodic event.",
            date="2026-06-08",
        ),
    ]
    drills: list[dict[str, Any]] = []
    for index in range(variants):
        for template in templates:
            drills.append(template(index))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in drills:
        key = hashlib.sha256(json.dumps(item["expected"], sort_keys=True).encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        _, issues = validate_training_row(item)
        if issues:
            raise ValueError(f"invalid drill {item['id']}: {issues}")
        unique.append(item)
    return unique


def build_gate4_complete_tag_drills(*, variants: int = 120, include_fact_drills: bool = True) -> list[dict[str, Any]]:
    rows = build_complete_tag_drills(variants=variants)
    if include_fact_drills:
        rows.extend(build_fact_format_drills(variants=max(40, variants // 3)))
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build complete tagged-output drills (F + Q + R + END).")
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", type=int, default=120)
    args = parser.parse_args()
    rows = build_gate4_complete_tag_drills(variants=args.variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
