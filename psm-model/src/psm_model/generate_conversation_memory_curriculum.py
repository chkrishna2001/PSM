from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.data import validate_training_row
from psm_model.schema import validate_storage_decision

SPEAKERS = ["Caroline", "Melanie", "Avery", "Jordan", "Riley", "Sam"]
EVENT_TEMPLATES = [
    ("went to a LGBTQ support group yesterday and found it powerful", ["event", "lgbtq"], "yesterday"),
    ("painted a sunrise over a lake last weekend", ["event", "art"], "last weekend"),
    ("started counseling classes this month", ["education", "career"], "this month"),
    ("volunteered at an animal shelter on Saturday", ["event", "volunteering"], "on Saturday"),
]
PREFERENCE_TEMPLATES = [
    ("prefers concise weekly planning checklists", ["preference", "planning"]),
    ("wants direct feedback in code reviews", ["preference", "feedback"]),
    ("likes to keep commits small and focused", ["preference", "workflow"]),
]
NOISE_TEMPLATES = [
    "said hi and asked how the day was going",
    "asked if there were any updates with no new details",
    "said sounds good and thanked everyone",
]
CONFLICT_PREFERENCES = [
    ("prefers async updates in chat", "prefers a short live sync each morning"),
    ("prefers dark mode in all apps", "uses light mode during daytime"),
]
IMAGE_TEMPLATES = [
    (
        "said the transgender stories were inspiring",
        "transgender pride flag mural",
        "a photo of a dog walking past a wall mural",
    ),
    (
        "said she took a picture of her hiking trail yesterday",
        "mountain trail photo",
        "a wide trail at sunset with pine trees",
    ),
]
CURRICULUM_BLEED_TOKENS = ("checkpoint", "PowerShell", "gate datasets", "nvidia-smi", "StorageDecision")


def _base_input(conversation: str, index: int, *, source_timestamp: str = "2026-06-16T12:00:00Z") -> dict[str, Any]:
    return {
        "conversation": conversation,
        "operation": "remember",
        "source_kind": "synthetic_dialogue",
        "source_id": f"session-{index // 10}:turn-{index}",
        "source_timestamp": source_timestamp,
    }


def _memory(
    *,
    content: str,
    memory_type: str,
    tags: list[str],
    temporal_expression: str | None = None,
    resolved_time: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "content": content,
        "type": memory_type,
        "strength": 0.86,
        "decay_rate": 0.02,
        "emotional_weight": 0.30 if memory_type == "episodic" else 0.18,
        "confidence": 0.92,
        "tags": tags,
    }
    if temporal_expression:
        payload["temporal_expression"] = temporal_expression
    if resolved_time:
        payload["resolved_time"] = resolved_time
    return payload


def _validate_row(row: dict[str, Any]) -> None:
    _, issues = validate_training_row(row)
    if issues:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
        raise ValueError(f"{row['id']}: {formatted}")
    result = validate_storage_decision(row["expected"])
    if not result.ok:
        raise ValueError(f"{row['id']}: invalid storage decision")


def _filter_row(row: dict[str, Any]) -> bool:
    conversation = str(row.get("input", {}).get("conversation", ""))
    if not conversation.startswith("User: "):
        return False
    expected = row.get("expected", {})
    memory = expected.get("memory")
    if isinstance(memory, dict):
        content = str(memory.get("content", ""))
        if len(content) > 200:
            return False
        lowered = content.lower()
        if any(token.lower() in lowered for token in CURRICULUM_BLEED_TOKENS):
            return False
    return True


def build_rows(count: int, *, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    targets = ["store_episodic"] * 40 + ["ignore"] * 35 + ["promote_semantic"] * 15 + ["flag_conflict"] * 10
    for index in range(count):
        action = targets[index % len(targets)]
        speaker = SPEAKERS[index % len(SPEAKERS)]
        row_id = f"conv-mem-{action}-{index}"
        input_payload = _base_input("", index)
        expected: dict[str, Any]

        if action == "store_episodic":
            event, tags, temporal = EVENT_TEMPLATES[index % len(EVENT_TEMPLATES)]
            conversation = f'User: {speaker} said "{event}."'
            content = f"{speaker} {event}."
            input_payload["conversation"] = conversation
            expected = {
                "action": "store_episodic",
                "memory": _memory(
                    content=content,
                    memory_type="episodic",
                    tags=tags,
                    temporal_expression=temporal,
                    resolved_time="2026-06-15" if temporal == "yesterday" else None,
                ),
                "facts": [],
                "reasoning": "The speaker described a durable personal event.",
            }
        elif action == "ignore":
            noise = NOISE_TEMPLATES[index % len(NOISE_TEMPLATES)]
            input_payload["conversation"] = f"User: {speaker} {noise}."
            expected = {"action": "ignore", "memory": None, "facts": [], "reasoning": "No durable memory signal."}
        elif action == "promote_semantic":
            pref, tags = PREFERENCE_TEMPLATES[index % len(PREFERENCE_TEMPLATES)]
            input_payload["conversation"] = f"User: {speaker} said she {pref}."
            expected = {
                "action": "promote_semantic",
                "memory": _memory(
                    content=f"{speaker} {pref}.",
                    memory_type="semantic",
                    tags=tags,
                ),
                "facts": [],
                "reasoning": "The statement is a stable preference.",
            }
        else:
            old_pref, new_pref = CONFLICT_PREFERENCES[index % len(CONFLICT_PREFERENCES)]
            input_payload["conversation"] = f"User: {speaker} said she changed her mind and now {new_pref}."
            input_payload["context"] = json.dumps(
                {"memory_store": [{"id": "m1", "content": f"{speaker} {old_pref}."}]},
                ensure_ascii=False,
                sort_keys=True,
            )
            expected = {
                "action": "flag_conflict",
                "memory": _memory(
                    content=f"{speaker} now {new_pref}.",
                    memory_type="semantic",
                    tags=["conflict", "preference"],
                ),
                "facts": [],
                "reasoning": "New statement contradicts existing stored preference.",
            }

        if index % 11 == 0:
            event, query, caption = IMAGE_TEMPLATES[index % len(IMAGE_TEMPLATES)]
            input_payload["conversation"] = (
                f'User: {speaker} said "{event}!". Image query: {query}. Image caption: {caption}.'
            )
            expected = {
                "action": "store_episodic",
                "memory": _memory(
                    content=f"{speaker} said {event} and shared an image context.",
                    memory_type="episodic",
                    tags=["event", "image"],
                ),
                "facts": [],
                "reasoning": "Image-augmented utterance still contains a durable personal event.",
            }

        row = {
            "id": row_id,
            "input": input_payload,
            "expected": expected,
            "source": "gate6:synthetic-dialogue",
            "split": "train",
        }
        if _filter_row(row):
            _validate_row(row)
            rows.append(row)
    rng.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic conversation-memory curriculum rows.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("psm-model/data/curriculum/conversation-memory-synthetic-v1.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_rows(args.rows, seed=args.seed)
    write_jsonl(args.out, rows)
    action_counts = Counter(row["expected"]["action"] for row in rows)
    avg_memory_len = (
        sum(len(row["expected"]["memory"]["content"]) for row in rows if isinstance(row["expected"].get("memory"), dict))
        / max(1, sum(1 for row in rows if isinstance(row["expected"].get("memory"), dict)))
    )
    summary = {
        "rows_requested": args.rows,
        "rows_written": len(rows),
        "output": str(args.out),
        "seed": args.seed,
        "action_counts": dict(sorted(action_counts.items())),
        "avg_memory_content_length": round(avg_memory_len, 2),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
