from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTIONS = [
    "ignore",
    "store_episodic",
    "promote_semantic",
    "update_existing",
    "flag_conflict",
    "flag_and_store",
    "recall_context",
]

MEMORY_TYPES = ["none", "episodic", "semantic"]


@dataclass(frozen=True)
class TrainingExample:
    id: str
    instruction: str
    input: dict[str, Any]
    output: dict[str, Any]


def action_to_id(action: str) -> int:
    try:
        return ACTIONS.index(action)
    except ValueError as exc:
        raise ValueError(f"Unsupported action: {action}") from exc


def memory_type_to_id(memory_type: str | None) -> int:
    if not memory_type:
        return 0
    try:
        return MEMORY_TYPES.index(memory_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported memory type: {memory_type}") from exc

