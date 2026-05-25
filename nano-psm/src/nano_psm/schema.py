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

ACTION_ALIASES = {
    "detect_interference": "flag_conflict",
    "flag_and_update": "flag_and_store",
    "flag_contradiction": "flag_conflict",
    "ignore_noise": "ignore",
    "merge_results": "recall_context",
    "promote": "promote_semantic",
    "recall": "recall_context",
    "retrieve_plan": "recall_context",
    "store": "store_episodic",
    "store_episodic_with_emotional_weighting": "store_episodic",
    "update": "update_existing",
}

MEMORY_TYPE_ALIASES = {
    "": "none",
    "null": "none",
    "none": "none",
    "fact": "semantic",
    "profile": "semantic",
    "event": "episodic",
}


@dataclass(frozen=True)
class TrainingExample:
    id: str
    instruction: str
    input: dict[str, Any]
    output: dict[str, Any]


def normalize_action(action: str | None) -> str:
    normalized = str(action or "").strip().lower()
    return ACTION_ALIASES.get(normalized, normalized)


def action_to_id(action: str) -> int:
    normalized = normalize_action(action)
    try:
        return ACTIONS.index(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported action: {action} (normalized: {normalized})") from exc


def memory_type_to_id(memory_type: str | None) -> int:
    normalized = MEMORY_TYPE_ALIASES.get(str(memory_type or "").strip().lower(), str(memory_type or "").strip().lower())
    if not normalized:
        normalized = "none"
    try:
        return MEMORY_TYPES.index(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported memory type: {memory_type}") from exc
