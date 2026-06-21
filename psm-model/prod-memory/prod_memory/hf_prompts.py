from __future__ import annotations

import json
from typing import Any

from psm_model.prompts import (
    RECALL_SYSTEM_INSTRUCTION,
    TAGGED_SYSTEM_INSTRUCTION,
    JSON_SYSTEM_INSTRUCTION,
    render_expected_output,
    row_task,
)

from prod_memory.row_validation import remember_target_from_input

PROD_STORAGE_USER_PREFIX = (
    "Extract durable memory from the assistant response below.\n"
    "Choose ignore, store_episodic, or promote_semantic.\n"
    "When storing, emit grounded memory.content, facts[], and indexables[] from the text.\n\n"
    "Assistant response:\n"
)


def compact_storage_json(decision: dict[str, Any]) -> str:
    payload = {
        "action": decision["action"],
        "memory": decision.get("memory"),
        "facts": decision.get("facts") or [],
        "indexables": decision.get("indexables") or [],
        "reasoning": decision["reasoning"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_storage_assistant_output(expected: dict[str, Any], *, output_format: str) -> str:
    if output_format == "json":
        return compact_storage_json(expected)
    return render_expected_output(expected, output_format=output_format)


def row_messages(
    row: dict[str, Any],
    *,
    output_format: str = "tagged",
) -> list[dict[str, str]]:
    input_payload = row["input"]
    expected = row["expected"]
    task = row_task(input_payload)

    if task in {"recall_plan", "context_plan"}:
        system = RECALL_SYSTEM_INSTRUCTION
        user = (
            "Create a recall plan as JSON only with intent, target_tables, filters, ranking_hints, "
            "temporal_intent, and top_k. PSM owns memory planning; do not answer the user.\n"
            + json.dumps(input_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        assistant = json.dumps(expected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]

    llm_response = remember_target_from_input(input_payload)
    system = JSON_SYSTEM_INSTRUCTION if output_format == "json" else TAGGED_SYSTEM_INSTRUCTION
    user = f"{PROD_STORAGE_USER_PREFIX}{llm_response.strip()}"
    assistant = render_storage_assistant_output(expected, output_format=output_format)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def apply_chat_text(messages: list[dict[str, str]], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def apply_chat_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    prompt_messages = messages[:-1]
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
