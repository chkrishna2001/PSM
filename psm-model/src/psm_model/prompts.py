from __future__ import annotations

import json
from typing import Any

from psm_model.lean_format import encode_at_tag_decision, encode_tagged_decision


JSON_SYSTEM_INSTRUCTION = """You are the PSM storage model.
Return one strict JSON object compatible with the PSM StorageDecision schema.
Do not include markdown, prose, comments, or fallback text outside JSON.
Facts must be explicit and supported by evidence_text from the current input."""


TAGGED_SYSTEM_INSTRUCTION = """You are the PSM storage model.
Return one strict tagged StorageDecision.
Use only these line prefixes: A:, M:, T:, C:, Q:, G:, TE:, RT:, F:, R:, END.
Do not include markdown, prose, comments, JSON, or fallback text outside the tagged decision.
Facts must be explicit and supported by evidence_text from the current input."""


AT_TAG_SYSTEM_INSTRUCTION = """You are the PSM storage model.
Return one strict at-tag StorageDecision.
Use only these line prefixes: @a, @m, @t, @c, @s, @d, @e, @p, @g, @te, @rt, @f, @ef, @r, @end.
Do not include markdown, prose, comments, JSON, or fallback text outside the at-tag decision.
Facts must be explicit and supported by evidence_text from the current input."""


ACTION_SYSTEM_INSTRUCTION = """You are the PSM storage model action selector.
Return only the storage action in this strict format:
A:<action>
END
Choose from: ignore, store_episodic, promote_semantic, update_existing, flag_conflict, flag_and_store."""

RECALL_SYSTEM_INSTRUCTION = """You are the PSM memory planner.
Return one strict JSON recall plan object with intent, target_tables, filters, ranking_hints, temporal_intent, and top_k.
Choose memory tiers from episodic, semantic, and archival. Do not answer the user."""


def row_task(input_payload: dict[str, Any]) -> str:
    operation = input_payload.get("operation")
    if operation == "recall_plan":
        return "recall_plan"
    if operation == "context_plan":
        return "context_plan"
    return "storage"


def row_output_format(input_payload: dict[str, Any], *, default: str = "tagged") -> str:
    if row_task(input_payload) in {"recall_plan", "context_plan"}:
        return "json"
    return default


def render_recall_plan_prompt(input_payload: dict[str, Any]) -> str:
    payload = json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
    return (
        "<|system|>\n"
        f"{RECALL_SYSTEM_INSTRUCTION}\n"
        "<|user|>\n"
        "Create a recall plan as JSON only with intent, target_tables, filters, ranking_hints, temporal_intent, and top_k. "
        "PSM owns memory planning: choose the memory tiers that should be searched, but do not answer the user.\n"
        f"{payload}\n"
        "<|assistant|>\n"
    )


def render_context_plan_prompt(input_payload: dict[str, Any]) -> str:
    payload = json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
    return (
        "<|system|>\n"
        f"{RECALL_SYSTEM_INSTRUCTION}\n"
        "<|user|>\n"
        "Create a memory context recall plan as JSON only with intent, target_tables, filters, ranking_hints, temporal_intent, and top_k. "
        "PSM owns memory planning: choose the memory tiers that should be searched, but do not answer the user.\n"
        f"{payload}\n"
        "<|assistant|>\n"
    )


def render_row_prompt(input_payload: dict[str, Any], *, output_format: str = "tagged") -> str:
    task = row_task(input_payload)
    if task == "recall_plan":
        return render_recall_plan_prompt(input_payload)
    if task == "context_plan":
        return render_context_plan_prompt(input_payload)
    return render_storage_prompt(input_payload, output_format=output_format)


def render_storage_prompt(input_payload: dict[str, Any], *, output_format: str = "tagged") -> str:
    payload = json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
    instruction = _system_instruction(output_format)
    output_name = _output_name(output_format)
    return (
        "<|system|>\n"
        f"{instruction}\n"
        "<|user|>\n"
        f"Analyze this input and produce the PSM storage {output_name}.\n"
        f"{payload}\n"
        "<|assistant|>\n"
    )


def render_training_text(input_payload: dict[str, Any], expected_output: dict[str, Any], *, output_format: str = "tagged") -> str:
    active_format = row_output_format(input_payload, default=output_format)
    prompt = render_row_prompt(input_payload, output_format=active_format)
    if row_task(input_payload) == "storage":
        output = render_expected_output(expected_output, output_format=active_format)
    else:
        output = json.dumps(expected_output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prompt}{output}<|end|>"


def render_expected_output(expected_output: dict[str, Any], *, output_format: str = "tagged") -> str:
    if output_format == "json":
        return json.dumps(expected_output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if output_format == "tagged":
        return encode_tagged_decision(expected_output)
    if output_format == "at_tag":
        return encode_at_tag_decision(expected_output)
    if output_format == "action":
        return f"A:{expected_output['action']}\nEND"
    raise ValueError(f"unsupported output format: {output_format}")


def _system_instruction(output_format: str) -> str:
    if output_format == "json":
        return JSON_SYSTEM_INSTRUCTION
    if output_format == "tagged":
        return TAGGED_SYSTEM_INSTRUCTION
    if output_format == "at_tag":
        return AT_TAG_SYSTEM_INSTRUCTION
    if output_format == "action":
        return ACTION_SYSTEM_INSTRUCTION
    raise ValueError(f"unsupported output format: {output_format}")


def _output_name(output_format: str) -> str:
    if output_format == "json":
        return "JSON"
    if output_format == "tagged":
        return "tagged DSL"
    if output_format == "at_tag":
        return "at-tag DSL"
    if output_format == "action":
        return "action-only DSL"
    raise ValueError(f"unsupported output format: {output_format}")
