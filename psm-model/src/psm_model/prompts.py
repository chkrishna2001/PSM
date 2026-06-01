from __future__ import annotations

import json
from typing import Any

from psm_model.lean_format import encode_at_tag_decision, encode_tagged_decision


SYSTEM_INSTRUCTION = """You are the PSM storage model.
Return one strict JSON object compatible with the PSM StorageDecision schema.
Do not include markdown, prose, comments, or fallback text outside JSON.
Facts must be explicit and supported by evidence_text from the current input."""


def render_storage_prompt(input_payload: dict[str, Any]) -> str:
    payload = json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
    return (
        "<|system|>\n"
        f"{SYSTEM_INSTRUCTION}\n"
        "<|user|>\n"
        "Analyze this input and produce the PSM storage JSON.\n"
        f"{payload}\n"
        "<|assistant|>\n"
    )


def render_training_text(input_payload: dict[str, Any], expected_output: dict[str, Any], *, output_format: str = "tagged") -> str:
    output = render_expected_output(expected_output, output_format=output_format)
    return f"{render_storage_prompt(input_payload)}{output}<|end|>"


def render_expected_output(expected_output: dict[str, Any], *, output_format: str = "tagged") -> str:
    if output_format == "json":
        return json.dumps(expected_output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if output_format == "tagged":
        return encode_tagged_decision(expected_output)
    if output_format == "at_tag":
        return encode_at_tag_decision(expected_output)
    raise ValueError(f"unsupported output format: {output_format}")
