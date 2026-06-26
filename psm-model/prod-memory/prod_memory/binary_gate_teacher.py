from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from prod_memory.curriculum_sources import _ignore_expected, remember_input
from prod_memory.hf_prompts import storage_inference_messages
from prod_memory.openrouter_teacher import TeacherConfig

DEFAULT_BINARY_MODEL = "google/gemma-4-31b-it"

NOISE_GEN_PROMPT = """You generate training data for a memory gate classifier.
Return a JSON array of exactly {count} strings. Each string is one assistant reply that must be labeled IGNORE
(no durable facts, procedures, or preferences worth remembering).

Include diverse categories:
- short acknowledgments and filler
- greetings and closings
- meta statements about having nothing to store
- empty progress updates with no concrete facts
- polite deferrals without new information

Each string 20-200 characters. No markdown. No numbering. JSON array only."""


@dataclass(frozen=True)
class BinaryLabel:
    text: str
    label: str  # ignore | store
    model: str
    raw: str


def _chat(config: TeacherConfig, *, model: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    return str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")


def parse_binary_line(raw: str) -> str:
    for part in raw.splitlines():
        stripped = part.strip().lower()
        if stripped:
            return "store" if stripped == "store" else "ignore"
    return "ignore"


def classify_binary(
    llm_response: str,
    *,
    config: TeacherConfig | None = None,
    model: str | None = None,
) -> BinaryLabel:
    cfg = config or TeacherConfig.from_env(model=model or DEFAULT_BINARY_MODEL)
    if not cfg.api_key:
        raise ValueError("OPENROUTER_API_KEY required for binary gate teacher")
    use_model = model or cfg.model
    msgs = storage_inference_messages(llm_response, output_format="binary")
    raw = _chat(cfg, model=use_model, messages=msgs, max_tokens=16)
    return BinaryLabel(text=llm_response, label=parse_binary_line(raw), model=use_model, raw=raw.strip())


def generate_noise_variants(
    *,
    count: int = 30,
    config: TeacherConfig | None = None,
    model: str | None = None,
) -> list[str]:
    cfg = config or TeacherConfig.from_env(model=model or DEFAULT_BINARY_MODEL)
    if not cfg.api_key:
        raise ValueError("OPENROUTER_API_KEY required for noise generation")
    use_model = model or cfg.model
    prompt = NOISE_GEN_PROMPT.format(count=count)
    raw = _chat(
        cfg,
        model=use_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
    )
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    variants = json.loads(text)
    if not isinstance(variants, list):
        raise ValueError("noise generator did not return a JSON array")
    out: list[str] = []
    for item in variants:
        s = str(item).strip()
        if len(s) >= 12:
            out.append(s)
    return out[:count]


def binary_ignore_row(row_id: str, llm_response: str, *, source: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "input": remember_input(llm_response, source_id=row_id, source_kind=source),
        "expected": _ignore_expected("No durable memory."),
        "source": source,
    }


def binary_store_row(row_id: str, llm_response: str, *, source: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "input": remember_input(llm_response, source_id=row_id, source_kind=source),
        "expected": {
            "action": "store_episodic",
            "memory": {"content": "classify-store", "type": "episodic"},
            "facts": [],
            "indexables": [],
            "reasoning": "Durable information present.",
        },
        "source": source,
    }
