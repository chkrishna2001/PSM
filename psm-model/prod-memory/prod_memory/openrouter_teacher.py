from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from prod_memory.curriculum_sources import _ignore_expected, _store_expected, remember_input
from prod_memory.grounding import has_curriculum_bleed
from prod_memory.indexable_labels import build_indexable_labels
from prod_memory.label_from_assistant import (
    _labels_invalid,
    _make_fact,
    _substring_grounded,
    build_expected_from_assistant,
    should_ignore_assistant_text,
)

DEFAULT_MODEL = "google/gemma-4-31b-it"
FALLBACK_MODEL = "z-ai/glm-5.2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
ALLOWED_ACTIONS = {"ignore", "store_episodic", "promote_semantic"}

SYSTEM_PROMPT = """You are a strict PSM production-memory training labeler.
Return one complete JSON object only. No markdown fences. No commentary. Start with { and end with }.

Task: label remember({ llmResponse }) — extract durable memory from assistant text only.
The model must learn grounded extraction, not template memorization.

Allowed actions: ignore, store_episodic, promote_semantic.
Use ignore for transient progress, greetings, raw command logs, or text with no durable value.
Use promote_semantic for reusable rules, preferences, architecture decisions, or product constraints.
Use store_episodic for concrete milestones, completed outcomes, handoffs, or session-specific durable facts.

Rules:
- Do not invent facts. memory.content and every fact must be grounded in verbatim spans from llm_response.
- evidence_text must be an exact substring (or whitespace-normalized match) from llm_response.
- memory.content must be concise (under 480 chars), not a raw transcript.
- Return at most 4 facts. Each fact needs subject, predicate (snake_case), value, evidence_text.
- Return indexables: [] — indexables are built locally.
- Avoid curriculum bleed tokens in labels: runpod, gate6, direct probe, nvidia-smi, token budget, expanded probe.
- Prefer ignore over weak storage.

Return exactly:
{"action":"ignore|store_episodic|promote_semantic","memory":null|{"content":"...","type":"episodic|semantic","confidence":0.9,"tags":["..."]},"facts":[{"subject":"...","predicate":"snake_case","value":"...","evidence_text":"..."}],"reasoning":"short reason"}"""


@dataclass(frozen=True)
class TeacherConfig:
    model: str = DEFAULT_MODEL
    fallback_model: str = FALLBACK_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    request_delay_ms: int = 300
    request_max_retries: int = 6
    request_timeout_s: int = 90
    max_tokens: int = 900

    @classmethod
    def from_env(cls, *, model: str | None = None) -> TeacherConfig:
        return cls(
            model=model or os.environ.get("PROD_TEACHER_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
            base_url=os.environ.get("OPENROUTER_BASE_URL", os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)),
            request_delay_ms=int(os.environ.get("PROD_TEACHER_DELAY_MS", "1800")),
            request_max_retries=int(os.environ.get("PROD_TEACHER_MAX_RETRIES", "6")),
        )


def build_expected_from_teacher(
    llm_response: str,
    teacher_output: dict[str, Any] | None,
    *,
    parse_error: str | None = None,
) -> dict[str, Any]:
    if parse_error or not teacher_output:
        return _ignore_expected(parse_error or "Teacher response could not be parsed.")

    action = str(teacher_output.get("action") or "ignore").strip()
    if action not in ALLOWED_ACTIONS:
        action = "ignore"
    reasoning = str(teacher_output.get("reasoning") or "Teacher labeled assistant response.").strip()

    if action == "ignore":
        return _ignore_expected(reasoning)

    memory_payload = teacher_output.get("memory")
    if not isinstance(memory_payload, dict):
        return _ignore_expected("Teacher chose store but returned no memory object.")

    content = _clean_text(memory_payload.get("content"))
    if len(content) < 12:
        return _ignore_expected("Teacher memory content too short.")

    memory_type = str(memory_payload.get("type") or ("semantic" if action == "promote_semantic" else "episodic"))
    if memory_type not in {"episodic", "semantic"}:
        memory_type = "episodic" if action == "store_episodic" else "semantic"

    tags = _normalize_tags(memory_payload.get("tags"))
    facts = _normalize_teacher_facts(teacher_output.get("facts"), llm_response)
    expected = _store_expected(
        llm_response,
        content,
        tags=tags or ["assistant_handoff"],
        reasoning=reasoning,
        facts=facts,
        memory_type=memory_type,
    )
    if _labels_invalid(llm_response, expected):
        return _ignore_expected("Teacher labels failed prod grounding validation.")
    return expected


def label_assistant_with_teacher(
    text: str,
    *,
    config: TeacherConfig,
    use_heuristic_fallback: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {"model": config.model, "fallback_used": False}
    if should_ignore_assistant_text(text):
        return _ignore_expected("Short or low-signal assistant text; nothing durable to store."), meta

    raw_response = ""
    parse_error: str | None = None
    teacher_output: dict[str, Any] | None = None
    models = [config.model]
    if config.fallback_model and config.fallback_model != config.model:
        models.append(config.fallback_model)

    for model_index, model in enumerate(models):
        try:
            raw_response = _chat_completion(
                config,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "operation": "remember_llm_response",
                                "llm_response": _compact_for_teacher(text),
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            teacher_output, parse_error = _parse_teacher_json(raw_response)
            meta["model"] = model
            meta["fallback_used"] = model_index > 0
            if not parse_error:
                break
        except Exception as exc:
            parse_error = str(exc)
            meta["error"] = parse_error

    meta["raw_response"] = raw_response[:4000] if raw_response else ""
    meta["parse_error"] = parse_error

    expected = build_expected_from_teacher(text, teacher_output, parse_error=parse_error)
    if expected.get("action") != "ignore":
        return expected, meta

    if use_heuristic_fallback:
        heuristic = build_expected_from_assistant(text)
        if heuristic is not None:
            meta["fallback"] = "heuristic"
            return heuristic, meta

    return expected, meta


def build_row_from_teacher(
    text: str,
    *,
    row_id: str,
    source_id: str,
    source_kind: str,
    config: TeacherConfig,
    use_heuristic_fallback: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    expected, meta = label_assistant_with_teacher(
        text,
        config=config,
        use_heuristic_fallback=use_heuristic_fallback,
    )
    if expected is None:
        return None, meta
    return {
        "id": row_id,
        "input": remember_input(text, source_id=source_id, source_kind=source_kind),
        "expected": expected,
        "source": f"prod_extraction_v3_teacher:{source_kind}",
    }, meta


def _chat_completion(config: TeacherConfig, *, model: str, messages: list[dict[str, str]]) -> str:
    if not config.api_key:
        raise ValueError("OPENROUTER_API_KEY is required for teacher labeling.")

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": config.max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    last_error = ""
    for attempt in range(config.request_max_retries + 1):
        if config.request_delay_ms > 0:
            time.sleep(config.request_delay_ms / 1000)
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.request_timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {err_body[:500]}"
            if exc.code != 429 or attempt >= config.request_max_retries:
                break
            time.sleep(min(120, 4 * (2**attempt)))
        except Exception as exc:
            last_error = str(exc)
            if attempt >= config.request_max_retries:
                break
            time.sleep(min(60, 2 * (2**attempt)))

    raise RuntimeError(last_error or "OpenRouter chat completion failed")


def _parse_teacher_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    trimmed = raw.strip()
    if not trimmed:
        return None, "empty teacher response"
    for candidate in reversed(_complete_json_objects(trimmed)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and parsed.get("action"):
                return parsed, None
        except json.JSONDecodeError:
            continue
    return None, "no complete JSON object with action"


def _complete_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start = -1
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objects.append(text[start : index + 1])
                start = -1
    return objects


def _normalize_teacher_facts(value: Any, source_text: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    facts: list[dict[str, Any]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        subject = _clean_text(item.get("subject"))
        predicate = _snake_case(item.get("predicate") or "states")
        fact_value = _clean_text(item.get("value"))
        evidence = _clean_text(item.get("evidence_text") or fact_value)
        if not subject or not predicate or not fact_value or not evidence:
            continue
        if has_curriculum_bleed(f"{fact_value} {evidence}"):
            continue
        if not _substring_grounded(source_text, evidence) and not _substring_grounded(source_text, fact_value):
            continue
        grounded_evidence = evidence if _substring_grounded(source_text, evidence) else fact_value
        facts.append(_make_fact(
            subject=subject,
            predicate=predicate,
            value=fact_value,
            evidence_text=grounded_evidence,
            confidence=0.86,
        ))
    return facts


def _normalize_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    for item in value:
        tag = re.sub(r"[^a-z0-9_]+", "_", str(item).strip().lower()).strip("_")
        if tag:
            tags.append(tag)
    return tags[:6]


def _compact_for_teacher(text: str, *, max_len: int = 6000) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    compact = re.sub(r"hf_[A-Za-z0-9_=-]{16,}", "[REDACTED_HF_TOKEN]", compact)
    compact = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED_API_KEY]", compact)
    if len(compact) > max_len:
        compact = compact[: max_len - 3].rsplit(" ", 1)[0] + "..."
    return compact


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _snake_case(value: Any) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return key if re.match(r"^[a-z]", key) else "states"
