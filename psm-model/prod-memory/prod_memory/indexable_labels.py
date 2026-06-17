from __future__ import annotations

import re
from typing import Any

WORKFLOW_HEADER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"review.*pull request|pull request review", re.IGNORECASE), "review-pr"),
    (re.compile(r"runpod.*train|gpu train|two-phase launch", re.IGNORECASE), "runpod-gpu-train"),
    (re.compile(r"grounding bar|promotion bar", re.IGNORECASE), "grounding-bar"),
]


def infer_workflow_key(text: str, tags: list[str] | None = None) -> str | None:
    for tag in tags or []:
        match = re.match(r"^workflow:([a-z0-9-]+)$", str(tag), re.IGNORECASE)
        if match:
            return match.group(1).lower()
    header = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    haystack = f"{header.group(1) if header else ''}\n{text[:240]}"
    for pattern, key in WORKFLOW_HEADER_PATTERNS:
        if pattern.search(haystack):
            return key
    return None


def extract_workflow_steps(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*\d+\.\s+(.+?)\s*$", line)
        if match and match.group(1):
            steps.append(_step_to_id(match.group(1)))
    return _unique(steps)


def build_indexable_labels(
    *,
    llm_response: str,
    memory_content: str,
    tags: list[str] | None = None,
    facts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    workflow_key = infer_workflow_key(llm_response, tags)
    steps = extract_workflow_steps(llm_response)
    if workflow_key and len(steps) >= 2:
        return [{
            "kind": "workflow",
            "key": workflow_key,
            "steps": steps,
            "salience": 0.95,
            "reconstructive_hint": _reconstructive_hint(memory_content),
            "evidence_text": llm_response[:500],
            "tags": _unique_tags([f"workflow:{workflow_key}", "workflow", *(tags or [])]),
        }]

    mnemonic_key = _mnemonic_key(memory_content, tags or [])
    rows: list[dict[str, Any]] = [{
        "kind": "mnemonic",
        "key": mnemonic_key,
        "salience": _salience_for(memory_content, tags or []),
        "reconstructive_hint": _reconstructive_hint(memory_content),
        "evidence_text": memory_content,
        "tags": _unique_tags(tags or [])[:6],
    }]
    fact_key = _fact_anchor_key(facts or [])
    if fact_key and fact_key != mnemonic_key:
        rows.append({
            "kind": "fact_anchor",
            "key": fact_key,
            "salience": max(rows[0]["salience"], 0.82),
            "reconstructive_hint": _reconstructive_hint(memory_content),
            "evidence_text": memory_content,
            "tags": _unique_tags(tags or [])[:6],
        })
    return rows


def _step_to_id(step: str) -> str:
    cleaned = re.sub(r"`([^`]+)`", r"\1", step)
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    return slug[:48] or "step"


def _mnemonic_key(content: str, tags: list[str]) -> str:
    tokens = _unique(_meaningful_tokens(" ".join(tags)) + _meaningful_tokens(content))[:4]
    return "-".join(tokens) if tokens else "memory-anchor"


def _fact_anchor_key(facts: list[dict[str, Any]]) -> str:
    for fact in facts:
        subject = str(fact.get("subject") or "").strip()
        predicate = str(fact.get("predicate") or "").strip()
        value = str(fact.get("value_text") or fact.get("value") or "").strip()
        if subject and predicate and value:
            tokens = _unique(
                _meaningful_tokens(subject) + _meaningful_tokens(predicate) + _meaningful_tokens(value)
            )[:4]
            if tokens:
                return "-".join(tokens)
    return ""


def _salience_for(content: str, tags: list[str]) -> float:
    lower = content.lower()
    score = 0.68
    if re.search(r"\b\d{4}\b|yesterday|workflow|review|procedure", lower):
        score += 0.08
    if re.search(r"decision|prefer|constraint|indexable|mnemonic|recall", lower):
        score += 0.12
    if tags:
        score += 0.04
    return round(min(score, 0.98), 2)


def _reconstructive_hint(content: str) -> str:
    match = re.match(r"^(.+?[.!?])(?:\s|$)", content, re.DOTALL)
    sentence = match.group(1) if match else content
    return sentence if len(sentence) <= 160 else f"{sentence[:157].rstrip()}..."


def _meaningful_tokens(text: str) -> list[str]:
    stop = {"the", "and", "for", "that", "this", "with", "from", "into", "said", "user", "memory"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop
    ]


def _unique_tags(tags: list[str]) -> list[str]:
    return _unique(tag.strip().replace(" ", "_") for tag in tags if str(tag).strip())


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
