from __future__ import annotations

import re
from typing import Any

from prod_memory.curriculum_sources import _ignore_expected, _store_expected, remember_input
from prod_memory.grounding import has_curriculum_bleed, is_grounded_in_source
from prod_memory.indexable_labels import build_indexable_labels

IGNORE_MARKERS = (
    "chatgpt can make mistakes",
    "exported from [chatgpt]",
    "check important info",
)
NOISE_ONLY = re.compile(
    r"^(thanks|thank you|ok|okay|sure|got it|sounds good|yes|no|yep|nope)[.!?\s]*$",
    re.IGNORECASE,
)
BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+(.+?)\s*$")
HEADER_LINE = re.compile(r"^#{1,6}\s+")
FACT_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"\buse\s+`([^`]+)`", re.I), "tooling", "uses"),
    (re.compile(r"\bprefer(?:s|red)?\s+(.{8,120}?)(?:[.;]|$)", re.I), "preference", "prefers"),
    (re.compile(r"\b(?:must|should|always)\s+(.{8,120}?)(?:[.;]|$)", re.I), "constraint", "requires"),
    (re.compile(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", re.I), "event", "occurred_on"),
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), "date", "on_date"),
)
WORKFLOW_TAG = re.compile(r"\b(workflow|procedure|process|steps|phase\s+\d+|review)\b", re.I)
PLAN_TAG = re.compile(r"\b(plan|handoff|roadmap|phase\s+\d+|next steps|baseline)\b", re.I)
TECHNICAL_TAG = re.compile(r"\b(api|typescript|eslint|schema|config|module|function|test)\b", re.I)


def should_ignore_assistant_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 80:
        return True
    if NOISE_ONLY.match(stripped):
        return True
    lower = stripped.lower()
    if any(marker in lower for marker in IGNORE_MARKERS):
        return True
    alpha = sum(1 for char in stripped if char.isalpha())
    if alpha < 40:
        return True
    question_lines = sum(1 for line in stripped.splitlines() if line.strip().endswith("?"))
    if question_lines >= 3 and question_lines * 2 >= len(stripped.splitlines()):
        return True
    return False


def extract_memory_content(text: str, *, max_len: int = 520) -> str:
    bullets: list[str] = []
    for line in text.splitlines():
        if HEADER_LINE.match(line):
            header = re.sub(r"^#{1,6}\s+", "", line).strip()
            if len(header) >= 12:
                bullets.append(header)
            continue
        match = BULLET_LINE.match(line)
        if match:
            candidate = _clean_clause(match.group(1))
            if len(candidate) >= 12:
                bullets.append(candidate)
    if bullets:
        content = "; ".join(_unique_preserve(bullets)[:4])
        clipped = _clip_grounded(text, content, max_len=max_len)
        if clipped:
            return clipped

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for paragraph in paragraphs:
        if HEADER_LINE.match(paragraph):
            continue
        sentence = _first_sentence(paragraph)
        if len(sentence) >= 24:
            return _clip_grounded(text, sentence, max_len=max_len)
    return ""


def _make_fact(
    *,
    subject: str,
    predicate: str,
    value: str,
    evidence_text: str,
    confidence: float = 0.84,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "value_text": value,
        "inference_kind": "explicit",
        "confidence": confidence,
        "evidence_text": evidence_text,
    }


def extract_facts(text: str, *, max_facts: int = 4) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pattern, subject, predicate in FACT_PATTERNS:
        for match in pattern.finditer(text):
            value = _clean_clause(match.group(1))
            if len(value) < 4 or value.lower() in seen:
                continue
            if not _substring_grounded(text, value):
                continue
            seen.add(value.lower())
            facts.append(_make_fact(
                subject=subject,
                predicate=predicate,
                value=value,
                evidence_text=value,
                confidence=0.88,
            ))
            if len(facts) >= max_facts:
                return facts
    for line in text.splitlines():
        match = BULLET_LINE.match(line)
        if not match:
            continue
        clause = _clean_clause(match.group(1))
        if len(clause) < 12 or clause.lower() in seen:
            continue
        if not _substring_grounded(text, clause):
            continue
        subject = _subject_from_clause(clause)
        seen.add(clause.lower())
        facts.append(_make_fact(
            subject=subject,
            predicate="states",
            value=clause,
            evidence_text=clause,
            confidence=0.84,
        ))
        if len(facts) >= max_facts:
            break
    if len(facts) < max_facts:
        for sentence in _candidate_sentences(text):
            if len(sentence) < 24 or sentence.lower() in seen:
                continue
            if not re.search(r"`[^`]+`|\b\d{4}-\d{2}-\d{2}\b|\b(use|must|should|prefer)\b", sentence, re.I):
                continue
            if not _substring_grounded(text, sentence):
                continue
            seen.add(sentence.lower())
            facts.append(_make_fact(
                subject=_subject_from_clause(sentence),
                predicate="states",
                value=sentence,
                evidence_text=sentence,
                confidence=0.8,
            ))
            if len(facts) >= max_facts:
                break
    return facts


def infer_tags(text: str) -> list[str]:
    tags: list[str] = []
    if WORKFLOW_TAG.search(text):
        tags.append("workflow")
    if PLAN_TAG.search(text):
        tags.append("plan")
    if TECHNICAL_TAG.search(text):
        tags.append("technical")
    if not tags:
        tags.append("assistant_handoff")
    return tags


def build_expected_from_assistant(text: str) -> dict[str, Any] | None:
    if should_ignore_assistant_text(text):
        return _ignore_expected("Short or low-signal assistant text; nothing durable to store.")

    memory_content = extract_memory_content(text)
    if not memory_content:
        return _ignore_expected("No grounded durable summary could be extracted from assistant text.")

    facts = extract_facts(text)
    tags = infer_tags(text)
    expected = _store_expected(
        text,
        memory_content,
        tags=tags,
        reasoning="Grounded extraction from long assistant response text.",
        facts=facts,
        memory_type="episodic",
    )
    if _labels_invalid(text, expected):
        return None
    return expected


def build_row_from_assistant(
    text: str,
    *,
    row_id: str,
    source_id: str,
    source_kind: str,
) -> dict[str, Any] | None:
    expected = build_expected_from_assistant(text)
    if expected is None:
        return None
    return {
        "id": row_id,
        "input": remember_input(text, source_id=source_id, source_kind=source_kind),
        "expected": expected,
        "source": f"prod_extraction_v2:{source_kind}",
    }


def _labels_invalid(source_text: str, expected: dict[str, Any]) -> bool:
    if str(expected.get("action") or "") == "ignore":
        return False
    memory = expected.get("memory")
    content = str(memory.get("content") or "") if isinstance(memory, dict) else ""
    facts = [
        fact
        for fact in expected.get("facts") or []
        if isinstance(fact, dict) and not _fact_label_bleeds(fact)
    ]
    expected["facts"] = facts
    fact_parts = [
        str(fact.get("value") or fact.get("value_text") or fact.get("evidence_text") or "")
        for fact in facts
    ]
    label_text = " ".join(part for part in [content, *fact_parts] if part)
    if content and has_curriculum_bleed(content):
        return True
    if content and not is_grounded_in_source(source_text, content):
        return True
    for fact in facts:
        evidence = str(fact.get("evidence_text") or fact.get("value") or "")
        if evidence and not _substring_grounded(source_text, evidence):
            return True
    indexables = build_indexable_labels(
        llm_response=source_text,
        memory_content=content,
        tags=(memory.get("tags") if isinstance(memory, dict) else []) or [],
        facts=facts,
    )
    expected["indexables"] = indexables
    return False


def _fact_label_bleeds(fact: dict[str, Any]) -> bool:
    parts = [
        str(fact.get("value") or ""),
        str(fact.get("value_text") or ""),
        str(fact.get("evidence_text") or ""),
    ]
    return has_curriculum_bleed(" ".join(part for part in parts if part))


def _substring_grounded(source: str, fragment: str) -> bool:
    normalized = re.sub(r"\s+", " ", fragment.strip())
    if not normalized:
        return False
    if normalized in source:
        return True
    compact_source = re.sub(r"\s+", " ", source)
    return normalized in compact_source


def _clip_grounded(source: str, content: str, *, max_len: int) -> str:
    clipped = content.strip()
    if len(clipped) > max_len:
        clipped = clipped[: max_len - 3].rsplit(" ", 1)[0] + "..."
    if _substring_grounded(source, clipped):
        return clipped
    shorter = clipped[: max(24, len(clipped) // 2)].rsplit(" ", 1)[0]
    return shorter if _substring_grounded(source, shorter) else ""


def _first_sentence(text: str) -> str:
    chunk = text.strip()
    for sep in ("\n", ". "):
        if sep in chunk:
            chunk = chunk.split(sep, 1)[0].strip()
            break
    return _clean_clause(chunk)


def _clean_clause(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = cleaned.strip(" *-_")
    return cleaned


def _subject_from_clause(clause: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", clause.lower())
    return "-".join(tokens[:3]) or "detail"


def _candidate_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph or HEADER_LINE.match(paragraph):
            continue
        for chunk in re.split(r"(?<=[.!?])\s+", paragraph):
            cleaned = _clean_clause(chunk)
            if cleaned:
                sentences.append(cleaned)
    return sentences


def _unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
