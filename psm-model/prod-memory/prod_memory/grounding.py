from __future__ import annotations

import re
from typing import Any

BLEED_PATTERN = re.compile(
    r"checkpoint|powershell|gate datasets|nvidia-smi|direct probe|token budget|runpod|"
    r"fact parser|malformed parser|constoursated|gate6|expanded probe|gate-?\d",
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "what", "when", "where", "why", "how", "are",
    "was", "were", "has", "have", "had", "from", "about", "into", "onto", "then", "than", "they", "them",
    "does", "did", "doing", "done", "will", "would", "could", "should", "their", "there", "here", "also",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def significant_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if len(token) >= 3 and not token.isdigit()]


def has_curriculum_bleed(text: str) -> bool:
    return bool(BLEED_PATTERN.search(text))


def grounding_overlap_score(remember_target: str, stored_text: str) -> dict[str, int | bool]:
    input_tokens = significant_tokens(remember_target)
    if not input_tokens:
        return {"overlap": 0, "required": 0, "grounded": True}
    stored_set = set(significant_tokens(stored_text))
    overlap = sum(1 for token in input_tokens if token in stored_set)
    required = min(2, max(1, (len(input_tokens) + 9) // 10))
    return {"overlap": overlap, "required": required, "grounded": overlap >= required}


def is_grounded_in_source(remember_target: str, stored_text: str) -> bool:
    return bool(grounding_overlap_score(remember_target, stored_text)["grounded"])


def key_tokens_grounded(key_tokens: list[str], stored_text: str) -> bool:
    if not key_tokens:
        return False
    haystack = stored_text.lower()
    hits = sum(1 for token in key_tokens if token.lower() in haystack)
    return hits >= min(2, len(key_tokens))


def stored_text_from_decision(decision: dict[str, Any]) -> str:
    memory = decision.get("memory")
    content = ""
    if isinstance(memory, dict):
        content = str(memory.get("content") or "").strip()
    fact_parts: list[str] = []
    facts = decision.get("facts")
    if isinstance(facts, list):
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            for key in ("subject", "predicate", "value_text", "evidence_text"):
                value = fact.get(key)
                if value:
                    fact_parts.append(str(value))
    return " ".join(part for part in [content, *fact_parts] if part)


def would_model_store(decision: dict[str, Any]) -> bool:
    action = str(decision.get("action") or "").strip().lower()
    if action in {"ignore", "ignore_noise", "rank", "recall_weighting"}:
        return False
    return bool(stored_text_from_decision(decision).strip())


def apply_storage_guards(remember_target: str, decision: dict[str, Any]) -> dict[str, Any]:
    if not would_model_store(decision):
        return {"rejected": False, "route": None, "reason": None}
    stored_text = stored_text_from_decision(decision)
    if has_curriculum_bleed(stored_text):
        return {
            "rejected": True,
            "route": "grounding_reject_bleed",
            "reason": "Stored content matches curriculum bleed blocklist.",
        }
    if not is_grounded_in_source(remember_target, stored_text):
        return {
            "rejected": True,
            "route": "grounding_reject",
            "reason": "Stored content is not grounded in remember_target tokens.",
        }
    return {"rejected": False, "route": None, "reason": None}


def is_fail_safe_report(report: dict[str, Any]) -> bool:
    status = str(report.get("repair_status") or "")
    if status == "failed_safe":
        return True
    parsed = report.get("parsed")
    if isinstance(parsed, dict):
        reasoning = str(parsed.get("reasoning") or "")
        if re.search(r"model output unparseable|storing nothing", reasoning, re.IGNORECASE):
            return True
    return False
