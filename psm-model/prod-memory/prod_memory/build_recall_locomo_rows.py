"""Real recall-plan/context-plan curriculum from LoCoMo QA questions.

The only prior recall_plan/context_plan training data (`generate_recall_curriculum.py`'s
`build_recall_probe_rows()`, 23 hand-written scenarios) had zero real-question diversity and
its "eval" (gate5) tested the model on the exact same 23 scenarios it trained on -- pure
memorization, not a real signal (see docs/psm-model/PSM-MEMORY.md, 2026-07-10 finding).

This builds real training rows from LoCoMo's own `qa` field (1,986 real questions across 10
conversations) -- genuinely diverse questions a memory system would actually be asked. Source
conversations conv-47/48/49/50 (never touched by anything else this project). conv-26 is
deliberately EXCLUDED here and reserved as a genuinely held-out eval set
(build_recall_locomo_eval_cases below) -- conv-30/41/42/43/44 stay reserved for the storage
adapter's holdout gates and are not touched.

Labeling is a heuristic (LoCoMo has no ground-truth "which PSM table" label) based on LoCoMo's
own question category: 1=single-hop event, 2=temporal, 3=open-domain/inference,
4=multi-hop, 5=this dataset's variant (has real evidence despite the "adversarial" name).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from psm_model.generate_recall_curriculum import AVAILABLE_TABLES, _recall_row

LOCOMO_PATH = Path(__file__).resolve().parents[3] / "benchmark" / "locomo" / "data" / "locomo10.json"

TRAIN_CONVERSATIONS = ("conv-47", "conv-48", "conv-49", "conv-50")
EVAL_CONVERSATION = "conv-26"

_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "did", "does", "do",
    "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on", "for", "and", "or",
    "with", "about", "would", "could", "should", "have", "has", "had", "his", "her", "their",
    "both", "any", "at", "from", "that", "this",
}

_TEMPORAL_PATTERNS = [
    re.compile(r"\b(19|20)\d{2}\b"),
    re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I),
    re.compile(r"\b(yesterday|today|last (week|month|year|summer|winter|spring|fall))\b", re.I),
]


def _ranking_hints(question: str, max_hints: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", question)
    hints: list[str] = []
    seen: set[str] = set()
    for w in words:
        lw = w.lower()
        if lw in _STOPWORDS or len(w) < 3:
            continue
        if lw in seen:
            continue
        seen.add(lw)
        hints.append(w)
        if len(hints) >= max_hints:
            break
    return hints


def _temporal_intent(question: str) -> str | None:
    for pat in _TEMPORAL_PATTERNS:
        m = pat.search(question)
        if m:
            return m.group(0)
    return None


def _target_tables(category: int, temporal: str | None) -> list[str]:
    if category == 1:
        return ["episodic"]
    if category == 2:
        return ["episodic"]
    if category == 3:
        return ["semantic", "episodic"]
    if category == 4:
        return ["episodic", "semantic", "archival"]
    if category == 5:
        return ["episodic", "semantic"]
    return ["episodic", "semantic", "archival"]


def _top_k(category: int) -> int:
    return 8 if category == 4 else 5


def _row_from_qa(conv_id: str, idx: int, qa: dict[str, Any]) -> dict[str, Any] | None:
    question = qa.get("question")
    category = qa.get("category")
    if not isinstance(question, str) or not question.strip() or not isinstance(category, int):
        return None
    temporal = _temporal_intent(question)
    expected = {
        "intent": "recall",
        "target_tables": _target_tables(category, temporal),
        "filters": {},
        "ranking_hints": _ranking_hints(question),
        "temporal_intent": temporal,
        "top_k": _top_k(category),
    }
    row = _recall_row(
        f"recall-locomo-{conv_id}-{idx}",
        operation="recall_plan",
        question=question,
        expected=expected,
        top_k=_top_k(category),
    )
    row["source"] = f"locomo_recall:{conv_id}:category-{category}"
    return row


def _load_locomo() -> list[dict[str, Any]]:
    return json.loads(LOCOMO_PATH.read_text(encoding="utf-8"))


def build_recall_locomo_train_rows() -> list[dict[str, Any]]:
    convs = {c["sample_id"]: c for c in _load_locomo()}
    rows: list[dict[str, Any]] = []
    for conv_id in TRAIN_CONVERSATIONS:
        conv = convs[conv_id]
        for idx, qa in enumerate(conv.get("qa") or []):
            row = _row_from_qa(conv_id, idx, qa)
            if row:
                rows.append(row)
    return rows


def build_recall_locomo_eval_cases() -> dict[str, Any]:
    """Genuinely held-out recall-plan eval fixtures from conv-26 -- never used in training."""
    convs = {c["sample_id"]: c for c in _load_locomo()}
    conv = convs[EVAL_CONVERSATION]
    cases: list[dict[str, Any]] = []
    for idx, qa in enumerate(conv.get("qa") or []):
        row = _row_from_qa(EVAL_CONVERSATION, idx, qa)
        if not row:
            continue
        cases.append({
            "id": row["id"],
            "suite": "recall_locomo_holdout",
            "question": row["input"]["question"],
            "availableTables": AVAILABLE_TABLES,
            "requestedTopK": row["input"]["requested_top_k"],
            "expected": row["expected"],
        })
    return {"cases": cases}
