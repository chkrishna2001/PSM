"""Storage-decision curriculum mined from LoCoMo conversation turns (personal/social memory content),
for the conversational_storage adapter.

The coding-domain storage adapter's curriculum (prod_extraction_v1..v7 etc.) was built from Claude
Code/Codex/ChatGPT assistant turns, teacher-labeled into the same action/memory/facts/indexables JSON
schema. This does the same thing for personal conversations instead of coding-agent technical text --
same schema, same store-decision vocabulary (ignore/store_episodic/promote_semantic), new content
domain. Reuses row_messages() (hf_prompts.py) for the final {input, expected} -> {id, messages} SFT
conversion, so output rows are byte-compatible with hf_lora_train.py without any new formatting code.

Reuses the exact TRAIN_CONVERSATIONS/EVAL_CONVERSATION split as build_recall_locomo_rows.py
(conv-47/48/49/50 for training, conv-26 held out for eval) so every conversational adapter shares the
same train/holdout discipline -- conv-30/41/42/43/44 remain untouched, reserved purely for the eventual
LoCoMo answer-accuracy benchmark (Phase 4 in transient-dazzling-lake.md), never used for training or
per-adapter gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from prod_memory.curriculum_sources import _ignore_expected, _store_expected, remember_input
from prod_memory.hf_prompts import row_messages
from prod_memory.label_from_assistant import _make_fact, _substring_grounded
from prod_memory.teacher_client import AllProvidersExhausted, TeacherClient, complete_json

LOCOMO_PATH = Path(__file__).resolve().parents[3] / "benchmark" / "locomo" / "data" / "locomo10.json"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "conversational-storage-teacher-cache.jsonl"

TRAIN_CONVERSATIONS = ("conv-47", "conv-48", "conv-49", "conv-50")
EVAL_CONVERSATION = "conv-26"

ALLOWED_ACTIONS = {"ignore", "store_episodic", "promote_semantic"}

SYSTEM_PROMPT = """You are a strict PSM production-memory training labeler for PERSONAL conversations
between friends, family, or partners -- NOT technical/coding content.
Return one complete JSON object only. No markdown fences. No commentary. Start with { and end with }.

Task: label a single message from an ongoing personal conversation. The message is formatted as
"Speaker: text" -- attribute any facts to that named speaker. Decide whether it contains a durable
fact worth remembering about that person for future conversations.

Allowed actions: ignore, store_episodic, promote_semantic.
Use ignore for greetings, small talk, reactions ("that's great!", "aw thanks"), questions with no
disclosed fact, or anything with no durable personal information.
Use promote_semantic for durable personal facts: stable preferences, relationships, identity, ongoing
life circumstances (job, health, family, home), recurring habits or hobbies.
Use store_episodic for concrete one-time events, milestones, plans, or experiences the speaker mentions
as having happened or being about to happen (a trip, a decision, something that occurred on an occasion).

Rules:
- Do not invent facts. memory.content and every fact must be grounded in verbatim spans from the message text.
- evidence_text must be an exact substring (or whitespace-normalized match) from the message text.
- memory.content must be concise (under 480 chars), written in third person naming the speaker (e.g.
  "Caroline volunteers at an LGBTQ support group"), not a raw quote.
- Return at most 4 facts. Each fact needs subject, predicate (snake_case), value, evidence_text.
- Return indexables: [] -- indexables are built locally.
- Prefer ignore over weak storage -- most single conversational turns are small talk with nothing durable.

Return exactly:
{"action":"ignore|store_episodic|promote_semantic","memory":null|{"content":"...","type":"episodic|semantic","confidence":0.9,"tags":["..."]},"facts":[{"subject":"...","predicate":"snake_case","value":"...","evidence_text":"..."}],"reasoning":"short reason"}"""


def _load_locomo() -> list[dict[str, Any]]:
    return json.loads(LOCOMO_PATH.read_text(encoding="utf-8"))


def _turns_for(conv: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = sorted(
        (k for k in conv["conversation"] if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda s: int(s.split("_")[1]),
    )
    turns: list[dict[str, Any]] = []
    for session in sessions:
        for turn in conv["conversation"][session]:
            speaker = turn.get("speaker")
            text = turn.get("text")
            dia_id = turn.get("dia_id")
            if not speaker or not text or not dia_id:
                continue
            turns.append({"speaker": speaker, "text": text, "dia_id": dia_id, "session": session})
    return turns


def _normalize_teacher_facts(value: Any, source_text: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    facts: list[dict[str, Any]] = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "states").strip().lower().replace(" ", "_")
        fact_value = str(item.get("value") or "").strip()
        evidence = str(item.get("evidence_text") or fact_value).strip()
        if not subject or not fact_value or not evidence:
            continue
        if not _substring_grounded(source_text, evidence) and not _substring_grounded(source_text, fact_value):
            continue
        grounded_evidence = evidence if _substring_grounded(source_text, evidence) else fact_value
        facts.append(_make_fact(subject=subject, predicate=predicate, value=fact_value, evidence_text=grounded_evidence, confidence=0.86))
    return facts


def label_turn_with_teacher(client: TeacherClient, speaker: str, text: str) -> dict[str, Any]:
    """Returns an `expected` dict (action/memory/facts/reasoning), same shape as coding-domain labeling."""
    formatted = f"{speaker}: {text}"
    user_payload = json.dumps({"message": formatted}, ensure_ascii=False)
    out = complete_json(client, user_payload, system=SYSTEM_PROMPT, max_tokens=500)
    if out is None:
        return _ignore_expected("Teacher response could not be parsed.")

    action = str(out.get("action") or "ignore").strip()
    if action not in ALLOWED_ACTIONS:
        action = "ignore"
    reasoning = str(out.get("reasoning") or "Teacher labeled conversational turn.").strip()

    if action == "ignore":
        return _ignore_expected(reasoning)

    memory_payload = out.get("memory")
    if not isinstance(memory_payload, dict):
        return _ignore_expected("Teacher chose store but returned no memory object.")

    content = str(memory_payload.get("content") or "").strip()
    if len(content) < 12:
        return _ignore_expected("Teacher memory content too short.")

    memory_type = str(memory_payload.get("type") or ("semantic" if action == "promote_semantic" else "episodic"))
    if memory_type not in {"episodic", "semantic"}:
        memory_type = "episodic" if action == "store_episodic" else "semantic"

    tags = [str(t).strip().lower().replace(" ", "_") for t in (memory_payload.get("tags") or []) if str(t).strip()][:6]
    facts = _normalize_teacher_facts(out.get("facts"), formatted)

    return _store_expected(
        formatted,
        content,
        tags=tags or ["personal_conversation"],
        reasoning=reasoning,
        facts=facts,
        memory_type=memory_type,
    )


def _load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["dia_id"]] = row
    return cache


def build_storage_locomo_rows(*, conversations: tuple[str, ...] = TRAIN_CONVERSATIONS, limit: int | None = None) -> list[dict[str, Any]]:
    """Mines turns from `conversations`, teacher-labels them (resumable via CACHE_PATH), and returns
    final {id, messages} rows ready for hf_lora_train.py.
    """
    convs = {c["sample_id"]: c for c in _load_locomo()}
    cache = _load_cache()
    client = TeacherClient()

    all_turns: list[tuple[str, dict[str, Any]]] = []
    for conv_id in conversations:
        for turn in _turns_for(convs[conv_id]):
            all_turns.append((conv_id, turn))
    if limit:
        all_turns = all_turns[:limit]

    rows: list[dict[str, Any]] = []
    new_labels = 0
    cache_file = CACHE_PATH.open("a", encoding="utf-8")
    try:
        for i, (conv_id, turn) in enumerate(all_turns):
            dia_id = f"{conv_id}:{turn['dia_id']}"
            cached = cache.get(dia_id)
            if cached is not None:
                expected = cached["expected"]
            else:
                try:
                    expected = label_turn_with_teacher(client, turn["speaker"], turn["text"])
                except AllProvidersExhausted as exc:
                    print(f"[{i + 1}/{len(all_turns)}] all teacher providers exhausted: {exc}", file=sys.stderr)
                    break
                cache_file.write(json.dumps({"dia_id": dia_id, "expected": expected}, ensure_ascii=False) + "\n")
                cache_file.flush()
                new_labels += 1

            formatted = f"{turn['speaker']}: {turn['text']}"
            row = {
                "id": f"locomo-storage-{dia_id}",
                "input": remember_input(formatted, source_id=dia_id, source_kind="locomo_conversation_turn"),
                "expected": expected,
                "source": f"locomo_storage_teacher:{conv_id}",
            }
            rows.append({"id": row["id"], "messages": row_messages(row, output_format="json")})

            if (i + 1) % 25 == 0:
                stored = sum(1 for r in rows if json.loads(r["messages"][2]["content"]).get("action") != "ignore")
                print(f"[{i + 1}/{len(all_turns)}] labeled ({new_labels} new this run), {stored} store so far", file=sys.stderr)
    finally:
        cache_file.close()

    print(f"teacher stats: {client.stats}", file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap total turns mined (0 = all)")
    ap.add_argument("--conversations", nargs="+", default=list(TRAIN_CONVERSATIONS))
    args = ap.parse_args()

    rows = build_storage_locomo_rows(conversations=tuple(args.conversations), limit=args.limit or None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
