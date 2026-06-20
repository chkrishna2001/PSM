from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from prod_memory.label_from_assistant import build_row_from_assistant, should_ignore_assistant_text
from prod_memory.row_validation import validate_prod_row
from prod_memory.segment_text import segment_llm_response

CHATGPT_USER_LINE = re.compile(r"^#{1,6}\s*(?:🧑\s*)?\*{0,2}User\*{0,2}\s*$", re.IGNORECASE | re.MULTILINE)
ASSISTANT_HEADER = re.compile(r"^#{1,6}\s*(?:🤖\s*)?\*{0,2}Assistant\*{0,2}\s*$", re.IGNORECASE | re.MULTILINE)
USER_HEADER = CHATGPT_USER_LINE
EXPORT_BOILERPLATE = re.compile(
    r"^(\*Exported from|title:|source:|platform:|exportDate:|---|\s*# )",
    re.IGNORECASE,
)

DEFAULT_MIN_CHARS = 320
DEFAULT_MAX_CHARS = 12_000


def ingest_training_directory(
    root: Path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    include_codex_commentary: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "root": str(root),
        "sources": {},
        "candidates": 0,
        "accepted": 0,
        "skipped": {},
    }

    for source_kind, iterator in (
        ("codex_session", _iter_codex_messages(root, include_commentary=include_codex_commentary)),
        ("chatgpt_export", _iter_chatgpt_messages(root)),
        ("gemini_session", _iter_gemini_messages(root)),
    ):
        source_report = _ingest_messages(
            iterator,
            source_kind=source_kind,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        rows.extend(source_report["rows"])
        report["sources"][source_kind] = {key: value for key, value in source_report.items() if key != "rows"}
        report["candidates"] += int(source_report["candidates"])
        report["accepted"] += int(source_report["accepted"])
        for reason, count in source_report.get("skipped", {}).items():
            report["skipped"][reason] = report["skipped"].get(reason, 0) + count

    deduped, deduped_count = _dedupe_rows(rows)
    report["deduped"] = deduped_count
    report["rows"] = len(deduped)
    return deduped, report


def _ingest_messages(
    messages: Iterator[dict[str, str]],
    *,
    source_kind: str,
    min_chars: int,
    max_chars: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    candidates = 0

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for message in messages:
        text = message["text"].strip()
        if len(text) < min_chars:
            skip("too_short")
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        if should_ignore_assistant_text(text):
            skip("ignored_noise")
            continue

        segments = segment_llm_response(text)
        if not segments:
            skip("no_segments")
            continue

        for segment in segments:
            candidates += 1
            segment_text = segment.text.strip()
            if len(segment_text) < min_chars:
                skip("segment_too_short")
                continue
            row_id = f"session-{source_kind}-{message['source_key']}-c{segment.index}"
            source_id = f"{source_kind}:{message['source_key']}:chunk-{segment.index}"
            try:
                row = build_row_from_assistant(
                    segment_text,
                    row_id=row_id,
                    source_id=source_id,
                    source_kind=source_kind,
                )
            except ValueError:
                skip("validation_error")
                continue
            if row is None:
                skip("label_reject")
                continue
            try:
                validate_prod_row(row)
            except ValueError:
                skip("validation_error")
                continue
            rows.append(row)

    return {
        "candidates": candidates,
        "accepted": len(rows),
        "skipped": skipped,
        "rows": rows,
    }


def _iter_codex_messages(root: Path, *, include_commentary: bool) -> Iterator[dict[str, str]]:
    codex_dir = root / "codex-sessions"
    if not codex_dir.exists():
        return
    for path in sorted(codex_dir.glob("*.jsonl")):
        for index, text in enumerate(_extract_codex_assistant_text(path, include_commentary=include_commentary)):
            yield {
                "text": text,
                "source_key": f"{path.stem}-{index}",
            }


def _extract_codex_assistant_text(path: Path, *, include_commentary: bool) -> list[str]:
    messages: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "response_item":
                continue
            item = payload.get("payload") or {}
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            phase = str(item.get("phase") or "")
            if phase != "final_answer" and not (include_commentary and phase == "commentary"):
                continue
            text = _join_message_content(item.get("content"))
            if text.strip():
                messages.append(text.strip())
    return messages


def _iter_chatgpt_messages(root: Path) -> Iterator[dict[str, str]]:
    chat_dir = root / "chatgpt_chats"
    if not chat_dir.exists():
        return
    for path in sorted(chat_dir.glob("*.md")):
        for index, text in enumerate(_extract_chatgpt_assistant_text(path)):
            yield {
                "text": text,
                "source_key": f"{path.stem}-{index}",
            }


def _extract_chatgpt_assistant_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    messages: list[str] = []
    for part in CHATGPT_USER_LINE.split(text)[1:]:
        body = part.split("\n---\n", 1)[0]
        lines = body.splitlines()
        if not lines:
            continue
        assistant_lines: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            if CHATGPT_USER_LINE.match(stripped) or ASSISTANT_HEADER.match(stripped):
                break
            if EXPORT_BOILERPLATE.match(stripped):
                continue
            assistant_lines.append(line.rstrip())
        chunk = "\n".join(assistant_lines).strip()
        if chunk:
            messages.append(chunk)
    return messages


def _iter_gemini_messages(root: Path) -> Iterator[dict[str, str]]:
    gemini_dir = root / "gemini-sessions"
    if not gemini_dir.exists():
        return
    for path in sorted(list(gemini_dir.glob("*.json")) + list(gemini_dir.glob("*.jsonl"))):
        for index, text in enumerate(_extract_gemini_assistant_text(path)):
            yield {
                "text": text,
                "source_key": f"{path.stem}-{index}",
            }


def _extract_gemini_assistant_text(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        messages: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _gemini_payload_text(payload)
                if text:
                    messages.append(text)
        return messages

    payload = json.loads(path.read_text(encoding="utf-8"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("type") or "").lower() not in {"gemini", "model"}:
            continue
        text = _gemini_payload_text(message)
        if text:
            texts.append(text)
    return texts


def _gemini_payload_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            if text.strip().startswith("--- Content from referenced files ---"):
                continue
            parts.append(text.strip())
    return "\n\n".join(parts).strip()


def _join_message_content(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0
    for row in rows:
        conversation = row.get("input", {}).get("conversation")
        if isinstance(conversation, list) and conversation:
            key = str(conversation[0].get("content") or "").strip().lower()
        else:
            key = json.dumps(row.get("input"), sort_keys=True)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed
