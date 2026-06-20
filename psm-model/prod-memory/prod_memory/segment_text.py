from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SegmentSplitReason = Literal["single", "markdown_header", "numbered_block", "paragraph", "hard_max"]

DEFAULT_MAX_CHUNK_TOKENS = 1200
DEFAULT_MIN_CHUNK_TOKENS = 200


@dataclass(frozen=True)
class TextSegment:
    index: int
    text: str
    estimated_tokens: int
    split_reason: SegmentSplitReason


def estimate_text_tokens(text: str) -> int:
    trimmed = text.strip()
    if not trimmed:
        return 0
    return max(1, (len(trimmed) + 3) // 4)


def segment_llm_response(
    text: str,
    *,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
) -> list[TextSegment]:
    trimmed = text.strip()
    if not trimmed:
        return []
    if estimate_text_tokens(trimmed) <= max_chunk_tokens:
        return [_make_segment(0, trimmed, "single")]

    header_sections = _split_by_markdown_headers(trimmed)
    units: list[tuple[str, SegmentSplitReason]] = []
    for section in header_sections:
        reason: SegmentSplitReason = "markdown_header" if len(header_sections) > 1 else "paragraph"
        for unit in _split_preserving_numbered_blocks(section):
            unit_reason: SegmentSplitReason = "numbered_block" if _is_numbered_workflow_block(unit) else reason
            if estimate_text_tokens(unit) <= max_chunk_tokens:
                units.append((unit, unit_reason))
            else:
                for piece in _hard_split_by_tokens(unit, max_chunk_tokens):
                    units.append((piece, "hard_max"))

    merged = _merge_small_segments(units, min_chunk_tokens, max_chunk_tokens)
    return [_make_segment(index, unit_text, reason) for index, (unit_text, reason) in enumerate(merged)]


def _make_segment(index: int, text: str, split_reason: SegmentSplitReason) -> TextSegment:
    return TextSegment(index=index, text=text, estimated_tokens=estimate_text_tokens(text), split_reason=split_reason)


def _split_by_markdown_headers(text: str) -> list[str]:
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append("\n".join(current).strip())
        current = []

    for line in lines:
        if re.match(r"^#{1,3}\s+\S", line):
            flush()
            current.append(line)
            continue
        current.append(line)
    flush()
    return [section for section in sections if section.strip()] or [text]


def _split_preserving_numbered_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_numbered = False

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append("\n".join(current).strip())
        current = []

    for line in lines:
        numbered = bool(re.match(r"^\s*\d+\.\s+\S", line))
        if numbered and not in_numbered:
            flush()
            in_numbered = True
        elif not numbered and in_numbered and line.strip():
            flush()
            in_numbered = numbered
        elif not numbered:
            in_numbered = False
        current.append(line)
    flush()
    return [block for block in blocks if block.strip()] or [text]


def _is_numbered_workflow_block(text: str) -> bool:
    lines = [line for line in text.splitlines() if re.match(r"^\s*\d+\.\s+\S", line)]
    return len(lines) >= 2


def _hard_split_by_tokens(text: str, max_chunk_tokens: int) -> list[str]:
    max_chars = max_chunk_tokens * 4
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind("\n\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


def _merge_small_segments(
    units: list[tuple[str, SegmentSplitReason]],
    min_chunk_tokens: int,
    max_chunk_tokens: int,
) -> list[tuple[str, SegmentSplitReason]]:
    if not units:
        return []
    merged: list[tuple[str, SegmentSplitReason]] = []
    buffer_text = ""
    buffer_reason: SegmentSplitReason = "paragraph"

    def flush() -> None:
        nonlocal buffer_text, buffer_reason
        if buffer_text.strip():
            merged.append((buffer_text.strip(), buffer_reason))
        buffer_text = ""

    for text, reason in units:
        if not buffer_text:
            buffer_text = text
            buffer_reason = reason
            continue
        combined = f"{buffer_text}\n\n{text}"
        if estimate_text_tokens(buffer_text) < min_chunk_tokens and estimate_text_tokens(combined) <= max_chunk_tokens:
            buffer_text = combined
            continue
        flush()
        buffer_text = text
        buffer_reason = reason
    flush()
    return merged
