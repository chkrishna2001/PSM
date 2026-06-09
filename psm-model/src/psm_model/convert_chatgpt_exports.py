from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

from psm_model.data import validate_training_row
from psm_model.generate_action_foundation_curriculum import expected_memory, row

USER_HEADER = re.compile(r"^#{1,6}\s*(?:🧑\s*)?\*{0,2}User\*{0,2}\s*$", re.IGNORECASE)
ASSISTANT_HEADER = re.compile(r"^#{1,6}\s*(?:🤖\s*)?\*{0,2}Assistant\*{0,2}\s*$", re.IGNORECASE)
PERSONAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bI prefer\b", re.I), "preference"),
    (re.compile(r"\bI always\b", re.I), "rule"),
    (re.compile(r"\balways use\b", re.I), "rule"),
    (re.compile(r"\bmy name is\b", re.I), "identity"),
    (re.compile(r"\bI work at\b", re.I), "identity"),
    (re.compile(r"\bI live in\b", re.I), "identity"),
    (re.compile(r"\bremember that\b", re.I), "preference"),
    (re.compile(r"\bfor future\b", re.I), "rule"),
    (re.compile(r"\bI'm using\b|\bI am using\b", re.I), "preference"),
    (re.compile(r"\bmy (project|app|stack|setup|database|server)\b", re.I), "identity"),
    (re.compile(r"\bwe (use|run|deploy|host)\b", re.I), "rule"),
    (re.compile(r"\bdo not (use|delete|commit)\b", re.I), "rule"),
)
EPISODIC_MARKERS = re.compile(r"\b(yesterday|today|last week|last month|on \d{4}-\d{2}-\d{2})\b", re.I)
QUESTION_ONLY = re.compile(r"^(how|what|why|when|where|is it|can you|could you|should i)\b", re.I)


def _iter_user_turns(markdown: str) -> Iterator[str]:
    role: str | None = None
    buffer: list[str] = []

    def flush() -> Iterator[str]:
        nonlocal buffer, role
        if role == "user" and buffer:
            text = "\n".join(buffer).strip()
            if text:
                yield text
        buffer = []

    for line in markdown.splitlines():
        if USER_HEADER.match(line.strip()):
            yield from flush()
            role = "user"
            continue
        if ASSISTANT_HEADER.match(line.strip()):
            yield from flush()
            role = "assistant"
            continue
        if line.strip() == "---":
            continue
        if role == "user":
            buffer.append(line.rstrip())
    yield from flush()


def _first_sentence(text: str, *, max_len: int = 320) -> str:
    chunk = text.strip()
    for sep in ("\n\n", "\n"):
        if sep in chunk:
            chunk = chunk.split(sep, 1)[0].strip()
    if len(chunk) > max_len:
        chunk = chunk[: max_len - 3].rsplit(" ", 1)[0] + "..."
    return chunk


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned[:48] or "fact"


def _label_turn(text: str, *, source_stem: str, index: int) -> dict[str, Any] | None:
    sentence = _first_sentence(text)
    if len(sentence) < 24 or len(sentence) > 400:
        return None
    if sentence.count("?") >= 2 or (sentence.endswith("?") and QUESTION_ONLY.match(sentence) and " I " not in sentence):
        return None
    if not any(pattern.search(sentence) for pattern, _ in PERSONAL_PATTERNS):
        if not EPISODIC_MARKERS.search(sentence):
            return None

    conversation = f"User: {sentence}"
    base_input = {
        "conversation": conversation,
        "operation": "remember",
        "source_id": f"chatgpt:{source_stem}:{index}",
        "source_kind": "chatgpt_export",
        "source_timestamp": "2026-06-07T12:00:00Z",
    }
    row_id = f"chatgpt-{source_stem}-{index}"

    if EPISODIC_MARKERS.search(sentence):
        marker = EPISODIC_MARKERS.search(sentence)
        assert marker is not None
        temporal = marker.group(1)
        return row(
            row_id,
            base_input,
            expected_memory(
                "store_episodic",
                "episodic",
                f"The user reported: {sentence}",
                ["event", "chatgpt"],
                "User",
                "reported_event",
                _slug(sentence),
                sentence,
                "Personal episodic statement from ChatGPT export.",
                temporal_expression=temporal.lower(),
                resolved_time="2026-06-07",
                decay_rate=0.05,
            ),
        )

    kind = next(kind for pattern, kind in PERSONAL_PATTERNS if pattern.search(sentence))
    return row(
        row_id,
        base_input,
        expected_memory(
            "promote_semantic",
            "semantic",
            sentence.rstrip("."),
            ["chatgpt", kind],
            "User",
            kind,
            _slug(sentence),
            sentence,
            "Personal durable statement from ChatGPT export.",
        ),
    )


def convert_chatgpt_exports(input_dir: Path, *, limit_files: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted(input_dir.glob("*.md"))
    if limit_files is not None:
        files = files[:limit_files]

    rows: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    skipped = {"too_short": 0, "not_personal": 0, "duplicate": 0, "invalid": 0}

    for path in files:
        markdown = path.read_text(encoding="utf-8", errors="replace")
        for index, turn in enumerate(_iter_user_turns(markdown)):
            labeled = _label_turn(turn, source_stem=path.stem[:40], index=index)
            if labeled is None:
                if len(_first_sentence(turn)) < 24:
                    skipped["too_short"] += 1
                else:
                    skipped["not_personal"] += 1
                continue
            text_key = hashlib.sha256(labeled["input"]["conversation"].encode()).hexdigest()
            if text_key in seen_text:
                skipped["duplicate"] += 1
                continue
            _, issues = validate_training_row(labeled)
            if issues:
                skipped["invalid"] += 1
                continue
            seen_text.add(text_key)
            rows.append(labeled)

    summary = {
        "input_dir": str(input_dir),
        "files_scanned": len(files),
        "rows": len(rows),
        "skipped": skipped,
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ChatGPT markdown exports into PSM training rows.")
    parser.add_argument("input_dir", type=Path, help="Directory of chatgpt_*.md exports.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    rows, summary = convert_chatgpt_exports(args.input_dir, limit_files=args.limit_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
