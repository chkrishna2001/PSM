#!/usr/bin/env python3
"""Build teacher-labeled prod-memory rows from ChatGPT chat exports."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.ingest_training_data import (
    _extract_codex_assistant_text,
    _extract_gemini_assistant_text,
)
from prod_memory.openrouter_teacher import TeacherConfig, build_row_from_teacher

USER_HEADER = re.compile(r"^#{1,6}\s*(?:🧑\s*)?\*{0,2}User\*{0,2}\s*$", re.IGNORECASE)
ASSISTANT_HEADER = re.compile(r"^#{1,6}\s*(?:🤖\s*)?\*{0,2}Assistant\*{0,2}\s*$", re.IGNORECASE)


def _clean_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return compact


def _iter_markdown_turns(path: Path, *, role: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    current_role: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_role
        if current_role == role and buffer:
            text = _clean_text("\n".join(buffer))
            if text:
                turns.append((path.stem, text))
        buffer = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if USER_HEADER.match(stripped):
            flush()
            current_role = "user"
            continue
        if ASSISTANT_HEADER.match(stripped):
            flush()
            current_role = "assistant"
            continue
        if stripped == "---":
            continue
        if current_role in {"user", "assistant"}:
            buffer.append(line)
    flush()
    if turns:
        return turns
    return _iter_compact_chatgpt_markdown_turns(path, role=role)


def _iter_compact_chatgpt_markdown_turns(path: Path, *, role: str) -> list[tuple[str, str]]:
    """Handle exports with one user header and interleaved message blocks."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = -1
    for idx, line in enumerate(lines):
        if USER_HEADER.match(line.strip()):
            start = idx + 1
            break
    if start < 0:
        return []

    content: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped == "---" or stripped.startswith("ChatGPT can make mistakes"):
            continue
        if stripped.startswith("# "):
            continue
        if stripped.startswith("*Exported from"):
            continue
        if stripped.startswith("!["):
            continue
        content.append(line.rstrip())

    raw_chunks: list[str] = []
    chunk: list[str] = []
    blank_run = 0
    for line in content:
        if not line.strip():
            blank_run += 1
            if blank_run >= 2 and chunk:
                raw_chunks.append(_clean_text("\n".join(chunk)))
                chunk = []
            continue
        blank_run = 0
        chunk.append(line)
    if chunk:
        raw_chunks.append(_clean_text("\n".join(chunk)))

    raw_chunks = [c for c in raw_chunks if c]
    inferred = _infer_roles_for_compact_chunks(raw_chunks)
    turns: list[tuple[str, str]] = []
    for inferred_role, block in inferred:
        if inferred_role == role:
            turns.append((path.stem, block))
    return turns


def _infer_roles_for_compact_chunks(chunks: list[str]) -> list[tuple[str, str]]:
    stitched: list[tuple[str, str]] = []
    last_role: str | None = None
    for chunk in chunks:
        role = _classify_compact_chunk_role(chunk, last_role=last_role)
        if stitched and stitched[-1][0] == role:
            stitched[-1] = (role, f"{stitched[-1][1]} {chunk}".strip())
        else:
            stitched.append((role, chunk))
        last_role = role
    return stitched


def _classify_compact_chunk_role(chunk: str, *, last_role: str | None) -> str:
    lower = chunk.lower()
    if "chatgpt can make mistakes" in lower:
        return "assistant"
    if re.search(r"\b(you need to|can you|could you|please|i need|what's|whats|why|how)\b", lower) and len(chunk) < 420:
        return "user"
    if chunk.endswith("?") and len(chunk) < 320:
        return "user"
    if chunk.startswith(("##", "###", "-", "1.", "2.", "3.", "```")):
        return "assistant"
    if re.search(r"\b(here's|let's|i think|therefore|hypothesis|experiment|conclusion)\b", lower):
        return "assistant"
    # ponytail: fallback alternation has a ceiling on noisy prose; upgrade path is richer metadata parser.
    return "assistant" if last_role == "user" else "user"


def _is_durable_candidate(text: str) -> bool:
    lower = text.lower()
    if len(text) < 40:
        return False
    strong_patterns = (
        "i prefer",
        "i always",
        "my setup",
        "my stack",
        "my project",
        "we use",
        "we run",
        "we deploy",
        "don't ",
        "do not ",
        "never ",
        "yesterday",
        "last week",
        "last month",
        "today",
        "tomorrow",
        "next week",
        "next month",
        "i'm using",
        "i am using",
        "i decided",
        "we decided",
        "from now on",
        "i will",
        "we will",
        "rule",
        "constraint",
    )
    return any(p in lower for p in strong_patterns)


def _iter_conversations_json_turns(path: Path, *, role: str) -> list[tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    turns: list[tuple[str, str]] = []
    for convo in items:
        if not isinstance(convo, dict):
            continue
        convo_id = str(convo.get("id") or "convo")
        mapping = convo.get("mapping")
        if not isinstance(mapping, dict):
            continue
        rows: list[tuple[float, str]] = []
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            author = message.get("author") or {}
            if str(author.get("role") or "") != role:
                continue
            content = message.get("content") or {}
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            text = _clean_text(" ".join(str(p) for p in parts if isinstance(p, str)))
            if not text:
                continue
            created = message.get("create_time")
            created_ts = float(created) if isinstance(created, (int, float)) else 0.0
            rows.append((created_ts, text))
        rows.sort(key=lambda r: r[0])
        for idx, (_, text) in enumerate(rows):
            turns.append((f"{convo_id[:32]}-{idx}", text))
    return turns


def _collect_training_data_turns(root: Path, *, role: str, sources: set[str]) -> list[tuple[str, str]]:
    """Collect turns from training-data/{chatgpt_chats,codex-sessions,gemini-sessions}."""
    if role != "assistant":
        raise SystemExit("training-data mode currently supports --role assistant only")
    turns: list[tuple[str, str]] = []
    if "chatgpt" in sources:
        chat_dir = root / "chatgpt_chats"
        if chat_dir.is_dir():
            for path in sorted(chat_dir.glob("*.md")):
                for idx, (_, text) in enumerate(_iter_markdown_turns(path, role="assistant")):
                    turns.append((f"{path.stem}-{idx}", text))
    if "codex" in sources:
        codex_dir = root / "codex-sessions"
        if codex_dir.is_dir():
            for path in sorted(codex_dir.glob("*.jsonl")):
                for idx, text in enumerate(_extract_codex_assistant_text(path, include_commentary=False)):
                    turns.append((f"{path.stem}-{idx}", text))
    if "gemini" in sources:
        gemini_dir = root / "gemini-sessions"
        if gemini_dir.is_dir():
            for path in sorted(list(gemini_dir.glob("*.json")) + list(gemini_dir.glob("*.jsonl"))):
                for idx, text in enumerate(_extract_gemini_assistant_text(path)):
                    turns.append((f"{path.stem}-{idx}", text))
    return turns


def _source_kind(source_key: str) -> str:
    if source_key.startswith("rollout-"):
        return "codex_session_assistant"
    if source_key.startswith("session-"):
        return "gemini_session_assistant"
    return "chatgpt_export_assistant"


def _collect_turns(input_path: Path, *, role: str, sources: set[str] | None = None) -> list[tuple[str, str]]:
    if input_path.is_dir() and (input_path / "chatgpt_chats").is_dir():
        return _collect_training_data_turns(input_path, role=role, sources=sources or {"chatgpt", "codex", "gemini"})
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.md")) + sorted(input_path.rglob("conversations.json"))

    turns: list[tuple[str, str]] = []
    for file_path in files:
        lower = file_path.name.lower()
        if lower.endswith(".md"):
            turns.extend(_iter_markdown_turns(file_path, role=role))
        elif lower == "conversations.json":
            turns.extend(_iter_conversations_json_turns(file_path, role=role))
    return turns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ChatGPT export folder or file path.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "cloudflare"],
        default="",
        help="Teacher API backend (default: cloudflare if CF env set, else openrouter).",
    )
    parser.add_argument("--role", choices=["assistant", "user"], default="assistant")
    parser.add_argument("--limit", type=int, default=0, help="0 means all turns.")
    parser.add_argument("--min-chars", type=int, default=24)
    parser.add_argument("--durable-only", action="store_true", help="Prefilter to likely durable-memory turns.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N turns after filtering.")
    parser.add_argument(
        "--sources",
        default="chatgpt,codex,gemini",
        help="Comma-separated sources when --input is training-data root.",
    )
    args = parser.parse_args(argv)

    cfg = TeacherConfig.from_env(model=args.model, provider=args.provider or None)
    if not cfg.api_key:
        raise SystemExit(f"{cfg.provider} API key required (set OPENROUTER_API_KEY or CLOUDFLARE_API_TOKEN)")

    source_set = {s.strip() for s in args.sources.split(",") if s.strip()}
    turns = _collect_turns(args.input, role=args.role, sources=source_set)
    turns = [(src, text) for (src, text) in turns if len(text) >= args.min_chars]
    if args.durable_only:
        turns = [(src, text) for (src, text) in turns if _is_durable_candidate(text)]
    if args.offset > 0:
        turns = turns[args.offset :]
    if args.limit > 0:
        turns = turns[: args.limit]
    if not turns:
        raise SystemExit("no turns found to label")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[str] = set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            conv = row.get("input", {}).get("conversation")
            if isinstance(conv, list) and conv:
                existing_keys.add(str(conv[0].get("content") or "").strip().lower())

    rows: list[dict[str, Any]] = []
    action_counts: dict[str, int] = {}
    skipped = 0
    for idx, (source_stem, text) in enumerate(turns):
        key = text.strip().lower()
        if key in existing_keys:
            skipped += 1
            print(f"[{idx + 1}/{len(turns)}] SKIP resume duplicate", flush=True)
            continue
        kind = _source_kind(source_stem)
        row_id = f"conv-{kind}-{idx:05d}"
        row, meta = build_row_from_teacher(
            text,
            row_id=row_id,
            source_id=f"{kind}:{source_stem}:{idx}",
            source_kind=kind,
            config=cfg,
            use_heuristic_fallback=False,
        )
        if row is None:
            skipped += 1
            print(f"[{idx + 1}/{len(turns)}] {row_id} SKIP {meta.get('error') or meta.get('parse_error')}", flush=True)
            continue
        action = str(row["expected"].get("action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        rows.append(row)
        existing_keys.add(key)
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{idx + 1}/{len(turns)}] {row_id} -> {action}", flush=True)

    summary_rows = len(rows)
    if args.out.is_file():
        summary_rows = sum(1 for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip())

    print(
        json.dumps(
            {
                "output": str(args.out),
                "rows": summary_rows,
                "rows_this_run": len(rows),
                "skipped": skipped,
                "actions": action_counts,
                "role": args.role,
                "model": args.model,
                "durable_only": args.durable_only,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
