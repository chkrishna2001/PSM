#!/usr/bin/env python3
"""One-off reviewing aid: pull candidate assistant-turn texts from real, genuinely
held-out coding-agent sources (Claude Code transcripts, Codex rollouts, ChatGPT
exports) so they can be hand-labeled into holdout-coding-agent-cases.json.

Not part of the training pipeline -- outputs a flat JSONL for manual review only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CLAUDE_CODE_SESSIONS = [
    Path.home() / ".claude/projects/C--Users-chkri-source-repos-PSM/79c7a744-d4f6-41fd-a9b1-57432061b636.jsonl",
    Path.home() / ".claude/projects/C--Users-chkri-source-repos-PSM/e8bce25e-2c17-4eb6-8dcd-526013a0667c.jsonl",
]

CODEX_ROLLOUTS = [
    Path.home() / ".codex/sessions/2026/07/07/rollout-2026-07-07T04-40-06-019f3bbb-f2fb-7933-80c2-0af9d4e1fa23.jsonl",
    Path.home() / ".codex/sessions/2026/07/07/rollout-2026-07-07T10-35-43-019f3d01-8360-7f21-ad49-9c0694a13ca1.jsonl",
    Path.home() / ".codex/sessions/2026/07/07/rollout-2026-07-07T11-34-42-019f3d37-8726-7fa0-b0b2-4490009382f9.jsonl",
    # 2026-07-11: 2 more genuinely untouched sessions found -- of 34 total Codex sessions on
    # disk, 29 were already consumed by prod_extraction_v1 (verified via rollout-id grep
    # against hf-prod-v5n.jsonl); these 2 plus the 3 above are the only untouched ones.
    Path.home() / ".codex/sessions/2026/04/19/rollout-2026-04-19T07-51-08-019da594-88c6-7581-b280-eb2ae69160ea.jsonl",
    Path.home() / ".codex/sessions/2026/05/12/rollout-2026-05-12T05-00-58-019e1b6b-01e2-7dd1-ab34-b5df3176c390.jsonl",
]

CHATGPT_TECHNICAL_FILES = [
    "chatgpt_Blazor_vs_Angular_Comparison_20260607_091611.md",
    "chatgpt_Bulkhead_Pattern_in_C#_20260607_092956.md",
    "chatgpt_Kubectl_jq_watch_fix_20260607_092835.md",
    "chatgpt_DuckDB_JSON_query_20260607_093508.md",
    "chatgpt_Azure_Pipeline_Secret_Retrieval_20260607_092107.md",
    "chatgpt_Postgres_Docker_ASP.NET_setup_20260607_093722.md",
    "chatgpt_Fine-Tuning_Loss_Analysis_20260607_091405.md",
    "chatgpt_Strangler_Fig_Pattern_20260607_093003.md",
    "chatgpt_PowerShell_FileNotFoundError_20260607_092656.md",
    "chatgpt_YARP_Microsoft_Reverse_Proxy_20260607_092212.md",
]
CHATGPT_DIR = Path.home() / "Downloads/training-data/chatgpt_chats"

OUT = Path(__file__).resolve().parents[1] / "results" / "coding-agent-candidate-turns.jsonl"

MIN_CHARS = 80
MAX_CHARS = 2000


def _text_from_content_blocks(blocks) -> str:
    if not isinstance(blocks, list):
        return ""
    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("text", "output_text"):
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts)


def extract_claude_code(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "assistant":
            continue
        message = d.get("message") or {}
        if message.get("role") != "assistant":
            continue
        text = _text_from_content_blocks(message.get("content"))
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            out.append({"source": f"claude_code:{path.stem}:{i}", "turn_text": text})
    return out


def extract_codex(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "response_item":
            continue
        payload = d.get("payload") or {}
        if payload.get("type") != "message" or payload.get("role") != "assistant":
            continue
        text = _text_from_content_blocks(payload.get("content"))
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            out.append({"source": f"codex_session:{path.stem}:{i}", "turn_text": text})
    return out


def extract_chatgpt_md(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    # These exports are single-turn: one "### ... User ..." header, one question
    # paragraph, then the assistant's (possibly multi-paragraph, multi-heading)
    # response for the rest of the file.
    match = re.search(r"^###.*User.*$", raw, flags=re.MULTILINE)
    if not match:
        return []
    remainder = raw[match.end():].lstrip("\n")
    parts = remainder.split("\n\n", 1)
    if len(parts) < 2:
        return []
    text = parts[1].strip()
    text = text[:MAX_CHARS]
    if MIN_CHARS <= len(text):
        return [{"source": f"chatgpt_export:{path.stem}:0", "turn_text": text}]
    return []


def main() -> int:
    candidates: list[dict] = []
    for p in CLAUDE_CODE_SESSIONS:
        candidates.extend(extract_claude_code(p))
    for p in CODEX_ROLLOUTS:
        candidates.extend(extract_codex(p))
    for name in CHATGPT_TECHNICAL_FILES:
        candidates.extend(extract_chatgpt_md(CHATGPT_DIR / name))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in candidates:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_kind: dict[str, int] = {}
    for row in candidates:
        kind = row["source"].split(":")[0]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(json.dumps({"total": len(candidates), "by_kind": by_kind, "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
