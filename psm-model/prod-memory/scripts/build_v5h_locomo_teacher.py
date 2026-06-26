#!/usr/bin/env python3
"""Label LoCoMo D1 turns via OpenRouter teacher for v5h curriculum anchors."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prod_memory.openrouter_teacher import TeacherConfig, build_row_from_teacher

DEFAULT_DATA = ROOT.parent.parent / "benchmark" / "locomo" / "data" / "locomo10.json"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_OUT = ROOT / "data" / "hf-prod-v5h-locomo.jsonl"
DEFAULT_MODEL = "openai/gpt-4o"


def _ensure_data(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LOCOMO_URL, timeout=120) as resp:
        path.write_bytes(resp.read())


def _flatten_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = sample.get("conversation") or {}
    keys = sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", str(k))),
        key=lambda k: int(str(k).split("_")[1]),
    )
    turns: list[dict[str, Any]] = []
    for key in keys:
        block = conversation.get(key)
        if not isinstance(block, list):
            continue
        for turn in block:
            if isinstance(turn, dict):
                turns.append({**turn, "session": key})
    return turns


def _product_text(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "Unknown").strip()
    utterance = str(turn.get("text") or "").strip()
    bits = [
        f"Image query: {turn['query']}." if turn.get("query") else "",
        f"Image caption: {turn['blip_caption']}." if turn.get("blip_caption") else "",
    ]
    base = f'{speaker} said "{utterance}".' if utterance else f"{speaker} shared an image."
    extra = " ".join(b for b in bits if b).strip()
    return f"{base} {extra}".strip() if extra else base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--delay-sec", type=float, default=1.5)
    args = parser.parse_args(argv)

    _ensure_data(args.data)
    samples = json.loads(args.data.read_text(encoding="utf-8"))
    turns: list[dict[str, Any]] = []
    for sample in samples:
        turns.extend(_flatten_turns(sample))
    turns = turns[: args.limit]
    config = TeacherConfig.from_env(model=args.model)
    config = replace(config, request_delay_ms=0)
    if not config.api_key:
        raise SystemExit("OPENROUTER_API_KEY required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i, turn in enumerate(turns):
        text = _product_text(turn)
        row_id = f"locomo-d1-{i:02d}"
        row, meta = build_row_from_teacher(
            text,
            row_id=row_id,
            source_id=row_id,
            source_kind="locomo_dialogue",
            config=config,
        )
        if row is None:
            print(f"[{i + 1}/{len(turns)}] {row_id} SKIP {meta.get('error')}")
            continue
        row["source"] = "exp_a_locomo_teacher"
        rows.append(row)
        print(f"[{i + 1}/{len(turns)}] {row_id} -> {row['expected'].get('action')}")
        if i + 1 < len(turns):
            time.sleep(args.delay_sec)

    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
