from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _iter_locomo_utterances(locomo: Any) -> set[str]:
    utterances: set[str] = set()
    if isinstance(locomo, list):
        dialogues = locomo
    elif isinstance(locomo, dict):
        dialogues = locomo.get("dialogues", [])
    else:
        dialogues = []
    for dialogue in dialogues:
        if not isinstance(dialogue, dict):
            continue
        conversation = dialogue.get("conversation", [])
        if not isinstance(conversation, list):
            continue
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            text = turn.get("text")
            if isinstance(text, str) and text.strip():
                utterances.add(_normalize_text(text))
    return utterances


def _contains_locomo_phrase(conversation: str, utterances: set[str]) -> bool:
    normalized = _normalize_text(conversation)
    if normalized in utterances:
        return True
    for phrase in utterances:
        if len(phrase) < 30:
            continue
        if phrase in normalized:
            return True
    return False


def filter_rows(rows_path: Path, locomo_path: Path, output_path: Path) -> dict[str, Any]:
    locomo = json.loads(locomo_path.read_text(encoding="utf-8"))
    utterances = _iter_locomo_utterances(locomo)
    kept: list[dict[str, Any]] = []
    dropped = 0

    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            input_payload = row.get("input", {})
            conversation = str(input_payload.get("conversation") or "")
            source_id = str(input_payload.get("source_id") or "")
            row_id = str(row.get("id") or "")
            if "conv-26:d" in source_id.lower() or "conv-26:d" in row_id.lower():
                dropped += 1
                continue
            if conversation and _contains_locomo_phrase(conversation, utterances):
                dropped += 1
                continue
            kept.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept), encoding="utf-8")
    return {
        "input": str(rows_path),
        "locomo": str(locomo_path),
        "output": str(output_path),
        "rows_kept": len(kept),
        "rows_dropped": dropped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter LoCoMo holdout overlap from curriculum rows.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--locomo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = filter_rows(args.rows, args.locomo, args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["rows_kept"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
