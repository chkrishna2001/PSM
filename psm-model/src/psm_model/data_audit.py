from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from psm_model.prompts import render_expected_output, render_training_text
from psm_model.train import DEFAULT_REAL_TOKENIZER
from psm_model.tokenizer import load_tokenizer


def audit_dataset(path: Path, *, tokenizer_path: Path, output_format: str = "tagged", opening_words: int = 8) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    action_counts: Counter[str] = Counter()
    memory_counts: Counter[str] = Counter()
    train_tokens: Counter[str] = Counter()
    output_tokens: Counter[str] = Counter()
    openings: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        expected = row["expected"]
        action = expected["action"]
        memory = expected.get("memory")
        conversation = str(row.get("input", {}).get("conversation", ""))
        opening = " ".join(conversation.split()[:opening_words])
        action_counts[action] += 1
        memory_counts[memory["type"] if memory else "none"] += 1
        openings[action][opening] += 1
        train_tokens[action] += len(tokenizer.encode(render_training_text(row["input"], expected, output_format=output_format), add_bos=True, add_eos=True))
        output_tokens[action] += len(tokenizer.encode(render_expected_output(expected, output_format=output_format), add_eos=True))

    action_report: dict[str, Any] = {}
    for action in sorted(action_counts):
        top_openings = openings[action].most_common(3)
        top3 = sum(count for _, count in top_openings)
        action_report[action] = {
            "rows": action_counts[action],
            "train_tokens": train_tokens[action],
            "output_tokens": output_tokens[action],
            "avg_output_tokens": output_tokens[action] / action_counts[action],
            "unique_openings": len(openings[action]),
            "top3_opening_fraction": top3 / action_counts[action],
            "top_openings": [{"opening": opening, "count": count} for opening, count in top_openings],
        }

    return {
        "path": str(path),
        "tokenizer": str(tokenizer_path),
        "output_format": output_format,
        "rows": len(rows),
        "actions": action_report,
        "memory_types": dict(sorted(memory_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit PSM training data balance and rough template diversity.")
    parser.add_argument("data", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_REAL_TOKENIZER)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag"], default="tagged")
    parser.add_argument("--opening-words", type=int, default=8)
    args = parser.parse_args()

    print(
        json.dumps(
            audit_dataset(args.data, tokenizer_path=args.tokenizer, output_format=args.output_format, opening_words=args.opening_words),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
