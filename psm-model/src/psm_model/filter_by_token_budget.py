from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psm_model.prompts import render_training_text
from psm_model.tokenizer import load_tokenizer


def filter_jsonl_by_token_budget(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer_path: Path,
    max_tokens: int,
    output_format: str = "tagged",
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    kept = 0
    dropped = 0
    lengths: list[int] = []
    dropped_rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as input_handle, output_path.open("w", encoding="utf-8") as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = render_training_text(row["input"], row["expected"], output_format=output_format)
            token_count = len(tokenizer.encode(text, add_bos=True, add_eos=True))
            lengths.append(token_count)
            if token_count <= max_tokens:
                output_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                kept += 1
            else:
                dropped += 1
                if len(dropped_rows) < 25:
                    dropped_rows.append({"id": row.get("id"), "line": line_number, "tokens": token_count})
    lengths.sort()
    return {
        "input": str(input_path),
        "output": str(output_path),
        "tokenizer": str(tokenizer_path),
        "max_tokens": max_tokens,
        "rows": len(lengths),
        "kept": kept,
        "dropped": dropped,
        "min_tokens": lengths[0] if lengths else 0,
        "p50_tokens": _percentile(lengths, 0.50),
        "p90_tokens": _percentile(lengths, 0.90),
        "p95_tokens": _percentile(lengths, 0.95),
        "p99_tokens": _percentile(lengths, 0.99),
        "max_tokens_seen": lengths[-1] if lengths else 0,
        "sample_dropped": dropped_rows,
    }


def _percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, int(len(values) * ratio))
    return values[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter psm-model JSONL rows to a tokenizer/context token budget.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2049)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"], default="tagged")
    args = parser.parse_args()

    report = filter_jsonl_by_token_budget(
        args.input,
        args.output,
        tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
        output_format=args.output_format,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["kept"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
