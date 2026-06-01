from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.data import validate_training_row
from psm_model.prompts import render_training_text


def prepare_jsonl(input_path: Path, output_path: Path, *, output_format: str = "tagged") -> int:
    written = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row, issues = validate_training_row(raw)
            if issues or row is None:
                formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
                raise ValueError(f"{input_path}:{line_number}: invalid row: {formatted}")
            target.write(
                json.dumps(
                    {
                        "id": row.id,
                        "text": render_training_text(raw["input"], raw["expected"], output_format=output_format),
                        "source": row.source,
                        "split": row.split,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            target.write("\n")
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare canonical PSM JSONL rows as prompt/completion text.")
    parser.add_argument("input", type=Path, help="Canonical JSONL rows with id/input/expected")
    parser.add_argument("output", type=Path, help="Output JSONL with id/text/source/split")
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag"], default="tagged")
    args = parser.parse_args()

    count = prepare_jsonl(args.input, args.output, output_format=args.output_format)
    print(json.dumps({"written": count, "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
