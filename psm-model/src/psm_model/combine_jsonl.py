from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def combine_jsonl(inputs: list[Path], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id"))
            if row_id in seen:
                continue
            seen.add(row_id)
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "rows": len(rows),
        "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine JSONL training files, dropping duplicate ids.")
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    print(json.dumps(combine_jsonl(args.inputs, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
