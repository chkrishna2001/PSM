from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def make_action_first_curriculum(inputs: list[Path], output: Path, *, copies: int = 1) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in inputs:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected = row.get("expected")
            if not isinstance(expected, dict) or not isinstance(expected.get("action"), str):
                raise ValueError(f"{path}:{line_number}: missing expected.action")
            input_payload = row.get("input")
            if not isinstance(input_payload, dict):
                raise ValueError(f"{path}:{line_number}: missing input object")
            row_id = str(row.get("id") or row.get("case") or f"{path.stem}-{line_number}")
            for copy_index in range(copies):
                action_row_id = f"action-first:{copy_index}:{row_id}"
                if action_row_id in seen:
                    continue
                seen.add(action_row_id)
                rows.append(
                    {
                        "id": action_row_id,
                        "input": input_payload,
                        "expected": {
                            "action": expected["action"],
                            "memory": None,
                            "facts": [],
                            "reasoning": "Action-only curriculum row.",
                        },
                        "source": "action_first_curriculum",
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "rows": len(rows),
        "copies": copies,
        "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build action-only rows for 50M action-first decoder training.")
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--copies", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(make_action_first_curriculum(args.inputs, args.output, copies=args.copies), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
