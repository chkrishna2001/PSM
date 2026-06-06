from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from psm_model.train import ACTION_ORDER


def make_probe(input_path: Path, output_path: Path, *, per_action: int) -> dict[str, object]:
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        buckets[row["expected"]["action"]].append(row)

    selected: list[dict[str, object]] = []
    for action in ACTION_ORDER:
        selected.extend(buckets[action][:per_action])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in selected) + "\n",
        encoding="utf-8",
    )
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(selected),
        "per_action": per_action,
        "actions": {action: sum(1 for row in selected if row["expected"]["action"] == action) for action in ACTION_ORDER},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a balanced action probe from a PSM JSONL split.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-action", type=int, default=10)
    args = parser.parse_args()

    print(json.dumps(make_probe(args.input, args.output, per_action=args.per_action), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
