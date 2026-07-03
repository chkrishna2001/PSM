#!/usr/bin/env python3
"""Merge teacher-labeled conversation JSONL shards into one train-ready file."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.row_validation import validate_prod_row

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-conversation-gemma-combined.jsonl"


def _row_key(row: dict) -> str:
    conv = row.get("input", {}).get("conversation")
    if isinstance(conv, list) and conv:
        return str(conv[0].get("content") or "").strip().lower()
    return json.dumps(row.get("input"), sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--store-only", action="store_true")
    args = parser.parse_args(argv)

    seen: set[str] = set()
    rows: list[dict] = []
    invalid = 0
    for path in args.inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                validate_prod_row(row)
            except ValueError:
                invalid += 1
                continue
            if args.store_only and str(row["expected"].get("action") or "") == "ignore":
                continue
            key = _row_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    actions = Counter(str(r["expected"].get("action") or "") for r in rows)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "rows": len(rows),
                "invalid_skipped": invalid,
                "actions": dict(actions),
                "store_only": args.store_only,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
