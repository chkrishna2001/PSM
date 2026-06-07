from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        expected = row.get("expected")
        if not isinstance(expected, dict) or not isinstance(expected.get("action"), str):
            raise ValueError(f"{path}:{line_number}: missing expected.action")
        if not isinstance(row.get("input"), dict):
            raise ValueError(f"{path}:{line_number}: missing input object")
        rows.append(row)
    return rows


def _copy_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    copies: int,
    seen: set[str],
    output: list[dict[str, Any]],
    action_filter: set[str] | None = None,
) -> int:
    added = 0
    for row in rows:
        action = str(row["expected"]["action"])
        if action_filter is not None and action not in action_filter:
            continue
        row_id = str(row.get("id") or row.get("case") or "row")
        for copy_index in range(copies):
            copied_id = f"{prefix}:{copy_index}:{row_id}"
            if copied_id in seen:
                continue
            seen.add(copied_id)
            output.append(
                {
                    "id": copied_id,
                    "input": row["input"],
                    "expected": row["expected"],
                    "source": f"gate4_curriculum:{prefix}",
                }
            )
            added += 1
    return added


def build_gate4_curriculum(
    base: Path,
    output: Path,
    *,
    direct_probes: Path,
    expanded_probes: Path,
    direct_copies: int = 500,
    expanded_copies: int = 8,
    ignore_extra_copies: int = 4,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_count = 0

    for row in _load_rows(base):
        row_id = str(row.get("id") or f"{base.stem}-{base_count}")
        if row_id in seen:
            continue
        seen.add(row_id)
        rows.append(row)
        base_count += 1

    direct_rows = _load_rows(direct_probes)
    expanded_rows = _load_rows(expanded_probes)

    anchor_added = _copy_rows(
        direct_rows,
        prefix="direct-anchor",
        copies=direct_copies,
        seen=seen,
        output=rows,
    )
    expanded_added = _copy_rows(
        expanded_rows,
        prefix="expanded-anchor",
        copies=expanded_copies,
        seen=seen,
        output=rows,
    )
    ignore_added = _copy_rows(
        expanded_rows,
        prefix="expanded-ignore",
        copies=ignore_extra_copies,
        seen=seen,
        output=rows,
        action_filter={"ignore"},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "base": str(base),
        "direct_probes": str(direct_probes),
        "expanded_probes": str(expanded_probes),
        "output": str(output),
        "rows": len(rows),
        "base_rows": base_count,
        "direct_anchor_rows": anchor_added,
        "expanded_anchor_rows": expanded_added,
        "ignore_extra_rows": ignore_added,
        "direct_copies": direct_copies,
        "expanded_copies": expanded_copies,
        "ignore_extra_copies": ignore_extra_copies,
        "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Gate 4 curriculum: full-storage base + direct/expanded anchors + ignore oversample."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", type=Path, required=True, help="Filtered full-storage JSONL.")
    parser.add_argument("--direct-probes", type=Path, required=True)
    parser.add_argument("--expanded-probes", type=Path, required=True)
    parser.add_argument("--direct-copies", type=int, default=500)
    parser.add_argument("--expanded-copies", type=int, default=8)
    parser.add_argument("--ignore-extra-copies", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            build_gate4_curriculum(
                args.base,
                args.output,
                direct_probes=args.direct_probes,
                expanded_probes=args.expanded_probes,
                direct_copies=args.direct_copies,
                expanded_copies=args.expanded_copies,
                ignore_extra_copies=args.ignore_extra_copies,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
