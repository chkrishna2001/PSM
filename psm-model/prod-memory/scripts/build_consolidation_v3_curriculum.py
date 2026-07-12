#!/usr/bin/env python3
"""Build hf-prod-consolidation-v3.jsonl -- gentler retry of the v2 store_episodic boost.

v2 (store_episodic dup'd 4x vs 1x for update/conflict, continued from v1, 150 steps) fixed the
always-update_existing shortcut (0/7 -> 5/7 correct) but overshot slightly, flipping 3
previously-correct update_existing cases to store_episodic (net 14/19 vs v1's 12/19 -- still a
real win, but this tries a lighter 2x boost to see if the boundary lands more precisely instead
of overshooting past it."""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
SRC_ROOT = PACKAGE_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_consolidation_rows import (  # noqa: E402
    _row,
    _TRAIN_CONFLICT_PAIRS,
    _TRAIN_STORE_PAIRS,
    _TRAIN_UPDATE_PAIRS,
)
from prod_memory.hf_prompts import row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-v3.jsonl"


def main() -> int:
    update_rows = [_row(rid, new, existing, action="update_existing") for rid, new, existing in _TRAIN_UPDATE_PAIRS]
    store_rows = [_row(rid, new, existing, action="store_episodic") for rid, new, existing in _TRAIN_STORE_PAIRS]
    conflict_rows = [_row(rid, new, existing, action="flag_conflict") for rid, new, existing in _TRAIN_CONFLICT_PAIRS]

    rows = list(update_rows) + list(store_rows) + list(conflict_rows)
    # Lighter boost than v2 (2x total vs v2's 4x) -- v2 overshot the boundary.
    rows.extend(_dup_rows(store_rows, prefix="consv3store", copies=1))

    hf_rows = []
    for row in rows:
        hf_rows.append({
            "id": row["id"],
            "task": row.get("task"),
            "messages": row_messages(row, output_format="json"),
            "source": row.get("source"),
        })
    write_jsonl(DEFAULT_OUT, hf_rows)
    print({
        "total_rows": len(hf_rows),
        "update_pairs": len(update_rows),
        "store_pairs_boosted": len(store_rows) * 2,
        "conflict_pairs": len(conflict_rows),
        "output": str(DEFAULT_OUT),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
