#!/usr/bin/env python3
"""Build hf-prod-consolidation-v5.jsonl -- round 5: v4 (2x store boost, 133 rows) reached
0.826 (19/23) with update_existing and flag_conflict both saturated at 100% and all 4 residual
misses on the store_episodic side (topically-similar-but-factually-distinct pairs the model
still calls update_existing). Since the other two classes have headroom to absorb more
counter-pressure, this round pushes the store_episodic boost to 3x, continuing from v4."""
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

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-v5.jsonl"


def main() -> int:
    update_rows = [_row(rid, new, existing, action="update_existing") for rid, new, existing in _TRAIN_UPDATE_PAIRS]
    store_rows = [_row(rid, new, existing, action="store_episodic") for rid, new, existing in _TRAIN_STORE_PAIRS]
    conflict_rows = [_row(rid, new, existing, action="flag_conflict") for rid, new, existing in _TRAIN_CONFLICT_PAIRS]

    rows = list(update_rows) + list(store_rows) + list(conflict_rows)
    rows.extend(_dup_rows(store_rows, prefix="consv5store", copies=2))

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
        "store_pairs_boosted": len(store_rows) * 3,
        "conflict_pairs": len(conflict_rows),
        "output": str(DEFAULT_OUT),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
