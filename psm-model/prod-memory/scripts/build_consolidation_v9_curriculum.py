#!/usr/bin/env python3
"""Build hf-prod-consolidation-v9.jsonl -- round 6: v8 (this session) removed v4's 2x
store_episodic boost entirely while adding 10 new real update_existing pairs, expecting the
extra real data to make the boost unnecessary. Result: 0.652, same aggregate as v7 but a
mirror-image failure pattern -- update_existing improved to 8/10 (close to v4's 10/10) but
store_episodic collapsed to 3/9 (worse than v4's 6/9). This proves the boost was load-bearing,
not a crutch the extra data could replace outright.

v9 isolates the one remaining real variable: keep v4's exact proven recipe AND its 2x
store_episodic boost, and ONLY add the 10 new real update_existing pairs on top (44 total vs
v4's 34). If this improves on v4's 0.826 without disturbing update_existing/flag_conflict's
100%, the extra real data was the missing piece; if not, consolidation's ceiling right now is
genuinely the data itself, not the boost ratio.
"""
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

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-v9.jsonl"


def main() -> int:
    update_rows = [_row(rid, new, existing, action="update_existing") for rid, new, existing in _TRAIN_UPDATE_PAIRS]
    store_rows = [_row(rid, new, existing, action="store_episodic") for rid, new, existing in _TRAIN_STORE_PAIRS]
    conflict_rows = [_row(rid, new, existing, action="flag_conflict") for rid, new, existing in _TRAIN_CONFLICT_PAIRS]

    rows = list(update_rows) + list(store_rows) + list(conflict_rows)
    rows.extend(_dup_rows(store_rows, prefix="consv9store", copies=1))  # same 2x boost as v4

    hf_rows = []
    for row in rows:
        hf_rows.append({
            "id": row["id"],
            "task": row.get("task"),
            "messages": row_messages(row, output_format="json"),  # reasoning-first
            "source": row.get("source"),
        })
    write_jsonl(DEFAULT_OUT, hf_rows)
    print({
        "total_rows": len(hf_rows),
        "update_pairs": len(update_rows),
        "store_pairs_boosted": len(store_rows) * 2,
        "conflict_pairs": len(conflict_rows),
        "output": str(DEFAULT_OUT),
        "change": "44 update pairs (10 new) + v4's exact 2x store boost, same recipe as v4",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
