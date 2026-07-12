#!/usr/bin/env python3
"""Build hf-prod-consolidation-v6.jsonl -- methodology fix (same recipe as storage-v10),
not another data-volume test:

1. Reasoning-first JSON key order. Every prior round (v1-v5) used sort_keys=True
   (alphabetical), putting "action" before "reasoning" even though the schema requires
   both -- the model committed to the classification before articulating why. Fixed in
   `hf_prompts.compact_consolidation_json` (reasoning now first). json.loads() (used by
   every consumer, including eval_hf_consolidation.py) is order-independent, so this is
   safe without touching any parser.
2. Learning rate. v1-v5 all used 5e-6 (a small resume-patch LR); using 1e-4 instead,
   matching the recipe that worked for storage-v10.
3. Full retrain, not a resume-patch. v4/v5 resumed from v1's checkpoint for 120-150 steps
   -- an incremental nudge, not a full convergent pass. This trains from the base model on
   the FULL accumulated real dataset (same 91 hand-verified pairs as v4/v5: 34 update, 42
   store x2 boost, 15 conflict = 133 rows) for ~6 epochs.
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

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-v6.jsonl"


def main() -> int:
    update_rows = [_row(rid, new, existing, action="update_existing") for rid, new, existing in _TRAIN_UPDATE_PAIRS]
    store_rows = [_row(rid, new, existing, action="store_episodic") for rid, new, existing in _TRAIN_STORE_PAIRS]
    conflict_rows = [_row(rid, new, existing, action="flag_conflict") for rid, new, existing in _TRAIN_CONFLICT_PAIRS]

    rows = list(update_rows) + list(store_rows) + list(conflict_rows)
    rows.extend(_dup_rows(store_rows, prefix="consv6store", copies=1))

    hf_rows = []
    for row in rows:
        hf_rows.append({
            "id": row["id"],
            "task": row.get("task"),
            "messages": row_messages(row, output_format="json"),  # reasoning-first now
            "source": row.get("source"),
        })
    write_jsonl(DEFAULT_OUT, hf_rows)
    print({
        "total_rows": len(hf_rows),
        "update_pairs": len(update_rows),
        "store_pairs_boosted": len(store_rows) * 2,
        "conflict_pairs": len(conflict_rows),
        "output": str(DEFAULT_OUT),
        "format_change": "reasoning-first key order (was action-first alphabetical)",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
