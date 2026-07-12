#!/usr/bin/env python3
"""Build hf-prod-consolidation-v8.jsonl -- round 5: fix the update_existing recall gap with
more real DATA instead of another LR/methodology change, after v6/v7 (this session's
storage-style full-retrain-at-high-LR fix) regressed hard (0.826 -> 0.565 -> partial recovery
0.652) by collapsing a small, artificially-boosted 133-row curriculum toward the majority class.

Two changes from v4/v6/v7, isolating "does real data help" as the only new variable:
1. Ten new real, hand-verified update_existing pairs mined from conv-30 (Jon/Gina) and conv-41
   (John/Maria) -- two LoCoMo conversations untouched by any prior consolidation OR
   retrieval-plan training/eval, so no contamination risk. Grows update_existing from 34 to 44
   real pairs (see build_consolidation_rows.py "Fifth batch" for the pairs and provenance).
2. No store_episodic boost. v1-v7 all used a 2x (or 3x) artificial store_episodic boost to
   counter a *different* collapse direction discovered in v1 (always-predict update_existing).
   With 44 real update pairs now roughly matching the 42 real store pairs, the natural,
   unboosted data is already close to balanced -- re-applying the old boost on top would
   just re-introduce the store_episodic-dominant skew that caused v6's collapse.

Recipe: matches v4's own proven-safe hyperparameters exactly (resume from v1, learning_rate
5e-6) rather than v10's storage-style full-retrain-at-1e-4 recipe, since this session's lesson
was that the aggressive recipe only works on large/diverse curricula -- consolidation's is
still small. Step count is scaled down from v4's 120 (133 rows) to keep the same epoch count
on this run's 101 rows.
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

from prod_memory.build_consolidation_rows import (  # noqa: E402
    _row,
    _TRAIN_CONFLICT_PAIRS,
    _TRAIN_STORE_PAIRS,
    _TRAIN_UPDATE_PAIRS,
)
from prod_memory.hf_prompts import row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-v8.jsonl"


def main() -> int:
    update_rows = [_row(rid, new, existing, action="update_existing") for rid, new, existing in _TRAIN_UPDATE_PAIRS]
    store_rows = [_row(rid, new, existing, action="store_episodic") for rid, new, existing in _TRAIN_STORE_PAIRS]
    conflict_rows = [_row(rid, new, existing, action="flag_conflict") for rid, new, existing in _TRAIN_CONFLICT_PAIRS]

    rows = list(update_rows) + list(store_rows) + list(conflict_rows)  # no boost this round

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
        "store_pairs_unboosted": len(store_rows),
        "conflict_pairs": len(conflict_rows),
        "output": str(DEFAULT_OUT),
        "change": "10 new real update_existing pairs (conv-30/conv-41), no store_episodic boost",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
