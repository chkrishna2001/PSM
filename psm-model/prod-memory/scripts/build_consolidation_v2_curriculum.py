#!/usr/bin/env python3
"""Build hf-prod-consolidation-v2.jsonl -- fix consolidation's store_episodic vs
update_existing boundary. attempt 3 (hf-prod-consolidation-v1-qwen0.5b) scores 100% on
update_existing (8/8) and flag_conflict (4/4) but 0/7 on store_episodic -- the model has
learned a shortcut of always predicting update_existing whenever an existing memory is shown,
regardless of whether the new memory actually restates it. A prior DPO attempt to fix this
collapsed the model (homogeneous chosen-text) and was reverted.

This is the same systematic-bias shape as retrieval-plan's target_tables_exact gap, which was
fixed cleanly (zero regressions) via SFT-boost: duplicate the underrepresented/weak class more
heavily and continue-train from the existing checkpoint, rather than a DPO contrastive pass.
Here store_episodic isn't underrepresented in raw count (32 vs 29 update pairs) -- the bias is
a learned shortcut, not a data-volume gap -- so this boosts store_episodic harder than the
other two classes to counter-weight it specifically.
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

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-v2.jsonl"


def main() -> int:
    update_rows = [_row(rid, new, existing, action="update_existing") for rid, new, existing in _TRAIN_UPDATE_PAIRS]
    store_rows = [_row(rid, new, existing, action="store_episodic") for rid, new, existing in _TRAIN_STORE_PAIRS]
    conflict_rows = [_row(rid, new, existing, action="flag_conflict") for rid, new, existing in _TRAIN_CONFLICT_PAIRS]

    rows = list(update_rows) + list(store_rows) + list(conflict_rows)
    # Keep update/conflict at the existing 1x (already saturated -- 100% on both in attempt 3);
    # boost store_episodic 3x extra to counter-weight the update_existing shortcut.
    rows.extend(_dup_rows(store_rows, prefix="consv2store", copies=3))

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
        "store_pairs_boosted": len(store_rows) * 4,
        "conflict_pairs": len(conflict_rows),
        "output": str(DEFAULT_OUT),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
