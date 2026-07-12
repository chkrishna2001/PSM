#!/usr/bin/env python3
"""Build hf-prod-consolidation-v1.jsonl -- consolidation adapter v1 curriculum (26 hand-verified
rows from real LoCoMo observation pairs + synthesized flag_conflict examples). Standalone
adapter, not multi-tasked with storage or retrieval-plan."""
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
from prod_memory.build_consolidation_rows import build_consolidation_train_rows  # noqa: E402
from prod_memory.hf_prompts import row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-v1.jsonl"


def main() -> int:
    rows = list(build_consolidation_train_rows())
    # Small hand-verified set (26 rows) -- duplicate for a few epochs' worth of steps,
    # same convention as other small curricula this session (v5n-dpo2/3 calibration).
    rows.extend(_dup_rows(rows, prefix="cons2", copies=3))

    hf_rows = []
    for row in rows:
        hf_rows.append({
            "id": row["id"],
            "task": row.get("task"),
            "messages": row_messages(row, output_format="json"),
            "source": row.get("source"),
        })
    write_jsonl(DEFAULT_OUT, hf_rows)
    print({"total_rows": len(hf_rows), "output": str(DEFAULT_OUT)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
