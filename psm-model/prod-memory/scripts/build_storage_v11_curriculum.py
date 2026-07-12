#!/usr/bin/env python3
"""Build hf-prod-storage-v11.jsonl -- targeted fix for v10's residual under-storing pattern
(0.7059, 12/17; all 5 misses were false negatives on terse-but-substantive findings/decisions
misclassified as transient narration). Not a methodology change (v10 already fixed reasoning
order/LR/epochs) -- this adds 22 new real, hand-labeled examples (15 store, 7 ignore) targeting
that exact boundary, mined from the same provenance-checked candidate pool used for the eval
gate and prior training rounds (see build_storage_v11_rows.py for selection detail and
contamination-avoidance notes).

v10's `hf-prod-storage-v10.jsonl` is already in the reasoning-first format (built by
`build_storage_v10_curriculum.py`), so this script just loads it directly and appends the new
rows -- no re-rendering needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_storage_v11_rows import build_storage_v11_new_rows  # noqa: E402
from prod_memory.hf_prompts import row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

V10_CURRICULUM = PACKAGE_ROOT / "data" / "hf-prod-storage-v10.jsonl"
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-storage-v11.jsonl"


def _load_v10_rows() -> list[dict]:
    rows = []
    with V10_CURRICULUM.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_ignore(hf_row: dict) -> bool:
    for msg in hf_row.get("messages", []):
        if msg.get("role") == "assistant":
            try:
                return json.loads(msg["content"]).get("action") == "ignore"
            except Exception:
                return False
    return False


def main() -> int:
    v10_rows = _load_v10_rows()
    v10_ignore = sum(1 for r in v10_rows if _is_ignore(r))
    print(f"v10 base: {len(v10_rows)} rows, {v10_ignore} ignore ({v10_ignore/len(v10_rows):.1%})")

    new_rows = build_storage_v11_new_rows()

    def to_hf(row: dict) -> dict:
        return {
            "id": row["id"],
            "task": "storage",
            "messages": row_messages(row, output_format="json"),  # reasoning-first
            "source": row.get("source"),
        }

    new_hf = [to_hf(r) for r in new_rows]
    new_ignore = sum(1 for r in new_hf if _is_ignore(r))
    print(f"new v11 rows: {len(new_hf)} ({new_ignore} ignore, {len(new_hf)-new_ignore} store)")

    rows = list(v10_rows) + new_hf
    final_ignore = sum(1 for r in rows if _is_ignore(r))
    write_jsonl(DEFAULT_OUT, rows)
    manifest = {
        "profile": "hf-prod-storage-v11",
        "output": str(DEFAULT_OUT),
        "total_rows": len(rows),
        "v10_base_rows": len(v10_rows),
        "new_rows": len(new_hf),
        "new_store_rows": len(new_hf) - new_ignore,
        "new_ignore_rows": new_ignore,
        "final_ignore_count": final_ignore,
        "final_ignore_ratio": round(final_ignore / len(rows), 4),
        "change": "22 new hand-labeled examples targeting terse-finding-vs-narration boundary "
        "(v10's 5 primary-gate misses were all false negatives of this exact shape)",
    }
    DEFAULT_OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
