#!/usr/bin/env python3
"""Build hf-prod-conversational-storage-v2.jsonl -- rebalance fix for v1's under-storing bias.

v1 gate result (419 held-out conv-26 cases): 65.9% action-match, store_recall only 86/189 (45.5%)
vs ignore_recall 218/230 (94.8%) -- heavily under-storing. v1's curriculum was 67% ignore / 33%
store (1641/481/325), noticeably below the real gate's own 55%/45% split.

This mirrors the exact two-part fix that got the coding-domain storage adapter past the same
failure mode (storage-v6's rebalance, then storage-v10's methodology fix):
  1. Duplicate the store-labeled subset (promote_semantic + store_episodic, 806 rows) one extra
     pass, lifting the store ratio from 33% to ~50% -- close to the real 45% without wild
     overcorrection, same conservative approach v6 used for its ignore-ratio fix.
  2. Pair this curriculum with storage-v10/v11's hyperparameters (LR 1e-4 instead of 1e-5, more
     steps) in the conversational-storage-v2 training profile -- v6's rebalance alone did NOT fix
     the coding-domain adapter; v10's LR bump was needed on top of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.row_validation import write_jsonl  # noqa: E402

BASE_CURRICULUM = PACKAGE_ROOT / "data" / "hf-prod-conversational-storage-v1.jsonl"
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-conversational-storage-v2.jsonl"


def _load_rows() -> list[dict]:
    rows = []
    with BASE_CURRICULUM.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _action(row: dict) -> str | None:
    for msg in row.get("messages", []):
        if msg.get("role") == "assistant":
            try:
                return json.loads(msg["content"]).get("action")
            except Exception:
                return None
    return None


def main() -> int:
    base_rows = _load_rows()
    store_rows = [r for r in base_rows if _action(r) != "ignore"]
    ignore_rows = [r for r in base_rows if _action(r) == "ignore"]
    print(f"base: {len(base_rows)} rows, {len(store_rows)} store ({len(store_rows) / len(base_rows):.1%})")

    store_dup = [{**r, "id": f"{r['id']}-dup1"} for r in store_rows]
    rows = base_rows + store_dup

    final_store = sum(1 for r in rows if _action(r) != "ignore")
    write_jsonl(DEFAULT_OUT, rows)
    manifest = {
        "profile": "hf-prod-conversational-storage-v2",
        "output": str(DEFAULT_OUT),
        "total_rows": len(rows),
        "base_rows": len(base_rows),
        "base_store_rows": len(store_rows),
        "base_ignore_rows": len(ignore_rows),
        "duplicated_store_rows": len(store_dup),
        "final_store_count": final_store,
        "final_store_ratio": round(final_store / len(rows), 4),
    }
    DEFAULT_OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
