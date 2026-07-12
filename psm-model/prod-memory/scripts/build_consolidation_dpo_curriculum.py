#!/usr/bin/env python3
"""Build hf-prod-consolidation-dpo-v1.jsonl -- targeted DPO fix for the store_episodic vs
update_existing confusion diagnosed after 3 SFT rounds. Continues from the SFT checkpoint,
not a retrain from scratch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
SRC_ROOT = PACKAGE_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_consolidation_dpo_rows import build_consolidation_dpo_pairs  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-consolidation-dpo-v1.jsonl"


def main() -> int:
    pairs = build_consolidation_dpo_pairs()
    rows = list(pairs)
    rows.extend(_dup_rows(pairs, prefix="consdpo", copies=3))

    hf_rows = [
        {
            "id": row["id"],
            "task": "consolidation_dpo",
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "source": row.get("source"),
            "variant": row.get("variant"),
        }
        for row in rows
    ]
    write_jsonl(DEFAULT_OUT, hf_rows)
    manifest = {
        "profile": "hf-prod-consolidation-dpo-v1",
        "output": str(DEFAULT_OUT),
        "total_rows": len(hf_rows),
        "seed_pairs": len(pairs),
        "format": "dpo",
    }
    DEFAULT_OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
