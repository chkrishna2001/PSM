#!/usr/bin/env python3
"""Build hf-prod-v5q-dpo.jsonl — enum-contrast DPO on top of v5q SFT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_v5q_enum_dpo_rows import (  # noqa: E402
    IGNORE_ENUM_VARIANTS,
    STORE_ENUM_VARIANTS,
    V5Q_ENUM_FAILURE_FIXTURE_IDS,
    build_v5q_enum_dpo_rows,
)
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5q-dpo.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture-copies", type=int, default=8)
    parser.add_argument("--failure-copies", type=int, default=6)
    args = parser.parse_args()

    rows = build_v5q_enum_dpo_rows(
        fixture_copies=args.fixture_copies,
        failure_copies=args.failure_copies,
    )
    hf_rows = [
        {
            "id": row["id"],
            "task": "storage_dpo",
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "source": row.get("source"),
            "variant": row.get("variant"),
        }
        for row in rows
    ]
    write_jsonl(args.out, hf_rows)
    manifest = {
        "profile": "hf-prod-v5q-dpo",
        "output": str(args.out),
        "total_rows": len(hf_rows),
        "format": "dpo",
        "base_adapter": "hf-prod-v5q-qwen0.5b",
        "failure_fixture_ids": sorted(V5Q_ENUM_FAILURE_FIXTURE_IDS),
        "store_variants": list(STORE_ENUM_VARIANTS),
        "ignore_variants": list(IGNORE_ENUM_VARIANTS),
        "variants": sorted({str(r.get("variant") or "") for r in rows}),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
