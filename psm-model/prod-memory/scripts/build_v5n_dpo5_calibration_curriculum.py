#!/usr/bin/env python3
"""Build hf-prod-v5n-dpo5-calibration.jsonl -- isolated retry of the over-store fix for
v5n-dpo3's 2 remaining ignore-case misses (tests-pass-nextstep, docs-agree-sanity-check).

v5n-dpo4 combined this direction with an understore fix in one round and regressed (4
previously-fixed ignore-cases flipped back to wrongly stored); the lesson recorded was to fix
one direction per round. This round trains ONLY the 3 new batch-3 ignore seeds that v5n-dpo3's
frozen curriculum never saw, resuming from v5n-dpo3, with nothing else changed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_v5n_dpo2_calibration_rows import build_v5n_dpo5_calibration_pairs  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5n-dpo5-calibration.jsonl"


def main() -> int:
    pairs = build_v5n_dpo5_calibration_pairs()
    rows = list(pairs)
    rows.extend(_dup_rows(pairs, prefix="dpo5os", copies=3))

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
    write_jsonl(DEFAULT_OUT, hf_rows)
    manifest = {
        "profile": "hf-prod-v5n-dpo5-calibration",
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
