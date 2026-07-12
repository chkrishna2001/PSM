#!/usr/bin/env python3
"""Build hf-prod-v5n-dpo4-calibration.jsonl -- targeted fix for v5n-dpo3's 6 remaining
coding-agent-gate misses: 2 over-store (tests-pass-nextstep, docs-agree-sanity-check, 3 new
ignore-seed examples added) and 4 under-store (parse-failure-finding, run-complete-hf,
hallucination-not-truncation, powershell-path-resolution, 10 new store-seed examples added).
Continues from v5n-dpo3, not a retrain from scratch."""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_v5n_dpo2_calibration_rows import (  # noqa: E402
    build_v5n_dpo2_calibration_pairs,
    build_v5n_dpo4_calibration_pairs,
)
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5n-dpo4-calibration.jsonl"


def main() -> int:
    overstore_pairs = build_v5n_dpo2_calibration_pairs()
    understore_pairs = build_v5n_dpo4_calibration_pairs()

    rows = list(overstore_pairs) + list(understore_pairs)
    rows.extend(_dup_rows(overstore_pairs, prefix="dpo4os", copies=1))
    rows.extend(_dup_rows(understore_pairs, prefix="dpo4us", copies=2))

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
        "profile": "hf-prod-v5n-dpo4-calibration",
        "output": str(DEFAULT_OUT),
        "total_rows": len(hf_rows),
        "overstore_seed_pairs": len(overstore_pairs),
        "understore_seed_pairs": len(understore_pairs),
    }
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
