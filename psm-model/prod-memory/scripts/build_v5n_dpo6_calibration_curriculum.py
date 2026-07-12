#!/usr/bin/env python3
"""Build hf-prod-v5n-dpo6-calibration.jsonl -- second retry of the over-store fix for
v5n-dpo3's 2 remaining ignore-case misses (tests-pass-nextstep, docs-agree-sanity-check).

v5n-dpo5 isolated the fix to JUST the 3 new batch-3 seed rows (dup'd 4x = 12 rows, 15 steps)
and regressed (0.65 -> 0.53): only 3 distinct examples wasn't enough diversity to generalize
the right feature (transient status narration) -- the model latched onto surface patterns of
those specific 3 texts and overcorrected broadly toward under-storing, flipping 3 previously-
correct store cases while still not fixing the 2 targets.

This round uses the FULL accumulated ignore-seed pool (all 31 rows, batches 1+2+3) -- the
same recipe that worked cleanly for v5n-dpo3 (batches 1+2 only, 28 rows) -- so the 3 new
examples are learned alongside 28 already-diverse, already-proven-safe examples instead of in
isolation. Resumes from v5n-dpo3 (which already generalizes batches 1+2 correctly), same
step/beta recipe as the v5n-dpo3 round that worked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_v5n_dpo2_calibration_rows import build_v5n_dpo2_calibration_pairs  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5n-dpo6-calibration.jsonl"


def main() -> int:
    pairs = build_v5n_dpo2_calibration_pairs()  # all 31 rows (batches 1+2+3)
    rows = list(pairs)
    rows.extend(_dup_rows(pairs, prefix="dpo6os", copies=1))  # 2x total, same as v5n-dpo3

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
        "profile": "hf-prod-v5n-dpo6-calibration",
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
