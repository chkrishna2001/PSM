#!/usr/bin/env python3
"""Build hf-prod-v5n-dpo2-calibration.jsonl — targeted DPO fix for v5n-dpo's over-storing
bias (corrected 2026-07-09 eval) plus two known schema bugs, as a small second DPO pass
resuming from hf-prod-v5n-dpo-qwen0.5b (not a retrain from scratch)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_v5n_dpo2_calibration_rows import build_v5n_dpo2_calibration_pairs  # noqa: E402
from prod_memory.build_v5q_enum_dpo_rows import build_v5q_enum_dpo_rows  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5n-dpo3-calibration.jsonl"

# Schema-bug variants worth defensive hardening against, even though v5n-dpo doesn't
# currently exhibit them (v5q-dpo does) -- cheap, well-understood, orthogonal to the
# over-storing fix. Deliberately NOT pulling in the full v5q_enum_dpo_rows() set (the
# grounded/empty memory-type and semantic/fact/explicit indexable-kind variants) since
# v5n-dpo has never shown those specific hallucinations -- keep this pass narrow.
SCHEMA_BUG_VARIANTS = {"missing_memory_reasoning", "bad_indexable_kind_episodic"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overstore-copies", type=int, default=2)
    parser.add_argument("--schema-bug-copies", type=int, default=2)
    args = parser.parse_args()

    overstore_pairs = build_v5n_dpo2_calibration_pairs()
    schema_pairs = [
        p for p in build_v5q_enum_dpo_rows(fixture_copies=1, failure_copies=0)
        if p.get("variant") in SCHEMA_BUG_VARIANTS
    ]

    rows = list(overstore_pairs)
    rows.extend(_dup_rows(overstore_pairs, prefix="dpo2os", copies=args.overstore_copies - 1))
    rows.extend(schema_pairs)
    rows.extend(_dup_rows(schema_pairs, prefix="dpo2sb", copies=args.schema_bug_copies - 1))

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
        "profile": "hf-prod-v5n-dpo2-calibration",
        "output": str(args.out),
        "total_rows": len(hf_rows),
        "overstore_seed_pairs": len(overstore_pairs),
        "schema_bug_seed_pairs": len(schema_pairs),
        "format": "dpo",
        "variants": sorted({str(r.get("variant") or "") for r in rows}),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
