#!/usr/bin/env python3
"""Build hf-prod-v5q-sft.jsonl — indexables + temporal micro-curriculum (Phase 2)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_hf_curriculum import build_hf_curriculum  # noqa: E402
from prod_memory.build_v5q_indexables_rows import build_v5q_anchor_rows  # noqa: E402
from prod_memory.row_validation import validate_prod_row  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5q-sft.jsonl"
DEFAULT_FIXTURES = PACKAGE_ROOT / "fixtures" / "cases.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--boost-copies", type=int, default=24)
    args = parser.parse_args()

    anchors = build_v5q_anchor_rows(boost_copies=args.boost_copies)
    for row in anchors:
        validate_prod_row(row)

    manifest = build_hf_curriculum(
        args.out,
        source=DEFAULT_FIXTURES,
        output_format="json",
        recall_fraction=0.0,
        min_input_chars=0,
        download=False,
        fixture_copies=0,
        profile="hf-prod-v5q",
        anchor_rows=anchors,
        ignore_fraction=0.0,
        simplify_labels=False,
        include_source_storage=False,
    )
    with_idx = sum(
        1
        for row in anchors
        if row.get("expected", {}).get("indexables")
    )
    manifest["anchor_rows"] = len(anchors)
    manifest["anchors_with_indexables"] = with_idx
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
