#!/usr/bin/env python3
"""Build hf-prod-v5l-gate.jsonl — binary gate with LoCoMo QA-evidence labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prod_memory.build_hf_curriculum import MIN_STORAGE_V5_CHARS, build_hf_curriculum
from prod_memory.build_locomo_gate_rows import DEFAULT_DATA, build_v5l_gate_rows

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5l-gate.jsonl"
DEFAULT_SOURCE = PACKAGE_ROOT / "data" / "hf-prod-v3.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    anchors = build_v5l_gate_rows(locomo_path=args.data)
    manifest = build_hf_curriculum(
        args.out,
        source=args.source,
        output_format="binary",
        recall_fraction=0.0,
        min_input_chars=MIN_STORAGE_V5_CHARS,
        download=False,
        fixture_copies=0,
        profile="hf-prod-v5l-gate",
        anchor_rows=anchors,
        include_source_storage=False,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
