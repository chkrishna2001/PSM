#!/usr/bin/env python3
"""Build hf-prod-v5p.jsonl — v5n-style v3 mix + full conversation pool, no label simplification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_hf_curriculum import (  # noqa: E402
    DEFAULT_IGNORE_FRACTION_V5,
    DEFAULT_SOURCE,
    MIN_STORAGE_P50_CHARS,
    V5D_BOOST_FIXTURE_IDS,
    _copy_rows,
    build_hf_curriculum,
)
from prod_memory.build_minimal_fixture_rows import build_json_fixture_rows  # noqa: E402

DEFAULT_CONVERSATION = PACKAGE_ROOT / "data" / "hf-prod-conversation-gemma-all.jsonl"
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5p.jsonl"


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation", type=Path, default=DEFAULT_CONVERSATION)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--store-copies", type=int, default=1, help="Extra copies of non-ignore conversation rows.")
    args = parser.parse_args()

    if not args.conversation.is_file():
        raise SystemExit(f"missing conversation anchors: {args.conversation}")

    conversation = _load_rows(args.conversation)
    anchors: list[dict] = list(conversation)
    store_rows = [
        row
        for row in conversation
        if str(row.get("expected", {}).get("action") or "") not in {"ignore", "ignore_noise"}
    ]
    if args.store_copies > 1 and store_rows:
        anchors.extend(_copy_rows(store_rows, prefix="convp", copies=args.store_copies - 1))

    seed = build_json_fixture_rows()
    anchors.extend(seed)
    anchors.extend(_copy_rows(seed, prefix="fxp", copies=5))
    boost_seed = [row for row in seed if any(fid in row["id"] for fid in V5D_BOOST_FIXTURE_IDS)]
    if boost_seed:
        anchors.extend(_copy_rows(boost_seed, prefix="fxpb", copies=15))

    manifest = build_hf_curriculum(
        args.out,
        source=args.source,
        output_format="json",
        recall_fraction=0.0,
        min_input_chars=MIN_STORAGE_P50_CHARS,
        download=False,
        fixture_copies=0,
        profile="hf-prod-v5p",
        anchor_rows=anchors,
        ignore_fraction=DEFAULT_IGNORE_FRACTION_V5,
        simplify_labels=False,
        include_source_storage=True,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
