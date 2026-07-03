#!/usr/bin/env python3
"""Build hf-prod-v5o-sft.jsonl (conversation-only) and hf-prod-v5o-dpo.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_hf_curriculum import (  # noqa: E402
    MIN_STORAGE_P50_CHARS,
    V5D_BOOST_FIXTURE_IDS,
    _copy_rows,
    build_hf_curriculum,
)
from prod_memory.build_minimal_fixture_rows import build_json_fixture_rows  # noqa: E402
from prod_memory.build_v5o_storage_dpo_rows import build_v5o_storage_dpo_rows  # noqa: E402
from prod_memory.curriculum_sources import build_noise_rows  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

DEFAULT_CONVERSATION = PACKAGE_ROOT / "data" / "hf-prod-conversation-gemma-all.jsonl"
DEFAULT_SFT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5o-sft.jsonl"
DEFAULT_DPO_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5o-dpo.jsonl"


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_sft(out: Path, conversation: Path, *, store_copies: int) -> dict:
    if not conversation.is_file():
        raise SystemExit(f"missing conversation anchors: {conversation}")

    conversation_rows = _load_rows(conversation)
    anchors: list[dict] = list(conversation_rows)
    store_rows = [
        row
        for row in conversation_rows
        if str(row.get("expected", {}).get("action") or "") not in {"ignore", "ignore_noise"}
    ]
    if store_copies > 1 and store_rows:
        anchors.extend(_copy_rows(store_rows, prefix="conv", copies=store_copies - 1))

    anchors.extend(build_noise_rows())
    seed = build_json_fixture_rows()
    anchors.extend(seed)
    anchors.extend(_copy_rows(seed, prefix="fxo", copies=9))
    boost_seed = [row for row in seed if any(fid in row["id"] for fid in V5D_BOOST_FIXTURE_IDS)]
    if boost_seed:
        anchors.extend(_copy_rows(boost_seed, prefix="fxob", copies=35))

    return build_hf_curriculum(
        out,
        source=conversation,
        output_format="json",
        recall_fraction=0.0,
        min_input_chars=MIN_STORAGE_P50_CHARS,
        download=False,
        fixture_copies=0,
        profile="hf-prod-v5o",
        anchor_rows=anchors,
        ignore_fraction=0.0,
        simplify_labels=True,
        include_source_storage=False,
    )


def build_dpo(out: Path, conversation: Path) -> dict:
    dpo_rows = build_v5o_storage_dpo_rows(conversation_path=conversation)
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
        for row in dpo_rows
    ]
    write_jsonl(out, hf_rows)
    manifest = {
        "profile": "hf-prod-v5o-dpo",
        "conversation": str(conversation),
        "output": str(out),
        "total_rows": len(hf_rows),
        "format": "dpo",
        "variants": sorted({str(r.get("variant") or "") for r in dpo_rows}),
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation", type=Path, default=DEFAULT_CONVERSATION)
    parser.add_argument("--sft-out", type=Path, default=DEFAULT_SFT_OUT)
    parser.add_argument("--dpo-out", type=Path, default=DEFAULT_DPO_OUT)
    parser.add_argument("--store-copies", type=int, default=3)
    parser.add_argument("--sft-only", action="store_true")
    parser.add_argument("--dpo-only", action="store_true")
    args = parser.parse_args()

    result: dict = {}
    if not args.dpo_only:
        result["sft"] = build_sft(args.sft_out, args.conversation, store_copies=args.store_copies)
    if not args.sft_only:
        result["dpo"] = build_dpo(args.dpo_out, args.conversation)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
