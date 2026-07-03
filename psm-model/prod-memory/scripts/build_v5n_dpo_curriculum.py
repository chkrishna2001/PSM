#!/usr/bin/env python3
"""Build hf-prod-v5n-dpo.jsonl — StorageDecision DPO from v5n base fixtures + failures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_minimal_fixture_rows import build_json_fixture_rows  # noqa: E402
from prod_memory.build_v5o_storage_dpo_rows import _dpo_pair, _load_rows  # noqa: E402
from prod_memory.build_v5o_storage_dpo_rows import (  # noqa: E402
    _malformed_rejected,
    _truncated_rejected,
    _wrong_action_rejected,
)
from prod_memory.storage_rewards import compact_expected_json  # noqa: E402
from prod_memory.row_validation import remember_target_from_input, validate_prod_row, write_jsonl  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-v5n-dpo.jsonl"
FIXTURES = PACKAGE_ROOT / "fixtures" / "cases.json"


def build_v5n_fixture_dpo_rows(*, fixture_copies: int = 8) -> list[dict]:
    """DPO pairs from prod fixtures only — target truncation + wrong action on store cases."""
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    seed_rows = build_json_fixture_rows()
    pairs: list[dict] = []
    for row in seed_rows:
        try:
            validate_prod_row(row)
        except ValueError:
            continue
        remember_target = remember_target_from_input(row["input"])
        expected = row["expected"]
        row_id = str(row.get("id") or "fx")
        chosen_raw = compact_expected_json(expected)
        variants = {
            "wrong_action": _wrong_action_rejected(expected),
            "truncated": _truncated_rejected(chosen_raw),
            "malformed": _malformed_rejected(chosen_raw),
        }
        for variant, rejected_raw in variants.items():
            pair = _dpo_pair(
                row_id,
                remember_target,
                expected,
                rejected_raw,
                source="v5n_fixture_dpo",
                variant=variant,
            )
            if pair:
                pairs.append(pair)

    store_pairs = [
        p
        for p in pairs
        if json.loads(p["chosen"][0]["content"]).get("action") not in {"ignore", "ignore_noise"}
    ]
    ignore_pairs = [
        p
        for p in pairs
        if json.loads(p["chosen"][0]["content"]).get("action") in {"ignore", "ignore_noise"}
    ]
    out = list(pairs)
    out.extend(_dup_rows(store_pairs, prefix="dpos", copies=fixture_copies))
    out.extend(_dup_rows(ignore_pairs, prefix="dpoi", copies=max(2, fixture_copies // 2)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixture-copies", type=int, default=8)
    args = parser.parse_args()

    rows = build_v5n_fixture_dpo_rows(fixture_copies=args.fixture_copies)
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
        "profile": "hf-prod-v5n-dpo",
        "output": str(args.out),
        "total_rows": len(hf_rows),
        "format": "dpo",
        "variants": sorted({str(r.get("variant") or "") for r in rows}),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
