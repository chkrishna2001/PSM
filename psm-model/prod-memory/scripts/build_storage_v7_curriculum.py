#!/usr/bin/env python3
"""Build hf-prod-storage-v7.jsonl -- moderate rebalance retry after v6 overcorrected.

v6 (28.9% ignore ratio, 2 extra duplication passes over the combined 241-row ignore pool,
full fresh retrain, 600 steps) massively overcorrected: coding-agent gate action_match dropped
to 0.47 (9/11 store cases wrongly flipped to ignore, even though all 6 ignore cases were
fixed, up from 4/6). A full convergent SFT retrain absorbs a distribution shift far more
completely than a small DPO patch ever did -- the same 3x ignore-rate lift that only nudged
the model under DPO patching fully rewired it under a real retrain.

v7 targets a much more moderate ~12% ignore ratio -- roughly 1.4x the original 8.6%, not
3.4x -- by adding the new hand-labeled rows (v5n-dpo2's 31-row ignore pool + the new v6 45-row
ignore/29-row store pool) exactly once, with NO extra duplication passes. This is a genuinely
different, more conservative point on the same lever, not a repeat of v6."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_storage_v6_rows import build_storage_v6_rows  # noqa: E402
from prod_memory.build_v5n_dpo2_calibration_rows import (  # noqa: E402
    build_v5n_dpo2_ignore_rows,
    build_v5n_dpo4_store_rows,
)
from prod_memory.hf_prompts import row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

BASE_CURRICULUM = PACKAGE_ROOT / "data" / "hf-prod-v5n.jsonl"
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-storage-v7.jsonl"


def _load_base_rows() -> list[dict]:
    rows = []
    with BASE_CURRICULUM.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_ignore(hf_row: dict) -> bool:
    for msg in hf_row.get("messages", []):
        if msg.get("role") == "assistant":
            try:
                return json.loads(msg["content"]).get("action") == "ignore"
            except Exception:
                return False
    return False


def main() -> int:
    base_rows = _load_base_rows()
    base_ignore = [r for r in base_rows if _is_ignore(r)]
    print(f"base: {len(base_rows)} rows, {len(base_ignore)} ignore ({len(base_ignore)/len(base_rows):.1%})")

    v5n_dpo2_ignore = build_v5n_dpo2_ignore_rows()
    v5n_dpo4_store = build_v5n_dpo4_store_rows()
    v6_rows = build_storage_v6_rows()

    def to_hf(row: dict, source_tag: str) -> dict:
        return {
            "id": row["id"],
            "task": "storage",
            "messages": row_messages(row, output_format="json"),
            "source": source_tag,
        }

    new_ignore_hf = [to_hf(r, "v5n_dpo2_calibration") for r in v5n_dpo2_ignore]
    new_ignore_hf += [to_hf(r, "storage_v6") for r in v6_rows if r["expected"]["action"] == "ignore"]
    new_store_hf = [to_hf(r, "v5n_dpo4_calibration") for r in v5n_dpo4_store]
    new_store_hf += [to_hf(r, "storage_v6") for r in v6_rows if r["expected"]["action"] != "ignore"]

    # No extra duplication passes this time -- just add the new hand-labeled rows once.
    rows = list(base_rows) + new_ignore_hf + new_store_hf

    final_ignore = sum(1 for r in rows if _is_ignore(r))
    write_jsonl(DEFAULT_OUT, rows)
    manifest = {
        "profile": "hf-prod-storage-v7",
        "output": str(DEFAULT_OUT),
        "total_rows": len(rows),
        "base_rows": len(base_rows),
        "new_ignore_rows": len(new_ignore_hf),
        "new_store_rows": len(new_store_hf),
        "final_ignore_count": final_ignore,
        "final_ignore_ratio": round(final_ignore / len(rows), 4),
    }
    DEFAULT_OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
