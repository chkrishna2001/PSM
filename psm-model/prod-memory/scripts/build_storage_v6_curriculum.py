#!/usr/bin/env python3
"""Build hf-prod-storage-v6.jsonl -- a full, fresh SFT retrain curriculum for the storage
adapter, replacing the v5n-dpo(2,3,4,5,6) patch chain.

Root-cause finding (2026-07-10, after 3 consecutive failed DPO patch rounds): the BASE SFT
curriculum (`hf-prod-v5n.jsonl`, which everything since v5n has resumed from) is itself
massively imbalanced -- `prod_extraction_v1` (1389 of 1908 rows, 73% of the whole curriculum)
has only 8 ignore-labeled rows out of 1389 (0.6%). The model's prior was baked from the start
to almost never predict "ignore". Every DPO patch since (v5n-dpo through v5n-dpo6) has been
fighting this base distribution with 12-60 example contrastive nudges and, unsurprisingly,
losing ground on one side whenever it gains on the other -- a small patch cannot durably
overcome a systematic 0.6% base rate learned over 1000+ examples.

Fix: rebalance the base curriculum itself and do a FULL fresh SFT retrain from base Qwen (no
resume-adapter chain), not another patch. Combines:
  - all 1908 original hf-prod-v5n.jsonl rows (unchanged, real teacher-labeled content)
  - the existing 31-row hand-verified ignore-seed pool (build_v5n_dpo2_ignore_rows(), real
    Claude Code/Codex turns, already proven safe/correct across 3 prior rounds)
  - the new 45-row v6 ignore pool + 29-row v6 store pool (build_storage_v6_rows.py, freshly
    mined from the same real candidate pool, genuinely new content)
  - 2 extra duplicate passes of the combined ignore pool (165 original + 31 + 45 = 241 unique
    real examples) to lift the overall ignore ratio from 8.6% to ~27%, closer to the coding-agent
    gate's own real-world ratio (6/17 = 35%) without overcorrecting past it.
"""
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
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-storage-v6.jsonl"


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

    # New hand-labeled real rows (build_row -> hf-format via row_messages)
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

    # Combined real ignore pool: base + new hand-labeled (241 unique) -- duplicate 2 extra
    # passes to lift the ratio from 8.6% to ~27% without overcorrecting.
    combined_ignore_pool = base_ignore + new_ignore_hf
    ignore_dup_1 = [{**r, "id": f"{r['id']}-dup1"} for r in combined_ignore_pool]
    ignore_dup_2 = [{**r, "id": f"{r['id']}-dup2"} for r in combined_ignore_pool]

    rows = list(base_rows) + new_ignore_hf + new_store_hf + ignore_dup_1 + ignore_dup_2

    final_ignore = sum(1 for r in rows if _is_ignore(r))
    write_jsonl(DEFAULT_OUT, rows)
    manifest = {
        "profile": "hf-prod-storage-v6",
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
