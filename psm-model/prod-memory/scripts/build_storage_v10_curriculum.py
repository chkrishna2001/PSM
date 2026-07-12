#!/usr/bin/env python3
"""Build hf-prod-storage-v10.jsonl -- fixes 3 real methodology gaps found via research
(arXiv:2606.08051, "How Small Can You Go?"), not another data-volume test:

1. Reasoning-first JSON key order. Every prior round's assistant target used
   `sort_keys=True` (alphabetical), forcing "action" first and "reasoning" last -- the
   model had to commit to the classification before articulating any justification.
   The paper's Free-Thinking (reason-then-classify) vs JSON-Only fine-tuning comparison
   showed the LARGEST gains from reasoning-first framing go to the SMALLEST models
   (+1.82 F1 for their 270M model vs +0.18 for their 8B model) -- exactly the shape of
   gap we have. Fixed in `hf_prompts.compact_storage_json` (reasoning now first); this
   script re-renders the ENTIRE base curriculum (not just the new hand-labeled rows) so
   the whole training set uses one consistent format, not old+new mismatched orderings.
2. Learning rate. Every full retrain this session used 1e-5; the paper's recipe for
   comparable small-model structured-output fine-tuning uses 1e-4 (10x higher).
3. Undertraining. 500-600 steps on ~2000-2500 rows at effective batch 8 is only
   ~2 epochs; the paper trains 6 full epochs. This script keeps the data volume from
   v9 (149 hand-verified real rows, same as before) and only changes the recipe --
   isolating whether methodology, not more data, was the actual lever.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_storage_v6_rows import build_storage_v6_rows  # noqa: E402
from prod_memory.build_storage_v7_rows import build_storage_v7_new_rows  # noqa: E402
from prod_memory.build_v5n_dpo2_calibration_rows import (  # noqa: E402
    build_v5n_dpo2_ignore_rows,
    build_v5n_dpo4_store_rows,
)
from prod_memory.hf_prompts import compact_storage_json, row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402
from psm_model.prompts import JSON_SYSTEM_INSTRUCTION  # noqa: E402

BASE_CURRICULUM = PACKAGE_ROOT / "data" / "hf-prod-v5n.jsonl"
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-storage-v10.jsonl"


def _load_base_rows() -> list[dict]:
    rows = []
    with BASE_CURRICULUM.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _reorder_assistant_message(hf_row: dict) -> dict:
    """Re-render the assistant message's JSON with the new reasoning-first key order,
    preserving the exact decision content -- only the serialization order changes."""
    messages = hf_row["messages"]
    new_messages = []
    for msg in messages:
        if msg["role"] == "system":
            new_messages.append({"role": "system", "content": JSON_SYSTEM_INSTRUCTION})
            continue
        if msg["role"] != "assistant":
            new_messages.append(msg)
            continue
        try:
            decision = json.loads(msg["content"])
        except Exception:
            new_messages.append(msg)
            continue
        new_messages.append({"role": "assistant", "content": compact_storage_json(decision)})
    return {**hf_row, "messages": new_messages}


def _is_ignore(hf_row: dict) -> bool:
    for msg in hf_row.get("messages", []):
        if msg.get("role") == "assistant":
            try:
                return json.loads(msg["content"]).get("action") == "ignore"
            except Exception:
                return False
    return False


def main() -> int:
    base_rows_raw = _load_base_rows()
    base_rows = [_reorder_assistant_message(r) for r in base_rows_raw]
    base_ignore = [r for r in base_rows if _is_ignore(r)]
    print(f"base: {len(base_rows)} rows, {len(base_ignore)} ignore ({len(base_ignore)/len(base_rows):.1%}) -- re-rendered reasoning-first")

    v5n_dpo2_ignore = build_v5n_dpo2_ignore_rows()
    v5n_dpo4_store = build_v5n_dpo4_store_rows()
    v6_rows = build_storage_v6_rows()
    v7_rows = build_storage_v7_new_rows()

    def to_hf(row: dict, source_tag: str) -> dict:
        return {
            "id": row["id"],
            "task": "storage",
            "messages": row_messages(row, output_format="json"),  # already reasoning-first
            "source": source_tag,
        }

    new_ignore_hf = [to_hf(r, "v5n_dpo2_calibration") for r in v5n_dpo2_ignore]
    new_ignore_hf += [to_hf(r, "storage_v6") for r in v6_rows if r["expected"]["action"] == "ignore"]
    new_ignore_hf += [to_hf(r, "storage_v7") for r in v7_rows if r["expected"]["action"] == "ignore"]
    new_store_hf = [to_hf(r, "v5n_dpo4_calibration") for r in v5n_dpo4_store]
    new_store_hf += [to_hf(r, "storage_v6") for r in v6_rows if r["expected"]["action"] != "ignore"]
    new_store_hf += [to_hf(r, "storage_v7") for r in v7_rows if r["expected"]["action"] != "ignore"]

    rows = list(base_rows) + new_ignore_hf + new_store_hf

    final_ignore = sum(1 for r in rows if _is_ignore(r))
    write_jsonl(DEFAULT_OUT, rows)
    manifest = {
        "profile": "hf-prod-storage-v10",
        "output": str(DEFAULT_OUT),
        "total_rows": len(rows),
        "base_rows": len(base_rows),
        "new_ignore_rows": len(new_ignore_hf),
        "new_store_rows": len(new_store_hf),
        "final_ignore_count": final_ignore,
        "final_ignore_ratio": round(final_ignore / len(rows), 4),
        "format_change": "reasoning-first key order (was action-first alphabetical)",
    }
    DEFAULT_OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
