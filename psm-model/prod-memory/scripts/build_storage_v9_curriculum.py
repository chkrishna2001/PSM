#!/usr/bin/env python3
"""Build hf-prod-storage-v9.jsonl -- combines ALL hand-verified real data accumulated across
this session (v5n-dpo2/dpo4 seeds, storage-v6 rows, storage-v7 rows -- 149 new rows total, up
from v7's 90) at the same moderate ~12% ignore ratio that landed flat (0.647) rather than
overcorrecting (v6's 28.9% collapsed to 0.47). Four calibration points so far (8.6% baseline,
28.9% collapse, 11.9% flat) suggest the aggregate ignore ratio itself is not the lever that
moves the needle further -- so this round holds the ratio roughly constant and tests whether
volume/diversity of real hand-verified examples (149 vs 90) does instead. Full fresh SFT
retrain from base Qwen 0.5B (project's hard size constraint), no resume-adapter chain."""
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
from prod_memory.hf_prompts import row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

BASE_CURRICULUM = PACKAGE_ROOT / "data" / "hf-prod-v5n.jsonl"
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-storage-v9.jsonl"


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
    v7_rows = build_storage_v7_new_rows()

    def to_hf(row: dict, source_tag: str) -> dict:
        return {
            "id": row["id"],
            "task": "storage",
            "messages": row_messages(row, output_format="json"),
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
        "profile": "hf-prod-storage-v9",
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
