from __future__ import annotations

import json
from pathlib import Path

from prod_memory.build_minimal_fixture_rows import build_minimal_fixture_rows


def build_binary_fixture_rows(fixture_ids: list[str], *, fixtures_path: Path | None = None) -> list[dict]:
    rows = build_minimal_fixture_rows(fixture_ids, fixtures_path=fixtures_path)
    for row in rows:
        action = str(row["expected"].get("action") or "ignore")
        if action != "ignore":
            row["expected"] = {
                "action": "store_episodic",
                "memory": {
                    "content": "classify-store",
                    "type": "episodic",
                    "strength": 0.86,
                    "decay_rate": 0.02,
                    "emotional_weight": 0.22,
                    "confidence": 0.92,
                    "tags": [],
                },
                "facts": [],
                "indexables": [],
                "reasoning": "Durable information present.",
            }
    return rows


def write_binary_fixture_jsonl(out_path: Path, fixture_ids: list[str], *, fixtures_path: Path | None = None) -> int:
    rows = build_binary_fixture_rows(fixture_ids, fixtures_path=fixtures_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
