#!/usr/bin/env python
"""
Score LOCOMO evidence retrieval from memories already stored in SQLite.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from locomo_benchmark import ensure_locomo, hit_at, rank_mem0_style, summarize, render_markdown, Candidate


@dataclass(frozen=True)
class StoredMemory:
    sample_id: str
    dia_id: str
    content: str
    table: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Test LOCOMO retrieval from PSM SQLite memory.")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--db", default="locomo-psm-memory.db")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--require-all-evidence", action="store_true")
    parser.add_argument("--out", default="results/locomo/locomo-db-results.md")
    args = parser.parse_args()

    data_path = Path(args.data)
    ensure_locomo(data_path)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    memories = load_memories(args.db)
    by_sample: dict[str, list[StoredMemory]] = {}
    for memory in memories:
        by_sample.setdefault(memory.sample_id, []).append(memory)

    records: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", "unknown"))
        sample_memories = by_sample.get(sample_id, [])
        if not sample_memories:
            continue
        candidates = tuple(
            Candidate(m.dia_id, "stored", m.table, m.content, i)
            for i, m in enumerate(sample_memories)
        )
        stored_ids = {m.dia_id for m in sample_memories}

        for qa in sample.get("qa", []):
            evidence = tuple(str(x) for x in qa.get("evidence", []) if str(x).strip())
            if not evidence:
                continue
            evidence_set = set(evidence)
            if args.require_all_evidence:
                if not evidence_set.issubset(stored_ids):
                    continue
            elif not evidence_set.intersection(stored_ids):
                continue

            ranked = rank_mem0_style(str(qa.get("question", "")), candidates)
            selected = [c.dia_id for c, _ in ranked[: args.top_k]]
            records.append({
                "sample_id": sample_id,
                "category": str(qa.get("category", "unknown")),
                "question": str(qa.get("question", "")),
                "answer": str(qa.get("answer", "")),
                "evidence": list(evidence),
                "mem0_ids": selected,
                "psm_ids": selected,
                "psm_error": None,
                "mem0_hit_at_1": hit_at(evidence, selected, 1),
                "mem0_hit_at_k": hit_at(evidence, selected, args.top_k),
                "psm_hit_at_1": hit_at(evidence, selected, 1),
                "psm_hit_at_k": hit_at(evidence, selected, args.top_k),
            })

    if not records:
        raise SystemExit("No evaluable LOCOMO questions found for the memories stored in the DB.")

    summary = summarize(records, top_k=args.top_k, elapsed_seconds=0.0)
    markdown = render_markdown(summary).replace(
        "LOCOMO Evidence-Retrieval Results",
        "LOCOMO SQLite Memory Retrieval Results",
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {out_path}")
    return 0


def load_memories(db_path: str) -> list[StoredMemory]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    memories: list[StoredMemory] = []
    queries = {
        "episodic": "select content, tags, null as source_episodes from episodic",
        "semantic": "select content, tags, source_episodes from semantic",
    }
    for table, query in queries.items():
        for row in con.execute(query):
            tags = parse_json_list(row["tags"])
            source_episodes = parse_json_list(row["source_episodes"]) if "source_episodes" in row.keys() else []
            sample_id = tag_value(tags, "locomo_sample_id")
            dia_id = tag_value(tags, "locomo_dia_id")
            if (not sample_id or not dia_id) and source_episodes:
                parsed = parse_source(str(source_episodes[0]))
                sample_id = sample_id or parsed[0]
                dia_id = dia_id or parsed[1]
            if sample_id and dia_id:
                memories.append(StoredMemory(sample_id, dia_id, str(row["content"]), table))
    con.close()
    return memories


def parse_json_list(value: Any) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def tag_value(tags: list[Any], key: str) -> str:
    prefix = f"{key}:"
    for tag in tags:
        text = str(tag)
        if text.startswith(prefix):
            return text[len(prefix):]
    return ""


def parse_source(source: str) -> tuple[str, str]:
    match = re.fullmatch(r"([^:]+):(D\d+:\d+)", source)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


if __name__ == "__main__":
    raise SystemExit(main())
