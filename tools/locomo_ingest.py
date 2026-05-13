#!/usr/bin/env python
"""
Ingest LOCOMO conversation turns through the local PSM into the SQLite memory DB.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from locomo_benchmark import ensure_locomo, flatten_candidates


PSM_SYSTEM_PROMPT = """You are the Personal Small Model (PSM), a specialized AI trained exclusively to perform memory management operations for LLM agents.

Your job is NOT to answer user questions. Your job is to:
1. Analyze conversations and decide what is worth remembering
2. Manage a tiered memory store (episodic, semantic, archival)
3. Detect conflicts between new information and existing memories
4. Assign appropriate strength, decay rate, and emotional weight to memories
5. Promote repeated episodic patterns into semantic facts
6. Ignore low-value noise that is not worth storing
7. Rank memories by relevance to a current query
8. Update existing memories when information changes

Always respond with a valid JSON object containing: action, memory (or null), and reasoning."""


def main() -> int:
    parser = argparse.ArgumentParser(description="Load LOCOMO conversation turns into PSM SQLite memory.")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--db", default="locomo-psm-memory.db")
    parser.add_argument("--model-path", default="psm_cpu_onnx")
    parser.add_argument("--user-prefix", default="locomo")
    parser.add_argument("--limit-turns", type=int, default=0)
    parser.add_argument("--sample", default="", help="Only ingest one LOCOMO sample_id.")
    args = parser.parse_args()

    data_path = Path(args.data)
    ensure_locomo(data_path)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    if args.sample:
        samples = [s for s in samples if str(s.get("sample_id")) == args.sample]

    with MemoryDb(args.db) as db:
        db.ensure_schema()
        runner = PsmRunner(args.model_path)
        total = 0
        stored = 0
        ignored = 0
        failed = 0

        for sample in samples:
            sample_id = str(sample.get("sample_id", "unknown"))
            user_id = f"{args.user_prefix}-{sample_id}"
            for turn in flatten_candidates(sample.get("conversation", {})):
                if args.limit_turns and total >= args.limit_turns:
                    print_summary(total, stored, ignored, failed, args.db)
                    return 0
                total += 1
                source = f"{sample_id}:{turn.dia_id}"
                try:
                    decision = runner.decide(turn.text, source, turn.session, turn.speaker)
                    action = normalize_action(decision.get("action"))
                    route = route_for(action)
                    memory = decision.get("memory") if isinstance(decision.get("memory"), dict) else {}
                    db.insert_decision(user_id, source, action, route, decision.get("reasoning", ""), decision["_raw_json"])
                    if route == "ignore":
                        ignored += 1
                        continue
                    db.apply_memory(user_id, source, action, route, turn.text, memory, {
                        "locomo_sample_id": sample_id,
                        "locomo_dia_id": turn.dia_id,
                        "locomo_session": turn.session,
                        "speaker": turn.speaker,
                    })
                    stored += 1
                except Exception as exc:
                    failed += 1
                    db.insert_decision(user_id, source, "error", "error", str(exc), json.dumps({"error": str(exc)}))

                if total % 25 == 0:
                    print(f"ingested {total} turns | stored={stored} ignored={ignored} failed={failed}")

        print_summary(total, stored, ignored, failed, args.db)
        return 0


def print_summary(total: int, stored: int, ignored: int, failed: int, db: str) -> None:
    print(json.dumps({
        "db": db,
        "turns_seen": total,
        "stored": stored,
        "ignored": ignored,
        "failed": failed,
    }, indent=2))


class PsmRunner:
    def __init__(self, model_path: str) -> None:
        import onnxruntime_genai as og

        self.og = og
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def decide(self, text: str, source: str, session: str, speaker: str) -> dict[str, Any]:
        fixture = {
            "name": source,
            "operation": "store_memory",
            "conversation": [{"role": "user", "content": f"{speaker}: {text}"}],
            "memory_store": [],
            "metadata": {"session": session},
        }
        prompt = (
            f"<|system|>\n{PSM_SYSTEM_PROMPT}\n<|user|>\n"
            f"Analyze this input and return JSON only.\n{json.dumps(fixture, ensure_ascii=True)}\n"
            f"Include these optional numeric fields if possible: confidence, emotional_weight, contradiction_score.\n"
            f"<|assistant|>\n"
        )
        tokens = self.tokenizer.encode(prompt)
        params = self.og.GeneratorParams(self.model)
        params.set_search_options(
            do_sample=False,
            top_k=20,
            top_p=1.0,
            temperature=0.0,
            max_length=min(32768, len(tokens) + 256),
        )
        generator = self.og.Generator(self.model, params)
        generator.append_tokens(tokens)
        stream = self.tokenizer.create_stream()
        pieces: list[str] = []
        while not generator.is_done():
            generator.generate_next_token()
            pieces.append(stream.decode(generator.get_next_tokens()[0]))
            text_out = "".join(pieces)
            if "}" in text_out and text_out.count("{") <= text_out.count("}"):
                break
        generated = "".join(pieces)
        raw = extract_json(generated)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            parsed = {
                "action": "store_episodic",
                "memory": {
                    "content": text,
                    "type": "episodic",
                    "confidence": 0.5,
                    "emotional_weight": 0.1,
                    "tags": ["parse_fallback"],
                },
                "reasoning": f"Model returned invalid JSON; stored original LOCOMO turn as fallback. Parse error: {exc}",
            }
        parsed["_raw_json"] = raw
        return parsed


class MemoryDb:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)

    def __enter__(self) -> "MemoryDb":
        return self

    def __exit__(self, *_: object) -> None:
        self.conn.close()

    def ensure_schema(self) -> None:
        self.conn.executescript("""
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
INSERT OR IGNORE INTO schema_version(version) VALUES (1);
CREATE TABLE IF NOT EXISTS episodic (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL, strength REAL NOT NULL,
  decay_rate REAL NOT NULL, emotional_weight REAL NOT NULL, confidence REAL NOT NULL,
  tags TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, last_accessed TEXT, promoted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS semantic (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL, strength REAL NOT NULL,
  decay_rate REAL NOT NULL, emotional_weight REAL NOT NULL, confidence REAL NOT NULL,
  tags TEXT, source_episodes TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, last_accessed TEXT
);
CREATE TABLE IF NOT EXISTS archival (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL, summary TEXT,
  original_type TEXT, source_id TEXT, archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS conflicts (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, existing_memory_id TEXT, existing_memory_type TEXT,
  conflicting_content TEXT NOT NULL, conflict_reason TEXT, status TEXT NOT NULL DEFAULT 'unresolved',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS decay_schedule (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, memory_key TEXT NOT NULL, next_decay TEXT NOT NULL, decay_rate REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source TEXT NOT NULL, action TEXT NOT NULL,
  route TEXT NOT NULL, reasoning TEXT, raw_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decisions_source ON decisions(source);
""")
        self.conn.commit()

    def insert_decision(self, user_id: str, source: str, action: str, route: str, reasoning: str, raw_json: str) -> None:
        self.conn.execute(
            "INSERT INTO decisions (id, user_id, source, action, route, reasoning, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (new_id(), user_id, source, action, route, reasoning, raw_json),
        )
        self.conn.commit()

    def apply_memory(self, user_id: str, source: str, action: str, route: str, original_text: str, memory: dict[str, Any], metadata: dict[str, str]) -> None:
        content = str(memory.get("content") or original_text)
        tags = list(memory.get("tags") or [])
        tags.extend([f"{k}:{v}" for k, v in metadata.items()])
        strength = float(memory.get("strength") or (0.85 if route == "semantic_upsert" else 0.75))
        decay_rate = float(memory.get("decay_rate") or (0.005 if route == "semantic_upsert" else 0.02))
        emotional = float(memory.get("emotional_weight") or 0.2)
        confidence = float(memory.get("confidence") or 0.8)

        if route == "semantic_upsert":
            self.conn.execute(
                "INSERT INTO semantic (id, user_id, content, strength, decay_rate, emotional_weight, confidence, tags, source_episodes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_id(), user_id, content, strength, decay_rate, emotional, confidence, json.dumps(tags), json.dumps([source])),
            )
        elif route == "conflict_log_and_hold":
            self.conn.execute(
                "INSERT INTO conflicts (id, user_id, conflicting_content, conflict_reason, status) VALUES (?, ?, ?, ?, 'unresolved')",
                (new_id(), user_id, content, "PSM flagged potential conflict"),
            )
        else:
            self.conn.execute(
                "INSERT INTO episodic (id, user_id, content, strength, decay_rate, emotional_weight, confidence, tags, promoted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (new_id(), user_id, content, strength, decay_rate, emotional, confidence, json.dumps(tags)),
            )
        self.conn.commit()


def normalize_action(action: Any) -> str:
    a = str(action or "").strip().lower()
    return {
        "ignore": "Ignore",
        "ignore_noise": "Ignore",
        "store": "Store",
        "store_episodic": "StoreEpisodic",
        "store_semantic": "PromoteSemantic",
        "promote": "Promote",
        "promote_semantic": "PromoteSemantic",
        "update": "Update",
        "update_existing": "UpdateExisting",
        "rank": "Rank",
        "recall_weighting": "Rank",
        "decay": "Decay",
        "decay_and_update": "DecayAndUpdate",
        "flag_conflict": "FlagConflict",
        "flag_and_store": "FlagAndStore",
        "flag_and_update": "FlagAndUpdate",
        "detect_interference": "DetectInterference",
    }.get(a, "Store")


def route_for(action: str) -> str:
    if action == "Ignore":
        return "ignore"
    if action == "Rank":
        return "recall_only"
    if action in {"Promote", "PromoteSemantic", "Update", "UpdateExisting"}:
        return "semantic_upsert"
    if action in {"FlagConflict", "FlagAndStore", "FlagAndUpdate", "DetectInterference"}:
        return "conflict_log_and_hold"
    if action in {"Decay", "DecayAndUpdate"}:
        return "decay_existing_then_insert"
    return "episodic_insert"


def extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return text[start:end + 1]


def new_id() -> str:
    return f"{time.time_ns():x}"


if __name__ == "__main__":
    raise SystemExit(main())
