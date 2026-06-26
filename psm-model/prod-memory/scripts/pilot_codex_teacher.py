#!/usr/bin/env python3
"""Pilot teacher model on codex session samples (compare to cached gpt-4o labels)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from prod_memory.grounding import is_grounded_in_source, stored_text_from_decision
from prod_memory.openrouter_teacher import TeacherConfig, label_assistant_with_teacher
from prod_memory.row_validation import validate_prod_row

DEFAULT_CACHE = PACKAGE_ROOT / "data" / "prod-teacher-cache-v4-4o.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "gemma-codex-pilot.json"


def _load_codex_samples(cache: Path, *, limit: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        row = entry.get("row") or {}
        conv = (row.get("input") or {}).get("conversation") or []
        if not conv:
            continue
        text = str(conv[0].get("content") or "").strip()
        if len(text) < 80:
            continue
        expected = row.get("expected") or {}
        rows.append({
            "id": str(row.get("id") or entry.get("id") or ""),
            "text": text,
            "gpt4o_action": str(expected.get("action") or "ignore"),
            "gpt4o_facts": len(expected.get("facts") or []),
        })
    random.seed(seed)
    return random.sample(rows, min(limit, len(rows)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    samples = _load_codex_samples(args.cache, limit=args.limit, seed=args.seed)
    cfg = TeacherConfig.from_env(model=args.model)
    cfg = TeacherConfig(
        model=args.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        request_delay_ms=1200,
    )

    results: list[dict] = []
    stats = {"valid": 0, "action_agree": 0, "grounded_store": 0, "store": 0, "ignore": 0}

    for sample in samples:
        expected, meta = label_assistant_with_teacher(
            sample["text"],
            config=cfg,
            use_heuristic_fallback=False,
        )
        row = {
            "id": sample["id"],
            "input": {
                "operation": "remember_llm_response",
                "conversation": [{"role": "assistant", "content": sample["text"]}],
            },
            "expected": expected,
        }
        valid = True
        valid_err = ""
        try:
            validate_prod_row(row)
            stats["valid"] += 1
        except ValueError as exc:
            valid = False
            valid_err = str(exc)

        action = str(expected.get("action") or "ignore")
        stored = stored_text_from_decision(expected)
        grounded = bool(stored and is_grounded_in_source(sample["text"], stored))
        gpt_bin = "ignore" if sample["gpt4o_action"] == "ignore" else "store"
        got_bin = "ignore" if action == "ignore" else "store"
        agree = gpt_bin == got_bin
        if agree:
            stats["action_agree"] += 1
        if action != "ignore":
            stats["store"] += 1
            if grounded:
                stats["grounded_store"] += 1
        else:
            stats["ignore"] += 1

        results.append({
            "id": sample["id"],
            "input_chars": len(sample["text"]),
            "gpt4o_action": sample["gpt4o_action"],
            "got_action": action,
            "action_agree": agree,
            "grounded": grounded,
            "valid": valid,
            "valid_error": valid_err,
            "facts": len(expected.get("facts") or []),
            "memory_preview": (stored or "")[:140],
            "reasoning": str(expected.get("reasoning") or "")[:120],
            "model": meta.get("model"),
            "parse_error": meta.get("parse_error"),
        })

    n = len(results)
    report = {
        "model": args.model,
        "samples": n,
        "valid_rate": round(stats["valid"] / n, 3) if n else 0,
        "action_agree_gpt4o": f"{stats['action_agree']}/{n}",
        "grounded_store": stats["grounded_store"],
        "store_count": stats["store"],
        "ignore_count": stats["ignore"],
        "approve_bulk_teacher": stats["valid"] >= max(3, n - 1) and stats["action_agree"] >= n // 2,
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2))
    print(f"detail: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
