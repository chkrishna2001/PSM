#!/usr/bin/env python3
"""In-domain v5h sanity: v3 rows with training message format."""
from __future__ import annotations

import json
import random
from pathlib import Path

from prod_memory.eval_hf_grounding import open_hf_session
from prod_memory.grounding import grounding_overlap_score, stored_text_from_decision, would_model_store
from prod_memory.hf_prompts import storage_inference_messages
from prod_memory.row_validation import label_text_from_expected, remember_target_from_input
from psm_model.remember_cli import apply_product_boundary

REPO = Path(__file__).resolve().parents[2]
V3 = REPO / "psm-model/prod-memory/data/prod-extraction-v3.jsonl"
ADAPTER = REPO / "psm-model/prod-memory/checkpoints/hf-prod-v5h-qwen0.5b/adapter"


def storage_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            inp = r.get("input") or {}
            if isinstance(inp.get("conversation"), list):
                action = str((r.get("expected") or {}).get("action") or "")
                if action != "ignore":
                    rows.append(r)
    return rows


def main() -> None:
    random.seed(42)
    pool = storage_rows(V3)
    short = [r for r in pool if 400 <= len(remember_target_from_input(r["input"])) <= 1800]
    sample = random.sample(short or pool, min(5, len(short or pool)))
    session = open_hf_session(ADAPTER, device="cpu")
    results = []
    for i, row in enumerate(sample):
        print(f"gen {i + 1}/{len(sample)} {row['id'][:50]}", flush=True)
        rid = row["id"]
        remember = remember_target_from_input(row["input"])
        expected = row["expected"]
        raw = session.generate(remember, output_format="json", max_new_tokens=256)
        report = apply_product_boundary(raw, output_format="json")
        decision = report.get("parsed") or {}
        stored = stored_text_from_decision(decision)
        label = label_text_from_expected(expected)
        overlap = grounding_overlap_score(remember, stored)
        label_overlap = grounding_overlap_score(remember, label)
        exp_action = expected.get("action")
        got_action = decision.get("action")
        store_ok = would_model_store(decision) == (exp_action != "ignore")
        results.append(
            {
                "id": rid,
                "parse_ok": report.get("repair_status") == "parsed",
                "exp_action": exp_action,
                "got_action": got_action,
                "store_match": store_ok,
                "has_facts": bool(decision.get("facts")),
                "exp_has_facts": bool(expected.get("facts")),
                "content_overlap": overlap,
                "label_overlap": label_overlap,
                "remember_chars": len(remember),
                "memory_preview": (decision.get("memory") or {}).get("content", "")[:120],
                "exp_memory_preview": (expected.get("memory") or {}).get("content", "")[:120],
            }
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    n = len(results)
    print(
        "SUMMARY",
        f"parse={sum(r['parse_ok'] for r in results)}/{n}",
        f"store={sum(r['store_match'] for r in results)}/{n}",
        f"facts={sum(r['has_facts'] for r in results)}/{n}",
        f"grounded={sum(r['content_overlap']['grounded'] for r in results)}/{n}",
    )


if __name__ == "__main__":
    main()
