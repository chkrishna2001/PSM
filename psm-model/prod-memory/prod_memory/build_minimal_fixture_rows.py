from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prod_memory.curriculum_sources import load_fixture_cases, remember_input


def grounded_store_content(llm_response: str, key_tokens: list[str]) -> str:
    """Build a one-line store label with tokens present in llmResponse (for grounding guards)."""
    text = llm_response.strip()
    snippets: list[str] = []
    for key in key_tokens[:4]:
        idx = text.lower().find(key.lower())
        if idx >= 0:
            start = max(0, idx - 24)
            end = min(len(text), idx + len(key) + 48)
            snippets.append(text[start:end].strip())
    if snippets:
        return " ".join(snippets)[:220]
    return text[:180].replace("\n", " ").strip()


def build_minimal_fixture_rows(
    fixture_ids: list[str] | None = None,
    *,
    fixtures_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    want = set(fixture_ids) if fixture_ids else None
    for case in load_fixture_cases(fixtures_path):
        case_id = str(case.get("id") or "")
        if want is not None and case_id not in want:
            continue
        llm_response = str(case.get("llmResponse") or "").strip()
        if not llm_response:
            continue
        expect_action = str(case.get("expectAction") or "store")
        key_tokens = [str(k) for k in (case.get("keyTokens") or [])]
        if expect_action == "ignore":
            expected = {
                "action": "ignore",
                "memory": None,
                "facts": [],
                "indexables": [],
                "reasoning": "No durable memory.",
            }
        else:
            content = grounded_store_content(llm_response, key_tokens)
            expected = {
                "action": "store_episodic",
                "memory": {
                    "content": content,
                    "type": "episodic",
                    "strength": 0.86,
                    "decay_rate": 0.02,
                    "emotional_weight": 0.22,
                    "confidence": 0.92,
                    "tags": [f"prod_eval_id:{case_id}"],
                },
                "facts": [],
                "indexables": [],
                "reasoning": content,
            }
        rows.append({
            "id": f"minimal-fixture-{case_id}",
            "input": remember_input(llm_response, source_id=case_id, source_kind="prod_minimal"),
            "expected": expected,
            "source": "exp_a_minimal",
        })
    return rows


def write_minimal_fixture_jsonl(
    out_path: Path,
    fixture_ids: list[str],
    *,
    fixtures_path: Path | None = None,
) -> int:
    rows = build_minimal_fixture_rows(fixture_ids, fixtures_path=fixtures_path)
    if not rows:
        raise SystemExit(f"no minimal rows for fixture_ids={fixture_ids}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
