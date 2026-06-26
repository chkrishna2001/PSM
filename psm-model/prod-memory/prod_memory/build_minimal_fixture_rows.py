from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prod_memory.curriculum_sources import (
    MEMORY_SUMMARIES,
    _ignore_expected,
    _store_expected,
    build_noise_rows,
    build_plan_handoff_rows,
    build_technical_rows,
    load_fixture_cases,
    remember_input,
)
from prod_memory.grounding import is_grounded_in_source, key_tokens_grounded
from prod_memory.label_from_assistant import _make_fact

V5E_FORCED_STORE_IDS = frozenset({"plan-01-handoff", "workflow-runpod"})
V5E_BOOST_FIXTURE_IDS = (
    "plan-01-handoff",
    "workflow-runpod",
    "noise-filler",
    "noise-meta",
)


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


def forced_grounded_store_content(llm_response: str, key_tokens: list[str]) -> str:
    """Snippet label with ≥2 keyTokens verbatim (for guard + eval grounding)."""
    text = llm_response.strip()
    hits: list[str] = []
    for key in key_tokens:
        idx = text.lower().find(key.lower())
        if idx < 0:
            continue
        start = max(0, idx - 20)
        end = min(len(text), idx + len(key) + 40)
        hits.append(text[start:end].strip())
    if not hits:
        return grounded_store_content(llm_response, key_tokens)
    content = " ".join(hits)[:220]
    if key_tokens and not key_tokens_grounded(key_tokens, content):
        # ponytail: widen windows until min(2, n) key hits land in label
        for key in key_tokens:
            if key_tokens_grounded(key_tokens, content):
                break
            idx = text.lower().find(key.lower())
            if idx >= 0:
                content = (content + " " + text[max(0, idx - 8) : min(len(text), idx + len(key) + 56)]).strip()[:220]
    return content


def _store_content_for_case(llm_response: str, case_id: str, key_tokens: list[str]) -> str:
    """Prefer MEMORY_SUMMARIES when grounded in source; else snippet extraction."""
    summary = MEMORY_SUMMARIES.get(case_id, "").strip()
    if summary and is_grounded_in_source(llm_response, summary):
        return summary
    return grounded_store_content(llm_response, key_tokens)


def build_summary_fixture_rows(
    fixture_ids: list[str] | None = None,
    *,
    fixtures_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Fixture rows with human summaries (not snippet extraction) for minimal store labels."""
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
            content = _store_content_for_case(llm_response, case_id, key_tokens)
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
            "id": f"summary-fixture-{case_id}",
            "input": remember_input(llm_response, source_id=case_id, source_kind="prod_summary"),
            "expected": expected,
            "source": "exp_a_summary",
        })
    return rows


def build_hybrid_fixture_rows(
    fixture_ids: list[str] | None = None,
    *,
    fixtures_path: Path | None = None,
) -> list[dict[str, Any]]:
    """v5e: snippet labels; forced key-token grounding on weak store fixtures."""
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
            if case_id in V5E_FORCED_STORE_IDS:
                content = forced_grounded_store_content(llm_response, key_tokens)
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
            "id": f"hybrid-fixture-{case_id}",
            "input": remember_input(llm_response, source_id=case_id, source_kind="prod_hybrid"),
            "expected": expected,
            "source": "exp_a_hybrid",
        })
    return rows


def _facts_from_key_tokens(llm_response: str, key_tokens: list[str]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for key in key_tokens[:4]:
        idx = llm_response.lower().find(key.lower())
        if idx < 0:
            continue
        evidence = llm_response[max(0, idx - 12) : min(len(llm_response), idx + len(key) + 48)].strip()
        facts.append(
            _make_fact(
                subject=key,
                predicate="mentions",
                value=key,
                evidence_text=evidence,
                confidence=0.9,
            )
        )
    return facts[:3]


def build_json_fixture_rows(
    fixture_ids: list[str] | None = None,
    *,
    fixtures_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Fixture anchors with grounded JSON labels + fact spans (anti template-collapse)."""
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
        suite = str(case.get("suite") or "prod")
        if expect_action == "ignore":
            expected = _ignore_expected("No durable memory to store from this assistant text.")
        else:
            if case_id in V5E_FORCED_STORE_IDS:
                content = forced_grounded_store_content(llm_response, key_tokens)
            else:
                content = grounded_store_content(llm_response, key_tokens)
            facts = _facts_from_key_tokens(llm_response, key_tokens)
            expected = _store_expected(
                llm_response,
                content,
                tags=[f"prod_eval_suite:{suite}", f"prod_eval_id:{case_id}"],
                reasoning=content[:160],
                facts=facts,
                memory_type="semantic" if suite == "technical" else "episodic",
            )
        rows.append({
            "id": f"json-fixture-{case_id}",
            "input": remember_input(llm_response, source_id=case_id, source_kind=f"prod_{suite}"),
            "expected": expected,
            "source": "exp_a_json_fixture",
        })
    return rows


def _dup_rows(rows: list[dict[str, Any]], *, prefix: str, copies: int) -> list[dict[str, Any]]:
    if copies <= 0:
        return []
    out: list[dict[str, Any]] = []
    for copy_idx in range(copies):
        for row in rows:
            cloned = dict(row)
            cloned["id"] = f"{prefix}-{copy_idx:03d}-{row['id']}"
            cloned["source"] = f"{prefix}:{row.get('source', 'row')}"
            out.append(cloned)
    return out


def build_v5j_anchor_rows() -> list[dict[str, Any]]:
    """Fixture-first anchors: per-input grounded labels + heavy ignore (anti template-collapse)."""
    rows: list[dict[str, Any]] = []
    seed = build_hybrid_fixture_rows()
    rows.extend(seed)
    rows.extend(_dup_rows(seed, prefix="fxj", copies=19))
    noise_seed = [row for row in seed if str((row.get("expected") or {}).get("action") or "").lower() in {"ignore", "ignore_noise"}]
    store_seed = [row for row in seed if row not in noise_seed]
    if noise_seed:
        rows.extend(_dup_rows(noise_seed, prefix="fxjn", copies=79))
    if store_seed:
        rows.extend(_dup_rows(store_seed, prefix="fxjs", copies=14))
    for block in (build_noise_rows(), build_plan_handoff_rows(), build_technical_rows()):
        rows.extend(block)
        rows.extend(_dup_rows(block, prefix="fxjd", copies=24))
    return rows


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


def build_v5k_extract_rows(
    fixture_ids: list[str] | None = None,
    *,
    fixtures_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Store-only rows with substring-grounded labels (minimal_extract training)."""
    rows: list[dict[str, Any]] = []
    for case in load_fixture_cases(fixtures_path):
        case_id = str(case["id"])
        if fixture_ids is not None and case_id not in fixture_ids:
            continue
        if str(case.get("expectAction") or "store") == "ignore":
            continue
        llm_response = str(case["llmResponse"])
        key_tokens = [str(t) for t in (case.get("keyTokens") or []) if str(t).strip()]
        content = forced_grounded_store_content(llm_response, key_tokens)
        rows.append({
            "id": f"extract-fixture-{case_id}",
            "input": remember_input(llm_response, source_id=case_id, source_kind="prod_extract"),
            "expected": _store_expected(
                llm_response,
                content,
                tags=[f"prod_eval_id:{case_id}"],
                reasoning=content,
            ),
            "source": "v5k_extract_fixture",
        })
    return rows
