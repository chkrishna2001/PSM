from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prod_memory.indexable_labels import build_indexable_labels

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = PACKAGE_ROOT / "fixtures" / "cases.json"

MEMORY_SUMMARIES: dict[str, str] = {
    "plan-01-handoff": (
        "Phase 1 baseline eval measures saved model steps with grounding, bleed, and fail-safe metrics."
    ),
    "plan-02-chunking": (
        "Chunk long assistant handoffs on markdown headers and numbered steps near 600-1200 tokens per chunk."
    ),
    "cursor-01-summary": (
        "Prod remember path rejects ungrounded storage, blocks curriculum bleed, and uses 384 max_new_tokens."
    ),
    "cursor-02-debug": (
        "Training templates hurt assistant-text extraction; high Hit@k can hide missing grounded facts."
    ),
    "workflow-review-pr": (
        "Review a pull request by getting PR info, checking the target branch, listing changed files, "
        "reviewing each file, then approving or requesting changes."
    ),
    "workflow-runpod": (
        "RunPod launch uses two-phase deploy, verify tmux and GPU util, "
        "and export PSM_RUNPOD=1 so training uses the GPU."
    ),
    "technical-eslint": (
        "Use explicit return types on exported psm-core functions and avoid default exports."
    ),
    "technical-api": (
        "remember() extracts from llmResponse assistant text; userMessage is optional real user context."
    ),
}


def remember_input(
    llm_response: str,
    *,
    source_id: str,
    source_kind: str = "llm_response",
    user_message: str | None = None,
) -> dict[str, Any]:
    conversation: list[dict[str, str]] | str
    if user_message:
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": llm_response},
        ]
    else:
        conversation = [{"role": "assistant", "content": llm_response}]
    payload: dict[str, Any] = {
        "operation": "remember_llm_response",
        "conversation": conversation,
        "source_kind": source_kind,
        "source_id": source_id,
    }
    return payload


def _memory_payload(content: str, *, memory_type: str = "episodic", tags: list[str]) -> dict[str, Any]:
    return {
        "content": content,
        "type": memory_type,
        "strength": 0.86,
        "decay_rate": 0.02,
        "emotional_weight": 0.22,
        "confidence": 0.92,
        "tags": tags,
    }


def _store_expected(
    llm_response: str,
    memory_content: str,
    *,
    tags: list[str],
    reasoning: str,
    facts: list[dict[str, Any]] | None = None,
    memory_type: str = "episodic",
) -> dict[str, Any]:
    facts = facts or []
    return {
        "action": "store_episodic" if memory_type == "episodic" else "promote_semantic",
        "memory": _memory_payload(memory_content, memory_type=memory_type, tags=tags),
        "facts": facts,
        "indexables": build_indexable_labels(
            llm_response=llm_response,
            memory_content=memory_content,
            tags=tags,
            facts=facts,
        ),
        "reasoning": reasoning,
    }


def _ignore_expected(reasoning: str) -> dict[str, Any]:
    return {"action": "ignore", "memory": None, "facts": [], "indexables": [], "reasoning": reasoning}


def load_fixture_cases(path: Path | None = None) -> list[dict[str, Any]]:
    fixture_path = path or FIXTURES
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    return cases if isinstance(cases, list) else []


def build_fixture_rows(path: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in load_fixture_cases(path):
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "case")
        llm_response = str(case.get("llmResponse") or "").strip()
        if not llm_response:
            continue
        expect_action = str(case.get("expectAction") or "store")
        suite = str(case.get("suite") or "prod")
        tags = [f"prod_eval_suite:{suite}", f"prod_eval_id:{case_id}"]

        if expect_action == "ignore":
            expected = _ignore_expected("No durable memory to store from this assistant text.")
        else:
            memory_content = MEMORY_SUMMARIES.get(case_id)
            if not memory_content:
                continue
            memory_type = "semantic" if suite == "technical" else "episodic"
            expected = _store_expected(
                llm_response,
                memory_content,
                tags=tags,
                reasoning="Extract grounded durable information from assistant remember_target.",
                memory_type=memory_type,
            )

        rows.append({
            "id": f"fixture-{case_id}",
            "input": remember_input(llm_response, source_id=case_id, source_kind=f"prod_{suite}"),
            "expected": expected,
            "source": f"prod_fixture:{suite}",
        })
    return rows


def build_plan_handoff_rows() -> list[dict[str, Any]]:
    templates = [
        (
            "handoff-auth",
            "## Auth rollout\n\nShip OAuth refresh tokens with 15-minute access TTL. "
            "Document migration steps for existing sessions and add smoke tests for token refresh.",
            "Auth rollout ships OAuth refresh tokens with 15-minute access TTL and session migration steps.",
            ["auth", "oauth", "rollout"],
        ),
        (
            "handoff-observability",
            "### Observability\n\nAdd structured logs for remember() grounding rejects and indexable recalls. "
            "Track fail-safe ignore rate per suite in prod-memory eval artifacts.",
            "Observability adds structured logs for grounding rejects, indexable recalls, and fail-safe rates.",
            ["observability", "grounding", "indexables"],
        ),
        (
            "handoff-release",
            "## Release checklist\n\nRun prod grounding eval, verify regression gate on x2 expanded subset, "
            "attach eval JSON to HF manifest before deleting the pod.",
            "Release checklist requires prod grounding eval, regression gate, and HF eval manifest before pod delete.",
            ["release", "grounding", "manifest"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for row_id, llm_response, memory_content, tags in templates:
        rows.append({
            "id": f"plan-{row_id}",
            "input": remember_input(llm_response, source_id=row_id, source_kind="agent_plan"),
            "expected": _store_expected(
                llm_response,
                memory_content,
                tags=["plan", *tags],
                reasoning="Store durable plan facts from the assistant handoff.",
            ),
            "source": "prod_plan_handoff",
        })
    return rows


def build_technical_rows() -> list[dict[str, Any]]:
    templates = [
        (
            "api-versioning",
            "API rule: version all breaking remember() response shape changes and keep indexables optional in schema validation.",
            "Breaking remember() response changes must be versioned; indexables stay optional in schema validation.",
            ["api", "versioning", "indexables"],
        ),
        (
            "test-policy",
            "Testing rule: every prod-memory phase adds unit tests; grounding eval rows must pass bleed and overlap checks at build time.",
            "Prod-memory phases require unit tests and build-time grounding validation for curriculum rows.",
            ["testing", "grounding", "curriculum"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for row_id, llm_response, memory_content, tags in templates:
        rows.append({
            "id": f"technical-{row_id}",
            "input": remember_input(llm_response, source_id=row_id, source_kind="technical_rule"),
            "expected": _store_expected(
                llm_response,
                memory_content,
                tags=["technical", *tags],
                reasoning="Store durable technical rule from assistant text.",
                memory_type="semantic",
            ),
            "source": "prod_technical_rule",
        })
    return rows


def build_noise_rows() -> list[dict[str, Any]]:
    templates = [
        ("noise-ack", "Okay, sounds good. I will wait for your update."),
        ("noise-thanks", "Thanks, that helps. Let me know if anything else comes up."),
        ("noise-empty", "I don't have any durable facts to store from this assistant reply."),
        ("noise-lol", "Haha, fair point."),
    ]
    rows: list[dict[str, Any]] = []
    for row_id, llm_response in templates:
        rows.append({
            "id": row_id,
            "input": remember_input(llm_response, source_id=row_id, source_kind="noise"),
            "expected": _ignore_expected("Assistant reply has no durable memory content."),
            "source": "prod_noise",
        })
    return rows


def build_primary_source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(build_fixture_rows())
    rows.extend(build_plan_handoff_rows())
    rows.extend(build_technical_rows())
    rows.extend(build_noise_rows())

    seen_ids: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        unique_rows.append(row)
    return unique_rows
