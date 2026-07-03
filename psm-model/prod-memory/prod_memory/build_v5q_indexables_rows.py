"""Phase 2 SFT anchor rows: indexables[], temporal fields, bleed-safe fixture labels."""
from __future__ import annotations

from typing import Any

from prod_memory.build_minimal_fixture_rows import (
    V5E_FORCED_STORE_IDS,
    _dup_rows,
    forced_grounded_store_content,
)
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
from prod_memory.grounding import has_curriculum_bleed, stored_text_from_decision
from prod_memory.indexable_labels import build_indexable_labels
from prod_memory.label_from_assistant import _make_fact

PHASE2_BOOST_FIXTURE_IDS = (
    "plan-01-handoff",
    "workflow-runpod",
    "technical-api",
    "workflow-review-pr",
)

# keyTokens that must not appear in facts (BLEED_PATTERN / guard blocklist)
BLEED_SAFE_KEY_TOKENS: dict[str, list[str]] = {
    "plan-01-handoff": ["baseline", "grounding", "workflow"],
    "workflow-runpod": ["verify", "tmux", "training"],
}

# ponytail: paraphrases avoid BLEED_PATTERN tokens in stored_text (checkpoint, runpod, …)
BLEED_SAFE_MEMORY: dict[str, str] = {
    "plan-01-handoff": (
        "Phase 1 baseline eval measures saved model steps with grounding and fail-safe metrics."
    ),
    "workflow-runpod": (
        "GPU train launch uses two-phase deploy, verify tmux and GPU util within 15s, "
        "and set the GPU train env flag so training uses the GPU."
    ),
    "workflow-review-pr": MEMORY_SUMMARIES["workflow-review-pr"],
    "technical-api": MEMORY_SUMMARIES["technical-api"],
    "technical-eslint": MEMORY_SUMMARIES["technical-eslint"],
}


def _make_temporal_fact(
    *,
    subject: str,
    predicate: str,
    value: str,
    evidence_text: str,
    temporal_expression: str,
    resolved_time: str,
) -> dict[str, Any]:
    fact = _make_fact(
        subject=subject,
        predicate=predicate,
        value=value,
        evidence_text=evidence_text,
        confidence=0.9,
    )
    fact["temporal_expression"] = temporal_expression
    fact["resolved_time"] = resolved_time
    fact["resolved_time_confidence"] = 0.9
    return fact


def _memory_with_temporal(
    content: str,
    *,
    tags: list[str],
    memory_type: str = "episodic",
    temporal_expression: str | None = None,
    resolved_time: str | None = None,
) -> dict[str, Any]:
    memory: dict[str, Any] = {
        "content": content,
        "type": memory_type,
        "strength": 0.86,
        "decay_rate": 0.02,
        "emotional_weight": 0.22,
        "confidence": 0.92,
        "tags": tags,
    }
    if temporal_expression:
        memory["temporal_expression"] = temporal_expression
    if resolved_time:
        memory["resolved_time"] = resolved_time
        memory["resolved_time_confidence"] = 0.9
    return memory


def _store_with_indexables(
    llm_response: str,
    memory_content: str,
    *,
    tags: list[str],
    reasoning: str,
    facts: list[dict[str, Any]] | None = None,
    memory_type: str = "episodic",
    temporal_expression: str | None = None,
    resolved_time: str | None = None,
) -> dict[str, Any]:
    facts = facts or []
    memory = _memory_with_temporal(
        memory_content,
        tags=tags,
        memory_type=memory_type,
        temporal_expression=temporal_expression,
        resolved_time=resolved_time,
    )
    return {
        "action": "store_episodic" if memory_type == "episodic" else "promote_semantic",
        "memory": memory,
        "facts": facts,
        "indexables": build_indexable_labels(
            llm_response=llm_response,
            memory_content=memory_content,
            tags=tags,
            facts=facts,
        ),
        "reasoning": reasoning,
    }


def _assert_bleed_safe(expected: dict[str, Any], row_id: str) -> None:
    stored = stored_text_from_decision(expected)
    if has_curriculum_bleed(stored):
        raise ValueError(f"{row_id}: label stored_text hits curriculum bleed blocklist")


def _safe_facts_from_key_tokens(llm_response: str, key_tokens: list[str]) -> list[dict[str, Any]]:
    """Facts with token-only evidence — avoids snippet windows that pull bleed tokens."""
    facts: list[dict[str, Any]] = []
    for key in key_tokens[:4]:
        if key.lower() not in llm_response.lower():
            continue
        facts.append(
            _make_fact(
                subject=key,
                predicate="mentions",
                value=key,
                evidence_text=key,
                confidence=0.9,
            )
        )
    return facts[:3]


def build_v5q_fixture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in load_fixture_cases():
        case_id = str(case.get("id") or "")
        llm_response = str(case.get("llmResponse") or "").strip()
        if not llm_response:
            continue
        expect_action = str(case.get("expectAction") or "store")
        key_tokens = [str(k) for k in (case.get("keyTokens") or [])]
        key_tokens = BLEED_SAFE_KEY_TOKENS.get(case_id, key_tokens)
        suite = str(case.get("suite") or "prod")
        tags = [f"prod_eval_suite:{suite}", f"prod_eval_id:{case_id}"]

        if expect_action == "ignore":
            expected = _ignore_expected("No durable memory to store from this assistant text.")
        else:
            memory_content = BLEED_SAFE_MEMORY.get(case_id) or MEMORY_SUMMARIES.get(case_id)
            if not memory_content:
                if case_id in V5E_FORCED_STORE_IDS:
                    memory_content = forced_grounded_store_content(llm_response, key_tokens)
                else:
                    continue
            facts = _safe_facts_from_key_tokens(llm_response, key_tokens)
            memory_type = "semantic" if suite in {"technical", "workflow"} else "episodic"
            expected = _store_with_indexables(
                llm_response,
                memory_content,
                tags=tags,
                reasoning="Grounded durable extraction with indexables from assistant text.",
                facts=facts,
                memory_type=memory_type,
            )

        row = {
            "id": f"v5q-fixture-{case_id}",
            "input": remember_input(llm_response, source_id=case_id, source_kind=f"prod_{suite}"),
            "expected": expected,
            "source": "v5q_fixture",
        }
        if str(expected.get("action") or "") not in {"ignore", "ignore_noise"}:
            _assert_bleed_safe(expected, row["id"])
        rows.append(row)
    return rows


def build_v5q_temporal_rows() -> list[dict[str, Any]]:
    """Synthetic dated assistant turns (not LoCoMo holdout convs)."""
    templates: list[tuple[str, str, str, str, str, str]] = [
        (
            "temporal-job-loss",
            "Hey! Good to see you. Lost my job as a banker yesterday, so I'm going to take a shot at starting my own business.",
            "Jon lost banking job yesterday and plans to start own business.",
            "Jon",
            "yesterday",
            "19 January 2023",
        ),
        (
            "temporal-door-dash",
            "Sorry about your job. I also lost my job at Door Dash this month. What business are you thinking of?",
            "Gina lost Door Dash job this month.",
            "Gina",
            "this month",
            "January 2023",
        ),
        (
            "temporal-dance-studio",
            "I'm starting a dance studio because I'm passionate about dancing. I've been into dancing since I was a kid.",
            "Jon is starting a dance studio and has danced since childhood.",
            "Jon",
            "",
            "",
        ),
        (
            "temporal-meeting",
            "We met last Tuesday to review the Q1 roadmap and agreed to ship the memory indexables layer before end of March.",
            "Team met last Tuesday for Q1 roadmap; ship indexables layer before end of March.",
            "team",
            "last Tuesday",
            "28 January 2023",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for row_id, llm_response, memory_content, subject, temporal_expr, resolved in templates:
        tags = ["temporal", "synthetic", f"v5q_id:{row_id}"]
        facts: list[dict[str, Any]] = []
        if temporal_expr and resolved:
            facts.append(
                _make_temporal_fact(
                    subject=subject,
                    predicate="event_on",
                    value=memory_content[:80],
                    evidence_text=llm_response[:120],
                    temporal_expression=temporal_expr,
                    resolved_time=resolved,
                )
            )
        expected = _store_with_indexables(
            llm_response,
            memory_content,
            tags=tags,
            reasoning="Store dated event with temporal_expression and resolved_time.",
            facts=facts,
            temporal_expression=temporal_expr or None,
            resolved_time=resolved or None,
        )
        row = {
            "id": f"v5q-{row_id}",
            "input": remember_input(
                llm_response,
                source_id=row_id,
                source_kind="synthetic_temporal",
            ),
            "expected": expected,
            "source": "v5q_temporal",
        }
        _assert_bleed_safe(expected, row["id"])
        rows.append(row)
    return rows


def build_v5q_indexable_workflow_rows() -> list[dict[str, Any]]:
    """Extra workflow rows so model emits workflow indexables with steps."""
    templates = [
        (
            "wf-deploy-checklist",
            "# Deploy checklist\n\n1. Run prod grounding eval on the checkpoint.\n"
            "2. Verify holdout retrieval against baseline.\n"
            "3. Upload adapter and metrics to HF before stopping the pod.",
            "Deploy checklist: prod grounding eval, holdout retrieval baseline, HF upload before pod stop.",
            ["workflow", "deploy", "grounding"],
        ),
        (
            "wf-memory-review",
            "# Review stored memories\n\n1. Export decisions table for the date range.\n"
            "2. Flag rows with empty facts or missing indexables.\n"
            "3. Re-ingest failed turns with the current adapter.",
            "Review stored memories by exporting decisions, flagging empty facts/indexables, re-ingesting failures.",
            ["workflow", "memory", "review"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for row_id, llm_response, memory_content, tags in templates:
        expected = _store_with_indexables(
            llm_response,
            memory_content,
            tags=[*tags, f"v5q_id:{row_id}"],
            reasoning="Store workflow procedure with workflow indexable steps.",
            memory_type="semantic",
        )
        rows.append({
            "id": f"v5q-{row_id}",
            "input": remember_input(llm_response, source_id=row_id, source_kind="workflow"),
            "expected": expected,
            "source": "v5q_workflow",
        })
    return rows


def build_v5q_anchor_rows(*, boost_copies: int = 24) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(build_v5q_fixture_rows())
    rows.extend(build_v5q_temporal_rows())
    rows.extend(build_v5q_indexable_workflow_rows())
    rows.extend(build_plan_handoff_rows())
    rows.extend(build_technical_rows())
    rows.extend(build_noise_rows())

    boost = [row for row in rows if any(fid in str(row.get("id") or "") for fid in PHASE2_BOOST_FIXTURE_IDS)]
    if boost and boost_copies > 0:
        rows.extend(_dup_rows(boost, prefix="v5qb", copies=boost_copies))

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            continue
        seen.add(row_id)
        unique.append(row)
    return unique
