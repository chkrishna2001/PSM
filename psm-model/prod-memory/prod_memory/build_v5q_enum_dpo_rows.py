"""Phase 3 DPO pairs: valid enums (chosen) vs v5q hallucinated enums (rejected)."""
from __future__ import annotations

import copy
import json
from typing import Any

from prod_memory.build_binary_fixture_rows import _dup_rows
from prod_memory.build_v5o_storage_dpo_rows import (
    _dpo_pair,
    _hallucinated_tags_fields_rejected,
    _truncated_rejected,
)
from prod_memory.build_v5q_indexables_rows import (
    build_v5q_fixture_rows,
    build_v5q_indexable_workflow_rows,
    build_v5q_temporal_rows,
)
from prod_memory.hf_prompts import compact_storage_json
from prod_memory.row_validation import remember_target_from_input, validate_prod_row
from prod_memory.storage_rewards import compact_expected_json

# Fixtures that failed v5q eval with enum drift (extra DPO boost).
V5Q_ENUM_FAILURE_FIXTURE_IDS = frozenset({
    "cursor-02-debug",
    "workflow-review-pr",
    "technical-eslint",
    "technical-api",
    "noise-filler",
    "noise-meta",
    # v5q-dpo real-output diagnosis (2026-07-05): valid JSON, missing memory+reasoning.
    "workflow-runpod",
})

STORE_ENUM_VARIANTS = (
    "bad_memory_type_grounded",
    "bad_memory_type_empty",
    "bad_indexable_kind_semantic",
    "bad_indexable_kind_fact",
    "bad_indexable_kind_explicit",
    "truncated",
    # v5q-dpo real-output diagnosis (2026-07-05): actual failures were not truncation —
    # workflow-runpod omits memory+reasoning entirely; technical-eslint hallucinates
    # non-schema fields inside memory.tags. These variants reject those exact shapes.
    "missing_memory_reasoning",
    "hallucinated_tags_fields",
    # v5q-dpo real-output diagnosis (2026-07-09, corrected --output-format json eval on
    # holdout-coding-agent-cases.json): hallucinates indexable kind "episodic" (a memory
    # *type* value, not a valid indexable kind) — a variant not covered by the existing
    # bad_indexable_kind_{semantic,fact,explicit} set.
    "bad_indexable_kind_episodic",
)

IGNORE_ENUM_VARIANTS = (
    "ignore_store_bad_kind_fact",
    "ignore_store_bad_kind_explicit",
)


def _deep_copy_expected(expected: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(expected)


def _variant_applies(expected: dict[str, Any], variant: str) -> bool:
    memory = expected.get("memory")
    indexables = expected.get("indexables") or []
    if variant in {"bad_memory_type_grounded", "bad_memory_type_empty"}:
        return isinstance(memory, dict)
    if variant.startswith("bad_indexable_kind_"):
        return bool(indexables)
    if variant == "truncated":
        return True
    if variant in {"missing_memory_reasoning", "hallucinated_tags_fields"}:
        # Scoped to promote_semantic (technical/workflow suite) only — matches where the
        # real v5q-dpo bugs were observed (workflow-runpod, technical-eslint). A 2026-07-05
        # retrain that applied these coarse-edit variants to all store rows caused broad
        # output-quality collapse on unrelated episodic fixtures; narrowing scope + raising
        # dpo_beta (_run_hf_lora.py v5q-dpo profile) is the follow-up mitigation.
        if not (isinstance(memory, dict) and memory.get("type") == "semantic"):
            return False
        if variant == "hallucinated_tags_fields":
            return bool(memory.get("tags"))
        return True
    return False


def _mutate_expected_for_enum_reject(
    expected: dict[str, Any],
    variant: str,
    *,
    chosen_raw: str | None = None,
) -> str:
    if variant == "truncated":
        raw = chosen_raw or compact_expected_json(expected)
        return _truncated_rejected(raw)
    if variant == "hallucinated_tags_fields":
        raw = chosen_raw or compact_expected_json(expected)
        return _hallucinated_tags_fields_rejected(raw)
    if variant == "missing_memory_reasoning":
        # Mirrors the real v5q-dpo workflow-runpod bug: valid JSON for action/facts/
        # indexables, but memory+reasoning keys omitted entirely even though the action
        # requires them. compact_storage_json can't express this (it requires
        # decision["reasoning"]), so build the payload directly.
        payload = _deep_copy_expected(expected)
        return json.dumps(
            {
                "action": payload["action"],
                "facts": payload.get("facts") or [],
                "indexables": payload.get("indexables") or [],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    payload = _deep_copy_expected(expected)
    memory = payload.get("memory")
    indexables = payload.get("indexables") or []

    if variant == "bad_memory_type_grounded" and isinstance(memory, dict):
        memory["type"] = "grounded"
    elif variant == "bad_memory_type_empty" and isinstance(memory, dict):
        memory["type"] = ""
    elif variant == "bad_indexable_kind_semantic" and indexables:
        indexables[0] = dict(indexables[0])
        indexables[0]["kind"] = "semantic"
        payload["indexables"] = indexables
    elif variant == "bad_indexable_kind_fact" and indexables:
        indexables[0] = dict(indexables[0])
        indexables[0]["kind"] = "fact"
        payload["indexables"] = indexables
    elif variant == "bad_indexable_kind_explicit" and indexables:
        indexables[0] = dict(indexables[0])
        indexables[0]["kind"] = "explicit"
        payload["indexables"] = indexables
    elif variant == "bad_indexable_kind_episodic" and indexables:
        indexables[0] = dict(indexables[0])
        indexables[0]["kind"] = "episodic"
        payload["indexables"] = indexables

    return compact_storage_json(payload)


def _ignore_store_bad_indexables_rejected(remember_target: str, *, bad_kind: str) -> str:
    snippet = remember_target[:120].strip() or "acknowledgment"
    payload = {
        "action": "store_episodic",
        "memory": {"content": snippet, "type": "episodic"},
        "facts": [],
        "indexables": [{
            "kind": bad_kind,
            "key": "noise-anchor",
            "salience": 0.72,
            "reconstructive_hint": snippet[:80],
            "evidence_text": snippet,
            "tags": ["noise"],
        }],
        "reasoning": "wrong gate with hallucinated indexable kind",
    }
    return compact_storage_json(payload)


def _fixture_id_from_row(row: dict[str, Any]) -> str:
    row_id = str(row.get("id") or "")
    if row_id.startswith("v5q-fixture-"):
        return row_id.removeprefix("v5q-fixture-")
    input_payload = row.get("input") or {}
    source_id = str(input_payload.get("source_id") or "")
    if source_id:
        return source_id
    return row_id


def _fixture_id_from_pair(pair: dict[str, Any]) -> str:
    pair_id = str(pair.get("id") or "")
    for fixture_id in V5Q_ENUM_FAILURE_FIXTURE_IDS:
        if fixture_id in pair_id:
            return fixture_id
    return ""


def build_v5q_enum_dpo_rows(
    *,
    fixture_copies: int = 8,
    failure_copies: int = 6,
    include_temporal: bool = True,
    include_workflow: bool = True,
) -> list[dict[str, Any]]:
    """DPO pairs penalizing v5q enum hallucinations while keeping valid indexables."""
    seed_rows: list[dict[str, Any]] = list(build_v5q_fixture_rows())
    if include_temporal:
        seed_rows.extend(build_v5q_temporal_rows())
    if include_workflow:
        seed_rows.extend(build_v5q_indexable_workflow_rows())

    pairs: list[dict[str, Any]] = []
    for row in seed_rows:
        try:
            validate_prod_row(row)
        except ValueError:
            continue

        remember_target = remember_target_from_input(row["input"])
        if not remember_target:
            continue

        expected = row["expected"]
        row_id = str(row.get("id") or "v5q")
        action = str(expected.get("action") or "").lower()
        chosen_raw = compact_expected_json(expected)

        if action in {"ignore", "ignore_noise"}:
            ignore_variants = {
                "ignore_store_bad_kind_fact": _ignore_store_bad_indexables_rejected(
                    remember_target, bad_kind="fact",
                ),
                "ignore_store_bad_kind_explicit": _ignore_store_bad_indexables_rejected(
                    remember_target, bad_kind="explicit",
                ),
            }
            for variant, rejected_raw in ignore_variants.items():
                pair = _dpo_pair(
                    row_id,
                    remember_target,
                    expected,
                    rejected_raw,
                    source="v5q_enum_dpo",
                    variant=variant,
                )
                if pair:
                    pairs.append(pair)
            continue

        for variant in STORE_ENUM_VARIANTS:
            if not _variant_applies(expected, variant):
                continue
            rejected_raw = _mutate_expected_for_enum_reject(
                expected,
                variant,
                chosen_raw=chosen_raw,
            )
            pair = _dpo_pair(
                row_id,
                remember_target,
                expected,
                rejected_raw,
                source="v5q_enum_dpo",
                variant=variant,
            )
            if pair:
                pairs.append(pair)

    store_pairs = [
        p
        for p in pairs
        if json_action(p) not in {"ignore", "ignore_noise"}
    ]
    ignore_pairs = [
        p
        for p in pairs
        if json_action(p) in {"ignore", "ignore_noise"}
    ]
    failure_pairs = [p for p in pairs if _fixture_id_from_pair(p) in V5Q_ENUM_FAILURE_FIXTURE_IDS]

    out = list(pairs)
    out.extend(_dup_rows(store_pairs, prefix="dpos", copies=fixture_copies))
    out.extend(_dup_rows(ignore_pairs, prefix="dpoi", copies=max(2, fixture_copies // 2)))
    if failure_copies > 0 and failure_pairs:
        out.extend(_dup_rows(failure_pairs, prefix="dpof", copies=failure_copies))
    return out


def json_action(pair: dict[str, Any]) -> str:
    content = pair["chosen"][0]["content"]
    return str(json.loads(content).get("action") or "").lower()
