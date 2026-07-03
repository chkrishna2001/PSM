from __future__ import annotations

import json
from typing import Any

from prod_memory.grounding import (
    has_curriculum_bleed,
    grounding_overlap_score,
    stored_text_from_decision,
    would_model_store,
)
from prod_memory.hf_prompts import compact_storage_json
from psm_model.remember_cli import apply_product_boundary


def _normalize_action(action: str) -> str:
    return str(action or "").strip().lower()


def _expected_store(expected: dict[str, Any]) -> bool:
    return _normalize_action(str(expected.get("action") or "")) not in {"ignore", "ignore_noise"}


def score_storage_decision(
    *,
    remember_target: str,
    expected: dict[str, Any],
    raw: str,
) -> dict[str, Any]:
    """Rule reward for StorageDecision JSON (0..1). ponytail: no LLM judge."""
    report = apply_product_boundary(raw, output_format="json")
    parsed = report.get("parsed") if isinstance(report.get("parsed"), dict) else {}
    parse_valid = bool(parsed) and report.get("repair_status") != "unrecoverable"
    exp_store = _expected_store(expected)
    pred_store = would_model_store(parsed)
    exp_action = _normalize_action(str(expected.get("action") or ""))
    pred_action = _normalize_action(str(parsed.get("action") or ""))
    action_match = exp_action == pred_action
    store_match = exp_store == pred_store

    stored_text = stored_text_from_decision(parsed)
    overlap = grounding_overlap_score(remember_target, stored_text)
    content_grounded = (not pred_store) or bool(overlap["grounded"])
    no_bleed = not (pred_store and has_curriculum_bleed(stored_text))
    complete_json = parse_valid and raw.strip().endswith("}") and "..." not in raw[-24:]

    components = {
        "parse_valid": 1.0 if parse_valid else 0.0,
        "action_match": 1.0 if action_match else 0.0,
        "store_match": 1.0 if store_match else 0.0,
        "content_grounded": 1.0 if content_grounded else 0.0,
        "no_bleed": 1.0 if no_bleed else 0.0,
        "complete_json": 1.0 if complete_json else 0.0,
    }
    weights = {
        "parse_valid": 0.25,
        "action_match": 0.20,
        "store_match": 0.15,
        "content_grounded": 0.20,
        "no_bleed": 0.10,
        "complete_json": 0.10,
    }
    reward = sum(components[k] * weights[k] for k in components)
    return {
        "reward": round(reward, 4),
        "components": components,
        "parse_valid": parse_valid,
        "action_match": action_match,
        "store_match": store_match,
        "content_grounded": content_grounded,
        "no_bleed": no_bleed,
        "complete_json": complete_json,
        "repair_status": report.get("repair_status"),
        "issues": report.get("issues"),
    }


def chosen_rejected_gap(chosen_raw: str, rejected_raw: str, **kwargs: Any) -> float:
    chosen = score_storage_decision(raw=chosen_raw, **kwargs)
    rejected = score_storage_decision(raw=rejected_raw, **kwargs)
    return float(chosen["reward"] - rejected["reward"])


def compact_expected_json(expected: dict[str, Any]) -> str:
    return compact_storage_json(expected)
