from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prod_memory.build_binary_fixture_rows import _dup_rows
from prod_memory.build_minimal_fixture_rows import build_json_fixture_rows
from prod_memory.curriculum_sources import build_noise_rows
from prod_memory.hf_prompts import compact_storage_json, storage_inference_messages
from prod_memory.row_validation import remember_target_from_input, validate_prod_row
from prod_memory.storage_rewards import chosen_rejected_gap, compact_expected_json

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERSATION = PACKAGE_ROOT / "data" / "hf-prod-conversation-gemma-all.jsonl"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _wrong_action_rejected(expected: dict[str, Any]) -> str:
    action = str(expected.get("action") or "").lower()
    if action in {"ignore", "ignore_noise"}:
        payload = {
            "action": "store_episodic",
            "memory": {"content": "unrelated synthetic fact", "type": "episodic"},
            "facts": [],
            "indexables": [],
            "reasoning": "wrong gate",
        }
    else:
        payload = {
            "action": "ignore",
            "memory": None,
            "facts": [],
            "indexables": [],
            "reasoning": "wrong gate",
        }
    return compact_storage_json(payload)


def _truncated_rejected(chosen: str) -> str:
    cut = max(32, int(len(chosen) * 0.62))
    return chosen[:cut]


def _malformed_rejected(chosen: str) -> str:
    return chosen.replace("}", "", 1)


def _ungrounded_store_rejected(remember_target: str) -> str:
    payload = {
        "action": "store_episodic",
        "memory": {"content": "checkpoint gate6 runpod nvidi-smi probe", "type": "episodic"},
        "facts": [],
        "indexables": [],
        "reasoning": "synthetic ungrounded",
    }
    return compact_storage_json(payload)


def _dpo_pair(
    row_id: str,
    remember_target: str,
    expected: dict[str, Any],
    rejected_raw: str,
    *,
    source: str,
    variant: str,
) -> dict[str, Any] | None:
    chosen_raw = compact_expected_json(expected)
    gap = chosen_rejected_gap(
        chosen_raw,
        rejected_raw,
        remember_target=remember_target,
        expected=expected,
    )
    if gap < 0.15:
        return None
    prompt = storage_inference_messages(remember_target, output_format="json")
    return {
        "id": f"{row_id}-dpo-{variant}",
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen_raw}],
        "rejected": [{"role": "assistant", "content": rejected_raw}],
        "source": source,
        "variant": variant,
        "reward_gap": round(gap, 4),
    }


def build_v5o_storage_dpo_rows(
    *,
    conversation_path: Path = DEFAULT_CONVERSATION,
    include_fixtures: bool = True,
    include_noise: bool = True,
) -> list[dict[str, Any]]:
    """Full StorageDecision DPO pairs (not binary gate)."""
    seed_rows: list[dict[str, Any]] = []
    if conversation_path.is_file():
        for row in _load_rows(conversation_path):
            try:
                validate_prod_row(row)
            except ValueError:
                continue
            seed_rows.append(row)
    if include_fixtures:
        seed_rows.extend(build_json_fixture_rows())
    if include_noise:
        seed_rows.extend(build_noise_rows())

    pairs: list[dict[str, Any]] = []
    for row in seed_rows:
        remember_target = remember_target_from_input(row["input"])
        if not remember_target:
            continue
        expected = row["expected"]
        row_id = str(row.get("id") or "row")
        source = str(row.get("source") or "v5o_dpo")
        chosen_raw = compact_expected_json(expected)
        variants = {
            "wrong_action": _wrong_action_rejected(expected),
            "truncated": _truncated_rejected(chosen_raw),
            "malformed": _malformed_rejected(chosen_raw),
        }
        if str(expected.get("action") or "").lower() not in {"ignore", "ignore_noise"}:
            variants["ungrounded"] = _ungrounded_store_rejected(remember_target)
        for variant, rejected_raw in variants.items():
            pair = _dpo_pair(
                row_id,
                remember_target,
                expected,
                rejected_raw,
                source=source,
                variant=variant,
            )
            if pair:
                pairs.append(pair)

    store_pairs = [
        p
        for p in pairs
        if json.loads(p["chosen"][0]["content"]).get("action") not in {"ignore", "ignore_noise"}
    ]
    ignore_pairs = [
        p
        for p in pairs
        if json.loads(p["chosen"][0]["content"]).get("action") in {"ignore", "ignore_noise"}
    ]
    out = list(pairs)
    out.extend(_dup_rows(store_pairs, prefix="dpos", copies=2))
    out.extend(_dup_rows(ignore_pairs, prefix="dpoi", copies=4))
    return out
