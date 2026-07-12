"""Consolidation DPO calibration: fix the store_episodic vs update_existing confusion.

Three rounds of plain SFT (build_consolidation_rows.py) left the model unable to tell
"genuinely distinct fact" (store_episodic) apart from "elaboration of the same fact"
(update_existing) -- it predicts update_existing for every store_episodic case, 0/7 on the
2026-07-10 round-3 held-out eval, despite 100% on the other two action types. Plain SFT
volume of each class independently hasn't fixed this; this builds explicit contrastive DPO
pairs on exactly the confused cases: chosen=correct store_episodic decision,
rejected=the specific wrong update_existing decision the model actually produces.
"""
from __future__ import annotations

import json
from typing import Any

from prod_memory.build_consolidation_rows import (
    _TRAIN_STORE_PAIRS,
    _memory,
)
from prod_memory.hf_prompts import apply_chat_prompt  # noqa: F401 (re-exported for callers)


def _consolidation_input(new_text: str, existing_id: str, existing_text: str) -> dict[str, Any]:
    return {
        "operation": "consolidate",
        "new_memory": _memory(new_text),
        "existing_memory": {"id": existing_id, **_memory(existing_text)},
    }


def _prompt_messages(input_payload: dict[str, Any]) -> list[dict[str, str]]:
    from psm_model.prompts import CONSOLIDATION_SYSTEM_INSTRUCTION

    user = (
        "Decide store_episodic, update_existing, or flag_conflict as JSON only with action, "
        "target_memory_id, merged_content, and reasoning.\n"
        + json.dumps(input_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": CONSOLIDATION_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user},
    ]


def build_consolidation_dpo_pairs() -> list[dict[str, Any]]:
    """DPO pairs: chosen=store_episodic (correct), rejected=update_existing (the model's
    actual confusion) -- one pair per hand-verified store_episodic training example."""
    pairs: list[dict[str, Any]] = []
    for row_id, new_text, existing_text in _TRAIN_STORE_PAIRS:
        existing_id = f"mem-{row_id}-existing"
        input_payload = _consolidation_input(new_text, existing_id, existing_text)
        prompt = _prompt_messages(input_payload)

        chosen = {
            "action": "store_episodic",
            "target_memory_id": None,
            "merged_content": None,
            "reasoning": "New memory is a distinct fact, not a restatement or contradiction of the existing one; store independently.",
        }
        rejected = {
            "action": "update_existing",
            "target_memory_id": existing_id,
            "merged_content": f"{existing_text} {new_text}",
            "reasoning": "New memory restates/elaborates the same underlying fact as the existing one; merge into a single updated memory.",
        }
        pairs.append({
            "id": f"consolidation-dpo-{row_id}",
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": json.dumps(chosen, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}],
            "rejected": [{"role": "assistant", "content": json.dumps(rejected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}],
            "source": "consolidation_dpo_store_vs_update",
            "variant": "store_vs_update",
        })
    return pairs
