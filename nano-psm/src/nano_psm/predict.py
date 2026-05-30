from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from nano_psm.dataset import serialize_example
from nano_psm.evaluate import resolve_device
from nano_psm.model import NanoPsmConfig, build_model, require_torch
from nano_psm.schema import ACTIONS, MEMORY_TYPES, TrainingExample
from nano_psm.tokenizer import HashTokenizer


INSTRUCTION = (
    "Perform the PSM memory operation for the current input. Return JSON only using the target schema. "
    "Do not use legacy keys such as operation or assistant_response. Do not write generic User when a speaker name is available. "
    "Only extract facts that are explicitly supported by evidence_text. Create compact indexables for stored memories so later recall can use mnemonic cues. "
    "For recall inputs, select grounded memory ids and indexable keys; do not answer from general knowledge."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Nano PSM storage predictions as JSONL.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch, _ = require_torch()
    config_doc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = NanoPsmConfig(**config_doc["model"])
    device = resolve_device(torch, args.device)
    tokenizer = HashTokenizer(model_config.vocab_size, model_config.max_sequence_length)

    wrapped = build_model(model_config)
    model = wrapped.module.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            prediction = predict_one(torch, model, tokenizer, row, device)
        except Exception as exc:  # noqa: BLE001 - streaming CLI should return row-level errors.
            prediction = {
                "action": "ignore",
                "memory": None,
                "facts": [],
                "indexables": [],
                "reasoning": f"Nano PSM prediction failed: {exc}",
                "parse_error": str(exc),
            }
        print(json.dumps(prediction, ensure_ascii=False), flush=True)


def predict_one(torch, model, tokenizer: HashTokenizer, row: dict[str, Any], device) -> dict[str, Any]:
    example = TrainingExample(
        id=str(row.get("id") or "prediction"),
        instruction=str(row.get("instruction") or INSTRUCTION),
        input=dict(row.get("input") or {}),
        output={"action": "ignore"},
    )
    input_ids, attention_mask = tokenizer.encode(serialize_example(example))
    with torch.no_grad():
        output = model(
            torch.tensor([input_ids], dtype=torch.long, device=device),
            torch.tensor([attention_mask], dtype=torch.float32, device=device),
        )
        action_probs = output["action_logits"].softmax(dim=-1)[0]
        memory_type_probs = output["memory_type_logits"].softmax(dim=-1)[0]
        action_id = int(action_probs.argmax().item())
        memory_type_id = int(memory_type_probs.argmax().item())
        scores = [float(value) for value in output["scores"][0].detach().cpu().tolist()]
        indexable_probability = float(output["indexable_logits"].sigmoid()[0].detach().cpu().item())
        fact_count = int(output["fact_count_logits"].argmax(dim=-1)[0].item())
        recall_count = int(output["recall_count_logits"].argmax(dim=-1)[0].item())

    action = ACTIONS[action_id]
    memory_type = MEMORY_TYPES[memory_type_id]
    memory = None if action in {"ignore", "recall_context"} else {
        "content": durable_content(example.input),
        "type": "semantic" if memory_type == "semantic" else "episodic",
        "strength": round(scores[0], 6),
        "decay_rate": round(scores[1], 6),
        "emotional_weight": round(scores[2], 6),
        "confidence": round(scores[3], 6),
        "tags": compact_tags(example.input),
    }
    return {
        "action": action,
        "memory": memory,
        "facts": [],
        "indexables": [] if indexable_probability < 0.5 else [{
            "kind": "mnemonic",
            "key": mnemonic_key(memory["content"] if memory else durable_content(example.input)),
            "target_type": memory["type"] if memory else "none",
            "target_id": "",
            "salience": round(indexable_probability, 6),
            "reconstructive_hint": memory["content"] if memory else durable_content(example.input),
            "evidence_text": current_turn_text(example.input),
            "tags": compact_tags(example.input),
        }],
        "updates": [],
        "conflicts": [],
        "reasoning": "Nano PSM structured checkpoint prediction.",
        "confidence": round(float(action_probs[action_id].item()), 6),
        "nano": {
            "memory_type": memory_type,
            "memory_type_confidence": round(float(memory_type_probs[memory_type_id].item()), 6),
            "action_confidence": round(float(action_probs[action_id].item()), 6),
            "indexable_probability": round(indexable_probability, 6),
            "fact_count": fact_count,
            "recall_count": recall_count,
        },
    }


def durable_content(input_value: dict[str, Any]) -> str:
    current = input_value.get("current_turn") if isinstance(input_value.get("current_turn"), dict) else {}
    speaker = str(current.get("speaker") or "").strip()
    text = current_turn_text(input_value)
    if speaker and speaker.lower() not in {"user", "unknown"}:
        return f"{speaker} said: {text}" if text else speaker
    return text or json.dumps(input_value, ensure_ascii=False, sort_keys=True)[:500]


def current_turn_text(input_value: dict[str, Any]) -> str:
    current = input_value.get("current_turn") if isinstance(input_value.get("current_turn"), dict) else {}
    return str(current.get("text") or input_value.get("text") or "").strip()


def compact_tags(input_value: dict[str, Any]) -> list[str]:
    tags = []
    source_kind = str(input_value.get("source_kind") or "").strip()
    if source_kind:
        tags.append(source_kind)
    current = input_value.get("current_turn") if isinstance(input_value.get("current_turn"), dict) else {}
    session = str(current.get("session") or "").strip()
    if session:
        tags.append(f"session:{session}")
    speaker = str(current.get("speaker") or "").strip()
    if speaker:
        tags.append(f"speaker:{speaker}")
    return tags[:8]


def mnemonic_key(text: str) -> str:
    import re

    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
    return "-".join(tokens[:5]) or "memory"


if __name__ == "__main__":
    main()
