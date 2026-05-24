from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from nano_psm.dataset import NanoPsmJsonlDataset, collate_batch, load_jsonl
from nano_psm.evaluate import move_batch, resolve_device
from nano_psm.model import NanoPsmConfig, build_model, require_torch
from nano_psm.schema import ACTIONS, MEMORY_TYPES
from nano_psm.tokenizer import HashTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Nano PSM validation predictions.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-correct", action="store_true")
    args = parser.parse_args()

    torch, _ = require_torch()
    from torch.utils.data import DataLoader

    config_doc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = NanoPsmConfig(**config_doc["model"])
    device = resolve_device(torch, args.device)
    tokenizer = HashTokenizer(model_config.vocab_size, model_config.max_sequence_length)
    examples = load_jsonl(args.validation)
    dataset = NanoPsmJsonlDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

    wrapped = build_model(model_config)
    model = wrapped.module.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    by_id = {example.id: example for example in examples}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["input_ids"], batch["attention_mask"])
            action_pred = output["action_logits"].argmax(dim=-1)
            memory_type_pred = output["memory_type_logits"].argmax(dim=-1)
            indexable_pred = (output["indexable_logits"].sigmoid() >= 0.5).long()
            fact_count_pred = output["fact_count_logits"].argmax(dim=-1)
            recall_count_pred = output["recall_count_logits"].argmax(dim=-1)

            for index, example_id in enumerate(batch["ids"]):
                expected_action = ACTIONS[int(batch["action"][index].item())]
                predicted_action = ACTIONS[int(action_pred[index].item())]
                expected_memory_type = MEMORY_TYPES[int(batch["memory_type"][index].item())]
                predicted_memory_type = MEMORY_TYPES[int(memory_type_pred[index].item())]
                expected_indexable = bool(int(batch["has_indexables"][index].item()))
                predicted_indexable = bool(int(indexable_pred[index].item()))
                expected_fact_count = int(batch["fact_count"][index].item())
                predicted_fact_count = int(fact_count_pred[index].item())
                expected_recall_count = int(batch["recall_count"][index].item())
                predicted_recall_count = int(recall_count_pred[index].item())

                correct = (
                    expected_action == predicted_action
                    and expected_memory_type == predicted_memory_type
                    and expected_indexable == predicted_indexable
                    and expected_fact_count == predicted_fact_count
                    and expected_recall_count == predicted_recall_count
                )
                if correct and not args.show_correct:
                    continue

                example = by_id[example_id]
                rows.append(
                    {
                        "id": example_id,
                        "source": example.input.get("source"),
                        "correct": correct,
                        "expected": {
                            "action": expected_action,
                            "memory_type": expected_memory_type,
                            "has_indexables": expected_indexable,
                            "fact_count": expected_fact_count,
                            "recall_count": expected_recall_count,
                        },
                        "predicted": {
                            "action": predicted_action,
                            "memory_type": predicted_memory_type,
                            "has_indexables": predicted_indexable,
                            "fact_count": predicted_fact_count,
                            "recall_count": predicted_recall_count,
                        },
                        "input": compact_input(example.input),
                    }
                )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    print(json.dumps(rows[: args.limit], indent=2, ensure_ascii=False))


def compact_input(value: dict[str, object]) -> dict[str, object]:
    compact = dict(value)
    for key in ("conversation", "messages", "turns"):
        if key in compact:
            compact[key] = compact_text(compact[key])
    return compact


def compact_text(value: object, max_chars: int = 900) -> object:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    if len(text) <= max_chars:
        return value
    return text[:max_chars] + "..."


if __name__ == "__main__":
    main()
