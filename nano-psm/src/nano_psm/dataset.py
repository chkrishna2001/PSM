from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .schema import TrainingExample, action_to_id, memory_type_to_id, normalize_action
from .tokenizer import HashTokenizer


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    rows: list[TrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            output = dict(raw.get("output") or {})
            action = normalize_action(output.get("action"))
            if not action:
                continue
            try:
                action_to_id(action)
            except ValueError:
                continue
            rows.append(
                TrainingExample(
                    id=str(raw.get("id") or f"{Path(path).stem}-{line_number}"),
                    instruction=str(raw.get("instruction") or raw.get("task") or ""),
                    input=dict(raw.get("input") or {}),
                    output=output,
                )
            )
    return rows


def iter_labels(examples: list[TrainingExample]) -> Iterator[dict[str, int]]:
    for example in examples:
        memory = example.output.get("memory")
        memory_type = memory.get("type") if isinstance(memory, dict) else None
        yield {
            "action": action_to_id(str(example.output.get("action"))),
            "memory_type": memory_type_to_id(memory_type),
        }


def serialize_example(example: TrainingExample) -> str:
    payload = {
        "instruction": example.instruction,
        "input": example.input,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def targets_for_example(example: TrainingExample) -> dict[str, Any]:
    output = example.output
    action = normalize_action(output.get("action"))
    memory = output.get("memory")
    memory_type = memory.get("type") if isinstance(memory, dict) else None
    recall = output.get("recall") if isinstance(output.get("recall"), dict) else {}
    has_memory_target = isinstance(memory, dict) and action not in {"ignore", "recall_context"}
    return {
        "action": action_to_id(action),
        "memory_type": memory_type_to_id(memory_type),
        "scores": memory_scores(memory) if has_memory_target else [0.0, 0.0, 0.0, 0.0],
        "has_score_target": 1.0 if has_memory_target else 0.0,
        "has_indexables": 1.0 if len(output.get("indexables") or []) > 0 else 0.0,
        "fact_count": min(len(output.get("facts") or []), 8),
        "recall_count": min(len(recall.get("selected_memory_ids") or []), 8),
    }


def memory_scores(memory: object) -> list[float]:
    return [
        score_value(memory, "strength", default=0.75),
        score_value(memory, "decay_rate", default=0.02),
        score_value(memory, "emotional_weight", default=0.1),
        score_value(memory, "confidence", default=0.85),
    ]


def score_value(memory: object, key: str, default: float = 0.0) -> float:
    if isinstance(memory, dict):
        value = memory.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return default


class NanoPsmJsonlDataset:
    def __init__(self, examples: list[TrainingExample], tokenizer: HashTokenizer) -> None:
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        input_ids, attention_mask = self.tokenizer.encode(serialize_example(example))
        return {
            "id": example.id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets_for_example(example),
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for Nano PSM training. Use the Colab notebook or install torch.") from exc

    targets = [item["targets"] for item in batch]
    return {
        "ids": [item["id"] for item in batch],
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.float32),
        "action": torch.tensor([target["action"] for target in targets], dtype=torch.long),
        "memory_type": torch.tensor([target["memory_type"] for target in targets], dtype=torch.long),
        "scores": torch.tensor([target["scores"] for target in targets], dtype=torch.float32),
        "has_score_target": torch.tensor([target["has_score_target"] for target in targets], dtype=torch.float32),
        "has_indexables": torch.tensor([target["has_indexables"] for target in targets], dtype=torch.float32),
        "fact_count": torch.tensor([target["fact_count"] for target in targets], dtype=torch.long),
        "recall_count": torch.tensor([target["recall_count"] for target in targets], dtype=torch.long),
    }
