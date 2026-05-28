from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from nano_psm.dataset import NanoPsmJsonlDataset, collate_batch, load_jsonl
from nano_psm.model import NanoPsmConfig, build_model, require_torch
from nano_psm.tokenizer import HashTokenizer


def evaluate_model(model, data_loader, device) -> dict[str, float]:
    torch, _ = require_torch()
    model.eval()
    totals = {
        "rows": 0,
        "action_correct": 0,
        "memory_type_correct": 0,
        "indexable_correct": 0,
        "fact_count_correct": 0,
        "recall_count_correct": 0,
        "score_rows": 0,
        "score_abs_error": 0.0,
        "score_component_abs_error": None,
    }
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch(batch, device)
            output = model(batch["input_ids"], batch["attention_mask"])
            action_pred = output["action_logits"].argmax(dim=-1)
            memory_type_pred = output["memory_type_logits"].argmax(dim=-1)
            indexable_pred = (output["indexable_logits"].sigmoid() >= 0.5).float()
            fact_count_pred = output["fact_count_logits"].argmax(dim=-1)
            recall_count_pred = output["recall_count_logits"].argmax(dim=-1)
            rows = int(batch["action"].shape[0])
            totals["rows"] += rows
            totals["action_correct"] += int((action_pred == batch["action"]).sum().item())
            totals["memory_type_correct"] += int((memory_type_pred == batch["memory_type"]).sum().item())
            totals["indexable_correct"] += int((indexable_pred == batch["has_indexables"]).sum().item())
            totals["fact_count_correct"] += int((fact_count_pred == batch["fact_count"]).sum().item())
            totals["recall_count_correct"] += int((recall_count_pred == batch["recall_count"]).sum().item())
            score_mask = batch["has_score_target"] > 0
            score_rows = int(score_mask.sum().item())
            if score_rows:
                score_component_error = (output["scores"][score_mask] - batch["scores"][score_mask]).abs()
                score_error = score_component_error.mean(dim=1)
                totals["score_abs_error"] += float(score_error.sum().item())
                component_sum = score_component_error.sum(dim=0)
                if totals["score_component_abs_error"] is None:
                    totals["score_component_abs_error"] = component_sum
                else:
                    totals["score_component_abs_error"] += component_sum
                totals["score_rows"] += score_rows

    rows = max(totals["rows"], 1)
    score_rows = max(totals["score_rows"], 1)
    score_component_names = ["strength", "decay_rate", "emotional_weight", "confidence"]
    if totals["score_component_abs_error"] is None:
        score_component_mae = {f"{name}_mae": 0.0 for name in score_component_names}
    else:
        score_component_mae = {
            f"{name}_mae": float(totals["score_component_abs_error"][index].item()) / score_rows
            for index, name in enumerate(score_component_names)
        }
    return {
        "rows": float(totals["rows"]),
        "action_accuracy": totals["action_correct"] / rows,
        "memory_type_accuracy": totals["memory_type_correct"] / rows,
        "indexable_accuracy": totals["indexable_correct"] / rows,
        "fact_count_accuracy": totals["fact_count_correct"] / rows,
        "recall_count_accuracy": totals["recall_count_correct"] / rows,
        "score_mae": totals["score_abs_error"] / score_rows,
        **score_component_mae,
        "score_rows": float(totals["score_rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Nano PSM.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch, _ = require_torch()
    from torch.utils.data import DataLoader

    config_doc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = NanoPsmConfig(**config_doc["model"])
    device = resolve_device(torch, args.device)
    tokenizer = HashTokenizer(model_config.vocab_size, model_config.max_sequence_length)
    validation = NanoPsmJsonlDataset(load_jsonl(args.validation), tokenizer)
    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    wrapped = build_model(model_config)
    model = wrapped.module.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate_model(model, loader, device)
    print(json.dumps(metrics, indent=2))


def resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_batch(batch, device):
    moved = {"ids": batch["ids"]}
    for key, value in batch.items():
        if key == "ids":
            continue
        moved[key] = value.to(device)
    return moved


if __name__ == "__main__":
    main()
