from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.generate import load_checkpoint_metadata
from psm_model.model import TinyDecoderModel
from psm_model.prompts import render_storage_prompt
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import ACTION_ORDER, resolve_device


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Action diagnostics require PyTorch. Install torch to run psm_model.action_diagnostics.") from exc
    return torch


def evaluate_action_prefixes(
    checkpoint: Path,
    data: Path,
    *,
    output_format: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    metadata = load_checkpoint_metadata(checkpoint)
    active_output_format = output_format or str(metadata.get("output_format", "tagged"))
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = TinyDecoderModel.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    model.eval()
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]

    reports: list[dict[str, Any]] = []
    correct = 0
    predicted_counts = {action: 0 for action in ACTION_ORDER}
    expected_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    gold_ranks: list[int] = []
    with torch.no_grad():
        for row in rows:
            scores = score_actions(model, tokenizer, _row_input(row), output_format=active_output_format, device=device_obj)
            ranked = sorted(scores.items(), key=lambda item: item[1])
            predicted = ranked[0][0]
            expected = _row_expected_action(row)
            expected_counts[expected] += 1
            predicted_counts[predicted] += 1
            correct += int(predicted == expected)
            correct_counts[expected] += int(predicted == expected)
            rank = [action for action, _ in ranked].index(expected) + 1
            gold_ranks.append(rank)
            reports.append(
                {
                    "id": _row_id(row),
                    "expected_action": expected,
                    "predicted_action": predicted,
                    "gold_rank": rank,
                    "scores": {action: round(loss, 6) for action, loss in ranked},
                }
            )

    total = len(rows)
    per_action_accuracy = {
        action: correct_counts[action] / expected_counts[action]
        for action in sorted(expected_counts)
        if expected_counts[action]
    }
    return {
        "checkpoint": str(checkpoint),
        "data": str(data),
        "device": str(device_obj),
        "output_format": active_output_format,
        "examples": total,
        "action_prefix_accuracy": correct / total if total else 0.0,
        "macro_action_prefix_accuracy": sum(per_action_accuracy.values()) / len(per_action_accuracy) if per_action_accuracy else 0.0,
        "mean_gold_rank": sum(gold_ranks) / len(gold_ranks) if gold_ranks else None,
        "expected_action_counts": dict(sorted(expected_counts.items())),
        "per_action_accuracy": per_action_accuracy,
        "predicted_action_counts": predicted_counts,
        "collapse_fraction": collapse_fraction(predicted_counts),
        "reports": reports,
    }


def collapse_fraction(predicted_action_counts: dict[str, int]) -> float:
    total = sum(predicted_action_counts.values())
    if not total:
        return 0.0
    return max(predicted_action_counts.values()) / total


def evaluate_action_head(
    checkpoint: Path,
    data: Path,
    *,
    output_format: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    metadata = load_checkpoint_metadata(checkpoint)
    active_output_format = output_format or str(metadata.get("output_format", "tagged"))
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = TinyDecoderModel.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    model.eval()
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]

    reports: list[dict[str, Any]] = []
    correct = 0
    predicted_counts = {action: 0 for action in ACTION_ORDER}
    expected_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    with torch.no_grad():
        for row in rows:
            scores = score_action_head(model, tokenizer, _row_input(row), output_format=active_output_format, device=device_obj)
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            predicted = ranked[0][0]
            expected = _row_expected_action(row)
            expected_counts[expected] += 1
            predicted_counts[predicted] += 1
            correct += int(predicted == expected)
            correct_counts[expected] += int(predicted == expected)
            reports.append(
                {
                    "id": _row_id(row),
                    "expected_action": expected,
                    "predicted_action": predicted,
                    "scores": {action: round(score, 6) for action, score in ranked},
                }
            )

    total = len(rows)
    per_action_accuracy = {
        action: correct_counts[action] / expected_counts[action]
        for action in sorted(expected_counts)
        if expected_counts[action]
    }
    return {
        "checkpoint": str(checkpoint),
        "data": str(data),
        "device": str(device_obj),
        "output_format": active_output_format,
        "examples": total,
        "action_head_accuracy": correct / total if total else 0.0,
        "macro_action_head_accuracy": sum(per_action_accuracy.values()) / len(per_action_accuracy) if per_action_accuracy else 0.0,
        "expected_action_counts": dict(sorted(expected_counts.items())),
        "per_action_accuracy": per_action_accuracy,
        "predicted_action_counts": predicted_counts,
        "reports": reports,
    }


def score_actions(model: Any, tokenizer: Any, input_payload: dict[str, Any], *, output_format: str, device: Any) -> dict[str, float]:
    torch = _torch()
    prompt = render_storage_prompt(input_payload, output_format=output_format)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    reserve = max(len(tokenizer.encode(_action_prefix(action, output_format=output_format))) for action in ACTION_ORDER) + 2
    max_prompt = model.config.context_length - reserve
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
    scores: dict[str, float] = {}
    for action in ACTION_ORDER:
        action_ids = tokenizer.encode(_action_prefix(action, output_format=output_format))
        sequence = prompt_ids + action_ids
        input_ids = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
        labels = torch.full_like(input_ids, -100)
        labels[0, len(prompt_ids) - 1 :] = torch.tensor(action_ids, dtype=torch.long, device=device)
        result = model(input_ids, labels=labels)
        scores[action] = float(result["lm_loss"].detach().cpu())
    return scores


def score_action_head(model: Any, tokenizer: Any, input_payload: dict[str, Any], *, output_format: str, device: Any) -> dict[str, float]:
    torch = _torch()
    prompt = render_storage_prompt(input_payload, output_format=output_format)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    if len(prompt_ids) > model.config.context_length:
        prompt_ids = prompt_ids[-model.config.context_length :]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    action_positions = torch.tensor([len(prompt_ids) - 1], dtype=torch.long, device=device)
    result = model(input_ids, action_positions=action_positions)
    probabilities = torch.nn.functional.softmax(result["action_logits"][0], dim=-1)
    return {action: float(probabilities[index].detach().cpu()) for index, action in enumerate(ACTION_ORDER)}


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("case") or row.get("source_id") or "row")


def _row_input(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("input"), dict):
        return row["input"]
    if isinstance(row.get("text"), str):
        return {"conversation": row["text"], "operation": "remember"}
    raise KeyError(f"{_row_id(row)} has no input payload")


def _row_expected_action(row: dict[str, Any]) -> str:
    expected = row.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("action"), str):
        return expected["action"]
    if isinstance(row.get("expected_action"), str):
        return row["expected_action"]
    raise KeyError(f"{_row_id(row)} has no expected action")


def _action_prefix(action: str, *, output_format: str) -> str:
    if output_format in {"tagged", "action"}:
        return f"A:{action}\n"
    if output_format == "at_tag":
        return f"@a {action}\n"
    if output_format == "json":
        return f'{{"action":"{action}",'
    raise ValueError(f"unsupported output format: {output_format}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score legal action prefixes under a saved PSM checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"])
    parser.add_argument("--device", default="cpu", help="Evaluation device: cpu, cuda, or auto.")
    parser.add_argument("--mode", choices=["prefix", "head"], default="prefix")
    args = parser.parse_args()

    evaluator = evaluate_action_head if args.mode == "head" else evaluate_action_prefixes
    print(
        json.dumps(
            evaluator(
                args.checkpoint,
                args.data,
                output_format=args.output_format,
                device=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
