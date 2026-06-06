from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from psm_model.gates import gate_report
from psm_model.lean_format import parse_at_tag_decision, parse_tagged_decision
from psm_model.model import TinyDecoderConfig
from psm_model.prompts import render_storage_prompt
from psm_model.schema import parse_and_validate_storage_decision, validate_storage_decision
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import ACTION_ORDER, ACTION_TO_ID, load_training_texts, overfit_texts, resolve_device


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Generation evaluation requires PyTorch. Install torch to run psm_model.eval_generation.") from exc
    return torch


def evaluate_generation(
    train_path: Path,
    *,
    eval_path: Path | None = None,
    output_format: str = "tagged",
    steps: int = 300,
    max_new_tokens: int = 1200,
    tokenizer_path: Path | None = None,
    device: str = "cpu",
    force_action_head: bool = False,
) -> dict[str, Any]:
    eval_source = eval_path or train_path
    rows = [json.loads(line) for line in eval_source.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else ByteTokenizer()
    config = TinyDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=2048,
        n_layer=2,
        n_head=4,
        n_embd=128,
    )
    model, losses = overfit_texts(
        load_training_texts(train_path, output_format=output_format),
        config=config,
        tokenizer=tokenizer,
        steps=steps,
        learning_rate=1e-3,
        device=device,
    )

    report = evaluate_model_rows(
        model,
        tokenizer,
        rows,
        output_format=output_format,
        max_new_tokens=max_new_tokens,
        device=device,
        force_action_head=force_action_head,
    )
    report.update(
        {
            "output_format": output_format,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "train_path": str(train_path),
            "eval_path": str(eval_source),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
        }
    )
    return report


def evaluate_model_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    output_format: str = "tagged",
    max_new_tokens: int = 1200,
    device: str = "cpu",
    force_action_head: bool = False,
) -> dict[str, Any]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    model.to(device_obj)
    reports: list[dict[str, Any]] = []
    parse_valid = 0
    schema_valid = 0
    action_correct = 0
    memory_type_correct = 0
    memory_content_exact = 0
    fact_count_correct = 0
    facts_exact = 0
    action_head_correct = 0
    action_head_available = 0
    generated_tokens = 0

    for row in rows:
        prompt = render_storage_prompt(row["input"], output_format=output_format)
        input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device_obj)
        action_head_prediction = None
        action_head_confidence = None
        action_head_prediction, action_head_confidence = predict_action_head(model, input_ids)
        if action_head_prediction is not None:
            action_head_available += 1
            action_head_correct += int(action_head_prediction == row["expected"]["action"])
            if force_action_head:
                forced_prefix = render_action_prefix(action_head_prediction, output_format=output_format)
                forced_ids = torch.tensor([tokenizer.encode(forced_prefix)], dtype=torch.long, device=device_obj)
                input_ids = torch.cat([input_ids, forced_ids], dim=1)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            temperature=0.0,
        )[0].tolist()
        text = tokenizer.decode(output_ids)
        raw = text.split("<|assistant|>\n", 1)[-1].split("<|end|>", 1)[0]
        generated_tokens += len(tokenizer.encode(raw))
        parsed, parse_issues = _parse_output(raw, output_format)
        parse_ok = parsed is not None and not parse_issues
        schema_result = validate_storage_decision(parsed) if parsed is not None else None
        schema_ok = bool(schema_result and schema_result.ok)
        parse_valid += int(parse_ok)
        schema_valid += int(schema_ok)

        expected = row["expected"]
        expected_memory = expected.get("memory")
        expected_type = expected_memory.get("type") if expected_memory else None
        predicted_type = None
        if parsed and parsed.get("memory"):
            predicted_type = parsed["memory"].get("type")
        if schema_ok:
            action_correct += int(parsed["action"] == expected["action"])
            memory_type_correct += int(predicted_type == expected_type)
            memory_content_exact += int(_memory_content(parsed) == _memory_content(expected))
            fact_count_correct += int(len(parsed.get("facts", [])) == len(expected.get("facts", [])))
            facts_exact += int(_fact_signature(parsed) == _fact_signature(expected))

        reports.append(
            {
                "id": row["id"],
                "parse_valid": parse_ok,
                "schema_valid": schema_ok,
                "expected_action": expected["action"],
                "predicted_action": parsed.get("action") if parsed else None,
                "action_head_prediction": action_head_prediction,
                "action_head_confidence": action_head_confidence,
                "expected_memory_type": expected_type,
                "predicted_memory_type": predicted_type,
                "memory_content_exact": schema_ok and _memory_content(parsed) == _memory_content(expected),
                "expected_fact_count": len(expected.get("facts", [])),
                "predicted_fact_count": len(parsed.get("facts", [])) if parsed else None,
                "facts_exact": schema_ok and _fact_signature(parsed) == _fact_signature(expected),
                "generated_tokens": len(tokenizer.encode(raw)),
                "issues": [
                    {"path": issue.path, "message": issue.message}
                    for issue in (parse_issues if parse_issues else (schema_result.issues if schema_result else ()))
                ],
            }
        )

    total = len(rows)
    return {
        "examples": total,
        "parse_valid_rate": parse_valid / total if total else 0.0,
        "schema_valid_rate": schema_valid / total if total else 0.0,
        "action_accuracy": action_correct / total if total else 0.0,
        "action_head_accuracy": action_head_correct / action_head_available if action_head_available else None,
        "action_head_examples": action_head_available,
        "memory_type_accuracy": memory_type_correct / total if total else 0.0,
        "memory_content_exact_rate": memory_content_exact / total if total else 0.0,
        "fact_count_accuracy": fact_count_correct / total if total else 0.0,
        "facts_exact_rate": facts_exact / total if total else 0.0,
        "avg_generated_tokens": generated_tokens / total if total else 0.0,
        "force_action_head": force_action_head,
        "reports": reports,
    }


def predict_action_head(model: Any, input_ids: Any) -> tuple[str | None, float | None]:
    if not hasattr(model, "action_head"):
        return None, None
    torch = _torch()
    with torch.no_grad():
        action_positions = torch.tensor([input_ids.size(1) - 1], dtype=torch.long, device=input_ids.device)
        action_result = model(input_ids, action_positions=action_positions)
        action_logits = action_result.get("action_logits")
        if action_logits is None:
            return None, None
        probs = torch.nn.functional.softmax(action_logits, dim=-1)
        predicted_id = int(torch.argmax(probs, dim=-1).detach().cpu()[0])
        return ACTION_ORDER[predicted_id], float(probs[0, predicted_id].detach().cpu())


def render_action_prefix(action: str, *, output_format: str) -> str:
    if output_format == "tagged":
        return f"A:{action}\n"
    if output_format == "at_tag":
        return f"@a {action}\n"
    if output_format == "json":
        return f'{{"action":"{action}",'
    raise ValueError(f"unsupported output format: {output_format}")


def _parse_output(raw: str, output_format: str) -> tuple[dict[str, Any] | None, tuple[Any, ...]]:
    if output_format == "json":
        result = parse_and_validate_storage_decision(raw)
        return (asdict(result.decision) if result.decision else None), result.issues
    if output_format == "tagged":
        return parse_tagged_decision(raw)
    if output_format == "at_tag":
        return parse_at_tag_decision(raw)
    raise ValueError(f"unsupported output format: {output_format}")


def _memory_content(decision: dict[str, Any] | None) -> str | None:
    if not decision or not decision.get("memory"):
        return None
    return decision["memory"].get("content")


def _fact_signature(decision: dict[str, Any] | None) -> list[tuple[str, str, str, str]]:
    if not decision:
        return []
    return [
        (
            str(fact.get("subject")),
            str(fact.get("predicate")),
            str(fact.get("value")),
            str(fact.get("evidence_text")),
        )
        for fact in decision.get("facts", [])
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate tiny-model generation over a canonical probe JSONL.")
    parser.add_argument("train", type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag"], default="tagged")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", default="cpu", help="Evaluation device: cpu, cuda, or auto.")
    parser.add_argument("--force-action-head", action="store_true", help="Use the auxiliary action head to force the generated action prefix.")
    args = parser.parse_args()

    report = evaluate_generation(
        args.train,
        eval_path=args.eval,
        output_format=args.output_format,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        tokenizer_path=args.tokenizer,
        device=args.device,
        force_action_head=args.force_action_head,
    )
    report["gate"] = gate_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
