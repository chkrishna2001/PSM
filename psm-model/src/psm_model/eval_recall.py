from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from psm_model.gates import RECALL_PROBE_THRESHOLDS, gate_report
from psm_model.generate import load_checkpoint_metadata
from psm_model.model import TinyDecoderModel
from psm_model.prompts import render_row_prompt
from psm_model.recall_schema import parse_recall_plan_json, score_recall_plan
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import resolve_device


def evaluate_recall_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> dict[str, Any]:
    import torch

    torch_module = torch
    device_obj = resolve_device(device, torch_module)
    model.to(device_obj)
    reports: list[dict[str, Any]] = []
    parse_valid = 0
    schema_valid = 0
    target_tables_exact = 0
    target_tables_primary = 0
    ranking_hints_total = 0.0
    top_k_exact = 0
    temporal_intent_exact = 0
    generated_tokens = 0
    context_length = int(getattr(getattr(model, "config", None), "context_length", 2048))

    for row in rows:
        input_payload = row["input"]
        prompt = render_row_prompt(input_payload, output_format="json")
        input_ids = torch_module.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device_obj)
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > context_length:
            reports.append(
                {
                    "id": row.get("id"),
                    "skipped": True,
                    "reason": "context_overflow",
                    "prompt_tokens": prompt_tokens,
                    "context_length": context_length,
                }
            )
            continue
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            temperature=0.0,
        )[0].tolist()
        text = tokenizer.decode(output_ids)
        raw = text.split("<|assistant|>\n", 1)[-1].split("<|end|>", 1)[0]
        generated_tokens += len(tokenizer.encode(raw))
        parsed, issues = parse_recall_plan_json(raw)
        parse_ok = parsed is not None and not issues
        schema_ok = parse_ok
        parse_valid += int(parse_ok)
        schema_valid += int(schema_ok)
        scores = score_recall_plan(row["expected"], parsed if parse_ok else None)
        target_tables_exact += int(scores["target_tables_exact"])
        target_tables_primary += int(scores["target_tables_primary"])
        ranking_hints_total += float(scores["ranking_hints_score"])
        top_k_exact += int(scores["top_k_exact"])
        temporal_intent_exact += int(scores["temporal_intent_exact"])
        reports.append(
            {
                "id": row.get("id"),
                "task": row.get("task") or input_payload.get("operation"),
                "raw": raw,
                "parsed": parsed,
                "parse_issues": list(issues),
                "scores": scores,
            }
        )

    evaluated = max(1, len([report for report in reports if not report.get("skipped")]))
    return {
        "rows": len(rows),
        "evaluated_rows": evaluated,
        "parse_valid_rate": parse_valid / evaluated,
        "schema_valid_rate": schema_valid / evaluated,
        "target_tables_exact_rate": target_tables_exact / evaluated,
        "target_tables_primary_rate": target_tables_primary / evaluated,
        "ranking_hints_score": ranking_hints_total / evaluated,
        "top_k_exact_rate": top_k_exact / evaluated,
        "temporal_intent_exact_rate": temporal_intent_exact / evaluated,
        "avg_generated_tokens": generated_tokens / evaluated,
        "reports": reports,
    }


def evaluate_recall_checkpoint(
    checkpoint: Path,
    data: Path,
    *,
    max_new_tokens: int = 256,
    device: str = "cpu",
) -> dict[str, object]:
    import torch

    device_obj = resolve_device(device, torch)
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = load_checkpoint_metadata(checkpoint)
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = TinyDecoderModel.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    report = evaluate_recall_rows(
        model,
        tokenizer,
        rows,
        max_new_tokens=max_new_tokens,
        device=str(device_obj),
    )
    report.update(
        {
            "checkpoint": str(checkpoint),
            "data": str(data),
            "output_format": "json",
            "device": str(device_obj),
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "gate_mode": "recall",
            "gate": gate_report(report, RECALL_PROBE_THRESHOLDS),
        }
    )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Gate a checkpoint on recall/context planning probes.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = evaluate_recall_checkpoint(
        args.checkpoint,
        args.data,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
