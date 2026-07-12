"""Held-out recall-plan eval for the HF LoRA retrieval-plan adapter.

Reuses the generic, model-agnostic scoring layer (parse_recall_plan_json/score_recall_plan
from psm_model.recall_schema -- these predate and are independent of the obsolete
TinyDecoderModel eval stack in psm_model.eval_recall) against the current Qwen HF LoRA
generation session (eval_hf_grounding.open_hf_session).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.recall_schema import parse_recall_plan_json, score_recall_plan
from psm_model.prompts import RECALL_SYSTEM_INSTRUCTION

from prod_memory.eval_hf_grounding import HfGenerationSession, open_hf_session

DEFAULT_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "holdout-recall-locomo-cases.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "recall-plan-eval.json"


def _recall_prompt(case: dict[str, Any]) -> str:
    input_payload = {
        "operation": "recall_plan",
        "available_tables": case["availableTables"],
        "requested_top_k": case["requestedTopK"],
        "question": case["question"],
    }
    return (
        "Create a recall plan as JSON only with intent, target_tables, filters, ranking_hints, "
        "temporal_intent, and top_k. PSM owns memory planning; do not answer the user.\n"
        + json.dumps(input_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _generate_recall_plan(session: HfGenerationSession, case: dict[str, Any], *, max_new_tokens: int) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": RECALL_SYSTEM_INSTRUCTION},
        {"role": "user", "content": _recall_prompt(case)},
    ]
    from prod_memory.hf_prompts import apply_chat_prompt

    prompt = apply_chat_prompt(messages, session.tokenizer)
    inputs = session.tokenizer(prompt, return_tensors="pt")
    device = session._input_device()
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    import torch

    with torch.inference_mode():
        output_ids = session.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=session.tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, prompt_len:]
    text = session.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, int(new_tokens.shape[0])


def evaluate_recall_cases(
    session: HfGenerationSession,
    cases: list[dict[str, Any]],
    *,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    parse_valid = 0
    target_tables_exact = 0
    target_tables_primary = 0
    ranking_hints_total = 0.0
    top_k_exact = 0
    temporal_intent_exact = 0

    for case in cases:
        raw, token_count = _generate_recall_plan(session, case, max_new_tokens=max_new_tokens)
        parsed, issues = parse_recall_plan_json(raw)
        parse_ok = parsed is not None and not issues
        parse_valid += int(parse_ok)
        scores = score_recall_plan(case["expected"], parsed if parse_ok else None)
        target_tables_exact += int(scores["target_tables_exact"])
        target_tables_primary += int(scores["target_tables_primary"])
        ranking_hints_total += float(scores["ranking_hints_score"])
        top_k_exact += int(scores["top_k_exact"])
        temporal_intent_exact += int(scores["temporal_intent_exact"])
        reports.append({
            "id": case["id"],
            "question": case["question"],
            "raw": raw,
            "generated_token_count": token_count,
            "parsed": parsed,
            "parse_issues": list(issues),
            "scores": scores,
        })

    n = max(1, len(cases))
    return {
        "cases": len(cases),
        "parse_valid_rate": parse_valid / n,
        "target_tables_exact_rate": target_tables_exact / n,
        "target_tables_primary_rate": target_tables_primary / n,
        "ranking_hints_score": ranking_hints_total / n,
        "top_k_exact_rate": top_k_exact / n,
        "temporal_intent_exact_rate": temporal_intent_exact / n,
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="")
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases (0 = all).")
    args = parser.parse_args()

    payload = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.limit > 0:
        cases = cases[: args.limit]

    session = open_hf_session(args.adapter_dir, model_key=args.model, device=args.device)
    report = evaluate_recall_cases(session, cases, max_new_tokens=args.max_new_tokens)
    report.update({
        "checkpoint": args.checkpoint_label or str(args.adapter_dir),
        "adapter_dir": str(args.adapter_dir),
        "fixtures": str(args.fixtures),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "reports"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
