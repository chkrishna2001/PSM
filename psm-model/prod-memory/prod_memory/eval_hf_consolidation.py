"""Held-out consolidation eval for the HF LoRA consolidation adapter.

New, standalone scoring layer (no prior generic scorer existed for this task, unlike
recall-plan which reused psm_model.recall_schema) -- parses the model's JSON output and
compares against the hand-labeled expected decision.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.prompts import CONSOLIDATION_SYSTEM_INSTRUCTION

from prod_memory.eval_hf_grounding import HfGenerationSession, open_hf_session

DEFAULT_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "holdout-consolidation-locomo-cases.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "consolidation-eval.json"

_VALID_ACTIONS = {"store_episodic", "update_existing", "flag_conflict"}


def _consolidation_prompt(case: dict[str, Any]) -> str:
    input_payload = {
        "operation": "consolidate",
        "new_memory": case["newMemory"],
        "existing_memory": case["existingMemory"],
    }
    return (
        "Decide store_episodic, update_existing, or flag_conflict as JSON only. "
        "First write reasoning, then action, target_memory_id, and merged_content.\n"
        + json.dumps(input_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _parse_consolidation_json(raw: str) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, ("no JSON object found",)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, (f"invalid JSON: {exc.msg}",)
    if not isinstance(parsed, dict):
        return None, ("not a JSON object",)
    issues: list[str] = []
    action = parsed.get("action")
    if action not in _VALID_ACTIONS:
        issues.append(f"unsupported action: {action!r}")
    if "target_memory_id" not in parsed:
        issues.append("missing target_memory_id")
    if "reasoning" not in parsed or not str(parsed.get("reasoning") or "").strip():
        issues.append("missing reasoning")
    if issues:
        return None, tuple(issues)
    return parsed, ()


def _score_case(expected: dict[str, Any], predicted: dict[str, Any] | None) -> dict[str, Any]:
    if predicted is None:
        return {
            "action_match": False,
            "target_memory_id_match": False,
            "merged_content_grounded": False,
        }
    action_match = predicted.get("action") == expected.get("action")
    exp_target = expected.get("target_memory_id")
    pred_target = predicted.get("target_memory_id")
    target_match = (exp_target is None and pred_target in (None, "")) or (
        exp_target is not None and pred_target == exp_target
    )
    grounded = True
    if expected.get("action") == "update_existing":
        merged = str(predicted.get("merged_content") or "")
        grounded = bool(merged) and len(merged) > 10
    return {
        "action_match": action_match,
        "target_memory_id_match": target_match,
        "merged_content_grounded": grounded,
    }


def evaluate_consolidation_cases(
    session: HfGenerationSession,
    cases: list[dict[str, Any]],
    *,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    parse_valid = 0
    action_match = 0
    target_match = 0
    grounded = 0

    for case in cases:
        messages = [
            {"role": "system", "content": CONSOLIDATION_SYSTEM_INSTRUCTION},
            {"role": "user", "content": _consolidation_prompt(case)},
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
        raw = session.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        parsed, issues = _parse_consolidation_json(raw)
        parse_ok = parsed is not None
        parse_valid += int(parse_ok)
        scores = _score_case(case["expected"], parsed)
        action_match += int(scores["action_match"])
        target_match += int(scores["target_memory_id_match"])
        grounded += int(scores["merged_content_grounded"])
        reports.append({
            "id": case["id"],
            "expected": case["expected"],
            "raw": raw,
            "parsed": parsed,
            "parse_issues": list(issues),
            "scores": scores,
        })

    n = max(1, len(cases))
    return {
        "cases": len(cases),
        "parse_valid_rate": parse_valid / n,
        "action_match_rate": action_match / n,
        "target_memory_id_match_rate": target_match / n,
        "merged_content_grounded_rate": grounded / n,
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
    args = parser.parse_args()

    payload = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = payload["cases"]

    session = open_hf_session(args.adapter_dir, model_key=args.model, device=args.device)
    report = evaluate_consolidation_cases(session, cases, max_new_tokens=args.max_new_tokens)
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
