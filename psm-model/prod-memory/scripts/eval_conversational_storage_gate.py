"""Gate eval for the conversational_storage adapter.

eval_hf_grounding.py hardcodes the coding-agent system prompt (storage_inference_messages() /
PROD_STORAGE_USER_PREFIX), which does not match how the conversational adapter was trained
(build_storage_locomo_rows.py's personal-conversation SYSTEM_PROMPT + "Speaker: text" input, not
"Assistant response: ..."). The held-out fixture (build_storage_locomo_gate.py) already stores the
exact system/user text used at training time per case, so this eval just replays those verbatim
instead of reconstructing a prompt.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from psm_model.hf_lora_train import DEFAULT_MODELS
from psm_model.remember_cli import PROD_STORAGE_MAX_NEW_TOKENS, apply_product_boundary
from prod_memory.hf_prompts import apply_chat_prompt

ALLOWED_ACTIONS = {"ignore", "store_episodic", "promote_semantic"}


def run_case(model: Any, tokenizer: Any, case: dict[str, Any], *, max_new_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": case["system"]},
        {"role": "user", "content": case["user"]},
    ]
    prompt = apply_chat_prompt(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    report = apply_product_boundary(raw, output_format="json")
    decision = report.get("parsed")
    if not isinstance(decision, dict):
        decision = {}
    action = decision.get("action")
    parse_valid = bool(decision) and report.get("repair_status") != "unrecoverable"
    return {
        "id": case["id"],
        "expectAction": case.get("expectAction"),
        "action": action,
        "action_match": action == case.get("expectAction"),
        "parse_valid": parse_valid,
        "repair_status": report.get("repair_status"),
        "raw_output": raw[:500],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(DEFAULT_MODELS), default="qwen0.5b")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=PROD_STORAGE_MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise SystemExit(f"Invalid fixtures file: {args.fixtures}")

    resolved = DEFAULT_MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.device == "cuda" and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map="auto" if args.device == "cuda" and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()

    results = [
        run_case(model, tokenizer, case, max_new_tokens=args.max_new_tokens)
        for case in cases
        if isinstance(case, dict)
    ]
    n = len(results)
    action_match = sum(1 for r in results if r["action_match"])
    parse_valid = sum(1 for r in results if r["parse_valid"])
    store_actions = {"store_episodic", "promote_semantic"}
    tp = sum(1 for r in results if r["expectAction"] in store_actions and r["action"] in store_actions)
    fn = sum(1 for r in results if r["expectAction"] in store_actions and r["action"] not in store_actions)
    fp = sum(1 for r in results if r["expectAction"] == "ignore" and r["action"] in store_actions)
    tn = sum(1 for r in results if r["expectAction"] == "ignore" and r["action"] == "ignore")
    aggregate = {
        "n": n,
        "action_match_rate": round(action_match / max(1, n), 4),
        "parse_valid_rate": round(parse_valid / max(1, n), 4),
        "store_recall": f"{tp}/{tp + fn}",
        "ignore_recall": f"{tn}/{tn + fp}",
        "false_store": fp,
        "false_ignore": fn,
    }
    report = {
        "adapter_dir": str(args.adapter_dir.resolve()),
        "model_key": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixtures": str(args.fixtures.resolve()),
        "aggregate": aggregate,
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
