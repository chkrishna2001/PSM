#!/usr/bin/env python3
"""Compare greedy vs logprob binary gate on fixtures (CPU-friendly)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.eval_grounding import DEFAULT_FIXTURES  # noqa: E402
from prod_memory.eval_hf_grounding import open_hf_session  # noqa: E402
from prod_memory.hf_prompts import apply_chat_prompt, storage_inference_messages  # noqa: E402
from psm_model.prompts import BINARY_SYSTEM_INSTRUCTION  # noqa: E402


GATE_USER_PREFIX = "Assistant response:\n"


def _messages(llm_response: str, *, gate_only_user: bool) -> list[dict[str, str]]:
    if gate_only_user:
        return [
            {"role": "system", "content": BINARY_SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"{GATE_USER_PREFIX}{llm_response.strip()}"},
        ]
    return storage_inference_messages(llm_response, output_format="binary")


def _completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Sum log P(token_i | prefix) for completion tokens after prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not comp_ids:
        return float("-inf")
    full = prompt_ids + comp_ids
    input_ids = torch.tensor([full], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids).logits[0]
    total = 0.0
    for i, tok in enumerate(comp_ids):
        pos = len(prompt_ids) + i - 1
        log_probs = F.log_softmax(logits[pos], dim=-1)
        total += float(log_probs[tok].item())
    return total


def _first_token_logprobs(model, tokenizer, prompt: str, labels: tuple[str, ...]) -> dict[str, float]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids).logits[0, -1]
    log_probs = F.log_softmax(logits, dim=-1)
    out: dict[str, float] = {}
    for label in labels:
        tok = tokenizer.encode(label, add_special_tokens=False)
        if len(tok) == 1:
            out[label] = float(log_probs[tok[0]].item())
        else:
            out[label] = _completion_logprob(model, tokenizer, prompt, label)
    return out


def _decide(
    session,
    llm_response: str,
    *,
    gate_only_user: bool,
    margin: float,
) -> dict:
    messages = _messages(llm_response, gate_only_user=gate_only_user)
    prompt = apply_chat_prompt(messages, session.tokenizer)
    greedy = session.generate(llm_response, output_format="binary", max_new_tokens=8)
    if gate_only_user:
        # regenerate with gate-only messages
        prompt = apply_chat_prompt(messages, session.tokenizer)
        inputs = session.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(session.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = session.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=session.tokenizer.pad_token_id,
            )
        new = out_ids[0, inputs["input_ids"].shape[1] :]
        greedy = session.tokenizer.decode(new, skip_special_tokens=True).strip()

    scores = _first_token_logprobs(session.model, session.tokenizer, prompt, ("ignore", "store"))
    pick = "ignore" if scores["ignore"] >= scores["store"] - margin else "store"
    return {
        "greedy": greedy.strip().splitlines()[0].strip().lower() if greedy.strip() else "",
        "logprob_pick": pick,
        "log_ignore": round(scores["ignore"], 4),
        "log_store": round(scores["store"], 4),
        "delta": round(scores["ignore"] - scores["store"], 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-only-user", action="store_true")
    parser.add_argument("--margin", type=float, default=0.0, help="bias toward ignore when delta >= -margin")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cases = json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8")).get("cases", [])
    session = open_hf_session(args.adapter_dir, model_key=args.model, device=args.device)

    greedy_hits = logprob_hits = 0
    rows: list[dict] = []
    for case in cases:
        exp = str(case.get("expectAction") or "store")
        row = _decide(
            session,
            str(case["llmResponse"]),
            gate_only_user=args.gate_only_user,
            margin=args.margin,
        )
        greedy_ok = (row["greedy"] == "store") == (exp == "store")
        log_ok = (row["logprob_pick"] == "store") == (exp == "store")
        greedy_hits += int(greedy_ok)
        logprob_hits += int(log_ok)
        rows.append({"id": case["id"], "expect": exp, **row, "greedy_ok": greedy_ok, "logprob_ok": log_ok})
        print(
            f"{case['id']}: expect={exp} greedy={row['greedy']!r} ({greedy_ok}) "
            f"logprob={row['logprob_pick']} ign={row['log_ignore']} store={row['log_store']} ({log_ok})"
        )

    summary = {
        "adapter": str(args.adapter_dir) if args.adapter_dir else "base",
        "gate_only_user": args.gate_only_user,
        "margin": args.margin,
        "greedy_match": f"{greedy_hits}/10",
        "logprob_match": f"{logprob_hits}/10",
        "cases": rows,
    }
    print(json.dumps({"greedy": summary["greedy_match"], "logprob": summary["logprob_match"]}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if logprob_hits == 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
