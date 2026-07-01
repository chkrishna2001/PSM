"""Long-lived two-pass HF LoRA remember server (gate + minimal_extract)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from psm_model.remember_cli import apply_product_boundary, to_model_input


def _llm_response_from_payload(payload: dict[str, Any]) -> str:
    conversation = payload.get("conversation")
    if isinstance(conversation, list):
        for message in reversed(conversation):
            if isinstance(message, dict) and str(message.get("role")) == "assistant":
                text = str(message.get("content") or "").strip()
                if text:
                    return text
    model_input = to_model_input(payload)
    text = str(model_input.get("conversation") or "").strip()
    if text.startswith("User:"):
        return text.split(":", 1)[1].strip()
    if text.startswith("Assistant:"):
        return text.split(":", 1)[1].strip()
    return text


def _remember_two_pass(
    binary_session,
    extract_session,
    payload: dict[str, Any],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    from prod_memory.eval_classify import binary_predicts_store

    llm_response = _llm_response_from_payload(payload)
    model_input = to_model_input(payload)
    raw_binary = binary_session.generate(llm_response, output_format="binary", max_new_tokens=16)
    if not binary_predicts_store(raw_binary):
        boundary = apply_product_boundary("ignore", output_format="minimal")
        return {
            "raw": raw_binary,
            "binary_output": raw_binary,
            "output_format": "minimal_extract",
            "model_input": model_input,
            **boundary,
        }
    raw_extract = extract_session.generate(
        llm_response,
        output_format="minimal_extract",
        max_new_tokens=max_new_tokens,
    )
    boundary = apply_product_boundary(raw_extract, output_format="minimal")
    return {
        "raw": raw_extract,
        "binary_output": raw_binary,
        "output_format": "minimal_extract",
        "model_input": model_input,
        **boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-adapter", type=Path, required=True)
    parser.add_argument("--extract-adapter", type=Path, required=True)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    from prod_memory.eval_hf_grounding import open_hf_two_pass_sessions

    binary_session, extract_session = open_hf_two_pass_sessions(
        args.binary_adapter,
        args.extract_adapter,
        model_key=args.model,
        device=args.device,
    )

    sys.stdout.write(json.dumps({"ready": True, "mode": "hf_two_pass"}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("op") == "shutdown":
            break
        payload = request.get("payload")
        if not isinstance(payload, dict):
            response: dict[str, Any] = {"error": "payload must be an object"}
        else:
            try:
                response = _remember_two_pass(
                    binary_session,
                    extract_session,
                    payload,
                    max_new_tokens=int(request.get("max_new_tokens", args.max_new_tokens)),
                )
            except Exception as exc:  # noqa: BLE001
                response = {"error": str(exc)}
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
