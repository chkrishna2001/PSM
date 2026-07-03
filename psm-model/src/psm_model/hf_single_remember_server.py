"""Long-lived single-adapter HF LoRA remember server (full StorageDecision JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from psm_model.remember_cli import PROD_STORAGE_MAX_NEW_TOKENS, apply_product_boundary

from prod_memory.eval_hf_grounding import open_hf_session


def _llm_response_from_payload(payload: dict[str, Any]) -> str:
    from psm_model.remember_cli import to_model_input

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


def _remember_single(
    session,
    payload: dict[str, Any],
    *,
    output_format: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    from psm_model.remember_cli import to_model_input

    llm_response = _llm_response_from_payload(payload)
    model_input = to_model_input(payload)
    raw = session.generate(llm_response, output_format=output_format, max_new_tokens=max_new_tokens)
    boundary = apply_product_boundary(raw, output_format=output_format)
    return {
        "raw": raw,
        "output_format": output_format,
        "model_input": model_input,
        **boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--output-format", default="json", choices=["json", "tagged", "minimal", "minimal_extract"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=PROD_STORAGE_MAX_NEW_TOKENS)
    args = parser.parse_args()

    session = open_hf_session(args.adapter, model_key=args.model, device=args.device)

    sys.stdout.write(json.dumps({"ready": True, "mode": "hf_single"}) + "\n")
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
                response = _remember_single(
                    session,
                    payload,
                    output_format=args.output_format,
                    max_new_tokens=int(request.get("max_new_tokens", args.max_new_tokens)),
                )
            except Exception as exc:  # noqa: BLE001
                response = {"error": str(exc)}
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
