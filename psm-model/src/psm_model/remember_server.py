"""Long-lived remember server: load 50M checkpoint once, serve many remember() calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from psm_model.generate import GenerationSession, open_generation_session


def _remember_with_session(
    session: GenerationSession,
    payload: dict,
    *,
    max_new_tokens: int,
) -> dict:
    if payload.get("operation") == "repair_remember_json":
        from psm_model.remember_cli import remember_from_repair_payload

        return remember_from_repair_payload(payload, output_format=session.output_format)
    conversation = payload.get("conversation")
    if isinstance(conversation, list) or payload.get("operation") == "remember_llm_response":
        from psm_model.remember_cli import to_model_input

        model_input = to_model_input(payload)
    elif isinstance(conversation, str):
        model_input = payload
    else:
        from psm_model.remember_cli import to_model_input

        model_input = to_model_input(payload)
    from psm_model.generate import generate_storage_json
    from psm_model.remember_cli import apply_product_boundary

    raw = generate_storage_json(
        session.checkpoint,
        model_input,
        max_new_tokens=max_new_tokens,
        output_format=session.output_format,
        device=str(session.device_obj),
        session=session,
    )
    boundary = apply_product_boundary(raw, output_format=session.output_format)
    return {
        "raw": raw,
        "output_format": session.output_format,
        "model_input": model_input,
        **boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-format", default="tagged", choices=["json", "tagged", "at_tag"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    session = open_generation_session(
        args.checkpoint,
        output_format=args.output_format,
        device=args.device,
    )
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
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
            response = {"error": "payload must be an object"}
        else:
            try:
                response = _remember_with_session(
                    session,
                    payload,
                    max_new_tokens=int(request.get("max_new_tokens", args.max_new_tokens)),
                )
            except Exception as exc:  # noqa: BLE001 — return error to caller
                response = {"error": str(exc)}
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
