from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psm_model.eval_generation import _parse_output, predict_action_head, render_action_prefix
from psm_model.model import TinyDecoderModel
from psm_model.prompts import render_storage_prompt
from psm_model.schema import validate_storage_decision
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import resolve_device


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Generation requires PyTorch. Install torch to run psm_model.generate.") from exc
    return torch


def generate_storage_json(
    checkpoint: Path,
    input_payload: dict[str, object],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    output_format: str | None = None,
    device: str = "cpu",
    force_action_head: bool = False,
) -> str:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    metadata = load_checkpoint_metadata(checkpoint)
    output_format = output_format or str(metadata.get("output_format", "json"))
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = TinyDecoderModel.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    prompt = render_storage_prompt(input_payload, output_format=output_format)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device_obj)
    if force_action_head:
        action, _ = predict_action_head(model, input_ids)
        if action is not None:
            forced_ids = torch.tensor([tokenizer.encode(render_action_prefix(action, output_format=output_format))], dtype=torch.long, device=device_obj)
            input_ids = torch.cat([input_ids, forced_ids], dim=1)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
    )[0].tolist()
    text = tokenizer.decode(output_ids)
    return text.split("<|assistant|>\n", 1)[-1].split("<|end|>", 1)[0]


def load_checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_path = checkpoint.with_suffix(".meta.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate a PSM storage decision with a checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", help="JSON object payload")
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", help="Generation device: cpu, cuda, or auto.")
    parser.add_argument("--force-action-head", action="store_true", help="Use the auxiliary action head to force the generated action prefix.")
    args = parser.parse_args()

    payload = json.loads(args.input)
    raw = generate_storage_json(
        args.checkpoint,
        payload,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_format=args.output_format,
        device=args.device,
        force_action_head=args.force_action_head,
    )
    output_format = args.output_format or str(load_checkpoint_metadata(args.checkpoint).get("output_format", "json"))
    parsed, parse_issues = _parse_output(raw, output_format)
    validation = validate_storage_decision(parsed) if parsed is not None else None
    issues = parse_issues if parse_issues else (validation.issues if validation else ())
    print(
        json.dumps(
            {
                "raw": raw,
                "parsed": parsed,
                "output_format": output_format,
                "valid": bool(validation and validation.ok and not parse_issues),
                "issues": [{"path": issue.path, "message": issue.message} for issue in issues],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation and validation.ok and not parse_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
