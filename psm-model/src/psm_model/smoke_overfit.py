from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.model import TinyDecoderConfig
from psm_model.prompts import render_storage_prompt
from psm_model.lean_format import parse_at_tag_decision, parse_tagged_decision
from psm_model.schema import parse_and_validate_storage_decision
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import load_training_texts, overfit_texts


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Overfit smoke requires PyTorch. Install torch to run psm_model.smoke_overfit.") from exc
    return torch


def run_smoke(
    probes_path: Path,
    *,
    probe_id: str,
    steps: int,
    max_new_tokens: int,
    output_format: str = "tagged",
    tokenizer_path: Path | None = None,
) -> dict[str, object]:
    torch = _torch()
    rows = [json.loads(line) for line in probes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        row = next(item for item in rows if item["id"] == probe_id)
    except StopIteration as exc:
        raise ValueError(f"probe id not found: {probe_id}") from exc

    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path else ByteTokenizer()
    config = TinyDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=2048,
        n_layer=2,
        n_head=4,
        n_embd=128,
    )
    texts = load_training_texts(probes_path, output_format=output_format)
    model, losses = overfit_texts(texts, config=config, tokenizer=tokenizer, steps=steps, learning_rate=1e-3)
    prompt = render_storage_prompt(row["input"], output_format=output_format)
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=0.0,
    )[0].tolist()
    text = tokenizer.decode(output_ids)
    raw = text.split("<|assistant|>\n", 1)[-1].split("<|end|>", 1)[0]
    if output_format == "json":
        validation = parse_and_validate_storage_decision(raw)
    elif output_format == "tagged":
        _, issues = parse_tagged_decision(raw)
        validation = _ValidationAdapter(not issues, issues)
    elif output_format == "at_tag":
        _, issues = parse_at_tag_decision(raw)
        validation = _ValidationAdapter(not issues, issues)
    else:
        raise ValueError(f"unsupported output format: {output_format}")
    return {
        "probe_id": probe_id,
        "output_format": output_format,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "examples": len(texts),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "generated": raw,
        "valid": validation.ok,
        "issues": [{"path": issue.path, "message": issue.message} for issue in validation.issues],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Overfit the tiny model on probes and validate one generated output.")
    parser.add_argument("probes", type=Path, help="Canonical probe JSONL")
    parser.add_argument("--probe-id", default="ignore_noise")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag"], default="tagged")
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()

    report = run_smoke(
        args.probes,
        probe_id=args.probe_id,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        output_format=args.output_format,
        tokenizer_path=args.tokenizer,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


class _ValidationAdapter:
    def __init__(self, ok: bool, issues: tuple[object, ...]):
        self.ok = ok
        self.issues = issues
