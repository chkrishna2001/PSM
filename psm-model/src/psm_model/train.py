from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from psm_model.configs import config_from_preset
from psm_model.data import validate_training_row
from psm_model.model import TinyDecoderConfig, TinyDecoderModel
from psm_model.prompts import render_training_text
from psm_model.tokenizer import ByteTokenizer, load_tokenizer


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Training requires PyTorch. Install torch to run psm_model.train.") from exc
    return torch


def load_training_texts(path: Path, *, output_format: str = "tagged") -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            row, issues = validate_training_row(raw)
            if issues or row is None:
                formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
                raise ValueError(f"{path}:{line_number}: invalid row: {formatted}")
            texts.append(render_training_text(raw["input"], raw["expected"], output_format=output_format))
    return texts


def build_lm_batch(texts: list[str], tokenizer: ByteTokenizer, *, context_length: int) -> tuple[Any, Any]:
    torch = _torch()
    if not texts:
        raise ValueError("at least one training text is required")

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for text in texts:
        ids, label_mask = _encode_training_text(tokenizer, text)
        if len(ids) < 2:
            raise ValueError("training text is too short")
        if len(ids) > context_length + 1:
            raise ValueError(f"training text length {len(ids)} exceeds context length plus label token {context_length + 1}")
        input_ids = ids[:-1]
        labels = [token_id if mask else -100 for token_id, mask in zip(ids[1:], label_mask[1:])]
        pad = context_length - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [tokenizer.pad_id] * pad
            labels = labels + [-100] * pad
        input_rows.append(input_ids)
        label_rows.append(labels)

    return torch.tensor(input_rows, dtype=torch.long), torch.tensor(label_rows, dtype=torch.long)


def _encode_training_text(tokenizer: Any, text: str) -> tuple[list[int], list[bool]]:
    marker = "<|assistant|>\n"
    if marker in text and hasattr(tokenizer, "encode_pieces"):
        prompt, output = text.split(marker, 1)
        prompt_ids = tokenizer.encode(prompt + marker, add_bos=True)
        output_ids = tokenizer.encode(output, add_eos=True)
        return prompt_ids + output_ids, [False] * len(prompt_ids) + [True] * len(output_ids)
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return ids, [True] * len(ids)


def overfit_texts(
    texts: list[str],
    *,
    config: TinyDecoderConfig,
    tokenizer: Any | None = None,
    steps: int = 100,
    learning_rate: float = 3e-4,
    seed: int = 7,
) -> tuple[TinyDecoderModel, list[float]]:
    torch = _torch()
    torch.manual_seed(seed)
    tokenizer = tokenizer or ByteTokenizer()
    model = TinyDecoderModel(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input_ids, labels = build_lm_batch(texts, tokenizer, context_length=config.context_length)
    losses: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids, labels=labels)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return model, losses


def train_texts(
    texts: list[str],
    *,
    config: TinyDecoderConfig,
    tokenizer: Any | None = None,
    steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    seed: int = 7,
) -> tuple[TinyDecoderModel, list[float]]:
    torch = _torch()
    if not texts:
        raise ValueError("at least one training text is required")
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = tokenizer or ByteTokenizer()
    model = TinyDecoderModel(config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses: list[float] = []

    for _ in range(steps):
        batch = [texts[random.randrange(len(texts))] for _ in range(batch_size)]
        input_ids, labels = build_lm_batch(batch, tokenizer, context_length=config.context_length)
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids, labels=labels)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return model, losses


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tiny debug PSM decoder on canonical JSONL rows.")
    parser.add_argument("data", type=Path, help="Canonical JSONL rows")
    parser.add_argument("--out", type=Path, default=Path("psm-model/checkpoints/tiny-debug.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preset", choices=["debug", "10m", "25m", "50m"], default="debug")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag"], default="tagged")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config and print parameter estimate without training")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else ByteTokenizer()
    config = config_from_preset(args.preset, vocab_size=tokenizer.vocab_size, context_length=args.context_length)
    texts = load_training_texts(args.data, output_format=args.output_format)
    report = {
        "config": asdict(config),
        "examples": len(texts),
        "output_format": args.output_format,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "preset": args.preset,
        "parameter_estimate": TinyDecoderModel.parameter_estimate(config),
    }
    if args.dry_run:
        print(json.dumps(report | {"dry_run": True}, indent=2, sort_keys=True))
        return 0

    model, losses = train_texts(
        texts,
        config=config,
        tokenizer=tokenizer,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    model.save_checkpoint(args.out)
    tokenizer.save(args.out.with_suffix(".tokenizer.json"))
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "output_format": args.output_format,
                "preset": args.preset,
                "parameter_estimate": TinyDecoderModel.parameter_estimate(config),
                "tokenizer_vocab_size": tokenizer.vocab_size,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report.update({
        "checkpoint": str(args.out),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
    })
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
