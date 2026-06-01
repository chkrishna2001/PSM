from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.tokenizer import ByteTokenizer, DslTokenizer, train_bpe_tokenizer, train_pattern_tokenizer
from psm_model.train import load_training_texts


def train_tokenizer_file(data: Path, output: Path, *, vocab_size: int, output_format: str, kind: str = "pattern") -> dict[str, object]:
    texts = load_training_texts(data, output_format=output_format)
    tokenizer_training_texts = _split_training_texts(texts)
    byte = ByteTokenizer()
    if kind == "bpe":
        tokenizer = train_bpe_tokenizer(tokenizer_training_texts, vocab_size=vocab_size)
    elif kind == "pattern":
        tokenizer = train_pattern_tokenizer(tokenizer_training_texts, vocab_size=vocab_size)
    elif kind == "dsl":
        tokenizer = DslTokenizer()
    else:
        raise ValueError(f"unsupported tokenizer kind: {kind}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output)
    byte_tokens = sum(len(byte.encode(text, add_bos=True, add_eos=True)) for text in texts)
    trained_tokens = sum(len(tokenizer.encode(text, add_bos=True, add_eos=True)) for text in texts)
    return {
        "kind": kind,
        "output": str(output),
        "examples": len(texts),
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "learned_units": tokenizer.vocab_size - byte.vocab_size,
        "byte_tokens": byte_tokens,
        "trained_tokens": trained_tokens,
        "token_savings": 1.0 - (trained_tokens / byte_tokens) if byte_tokens else 0.0,
    }


def _split_training_texts(texts: list[str]) -> list[str]:
    pieces: list[str] = []
    marker = "<|assistant|>\n"
    for text in texts:
        if marker in text:
            prompt, output = text.split(marker, 1)
            pieces.extend([prompt + marker, output])
        else:
            pieces.append(text)
    return pieces


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a dependency-free byte-level BPE tokenizer.")
    parser.add_argument("data", type=Path, help="Canonical JSONL rows")
    parser.add_argument("output", type=Path, help="Tokenizer JSON output")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag"], default="tagged")
    parser.add_argument("--kind", choices=["dsl", "pattern", "bpe"], default="dsl")
    args = parser.parse_args()
    print(
        json.dumps(
            train_tokenizer_file(args.data, args.output, vocab_size=args.vocab_size, output_format=args.output_format, kind=args.kind),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
