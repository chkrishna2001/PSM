from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.eval_generation import evaluate_model_rows
from psm_model.gates import gate_report
from psm_model.generate import load_checkpoint_metadata
from psm_model.model import TinyDecoderModel
from psm_model.tokenizer import ByteTokenizer, load_tokenizer


def evaluate_checkpoint(
    checkpoint: Path,
    data: Path,
    *,
    output_format: str | None = None,
    max_new_tokens: int = 1200,
) -> dict[str, object]:
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = load_checkpoint_metadata(checkpoint)
    active_output_format = output_format or str(metadata.get("output_format", "json"))
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = TinyDecoderModel.load_checkpoint(checkpoint)
    report = evaluate_model_rows(
        model,
        tokenizer,
        rows,
        output_format=active_output_format,
        max_new_tokens=max_new_tokens,
    )
    report.update(
        {
            "checkpoint": str(checkpoint),
            "data": str(data),
            "output_format": active_output_format,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "gate": gate_report(report),
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a saved PSM model checkpoint on generated output quality.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag"])
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    args = parser.parse_args()

    report = evaluate_checkpoint(
        args.checkpoint,
        args.data,
        output_format=args.output_format,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
