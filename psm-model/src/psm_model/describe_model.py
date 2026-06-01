from __future__ import annotations

import argparse
import json

from psm_model.configs import describe_preset
from psm_model.tokenizer import ByteTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe PSM model preset parameter estimates.")
    parser.add_argument("--preset", choices=["debug", "10m", "25m", "50m"], default="50m")
    parser.add_argument("--vocab-size", type=int, default=ByteTokenizer().vocab_size)
    parser.add_argument("--context-length", type=int)
    args = parser.parse_args()

    print(json.dumps(describe_preset(args.preset, vocab_size=args.vocab_size, context_length=args.context_length), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

