from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.data.seed import generate_seed_rows, split_rows, write_seed_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic seed rows for PSM model training.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--split-dir", type=Path)
    args = parser.parse_args()

    count = write_seed_jsonl(args.output)
    report = {"output": str(args.output), "rows": count}
    if args.split_dir:
        rows = generate_seed_rows()
        train, validation = split_rows(rows)
        args.split_dir.mkdir(parents=True, exist_ok=True)
        train_path = args.split_dir / "seed.train.jsonl"
        validation_path = args.split_dir / "seed.validation.jsonl"
        train_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in train) + "\n", encoding="utf-8")
        validation_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in validation) + "\n", encoding="utf-8"
        )
        report.update({"train": len(train), "validation": len(validation), "split_dir": str(args.split_dir)})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

