from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.data import load_jsonl_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate canonical PSM model JSONL training rows.")
    parser.add_argument("path", type=Path, help="JSONL file with id/input/expected rows")
    args = parser.parse_args()

    report = load_jsonl_rows(args.path)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

