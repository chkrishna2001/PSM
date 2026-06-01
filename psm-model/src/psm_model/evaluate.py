from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import validate_storage_decision


def evaluate_probe_file(path: Path) -> dict[str, Any]:
    total = 0
    valid = 0
    failures: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            result = validate_storage_decision(row["expected"])
            if result.ok:
                valid += 1
            else:
                failures.append(
                    {
                        "line": line_number,
                        "id": row.get("id"),
                        "issues": [{"path": issue.path, "message": issue.message} for issue in result.issues],
                    }
                )

    return {
        "total": total,
        "valid": valid,
        "schema_valid_rate": valid / total if total else 0.0,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PSM model probe fixtures.")
    parser.add_argument("path", type=Path, help="JSONL probe file")
    args = parser.parse_args()

    report = evaluate_probe_file(args.path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

