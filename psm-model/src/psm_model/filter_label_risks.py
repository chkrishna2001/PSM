from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.label_audit import row_label_issues


SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


def filter_label_risks(input_path: Path, output_path: Path, *, drop_severity: str = "high") -> dict[str, Any]:
    if drop_severity not in SEVERITY_ORDER:
        raise ValueError(f"unknown severity: {drop_severity}")

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    threshold = SEVERITY_ORDER[drop_severity]

    for row in rows:
        expected = row.get("expected", {})
        action = str(expected.get("action", "missing"))
        memory = expected.get("memory") if isinstance(expected.get("memory"), dict) else None
        memory_type = str(memory.get("type", "none")) if memory else "none"
        issues = row_label_issues(row, action=action, memory_type=memory_type)
        blocking_issues = [issue for issue in issues if SEVERITY_ORDER[issue.severity] >= threshold]
        if blocking_issues:
            dropped.append(
                {
                    "id": row.get("id"),
                    "action": action,
                    "memory_type": memory_type,
                    "issues": [issue.to_json() for issue in blocking_issues],
                }
            )
        else:
            kept.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept),
        encoding="utf-8",
    )
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "drop_severity": drop_severity,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(dropped),
        "kept_action_counts": dict(sorted(Counter(row["expected"]["action"] for row in kept).items())),
        "dropped_action_counts": dict(sorted(Counter(row["action"] for row in dropped).items())),
        "dropped_examples": dropped[:25],
    }
    output_path.with_suffix(output_path.suffix + ".audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a JSONL dataset with risky labels removed.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--drop-severity", choices=sorted(SEVERITY_ORDER), default="high")
    args = parser.parse_args()

    print(json.dumps(filter_label_risks(args.input, args.output, drop_severity=args.drop_severity), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
