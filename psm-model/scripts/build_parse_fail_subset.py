"""Extract probes that failed parse in a gate4 expanded eval report into a subset JSONL.

Usage:
    python psm-model/scripts/build_parse_fail_subset.py \
        --eval-report psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042000.json \
        --probes psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl \
        --output psm-model/data/direct-behavior-v1/parse-fails-step-042000.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    fail_ids = {r["id"] for r in report["reports"] if not r.get("parse_valid")}

    rows = [json.loads(line) for line in args.probes.read_text(encoding="utf-8").splitlines() if line.strip()]
    subset = [row for row in rows if row["id"] in fail_ids]
    missing = fail_ids - {row["id"] for row in subset}
    if missing:
        raise SystemExit(f"{len(missing)} failing ids not found in probe file: {sorted(missing)[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in subset),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "rows": len(subset)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
