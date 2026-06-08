from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psm_model.analyze_eval_report import classify_row


def _load_rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("id") or "")
        if not row_id:
            raise ValueError(f"{path}:{line_number}: row missing id")
        if row_id in by_id:
            raise ValueError(f"{path}:{line_number}: duplicate id {row_id!r}")
        by_id[row_id] = row
    return by_id


def mine_parse_failures(
    eval_report: Path,
    source_jsonl: Path,
    *,
    buckets: tuple[str, ...] = ("parse_fail", "schema_fail"),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(eval_report.read_text(encoding="utf-8"))
    eval_rows = report.get("reports")
    if not isinstance(eval_rows, list):
        raise ValueError(f"{eval_report}: missing reports list")

    source_by_id = _load_rows_by_id(source_jsonl)
    repair_rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    bucket_counts: dict[str, int] = {}

    for eval_row in eval_rows:
        if not isinstance(eval_row, dict):
            continue
        bucket = classify_row(eval_row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if bucket not in buckets:
            continue
        row_id = str(eval_row.get("id") or "")
        if not row_id:
            continue
        gold = source_by_id.get(row_id)
        if gold is None:
            missing_ids.append(row_id)
            continue
        repair_rows.append(
            {
                "id": f"gate4-parse-repair:{row_id}",
                "input": gold["input"],
                "expected": gold["expected"],
                "source": f"gate4-parse-repair:{bucket}",
            }
        )

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} parse/schema failure ids missing from {source_jsonl}: "
            f"{missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}"
        )

    summary = {
        "eval_report": str(eval_report),
        "source_jsonl": str(source_jsonl),
        "buckets": list(buckets),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "repair_rows": len(repair_rows),
        "checkpoint": report.get("checkpoint"),
        "data": report.get("data"),
    }
    return repair_rows, summary


def write_repair_pack(output: Path, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine parse/schema failures from Gate 4 eval report into gold replay training rows."
    )
    parser.add_argument("eval_report", type=Path, help="Full eval_checkpoint JSON (with reports list).")
    parser.add_argument("source_jsonl", type=Path, help="Eval probe JSONL with input/expected keyed by id.")
    parser.add_argument("output", type=Path, help="Output parse-repair JSONL.")
    parser.add_argument(
        "--include-schema-fail",
        action="store_true",
        help="Also mine schema_fail rows (default: parse_fail only).",
    )
    args = parser.parse_args()

    buckets: tuple[str, ...] = ("parse_fail", "schema_fail") if args.include_schema_fail else ("parse_fail",)
    rows, summary = mine_parse_failures(args.eval_report, args.source_jsonl, buckets=buckets)
    write_repair_pack(args.output, rows)
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
