"""Classify raw parse-failure outputs from an eval report with raw_output captured.

Buckets each failing row into mechanical classes so interventions can be targeted:
  - runaway_no_end: generation never emitted END (loop) — missing R:/END
  - fact_pipe_slip: F: line with < 6 pipe fields
  - q_numeric_slip: Q: quad containing a non-numeric field
  - escape_error: raw pipe/comma inside an escaped field (model skipped \\p / \\c)
  - other: anything else

Usage:
    python psm-model/scripts/classify_parse_failures.py \
        psm-model/checkpoints/gate-eval/parse-fails-042000-raw.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def classify(raw: str) -> list[str]:
    classes: list[str] = []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    has_end = any(line == "END" for line in lines)
    body = []
    for line in lines:
        body.append(line)
        if line == "END":
            break

    if not has_end:
        classes.append("runaway_no_end")

    for line in body:
        if line.startswith("F:"):
            parts = line[2:].split("|")
            if len(parts) < 6:
                classes.append("fact_pipe_slip")
        if line.startswith("Q:"):
            for part in line[2:].split(","):
                part = part.strip()
                if part and not re.fullmatch(r"-?\d*\.?\d+(e-?\d+)?", part):
                    classes.append("q_numeric_slip")

    if not any(line.startswith("R:") for line in body):
        classes.append("missing_reasoning_line")

    # repeated-line loop signature
    counts = collections.Counter(body)
    if counts and counts.most_common(1)[0][1] >= 5:
        classes.append("repeated_line_loop")

    return classes or ["other"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--samples", type=int, default=3, help="Raw samples to print per class.")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    fails = [r for r in report["reports"] if not r.get("parse_valid") and r.get("raw_output")]
    print(f"rows with raw_output: {len(fails)} / {len(report['reports'])}\n")

    by_class: dict[str, list[dict]] = collections.defaultdict(list)
    for row in fails:
        for cls in classify(row["raw_output"]):
            by_class[cls].append(row)

    for cls, rows in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        print(f"== {cls}: {len(rows)} rows ==")
        for row in rows[: args.samples]:
            preview = row["raw_output"][:400].replace("\n", " / ")
            print(f"  [{row['id']}] tokens={row['generated_tokens']}")
            print(f"    {preview}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
