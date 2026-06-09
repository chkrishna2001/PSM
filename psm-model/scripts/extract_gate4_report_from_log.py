"""Extract gate4-full-expanded JSON object from a RunPod eval session log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_gate4_report(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = "--- gate4-full-expanded ---"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"{log_path}: missing {marker!r}")
    chunk = text[start + len(marker) :]
    brace = chunk.find("{")
    if brace < 0:
        raise ValueError(f"{log_path}: no JSON after marker")
    decoder = json.JSONDecoder()
    report, _ = decoder.raw_decode(chunk, brace)
    if not isinstance(report, dict) or "reports" not in report:
        raise ValueError(f"{log_path}: parsed object is not an eval report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract gate4-full-expanded.json from eval terminal log.")
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = extract_gate4_report(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "parse_valid_rate": report.get("parse_valid_rate"),
                "action_accuracy": report.get("action_accuracy"),
                "examples": report.get("examples"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
