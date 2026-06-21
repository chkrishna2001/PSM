from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.data.rows import infer_row_task

from prod_memory.row_validation import remember_target_from_input, validate_prod_row, validate_prod_rows, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PACKAGE_ROOT / "data" / "prod-extraction-v3.jsonl"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "prod-extraction-v6.jsonl"
MIN_STORAGE_P50_CHARS = 500


def build_prod_extraction_v6_storage_only(
    output: Path,
    *,
    source: Path,
    min_input_chars: int = 0,
    require_facts: bool = False,
) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(f"source curriculum not found: {source}")

    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if infer_row_task(row) != "storage":
            continue
        validate_prod_row(row)
        if min_input_chars and len(remember_target_from_input(row["input"])) < min_input_chars:
            continue
        if require_facts and not row.get("expected", {}).get("facts"):
            continue
        rows.append(row)

    validation = validate_prod_rows(rows)
    if not validation["ok"]:
        raise ValueError(json.dumps(validation, indent=2))
    if not rows:
        raise ValueError("no storage rows after filter")

    lengths = sorted(len(remember_target_from_input(row["input"])) for row in rows)
    p50 = lengths[len(lengths) // 2]
    if p50 < MIN_STORAGE_P50_CHARS:
        raise ValueError(f"storage input p50 {p50} below minimum {MIN_STORAGE_P50_CHARS}")

    write_jsonl(output, rows)
    action_counts = Counter(str(row["expected"]["action"]) for row in rows)
    with_facts = sum(1 for row in rows if row.get("expected", {}).get("facts"))
    manifest = {
        "profile": "prod-extraction-v6-storage-only",
        "source": str(source),
        "output": str(output),
        "total_rows": len(rows),
        "storage_only": True,
        "input_chars_p50": p50,
        "input_chars_p90": lengths[int(len(lengths) * 0.9)],
        "input_chars_max": lengths[-1],
        "rows_with_facts": with_facts,
        "action_counts": dict(action_counts),
        "validation": validation,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build v6 storage-only mix from teacher v3 (prod-extraction-v3).")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-input-chars", type=int, default=0)
    parser.add_argument("--require-facts", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_prod_extraction_v6_storage_only(
        args.output,
        source=args.source,
        min_input_chars=args.min_input_chars,
        require_facts=args.require_facts,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
