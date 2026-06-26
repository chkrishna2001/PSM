#!/usr/bin/env python3
"""Label prod fixtures with Gemma teacher → jsonl for hf-prod-v5b anchors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from prod_memory.curriculum_sources import load_fixture_cases
from prod_memory.openrouter_teacher import TeacherConfig, build_row_from_teacher
from prod_memory.row_validation import validate_prod_row, write_jsonl


def build_rows(*, model: str, fixtures_path: Path, fixture_ids: list[str] | None = None) -> list[dict]:
    cfg = TeacherConfig.from_env(model=model)
    if not cfg.api_key:
        raise SystemExit("OPENROUTER_API_KEY required")
    want = set(fixture_ids) if fixture_ids else None
    rows: list[dict] = []
    for case in load_fixture_cases(fixtures_path):
        case_id = str(case["id"])
        if want is not None and case_id not in want:
            continue
        text = str(case.get("llmResponse") or "").strip()
        if not text:
            continue
        row, _meta = build_row_from_teacher(
            text,
            row_id=f"gemma-fixture-{case_id}",
            source_id=case_id,
            source_kind=f"prod_{case.get('suite', 'fixture')}",
            config=cfg,
            use_heuristic_fallback=False,
        )
        if row is None:
            continue
        validate_prod_row(row)
        row["source"] = "gemma_fixture_teacher"
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--fixtures", type=Path, default=PACKAGE_ROOT / "fixtures" / "cases.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "data" / "prod-extraction-fixtures-gemma.jsonl",
    )
    parser.add_argument(
        "--fixture-ids",
        default="",
        help="Comma-separated fixture ids (default: all)",
    )
    args = parser.parse_args()
    fixture_ids = [x.strip() for x in args.fixture_ids.split(",") if x.strip()] or None
    rows = build_rows(model=args.model, fixtures_path=args.fixtures, fixture_ids=fixture_ids)
    if not rows:
        raise SystemExit("no fixture rows")
    write_jsonl(args.output, rows)
    actions = {}
    for row in rows:
        act = str(row["expected"].get("action") or "")
        actions[act] = actions.get(act, 0) + 1
    print(json.dumps({"output": str(args.output), "rows": len(rows), "actions": actions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
