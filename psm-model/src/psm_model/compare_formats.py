from __future__ import annotations

import argparse
import json
from pathlib import Path

from psm_model.lean_format import (
    compact_json_array,
    encode_at_tag_decision,
    encode_tagged_decision,
    parse_at_tag_decision,
    parse_tagged_decision,
)
from psm_model.tokenizer import ByteTokenizer


def compare_file(path: Path) -> dict[str, object]:
    tokenizer = ByteTokenizer()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reports = []
    totals = {"json": 0, "array": 0, "tagged": 0, "at_tag": 0}
    for row in rows:
        decision = row["expected"]
        full_json = json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        array_json = compact_json_array(decision)
        tagged = encode_tagged_decision(decision)
        at_tag = encode_at_tag_decision(decision)
        parsed, issues = parse_tagged_decision(tagged)
        if issues:
            raise ValueError(f"{row['id']} tagged round-trip failed: {issues}")
        assert parsed is not None
        parsed_at_tag, at_tag_issues = parse_at_tag_decision(at_tag)
        if at_tag_issues:
            raise ValueError(f"{row['id']} @tag round-trip failed: {at_tag_issues}")
        assert parsed_at_tag is not None
        lengths = {
            "json": len(tokenizer.encode(full_json)),
            "array": len(tokenizer.encode(array_json)),
            "tagged": len(tokenizer.encode(tagged)),
            "at_tag": len(tokenizer.encode(at_tag)),
        }
        for key, value in lengths.items():
            totals[key] += value
        reports.append(
            {
                "id": row["id"],
                "tokens": lengths,
                "tagged_savings_vs_json": 1.0 - (lengths["tagged"] / lengths["json"]),
                "at_tag_savings_vs_json": 1.0 - (lengths["at_tag"] / lengths["json"]),
                "array_savings_vs_json": 1.0 - (lengths["array"] / lengths["json"]),
            }
        )
    return {
        "rows": len(rows),
        "totals": totals,
        "tagged_total_savings_vs_json": 1.0 - (totals["tagged"] / totals["json"]) if totals["json"] else 0.0,
        "at_tag_total_savings_vs_json": 1.0 - (totals["at_tag"] / totals["json"]) if totals["json"] else 0.0,
        "array_total_savings_vs_json": 1.0 - (totals["array"] / totals["json"]) if totals["json"] else 0.0,
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full JSON, compact array JSON, and tagged DSL token counts.")
    parser.add_argument("path", type=Path, help="Canonical JSONL probe/training rows")
    args = parser.parse_args()
    print(json.dumps(compare_file(args.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
