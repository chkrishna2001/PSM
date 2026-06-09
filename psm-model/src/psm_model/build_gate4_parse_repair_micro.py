from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.build_gate4_train_v1 import _parse_drill_rows
from psm_model.mine_gate4_parse_failures import mine_parse_failures, write_repair_pack


def build_gate4_parse_repair_micro(
    output: Path,
    *,
    direct_probes: Path,
    eval_report: Path,
    repair_source: Path,
    parse_repair: Path | None = None,
    direct_copies: int = 25,
    drill_rows_per_action: int = 120,
    drill_copies: int = 3,
    repair_copies: int = 12,
    seed: int = 42,
) -> dict[str, Any]:
    if parse_repair is not None and parse_repair.exists():
        repair_rows = _load_rows(parse_repair)
        repair_summary = None
    else:
        repair_rows, repair_summary = mine_parse_failures(eval_report, repair_source)
        if parse_repair is not None:
            write_repair_pack(parse_repair, repair_rows)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    drill_rows = _parse_drill_rows(drill_rows_per_action)

    repair_added = _copy_rows(
        repair_rows,
        prefix="parse-repair",
        copies=repair_copies,
        seen=seen,
        output=rows,
    )
    drill_added = _copy_rows(
        drill_rows,
        prefix="parse-drill",
        copies=drill_copies,
        seen=seen,
        output=rows,
    )
    direct_added = _copy_rows(
        _load_rows(direct_probes),
        prefix="direct-anchor",
        copies=direct_copies,
        seen=seen,
        output=rows,
    )

    rng = random.Random(seed)
    rng.shuffle(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    total = len(rows)
    parse_focus = repair_added + drill_added
    return {
        "curriculum": "gate4-parse-repair-micro",
        "output": str(output),
        "rows": total,
        "parse_repair_rows": repair_added,
        "parse_drill_rows": drill_added,
        "direct_anchor_rows": direct_added,
        "unique_repair_templates": len(repair_rows),
        "unique_drill_templates": len(drill_rows),
        "repair_copies": repair_copies,
        "drill_copies": drill_copies,
        "direct_copies": direct_copies,
        "mix_shares": {
            "parse_repair": round(repair_added / total if total else 0.0, 4),
            "parse_drill": round(drill_added / total if total else 0.0, 4),
            "parse_focus": round(parse_focus / total if total else 0.0, 4),
            "direct_anchor": round(direct_added / total if total else 0.0, 4),
        },
        "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
        "repair_mine": repair_summary,
        "eval_report": str(eval_report),
        "repair_source": str(repair_source),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build parse-repair-only micro curriculum for Gate 4 step bump (e.g. 40k→42k)."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--direct-probes", type=Path, required=True)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--repair-source", type=Path, required=True)
    parser.add_argument("--parse-repair", type=Path, default=None)
    parser.add_argument("--direct-copies", type=int, default=25)
    parser.add_argument("--drill-rows-per-action", type=int, default=120)
    parser.add_argument("--drill-copies", type=int, default=3)
    parser.add_argument("--repair-copies", type=int, default=12)
    args = parser.parse_args()
    print(
        json.dumps(
            build_gate4_parse_repair_micro(
                args.output,
                direct_probes=args.direct_probes,
                eval_report=args.eval_report,
                repair_source=args.repair_source,
                parse_repair=args.parse_repair,
                direct_copies=args.direct_copies,
                drill_rows_per_action=args.drill_rows_per_action,
                drill_copies=args.drill_copies,
                repair_copies=args.repair_copies,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
