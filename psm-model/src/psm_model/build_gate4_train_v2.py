from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.build_gate4_train_v1 import (
    PARSE_DRILL_ACTIONS,
    _parse_drill_rows,
    _sample_stratified_rows,
)
from psm_model.mine_gate4_parse_failures import mine_parse_failures, write_repair_pack

DEFAULT_V2 = {
    "direct_copies": 500,
    "expanded_copies": 25,
    "drill_rows_per_action": 120,
    "drill_copies": 50,
    "stratified_max": 1500,
    "repair_copies": 3,
}


def build_gate4_train_v2(
    output: Path,
    *,
    direct_probes: Path,
    expanded_probes: Path,
    parse_repair: Path | None = None,
    eval_report: Path | None = None,
    repair_source: Path | None = None,
    stratified_source: Path | None = None,
    direct_copies: int = DEFAULT_V2["direct_copies"],
    expanded_copies: int = DEFAULT_V2["expanded_copies"],
    drill_rows_per_action: int = DEFAULT_V2["drill_rows_per_action"],
    drill_copies: int = DEFAULT_V2["drill_copies"],
    stratified_max: int = DEFAULT_V2["stratified_max"],
    repair_copies: int = DEFAULT_V2["repair_copies"],
    stratified_seed: int = 42,
) -> dict[str, Any]:
    repair_rows: list[dict[str, Any]]
    repair_summary: dict[str, Any] | None = None

    if parse_repair is not None and parse_repair.exists():
        repair_rows = _load_rows(parse_repair)
    elif eval_report is not None and repair_source is not None:
        repair_rows, repair_summary = mine_parse_failures(eval_report, repair_source)
        if parse_repair is not None:
            write_repair_pack(parse_repair, repair_rows)
    else:
        repair_rows = []

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    direct_rows = _load_rows(direct_probes)
    expanded_rows = _load_rows(expanded_probes)
    drill_rows = _parse_drill_rows(drill_rows_per_action)
    stratified_rows = (
        _sample_stratified_rows(stratified_source, max_rows=stratified_max, seed=stratified_seed)
        if stratified_source is not None
        else []
    )

    direct_added = _copy_rows(
        direct_rows,
        prefix="direct-anchor",
        copies=direct_copies,
        seen=seen,
        output=rows,
    )
    expanded_added = _copy_rows(
        expanded_rows,
        prefix="expanded-full",
        copies=expanded_copies,
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
    repair_added = _copy_rows(
        repair_rows,
        prefix="parse-repair",
        copies=repair_copies,
        seen=seen,
        output=rows,
    )
    stratified_added = _copy_rows(
        stratified_rows,
        prefix="stratified-real",
        copies=1,
        seen=seen,
        output=rows,
    )

    rng = random.Random(stratified_seed)
    rng.shuffle(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    total = len(rows)
    parse_repair_share = repair_added / total if total else 0.0
    drill_share = drill_added / total if total else 0.0
    parse_focus_share = parse_repair_share + drill_share

    return {
        "curriculum": "gate4-train-v2",
        "direct_probes": str(direct_probes),
        "expanded_probes": str(expanded_probes),
        "parse_repair": str(parse_repair) if parse_repair is not None else None,
        "eval_report": str(eval_report) if eval_report is not None else None,
        "repair_source": str(repair_source) if repair_source is not None else None,
        "stratified_source": str(stratified_source) if stratified_source is not None else None,
        "output": str(output),
        "rows": total,
        "direct_anchor_rows": direct_added,
        "expanded_full_rows": expanded_added,
        "parse_drill_rows": drill_added,
        "parse_repair_rows": repair_added,
        "stratified_real_rows": stratified_added,
        "unique_drill_templates": len(drill_rows),
        "unique_repair_templates": len(repair_rows),
        "unique_stratified_templates": len(stratified_rows),
        "direct_copies": direct_copies,
        "expanded_copies": expanded_copies,
        "drill_rows_per_action": drill_rows_per_action,
        "drill_copies": drill_copies,
        "stratified_max": stratified_max,
        "repair_copies": repair_copies,
        "mix_shares": {
            "expanded_full": round(expanded_added / total if total else 0.0, 4),
            "parse_drill": round(drill_share, 4),
            "parse_repair": round(parse_repair_share, 4),
            "parse_focus": round(parse_focus_share, 4),
            "stratified_real": round(stratified_added / total if total else 0.0, 4),
            "direct_anchor": round(direct_added / total if total else 0.0, 4),
        },
        "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
        "repair_mine": repair_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Gate 4 curriculum v2: parse-heavy mix with failure-mined repair pack "
            "(expanded ×25, drills ×50, repair ×3, direct ×500, stratified max 1500)."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--direct-probes", type=Path, required=True)
    parser.add_argument("--expanded-probes", type=Path, required=True)
    parser.add_argument(
        "--parse-repair",
        type=Path,
        default=None,
        help="Pre-built parse-repair JSONL; if missing, mine from --eval-report + --repair-source.",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=None,
        help="Full Gate 4 expanded eval JSON (for on-the-fly repair mining).",
    )
    parser.add_argument(
        "--repair-source",
        type=Path,
        default=None,
        help="Eval probe JSONL with gold input/expected (typically token-budget filtered).",
    )
    parser.add_argument(
        "--stratified-source",
        type=Path,
        default=None,
        help="Converted storage JSONL; promote/store rows sampled only.",
    )
    parser.add_argument("--direct-copies", type=int, default=DEFAULT_V2["direct_copies"])
    parser.add_argument("--expanded-copies", type=int, default=DEFAULT_V2["expanded_copies"])
    parser.add_argument("--drill-rows-per-action", type=int, default=DEFAULT_V2["drill_rows_per_action"])
    parser.add_argument("--drill-copies", type=int, default=DEFAULT_V2["drill_copies"])
    parser.add_argument("--stratified-max", type=int, default=DEFAULT_V2["stratified_max"])
    parser.add_argument("--repair-copies", type=int, default=DEFAULT_V2["repair_copies"])
    parser.add_argument("--stratified-seed", type=int, default=42)
    args = parser.parse_args()

    if args.parse_repair is None and args.eval_report is None:
        raise SystemExit("Provide --parse-repair or --eval-report (+ --repair-source) for v2 curriculum.")

    print(
        json.dumps(
            build_gate4_train_v2(
                args.output,
                direct_probes=args.direct_probes,
                expanded_probes=args.expanded_probes,
                parse_repair=args.parse_repair,
                eval_report=args.eval_report,
                repair_source=args.repair_source,
                stratified_source=args.stratified_source,
                direct_copies=args.direct_copies,
                expanded_copies=args.expanded_copies,
                drill_rows_per_action=args.drill_rows_per_action,
                drill_copies=args.drill_copies,
                stratified_max=args.stratified_max,
                repair_copies=args.repair_copies,
                stratified_seed=args.stratified_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
