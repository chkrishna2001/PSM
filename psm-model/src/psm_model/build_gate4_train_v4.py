from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.build_gate4_train_v1 import _sample_stratified_rows
from psm_model.mine_gate4_parse_failures import mine_parse_failures, write_repair_pack

DEFAULT_V4 = {
    "direct_copies": 300,
    "expanded_copies": 100,
    "complete_tag_copies": 50,
    "stratified_max": 1500,
    "repair_copies": 1,
}


def build_gate4_train_v4(
    output: Path,
    *,
    direct_probes: Path,
    expanded_probes: Path,
    complete_tag_drills: Path,
    parse_repair: Path | None = None,
    eval_report: Path | None = None,
    repair_source: Path | None = None,
    stratified_source: Path | None = None,
    direct_copies: int = DEFAULT_V4["direct_copies"],
    expanded_copies: int = DEFAULT_V4["expanded_copies"],
    complete_tag_copies: int = DEFAULT_V4["complete_tag_copies"],
    stratified_max: int = DEFAULT_V4["stratified_max"],
    repair_copies: int = DEFAULT_V4["repair_copies"],
    stratified_seed: int = 42,
) -> dict[str, Any]:
    """Gate 4 v4: eval-matched expanded anchor (×100) + complete-tag drills; resume from 42k."""
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
    drill_rows = _load_rows(complete_tag_drills)
    stratified_rows = (
        _sample_stratified_rows(stratified_source, max_rows=stratified_max, seed=stratified_seed)
        if stratified_source is not None
        else []
    )

    direct_added = _copy_rows(direct_rows, prefix="direct-anchor", copies=direct_copies, seen=seen, output=rows)
    expanded_added = _copy_rows(expanded_rows, prefix="expanded-budget", copies=expanded_copies, seen=seen, output=rows)
    drill_added = _copy_rows(drill_rows, prefix="complete-tag", copies=complete_tag_copies, seen=seen, output=rows)
    repair_added = _copy_rows(repair_rows, prefix="parse-repair", copies=repair_copies, seen=seen, output=rows)
    stratified_added = _copy_rows(stratified_rows, prefix="stratified-real", copies=1, seen=seen, output=rows)

    rng = random.Random(stratified_seed)
    rng.shuffle(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    total = len(rows)
    parse_focus = drill_added + repair_added
    return {
        "curriculum": "gate4-train-v4",
        "canonical_resume": "psm-model/checkpoints/real-v3-50m-full-v2-step-042000.pt",
        "direct_probes": str(direct_probes),
        "expanded_probes": str(expanded_probes),
        "complete_tag_drills": str(complete_tag_drills),
        "parse_repair": str(parse_repair) if parse_repair is not None else None,
        "eval_report": str(eval_report) if eval_report is not None else None,
        "repair_source": str(repair_source) if repair_source is not None else None,
        "stratified_source": str(stratified_source) if stratified_source is not None else None,
        "output": str(output),
        "rows": total,
        "direct_anchor_rows": direct_added,
        "expanded_budget_rows": expanded_added,
        "complete_tag_rows": drill_added,
        "parse_repair_rows": repair_added,
        "stratified_real_rows": stratified_added,
        "mix_shares": {
            "expanded_budget": round(expanded_added / total if total else 0.0, 4),
            "complete_tag": round(drill_added / total if total else 0.0, 4),
            "parse_repair": round(repair_added / total if total else 0.0, 4),
            "parse_focus": round(parse_focus / total if total else 0.0, 4),
            "stratified_real": round(stratified_added / total if total else 0.0, 4),
            "direct_anchor": round(direct_added / total if total else 0.0, 4),
        },
        "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
        "repair_mine": repair_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Gate 4 v4: expanded budget ×100 + complete-tag drills; light repair from 42k."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--direct-probes", type=Path, required=True)
    parser.add_argument("--expanded-probes", type=Path, required=True)
    parser.add_argument("--complete-tag-drills", type=Path, required=True)
    parser.add_argument("--parse-repair", type=Path, default=None)
    parser.add_argument("--eval-report", type=Path, default=None)
    parser.add_argument("--repair-source", type=Path, default=None)
    parser.add_argument("--stratified-source", type=Path, default=None)
    parser.add_argument("--direct-copies", type=int, default=DEFAULT_V4["direct_copies"])
    parser.add_argument("--expanded-copies", type=int, default=DEFAULT_V4["expanded_copies"])
    parser.add_argument("--complete-tag-copies", type=int, default=DEFAULT_V4["complete_tag_copies"])
    parser.add_argument("--stratified-max", type=int, default=DEFAULT_V4["stratified_max"])
    parser.add_argument("--repair-copies", type=int, default=DEFAULT_V4["repair_copies"])
    parser.add_argument("--stratified-seed", type=int, default=42)
    args = parser.parse_args()

    print(
        json.dumps(
            build_gate4_train_v4(
                args.output,
                direct_probes=args.direct_probes,
                expanded_probes=args.expanded_probes,
                complete_tag_drills=args.complete_tag_drills,
                parse_repair=args.parse_repair,
                eval_report=args.eval_report,
                repair_source=args.repair_source,
                stratified_source=args.stratified_source,
                direct_copies=args.direct_copies,
                expanded_copies=args.expanded_copies,
                complete_tag_copies=args.complete_tag_copies,
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
