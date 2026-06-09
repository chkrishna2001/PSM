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

DEFAULT_V3 = {
    "direct_copies": 400,
    "expanded_copies": 60,
    "fact_drill_copies": 80,
    "chatgpt_copies": 4,
    "stratified_max": 2500,
    "repair_copies": 2,
}


def build_gate4_train_v3(
    output: Path,
    *,
    direct_probes: Path,
    expanded_probes: Path,
    fact_drills: Path,
    chatgpt_rows: Path | None = None,
    parse_repair: Path | None = None,
    eval_report: Path | None = None,
    repair_source: Path | None = None,
    stratified_source: Path | None = None,
    direct_copies: int = DEFAULT_V3["direct_copies"],
    expanded_copies: int = DEFAULT_V3["expanded_copies"],
    fact_drill_copies: int = DEFAULT_V3["fact_drill_copies"],
    chatgpt_copies: int = DEFAULT_V3["chatgpt_copies"],
    stratified_max: int = DEFAULT_V3["stratified_max"],
    repair_copies: int = DEFAULT_V3["repair_copies"],
    stratified_seed: int = 42,
) -> dict[str, Any]:
    """Production Gate 4 curriculum: eval-matched expanded anchor + format drills + real chat diversity."""
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
    drill_rows = _load_rows(fact_drills)
    chatgpt = _load_rows(chatgpt_rows) if chatgpt_rows is not None and chatgpt_rows.exists() else []
    stratified_rows = (
        _sample_stratified_rows(stratified_source, max_rows=stratified_max, seed=stratified_seed)
        if stratified_source is not None
        else []
    )

    direct_added = _copy_rows(direct_rows, prefix="direct-anchor", copies=direct_copies, seen=seen, output=rows)
    expanded_added = _copy_rows(expanded_rows, prefix="expanded-budget", copies=expanded_copies, seen=seen, output=rows)
    drill_added = _copy_rows(drill_rows, prefix="fact-format", copies=fact_drill_copies, seen=seen, output=rows)
    chatgpt_added = _copy_rows(chatgpt, prefix="chatgpt-real", copies=chatgpt_copies, seen=seen, output=rows) if chatgpt else 0
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
        "curriculum": "gate4-train-v3",
        "direct_probes": str(direct_probes),
        "expanded_probes": str(expanded_probes),
        "fact_drills": str(fact_drills),
        "chatgpt_rows": str(chatgpt_rows) if chatgpt_rows is not None else None,
        "parse_repair": str(parse_repair) if parse_repair is not None else None,
        "eval_report": str(eval_report) if eval_report is not None else None,
        "repair_source": str(repair_source) if repair_source is not None else None,
        "stratified_source": str(stratified_source) if stratified_source is not None else None,
        "output": str(output),
        "rows": total,
        "direct_anchor_rows": direct_added,
        "expanded_budget_rows": expanded_added,
        "fact_format_rows": drill_added,
        "chatgpt_real_rows": chatgpt_added,
        "parse_repair_rows": repair_added,
        "stratified_real_rows": stratified_added,
        "unique_expanded_templates": len(expanded_rows),
        "unique_fact_drill_templates": len(drill_rows),
        "unique_chatgpt_templates": len(chatgpt),
        "unique_repair_templates": len(repair_rows),
        "mix_shares": {
            "expanded_budget": round(expanded_added / total if total else 0.0, 4),
            "fact_format": round(drill_added / total if total else 0.0, 4),
            "parse_repair": round(repair_added / total if total else 0.0, 4),
            "parse_focus": round(parse_focus / total if total else 0.0, 4),
            "chatgpt_real": round(chatgpt_added / total if total else 0.0, 4),
            "stratified_real": round(stratified_added / total if total else 0.0, 4),
            "direct_anchor": round(direct_added / total if total else 0.0, 4),
        },
        "action_counts": dict(sorted(Counter(row["expected"]["action"] for row in rows).items())),
        "repair_mine": repair_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Gate 4 curriculum v3 for production: expanded budget ×60, fact-format drills ×80, "
            "ChatGPT real rows, light parse-repair ×2."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--direct-probes", type=Path, required=True)
    parser.add_argument(
        "--expanded-probes",
        type=Path,
        required=True,
        help="Use expanded-probe-v1-budget.jsonl (same file Gate 4 eval runs on).",
    )
    parser.add_argument("--fact-drills", type=Path, required=True)
    parser.add_argument("--chatgpt-rows", type=Path, default=None)
    parser.add_argument("--parse-repair", type=Path, default=None)
    parser.add_argument("--eval-report", type=Path, default=None)
    parser.add_argument("--repair-source", type=Path, default=None)
    parser.add_argument("--stratified-source", type=Path, default=None)
    parser.add_argument("--direct-copies", type=int, default=DEFAULT_V3["direct_copies"])
    parser.add_argument("--expanded-copies", type=int, default=DEFAULT_V3["expanded_copies"])
    parser.add_argument("--fact-drill-copies", type=int, default=DEFAULT_V3["fact_drill_copies"])
    parser.add_argument("--chatgpt-copies", type=int, default=DEFAULT_V3["chatgpt_copies"])
    parser.add_argument("--stratified-max", type=int, default=DEFAULT_V3["stratified_max"])
    parser.add_argument("--repair-copies", type=int, default=DEFAULT_V3["repair_copies"])
    parser.add_argument("--stratified-seed", type=int, default=42)
    args = parser.parse_args()

    print(
        json.dumps(
            build_gate4_train_v3(
                args.output,
                direct_probes=args.direct_probes,
                expanded_probes=args.expanded_probes,
                fact_drills=args.fact_drills,
                chatgpt_rows=args.chatgpt_rows,
                parse_repair=args.parse_repair,
                eval_report=args.eval_report,
                repair_source=args.repair_source,
                stratified_source=args.stratified_source,
                direct_copies=args.direct_copies,
                expanded_copies=args.expanded_copies,
                fact_drill_copies=args.fact_drill_copies,
                chatgpt_copies=args.chatgpt_copies,
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
