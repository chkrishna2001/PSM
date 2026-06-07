from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.data import validate_training_row
from psm_model.generate_direct_behavior_curriculum import build_rows as build_direct_behavior_rows

PARSE_DRILL_ACTIONS = frozenset({"promote_semantic", "store_episodic"})
STRATIFIED_ACTIONS = frozenset({"promote_semantic", "store_episodic"})


def _parse_drill_rows(rows_per_action: int) -> list[dict[str, Any]]:
    drills: list[dict[str, Any]] = []
    for row in build_direct_behavior_rows(rows_per_action):
        action = str(row["expected"]["action"])
        if action not in PARSE_DRILL_ACTIONS:
            continue
        drills.append(row)
    return drills


def _sample_stratified_rows(
    source: Path,
    *,
    max_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not source.exists():
        return []

    by_action: dict[str, list[dict[str, Any]]] = {action: [] for action in sorted(STRATIFIED_ACTIONS)}
    for row in _load_rows(source):
        action = str(row["expected"]["action"])
        if action not in STRATIFIED_ACTIONS:
            continue
        _, issues = validate_training_row(row)
        if issues:
            continue
        by_action[action].append(row)

    if not any(by_action.values()):
        return []

    per_action_cap = max(1, max_rows // len(STRATIFIED_ACTIONS))
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for action in sorted(STRATIFIED_ACTIONS):
        pool = by_action[action]
        rng.shuffle(pool)
        sampled.extend(pool[:per_action_cap])

    rng.shuffle(sampled)
    return sampled[:max_rows]


def build_gate4_train_v1(
    output: Path,
    *,
    direct_probes: Path,
    expanded_probes: Path,
    stratified_source: Path | None = None,
    direct_copies: int = 500,
    expanded_copies: int = 40,
    drill_rows_per_action: int = 120,
    drill_copies: int = 25,
    stratified_max: int = 2500,
    stratified_seed: int = 42,
) -> dict[str, Any]:
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
    stratified_added = _copy_rows(
        stratified_rows,
        prefix="stratified-real",
        copies=1,
        seen=seen,
        output=rows,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    action_counts = Counter(row["expected"]["action"] for row in rows)
    total = len(rows)
    expanded_share = expanded_added / total if total else 0.0
    drill_share = drill_added / total if total else 0.0
    stratified_share = stratified_added / total if total else 0.0

    return {
        "curriculum": "gate4-train-v1",
        "direct_probes": str(direct_probes),
        "expanded_probes": str(expanded_probes),
        "stratified_source": str(stratified_source) if stratified_source is not None else None,
        "output": str(output),
        "rows": total,
        "direct_anchor_rows": direct_added,
        "expanded_full_rows": expanded_added,
        "parse_drill_rows": drill_added,
        "stratified_real_rows": stratified_added,
        "unique_drill_templates": len(drill_rows),
        "unique_stratified_templates": len(stratified_rows),
        "direct_copies": direct_copies,
        "expanded_copies": expanded_copies,
        "drill_rows_per_action": drill_rows_per_action,
        "drill_copies": drill_copies,
        "stratified_max": stratified_max,
        "mix_shares": {
            "expanded_full": round(expanded_share, 4),
            "parse_drill": round(drill_share, 4),
            "stratified_real": round(stratified_share, 4),
            "direct_anchor": round(direct_added / total if total else 0.0, 4),
        },
        "action_counts": dict(sorted(action_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Gate 4 production curriculum v1: expanded full-DSL dominant + parse drills "
            "+ stratified promote/store from converted real dialogue (no 25k base dilution)."
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--direct-probes", type=Path, required=True)
    parser.add_argument("--expanded-probes", type=Path, required=True)
    parser.add_argument(
        "--stratified-source",
        type=Path,
        default=None,
        help="Converted storage JSONL (e.g. full-storage filtered); promote/store rows sampled only.",
    )
    parser.add_argument("--direct-copies", type=int, default=500)
    parser.add_argument("--expanded-copies", type=int, default=40)
    parser.add_argument("--drill-rows-per-action", type=int, default=120)
    parser.add_argument("--drill-copies", type=int, default=25)
    parser.add_argument("--stratified-max", type=int, default=2500)
    parser.add_argument("--stratified-seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build_gate4_train_v1(
                args.output,
                direct_probes=args.direct_probes,
                expanded_probes=args.expanded_probes,
                stratified_source=args.stratified_source,
                direct_copies=args.direct_copies,
                expanded_copies=args.expanded_copies,
                drill_rows_per_action=args.drill_rows_per_action,
                drill_copies=args.drill_copies,
                stratified_max=args.stratified_max,
                stratified_seed=args.stratified_seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
