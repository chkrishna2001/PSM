from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.data import load_jsonl_rows
from psm_model.data.rows import infer_row_task
from psm_model.generate_recall_curriculum import build_recall_probe_rows


def _copy_task_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    copies: int,
    seen: set[str],
    output: list[dict[str, Any]],
) -> int:
    added = 0
    for row in rows:
        row_id = str(row.get("id") or row.get("case") or "row")
        task = infer_row_task(row)
        for copy_index in range(copies):
            copied_id = f"{prefix}:{copy_index}:{row_id}"
            if copied_id in seen:
                continue
            seen.add(copied_id)
            copied = {
                "id": copied_id,
                "input": row["input"],
                "expected": row["expected"],
                "source": f"gate5_curriculum:{prefix}",
                "task": task,
            }
            if row.get("split"):
                copied["split"] = row["split"]
            output.append(copied)
            added += 1
    return added


def build_gate5_train_v1(
    output: Path,
    *,
    expanded_probes: Path,
    direct_probes: Path | None = None,
    expanded_copies: int = 25,
    direct_copies: int = 100,
    recall_copies: int = 20,
    recall_rows: list[dict[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Mixed storage+recall curriculum. Storage mass is high to prevent Gate 4 collapse."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    expanded_rows = _load_rows(expanded_probes)
    expanded_added = _copy_rows(
        expanded_rows,
        prefix="gate5-expanded",
        copies=expanded_copies,
        seen=seen,
        output=rows,
    )

    direct_added = 0
    if direct_probes and direct_probes.exists():
        direct_added = _copy_rows(
            _load_rows(direct_probes),
            prefix="gate5-direct",
            copies=direct_copies,
            seen=seen,
            output=rows,
        )

    recall_source = recall_rows if recall_rows is not None else build_recall_probe_rows()
    recall_added = _copy_task_rows(
        recall_source,
        prefix="gate5-recall",
        copies=recall_copies,
        seen=seen,
        output=rows,
    )

    rng = random.Random(seed)
    rng.shuffle(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    gate = load_jsonl_rows(output)
    task_counts = gate.task_counts
    storage_rows = task_counts.get("storage", 0)
    recall_total = sum(task_counts.get(key, 0) for key in ("recall_plan", "context_plan"))
    return {
        "curriculum": "gate5-train-v1",
        "output": str(output),
        "rows": len(rows),
        "expanded_anchor_rows": expanded_added,
        "direct_anchor_rows": direct_added,
        "recall_anchor_rows": recall_added,
        "expanded_copies": expanded_copies,
        "direct_copies": direct_copies,
        "recall_copies": recall_copies,
        "task_counts": task_counts,
        "storage_fraction": storage_rows / len(rows) if rows else 0.0,
        "recall_fraction": recall_total / len(rows) if rows else 0.0,
        "dataset_gate": gate.to_dict(),
        "action_counts": dict(
            sorted(
                Counter(
                    row["expected"]["action"]
                    for row in rows
                    if infer_row_task(row) == "storage" and isinstance(row.get("expected"), dict) and "action" in row["expected"]
                ).items()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("psm-model/data/curriculum/psm-50m-gate5-train-v1.jsonl"),
    )
    parser.add_argument(
        "--expanded-probes",
        type=Path,
        default=Path("psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl"),
    )
    parser.add_argument(
        "--direct-probes",
        type=Path,
        default=Path("psm-model/data/probes/direct_probes.jsonl"),
    )
    parser.add_argument("--expanded-copies", type=int, default=25)
    parser.add_argument("--direct-copies", type=int, default=100)
    parser.add_argument("--recall-copies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_gate5_train_v1(
        args.output,
        expanded_probes=args.expanded_probes,
        direct_probes=args.direct_probes,
        expanded_copies=args.expanded_copies,
        direct_copies=args.direct_copies,
        recall_copies=args.recall_copies,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["dataset_gate"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
