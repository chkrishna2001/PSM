from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.build_gate5_train_v1 import _copy_task_rows
from psm_model.data.rows import infer_row_task
from psm_model.generate_recall_curriculum import build_recall_probe_rows

from prod_memory.curriculum_sources import build_primary_source_rows
from prod_memory.row_validation import validate_prod_row, validate_prod_rows, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "prod-extraction-v1.jsonl"
DEFAULT_MANIFEST = PACKAGE_ROOT / "data" / "prod-extraction-v1.manifest.json"

PROD_EXTRACTION_V1_PROFILE: dict[str, int] = {
    "expanded_copies": 2,
    "recall_copies": 50,
    "plan_copies": 15,
    "workflow_copies": 10,
    "nano_copies": 10,
    "technical_copies": 5,
    "noise_copies": 8,
}


def _bucket_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "plan": [],
        "workflow": [],
        "cursor": [],
        "technical": [],
        "noise": [],
        "other": [],
    }
    for row in rows:
        source = str(row.get("source") or "")
        row_id = str(row.get("id") or "")
        if "workflow" in source or "workflow-" in row_id:
            buckets["workflow"].append(row)
        elif "plan" in source or row_id.startswith("plan-") or row_id.startswith("fixture-plan"):
            buckets["plan"].append(row)
        elif "cursor" in source or row_id.startswith("fixture-cursor"):
            buckets["cursor"].append(row)
        elif "technical" in source or row_id.startswith("technical-"):
            buckets["technical"].append(row)
        elif "noise" in source or row_id.startswith("noise-"):
            buckets["noise"].append(row)
        else:
            buckets["other"].append(row)
    return buckets


def _copy_primary_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    copies: int,
    seen: set[str],
    output: list[dict[str, Any]],
) -> int:
    added = 0
    for row in rows:
        row_id = str(row.get("id") or "row")
        for copy_index in range(copies):
            copied_id = f"{prefix}:{copy_index}:{row_id}"
            if copied_id in seen:
                continue
            seen.add(copied_id)
            output.append({
                "id": copied_id,
                "input": row["input"],
                "expected": row["expected"],
                "source": f"prod_extraction_v1:{prefix}",
            })
            added += 1
    return added


def build_prod_extraction_v1(
    output: Path,
    *,
    expanded_probes: Path | None = None,
    direct_probes: Path | None = None,
    nano_rows: Path | None = None,
    chatgpt_rows: Path | None = None,
    profile: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    copies = profile or PROD_EXTRACTION_V1_PROFILE
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    primary = build_primary_source_rows()
    for row in primary:
        validate_prod_row(row)
    buckets = _bucket_rows(primary)

    plan_source = [*buckets["plan"], *buckets["cursor"], *buckets["other"]]
    workflow_source = buckets["workflow"] or [
        row for row in primary if str(row.get("expected", {}).get("action")) == "store_episodic"
    ][:2]
    technical_source = buckets["technical"]
    noise_source = buckets["noise"]

    plan_added = _copy_primary_rows(plan_source, prefix="plan", copies=copies["plan_copies"], seen=seen, output=rows)
    workflow_added = _copy_primary_rows(
        workflow_source,
        prefix="workflow",
        copies=copies["workflow_copies"],
        seen=seen,
        output=rows,
    )
    technical_added = _copy_primary_rows(
        technical_source,
        prefix="technical",
        copies=copies["technical_copies"],
        seen=seen,
        output=rows,
    )
    noise_added = _copy_primary_rows(
        noise_source,
        prefix="noise",
        copies=copies["noise_copies"],
        seen=seen,
        output=rows,
    )

    expanded_path = expanded_probes
    if expanded_path is None or not expanded_path.exists():
        expanded_path = direct_probes
    expanded_added = 0
    if expanded_path and expanded_path.exists():
        expanded_added = _copy_rows(
            _load_rows(expanded_path),
            prefix="prod-expanded",
            copies=copies["expanded_copies"],
            seen=seen,
            output=rows,
        )

    recall_added = _copy_task_rows(
        build_recall_probe_rows(),
        prefix="prod-recall",
        copies=copies["recall_copies"],
        seen=seen,
        output=rows,
    )

    nano_added = 0
    if nano_rows and nano_rows.exists():
        nano_added = _copy_rows(
            _load_rows(nano_rows),
            prefix="prod-nano",
            copies=copies["nano_copies"],
            seen=seen,
            output=rows,
        )

    chatgpt_added = 0
    if chatgpt_rows and chatgpt_rows.exists():
        chatgpt_added = _copy_rows(
            _load_rows(chatgpt_rows),
            prefix="prod-chatgpt",
            copies=copies["nano_copies"],
            seen=seen,
            output=rows,
        )

    validation = validate_prod_rows([row for row in rows if infer_row_task(row) == "storage"])
    if not validation["ok"]:
        raise ValueError(json.dumps(validation, indent=2))

    write_jsonl(output, rows)
    action_counts = Counter(str(row["expected"]["action"]) for row in rows if infer_row_task(row) == "storage")
    task_counts = Counter(infer_row_task(row) for row in rows)
    manifest = {
        "profile": "prod-extraction-v1",
        "seed": seed,
        "output": str(output),
        "total_rows": len(rows),
        "copies": copies,
        "added": {
            "plan": plan_added,
            "workflow": workflow_added,
            "technical": technical_added,
            "noise": noise_added,
            "expanded_regression": expanded_added,
            "recall_regression": recall_added,
            "nano": nano_added,
            "chatgpt": chatgpt_added,
        },
        "sources": {
            "expanded_probes": str(expanded_path) if expanded_path else None,
            "direct_probes": str(direct_probes) if direct_probes else None,
            "nano_rows": str(nano_rows) if nano_rows else None,
            "chatgpt_rows": str(chatgpt_rows) if chatgpt_rows else None,
            "primary_seed_rows": len(primary),
        },
        "action_counts": dict(action_counts),
        "task_counts": dict(task_counts),
        "validation": validation,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build prod-extraction-v1 curriculum mix (isolated from gate6).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expanded-probes", type=Path, default=Path("psm-model/data/probes/expanded_probes.jsonl"))
    parser.add_argument("--direct-probes", type=Path, default=Path("psm-model/data/probes/direct_probes.jsonl"))
    parser.add_argument("--nano-rows", type=Path, default=None)
    parser.add_argument("--chatgpt-rows", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    manifest = build_prod_extraction_v1(
        args.output,
        expanded_probes=args.expanded_probes,
        direct_probes=args.direct_probes,
        nano_rows=args.nano_rows,
        chatgpt_rows=args.chatgpt_rows,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
