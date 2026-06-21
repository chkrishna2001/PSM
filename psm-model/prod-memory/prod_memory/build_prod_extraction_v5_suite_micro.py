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

from prod_memory.build_prod_extraction_v1 import _copy_primary_rows
from prod_memory.curriculum_sources import (
    build_fixture_rows,
    build_noise_rows,
    build_plan_handoff_rows,
    build_technical_rows,
)
from prod_memory.row_validation import validate_prod_row, validate_prod_rows, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "prod-extraction-v5.jsonl"
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DIRECT_PROBES = REPO_ROOT / "psm-model" / "data" / "probes" / "direct_probes.jsonl"

# ponytail: frozen from phase-5-failure-mining-2026-06-21.md — refresh after eval shifts
FOCUS_SUITE_FIXTURES: dict[str, frozenset[str]] = {
    "plan_chunks": frozenset({"fixture-plan-01-handoff", "fixture-plan-02-chunking"}),
    "workflow": frozenset({"fixture-workflow-review-pr", "fixture-workflow-runpod"}),
    "technical": frozenset({"fixture-technical-eslint", "fixture-technical-api"}),
    "cursor_shaped": frozenset({"fixture-cursor-01-summary", "fixture-cursor-02-debug"}),
}
ANCHOR_FIXTURE_IDS = frozenset({
    "fixture-cursor-01-summary",
    "fixture-noise-filler",
    "fixture-noise-meta",
})

# Phase 5 v1 philosophy: moderate primary copies + heavy recall anchor; NOT v4 ×40 fail-copy
PROD_EXTRACTION_V5_PROFILE: dict[str, int] = {
    "focus_copies": 5,
    "plan_seed_copies": 15,
    "technical_seed_copies": 5,
    "pass_copies": 5,
    "noise_copies": 8,
    "expanded_copies": 2,
    "recall_copies": 50,
}


def _anchor_row_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["id"]) for row in rows}


def build_prod_extraction_v5_suite_micro(
    output: Path,
    *,
    focus_suite: str = "plan_chunks",
    direct_probes: Path | None = None,
    profile: dict[str, int] | None = None,
    seed: int = 42,
    min_anchor_fraction: float = 0.5,
) -> dict[str, Any]:
    if focus_suite not in FOCUS_SUITE_FIXTURES:
        raise ValueError(f"unknown focus_suite {focus_suite!r}; choose from {sorted(FOCUS_SUITE_FIXTURES)}")

    copies = profile or PROD_EXTRACTION_V5_PROFILE
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    anchor_ids: set[str] = set()

    fixtures = build_fixture_rows()
    for row in fixtures:
        validate_prod_row(row)

    focus_ids = FOCUS_SUITE_FIXTURES[focus_suite]
    focus_rows = [row for row in fixtures if row["id"] in focus_ids]
    anchor_fixture_rows = [row for row in fixtures if row["id"] in ANCHOR_FIXTURE_IDS]

    focus_added = _copy_primary_rows(
        focus_rows, prefix=f"v5-{focus_suite}-focus", copies=copies["focus_copies"], seen=seen, output=rows
    )

    plan_seed_added = 0
    if focus_suite == "plan_chunks":
        plan_seed_added = _copy_primary_rows(
            build_plan_handoff_rows(),
            prefix="v5-plan-seed",
            copies=copies["plan_seed_copies"],
            seen=seen,
            output=rows,
        )

    technical_seed_added = 0
    if focus_suite == "technical":
        technical_seed_added = _copy_primary_rows(
            build_technical_rows(),
            prefix="v5-technical-seed",
            copies=copies["technical_seed_copies"],
            seen=seen,
            output=rows,
        )

    pass_added = _copy_primary_rows(
        [row for row in anchor_fixture_rows if row["id"] == "fixture-cursor-01-summary"],
        prefix="v5-anchor-pass",
        copies=copies["pass_copies"],
        seen=seen,
        output=rows,
    )
    noise_fixture_added = _copy_primary_rows(
        [row for row in anchor_fixture_rows if row["id"].startswith("fixture-noise")],
        prefix="v5-anchor-noise-fixture",
        copies=copies["noise_copies"],
        seen=seen,
        output=rows,
    )
    noise_seed_added = _copy_primary_rows(
        build_noise_rows(),
        prefix="v5-anchor-noise",
        copies=copies["noise_copies"],
        seen=seen,
        output=rows,
    )

    direct_path = direct_probes or DEFAULT_DIRECT_PROBES
    expanded_added = 0
    if direct_path.exists():
        expanded_rows: list[dict[str, Any]] = []
        expanded_added = _copy_rows(
            _load_rows(direct_path),
            prefix="v5-expanded",
            copies=copies["expanded_copies"],
            seen=seen,
            output=expanded_rows,
        )
        rows.extend(expanded_rows)
        anchor_ids.update(_anchor_row_ids(expanded_rows))

    recall_rows: list[dict[str, Any]] = []
    recall_added = _copy_task_rows(
        build_recall_probe_rows(),
        prefix="v5-recall",
        copies=copies["recall_copies"],
        seen=seen,
        output=recall_rows,
    )
    rows.extend(recall_rows)
    anchor_ids.update(_anchor_row_ids(recall_rows))
    anchor_ids.update(
        row_id
        for row in rows
        for row_id in [str(row["id"])]
        if row_id.startswith(("v5-anchor-", "v5-expanded:", "v5-recall:"))
    )

    storage_rows = [row for row in rows if infer_row_task(row) == "storage"]
    validation = validate_prod_rows(storage_rows)
    if not validation["ok"]:
        raise ValueError(json.dumps(validation, indent=2))

    anchor_count = sum(1 for row in rows if str(row["id"]) in anchor_ids or str(row["id"]).startswith("v5-anchor-"))
    anchor_fraction = anchor_count / max(1, len(rows))
    if anchor_fraction < min_anchor_fraction:
        raise ValueError(
            f"anchor_fraction {anchor_fraction:.3f} below min {min_anchor_fraction}; "
            "increase recall/expanded/noise copies"
        )

    write_jsonl(output, rows)
    action_counts = Counter(str(row["expected"]["action"]) for row in storage_rows)
    task_counts = Counter(infer_row_task(row) for row in rows)
    manifest = {
        "profile": f"prod-extraction-v5-suite-micro:{focus_suite}",
        "seed": seed,
        "output": str(output),
        "total_rows": len(rows),
        "focus_suite": focus_suite,
        "focus_fixture_ids": sorted(focus_ids),
        "copies": copies,
        "anchor_fraction": round(anchor_fraction, 4),
        "added": {
            "focus_fixture": focus_added,
            "plan_seed": plan_seed_added,
            "technical_seed": technical_seed_added,
            "anchor_pass": pass_added,
            "anchor_noise_fixture": noise_fixture_added,
            "anchor_noise_seed": noise_seed_added,
            "expanded_regression": expanded_added,
            "recall_regression": recall_added,
        },
        "action_counts": dict(action_counts),
        "task_counts": dict(task_counts),
        "validation": validation,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build prod-extraction-v5 suite-focused micro mix (Phase 5 plan; not v4 fail-copy)."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--focus-suite", choices=sorted(FOCUS_SUITE_FIXTURES), default="plan_chunks")
    parser.add_argument("--direct-probes", type=Path, default=DEFAULT_DIRECT_PROBES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest = build_prod_extraction_v5_suite_micro(
        args.output,
        focus_suite=args.focus_suite,
        direct_probes=args.direct_probes,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
