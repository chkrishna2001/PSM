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
from prod_memory.curriculum_sources import build_fixture_rows, build_noise_rows
from prod_memory.row_validation import validate_prod_row, validate_prod_rows, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "prod-extraction-v4.jsonl"
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_DIRECT_PROBES = REPO_ROOT / "psm-model" / "data" / "probes" / "direct_probes.jsonl"

# ponytail: fixed from prod-grounding-065000.json — re-run builder after eval shifts failures
FAILING_FIXTURE_IDS = frozenset({
    "fixture-plan-01-handoff",
    "fixture-plan-02-chunking",
    "fixture-cursor-02-debug",
    "fixture-workflow-review-pr",
    "fixture-workflow-runpod",
    "fixture-technical-eslint",
    "fixture-technical-api",
})
PASSING_FIXTURE_IDS = frozenset({"fixture-cursor-01-summary"})
NOISE_FIXTURE_IDS = frozenset({"fixture-noise-filler", "fixture-noise-meta"})

PROD_EXTRACTION_V4_PROFILE: dict[str, int] = {
    "fail_copies": 40,
    "pass_copies": 5,
    "noise_copies": 15,
    "expanded_copies": 2,
    "recall_copies": 10,
}


def build_prod_extraction_v4_fixture_repair(
    output: Path,
    *,
    direct_probes: Path | None = None,
    profile: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    copies = profile or PROD_EXTRACTION_V4_PROFILE
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    fixtures = build_fixture_rows()
    for row in fixtures:
        validate_prod_row(row)

    fail_rows = [row for row in fixtures if row["id"] in FAILING_FIXTURE_IDS]
    pass_rows = [row for row in fixtures if row["id"] in PASSING_FIXTURE_IDS]
    noise_fixture_rows = [row for row in fixtures if row["id"] in NOISE_FIXTURE_IDS]

    fail_added = _copy_primary_rows(
        fail_rows, prefix="repair-fail", copies=copies["fail_copies"], seen=seen, output=rows
    )
    pass_added = _copy_primary_rows(
        pass_rows, prefix="repair-pass", copies=copies["pass_copies"], seen=seen, output=rows
    )
    noise_fixture_added = _copy_primary_rows(
        noise_fixture_rows,
        prefix="repair-noise-fixture",
        copies=copies["noise_copies"],
        seen=seen,
        output=rows,
    )
    noise_added = _copy_primary_rows(
        build_noise_rows(),
        prefix="repair-noise",
        copies=copies["noise_copies"],
        seen=seen,
        output=rows,
    )

    direct_path = direct_probes or DEFAULT_DIRECT_PROBES
    expanded_added = 0
    if direct_path.exists():
        expanded_added = _copy_rows(
            _load_rows(direct_path),
            prefix="prod-direct",
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

    storage_rows = [row for row in rows if infer_row_task(row) == "storage"]
    validation = validate_prod_rows(storage_rows)
    if not validation["ok"]:
        raise ValueError(json.dumps(validation, indent=2))

    write_jsonl(output, rows)
    action_counts = Counter(str(row["expected"]["action"]) for row in storage_rows)
    task_counts = Counter(infer_row_task(row) for row in rows)
    manifest = {
        "profile": "prod-extraction-v4-fixture-repair",
        "seed": seed,
        "output": str(output),
        "total_rows": len(rows),
        "copies": copies,
        "failing_fixture_ids": sorted(FAILING_FIXTURE_IDS),
        "added": {
            "fixture_fail": fail_added,
            "fixture_pass": pass_added,
            "noise_fixture": noise_fixture_added,
            "noise_seed": noise_added,
            "direct_regression": expanded_added,
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
        description="Build prod-extraction-v4 fixture-repair mix (heavy copies on eval failures)."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--direct-probes", type=Path, default=DEFAULT_DIRECT_PROBES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest = build_prod_extraction_v4_fixture_repair(
        args.output,
        direct_probes=args.direct_probes,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
