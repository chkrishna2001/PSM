#!/usr/bin/env python3
"""Build hf-prod-recall-plan-v1.jsonl -- real LoCoMo-QA-derived recall_plan curriculum
(829 rows, genuinely diverse questions, see build_recall_locomo_rows.py) plus the 23
hand-written probe scenarios for coverage of edge cases the real questions don't hit
(entity/source-id/predicate lookups). Standalone adapter, not multi-tasked with storage."""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
SRC_ROOT = PACKAGE_ROOT.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prod_memory.build_binary_fixture_rows import _dup_rows  # noqa: E402
from prod_memory.build_recall_locomo_rows import build_recall_locomo_train_rows  # noqa: E402
from prod_memory.hf_prompts import row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402
from psm_model.generate_recall_curriculum import build_recall_probe_rows  # noqa: E402

DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-recall-plan-v1.jsonl"

# 2026-07-10 finding: the v1 checkpoint systematically under-predicts target_tables on
# category 3/5 questions (defaults to ["episodic"] alone when the correct answer includes
# "semantic" too) -- 36/199 held-out misses, all in this one direction. Category 3 is also
# the smallest slice of the training set (43/829 rows, vs 110-361 for the others). Boost
# both categories' representation to correct the systematic under-inclusion via ordinary
# SFT reinforcement (not DPO -- consolidation's DPO attempt this session regressed badly
# from a narrow/homogeneous contrastive set; this fix stays in the proven-safe SFT lane).
_BOOST_CATEGORIES = {"3": 4, "5": 2}


def main() -> int:
    rows = list(build_recall_locomo_train_rows())
    rows.extend(build_recall_probe_rows())

    for category, copies in _BOOST_CATEGORIES.items():
        boosted = [r for r in rows if str(r.get("source", "")).endswith(f"category-{category}")]
        if boosted:
            rows.extend(_dup_rows(boosted, prefix=f"recallboost{category}", copies=copies))

    hf_rows = []
    for row in rows:
        hf_rows.append({
            "id": row["id"],
            "task": row.get("task") or row["input"].get("operation"),
            "messages": row_messages(row, output_format="json"),
            "source": row.get("source"),
        })
    write_jsonl(DEFAULT_OUT, hf_rows)
    print({"total_rows": len(hf_rows), "output": str(DEFAULT_OUT)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
