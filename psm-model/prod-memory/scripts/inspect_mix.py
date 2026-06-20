"""Print prod-extraction-v1 mix breakdown for review before training."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIX = ROOT / "data" / "prod-extraction-v1.jsonl"


def main() -> int:
    sys.path.insert(0, str(ROOT.parent.parent / "src"))
    sys.path.insert(0, str(ROOT.parent))
    from prod_memory.curriculum_sources import build_primary_source_rows
    from psm_model.data.rows import infer_row_task

    rows = [json.loads(line) for line in MIX.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"FILE: {MIX}")
    print(f"ROWS: {len(rows)}")
    print()

    print("TASK MIX:")
    for task, count in Counter(infer_row_task(r) for r in rows).most_common():
        print(f"  {task}: {count}")
    print()

    print("BUCKET MIX (id prefix):")
    for prefix, count in Counter(str(r.get("id", "")).split(":")[0] for r in rows).most_common():
        print(f"  {prefix}: {count}")
    print()

    print("PRIMARY SEEDS (19 rows before copy multiplier):")
    for seed in build_primary_source_rows():
        exp = seed.get("expected") or {}
        mem = ((exp.get("memory") or {}).get("content") or "")[:70]
        print(f"  {seed['id']:35} {str(exp.get('action')):18} {mem}")
    print()

    plan = next(r for r in rows if r["id"].startswith("plan:0:fixture-plan-01"))
    print("SAMPLE PLAN ROW (input → label):")
    conv = plan["input"]["conversation"]
    text = conv[0]["content"] if isinstance(conv, list) else conv
    print("  INPUT:", text[:300].replace("\n", " "))
    print("  LABEL:", plan["expected"]["memory"]["content"])
    print("  INDEXABLES:", json.dumps(plan["expected"].get("indexables"), indent=2)[:400])
    print()

    wf = next(r for r in rows if "workflow-review-pr" in r["id"])
    print("SAMPLE WORKFLOW ROW:")
    print("  INPUT:", wf["input"]["conversation"][0]["content"][:200])
    print("  LABEL:", wf["expected"]["memory"]["content"][:200])
    idx = wf["expected"].get("indexables") or []
    if idx:
        print("  WORKFLOW KEY:", idx[0].get("key"), "STEPS:", idx[0].get("steps"))
    print()

    recall = next(r for r in rows if r["id"].startswith("prod-recall:0:"))
    print("SAMPLE RECALL ROW (regression, not storage):")
    print("  id:", recall["id"])
    print("  task:", infer_row_task(recall))
    print("  input keys:", list((recall.get("input") or {}).keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
