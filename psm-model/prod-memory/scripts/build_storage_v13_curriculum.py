#!/usr/bin/env python3
"""Build hf-prod-storage-v13.jsonl -- PromptMix-style consistent relabeling round on top of v11.

Three changes vs v11 (all keep v11's proven reasoning-first / LR 1e-4 / ~6-epoch recipe):
1. Relabel-fix: the v7 row whose text states an actual finding ("the ledger now reflects the new
   contradiction: flat comparison ignores small decisive evidence; veto-style separated
   comparison overreacts to noisy evidence") was labeled IGNORE, contradicting the STORE row
   "the ledger now records H22 as rejected...". Under one consistent rule (content-in-sentence =
   store) it is STORE. This script re-renders that specific row's assistant target to store.
2. +11 STORE rows: terse decisions/findings with content-in-sentence + standalone technical
   FACTS (the two under-store patterns from v11's 100-case failures).
3. +8 IGNORE rows: pure transient infra status (pod/billing/download), the over-store pattern.

All new rows are excluded from the 100-case gate and v11 training (verified in
build_storage_v13_rows.py provenance).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.build_storage_v13_rows import build_storage_v13_new_rows  # noqa: E402
from prod_memory.hf_prompts import compact_storage_json, row_messages  # noqa: E402
from prod_memory.row_validation import write_jsonl  # noqa: E402

V11_CURRICULUM = PACKAGE_ROOT / "data" / "hf-prod-storage-v11.jsonl"
DEFAULT_OUT = PACKAGE_ROOT / "data" / "hf-prod-storage-v13.jsonl"

# the mislabeled row's user text fragment (unique) and its corrected STORE target
MISLABELED_FRAGMENT = "flat comparison ignores small decisive evidence"
CORRECTED_STORE = {
    "action": "store_episodic",
    "memory": {
        "content": "The ledger now reflects the new contradiction: flat comparison ignores small "
        "decisive evidence; veto-style separated comparison overreacts to noisy evidence.",
        "type": "episodic",
    },
    "facts": [],
    "indexables": [],
    "reasoning": "The sentence states an actual finding (the two failure modes of the comparison "
    "rule), so it is durable and worth storing despite the 'ledger now reflects' framing.",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _relabel_if_mislabeled(row: dict) -> tuple[dict, bool]:
    for msg in row.get("messages", []):
        if msg.get("role") == "user" and MISLABELED_FRAGMENT in msg.get("content", ""):
            new_msgs = []
            for m in row["messages"]:
                if m["role"] == "assistant":
                    new_msgs.append({"role": "assistant", "content": compact_storage_json(CORRECTED_STORE)})
                else:
                    new_msgs.append(m)
            return {**row, "messages": new_msgs}, True
    return row, False


def main() -> int:
    base = _load(V11_CURRICULUM)
    relabeled = 0
    fixed = []
    for r in base:
        r2, did = _relabel_if_mislabeled(r)
        relabeled += int(did)
        fixed.append(r2)
    print(f"relabel-fix applied to {relabeled} row(s) (expected 1)")

    new_rows = build_storage_v13_new_rows()
    new_hf = [{
        "id": r["id"], "task": "storage",
        "messages": row_messages(r, output_format="json"),
        "source": r.get("source"),
    } for r in new_rows]

    rows = fixed + new_hf
    write_jsonl(DEFAULT_OUT, rows)
    manifest = {
        "profile": "hf-prod-storage-v13",
        "output": str(DEFAULT_OUT),
        "total_rows": len(rows),
        "v11_base_rows": len(base),
        "relabeled_rows": relabeled,
        "new_rows": len(new_hf),
        "change": "PromptMix consistent relabel + 11 store (terse-decision/technical-fact) + 8 ignore (transient status)",
    }
    DEFAULT_OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
