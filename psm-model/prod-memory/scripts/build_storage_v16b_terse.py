#!/usr/bin/env python3
"""Build hf-prod-storage-v16b.jsonl: the teacher's BI-DIRECTIONAL decisions (from the v16 distill
cache) applied on v11's TERSE, emittable extraction format.

Why: v16 (full teacher distillation with the teacher's verbose extraction — multi-fact, 5-indexable
promote_semantic) regressed to 0.72 on the gate, but the diagnosis showed 14 of 21 false-ignores were
PARSE FAILURES (Qwen-0.5B can't emit that verbose schema -> malformed JSON -> fail-safe ignore), not
under-store decisions. If those had parsed, v16 would have been ~0.87. So the teacher's DECISIONS are
good; the verbose extraction is what broke generation (train_loss 0.25->0.86, parse_valid 0.99->0.85).

v16b isolates the decision benefit from the extraction-complexity penalty:
- store<->store, ignore<->ignore (teacher agrees): keep the ORIGINAL v11 row verbatim (emittable at 0.82).
- store->ignore flip: replace assistant with a terse null-ignore decision.
- ignore->store flip: build a TERSE store (teacher action + reasoning + one-line memory content; no
  facts/indexables) so the 0.5B can actually emit it.
Non-auto (hand-labeled) rows: unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
from prod_memory.hf_prompts import compact_storage_json  # noqa: E402

SRC = ROOT / "data" / "hf-prod-storage-v11.jsonl"
CACHE = ROOT / "results" / "v16-distill-cache.jsonl"
OUT = ROOT / "data" / "hf-prod-storage-v16b.jsonl"
AUTO_SOURCES = ("prod_extraction_v1", "prod_extraction_v3_teacher")
STORE_ACTIONS = {"store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store"}


def _load_cache() -> dict:
    c = {}
    for line in CACHE.open(encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            c[d["id"]] = d["decision"]
    return c


def _orig_action(row):
    for m in row["messages"]:
        if m["role"] == "assistant" and m["content"].startswith("{"):
            try:
                return json.loads(m["content"]).get("action")
            except Exception:
                return None
    return None


def _terse_memory(mem, action):
    """Reduce the teacher's memory to a terse, emittable v11-style dict."""
    content = ""
    if isinstance(mem, dict):
        content = (mem.get("content") or "").strip()
    elif isinstance(mem, str):
        content = mem.strip()
    if not content:
        return None
    mtype = "semantic" if action == "promote_semantic" else "episodic"
    return {"confidence": 0.85, "content": content, "decay_rate": 0.02,
            "emotional_weight": 0.2, "strength": 0.82, "tags": [], "type": mtype}


def main() -> int:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    cache = _load_cache()
    out_rows = []
    counts = {"same_kept": 0, "store_to_ignore": 0, "ignore_to_store": 0, "no_cache_kept": 0, "nonauto": 0}
    for i, row in enumerate(rows):
        src = row.get("source", "")
        if not any(src.startswith(s) for s in AUTO_SOURCES):
            out_rows.append(row)
            counts["nonauto"] += 1
            continue
        rid = row.get("id", f"row{i}")
        dec = cache.get(rid)
        if dec is None:
            out_rows.append(row)
            counts["no_cache_kept"] += 1
            continue
        orig = _orig_action(row)
        orig_store = orig in STORE_ACTIONS
        new_store = dec.get("action") in STORE_ACTIONS
        if orig_store == new_store:
            out_rows.append(row)  # teacher agrees on direction -> keep emittable v11 row
            counts["same_kept"] += 1
            continue
        if orig_store and not new_store:  # store -> ignore
            terse = {"action": "ignore", "memory": None, "facts": [], "indexables": [],
                     "reasoning": dec.get("reasoning") or "Transient content with no durable fact, decision, or result to store."}
            counts["store_to_ignore"] += 1
        else:  # ignore -> store
            mem = _terse_memory(dec.get("memory"), dec.get("action"))
            if mem is None:
                out_rows.append(row)  # teacher store but no usable content -> keep original ignore
                counts["no_cache_kept"] += 1
                continue
            terse = {"action": dec.get("action"), "memory": mem, "facts": [], "indexables": [],
                     "reasoning": dec.get("reasoning") or "Durable technical fact/finding worth storing."}
            counts["ignore_to_store"] += 1
        new_msgs = [m if m["role"] != "assistant" else
                    {"role": "assistant", "content": compact_storage_json(terse)} for m in row["messages"]]
        out_rows.append({**row, "messages": new_msgs, "source": src + "+v16b"})

    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    store_after = sum(1 for r in out_rows for m in r["messages"] if m["role"] == "assistant"
                      and m["content"].startswith("{") and json.loads(m["content"]).get("action") in STORE_ACTIONS)
    manifest = {"output": str(OUT), "total_rows": len(out_rows), "counts": counts,
                "store_rows_after": store_after}
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
