#!/usr/bin/env python3
"""Build hf-prod-storage-v17.jsonl = v16b + the teacher-labeled AUGMENTATION set.

Why: v16b (0.84) leaves 16 gate errors in two buckets the data simply didn't cover —
  * ~5 operational-status OVER-stores; the curriculum had only ~22 ops->ignore examples.
  * ~7 technical-fact UNDER-stores; the real pool yielded only 40 ignore->store flips.
The mined+labeled augmentation adds ~190 ops->ignore and ~447 fact->store examples (~10x and deep),
directly targeting both. Rows use v16b's TERSE emittable format (v16 proved verbose extraction
breaks Qwen-0.5B's JSON generation: parse_valid 0.99->0.85, gate 0.72).

Prior note: we deliberately do NOT downsample real store rows just to match the gate's synthetic
50/50 split — the gate's balance is a measurement choice, not the production prior, so chasing it
would be gate-overfitting. We add discrimination signal instead. STORE_FRAC env can cap the store
fraction if a later experiment wants it.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
from prod_memory.hf_prompts import PROD_STORAGE_USER_PREFIX, compact_storage_json  # noqa: E402
from psm_model.prompts import _system_instruction  # noqa: E402

BASE = ROOT / "data" / "hf-prod-storage-v16b.jsonl"
LABELS = ROOT / "results" / "v17-label-cache.jsonl"
OUT = ROOT / "data" / "hf-prod-storage-v17.jsonl"
STORE_ACTIONS = {"store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store"}
STORE_FRAC = float(os.environ.get("STORE_FRAC", "0"))  # 0 = keep everything


def _terse(dec: dict) -> dict | None:
    action = dec.get("action")
    reasoning = (dec.get("reasoning") or "").strip()
    if action == "ignore":
        return {"reasoning": reasoning or "Transient content with no durable fact, decision, or result to store.",
                "action": "ignore", "memory": None, "facts": [], "indexables": []}
    if action not in ("store_episodic", "promote_semantic"):
        return None
    content = (dec.get("memory_content") or "").strip()
    if not content:
        return None
    mtype = "semantic" if action == "promote_semantic" else "episodic"
    return {"reasoning": reasoning or "Durable content worth storing for a future session.",
            "action": action,
            "memory": {"confidence": 0.85, "content": content, "decay_rate": 0.02,
                       "emotional_weight": 0.2, "strength": 0.82, "tags": [], "type": mtype},
            "facts": [], "indexables": []}


def main() -> int:
    system = _system_instruction("json")
    rows = [json.loads(l) for l in BASE.open(encoding="utf-8") if l.strip()]

    aug, skipped = [], 0
    for i, line in enumerate(LABELS.open(encoding="utf-8")):
        if not line.strip():
            continue
        d = json.loads(line)
        dec = _terse(d["decision"])
        if dec is None:
            skipped += 1
            continue
        turn = (d["turn_text"] or "").strip()
        if not turn:
            skipped += 1
            continue
        aug.append({
            "id": f"v17-aug-{i:05d}",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{PROD_STORAGE_USER_PREFIX}{turn}"},
                {"role": "assistant", "content": compact_storage_json(dec)},
            ],
            "source": f"v17_augment_{d['category']}",
            "task": "storage",
        })

    # optional prior cap (off by default)
    if STORE_FRAC:
        rng = random.Random(7)
        def is_store(r):
            c = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
            return c.startswith("{") and json.loads(c).get("action") in STORE_ACTIONS
        allr = rows + aug
        st = [r for r in allr if is_store(r)]
        ig = [r for r in allr if not is_store(r)]
        keep = int(len(ig) * STORE_FRAC / (1 - STORE_FRAC))
        if keep < len(st):
            rng.shuffle(st)
            st = st[:keep]
        out_rows = st + ig
        rng.shuffle(out_rows)
    else:
        out_rows = rows + aug

    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def act(r):
        c = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
        return json.loads(c).get("action") if c.startswith("{") else None
    store_n = sum(1 for r in out_rows if act(r) in STORE_ACTIONS)
    aug_store = sum(1 for r in aug if act(r) in STORE_ACTIONS)
    manifest = {
        "output": str(OUT), "total_rows": len(out_rows),
        "base_v16b_rows": len(rows), "augmentation_rows": len(aug), "aug_skipped": skipped,
        "aug_store": aug_store, "aug_ignore": len(aug) - aug_store,
        "store_rows": store_n, "ignore_rows": len(out_rows) - store_n,
        "store_frac": round(store_n / len(out_rows), 3),
        "store_frac_before_v16b": 0.788, "store_frac_cap_applied": STORE_FRAC or None,
    }
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
