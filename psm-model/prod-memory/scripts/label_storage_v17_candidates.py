#!/usr/bin/env python3
"""Teacher-label the mined v17 augmentation candidates (ops + technical-fact turns).

Asks for a TERSE decision only: action + one-line memory_content + reasoning. We deliberately do NOT
ask for the verbose schema (facts[]/indexables[]) — v16 proved Qwen-0.5B cannot emit that reliably
(parse_valid 0.99->0.85, gate 0.72); v16b's terse format scored 0.84.

Provider fallback (free before paid) is handled by prod_memory.teacher_client.TeacherClient.
Resumable via results/v17-label-cache.jsonl.

Usage: python scripts/label_storage_v17_candidates.py [categories] [limit]
       categories default "ops,fact"
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prod_memory.teacher_client import AllProvidersExhausted, TeacherClient, complete_json  # noqa: E402

CAND = ROOT / "results" / "v17-candidate-turns.jsonl"
CACHE = ROOT / "results" / "v17-label-cache.jsonl"

RULE = """You decide whether an AI coding-agent assistant turn contains something a persistent MEMORY
system should STORE for a future session to reuse.

STORE if the turn states DURABLE content: a decision made, a finding/diagnosis, a result or
measurement, a bug's root cause, a reusable technical rule/fact/constraint, or a user preference.
IGNORE if it is transient: status/progress narration, operational mechanics (a run/commit/upload/pod
completing, downloading, deploying, being verified), acknowledgements, greetings, clarifying
questions, or a restatement with no new durable content.
Length does NOT matter -- judge DURABILITY, not richness. A long status report is still IGNORE.
A one-line technical fact IS store.

Respond ONLY compact JSON, no markdown, no prose:
  ignore -> {"action":"ignore","reasoning":"<one short sentence>"}
  store  -> {"action":"store_episodic"|"promote_semantic","memory_content":"<ONE grounded sentence>","reasoning":"<one short sentence>"}
Use promote_semantic for reusable general technical knowledge; store_episodic for project-specific
events/decisions/results. memory_content must be ONE sentence grounded in the turn."""

VALID = {"ignore", "store_episodic", "promote_semantic"}


def main() -> int:
    cats = set((sys.argv[1] if len(sys.argv) > 1 else "ops,fact").split(","))
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    rows = [json.loads(l) for l in CAND.open(encoding="utf-8") if l.strip()]
    todo = [r for r in rows if r.get("category") in cats]
    cache = {}
    if CACHE.exists():
        for l in CACHE.open(encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                cache[d["source"]] = d
    pending = [r for r in todo if r["source"] not in cache]
    if limit:
        pending = pending[:limit]
    print(f"candidates {len(todo)} (cats {sorted(cats)}) | cached {len(cache)} | to label {len(pending)}",
          flush=True)

    tc = TeacherClient(min_interval_s=0.15)
    cf = CACHE.open("a", encoding="utf-8")
    done = fail = 0
    try:
        for i, r in enumerate(pending):
            j = complete_json(tc, r["turn_text"][:3000], system=RULE, max_tokens=220)
            if not j or j.get("action") not in VALID:
                fail += 1
                continue
            if j["action"] != "ignore" and not (j.get("memory_content") or "").strip():
                fail += 1  # store without content is unusable
                continue
            cf.write(json.dumps({"source": r["source"], "category": r["category"],
                                 "turn_text": r["turn_text"], "decision": j}, ensure_ascii=False) + "\n")
            cf.flush()
            done += 1
            if done % 50 == 0:
                print(f"labeled {done}/{len(pending)} | providers={tc.stats} | fails={fail}", flush=True)
    except AllProvidersExhausted as e:
        print(f"\nSTOPPED: {e}\nCache preserved ({done} new). Re-run later to resume.", file=sys.stderr)
    finally:
        cf.close()

    allc = {}
    for l in CACHE.open(encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            allc[d["source"]] = d
    acts = Counter(d["decision"]["action"] for d in allc.values())
    bycat = Counter(f"{d['category']}->{'ignore' if d['decision']['action']=='ignore' else 'store'}"
                    for d in allc.values())
    print(json.dumps({"total_labeled": len(allc), "new_this_run": done, "parse_fails": fail,
                      "providers_used": tc.stats, "actions": dict(acts),
                      "by_category": dict(bycat)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
