#!/usr/bin/env python3
"""Build hf-prod-storage-v16.jsonl: FULL teacher re-distillation of the real auto-labeled rows.

Why v16 (vs the v15 store->ignore-only relabel): the partial v15 gate showed store->ignore-only
cleaning is a seesaw — it removed operational over-stores but pushed the 0.5B into UNDER-storing
durable technical facts (gate stayed ~0.83). A strong teacher fixes BOTH directions. qwen3-coder-next
on Ollama Cloud (8/8 on a store+ignore probe, ~2-7s/call, free, no queue) regenerates the COMPLETE
StorageDecision (reasoning + action + memory + facts + indexables) for each auto row, so the label is
teacher-quality in both directions and the extraction is fresh — not just a flipped decision.

Policy:
- Auto-labeled rows (source prod_extraction_v1 / prod_extraction_v3_teacher): send the row's own user
  prompt to the teacher with the production JSON storage instruction; replace the assistant target with
  the teacher's decision IF it validates. On parse/validation failure or rate-limit, keep the original.
- Hand-labeled rows (storage_v6/v7/v11, calibration, fixtures): keep unchanged.

Resumable: caches the full teacher decision per row id to results/v16-distill-cache.jsonl.
Teacher via env: TEACHER_API_KEY, TEACHER_BASE_URL (default Ollama Cloud), TEACHER_MODEL.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
from prod_memory.hf_prompts import compact_storage_json  # noqa: E402
from psm_model.prompts import JSON_SYSTEM_INSTRUCTION  # noqa: E402

SRC = ROOT / "data" / "hf-prod-storage-v11.jsonl"
OUT = ROOT / "data" / "hf-prod-storage-v16.jsonl"
CACHE = ROOT / "results" / "v16-distill-cache.jsonl"
TEACHER_URL = os.environ.get("TEACHER_BASE_URL", "https://ollama.com/v1/chat/completions")
MODEL = os.environ.get("TEACHER_MODEL", "qwen3-coder-next")
MIN_INTERVAL_S = float(os.environ.get("TEACHER_MIN_INTERVAL_S", "0.3"))
AUTO_SOURCES = ("prod_extraction_v1", "prod_extraction_v3_teacher")
STORE_ACTIONS = {"store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store"}
VALID_ACTIONS = STORE_ACTIONS | {"ignore"}


def _load_cache() -> dict:
    c = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                c[d["id"]] = d
    return c


def _user_msg(row) -> str:
    for m in row["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def _norm_decision(j: dict) -> dict | None:
    """Coerce the teacher JSON into our StorageDecision shape; return None if unusable."""
    action = j.get("action")
    if action not in VALID_ACTIONS:
        return None
    reasoning = j.get("reasoning") or ""
    if action == "ignore":
        return {"action": "ignore", "memory": None, "facts": [], "indexables": [],
                "reasoning": reasoning or "No durable fact, decision, or result to store."}
    memory = j.get("memory")
    # A store decision must carry memory content; otherwise it's a malformed store — reject.
    if not memory or (isinstance(memory, dict) and not (memory.get("content") or "").strip()):
        return None
    return {"action": action, "memory": memory, "facts": j.get("facts") or [],
            "indexables": j.get("indexables") or [], "reasoning": reasoning}


def teacher_decision(tok, user_prompt):
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": JSON_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_prompt}],
        "temperature": 0, "max_tokens": 900}).encode()
    req = urllib.request.Request(TEACHER_URL, data=body, headers={
        "Authorization": f"Bearer {tok}", "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"})
    long_waits = 0
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
            content = data["choices"][0]["message"]["content"]
            try:
                j = json.loads(content)
            except Exception:
                s, e = content.find("{"), content.rfind("}")
                if s < 0 or e <= s:
                    return "__PARSEFAIL__"
                j = json.loads(content[s:e + 1])
            return _norm_decision(j) or "__PARSEFAIL__"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 3 * 2 ** attempt
                if wait > 120:
                    long_waits += 1
                    if wait > 1800 or long_waits > 3:
                        print(f"  persistent rate limit (retry-after {wait}s); stopping", file=sys.stderr)
                        return "__RATELIMIT__"
                    print(f"  429 cooldown: sleeping {wait:.0f}s", file=sys.stderr, flush=True)
                time.sleep(min(wait, 420))
            elif e.code in (401, 403):
                print(f"  auth/access error {e.code}: {e.read()[:150]}", file=sys.stderr)
                return "__FATAL__"
            elif attempt == 7:
                return "__PARSEFAIL__"
            else:
                time.sleep(3 * 2 ** attempt)
        except Exception:
            if attempt == 7:
                return "__PARSEFAIL__"
            time.sleep(3 * 2 ** attempt)
    return "__PARSEFAIL__"


def main() -> int:
    partial = bool(os.environ.get("PARTIAL_FROM_CACHE"))
    tok = (os.environ.get("TEACHER_API_KEY") or os.environ.get("OLLAMA_API_KEY") or "").strip()
    if not tok and not partial:
        print("Missing TEACHER_API_KEY", file=sys.stderr)
        return 1
    print(f"teacher: {MODEL} @ {TEACHER_URL} | partial={partial}", flush=True)
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    cache = _load_cache()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cf = CACHE.open("a", encoding="utf-8")

    out_rows = []
    stopped = False
    counts = {"relabeled": 0, "from_teacher": 0, "kept_original_fail": 0, "kept_original_uncalled": 0}
    changed = {"store_to_ignore": 0, "ignore_to_store": 0, "same": 0}
    last_call = 0.0
    for i, row in enumerate(rows):
        rid = row.get("id", f"row{i}")
        src = row.get("source", "")
        is_auto = any(src.startswith(s) for s in AUTO_SOURCES)
        if not is_auto:
            out_rows.append(row)
            continue
        counts["relabeled"] += 1
        # original action
        orig_action = None
        for m in row["messages"]:
            if m["role"] == "assistant":
                try:
                    orig_action = json.loads(m["content"]).get("action")
                except Exception:
                    pass
        orig_store = orig_action in STORE_ACTIONS

        if rid in cache:
            dec = cache[rid]["decision"]
        elif partial or stopped:
            out_rows.append(row)
            counts["kept_original_uncalled"] += 1
            continue
        else:
            wait = MIN_INTERVAL_S - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            dec = teacher_decision(tok, _user_msg(row))
            last_call = time.time()
            if dec == "__FATAL__":
                print("fatal teacher error; stopping", file=sys.stderr)
                stopped = True
                out_rows.append(row)
                counts["kept_original_uncalled"] += 1
                continue
            if dec == "__RATELIMIT__":
                stopped = True
                out_rows.append(row)
                counts["kept_original_uncalled"] += 1
                continue
            if dec in ("__PARSEFAIL__",) or not isinstance(dec, dict):
                out_rows.append(row)
                counts["kept_original_fail"] += 1
                continue
            cf.write(json.dumps({"id": rid, "decision": dec}) + "\n")
            cf.flush()

        if not isinstance(dec, dict):
            out_rows.append(row)
            counts["kept_original_fail"] += 1
            continue
        counts["from_teacher"] += 1
        new_store = dec["action"] in STORE_ACTIONS
        if orig_store and not new_store:
            changed["store_to_ignore"] += 1
        elif not orig_store and new_store:
            changed["ignore_to_store"] += 1
        else:
            changed["same"] += 1
        new_msgs = [m if m["role"] != "assistant" else
                    {"role": "assistant", "content": compact_storage_json(dec)}
                    for m in row["messages"]]
        out_rows.append({**row, "messages": new_msgs, "source": src + "+distill"})
        if i % 100 == 0:
            print(f"processed {i}/{len(rows)} | from_teacher {counts['from_teacher']} | "
                  f"s->i {changed['store_to_ignore']} i->s {changed['ignore_to_store']}", flush=True)

    cf.close()
    if stopped and not partial:
        newly = sum(1 for _ in CACHE.open(encoding="utf-8"))
        print(f"\nSTOPPED early (rate limit / fatal). Cache now {newly} rows (resumable). "
              f"NOT writing v16 curriculum yet.", file=sys.stderr)
        return 2
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    store_after = sum(1 for r in out_rows for m in r["messages"] if m["role"] == "assistant"
                      and (json.loads(m["content"]).get("action") in STORE_ACTIONS
                           if m["content"].startswith("{") else False))
    manifest = {"output": str(OUT), "total_rows": len(out_rows), "teacher": MODEL,
                "partial": partial or counts["kept_original_uncalled"] > 0,
                "counts": counts, "changed": changed, "store_rows_after": store_after}
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
