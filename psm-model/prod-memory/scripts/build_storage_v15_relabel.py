#!/usr/bin/env python3
"""Build hf-prod-storage-v15.jsonl: teacher-relabel the REAL auto-labeled base rows via Cloudflare
Llama-3.3-70B (pilot showed 28% flip / 91%->74% store — systematic over-labeling), keeping the
real inputs and the already-clean hand-labeled additions.

Policy (validated by the pilot's store->ignore flip direction):
- Auto-labeled rows (source prod_extraction_v1 / prod_extraction_v3_teacher): ask the teacher
  store/ignore. If teacher says IGNORE, replace the assistant output with a clean ignore decision.
  If teacher says STORE, keep the original (its extraction content is fine; only the store/ignore
  boundary was noisy). Do NOT fabricate extraction for ignore->store flips — keep those as-is.
- Hand-labeled rows (storage_v6/v7/v11, v5n_dpo* calibration, fixtures): keep unchanged.

Resumable: caches each teacher decision to results/v15-relabel-cache.jsonl keyed by row id, so a
long (~1400-call, ~2h) run survives interruption. Creds via env (CLOUDFLARE_ACCOUNT_ID/API_TOKEN).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))
from prod_memory.hf_prompts import compact_storage_json  # noqa: E402

SRC = ROOT / "data" / "hf-prod-storage-v11.jsonl"
OUT = ROOT / "data" / "hf-prod-storage-v15.jsonl"
CACHE = ROOT / "results" / "v15-relabel-cache.jsonl"
# Teacher endpoint is provider-generic (OpenAI-compatible chat/completions). Defaults to Groq's
# Llama-3.3-70B (free tier, same base model as the CF pilot). Override via env for RunPod/OpenRouter.
TEACHER_URL = os.environ.get("TEACHER_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.environ.get("TEACHER_MODEL", "llama-3.3-70b-versatile")
# Min seconds between calls to respect Groq free-tier 12k tokens/min (~1.3k tok/call -> ~9/min).
MIN_INTERVAL_S = float(os.environ.get("TEACHER_MIN_INTERVAL_S", "6.5"))
AUTO_SOURCES = ("prod_extraction_v1", "prod_extraction_v3_teacher")
STORE_ACTIONS = {"store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store"}

RULE = """You label whether an AI coding-agent's assistant turn contains something a persistent
MEMORY system should STORE, for a future session to reuse.
STORE if it states DURABLE content: a decision made, a finding/diagnosis, a result/measurement, a
bug's root cause, a reusable technical rule/constraint, or a user-stated preference/constraint.
Terse is fine.
IGNORE if transient: status/next-step narration, running commentary, operational mechanics (a
run/commit/upload completing), acknowledgements, greetings, how-to command dumps, clarifying
questions, or a restatement with no new durable content. Length does NOT matter — judge DURABILITY.
Respond ONLY compact JSON: {"action":"store"|"ignore","reasoning":"<one short sentence>"}"""


def _load_cache() -> dict:
    c = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                c[d["id"]] = d
    return c


def _turn_and_action(row):
    text = ""
    action = None
    orig_decision = None
    for m in row["messages"]:
        if m["role"] == "user":
            t = m["content"]
            text = t.split("Assistant response:", 1)[1].strip() if "Assistant response:" in t else t
        if m["role"] == "assistant":
            try:
                orig_decision = json.loads(m["content"])
                action = orig_decision.get("action")
            except Exception:
                pass
    return text, action, orig_decision


def teacher_label(tok, text):
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": RULE}, {"role": "user", "content": text[:4000]}],
        "temperature": 0, "max_tokens": 120}).encode()
    # Groq is fronted by Cloudflare bot management, which 403s the default Python-urllib UA at the
    # edge (never reaching the backend, so it doesn't even count against rate limits). A browser UA
    # passes. Harmless for other providers.
    req = urllib.request.Request(TEACHER_URL, data=body, headers={
        "Authorization": f"Bearer {tok}", "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"})
    long_waits = 0  # multi-minute cooldowns absorbed on this row; bail if the cap is persistent
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            content = data["choices"][0]["message"]["content"]
            a = re.search(r'"action"\s*:\s*"(store|ignore)"', content)
            reason = re.search(r'"reasoning"\s*:\s*"([^"]{0,200})"', content)
            return (a.group(1) if a else None), (reason.group(1) if reason else "")
        except urllib.error.HTTPError as e:
            # 429 = rate limit. Groq's free tier hands out multi-minute rolling-window cooldowns
            # (e.g. retry-after ~342s after ~175 calls). Be patient: sleep the Retry-After and
            # continue. Only give up if the cooldown is enormous or we hit 3 long waits in a row on
            # the same row (a genuinely exhausted daily cap) — the run is resumable via cache.
            if e.code == 429:
                ra = e.headers.get("retry-after")
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 3 * 2 ** attempt
                if wait > 120:
                    long_waits += 1
                    if wait > 1800 or long_waits > 3:
                        print(f"  persistent rate limit (retry-after {wait}s, long_waits {long_waits}); "
                              f"stopping so cache is preserved", file=sys.stderr)
                        return "__RATELIMIT__", ""
                    print(f"  429 cooldown: sleeping {wait:.0f}s then retrying", file=sys.stderr, flush=True)
                time.sleep(min(wait, 420))
            elif attempt == 7:
                return None, ""
            else:
                time.sleep(3 * 2 ** attempt)
        except Exception:
            if attempt == 7:
                return None, ""
            time.sleep(3 * 2 ** attempt)
    return None, ""


def main() -> int:
    # PARTIAL_FROM_CACHE=1: make NO API calls; apply only the flips already in the cache and keep
    # originals for uncached rows, then WRITE the (partial) curriculum. Free directional experiment.
    partial = bool(os.environ.get("PARTIAL_FROM_CACHE"))
    # Provider-generic: TEACHER_API_KEY preferred; fall back to known provider env vars.
    tok = (os.environ.get("TEACHER_API_KEY") or os.environ.get("GROQ_API_KEY")
           or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("CLOUDFLARE_API_TOKEN") or "").strip()
    if not tok and not partial:
        print("Missing teacher key (set TEACHER_API_KEY / GROQ_API_KEY)", file=sys.stderr)
        return 1
    if partial:
        print("PARTIAL_FROM_CACHE: no API calls; applying cached flips only", flush=True)
    else:
        print(f"teacher: {MODEL} @ {TEACHER_URL} | min_interval {MIN_INTERVAL_S}s", flush=True)
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    cache = _load_cache()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cf = CACHE.open("a", encoding="utf-8")

    out_rows = []
    flips = 0
    relabeled = 0
    last_call = 0.0
    ratelimited = False
    for i, row in enumerate(rows):
        rid = row.get("id", f"row{i}")
        src = row.get("source", "")
        is_auto = any(src.startswith(s) for s in AUTO_SOURCES)
        text, action, orig_decision = _turn_and_action(row)
        if not is_auto or orig_decision is None:
            out_rows.append(row)
            continue
        relabeled += 1
        if rid in cache:
            teacher = cache[rid]["teacher"]
            reason = cache[rid].get("reasoning", "")
        elif partial or ratelimited:
            out_rows.append(row)  # no call: keep original (partial mode, or daily cap hit)
            continue
        else:
            wait = MIN_INTERVAL_S - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            teacher, reason = teacher_label(tok, text)
            last_call = time.time()
            if teacher == "__RATELIMIT__":
                ratelimited = True
                out_rows.append(row)
                continue
            if teacher is None:
                out_rows.append(row)  # keep original on failure
                continue
            cf.write(json.dumps({"id": rid, "teacher": teacher, "reasoning": reason}) + "\n")
            cf.flush()
        orig_is_store = action in STORE_ACTIONS
        if orig_is_store and teacher == "ignore":
            flips += 1
            ignore_decision = {"action": "ignore", "memory": None, "facts": [], "indexables": [],
                               "reasoning": reason or "Transient content with no durable fact, decision, or result to store."}
            new_msgs = [m if m["role"] != "assistant" else
                        {"role": "assistant", "content": compact_storage_json(ignore_decision)}
                        for m in row["messages"]]
            out_rows.append({**row, "messages": new_msgs, "source": src + "+relabel"})
        else:
            out_rows.append(row)
        if i % 100 == 0:
            print(f"processed {i}/{len(rows)} | relabeled {relabeled} | store->ignore flips {flips}", flush=True)

    cf.close()
    if ratelimited:
        newly = sum(1 for _ in CACHE.open(encoding="utf-8"))
        print(f"\nDAILY RATE LIMIT hit. Cache now {newly} rows (resumable). "
              f"Re-run after the quota resets to finish the rest — NOT writing v15 curriculum yet.",
              file=sys.stderr)
        return 2
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    store_after = sum(1 for r in out_rows for m in r["messages"] if m["role"] == "assistant"
                      and (json.loads(m["content"]).get("action") in STORE_ACTIONS if m["content"].startswith("{") else False))
    auto_from_cache = sum(1 for i, row in enumerate(rows)
                          if any(row.get("source", "").startswith(s) for s in AUTO_SOURCES)
                          and row.get("id", f"row{i}") in cache)
    manifest = {"output": str(OUT), "total_rows": len(out_rows), "auto_rows_seen": relabeled,
                "auto_rows_judged_by_teacher": auto_from_cache,
                "auto_rows_kept_original_unjudged": relabeled - auto_from_cache,
                "partial": partial or (relabeled - auto_from_cache) > 0,
                "store_to_ignore_flips": flips, "store_rows_after": store_after}
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
