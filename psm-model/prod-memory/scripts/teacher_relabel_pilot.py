#!/usr/bin/env python3
"""Teacher-relabel PILOT: use Cloudflare Workers AI (Llama-3.3-70B) to re-label a sample of the
REAL base curriculum (prod_extraction_v1) store/ignore under one consistent durability rule, and
measure how many labels flip vs the noisy auto-labels. This validates the label-quality hypothesis
cheaply before committing to a full relabel + retrain.

Creds via the `o` password manager (keys: cloudflareaccountid, cloudflarekey) -> clipboard, same
pattern the repo uses for HF_TOKEN. Nothing is hardcoded.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "hf-prod-v5n.jsonl"
OUT = ROOT / "results" / "teacher-relabel-pilot.json"
MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

RULE = """You label whether an AI coding-agent's assistant turn contains something a persistent
MEMORY system should STORE, for a future session to reuse.

STORE if the turn states DURABLE content: a decision made, a finding/diagnosis, a result or
measurement, a bug's root cause, a reusable technical rule/constraint, or a user-stated
preference/constraint. Terse is fine — a one-line decision or fact still counts.

IGNORE if it is transient: status/next-step narration ("now let's...", "I'll check..."), running
commentary, operational mechanics (a run/pod completing, files uploaded), acknowledgements, or a
restatement of something with no new durable content. Length does NOT matter — a long transient
explanation is still IGNORE; judge DURABILITY, not richness.

Respond with ONLY compact JSON: {"action":"store"|"ignore","reasoning":"<one short sentence>"}"""


def _o_secret(key: str) -> str:
    subprocess.run(["o", key], check=False, capture_output=True)
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "(Get-Clipboard -Raw).Trim()"],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _creds() -> tuple[str, str]:
    tok = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not acct:
        acct = _o_secret("cloudflareaccountid")
    if not tok:
        tok = _o_secret("cloudflarekey")
    return tok, acct


def _turn_text(row: dict) -> str:
    for m in row["messages"]:
        if m["role"] == "user":
            t = m["content"]
            return t.split("Assistant response:", 1)[1].strip() if "Assistant response:" in t else t
    return ""


def _orig_label(row: dict) -> str:
    for m in row["messages"]:
        if m["role"] == "assistant":
            try:
                a = json.loads(m["content"]).get("action")
                return "store" if a in {"store_episodic", "promote_semantic", "update_existing", "flag_conflict", "flag_and_store"} else "ignore"
            except Exception:
                return "?"
    return "?"


def cf_label(tok: str, acct: str, text: str) -> str | None:
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": RULE}, {"role": "user", "content": text[:4000]}],
        "temperature": 0, "max_tokens": 120,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            content = data["choices"][0]["message"]["content"]
            m = re.search(r'"action"\s*:\s*"(store|ignore)"', content)
            return m.group(1) if m else None
        except Exception as e:
            if attempt == 4:
                print(f"  cf error: {e}", file=sys.stderr)
                return None
            time.sleep(3 * 2 ** attempt)
    return None


def main() -> int:
    tok, acct = _creds()
    if not tok or not acct:
        print("Missing Cloudflare creds (o: cloudflarekey / cloudflareaccountid)", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in BASE.open(encoding="utf-8") if l.strip()]
    v1 = [r for r in rows if r.get("source", "").startswith("prod_extraction_v1")]
    import random
    random.seed(7)
    # sample 100: stratified-ish (mostly store since that's the suspected over-label, some ignore)
    stores = [r for r in v1 if _orig_label(r) == "store"]
    ignores = [r for r in v1 if _orig_label(r) == "ignore"]
    sample = random.sample(stores, min(85, len(stores))) + random.sample(ignores, min(15, len(ignores)))
    random.shuffle(sample)

    results = []
    flips = 0
    store_to_ignore = 0
    for i, r in enumerate(sample):
        orig = _orig_label(r)
        text = _turn_text(r)
        new = cf_label(tok, acct, text)
        if new is None:
            continue
        flip = new != orig
        flips += int(flip)
        if orig == "store" and new == "ignore":
            store_to_ignore += 1
        results.append({"orig": orig, "teacher": new, "flip": flip, "text": text[:160]})
        if i % 20 == 0:
            print(f"labeled {i}/{len(sample)} | flips so far {flips}", flush=True)

    n = len(results)
    summary = {
        "model": MODEL,
        "n_labeled": n,
        "flip_rate": round(flips / n, 3) if n else 0,
        "store->ignore_flips": store_to_ignore,
        "store->ignore_rate": round(store_to_ignore / n, 3) if n else 0,
        "orig_store_frac": round(sum(1 for x in results if x["orig"] == "store") / n, 3) if n else 0,
        "teacher_store_frac": round(sum(1 for x in results if x["teacher"] == "store") / n, 3) if n else 0,
    }
    OUT.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nSample store->ignore flips (auto-labeled store, teacher says ignore):")
    for x in results:
        if x["orig"] == "store" and x["teacher"] == "ignore":
            print(f"  {x['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
