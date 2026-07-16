#!/usr/bin/env python3
"""Mine candidate turns for the storage-v17 AUGMENTATION set, targeting v16b's two error modes.

v16b (0.84) errors break into:
  - ~5 operational-status OVER-stores ("adapter verified on HF", "commits landed, pull and delete").
    The curriculum has only ~22 operational-status->ignore examples -> the bucket is nearly empty.
  - ~7 technical-fact UNDER-stores ("DuckDB cannot query XML", "overturns the hypothesis").
Plus the curriculum prior is 79% store vs the gate's 50/50 -> the model learned to over-store.

So: mine OPERATIONAL-STATUS turns (from agentic Codex sessions) and TECHNICAL-FACT turns (from
untouched ChatGPT technical Q&A), then teacher-label them and rebalance the prior.

CONTAMINATION GUARDS (critical — the 100-case gate must stay held-out):
  1. Skip the exact sources used to build the gate (the extractor's 10 ChatGPT files / 5 codex rollouts).
  2. Drop any candidate whose normalized text overlaps a gate case's llmResponse.
  3. Drop any candidate already present in the v16b curriculum (dedup).
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
GATE = ROOT / "fixtures" / "holdout-coding-agent-cases.json"
CURRICULUM = ROOT / "data" / "hf-prod-storage-v16b.jsonl"
OUT = ROOT / "results" / "v17-candidate-turns.jsonl"
CHATGPT_DIR = Path.home() / "Downloads/training-data/chatgpt_chats"
CODEX_GLOB = "*/*/*/rollout-*.jsonl"

# reuse the existing extractor's parsers (and its source lists, to EXCLUDE them)
spec = importlib.util.spec_from_file_location("cand", SCRIPTS / "extract_coding_agent_candidates.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

# Signatures for the two target categories
OPS = re.compile(r"\b(completed|finished|uploaded|synced|landed|deploying|deployed|downloaded|pulled|"
                 r"pushed|committed|verified|now running|will report|in the background|kicked off|"
                 r"started the|re-?running|deleting the pod|checkpoint-\d+)\b", re.I)
FACT = re.compile(r"\b(cannot|can't|does not|doesn't|is not|isn't|there is no|requires|must|always|never|"
                  r"because|the reason|root cause|turns out|actually|means that|so that|provides|supports)\b", re.I)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def _shingles(t: str, n: int = 8) -> set:
    w = _norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main() -> int:
    # --- guard 1: gate sources to exclude ---
    gate_chatgpt = set(cand.CHATGPT_TECHNICAL_FILES)
    gate_codex = {p.name for p in cand.CODEX_ROLLOUTS}
    gate_claude = {p.name for p in cand.CLAUDE_CODE_SESSIONS}
    print(f"excluding gate sources: {len(gate_chatgpt)} chatgpt, {len(gate_codex)} codex, "
          f"{len(gate_claude)} claude-code", flush=True)

    # --- guard 2: gate case texts ---
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    gate_texts = [c.get("llmResponse") or "" for c in gate["cases"]]
    gate_shingles = set()
    for t in gate_texts:
        gate_shingles |= _shingles(t)
    gate_norm = {_norm(t) for t in gate_texts}

    # --- guard 3: existing curriculum turn texts ---
    seen_curric = set()
    for line in CURRICULUM.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for m in r["messages"]:
            if m["role"] == "user":
                t = m["content"]
                t = t.split("Assistant response:", 1)[1] if "Assistant response:" in t else t
                seen_curric.add(_norm(t)[:400])

    candidates = []
    # ---- Codex sessions (operational-status rich), excluding gate's ----
    for p in sorted((Path.home() / ".codex/sessions").glob(CODEX_GLOB)):
        if p.name in gate_codex:
            continue
        try:
            candidates.extend(cand.extract_codex(p))
        except Exception:
            pass
    # ---- ChatGPT technical Q&A (technical-fact rich), excluding gate's 10 ----
    for p in sorted(CHATGPT_DIR.glob("*.md")):
        if p.name in gate_chatgpt:
            continue
        try:
            candidates.extend(cand.extract_chatgpt_md(p))
        except Exception:
            pass

    kept, drop_gate, drop_dup, drop_short = [], 0, 0, 0
    for c in candidates:
        t = c.get("turn_text") or ""
        if len(t.split()) < 12:
            drop_short += 1
            continue
        n = _norm(t)
        if n in gate_norm or (len(_shingles(t) & gate_shingles) >= 2):
            drop_gate += 1
            continue
        if n[:400] in seen_curric:
            drop_dup += 1
            continue
        seen_curric.add(n[:400])
        cat = "ops" if OPS.search(t[:1200]) else ("fact" if FACT.search(t[:1200]) else "other")
        kept.append({**c, "category": cat})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    from collections import Counter
    cats = Counter(c["category"] for c in kept)
    print(json.dumps({
        "raw_candidates": len(candidates), "kept": len(kept),
        "dropped_gate_overlap": drop_gate, "dropped_dup_in_curriculum": drop_dup,
        "dropped_too_short": drop_short, "by_category": dict(cats), "out": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
