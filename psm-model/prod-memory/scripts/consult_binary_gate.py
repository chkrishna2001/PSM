#!/usr/bin/env python3
"""One-shot OpenRouter consult on binary gate 8/10 ceiling."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

PROMPT = """You are an ML training architect. Be direct and skeptical — no cheerleading.

## Problem
We train Qwen2.5-0.5B with LoRA for a binary gate: given assistant text (llmResponse), output exactly one word: ignore or store.

Ship bar: 9/10 classify_match on 10 prod fixtures before a second extract LoRA.

## Stuck at 8/10
Student (Qwen 0.5B + LoRA) outputs store on ALL 10 fixtures including 2 noise cases.
- noise-filler: "Okay, sure. Let me know if you need anything else." -> should ignore
- noise-meta: "I don't have any durable facts to store..." -> should ignore
All 8 store fixtures pass.

## Runs tried (model-only)
1. v5k-gate: 404 rows (fixtures + noise + locomo ignores), ~80 steps -> 8/10
2. v5k-gate-fix: 280 rows, ONLY 10 fixtures (store x20 dup, ignore x60 dup), 150 steps @ 5e-5 -> still 8/10

Training: SFT, binary system prompt, assistant label is ignore or store only.

## OpenRouter binary gate pilot (identical student prompt, temp=0)
- claude-sonnet-4: 9/10 (only technical-api -> ignore wrongly)
- gemma-3-27b-it: 8/10 (plan-01-handoff, technical-eslint wrong)
- gpt-4o: 4/10 (ignores most store handoffs; gets both noise right)
- glm-5.2: 2/10 (broken outputs)

Pattern: strong models get noise right; 0.5B student is inverted (always-store prior).

## Constraints
- Model-only, no rule pre-filters
- 0.5B OK with gate+extract two-pass
- ~1 GPU hour budget for next experiment

## Questions
1. Why doesn't 60x ignore oversampling fix 2 trivial noise strings on 0.5B?
2. Is binary SFT wrong for teaching ignore? Better objective/format on 0.5B?
3. Distill from which teacher? Or DPO/contrastive on 10 fixtures?
4. Label mismatch in fixtures?
5. ONE concrete next experiment (data + hyperparams) most likely to hit 10/10 gate in ~1 GPU hour."""


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY required", file=sys.stderr)
        return 1
    model = os.environ.get("PROD_REVIEW_MODEL", "anthropic/claude-sonnet-4")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 3000,
            "temperature": 0.2,
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read().decode())
    text = out["choices"][0]["message"]["content"]
    path = PACKAGE_ROOT / "data" / "binary-gate-openrouter-consult.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
