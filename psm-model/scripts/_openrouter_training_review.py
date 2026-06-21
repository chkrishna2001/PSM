#!/usr/bin/env python3
"""One-shot OpenRouter consult on prod-memory training failures."""
from __future__ import annotations

import json
import os
import sys
import urllib.request

PROMPT = """You are an ML systems architect. Be direct and skeptical. We need an independent verdict on whether a custom 50M TinyDecoder is useless for our prod memory task.

## Product task
remember({ llmResponse }) -> store|ignore -> grounded memory.content (+ facts[] ideally)
Ship bar: 85% effective_stored on 10 prod fixtures (assistant handoffs, 600-2400 chars).
NOT gate probes (short "User: I prefer SQLite").

## 50M architecture
Custom TinyDecoder ~50M params, byte/pattern tokenizer, chosen for fast CPU inference.
Prior training was mostly gate curriculum (short user utterances) — misleading metrics.

## Empirical ladder (all on RunPod, same 3 fixtures: cursor-01-summary, cursor-02-debug, plan-01-handoff)

| Exp | Setup | Classify 3/3 | Ground 3/3 |
| A | Minimal one-line (store: fact), gate resume 058000, 800 steps | - | 0/3 |
| B | Binary only (store/ignore), gate resume, 800 steps | YES | - |
| C | Sequential: binary 800 then minimal 800 same weights | 1/3 | 0/3 |
| D | Two-pass inference: separate binary ckpt + extract ckpt (032000+1600 extract train) | YES | 0/3 (extract emits tagged gate DSL garbage) |
| E | Scratch 50M NO gate resume, v3 teacher storage ~1500 rows (p50 ~1100 chars), minimal_extract, 2500 steps, final_loss 4.0 | - | 0/3 |

Additional:
- Gate resume overfit 3 fixtures tagged/minimal: train loss 0.001, still 0/3 grounded at generation
- All gate donors (032000-058000) eval 1/10 effective_stored on full fixtures
- v6/v7 full prod trains: 0/10

## User hypothesis to evaluate
"Maybe generative extract failed only because we trained on small gate texts; scratch 50M on long llmResponse handoffs should work."

Exp E directly tests this (scratch + long v3) — failed. Overfit on 3 handoff fixtures with loss→0.001 also failed generation grounding.

## Questions
1. Is 50M **useless** for this product, or **useful for part of it** (classify only)? Give a clear verdict.
2. Is there ANY realistic 50M-only path left (scratch overfit 3 rows generative, more steps, different format) worth one more GPU hour?
3. vs 360M-500M HF model (SmolLM2-360M, Qwen2.5-0.5B): when is upgrading justified given CPU inference constraint?
4. Best architecture: single model, two-pass same size, 50M classify + extractive copy (no LM), or 50M + small HF extract?

No cheerleading. If 50M generative extract is dead, say so plainly."""


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        print("OPENROUTER_API_KEY required", file=sys.stderr)
        return 1
    model = os.environ.get("PROD_REVIEW_MODEL", "anthropic/claude-sonnet-4")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 2500,
            "temperature": 0.3,
        }
    ).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/psm",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = json.loads(resp.read())
    text = out["choices"][0]["message"]["content"]
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
