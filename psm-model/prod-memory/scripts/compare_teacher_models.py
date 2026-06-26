#!/usr/bin/env python3
"""Head-to-head OpenRouter teacher cost + quality on fixtures + codex samples."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.curriculum_sources import load_fixture_cases  # noqa: E402
from prod_memory.openrouter_teacher import (  # noqa: E402
    SYSTEM_PROMPT,
    TeacherConfig,
    _compact_for_teacher,
    _parse_teacher_json,
    build_expected_from_teacher,
)
from prod_memory.row_validation import validate_prod_row  # noqa: E402

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "z-ai/glm-5.2",
    "z-ai/glm-5",
    "z-ai/glm-4.6",
    "deepseek/deepseek-chat-v3-0324",
]

# OpenRouter $/token (live 2026-06-23)
PRICE_PER_TOKEN = {
    "openai/gpt-4o": (2.5e-6, 10e-6),
    "openai/gpt-4o-mini": (0.15e-6, 0.6e-6),
    "z-ai/glm-5.2": (0.98e-6, 3.08e-6),
    "z-ai/glm-5": (0.6e-6, 1.92e-6),
    "z-ai/glm-4.6": (0.43e-6, 1.74e-6),
    "deepseek/deepseek-chat-v3-0324": (0.2e-6, 0.77e-6),
}

FULL_RUN_ROWS = 1474


def _samples() -> list[dict]:
    out: list[dict] = []
    for case in load_fixture_cases():
        text = str(case.get("llmResponse") or "").strip()
        if text:
            out.append({
                "id": case["id"],
                "kind": "fixture",
                "text": text,
                "expect": case.get("expectAction"),
            })
    cache = PACKAGE_ROOT / "data" / "prod-teacher-cache-v4-4o.jsonl"
    codex = 0
    for line in cache.read_text(encoding="utf-8").splitlines():
        if codex >= 3:
            break
        row = json.loads(line)["row"]
        conv = row.get("input", {}).get("conversation") or []
        if conv:
            out.append({
                "id": str(row.get("id") or "")[:48],
                "kind": "codex",
                "text": str(conv[0].get("content") or ""),
                "expect": None,
            })
            codex += 1
    return out


def _chat(config: TeacherConfig, model: str, text: str) -> tuple[str, dict]:
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"operation": "remember_llm_response", "llm_response": _compact_for_teacher(text)},
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    content = str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")
    return content, payload.get("usage") or {}


def main() -> int:
    samples = _samples()
    base = TeacherConfig.from_env()
    if not base.api_key:
        print("OPENROUTER_API_KEY required", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for model in MODELS:
        cfg = TeacherConfig(
            model=model,
            api_key=base.api_key,
            base_url=base.base_url,
            request_delay_ms=1200,
        )
        pin, pout = PRICE_PER_TOKEN[model]
        stats = {
            "cost": 0.0,
            "prompt": 0,
            "completion": 0,
            "valid": 0,
            "store": 0,
            "fixture_match": 0,
            "fixture_n": 0,
            "errors": 0,
        }
        for sample in samples:
            try:
                raw, usage = _chat(cfg, model, sample["text"])
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
                stats["prompt"] += pt
                stats["completion"] += ct
                stats["cost"] += pt * pin + ct * pout
                parsed, err = _parse_teacher_json(raw)
                expected = build_expected_from_teacher(sample["text"], parsed, parse_error=err)
                row = {
                    "id": "x",
                    "input": {
                        "operation": "remember",
                        "conversation": [{"role": "assistant", "content": sample["text"]}],
                    },
                    "expected": expected,
                }
                try:
                    validate_prod_row(row)
                    stats["valid"] += 1
                except ValueError:
                    pass
                if expected.get("action") != "ignore":
                    stats["store"] += 1
                if sample["kind"] == "fixture" and sample.get("expect"):
                    stats["fixture_n"] += 1
                    got = "ignore" if expected.get("action") == "ignore" else "store"
                    if got == sample["expect"]:
                        stats["fixture_match"] += 1
            except Exception as exc:
                stats["errors"] += 1
                print(f"ERR {model} {sample['id']}: {exc}", file=sys.stderr)

        n = len(samples)
        usd_per = stats["cost"] / n if n else 0.0
        rows.append({
            "model": model,
            "samples": n,
            "total_usd_pilot": round(stats["cost"], 4),
            "usd_per_sample": round(usd_per, 5),
            "est_1474_run_usd": round(usd_per * FULL_RUN_ROWS, 2),
            "valid_rate": round(stats["valid"] / n, 3) if n else 0,
            "store_rate": round(stats["store"] / n, 3) if n else 0,
            "fixture_action_match": f"{stats['fixture_match']}/{stats['fixture_n']}",
            "avg_prompt_tokens": round(stats["prompt"] / n) if n else 0,
            "avg_completion_tokens": round(stats["completion"] / n) if n else 0,
            "errors": stats["errors"],
        })

    report = {"samples": len(samples), "models": rows}
    out = PACKAGE_ROOT / "data" / "teacher-model-compare.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
