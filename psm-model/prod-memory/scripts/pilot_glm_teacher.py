#!/usr/bin/env python3
"""Short fixture-only teacher pilot (default: z-ai/glm-5.2)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from prod_memory.curriculum_sources import load_fixture_cases
from prod_memory.grounding import is_grounded_in_source, stored_text_from_decision
from prod_memory.openrouter_teacher import (
    SYSTEM_PROMPT,
    TeacherConfig,
    _compact_for_teacher,
    _parse_teacher_json,
    build_expected_from_teacher,
)
from prod_memory.row_validation import validate_prod_row

PRICE_PER_TOKEN = {
    "openai/gpt-4o": (2.5e-6, 10e-6),
    "z-ai/glm-5.2": (0.98e-6, 3.08e-6),
    "liquid/lfm-2.5-1.2b-thinking:free": (0.0, 0.0),
    "google/gemma-4-31b-it:free": (0.0, 0.0),
    "google/gemma-4-31b-it": (0.12e-6, 0.35e-6),
}


def _chat(config: TeacherConfig, model: str, text: str, *, json_mode: bool = True) -> tuple[str, dict]:
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    payload: dict = {
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
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_err = ""
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode())
            content = str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")
            return content, payload.get("usage") or {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_err = f"HTTP {exc.code}: {err_body[:300]}"
            if exc.code == 429 and attempt < 7:
                time.sleep(min(120, 10 * (2**attempt)))
                continue
            raise RuntimeError(last_err) from exc
    raise RuntimeError(last_err or "chat failed")


def _eval_case(case: dict, expected: dict) -> dict:
    text = str(case.get("llmResponse") or "")
    want = str(case.get("expectAction") or "store")
    got_action = str(expected.get("action") or "ignore")
    got_bin = "ignore" if got_action == "ignore" else "store"
    stored = stored_text_from_decision(expected)
    grounded = bool(stored and is_grounded_in_source(text, stored))
    row = {
        "id": f"pilot-{case['id']}",
        "input": {
            "operation": "remember_llm_response",
            "conversation": [{"role": "assistant", "content": text}],
        },
        "expected": expected,
    }
    valid = True
    valid_err = ""
    try:
        validate_prod_row(row)
    except ValueError as exc:
        valid = False
        valid_err = str(exc)
    return {
        "id": case["id"],
        "suite": case.get("suite"),
        "expect_action": want,
        "got_action": got_action,
        "action_match": got_bin == want,
        "grounded": grounded,
        "valid": valid,
        "valid_error": valid_err,
        "memory_preview": (stored or "")[:160],
        "facts": len(expected.get("facts") or []),
        "reasoning": str(expected.get("reasoning") or "")[:120],
    }


def pilot(model: str, *, json_mode: bool = True, delay_ms: int = 800) -> dict:
    base = TeacherConfig.from_env(model=model)
    if not base.api_key:
        raise SystemExit("OPENROUTER_API_KEY required")

    cases = list(load_fixture_cases())
    pin, pout = PRICE_PER_TOKEN.get(model, (0.0, 0.0))
    results: list[dict] = []
    total_cost = 0.0
    prompt_tok = completion_tok = 0

    for case in cases:
        text = str(case.get("llmResponse") or "").strip()
        cfg = TeacherConfig(model=model, api_key=base.api_key, base_url=base.base_url, request_delay_ms=delay_ms)
        try:
            raw, usage = _chat(cfg, model, text, json_mode=json_mode)
        except Exception as exc:
            results.append({
                "id": case["id"],
                "suite": case.get("suite"),
                "expect_action": case.get("expectAction"),
                "got_action": "error",
                "action_match": False,
                "grounded": False,
                "valid": False,
                "valid_error": str(exc)[:200],
                "memory_preview": "",
                "facts": 0,
                "reasoning": "",
                "parse_error": str(exc)[:200],
                "raw_preview": "",
                "api_error": True,
            })
            continue
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        prompt_tok += pt
        completion_tok += ct
        total_cost += pt * pin + ct * pout
        parsed, err = _parse_teacher_json(raw)
        expected = build_expected_from_teacher(text, parsed, parse_error=err)
        row = _eval_case(case, expected)
        row["parse_error"] = err
        row["raw_preview"] = raw[:280]
        results.append(row)

    n = len(cases)
    summary = {
        "model": model,
        "fixtures": n,
        "action_match": sum(1 for r in results if r["action_match"]),
        "grounded_store": sum(1 for r in results if r["got_action"] != "ignore" and r["grounded"]),
        "valid_rate": round(sum(1 for r in results if r["valid"]) / n, 3),
        "store_count": sum(1 for r in results if r["got_action"] != "ignore"),
        "ignore_count": sum(1 for r in results if r["got_action"] == "ignore"),
        "total_usd": round(total_cost, 5),
        "usd_per_fixture": round(total_cost / n, 5),
        "avg_prompt_tokens": round(prompt_tok / n),
        "avg_completion_tokens": round(completion_tok / n),
        "cases": results,
    }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="z-ai/glm-5.2")
    parser.add_argument("--compare", default="openai/gpt-4o", help="Optional second model (re-runs fixtures).")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--no-json-mode", action="store_true", help="Skip response_format json_object (thinking models).")
    parser.add_argument("--delay-ms", type=int, default=0, help="Pause between API calls (default 800; 4000 for :free models).")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "data" / "glm-5.2-fixture-pilot.json")
    args = parser.parse_args()
    json_mode = not args.no_json_mode
    if ":thinking" in args.model and json_mode:
        json_mode = False
    delay_ms = args.delay_ms or (4000 if ":free" in args.model else 800)

    report = pilot(args.model, json_mode=json_mode, delay_ms=delay_ms)
    if not args.no_compare and args.compare:
        other = pilot(args.compare, json_mode=not (":thinking" in args.compare), delay_ms=delay_ms)
        report["compare"] = {
            "model": other["model"],
            "action_match": other["action_match"],
            "grounded_store": other["grounded_store"],
            "valid_rate": other["valid_rate"],
            "store_count": other["store_count"],
            "ignore_count": other["ignore_count"],
            "total_usd": other["total_usd"],
            "cases": other["cases"],
        }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    brief = {k: report[k] for k in (
        "model", "fixtures", "action_match", "grounded_store", "valid_rate",
        "store_count", "ignore_count", "total_usd", "usd_per_fixture",
    )}
    if "compare" in report:
        brief["compare"] = {k: report["compare"][k] for k in (
            "model", "action_match", "grounded_store", "valid_rate",
            "store_count", "ignore_count", "total_usd",
        )}
    print(json.dumps(brief, indent=2))
    print(f"detail: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
