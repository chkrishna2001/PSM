#!/usr/bin/env python3
"""Probe HF storage on curated LoCoMo turns using CLI-shaped remember payloads."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.remember_cli import apply_product_boundary

from probe_locomo_hf_storage_cpu import (
    DEFAULT_DATA,
    DEFAULT_EXTRACT,
    analyze_decision,
    build_cases,
    summarize,
)
from prod_memory.eval_hf_grounding import open_hf_session
from psm_model.remember_cli import PROD_STORAGE_MAX_NEW_TOKENS
from prod_memory.hf_prompts import (
    apply_chat_prompt,
    storage_inference_messages,
    storage_llm_response_from_input,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "benchmark/locomo/results/probe-locomo-hf-storage-v5h-cli-shape-json-cpu.jsonl"


def build_cli_payload(llm_response: str) -> dict[str, Any]:
    return {
        "operation": "remember_llm_response",
        "conversation": [{"role": "assistant", "content": llm_response}],
    }


def run_probe(args: argparse.Namespace) -> int:
    cases = build_cases(args.data)
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    if not cases:
        raise SystemExit(f"No cases built from {args.data}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    session = open_hf_session(
        args.adapter_dir,
        model_key=args.model,
        device=args.device,
    )
    adapter_label = str(args.adapter_dir.resolve()) if args.adapter_dir else "base"
    records: list[dict[str, Any]] = []
    for case in cases:
        llm_response = str(case["llm_response"])
        cli_payload = build_cli_payload(llm_response)
        train_text = storage_llm_response_from_input(cli_payload)
        messages = storage_inference_messages(train_text, output_format=args.output_format)
        prompt = apply_chat_prompt(messages, session.tokenizer)
        raw = session.generate(
            train_text,
            output_format=args.output_format,
            max_new_tokens=args.max_new_tokens,
        )
        report = apply_product_boundary(raw, output_format=args.output_format)
        analysis = analyze_decision(case, report, raw)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probe": "locomo_hf_storage_cli_shape",
            "output_format": args.output_format,
            "adapter_dir": adapter_label,
            "model_key": args.model,
            "device": args.device,
            "case": case,
            "cli_payload": cli_payload,
            "train_text": train_text,
            "prompt": prompt,
            "messages": messages,
            "raw_output": raw,
            "boundary_report": {
                "parsed": report.get("parsed"),
                "repair_status": report.get("repair_status"),
                "issues": report.get("issues"),
            },
            "analysis": analysis,
        }
        records.append(record)
        print(
            f"{case['id']}: action={analysis['action']!r} "
            f"store_match={analysis['checks']['store_decision_match']} "
            f"facts={analysis['facts_count']} "
            f"temporal={analysis['checks']['has_temporal_on_memory'] or analysis['checks']['has_temporal_on_facts']}",
            flush=True,
        )

    with args.out.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(records)
    summary["input_shape"] = "cli_remember_payload_via_remember_target"
    summary["train_text_example"] = records[0]["train_text"] if records else None
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-format", default="json", choices=["json", "tagged", "at_tag"])
    parser.add_argument("--max-new-tokens", type=int, default=PROD_STORAGE_MAX_NEW_TOKENS)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
