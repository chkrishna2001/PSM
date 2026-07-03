#!/usr/bin/env python3
"""Probe HF 0.6B storage on curated LoCoMo turns (write-path only).

Write-path only: store/ignore decision + extracted memory/facts/temporal.
Saves full prompts, raw outputs, and parsed decisions for review.

Device policy:
- **Local (Windows dev machine):** ``--device cpu`` + ``PSM_FORCE_CPU=1`` — no local CUDA.
- **RunPod / pod SSH:** ``--device cuda`` (or omit PSM_FORCE_CPU; set ``PSM_FORCE_CPU=0``).
  Filename says ``_cpu`` for historical local default, not a pod requirement.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from psm_model.remember_cli import PROD_STORAGE_MAX_NEW_TOKENS, apply_product_boundary

from prod_memory.eval_hf_grounding import open_hf_session, open_hf_two_pass_sessions
from prod_memory.eval_classify import binary_predicts_store
from prod_memory.grounding import (
    apply_storage_guards,
    grounding_overlap_score,
    stored_text_from_decision,
    would_model_store,
)
from prod_memory.hf_prompts import apply_chat_prompt, storage_inference_messages

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "benchmark/locomo/data/locomo10.json"
DEFAULT_EXTRACT = REPO / "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter"
DEFAULT_GATE = REPO / "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter"
DEFAULT_OUT = REPO / "benchmark/locomo/results/probe-locomo-hf-storage-tagged-cpu.jsonl"

# ponytail: hand-picked + evidence-linked cases; expand after first read
IGNORE_DIA_IDS = [
    ("conv-26", "D1:1"),   # greeting
    ("conv-30", "D1:2"),   # short ack
    ("conv-41", "D12:18"), # bye/filler (if exists)
]

PRIORITY_QUESTIONS = [
    # temporal
    ("conv-26", "When did Caroline go to the LGBTQ support group?", "D1:3"),
    ("conv-26", "When did Melanie paint a sunrise?", "D1:12"),
    # relationship / semantic-ish
    ("conv-26", "What is Caroline's identity?", "D1:9"),
    ("conv-26", "What fields would Caroline be interested in pursuing in her education?", "D1:8"),
    # episodic / event
    ("conv-26", "What did Caroline research?", "D3:4"),
    ("conv-30", "What did Gina lose?", "D1:3"),
    ("conv-41", "What did Maria do last week?", "D2:8"),
]


def quote_text(value: str | None) -> str:
    text = (value or "").strip()
    return f'"{text}"' if text else '""'


def locomo_source_timestamp(sample: dict[str, Any], session: str | None) -> str | None:
    if not session:
        return None
    conv = sample.get("conversation") or {}
    date_time = conv.get(f"{session}_date_time")
    if isinstance(date_time, str) and date_time.strip():
        return date_time.strip()
    match = re.match(r"^session_(\d+)$", session)
    if not match:
        return None
    events = (sample.get("event_summary") or {}).get(f"events_session_{match.group(1)}") or {}
    date = events.get("date")
    return date.strip() if isinstance(date, str) and date.strip() else None


def flatten_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    conv = sample.get("conversation") or {}
    turns: list[dict[str, Any]] = []
    for key in sorted(
        (k for k in conv if re.match(r"^session_\d+$", k)),
        key=lambda k: int(k.split("_")[1]),
    ):
        session_turns = conv.get(key)
        if not isinstance(session_turns, list):
            continue
        for turn in session_turns:
            if isinstance(turn, dict):
                turns.append({**turn, "session": key})
    return turns


def build_locomo_remember_text(
    sample: dict[str, Any],
    turns: list[dict[str, Any]],
    index: int,
    *,
    window_size: int = 2,
) -> str:
    turn = turns[index]
    sample_id = str(sample.get("sample_id") or "unknown")
    session = str(turn.get("session") or "")
    dia_id = str(turn.get("dia_id") or "")
    source_timestamp = locomo_source_timestamp(sample, session) or "unknown"
    window_start = max(0, index - window_size)
    nearby = turns[window_start:index]

    def prior_line(item: dict[str, Any]) -> str:
        bits = [f'{item.get("speaker") or "Unknown"} said: {quote_text(str(item.get("text") or ""))}']
        if item.get("query"):
            bits.append(f'image query: {item["query"]}')
        if item.get("blip_caption"):
            bits.append(f'image caption: {item["blip_caption"]}')
        return f'- [prior {item.get("session") or "unknown"} {item.get("dia_id") or "unknown"}] {"; ".join(bits)}'

    image_lines: list[str] = []
    if turn.get("query"):
        image_lines.append(f'Image query: {turn["query"]}')
    if turn.get("blip_caption"):
        image_lines.append(f'Image caption: {turn["blip_caption"]}')
    if turn.get("img_url"):
        image_lines.append(f'Image URLs: {", ".join(map(str, turn["img_url"]))}')

    prior_lines = [prior_line(item) for item in nearby] if nearby else ["- none"]
    lines = [
        f"Source id: {sample_id}:{dia_id}",
        f"Sample id: {sample_id}",
        f"Session: {session or 'unknown'}",
        f"Session time: {source_timestamp}",
        f'Current speaker: {turn.get("speaker") or "unknown"}',
        f'Current utterance: {quote_text(str(turn.get("text") or ""))}',
        *image_lines,
        "Previous context:",
        *prior_lines,
    ]
    return "\n".join(lines)


def find_turn_index(turns: list[dict[str, Any]], dia_id: str) -> int | None:
    for i, turn in enumerate(turns):
        if str(turn.get("dia_id") or "") == dia_id:
            return i
    return None


def build_cases(data_path: Path) -> list[dict[str, Any]]:
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    by_id = {str(s.get("sample_id")): s for s in samples if isinstance(s, dict)}
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_case(
        *,
        case_id: str,
        sample_id: str,
        dia_id: str,
        expect_store: bool,
        bucket: str,
        question: str | None = None,
        gold_answer: str | None = None,
    ) -> None:
        key = f"{sample_id}:{dia_id}"
        if key in seen:
            return
        sample = by_id.get(sample_id)
        if not sample:
            return
        turns = flatten_turns(sample)
        idx = find_turn_index(turns, dia_id)
        if idx is None:
            return
        turn = turns[idx]
        llm_response = build_locomo_remember_text(sample, turns, idx)
        seen.add(key)
        cases.append(
            {
                "id": case_id,
                "bucket": bucket,
                "sample_id": sample_id,
                "dia_id": dia_id,
                "speaker": turn.get("speaker"),
                "utterance": turn.get("text"),
                "session": turn.get("session"),
                "source_timestamp": locomo_source_timestamp(sample, str(turn.get("session") or "")),
                "expect_store": expect_store,
                "related_question": question,
                "gold_answer": gold_answer,
                "llm_response": llm_response,
            }
        )

    for sample_id, question, dia_id in PRIORITY_QUESTIONS:
        sample = by_id.get(sample_id)
        gold = None
        if sample:
            for qa in sample.get("qa") or []:
                if str(qa.get("question") or "") == question:
                    gold = qa.get("answer")
                    break
        bucket = "temporal" if any(w in question.lower() for w in ("when", "date", "year")) else "qa_evidence"
        if "identity" in question.lower() or "field" in question.lower():
            bucket = "semantic"
        add_case(
            case_id=f"{sample_id}_{dia_id.replace(':', '_')}",
            sample_id=sample_id,
            dia_id=dia_id,
            expect_store=True,
            bucket=bucket,
            question=question,
            gold_answer=str(gold) if gold is not None else None,
        )

    for sample_id, dia_id in IGNORE_DIA_IDS:
        add_case(
            case_id=f"ignore_{sample_id}_{dia_id.replace(':', '_')}",
            sample_id=sample_id,
            dia_id=dia_id,
            expect_store=False,
            bucket="ignore_noise",
        )

    return cases


def analyze_decision(case: dict[str, Any], report: dict[str, Any], raw: str) -> dict[str, Any]:
    decision = report.get("parsed") if isinstance(report.get("parsed"), dict) else {}
    memory = decision.get("memory") if isinstance(decision.get("memory"), dict) else {}
    facts = decision.get("facts") if isinstance(decision.get("facts"), list) else []
    fact_rows = [f for f in facts if isinstance(f, dict)]

    temporal_expr = memory.get("temporal_expression")
    resolved = memory.get("resolved_time")
    memory_type = memory.get("type")
    action = str(decision.get("action") or "")
    model_store = would_model_store(decision)
    guarded = apply_storage_guards(str(case["llm_response"]), decision)
    stored_text = stored_text_from_decision(decision)
    overlap = grounding_overlap_score(str(case["llm_response"]), stored_text)

    fact_temporal = sum(
        1
        for f in fact_rows
        if (f.get("temporal_expression") or f.get("resolved_time"))
    )

    checks = {
        "store_decision_match": model_store == bool(case["expect_store"]),
        "parse_ok": report.get("repair_status") == "parsed",
        "has_memory_content": bool(str(memory.get("content") or "").strip()),
        "has_facts": len(fact_rows) > 0,
        "has_temporal_on_memory": bool(temporal_expr or resolved),
        "has_temporal_on_facts": fact_temporal > 0,
        "has_semantic_type": str(memory_type or "").lower() == "semantic",
        "has_episodic_type": str(memory_type or "").lower() == "episodic",
        "content_grounded": bool(overlap.get("grounded")) if model_store else True,
        "guard_rejected": guarded.get("rejected"),
    }

    return {
        "action": action,
        "memory_type": memory_type,
        "memory_content": memory.get("content"),
        "temporal_expression": temporal_expr,
        "resolved_time": resolved,
        "facts_count": len(fact_rows),
        "facts": fact_rows,
        "indexables": decision.get("indexables") or [],
        "reasoning": decision.get("reasoning"),
        "model_would_store": model_store,
        "repair_status": report.get("repair_status"),
        "issues": report.get("issues"),
        "checks": checks,
        "stored_text": stored_text,
        "grounding_overlap": overlap,
        "guard": guarded,
    }


def run_two_pass_case(
    case: dict[str, Any],
    *,
    binary_session,
    extract_session,
    binary_max_tokens: int,
    extract_max_tokens: int,
) -> tuple[str, str, dict[str, Any]]:
    """Returns (raw_output, binary_output, boundary report)."""
    llm_response = str(case["llm_response"])
    raw_binary = binary_session.generate(
        llm_response, output_format="binary", max_new_tokens=binary_max_tokens
    )
    raw_extract = ""
    if binary_predicts_store(raw_binary):
        raw_extract = extract_session.generate(
            llm_response,
            output_format="minimal_extract",
            max_new_tokens=extract_max_tokens,
        )
        report = apply_product_boundary(raw_extract, output_format="minimal")
    else:
        report = apply_product_boundary("ignore", output_format="minimal")
    raw = raw_extract if raw_extract else raw_binary.strip()
    return raw, raw_binary.strip(), report


def run_probe(args: argparse.Namespace) -> int:
    cases = build_cases(args.data)
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    if not cases:
        raise SystemExit(f"No cases built from {args.data}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    adapter_label = str(args.adapter_dir.resolve()) if args.adapter_dir else "base"

    if args.two_pass:
        binary_session, extract_session = open_hf_two_pass_sessions(
            args.binary_adapter,
            args.extract_adapter,
            model_key=args.model,
            device=args.device,
        )
        adapter_label = (
            f"gate={args.binary_adapter.name}+extract={args.extract_adapter.name}"
        )
        tokenizer = binary_session.tokenizer
        for case in cases:
            raw, raw_binary, report = run_two_pass_case(
                case,
                binary_session=binary_session,
                extract_session=extract_session,
                binary_max_tokens=args.binary_max_tokens,
                extract_max_tokens=args.extract_max_tokens,
            )
            messages = storage_inference_messages(case["llm_response"], output_format="minimal_extract")
            prompt = apply_chat_prompt(messages, tokenizer)
            analysis = analyze_decision(case, report, raw)
            gate_open = binary_predicts_store(raw_binary)
            analysis["binary_output"] = raw_binary
            analysis["gate_open"] = gate_open
            analysis["checks"]["gate_match"] = gate_open == bool(case["expect_store"])
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "probe": "locomo_hf_storage_cpu_two_pass",
                "output_format": "minimal_extract",
                "adapter_dir": adapter_label,
                "binary_adapter": str(args.binary_adapter.resolve()),
                "extract_adapter": str(args.extract_adapter.resolve()),
                "model_key": args.model,
                "device": args.device,
                "case": case,
                "prompt": prompt,
                "messages": messages,
                "raw_output": raw,
                "binary_output": raw_binary,
                "boundary_report": {
                    "parsed": report.get("parsed"),
                    "repair_status": report.get("repair_status"),
                    "issues": report.get("issues"),
                },
                "analysis": analysis,
            }
            records.append(record)
            print(
                f"{case['id']}: gate={raw_binary!r} action={analysis['action']!r} "
                f"store_match={analysis['checks']['store_decision_match']} "
                f"facts={analysis['facts_count']}",
                flush=True,
            )
    else:
        session = open_hf_session(
            args.adapter_dir,
            model_key=args.model,
            device=args.device,
        )
        for case in cases:
            messages = storage_inference_messages(case["llm_response"], output_format=args.output_format)
            prompt = apply_chat_prompt(messages, session.tokenizer)
            raw = session.generate(
                case["llm_response"],
                output_format=args.output_format,
                max_new_tokens=args.max_new_tokens,
            )
            report = apply_product_boundary(
                raw,
                output_format="minimal" if args.output_format == "minimal_extract" else args.output_format,
            )
            analysis = analyze_decision(case, report, raw)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "probe": "locomo_hf_storage_cpu",
                "output_format": args.output_format,
                "adapter_dir": adapter_label,
                "model_key": args.model,
                "device": args.device,
                "case": case,
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
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_path}")
    return 0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records) or 1

    def rate(key: str) -> float:
        return round(sum(1 for r in records if r["analysis"]["checks"].get(key)) / n, 4)

    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_bucket.setdefault(str(row["case"]["bucket"]), []).append(row)

    return {
        "cases": len(records),
        "output_format": records[0]["output_format"] if records else None,
        "adapter_dir": records[0]["adapter_dir"] if records else None,
        "rates": {
            "store_decision_match": rate("store_decision_match"),
            "parse_ok": rate("parse_ok"),
            "has_memory_content": rate("has_memory_content"),
            "has_facts": rate("has_facts"),
            "gate_match": rate("gate_match") if any(
                r["analysis"]["checks"].get("gate_match") is not None for r in records
            ) else None,
            "has_temporal": round(
                sum(
                    1
                    for r in records
                    if r["analysis"]["checks"]["has_temporal_on_memory"]
                    or r["analysis"]["checks"]["has_temporal_on_facts"]
                )
                / n,
                4,
            ),
            "content_grounded": rate("content_grounded"),
        },
        "by_bucket": {
            bucket: {
                "count": len(rows),
                "store_decision_match": round(
                    sum(1 for r in rows if r["analysis"]["checks"]["store_decision_match"]) / max(1, len(rows)),
                    4,
                ),
                "has_facts": round(
                    sum(1 for r in rows if r["analysis"]["checks"]["has_facts"]) / max(1, len(rows)),
                    4,
                ),
            }
            for bucket, rows in sorted(by_bucket.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--no-adapter", action="store_true", help="Base Qwen only (no LoRA)")
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu = local dev only; cuda on RunPod pod (PSM_FORCE_CPU=0)",
    )
    parser.add_argument("--output-format", default="tagged", choices=["tagged", "json", "minimal", "minimal_extract"])
    parser.add_argument("--max-new-tokens", type=int, default=PROD_STORAGE_MAX_NEW_TOKENS)
    parser.add_argument("--two-pass", action="store_true", help="Binary gate + minimal_extract (prod path)")
    parser.add_argument("--binary-adapter", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--extract-adapter", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--binary-max-tokens", type=int, default=16)
    parser.add_argument("--extract-max-tokens", type=int, default=128)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.two_pass:
        args.adapter_dir = args.extract_adapter
    else:
        adapter_dir = None if args.no_adapter else args.adapter_dir
        args.adapter_dir = adapter_dir
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
