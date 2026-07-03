#!/usr/bin/env python3
"""Broad gate validation on LoCoMo turns (CPU).

Ground truth: a turn is "should store" iff its dia_id is cited as QA evidence.
Runs ONLY the binary gate (16 tokens, fast) and reports precision/recall/F1
so we can confirm the store/ignore failure is systematic before a GPU retrain.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prod_memory.eval_classify import binary_predicts_store
from prod_memory.eval_hf_grounding import open_hf_session

# reuse the exact LoCoMo remember-text builder from the write-path probe
from probe_locomo_hf_storage_cpu import (
    build_locomo_remember_text,
    flatten_turns,
    locomo_source_timestamp,
)
from prod_memory.hf_prompts import storage_inference_messages, apply_chat_prompt

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO / "benchmark/locomo/data/locomo10.json"
DEFAULT_GATE = REPO / "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter"
DEFAULT_OUT = REPO / "benchmark/locomo/results/validate-locomo-gate-cpu.jsonl"


def evidence_dia_ids(sample: dict[str, Any]) -> set[str]:
    ev: set[str] = set()
    for qa in sample.get("qa") or []:
        for e in qa.get("evidence") or []:
            ev.add(str(e))
    return ev


def build_cases(
    data_path: Path,
    *,
    per_conv_pos: int,
    per_conv_neg: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples = json.loads(data_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id") or "unknown")
        turns = flatten_turns(sample)
        ev = evidence_dia_ids(sample)
        pos_idx = [i for i, t in enumerate(turns) if str(t.get("dia_id")) in ev]
        neg_idx = [i for i, t in enumerate(turns) if str(t.get("dia_id")) not in ev]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        chosen = [(i, True) for i in pos_idx[:per_conv_pos]] + [
            (i, False) for i in neg_idx[:per_conv_neg]
        ]
        for idx, expect_store in chosen:
            turn = turns[idx]
            cases.append(
                {
                    "id": f"{sample_id}_{str(turn.get('dia_id')).replace(':', '_')}",
                    "sample_id": sample_id,
                    "dia_id": str(turn.get("dia_id")),
                    "speaker": turn.get("speaker"),
                    "utterance": turn.get("text"),
                    "session": turn.get("session"),
                    "source_timestamp": locomo_source_timestamp(
                        sample, str(turn.get("session") or "")
                    ),
                    "expect_store": expect_store,
                    "llm_response": build_locomo_remember_text(sample, turns, idx),
                }
            )
    return cases


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for r in records if r["expect_store"] and r["gate_store"])
    fp = sum(1 for r in records if not r["expect_store"] and r["gate_store"])
    fn = sum(1 for r in records if r["expect_store"] and not r["gate_store"])
    tn = sum(1 for r in records if not r["expect_store"] and not r["gate_store"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    store_rate = (tp + fp) / len(records) if records else 0.0
    return {
        "cases": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(records), 4) if records else 0.0,
        "predicted_store_rate": round(store_rate, 4),
        "true_store_rate": round(
            sum(1 for r in records if r["expect_store"]) / len(records), 4
        )
        if records
        else 0.0,
    }


def run(args: argparse.Namespace) -> int:
    cases = build_cases(
        args.data,
        per_conv_pos=args.per_conv_pos,
        per_conv_neg=args.per_conv_neg,
        seed=args.seed,
    )
    if not cases:
        raise SystemExit("No cases built")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    session = open_hf_session(args.gate_adapter, model_key=args.model, device=args.device)

    records: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        raw = session.generate(
            case["llm_response"], output_format="binary", max_new_tokens=args.max_new_tokens
        )
        gate_store = binary_predicts_store(raw)
        record = {**case, "gate_raw": raw.strip()[:60], "gate_store": gate_store}
        records.append(record)
        if (i + 1) % 20 == 0 or i + 1 == len(cases):
            print(f"  {i + 1}/{len(cases)} done", flush=True)

    with args.out.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "gate_adapter": str(args.gate_adapter.resolve()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "overall": metrics(records),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--gate-adapter", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--per-conv-pos", type=int, default=6)
    parser.add_argument("--per-conv-neg", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
