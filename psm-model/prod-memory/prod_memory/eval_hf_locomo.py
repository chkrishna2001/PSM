"""LoCoMo ingest smoke for HF LoRA prod-memory (storage quality, no SQLite)."""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from psm_model.remember_cli import apply_product_boundary

from prod_memory.eval_classify import binary_predicts_store
from prod_memory.eval_hf_grounding import HfGenerationSession, open_hf_session
from prod_memory.grounding import (
    apply_storage_guards,
    has_curriculum_bleed,
    stored_text_from_decision,
    would_model_store,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PACKAGE_ROOT.parent.parent / "benchmark" / "locomo" / "data" / "locomo10.json"
DEFAULT_OUT = PACKAGE_ROOT / "results" / "hf-locomo-smoke.json"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
BLEED = re.compile(r"checkpoint|powershell|gate datasets|nvidia-smi|direct probe|token budget|runpod", re.I)
WRAPPER = re.compile(r"current utterance:|source id:|locomo benchmark|extraction guidance:", re.I)
GOLD = (
    ("D1:3", (re.compile(r"lgbtq", re.I), re.compile(r"support group", re.I))),
    ("D1:5", (re.compile(r"transgender", re.I),)),
    ("D1:12", (re.compile(r"sunrise|paint", re.I),)),
)


def _ensure_data(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LOCOMO_URL, timeout=120) as resp:
        path.write_bytes(resp.read())


def _flatten_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = sample.get("conversation") or {}
    keys = sorted(
        (k for k in conversation if re.fullmatch(r"session_\d+", str(k))),
        key=lambda k: int(str(k).split("_")[1]),
    )
    turns: list[dict[str, Any]] = []
    for key in keys:
        block = conversation.get(key)
        if not isinstance(block, list):
            continue
        for turn in block:
            if isinstance(turn, dict):
                turns.append({**turn, "session": key})
    return turns


def _product_text(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "Unknown").strip()
    utterance = str(turn.get("text") or "").strip()
    bits = [
        f"Image query: {turn['query']}." if turn.get("query") else "",
        f"Image caption: {turn['blip_caption']}." if turn.get("blip_caption") else "",
    ]
    base = f'{speaker} said "{utterance}".' if utterance else f"{speaker} shared an image."
    extra = " ".join(b for b in bits if b).strip()
    return f"{base} {extra}".strip() if extra else base


def _quality_issue(source: str, check: str, detail: str) -> dict[str, str]:
    return {"source": source, "check": check, "detail": detail[:200]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HF LoRA LoCoMo ingest smoke (n turns).")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--binary-adapter", type=Path, default=None, help="Two-pass: gate adapter; extract uses --adapter-dir")
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output-format", default="minimal", choices=["json", "tagged", "minimal", "minimal_extract"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--label", default="hf-locomo-smoke")
    args = parser.parse_args(argv)

    _ensure_data(args.data)
    samples = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise SystemExit(f"invalid locomo data: {args.data}")

    session = open_hf_session(args.adapter_dir, model_key=args.model, device=args.device)
    binary_session: HfGenerationSession | None = None
    if args.binary_adapter is not None:
        binary_session = open_hf_session(args.binary_adapter, model_key=args.model, device=args.device)
    stats = {"seen": 0, "stored": 0, "ignored": 0, "failed": 0, "guard_rejected": 0}
    stored_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    ingested_ids: set[str] = set()

    for sample in samples:
        sample_id = str(sample.get("sample_id") or "unknown")
        for index, turn in enumerate(_flatten_turns(sample)):
            if args.limit > 0 and stats["seen"] >= args.limit:
                break
            dia_id = str(turn.get("dia_id") or "")
            source = f"{sample_id}:{dia_id or stats['seen']}"
            stats["seen"] += 1
            ingested_ids.add(dia_id)
            llm_response = _product_text(turn)
            try:
                if binary_session is not None:
                    raw_binary = binary_session.generate(
                        llm_response,
                        output_format="binary",
                        max_new_tokens=16,
                    )
                    if not binary_predicts_store(raw_binary):
                        stats["ignored"] += 1
                        continue
                    raw = session.generate(
                        llm_response,
                        output_format=args.output_format,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
                    raw = session.generate(
                        llm_response,
                        output_format=args.output_format,
                        max_new_tokens=args.max_new_tokens,
                    )
                parse_format = "minimal" if args.output_format == "minimal_extract" else args.output_format
                report = apply_product_boundary(raw, output_format=parse_format)
                decision = report.get("parsed") if isinstance(report.get("parsed"), dict) else {}
                action = str(decision.get("action") or "").lower()
                if action in {"ignore", "ignore_noise"}:
                    stats["ignored"] += 1
                    continue
                if not would_model_store(decision):
                    stats["ignored"] += 1
                    continue
                guarded = apply_storage_guards(llm_response, decision)
                if guarded["rejected"]:
                    stats["guard_rejected"] += 1
                    stats["failed"] += 1
                    errors.append({"source": source, "error": f"guard:{guarded['route']}"})
                    continue
                content = stored_text_from_decision(decision)
                stats["stored"] += 1
                stored_rows.append(
                    {
                        "source": source,
                        "dia_id": dia_id,
                        "content": content,
                        "raw_output": (raw or "")[:300],
                    }
                )
            except Exception as exc:  # ponytail: smoke script; one bad turn must not kill batch
                stats["failed"] += 1
                errors.append({"source": source, "error": str(exc)[:200]})

        if args.limit > 0 and stats["seen"] >= args.limit:
            break

    issues: list[dict[str, str]] = []
    for row in stored_rows:
        content = str(row.get("content") or "").strip()
        source = str(row["source"])
        if not content:
            issues.append(_quality_issue(source, "empty_content", ""))
        elif content.startswith("{") or WRAPPER.search(content):
            issues.append(_quality_issue(source, "wrapper_or_json", content))
        if BLEED.search(content) or has_curriculum_bleed(content):
            issues.append(_quality_issue(source, "curriculum_bleed", content))
        if re.match(r"^user prefers\b", content, re.I):
            issues.append(_quality_issue(source, "generic_user_pref", content))

    if stats["stored"] == 0:
        issues.append(_quality_issue("batch", "no_stored", "zero grounded stores"))

    by_dia = {str(r.get("dia_id") or ""): r for r in stored_rows}
    for dia_id, needles in GOLD:
        if dia_id not in ingested_ids:
            continue
        row = by_dia.get(dia_id)
        if not row:
            issues.append(_quality_issue(dia_id, "missing_gold_memory", "expected store"))
            continue
        text = str(row.get("content") or "")
        if not any(p.search(text) for p in needles):
            issues.append(_quality_issue(dia_id, "gold_fact_missing", text))

    summary = {
        "label": args.label,
        "adapter_dir": str(args.adapter_dir),
        "binary_adapter_dir": str(args.binary_adapter) if args.binary_adapter else None,
        "two_pass": args.binary_adapter is not None,
        "limit": args.limit,
        "output_format": args.output_format,
        **stats,
        "memory_rows": len(stored_rows),
        "issues": len(issues),
        "passed": len(issues) == 0,
    }
    report = {"summary": summary, "issues": issues, "errors": errors[:20], "stored_sample": stored_rows[:5]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
