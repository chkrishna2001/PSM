from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from prod_memory.ingest_training_data import ingest_training_directory
from prod_memory.label_from_assistant import build_expected_from_assistant, extract_facts, extract_memory_content
from prod_memory.openrouter_teacher import TeacherConfig, label_assistant_with_teacher
from prod_memory.row_validation import validate_prod_row

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DATA = Path.home() / "Downloads" / "training-data"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "prod-teacher-pilot-review.json"
DEFAULT_V2_MIX = PACKAGE_ROOT / "data" / "prod-extraction-v2.jsonl"


def _assistant_text_from_row(row: dict[str, Any]) -> str:
    conversation = row.get("input", {}).get("conversation")
    if isinstance(conversation, list) and conversation:
        return str(conversation[0].get("content") or "")
    return ""


def select_pilot_candidates(
    training_data_root: Path,
    *,
    limit: int,
    seed: int,
    mix_path: Path | None = None,
) -> list[dict[str, str]]:
    if mix_path and mix_path.exists():
        rows = [json.loads(line) for line in mix_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [row for row in rows if str(row.get("id", "")).startswith("session:")]
    else:
        rows, _report = ingest_training_directory(training_data_root)
    store_rows = [row for row in rows if str(row.get("expected", {}).get("action")) != "ignore"]
    if not store_rows:
        store_rows = rows
    random.seed(seed)
    sample = random.sample(store_rows, min(limit, len(store_rows)))
    candidates: list[dict[str, str]] = []
    for row in sample:
        text = _assistant_text_from_row(row)
        if not text:
            continue
        candidates.append({
            "row_id": str(row.get("id") or "row"),
            "source_kind": str(row.get("input", {}).get("source_kind") or "session"),
            "text": text,
            "heuristic_action": str(row.get("expected", {}).get("action") or ""),
        })
    return candidates


def review_pilot(
    training_data_root: Path,
    *,
    limit: int = 15,
    seed: int = 42,
    model: str | None = None,
    use_heuristic_fallback: bool = False,
    mix_path: Path | None = None,
) -> dict[str, Any]:
    config = TeacherConfig.from_env(model=model)
    candidates = select_pilot_candidates(
        training_data_root,
        limit=limit,
        seed=seed,
        mix_path=mix_path,
    )
    comparisons: list[dict[str, Any]] = []
    stats = {
        "teacher_store": 0,
        "heuristic_store": 0,
        "agreement": 0,
        "teacher_valid": 0,
        "teacher_facts": 0,
        "heuristic_facts": 0,
    }

    for index, candidate in enumerate(candidates):
        text = candidate["text"]
        heuristic = build_expected_from_assistant(text) or {"action": "ignore"}
        teacher, meta = label_assistant_with_teacher(
            text,
            config=config,
            use_heuristic_fallback=use_heuristic_fallback,
        )

        teacher_row = {
            "id": f"pilot-{index}",
            "input": {"operation": "remember_llm_response", "conversation": [{"role": "assistant", "content": text}]},
            "expected": teacher,
        }
        teacher_valid = False
        try:
            validate_prod_row(teacher_row)
            teacher_valid = True
            stats["teacher_valid"] += 1
        except ValueError as exc:
            meta["validation_error"] = str(exc)

        if teacher.get("action") != "ignore":
            stats["teacher_store"] += 1
            stats["teacher_facts"] += len(teacher.get("facts") or [])
        if heuristic.get("action") != "ignore":
            stats["heuristic_store"] += 1
            stats["heuristic_facts"] += len(heuristic.get("facts") or [])
        if teacher.get("action") == heuristic.get("action"):
            stats["agreement"] += 1

        comparisons.append({
            "row_id": candidate["row_id"],
            "source_kind": candidate["source_kind"],
            "input_chars": len(text),
            "heuristic": {
                "action": heuristic.get("action"),
                "memory": (heuristic.get("memory") or {}).get("content") if isinstance(heuristic.get("memory"), dict) else None,
                "facts": len(heuristic.get("facts") or []),
                "memory_preview": extract_memory_content(text)[:200],
                "fact_preview": [f.get("value") for f in extract_facts(text)[:2]],
            },
            "teacher": {
                "action": teacher.get("action"),
                "memory": (teacher.get("memory") or {}).get("content") if isinstance(teacher.get("memory"), dict) else None,
                "facts": len(teacher.get("facts") or []),
                "reasoning": teacher.get("reasoning"),
            },
            "teacher_valid": teacher_valid,
            "meta": {key: value for key, value in meta.items() if key != "raw_response"},
        })

    report = {
        "profile": "prod-teacher-pilot",
        "model": config.model,
        "fallback_model": config.fallback_model,
        "training_data_root": str(training_data_root),
        "samples": len(comparisons),
        "stats": stats,
        "verdict": _pilot_verdict(stats, comparisons),
        "comparisons": comparisons,
    }
    return report


def _pilot_verdict(stats: dict[str, int], comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return "no_samples"
    valid_rate = stats["teacher_valid"] / len(comparisons)
    if valid_rate < 0.6:
        return "reject_teacher_quality_low"
    if stats["teacher_store"] < max(3, len(comparisons) // 4):
        return "teacher_too_conservative_use_heuristic_fallback"
    if stats["teacher_facts"] > stats["heuristic_facts"] * 1.2:
        return "approve_teacher_for_session_labeling"
    if stats["agreement"] / len(comparisons) > 0.5:
        return "approve_teacher_mixed_with_heuristic"
    return "review_manually_before_full_run"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pilot-review OpenRouter teacher labels vs heuristics.")
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mix", type=Path, default=DEFAULT_V2_MIX, help="Existing v2 mix to sample (faster than re-ingest).")
    parser.add_argument("--heuristic-fallback", action="store_true")
    args = parser.parse_args(argv)

    report = review_pilot(
        args.training_data,
        limit=args.limit,
        seed=args.seed,
        model=args.model,
        use_heuristic_fallback=args.heuristic_fallback,
        mix_path=args.mix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "profile": report["profile"],
        "model": report["model"],
        "samples": report["samples"],
        "stats": report["stats"],
        "verdict": report["verdict"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
