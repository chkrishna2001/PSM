from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.build_gate5_train_v1 import _copy_task_rows
from psm_model.data.rows import infer_row_task
from psm_model.generate_recall_curriculum import build_recall_probe_rows
from psm_model.prompts import render_training_text

from prod_memory.build_prod_extraction_v1 import _bucket_rows, _copy_primary_rows
from prod_memory.curriculum_sources import build_primary_source_rows
from prod_memory.ingest_training_data import ingest_training_directory
from prod_memory.label_from_assistant import build_row_from_assistant
from prod_memory.openrouter_teacher import TeacherConfig, build_row_from_teacher
from prod_memory.row_validation import validate_prod_row, validate_prod_rows, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "prod-extraction-v2.jsonl"
DEFAULT_TRAINING_DATA = Path.home() / "Downloads" / "training-data"

PROD_EXTRACTION_V2_PROFILE: dict[str, int] = {
    "expanded_copies": 2,
    "recall_copies": 50,
    "plan_copies": 1,
    "workflow_copies": 1,
    "technical_copies": 1,
    "noise_copies": 1,
    "primary_copies": 1,
    "session_copies": 1,
}

DEFAULT_CONTEXT_LENGTH = 4096


def _filter_rows_by_token_budget(
    rows: list[dict[str, Any]],
    *,
    tokenizer_path: Path | None,
    max_training_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tokenizer_path is None or not tokenizer_path.exists():
        char_budget = max_training_tokens * 3
        kept = [
            row
            for row in rows
            if len(_assistant_text(row)) <= char_budget
        ]
        return kept, {
            "mode": "char_estimate",
            "char_budget": char_budget,
            "kept": len(kept),
            "dropped": len(rows) - len(kept),
        }

    from psm_model.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path)
    kept_rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    for row in rows:
        text = render_training_text(row["input"], row["expected"], output_format="tagged")
        token_count = len(tokenizer.encode(text, add_bos=True, add_eos=True))
        lengths.append(token_count)
        if token_count <= max_training_tokens:
            kept_rows.append(row)
    lengths.sort()
    return kept_rows, {
        "mode": "tokenizer",
        "tokenizer": str(tokenizer_path),
        "max_training_tokens": max_training_tokens,
        "kept": len(kept_rows),
        "dropped": len(rows) - len(kept_rows),
        "p50_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "p95_tokens": lengths[int(len(lengths) * 0.95)] if lengths else 0,
        "max_tokens_seen": lengths[-1] if lengths else 0,
    }


def _assistant_text(row: dict[str, Any]) -> str:
    conversation = row.get("input", {}).get("conversation")
    if isinstance(conversation, list) and conversation:
        return str(conversation[0].get("content") or "")
    if isinstance(conversation, str):
        return conversation
    return ""


def _load_teacher_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("cache_key") or row.get("id") or "")
        if key:
            cache[key] = row
    return cache


def _append_teacher_cache(path: Path | None, entry: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _relabel_sessions_with_teacher(
    session_rows: list[dict[str, Any]],
    *,
    config: TeacherConfig,
    cache_path: Path | None,
    limit: int | None,
    use_heuristic_fallback: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = _load_teacher_cache(cache_path)
    relabeled: list[dict[str, Any]] = []
    report = {
        "model": config.model,
        "cached": 0,
        "labeled": 0,
        "failed": 0,
        "fallback_heuristic": 0,
        "limit": limit,
    }

    targets = session_rows[:limit] if limit is not None else session_rows
    total = len(targets)
    for index, row in enumerate(targets):
        text = _assistant_text(row)
        row_id = str(row.get("id") or "row")
        source_id = str(row.get("input", {}).get("source_id") or row_id)
        source_kind = str(row.get("input", {}).get("source_kind") or "session")
        cache_key = source_id

        if cache_key in cache:
            cached_row = cache[cache_key].get("row")
            if isinstance(cached_row, dict):
                relabeled.append(cached_row)
                report["cached"] += 1
                continue

        try:
            teacher_row, meta = build_row_from_teacher(
                text,
                row_id=row_id,
                source_id=source_id,
                source_kind=source_kind,
                config=config,
                use_heuristic_fallback=use_heuristic_fallback,
            )
        except Exception as exc:
            report["failed"] += 1
            if use_heuristic_fallback:
                teacher_row = build_row_from_assistant(text, row_id=row_id, source_id=source_id, source_kind=source_kind)
                report["fallback_heuristic"] += 1
            else:
                teacher_row = None
            meta = {"error": str(exc)}

        if teacher_row is None:
            continue
        if meta.get("fallback") == "heuristic":
            report["fallback_heuristic"] += 1
        try:
            validate_prod_row(teacher_row)
        except ValueError as exc:
            report["validation_failed"] = report.get("validation_failed", 0) + 1
            if use_heuristic_fallback:
                teacher_row = build_row_from_assistant(text, row_id=row_id, source_id=source_id, source_kind=source_kind)
                if teacher_row is None:
                    continue
                report["fallback_heuristic"] += 1
                meta["validation_error"] = str(exc)
                try:
                    validate_prod_row(teacher_row)
                except ValueError:
                    continue
            else:
                meta["validation_error"] = str(exc)
                continue
        relabeled.append(teacher_row)
        report["labeled"] += 1
        _append_teacher_cache(cache_path, {"cache_key": cache_key, "id": row_id, "row": teacher_row, "meta": meta})
        done = index + 1
        if done % 25 == 0 or done == total:
            print(json.dumps({"event": "teacher_progress", "done": done, "total": total, **report}), flush=True)

    if limit is None:
        return relabeled, report
    remainder = session_rows[len(targets):]
    return relabeled + remainder, report


def build_prod_extraction_v2(
    output: Path,
    *,
    training_data_root: Path | None = None,
    expanded_probes: Path | None = None,
    direct_probes: Path | None = None,
    profile: dict[str, int] | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    tokenizer_path: Path | None = None,
    seed: int = 42,
    teacher_config: TeacherConfig | None = None,
    teacher_cache: Path | None = None,
    teacher_limit: int | None = None,
    teacher_heuristic_fallback: bool = True,
) -> dict[str, Any]:
    copies = profile or PROD_EXTRACTION_V2_PROFILE
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    session_rows: list[dict[str, Any]] = []
    ingest_report: dict[str, Any] | None = None
    teacher_report: dict[str, Any] | None = None
    if training_data_root and training_data_root.exists():
        session_rows, ingest_report = ingest_training_directory(training_data_root)
        if teacher_config is not None:
            session_rows, teacher_report = _relabel_sessions_with_teacher(
                session_rows,
                config=teacher_config,
                cache_path=teacher_cache,
                limit=teacher_limit,
                use_heuristic_fallback=teacher_heuristic_fallback,
            )
        _copy_primary_rows(
            session_rows,
            prefix="session",
            copies=copies["session_copies"],
            seen=seen,
            output=rows,
        )

    primary = build_primary_source_rows()
    for row in primary:
        validate_prod_row(row)
    buckets = _bucket_rows(primary)

    plan_source = [*buckets["plan"], *buckets["cursor"], *buckets["other"]]
    workflow_source = buckets["workflow"] or [
        row for row in primary if str(row.get("expected", {}).get("action")) == "store_episodic"
    ][:2]
    technical_source = buckets["technical"]
    noise_source = buckets["noise"]

    plan_added = _copy_primary_rows(plan_source, prefix="plan", copies=copies["plan_copies"], seen=seen, output=rows)
    workflow_added = _copy_primary_rows(
        workflow_source,
        prefix="workflow",
        copies=copies["workflow_copies"],
        seen=seen,
        output=rows,
    )
    technical_added = _copy_primary_rows(
        technical_source,
        prefix="technical",
        copies=copies["technical_copies"],
        seen=seen,
        output=rows,
    )
    noise_added = _copy_primary_rows(
        noise_source,
        prefix="noise",
        copies=copies["noise_copies"],
        seen=seen,
        output=rows,
    )

    expanded_path = expanded_probes
    if expanded_path is None or not expanded_path.exists():
        expanded_path = direct_probes
    expanded_added = 0
    if expanded_path and expanded_path.exists():
        expanded_added = _copy_rows(
            _load_rows(expanded_path),
            prefix="prod-expanded",
            copies=copies["expanded_copies"],
            seen=seen,
            output=rows,
        )

    recall_added = _copy_task_rows(
        build_recall_probe_rows(),
        prefix="prod-recall",
        copies=copies["recall_copies"],
        seen=seen,
        output=rows,
    )

    token_filter_report: dict[str, Any] | None = None
    if context_length > 0:
        rows, token_filter_report = _filter_rows_by_token_budget(
            rows,
            tokenizer_path=tokenizer_path,
            max_training_tokens=context_length + 1,
        )

    storage_rows = [row for row in rows if infer_row_task(row) == "storage"]
    validation = validate_prod_rows(storage_rows)
    if not validation["ok"]:
        raise ValueError(json.dumps(validation, indent=2))

    write_jsonl(output, rows)
    storage_with_facts = sum(
        1
        for row in storage_rows
        if str(row.get("expected", {}).get("action")) != "ignore" and row.get("expected", {}).get("facts")
    )
    input_lengths = sorted(len(_assistant_text(row)) for row in storage_rows if _assistant_text(row))
    action_counts = Counter(str(row["expected"]["action"]) for row in storage_rows)
    task_counts = Counter(infer_row_task(row) for row in rows)
    manifest = {
        "profile": "prod-extraction-v3" if "v3" in output.stem else "prod-extraction-v2",
        "seed": seed,
        "output": str(output),
        "context_length": context_length,
        "total_rows": len(rows),
        "copies": copies,
        "added": {
            "session": len(session_rows),
            "plan": plan_added,
            "workflow": workflow_added,
            "technical": technical_added,
            "noise": noise_added,
            "expanded_regression": expanded_added,
            "recall_regression": recall_added,
        },
        "sources": {
            "training_data_root": str(training_data_root) if training_data_root else None,
            "expanded_probes": str(expanded_path) if expanded_path else None,
            "direct_probes": str(direct_probes) if direct_probes else None,
            "primary_seed_rows": len(primary),
        },
        "ingest": ingest_report,
        "teacher": teacher_report,
        "token_filter": token_filter_report,
        "storage_stats": {
            "storage_rows": len(storage_rows),
            "rows_with_facts": storage_with_facts,
            "input_chars_p50": input_lengths[len(input_lengths) // 2] if input_lengths else 0,
            "input_chars_max": input_lengths[-1] if input_lengths else 0,
        },
        "action_counts": dict(action_counts),
        "task_counts": dict(task_counts),
        "validation": validation,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build prod-extraction-v2 from long assistant session exports.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-data", type=Path, default=DEFAULT_TRAINING_DATA)
    parser.add_argument("--expanded-probes", type=Path, default=Path("psm-model/data/probes/expanded_probes.jsonl"))
    parser.add_argument("--direct-probes", type=Path, default=Path("psm-model/data/probes/direct_probes.jsonl"))
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--tokenizer", type=Path, default=None, help="Tokenizer for exact 4096 token filtering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher", action="store_true", help="Label session rows with OpenRouter teacher model.")
    parser.add_argument("--teacher-model", type=str, default=None)
    parser.add_argument("--teacher-limit", type=int, default=None, help="Cap teacher API calls (pilot/smoke).")
    parser.add_argument("--teacher-cache", type=Path, default=PACKAGE_ROOT / "data" / "prod-teacher-cache.jsonl")
    parser.add_argument("--no-teacher-heuristic-fallback", action="store_true")
    args = parser.parse_args(argv)

    teacher_config = None
    if args.teacher:
        teacher_config = TeacherConfig.from_env(model=args.teacher_model)

    manifest = build_prod_extraction_v2(
        args.output,
        training_data_root=args.training_data,
        expanded_probes=args.expanded_probes,
        direct_probes=args.direct_probes,
        context_length=args.context_length,
        tokenizer_path=args.tokenizer,
        seed=args.seed,
        teacher_config=teacher_config,
        teacher_cache=args.teacher_cache if args.teacher else None,
        teacher_limit=args.teacher_limit,
        teacher_heuristic_fallback=not args.no_teacher_heuristic_fallback,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
