from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.data.rows import infer_row_task
from psm_model.generate_recall_curriculum import build_recall_probe_rows

from prod_memory.hf_prompts import row_messages
from prod_memory.row_validation import remember_target_from_input, validate_prod_row, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PACKAGE_ROOT / "data" / "prod-extraction-v3.jsonl"
DEFAULT_SOURCE_V5 = PACKAGE_ROOT / "data" / "prod-extraction-v5.jsonl"
DEFAULT_FIXTURES = PACKAGE_ROOT / "fixtures" / "cases.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "hf-prod-v1.jsonl"
DEFAULT_OUTPUT_V2 = PACKAGE_ROOT / "data" / "hf-prod-v2.jsonl"
DEFAULT_DATASET_REPO = "krishnach7262/psm-prod-memory-data"
DEFAULT_MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
MIN_STORAGE_P50_CHARS = 500
MIN_STORAGE_V5_CHARS = 80
DEFAULT_RECALL_FRACTION = 0.28
DEFAULT_FIXTURE_COPIES = 25


def _download_v3(source: Path, *, dataset_repo: str) -> None:
    if source.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(f"source missing and huggingface_hub unavailable: {source}") from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("DATASET_HF_TOKEN")
    for remote in (
        f"prod-memory/{source.name}",
        f"data/{source.name}",
    ):
        try:
            path = hf_hub_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                filename=remote,
                token=token,
            )
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(Path(path).read_text(encoding="utf-8"), encoding="utf-8")
            return
        except Exception:
            continue
    raise FileNotFoundError(f"could not download {source.name} from {dataset_repo}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _copy_rows(rows: list[dict[str, Any]], *, prefix: str, copies: int) -> list[dict[str, Any]]:
    if copies <= 0:
        return []
    out: list[dict[str, Any]] = []
    for copy_idx in range(copies):
        for row in rows:
            cloned = dict(row)
            cloned["id"] = f"{prefix}-{copy_idx:03d}-{row['id']}"
            cloned["source"] = f"{prefix}:{row.get('source', 'row')}"
            out.append(cloned)
    return out


def _storage_rows_from_source(
    source: Path,
    *,
    min_input_chars: int,
) -> list[dict[str, Any]]:
    storage_rows: list[dict[str, Any]] = []
    for row in _load_rows(source):
        if infer_row_task(row) != "storage":
            continue
        if not row.get("expected", {}).get("action"):
            continue
        validate_prod_row(row)
        if len(remember_target_from_input(row["input"])) < min_input_chars:
            continue
        storage_rows.append(row)
    return storage_rows


def _fixture_ids(fixtures_path: Path) -> set[str]:
    payload = json.loads(fixtures_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    return {str(case["id"]) for case in cases if isinstance(case, dict) and case.get("id")}


def build_hf_curriculum(
    output: Path,
    *,
    source: Path,
    output_format: str = "tagged",
    recall_fraction: float = DEFAULT_RECALL_FRACTION,
    min_input_chars: int = MIN_STORAGE_P50_CHARS,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    download: bool = True,
    extra_sources: list[Path] | None = None,
    extra_min_chars: dict[Path, int] | None = None,
    fixture_copies: int = 0,
    fixtures_path: Path | None = None,
    profile: str = "hf-prod-v1",
) -> dict[str, Any]:
    if download:
        _download_v3(source, dataset_repo=dataset_repo)

    storage_rows: list[dict[str, Any]] = _storage_rows_from_source(source, min_input_chars=min_input_chars)
    for extra in extra_sources or []:
        min_chars = (extra_min_chars or {}).get(extra, min_input_chars)
        if extra.exists():
            storage_rows.extend(_storage_rows_from_source(extra, min_input_chars=min_chars))

    if not storage_rows:
        raise ValueError("no storage rows after filter")

    fixture_ids = _fixture_ids(fixtures_path or DEFAULT_FIXTURES) if fixture_copies > 0 else set()
    fixture_rows = [
        row
        for row in storage_rows
        if str(row.get("input", {}).get("source_id") or "") in fixture_ids
        or any(fid in str(row.get("id") or "") for fid in fixture_ids)
    ]
    if fixture_copies > 0 and fixture_rows:
        storage_rows.extend(_copy_rows(fixture_rows, prefix="fx", copies=fixture_copies - 1))

    # ponytail: dedupe by id, keep fixture copies distinct via _copy_rows ids.
    seen: set[str] = set()
    unique_storage: list[dict[str, Any]] = []
    for row in storage_rows:
        row_id = str(row["id"])
        if row_id in seen:
            continue
        seen.add(row_id)
        unique_storage.append(row)
    storage_rows = unique_storage

    storage_lengths = sorted(len(remember_target_from_input(row["input"])) for row in storage_rows)
    storage_p50 = storage_lengths[len(storage_lengths) // 2]
    min_p50 = MIN_STORAGE_V5_CHARS if profile == "hf-prod-v2" else MIN_STORAGE_P50_CHARS
    if storage_p50 < min_p50:
        raise ValueError(f"storage input p50 {storage_p50} below {min_p50}")

    recall_seed = build_recall_probe_rows()
    if recall_fraction <= 0:
        recall_rows: list[dict[str, Any]] = []
    else:
        target_recall = max(1, int(len(storage_rows) * recall_fraction / max(1e-6, 1.0 - recall_fraction)))
        recall_copies = max(1, (target_recall + len(recall_seed) - 1) // len(recall_seed))
        recall_rows = _copy_rows(recall_seed, prefix="hf-recall", copies=recall_copies)

    rows = storage_rows + recall_rows
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            continue
        seen.add(row_id)
        unique_rows.append(row)

    hf_rows: list[dict[str, Any]] = []
    for row in unique_rows:
        hf_rows.append({
            "id": row["id"],
            "task": infer_row_task(row),
            "messages": row_messages(row, output_format=output_format),
            "source": row.get("source"),
        })

    write_jsonl(output, hf_rows)
    task_counts = Counter(item["task"] for item in hf_rows)
    action_counts = Counter(
        str(row.get("expected", {}).get("action") or "")
        for row in unique_rows
        if infer_row_task(row) == "storage"
    )
    with_facts = sum(1 for row in storage_rows if row.get("expected", {}).get("facts"))
    with_indexables = sum(1 for row in storage_rows if row.get("expected", {}).get("indexables"))
    manifest = {
        "profile": profile,
        "source": str(source),
        "extra_sources": [str(p) for p in (extra_sources or [])],
        "output": str(output),
        "output_format": output_format,
        "fixture_copies": fixture_copies,
        "fixture_rows": len(fixture_rows),
        "total_rows": len(hf_rows),
        "storage_rows": len(storage_rows),
        "recall_rows": len(recall_rows),
        "recall_fraction": len(recall_rows) / len(hf_rows) if hf_rows else 0.0,
        "storage_input_chars_p50": storage_p50,
        "storage_input_chars_p90": storage_lengths[int(len(storage_lengths) * 0.9)],
        "storage_input_chars_max": storage_lengths[-1],
        "rows_with_facts": with_facts,
        "rows_with_indexables": with_indexables,
        "task_counts": dict(task_counts),
        "action_counts": dict(action_counts),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build clean HF LoRA curriculum from teacher v3 + recall probes.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-format", choices=["tagged", "json"], default="tagged")
    parser.add_argument("--recall-fraction", type=float, default=DEFAULT_RECALL_FRACTION)
    parser.add_argument("--min-input-chars", type=int, default=MIN_STORAGE_P50_CHARS)
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--profile", choices=["hf-prod-v1", "hf-prod-v2"], default="hf-prod-v1")
    parser.add_argument("--source-v5", type=Path, default=DEFAULT_SOURCE_V5)
    parser.add_argument("--fixture-copies", type=int, default=0)
    args = parser.parse_args(argv)

    if args.profile == "hf-prod-v2":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V2
        recall_fraction = 0.12 if args.recall_fraction == DEFAULT_RECALL_FRACTION else args.recall_fraction
        fixture_copies = args.fixture_copies or DEFAULT_FIXTURE_COPIES
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format=args.output_format,
            recall_fraction=recall_fraction,
            min_input_chars=MIN_STORAGE_P50_CHARS,
            dataset_repo=args.dataset_repo,
            download=not args.no_download,
            extra_sources=[args.source_v5],
            extra_min_chars={args.source_v5: MIN_STORAGE_V5_CHARS},
            fixture_copies=fixture_copies,
            profile="hf-prod-v2",
        )
    else:
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format=args.output_format,
            recall_fraction=args.recall_fraction,
            min_input_chars=args.min_input_chars,
            dataset_repo=args.dataset_repo,
            download=not args.no_download,
            profile="hf-prod-v1",
        )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
