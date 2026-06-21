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
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "hf-prod-v1.jsonl"
DEFAULT_DATASET_REPO = "krishnach7262/psm-prod-memory-data"
DEFAULT_MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
MIN_STORAGE_P50_CHARS = 500
DEFAULT_RECALL_FRACTION = 0.28


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


def build_hf_curriculum(
    output: Path,
    *,
    source: Path,
    output_format: str = "tagged",
    recall_fraction: float = DEFAULT_RECALL_FRACTION,
    min_input_chars: int = MIN_STORAGE_P50_CHARS,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    download: bool = True,
) -> dict[str, Any]:
    if download:
        _download_v3(source, dataset_repo=dataset_repo)
    if not source.exists():
        raise FileNotFoundError(f"teacher source not found: {source}")

    storage_rows: list[dict[str, Any]] = []
    for row in _load_rows(source):
        if infer_row_task(row) != "storage":
            continue
        validate_prod_row(row)
        if len(remember_target_from_input(row["input"])) < min_input_chars:
            continue
        storage_rows.append(row)

    if not storage_rows:
        raise ValueError("no storage rows after filter")

    storage_lengths = sorted(len(remember_target_from_input(row["input"])) for row in storage_rows)
    storage_p50 = storage_lengths[len(storage_lengths) // 2]
    if storage_p50 < MIN_STORAGE_P50_CHARS:
        raise ValueError(f"storage input p50 {storage_p50} below {MIN_STORAGE_P50_CHARS}")

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
        "profile": "hf-prod-v1",
        "source": str(source),
        "output": str(output),
        "output_format": output_format,
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
    args = parser.parse_args(argv)
    manifest = build_hf_curriculum(
        args.output,
        source=args.source,
        output_format=args.output_format,
        recall_fraction=args.recall_fraction,
        min_input_chars=args.min_input_chars,
        dataset_repo=args.dataset_repo,
        download=not args.no_download,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
