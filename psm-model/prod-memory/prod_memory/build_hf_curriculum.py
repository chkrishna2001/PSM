from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.data.rows import infer_row_task
from psm_model.generate_recall_curriculum import build_recall_probe_rows

from prod_memory.build_binary_fixture_rows import (
    build_v5k_gate_distill_rows,
    build_v5k_gate_dpo_rows,
    build_v5k_gate_fixture_only_rows,
    build_v5k_gate_rows,
)
from prod_memory.build_minimal_fixture_rows import (
    V5E_BOOST_FIXTURE_IDS,
    build_hybrid_fixture_rows,
    build_json_fixture_rows,
    build_minimal_fixture_rows,
    build_summary_fixture_rows,
    build_v5j_anchor_rows,
    build_v5k_extract_rows,
)
from prod_memory.grounding import is_grounded_in_source, stored_text_from_decision
from prod_memory.hf_prompts import row_messages
from prod_memory.row_validation import remember_target_from_input, validate_prod_row, write_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PACKAGE_ROOT / "data" / "prod-extraction-v3.jsonl"
DEFAULT_SOURCE_V5 = PACKAGE_ROOT / "data" / "prod-extraction-v5.jsonl"
DEFAULT_FIXTURES = PACKAGE_ROOT / "fixtures" / "cases.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "data" / "hf-prod-v1.jsonl"
DEFAULT_OUTPUT_V2 = PACKAGE_ROOT / "data" / "hf-prod-v2.jsonl"
DEFAULT_OUTPUT_V4 = PACKAGE_ROOT / "data" / "hf-prod-v4.jsonl"
DEFAULT_OUTPUT_V5 = PACKAGE_ROOT / "data" / "hf-prod-v5.jsonl"
DEFAULT_OUTPUT_V5B = PACKAGE_ROOT / "data" / "hf-prod-v5b.jsonl"
DEFAULT_OUTPUT_V5C = PACKAGE_ROOT / "data" / "hf-prod-v5c.jsonl"
DEFAULT_OUTPUT_V5D = PACKAGE_ROOT / "data" / "hf-prod-v5d.jsonl"
DEFAULT_OUTPUT_V5E = PACKAGE_ROOT / "data" / "hf-prod-v5e.jsonl"
DEFAULT_OUTPUT_V5F = PACKAGE_ROOT / "data" / "hf-prod-v5f.jsonl"
DEFAULT_OUTPUT_V5G = PACKAGE_ROOT / "data" / "hf-prod-v5g.jsonl"
DEFAULT_OUTPUT_V5H = PACKAGE_ROOT / "data" / "hf-prod-v5h.jsonl"
DEFAULT_OUTPUT_V5I = PACKAGE_ROOT / "data" / "hf-prod-v5i.jsonl"
DEFAULT_OUTPUT_V5J = PACKAGE_ROOT / "data" / "hf-prod-v5j.jsonl"
DEFAULT_OUTPUT_V5K_GATE = PACKAGE_ROOT / "data" / "hf-prod-v5k-gate.jsonl"
DEFAULT_OUTPUT_V5K_GATE_FIX = PACKAGE_ROOT / "data" / "hf-prod-v5k-gate-fix.jsonl"
DEFAULT_OUTPUT_V5K_GATE_DISTILL = PACKAGE_ROOT / "data" / "hf-prod-v5k-gate-distill.jsonl"
DEFAULT_OUTPUT_V5K_GATE_DPO = PACKAGE_ROOT / "data" / "hf-prod-v5k-gate-dpo.jsonl"
DEFAULT_OUTPUT_V5K_EXTRACT = PACKAGE_ROOT / "data" / "hf-prod-v5k-extract.jsonl"
DEFAULT_LOCOMO_V5H = PACKAGE_ROOT / "data" / "hf-prod-v5h-locomo.jsonl"
V5D_BOOST_FIXTURE_IDS = ("plan-01-handoff", "workflow-runpod", "noise-filler", "noise-meta")
DEFAULT_GEMMA_FIXTURES = PACKAGE_ROOT / "data" / "prod-extraction-fixtures-gemma.jsonl"
DEFAULT_SOURCE_V4 = PACKAGE_ROOT / "data" / "prod-extraction-v4.jsonl"
DEFAULT_SOURCE_V6 = PACKAGE_ROOT / "data" / "prod-extraction-v6-v4.jsonl"
DEFAULT_IGNORE_FRACTION_V5 = 0.17
DEFAULT_DATASET_REPO = "krishnach7262/psm-prod-memory-data"
DEFAULT_MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
MIN_STORAGE_P50_CHARS = 500
MIN_STORAGE_V5_CHARS = 80
DEFAULT_RECALL_FRACTION = 0.28
DEFAULT_FIXTURE_COPIES = 25
DEFAULT_FIXTURE_COPIES_V4 = 5


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


def _grounded_storage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep ignore rows and store rows whose label tokens overlap the input (no template bleed)."""
    grounded: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected") or {}
        action = str(expected.get("action") or "").lower()
        if action in {"ignore", "ignore_noise"}:
            grounded.append(row)
            continue
        target = remember_target_from_input(row["input"])
        stored = stored_text_from_decision(expected)
        if stored == "classify-store":
            grounded.append(row)
            continue
        if stored and is_grounded_in_source(target, stored):
            grounded.append(row)
    return grounded


def _simplify_storage_expected(expected: dict[str, Any]) -> dict[str, Any]:
    """Strip mnemonic/indexable skeleton; keep grounded content + facts."""
    simplified = dict(expected)
    simplified["indexables"] = []
    return simplified


def _ignore_rows_from_source(source: Path, *, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _load_rows(source):
        if infer_row_task(row) != "storage":
            continue
        action = str((row.get("expected") or {}).get("action") or "").lower()
        if action not in {"ignore", "ignore_noise"}:
            continue
        validate_prod_row(row)
        rows.append(row)
    if max_rows > 0 and len(rows) > max_rows:
        rows = sorted(rows, key=lambda item: str(item["id"]))[:max_rows]
    return rows


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


def _diverse_v3_store_rows(source: Path, *, max_rows: int = 250) -> list[dict[str, Any]]:
    """Sample grounded v3 store rows with diverse memory.content (no template flood)."""
    seen_prefix: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for row in _storage_rows_from_source(source, min_input_chars=MIN_STORAGE_V5_CHARS):
        action = str((row.get("expected") or {}).get("action") or "").lower()
        if action in {"ignore", "ignore_noise"}:
            continue
        content = stored_text_from_decision(row.get("expected") or {})
        prefix = content[:48].lower()
        if seen_prefix.get(prefix, 0) >= 2:
            continue
        target = remember_target_from_input(row["input"])
        if content and not is_grounded_in_source(target, content):
            continue
        seen_prefix[prefix] = seen_prefix.get(prefix, 0) + 1
        cloned = dict(row)
        cloned["source"] = "exp_a_v3_diverse"
        rows.append(cloned)
        if len(rows) >= max_rows:
            break
    return rows


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
    anchor_rows: list[dict[str, Any]] | None = None,
    ignore_source: Path | None = None,
    ignore_fraction: float = 0.0,
    simplify_labels: bool = False,
    include_source_storage: bool = True,
) -> dict[str, Any]:
    if download and include_source_storage:
        _download_v3(source, dataset_repo=dataset_repo)

    storage_rows: list[dict[str, Any]] = list(anchor_rows or [])
    if include_source_storage:
        storage_rows.extend(_storage_rows_from_source(source, min_input_chars=min_input_chars))
        for extra in extra_sources or []:
            min_chars = (extra_min_chars or {}).get(extra, min_input_chars)
            if extra.exists():
                storage_rows.extend(_storage_rows_from_source(extra, min_input_chars=min_chars))

    if not storage_rows:
        raise ValueError("no storage rows after filter")

    store_count = sum(
        1
        for row in storage_rows
        if str((row.get("expected") or {}).get("action") or "").lower() not in {"ignore", "ignore_noise"}
    )
    if ignore_source and ignore_fraction > 0 and store_count > 0:
        if download:
            _download_v3(ignore_source, dataset_repo=dataset_repo)
        target_ignore = max(1, int(store_count * ignore_fraction / max(1e-6, 1.0 - ignore_fraction)))
        storage_rows.extend(_ignore_rows_from_source(ignore_source, max_rows=target_ignore))

    if simplify_labels:
        for row in storage_rows:
            action = str((row.get("expected") or {}).get("action") or "").lower()
            if action not in {"ignore", "ignore_noise"}:
                row["expected"] = _simplify_storage_expected(row["expected"])

    storage_rows = _grounded_storage_rows(storage_rows)
    if not storage_rows:
        raise ValueError("no grounded storage rows after filter")

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
    min_p50 = MIN_STORAGE_V5_CHARS if profile in {"hf-prod-v2", "hf-prod-v4"} else MIN_STORAGE_P50_CHARS
    if profile in {
        "hf-prod-v5b", "hf-prod-v5c", "hf-prod-v5d", "hf-prod-v5e", "hf-prod-v5f", "hf-prod-v5g",
        "hf-prod-v5h", "hf-prod-v5i", "hf-prod-v5j", "hf-prod-v5k-gate", "hf-prod-v5k-gate-fix", "hf-prod-v5k-gate-distill", "hf-prod-v5k-gate-dpo", "hf-prod-v5k-extract",
    }:
        min_p50 = MIN_STORAGE_V5_CHARS
    elif profile == "hf-prod-v5":
        min_p50 = MIN_STORAGE_P50_CHARS
    if profile in {
        "hf-prod-v5k-gate",
        "hf-prod-v5k-gate-fix",
        "hf-prod-v5k-gate-distill",
    }:
        min_p50 = 0
    elif storage_p50 < min_p50:
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
        "ignore_fraction_target": ignore_fraction,
        "simplify_labels": simplify_labels,
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
    parser.add_argument(
        "--profile",
        choices=[
            "hf-prod-v1", "hf-prod-v2", "hf-prod-v4", "hf-prod-v5", "hf-prod-v5b", "hf-prod-v5c",
            "hf-prod-v5d", "hf-prod-v5e", "hf-prod-v5f", "hf-prod-v5g", "hf-prod-v5h", "hf-prod-v5i",
            "hf-prod-v5j", "hf-prod-v5k-gate", "hf-prod-v5k-gate-fix", "hf-prod-v5k-gate-distill", "hf-prod-v5k-gate-dpo", "hf-prod-v5k-extract",
        ],
        default="hf-prod-v1",
    )
    parser.add_argument("--source-v5", type=Path, default=DEFAULT_SOURCE_V5)
    parser.add_argument("--source-v4", type=Path, default=DEFAULT_SOURCE_V4)
    parser.add_argument("--source-v6", type=Path, default=DEFAULT_SOURCE_V6)
    parser.add_argument("--ignore-fraction", type=float, default=DEFAULT_IGNORE_FRACTION_V5)
    parser.add_argument("--fixture-copies", type=int, default=0)
    args = parser.parse_args(argv)

    if args.profile == "hf-prod-v5k-gate-dpo":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5K_GATE_DPO
        dpo_rows = build_v5k_gate_dpo_rows()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for row in dpo_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "profile": "hf-prod-v5k-gate-dpo",
            "output": str(args.output),
            "total_rows": len(dpo_rows),
            "format": "dpo",
        }
        manifest_path = args.output.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    elif args.profile == "hf-prod-v5k-gate-distill":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5K_GATE_DISTILL
        anchors = build_v5k_gate_distill_rows()
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format="binary",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5k-gate-distill",
            anchor_rows=anchors,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5k-gate-fix":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5K_GATE_FIX
        anchors = build_v5k_gate_fixture_only_rows()
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format="binary",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5k-gate-fix",
            anchor_rows=anchors,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5k-gate":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5K_GATE
        locomo = DEFAULT_LOCOMO_V5H if DEFAULT_LOCOMO_V5H.is_file() else None
        anchors = build_v5k_gate_rows(locomo_path=locomo)
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format="binary",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5k-gate",
            anchor_rows=anchors,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5k-extract":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5K_EXTRACT
        if not args.no_download:
            _download_v3(args.source, dataset_repo=args.dataset_repo)
        seed = build_v5k_extract_rows()
        anchors = list(seed)
        anchors.extend(_copy_rows(seed, prefix="gext", copies=19))
        anchors.extend(_diverse_v3_store_rows(args.source, max_rows=80))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format="minimal_extract",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5k-extract",
            anchor_rows=anchors,
            include_source_storage=False,
            simplify_labels=True,
        )
    elif args.profile == "hf-prod-v5j":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5J
        if not args.no_download:
            _download_v3(args.source, dataset_repo=args.dataset_repo)
        anchors = build_v5j_anchor_rows()
        if DEFAULT_LOCOMO_V5H.is_file():
            for row in _grounded_storage_rows(_load_rows(DEFAULT_LOCOMO_V5H)):
                action = str((row.get("expected") or {}).get("action") or "").lower()
                copies = 24 if action in {"ignore", "ignore_noise"} else 8
                anchors.extend(_copy_rows([row], prefix="fxjl", copies=copies))
        anchors.extend(_diverse_v3_store_rows(args.source, max_rows=250))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5j",
            anchor_rows=anchors,
            ignore_source=args.source,
            ignore_fraction=0.4,
            simplify_labels=True,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5i":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5I
        if not args.no_download:
            _download_v3(args.source, dataset_repo=args.dataset_repo)
        seed = build_hybrid_fixture_rows()
        copies = args.fixture_copies or 20
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxi", copies=copies - 1))
        boost_seed = [row for row in seed if any(fid in row["id"] for fid in V5E_BOOST_FIXTURE_IDS)]
        if boost_seed:
            anchors.extend(_copy_rows(boost_seed, prefix="fxib", copies=50))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_P50_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5i",
            anchor_rows=anchors,
            ignore_fraction=DEFAULT_IGNORE_FRACTION_V5,
            simplify_labels=True,
            include_source_storage=True,
        )
    elif args.profile == "hf-prod-v5h":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5H
        if not args.no_download:
            _download_v3(args.source, dataset_repo=args.dataset_repo)
        seed = build_json_fixture_rows()
        copies = args.fixture_copies or 8
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxh", copies=copies - 1))
        boost_seed = [row for row in seed if any(fid in row["id"] for fid in V5E_BOOST_FIXTURE_IDS)]
        if boost_seed:
            anchors.extend(_copy_rows(boost_seed, prefix="fxhb", copies=30))
        if DEFAULT_LOCOMO_V5H.is_file():
            locomo = _grounded_storage_rows(_load_rows(DEFAULT_LOCOMO_V5H))
            anchors.extend(locomo)
            anchors.extend(_copy_rows(locomo, prefix="fxhl", copies=2))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source,
            output_format="json",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_P50_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5h",
            anchor_rows=anchors,
            ignore_fraction=DEFAULT_IGNORE_FRACTION_V5,
            simplify_labels=True,
            include_source_storage=True,
        )
    elif args.profile == "hf-prod-v5g":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5G
        seed = build_hybrid_fixture_rows()
        copies = args.fixture_copies or 24
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxg", copies=copies - 1))
        store_boost = [row for row in seed if any(fid in row["id"] for fid in ("plan-01-handoff", "workflow-runpod"))]
        if store_boost:
            anchors.extend(_copy_rows(store_boost, prefix="fxgs", copies=40))
        noise_boost = [row for row in seed if any(fid in row["id"] for fid in ("noise-filler", "noise-meta"))]
        if noise_boost:
            anchors.extend(_copy_rows(noise_boost, prefix="fxgn", copies=40))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5g",
            anchor_rows=anchors,
            ignore_fraction=0.0,
            simplify_labels=True,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5f":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5F
        seed = build_hybrid_fixture_rows()
        copies = args.fixture_copies or 30
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxf", copies=copies - 1))
        store_boost = [row for row in seed if any(fid in row["id"] for fid in ("plan-01-handoff", "workflow-runpod"))]
        if store_boost:
            anchors.extend(_copy_rows(store_boost, prefix="fxfs", copies=10))
        noise_boost = [row for row in seed if any(fid in row["id"] for fid in ("noise-filler", "noise-meta"))]
        if noise_boost:
            anchors.extend(_copy_rows(noise_boost, prefix="fxfn", copies=30))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5f",
            anchor_rows=anchors,
            ignore_fraction=0.0,
            simplify_labels=True,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5e":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5E
        seed = build_hybrid_fixture_rows()
        copies = args.fixture_copies or 30
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxe", copies=copies - 1))
        boost_seed = [row for row in seed if any(fid in row["id"] for fid in V5E_BOOST_FIXTURE_IDS)]
        if boost_seed:
            anchors.extend(_copy_rows(boost_seed, prefix="fxeb", copies=20))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5e",
            anchor_rows=anchors,
            ignore_fraction=0.0,
            simplify_labels=True,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5d":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5D
        seed = build_summary_fixture_rows()
        copies = args.fixture_copies or 30
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxd", copies=copies - 1))
        boost_seed = [row for row in seed if any(fid in row["id"] for fid in V5D_BOOST_FIXTURE_IDS)]
        if boost_seed:
            anchors.extend(_copy_rows(boost_seed, prefix="fxdb", copies=15))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5d",
            anchor_rows=anchors,
            ignore_fraction=0.0,
            simplify_labels=True,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5c":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5C
        seed = build_minimal_fixture_rows()
        copies = args.fixture_copies or 30
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxc", copies=copies - 1))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5c",
            anchor_rows=anchors,
            ignore_fraction=0.0,
            simplify_labels=True,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5b":
        from prod_memory.curriculum_sources import build_fixture_rows, build_noise_rows

        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5B
        # ponytail: curriculum_sources has correct store/ignore per fixture; Gemma bleed-filter drops runpod/plan rows.
        seed = build_fixture_rows() + build_noise_rows()
        copies = args.fixture_copies or 12
        anchors = list(seed)
        if copies > 1:
            anchors.extend(_copy_rows(seed, prefix="fxb", copies=copies - 1))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format="tagged",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_V5_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5b",
            anchor_rows=anchors,
            ignore_source=args.source_v4,
            ignore_fraction=0.22,
            simplify_labels=True,
            include_source_storage=False,
        )
    elif args.profile == "hf-prod-v5":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V5
        minimal = build_minimal_fixture_rows()
        copies = args.fixture_copies or DEFAULT_FIXTURE_COPIES_V4
        anchors = list(minimal)
        if copies > 1:
            anchors.extend(_copy_rows(minimal, prefix="fx", copies=copies - 1))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format="minimal",
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_P50_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v5",
            anchor_rows=anchors,
            ignore_source=args.source_v4,
            ignore_fraction=args.ignore_fraction,
            simplify_labels=True,
        )
    elif args.profile == "hf-prod-v4":
        args.output = args.output if args.output != DEFAULT_OUTPUT else DEFAULT_OUTPUT_V4
        minimal = build_minimal_fixture_rows()
        copies = args.fixture_copies or DEFAULT_FIXTURE_COPIES_V4
        anchors = list(minimal)
        if copies > 1:
            anchors.extend(_copy_rows(minimal, prefix="fx", copies=copies - 1))
        manifest = build_hf_curriculum(
            args.output,
            source=args.source_v6,
            output_format=args.output_format,
            recall_fraction=0.0,
            min_input_chars=MIN_STORAGE_P50_CHARS,
            dataset_repo=args.dataset_repo,
            download=False,
            fixture_copies=0,
            profile="hf-prod-v4",
            anchor_rows=anchors,
        )
    elif args.profile == "hf-prod-v2":
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
