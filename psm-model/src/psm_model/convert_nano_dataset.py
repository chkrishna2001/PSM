from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.data import validate_training_row
from psm_model.schema import ACTIONS, validate_storage_decision


STORAGE_ACTIONS = ACTIONS


def convert_nano_files(
    inputs: list[Path],
    output_dir: Path,
    *,
    validation_ratio: float = 0.08,
    test_ratio: float = 0.04,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    report: dict[str, Any] = {
        "inputs": [str(path) for path in inputs],
        "read": 0,
        "accepted": 0,
        "skipped": Counter(),
        "actions": Counter(),
        "memory_types": Counter(),
    }

    for path in inputs:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                report["read"] += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    report["skipped"]["invalid_json"] += 1
                    continue
                converted, reason = convert_nano_row(raw, source_path=path, line_number=line_number)
                if converted is None:
                    report["skipped"][reason] += 1
                    continue
                dedupe_key = _dedupe_key(converted)
                if dedupe_key in seen_keys:
                    report["skipped"]["duplicate"] += 1
                    continue
                seen_keys.add(dedupe_key)
                rows.append(converted)
                expected = converted["expected"]
                report["actions"][expected["action"]] += 1
                memory = expected.get("memory")
                report["memory_types"][memory["type"] if memory else "none"] += 1
                if limit is not None and len(rows) >= limit:
                    break
        if limit is not None and len(rows) >= limit:
            break

    splits = split_rows(rows, validation_ratio=validation_ratio, test_ratio=test_ratio)
    for split, split_rows_value in splits.items():
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row | {"split": split}, ensure_ascii=False, sort_keys=True) for row in split_rows_value) + "\n",
            encoding="utf-8",
        )
    all_path = output_dir / "all.jsonl"
    all_path.write_text(
        "\n".join(
            json.dumps(row | {"split": split}, ensure_ascii=False, sort_keys=True)
            for split, split_rows_value in splits.items()
            for row in split_rows_value
        )
        + "\n",
        encoding="utf-8",
    )

    report["accepted"] = len(rows)
    report["splits"] = {split: len(split_rows_value) for split, split_rows_value in splits.items()}
    report["actions"] = dict(sorted(report["actions"].items()))
    report["memory_types"] = dict(sorted(report["memory_types"].items()))
    report["skipped"] = dict(sorted(report["skipped"].items()))
    report_path = output_dir / "conversion-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def convert_nano_row(raw: dict[str, Any], *, source_path: Path, line_number: int) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "row_not_object"
    output = raw.get("output")
    if not isinstance(output, dict):
        return None, "missing_output"
    action = output.get("action")
    if action not in STORAGE_ACTIONS:
        return None, f"unsupported_action:{action}"

    expected = canonical_expected(output)
    result = validate_storage_decision(expected)
    if not result.ok:
        return None, "invalid_expected"

    input_payload = canonical_input(raw.get("input"))
    raw_id = str(raw.get("id") or f"{source_path.stem}:{line_number}")
    row = {
        "id": _stable_row_id(raw_id, source_path=source_path, line_number=line_number),
        "input": input_payload,
        "expected": expected,
        "source": f"nano:{source_path.as_posix()}:{line_number}",
    }
    _, issues = validate_training_row(row)
    if issues:
        return None, "invalid_training_row"
    return row, ""


def canonical_expected(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": output["action"],
        "memory": canonical_memory(output.get("memory")),
        "facts": [canonical_fact(fact) for fact in output.get("facts", []) if isinstance(fact, dict)],
        "reasoning": str(output.get("reasoning") or "No reasoning provided."),
    }


def canonical_memory(memory: Any) -> dict[str, Any] | None:
    if memory is None:
        return None
    if not isinstance(memory, dict):
        return None
    result: dict[str, Any] = {
        "content": str(memory.get("content") or ""),
        "type": str(memory.get("type") or ""),
        "tags": [str(tag) for tag in memory.get("tags", []) if isinstance(tag, str) and tag.strip()],
    }
    for key in ("strength", "decay_rate", "emotional_weight", "confidence", "resolved_time_confidence"):
        if isinstance(memory.get(key), (int, float)) and not isinstance(memory.get(key), bool):
            result[key] = float(memory[key])
    for key in ("temporal_expression", "resolved_time"):
        if isinstance(memory.get(key), str) and memory[key].strip():
            result[key] = memory[key]
    return result


def canonical_fact(fact: dict[str, Any]) -> dict[str, Any]:
    result = {
        "subject": str(fact.get("subject") or ""),
        "predicate": str(fact.get("predicate") or ""),
        "value": fact.get("value"),
        "confidence": fact.get("confidence") if isinstance(fact.get("confidence"), (int, float)) else None,
        "inference_kind": str(fact.get("inference_kind") or "explicit"),
        "evidence_text": str(fact.get("evidence_text") or ""),
    }
    for key in ("temporal_expression", "resolved_time"):
        if isinstance(fact.get(key), str) and fact[key].strip():
            result[key] = fact[key]
    if isinstance(fact.get("resolved_time_confidence"), (int, float)):
        result["resolved_time_confidence"] = float(fact["resolved_time_confidence"])
    return result


def canonical_input(input_value: Any) -> dict[str, Any]:
    if not isinstance(input_value, dict):
        return {"conversation": str(input_value or "")}
    parts: list[str] = []
    for item in input_value.get("prior_context", []):
        if isinstance(item, dict) and item.get("text"):
            speaker = item.get("speaker") or "Speaker"
            parts.append(f"{speaker}: {item['text']}")
    current = input_value.get("current_turn")
    if isinstance(current, dict) and current.get("text"):
        speaker = current.get("speaker") or "Speaker"
        parts.append(f"{speaker}: {current['text']}")
    if not parts and isinstance(input_value.get("conversation"), str):
        parts.append(input_value["conversation"])

    result: dict[str, Any] = {
        "conversation": "\n".join(parts).strip(),
    }
    memory_store = input_value.get("memory_store")
    if isinstance(memory_store, list) and memory_store:
        result["context"] = json.dumps({"memory_store": memory_store}, ensure_ascii=False, sort_keys=True)
    for key in ("operation", "source_kind", "source_id", "session_id"):
        if isinstance(input_value.get(key), str) and input_value[key].strip():
            result[key] = input_value[key]
    if isinstance(current, dict) and isinstance(current.get("timestamp"), str):
        result["source_timestamp"] = current["timestamp"]
    return result


def split_rows(
    rows: list[dict[str, Any]],
    *,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "validation": [], "test": []}
    for row in sorted(rows, key=lambda item: item["id"]):
        bucket = int(hashlib.sha256(row["id"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < test_ratio:
            splits["test"].append(row)
        elif bucket < test_ratio + validation_ratio:
            splits["validation"].append(row)
        else:
            splits["train"].append(row)
    return splits


def _dedupe_key(row: dict[str, Any]) -> str:
    payload = {
        "input": row["input"],
        "expected": row["expected"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _stable_row_id(raw_id: str, *, source_path: Path, line_number: int) -> str:
    suffix_payload = f"{source_path.as_posix()}:{line_number}:{raw_id}"
    suffix = hashlib.sha256(suffix_payload.encode("utf-8")).hexdigest()[:10]
    return f"{raw_id}:{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Nano PSM JSONL rows into psm-model canonical storage rows.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--validation-ratio", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.04)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    report = convert_nano_files(
        args.inputs,
        args.output_dir,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
