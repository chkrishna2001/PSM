"""Deterministic post-parse repair for tagged storage decisions (product boundary).

This is the Phase 2 "decode-boundary repair pass" from the 2026-06-10 parse
recovery plan. It never retrains or re-decodes; it salvages what free decode
produced, and when salvage is impossible it fails safe to `ignore` so a parse
failure can never corrupt the memory store.

Contract:
- `parsed`      strict parser accepted the raw output; decision passed through
- `repaired`    strict parse failed, deterministic repair produced a decision
                that passes full schema validation
- `failed_safe` unrepairable; canonical `ignore` decision returned (store
                nothing, log the row)

Repairs are field-local only (coerce a malformed Q: numeric, drop a broken F:
line, synthesize a missing R: from C:). Word-salad content stays word salad —
the repair pass is a *product* metric; the Gate 4 free-decode bar remains the
*model* metric. Track both, never blend them.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psm_model.lean_format import parse_tagged_decision
from psm_model.schema import ACTIONS, MEMORY_TYPES, validate_storage_decision

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

FAILSAFE_DECISION: dict[str, Any] = {
    "action": "ignore",
    "memory": None,
    "facts": [],
    "reasoning": "fail-safe: model output unparseable; storing nothing",
}


@dataclass
class RepairResult:
    status: str  # parsed | repaired | failed_safe
    decision: dict[str, Any]
    repairs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _unescape(value: str) -> str:
    output: list[str] = []
    escaping = False
    for char in value:
        if escaping:
            output.append({"n": "\n", "p": "|", "c": ","}.get(char, char))
            escaping = False
        elif char == "\\":
            escaping = True
        else:
            output.append(char)
    if escaping:
        output.append("\\")
    return "".join(output)


def _coerce_float(raw: str, repairs: list[str], label: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    match = _FLOAT_RE.search(raw)
    if match:
        repairs.append(f"coerced_{label}")
        try:
            return float(match.group(0))
        except ValueError:
            return None
    repairs.append(f"dropped_{label}")
    return None


def _snake_case(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned


def _repair_fact(value: str, repairs: list[str]) -> dict[str, Any] | None:
    parts = value.split("|")
    if len(parts) < 6:
        repairs.append("dropped_malformed_fact")
        return None
    if len(parts) > 6:
        parts = parts[:5] + ["|".join(parts[5:])]
    subject, predicate, fact_value, confidence, inference_kind, evidence = (_unescape(p) for p in parts)
    if not subject.strip() or not fact_value.strip() or not evidence.strip():
        repairs.append("dropped_malformed_fact")
        return None
    if not _PREDICATE_RE.fullmatch(predicate):
        fixed = _snake_case(predicate)
        if fixed and _PREDICATE_RE.fullmatch(fixed):
            repairs.append("snake_cased_predicate")
            predicate = fixed
        else:
            repairs.append("dropped_malformed_fact")
            return None
    if inference_kind != "explicit":
        repairs.append("forced_explicit_kind")
        inference_kind = "explicit"
    return {
        "subject": subject,
        "predicate": predicate,
        "value": fact_value,
        "confidence": _coerce_float(confidence, repairs, "fact_confidence"),
        "inference_kind": inference_kind,
        "evidence_text": evidence,
    }


def _lenient_rebuild(raw: str, repairs: list[str]) -> dict[str, Any]:
    """Rebuild a decision from raw tagged output, keeping only salvageable lines."""
    action: str | None = None
    memory: dict[str, Any] | None = None
    memory_none = False
    facts: list[dict[str, Any]] = []
    reasoning: str | None = None

    def mem() -> dict[str, Any]:
        nonlocal memory
        if memory is None:
            memory = {}
        return memory

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "END":
            break
        key, sep, value = line.partition(":")
        if not sep:
            repairs.append("dropped_untagged_line")
            continue
        if key == "A":
            if value in ACTIONS:
                action = value
        elif key == "M" and value == "-":
            memory_none = True
        elif key == "T":
            if value in MEMORY_TYPES:
                mem()["type"] = value
            else:
                repairs.append("dropped_bad_memory_type")
        elif key == "C":
            if value.strip():
                mem()["content"] = _unescape(value)
        elif key == "Q":
            parts = value.split(",")
            if len(parts) > 4:
                repairs.append("truncated_q_quad")
                parts = parts[:4]
            for fname, raw_field in zip(("strength", "decay_rate", "emotional_weight", "confidence"), parts):
                number = _coerce_float(raw_field, repairs, f"q_{fname}")
                if number is not None:
                    mem()[fname] = number
        elif key == "G":
            tags = [_unescape(item) for item in value.split(",") if item.strip()]
            if tags:
                mem()["tags"] = tags
        elif key == "TE":
            mem()["temporal_expression"] = _unescape(value)
        elif key == "RT":
            mem()["resolved_time"] = _unescape(value)
        elif key == "F":
            fact = _repair_fact(value, repairs)
            if fact is not None:
                facts.append(fact)
        elif key == "R":
            if value.strip():
                reasoning = _unescape(value)
        else:
            repairs.append("dropped_unknown_tag")

    if memory_none:
        memory = None
    if reasoning is None:
        content = (memory or {}).get("content")
        if content:
            repairs.append("synthesized_reasoning_from_content")
            reasoning = f"Stored: {content}"
        elif action == "ignore":
            repairs.append("synthesized_reasoning_generic")
            reasoning = "Nothing durable to store."
    return {"action": action, "memory": memory, "facts": facts, "reasoning": reasoning}


def _canonicalize_action_constraints(decision: dict[str, Any], repairs: list[str]) -> dict[str, Any]:
    normalized = dict(decision)
    if normalized.get("action") == "ignore":
        if normalized.get("memory") is not None:
            normalized["memory"] = None
            repairs.append("cleared_ignore_memory")
        if not str(normalized.get("reasoning") or "").strip():
            normalized["reasoning"] = "Nothing durable to store."
            repairs.append("synthesized_ignore_reasoning")
    return normalized


def repair_storage_decision(raw: str) -> RepairResult:
    decision, issues = parse_tagged_decision(raw)
    if decision is not None:
        repairs: list[str] = []
        canonical = _canonicalize_action_constraints(decision, repairs)
        if canonical != decision:
            return RepairResult(status="repaired", decision=canonical, repairs=repairs, issues=[])
        return RepairResult(status="parsed", decision=decision)

    repairs = []
    rebuilt = _canonicalize_action_constraints(_lenient_rebuild(raw, repairs), repairs)
    result = validate_storage_decision(rebuilt)
    if result.ok:
        return RepairResult(
            status="repaired",
            decision=rebuilt,
            repairs=repairs,
            issues=[f"{i.path}: {i.message}" for i in issues],
        )
    return RepairResult(
        status="failed_safe",
        decision=dict(FAILSAFE_DECISION),
        repairs=repairs,
        issues=[f"{i.path}: {i.message}" for i in result.issues],
    )


def evaluate_report(report_path: Path, *, samples: int = 0) -> dict[str, Any]:
    """Product-level metrics: re-run every row of an eval report through the repair pass."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [r for r in report.get("reports", []) if not r.get("skipped")]
    counts = {"parsed": 0, "repaired": 0, "failed_safe": 0, "no_raw_output": 0}
    repaired_action_correct = 0
    repaired_rows: list[dict[str, Any]] = []
    repair_ops: dict[str, int] = {}

    for row in rows:
        raw = row.get("raw_output")
        if raw is None:
            # raw_output only captured for parse fails in some reports; a strict
            # parse pass implies the repair pass would pass it through.
            counts["parsed" if row.get("parse_valid") else "no_raw_output"] += 1
            continue
        result = repair_storage_decision(raw)
        counts[result.status] += 1
        for op in result.repairs:
            repair_ops[op] = repair_ops.get(op, 0) + 1
        if result.status == "repaired":
            correct = result.decision.get("action") == row.get("expected_action")
            repaired_action_correct += int(correct)
            repaired_rows.append(
                {
                    "id": row.get("id"),
                    "expected_action": row.get("expected_action"),
                    "repaired_action": result.decision.get("action"),
                    "action_correct": correct,
                    "repairs": result.repairs,
                }
            )

    total = sum(counts.values())
    product_valid = counts["parsed"] + counts["repaired"]
    summary = {
        "report": str(report_path),
        "rows": total,
        "model_parse_valid_rate": report.get("parse_valid_rate"),
        "product_parse_valid_rate": product_valid / total if total else 0.0,
        "counts": counts,
        "repaired_action_accuracy": (repaired_action_correct / counts["repaired"]) if counts["repaired"] else None,
        "repair_ops": dict(sorted(repair_ops.items(), key=lambda kv: -kv[1])),
        "repaired_rows": repaired_rows if samples else [r["id"] for r in repaired_rows],
    }
    if samples:
        summary["repaired_rows"] = repaired_rows[:samples]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Eval report JSON with raw_output captured.")
    parser.add_argument("--samples", type=int, default=0, help="Include N repaired-row details.")
    args = parser.parse_args()
    print(json.dumps(evaluate_report(args.report, samples=args.samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
