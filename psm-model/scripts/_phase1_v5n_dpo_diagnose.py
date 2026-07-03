#!/usr/bin/env python3
"""Phase 1 diagnosis for v5n-dpo: fixture case table, holdout decision audit, retrieval misses."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "benchmark/locomo/results/holdout-gate-v5n-dpo-conv-30-conv-41.db"
DEFAULT_RETRIEVAL = REPO / "benchmark/locomo/results/holdout-gate-v5n-dpo-conv-30-conv-41-retrieval.json"
DEFAULT_ANSWER = REPO / "benchmark/locomo/results/holdout-gate-v5n-dpo-conv-30-conv-41-answer.json"
DEFAULT_FIXTURES = REPO / "psm-model/prod-memory/fixtures/cases.json"
DEFAULT_FIXTURE_REPORT = REPO / "psm-model/prod-memory/results/hf-prod-v5n-dpo-qwen0.5b-prod-grounding.json"
DEFAULT_OUT = REPO / "docs/psm-model/2026-07-03-v5n-dpo-phase1-diagnosis.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_decisions(db_path: Path, *, sample_n: int, seed: int) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, action, route, substr(reasoning,1,120) AS reasoning, raw_json "
        "FROM decisions WHERE action != 'ignore' ORDER BY id"
    ).fetchall()
    conn.close()

    rng = random.Random(seed)
    sample = rows if len(rows) <= sample_n else rng.sample(rows, sample_n)

    stats = Counter()
    samples: list[dict[str, Any]] = []
    for row in sample:
        raw = row["raw_json"] or ""
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            stats["parse_fail"] += 1
            samples.append({"id": row["id"], "action": row["action"], "parse_fail": True})
            continue

        memory = decision.get("memory") if isinstance(decision.get("memory"), dict) else {}
        facts = decision.get("facts") if isinstance(decision.get("facts"), list) else []
        indexables = decision.get("indexables") if isinstance(decision.get("indexables"), list) else []

        facts_with_temporal = 0
        facts_with_resolved = 0
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            if fact.get("temporal_expression"):
                facts_with_temporal += 1
            if fact.get("resolved_time"):
                facts_with_resolved += 1

        mem_temporal = bool(memory.get("temporal_expression") or memory.get("resolved_time"))
        content = str(memory.get("content") or "")
        bleed = any(
            needle in content.lower()
            for needle in ("storage decision", "current utterance:", "locomo", "curriculum")
        )

        stats["stored_turns"] += 1
        stats[f"action:{row['action']}"] += 1
        if facts:
            stats["with_facts"] += 1
        else:
            stats["no_facts"] += 1
        if indexables:
            stats["model_indexables"] += 1
        if facts_with_temporal or mem_temporal:
            stats["any_temporal"] += 1
        if facts_with_resolved or memory.get("resolved_time"):
            stats["any_resolved"] += 1
        if bleed:
            stats["curriculum_bleed_hint"] += 1

        samples.append(
            {
                "id": row["id"],
                "action": row["action"],
                "facts_n": len(facts),
                "indexables_n": len(indexables),
                "facts_temporal_n": facts_with_temporal,
                "facts_resolved_n": facts_with_resolved,
                "memory_temporal": mem_temporal,
                "memory_content_snip": content[:160] or None,
                "curriculum_bleed_hint": bleed,
                "fact_predicates": [
                    str(f.get("predicate"))
                    for f in facts[:5]
                    if isinstance(f, dict) and f.get("predicate")
                ],
            }
        )

    total_stored = len(rows)
    n = max(1, stats["stored_turns"])
    return {
        "db": str(db_path),
        "stored_decisions_total": total_stored,
        "sample_n": len(sample),
        "counts": dict(sorted(stats.items())),
        "rates": {
            "with_facts_rate": round(stats["with_facts"] / n, 4),
            "model_indexables_rate": round(stats["model_indexables"] / n, 4),
            "any_temporal_rate": round(stats["any_temporal"] / n, 4),
            "curriculum_bleed_hint_rate": round(stats["curriculum_bleed_hint"] / n, 4),
        },
        "samples": samples,
    }


def _retrieval_bucket(record: dict[str, Any]) -> str:
    evidence = record.get("evidence") if isinstance(record.get("evidence"), list) else []
    selected = record.get("selected_ids") if isinstance(record.get("selected_ids"), list) else []
    if not evidence:
        return "no_evidence_label"
    if record.get("hit_at_1"):
        return "hit"
    in_top_k = bool(record.get("hit_at_k"))
    ev_set = set(str(x) for x in evidence)
    sel_set = set(str(x) for x in selected)
    if ev_set & sel_set:
        return "evidence_in_top_k_not_rank1" if in_top_k else "evidence_partial_in_top_k"
    return "evidence_missing_from_top_k"


def retrieval_slice(retrieval_path: Path, answer_path: Path | None) -> dict[str, Any]:
    retrieval = _load_json(retrieval_path)
    records = retrieval.get("records") if isinstance(retrieval.get("records"), list) else []
    buckets = Counter(_retrieval_bucket(r) for r in records if isinstance(r, dict))

    misses = [r for r in records if isinstance(r, dict) and not r.get("hit_at_1")]
    miss_samples = [
        {
            "sample_id": r.get("sample_id"),
            "category": r.get("category"),
            "question": r.get("question"),
            "gold_answer": r.get("gold_answer"),
            "evidence": r.get("evidence"),
            "selected_ids": (r.get("selected_ids") or [])[:5],
            "bucket": _retrieval_bucket(r),
            "hit_at_k": r.get("hit_at_k"),
        }
        for r in misses[:40]
    ]

    by_cat = Counter(str(r.get("category")) for r in misses if isinstance(r, dict))
    answer_wrong: list[dict[str, Any]] = []
    if answer_path and answer_path.is_file():
        answer = _load_json(answer_path)
        for r in answer.get("records") or []:
            if not isinstance(r, dict):
                continue
            if r.get("answer_judgment") in ("incorrect", "wrong", "fail"):
                answer_wrong.append(
                    {
                        "sample_id": r.get("sample_id"),
                        "category": r.get("category"),
                        "question": r.get("question"),
                        "gold_answer": r.get("gold_answer"),
                        "model_answer": (r.get("model_answer") or r.get("answer") or "")[:200],
                        "hit_at_1": r.get("hit_at_1"),
                        "hit_at_k": r.get("hit_at_k"),
                    }
                )

    return {
        "retrieval_path": str(retrieval_path),
        "summary": retrieval.get("summary"),
        "questions": len(records),
        "misses": len(misses),
        "bucket_counts": dict(buckets),
        "miss_by_category": dict(by_cat),
        "miss_samples": miss_samples,
        "answer_wrong_n": len(answer_wrong),
        "answer_wrong_samples": answer_wrong[:20],
    }


def fixture_section(report_path: Path, fixtures_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        return {"status": "missing", "path": str(report_path)}
    report = _load_json(report_path)
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    fixture = _load_json(fixtures_path)
    by_id = {c["id"]: c for c in fixture.get("cases", []) if isinstance(c, dict) and c.get("id")}

    table = []
    for row in cases:
        if not isinstance(row, dict):
            continue
        fid = row.get("id")
        expect = row.get("expectAction") or (by_id.get(fid) or {}).get("expectAction")
        table.append(
            {
                "id": fid,
                "suite": row.get("suite"),
                "expectAction": expect,
                "action": row.get("action"),
                "effective_stored": row.get("effective_stored"),
                "content_grounded": row.get("content_grounded"),
                "guard_rejected": row.get("guard_rejected"),
                "guard_route": row.get("guard_route"),
                "fail_safe": row.get("fail_safe"),
                "curriculum_bleed": row.get("curriculum_bleed"),
                "repair_status": row.get("repair_status"),
                "memory_content": row.get("memory_content"),
            }
        )

    return {
        "status": "ok",
        "path": str(report_path),
        "checkpoint": report.get("checkpoint"),
        "timestamp": report.get("timestamp"),
        "max_new_tokens": report.get("max_new_tokens"),
        "aggregate": report.get("aggregate"),
        "case_table": table,
    }


def failure_taxonomy(fixture: dict[str, Any], audit: dict[str, Any], retrieval: dict[str, Any]) -> list[str]:
    items: list[str] = []
    agg = fixture.get("aggregate") or {}
    if fixture.get("status") == "ok":
        eff = agg.get("effective_stored", 0)
        if eff < 7:
            items.append(f"prod_fixtures_below_deploy_bar: {eff}/10 effective_stored (need ≥7)")
        for row in fixture.get("case_table") or []:
            if row.get("expectAction") == "ignore" and row.get("effective_stored"):
                items.append(f"false_store: {row.get('id')}")
            if row.get("expectAction") == "store" and not row.get("effective_stored"):
                items.append(f"missed_store_or_guard: {row.get('id')} action={row.get('action')}")
            elif row.get("expectAction") == "store" and row.get("effective_stored") and not row.get("content_grounded"):
                items.append(f"ungrounded_store: {row.get('id')}")

    if audit.get("rates", {}).get("model_indexables_rate", 0) == 0:
        items.append("holdout: model never emits indexables[] — all indexables auto-built downstream")
    if audit.get("rates", {}).get("any_temporal_rate", 0) < 0.15:
        items.append("holdout: low temporal fill in model decisions (<15% sampled stored turns)")

    bc = retrieval.get("bucket_counts") or {}
    missing = bc.get("evidence_missing_from_top_k", 0)
    if missing:
        items.append(f"retrieval: {missing} questions with evidence absent from top-k")
    rank1 = bc.get("evidence_in_top_k_not_rank1", 0) + bc.get("evidence_partial_in_top_k", 0)
    if rank1:
        items.append(f"retrieval: {rank1} questions evidence in top-k but not rank-1 (ranking)")

    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 v5n-dpo diagnosis bundle.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--answer", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--fixtures-report", type=Path, default=DEFAULT_FIXTURE_REPORT)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    args.fixtures_report.parent.mkdir(parents=True, exist_ok=True)
    if not args.fixtures_report.is_file():
        try:
            from huggingface_hub import hf_hub_download

            cached = hf_hub_download(
                repo_id="krishnach7262/psm-prod-memory-hf",
                filename="eval/hf-prod-v5n-dpo-qwen0.5b-prod-grounding.json",
                repo_type="model",
            )
            args.fixtures_report.write_bytes(Path(cached).read_bytes())
            print(f"pulled fixture report -> {args.fixtures_report}", file=sys.stderr)
        except Exception as exc:
            print(f"fixture report missing and HF pull failed: {exc}", file=sys.stderr)

    fixture = fixture_section(args.fixtures_report, args.fixtures)
    audit = audit_decisions(args.db, sample_n=args.sample_n, seed=args.seed)
    retrieval = retrieval_slice(args.retrieval, args.answer if args.answer.is_file() else None)
    taxonomy = failure_taxonomy(fixture, audit, retrieval)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "hf-prod-v5n-dpo-qwen0.5b",
        "holdout_convs": ["conv-30", "conv-41"],
        "prod_fixtures": fixture,
        "holdout_decision_audit": audit,
        "holdout_retrieval": retrieval,
        "failure_taxonomy": taxonomy,
        "phase2_hints": [
            "SFT rows teaching indexables[] + temporal_expression on dated assistant turns",
            "DPO pairs for plan-01-handoff / workflow-runpod guard+grounding failures",
            "Retrieval ranking: episodic salience for multi-evidence category-1 questions",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "taxonomy": taxonomy}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
