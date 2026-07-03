#!/usr/bin/env python3
"""One-shot diagnostics for train/probe format gap (ponytail: delete after handoff)."""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "psm-model/prod-memory/data"
LOCOMO_MARKERS = ("Current utterance", "Session time", "Source id:", "Previous context:")


def classify_input(inp: dict) -> str:
    conv = inp.get("conversation")
    if isinstance(conv, list):
        return "dict_conversation"
    if isinstance(conv, str):
        if any(m in conv for m in LOCOMO_MARKERS):
            return "locomo_flat"
        return "flat_other"
    return "other"


def audit_v3() -> dict:
    counts: Counter[str] = Counter()
    total = 0
    sample_dict = None
    with (DATA / "prod-extraction-v3.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            kind = classify_input(row.get("input") or {})
            counts[kind] += 1
            if kind == "dict_conversation" and sample_dict is None:
                sample_dict = row
    return {"total": total, "counts": dict(counts), "sample_dict": sample_dict}


def count_locomo_curriculum(name: str) -> tuple[int, int]:
    p = DATA / name
    n = locomo = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            src = str(row.get("source", ""))
            rid = str(row.get("id", ""))
            if "locomo" in src.lower() or "locomo" in rid.lower():
                locomo += 1
            elif classify_input(row.get("input") or {}) == "locomo_flat":
                locomo += 1
    return n, locomo


def compare_probes() -> list[dict]:
    def load(path: Path) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
        return out

    v5h = load(REPO / "benchmark/locomo/results/probe-locomo-hf-storage-v5h-json-cpu.jsonl")
    v5m = load(REPO / "benchmark/locomo/results/probe-locomo-hf-storage-v5m-json-cpu.jsonl")
    rows: list[dict] = []
    for cid in sorted(v5h):
        h, m = v5h[cid], v5m[cid]

        def action(r: dict) -> str:
            return str((r.get("parsed") or {}).get("action") or "?")

        def store_ok(r: dict, expect_store: bool) -> bool:
            a = action(r)
            if expect_store:
                return a in ("store_episodic", "promote_semantic")
            return a == "ignore"

        exp = bool(h.get("expect_store"))
        h_ok = store_ok(h, exp)
        m_ok = store_ok(m, exp)
        h_parse = bool(h.get("parse_ok"))
        m_parse = bool(m.get("parse_ok"))
        h_facts = len((h.get("parsed") or {}).get("facts") or []) > 0
        m_facts = len((m.get("parsed") or {}).get("facts") or []) > 0
        if m_ok and not h_ok:
            delta = "improved"
        elif h_ok and not m_ok:
            delta = "regressed"
        elif m_parse and not h_parse:
            delta = "parse_up"
        elif h_parse and not m_parse:
            delta = "parse_down"
        else:
            delta = "same"
        rows.append(
            {
                "id": cid,
                "bucket": h.get("bucket"),
                "expect_store": exp,
                "v5h_action": action(h),
                "v5m_action": action(m),
                "v5h_parse": h_parse,
                "v5m_parse": m_parse,
                "v5h_facts": h_facts,
                "v5m_facts": m_facts,
                "delta": delta,
            }
        )
    return rows


def main() -> None:
    v3 = audit_v3()
    print("=== prod-extraction-v3 ===")
    print(f"total={v3['total']}")
    for k, v in sorted(v3["counts"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100 * v / v3['total']:.1f}%)")

    for name in ("hf-prod-v5h.jsonl", "hf-prod-v5m.jsonl", "hf-prod-v5h-locomo.jsonl"):
        n, loc = count_locomo_curriculum(name)
        print(f"=== {name}: total={n} locomo={loc} ===")

    v5h_ids: set[str] = set()
    v5m_ids: set[str] = set()
    for name, s in (("hf-prod-v5h.jsonl", v5h_ids), ("hf-prod-v5m.jsonl", v5m_ids)):
        with (DATA / name).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s.add(json.loads(line)["id"])
    extra = v5m_ids - v5h_ids
    extra_rows = []
    with (DATA / "hf-prod-v5m.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["id"] in extra:
                    extra_rows.append(r)
    print(f"v5m extra vs v5h: {len(extra)}")
    print(f"  actions: {dict(Counter(r['expected']['action'] for r in extra_rows))}")
    print(f"  sources: {dict(Counter(str(r.get('source', ''))[:50] for r in extra_rows).most_common(5))}")

    print("=== probe matrix ===")
    for row in compare_probes():
        print(json.dumps(row, ensure_ascii=False))

    # training example
    from prod_memory.row_validation import remember_target_from_input
    from prod_memory.hf_prompts import storage_inference_messages

    sample = v3["sample_dict"]
    if sample:
        train_text = remember_target_from_input(sample["input"])
        print("=== TRAIN remember_target (first dict row) ===")
        print(train_text[:600])
        msgs = storage_inference_messages(train_text[:400] + "...", output_format="json")
        print("=== TRAIN user prefix (truncated) ===")
        print(msgs[1]["content"][:500])

    # probe example from script
    sys.path.insert(0, str(REPO / "psm-model/scripts"))
    from probe_locomo_hf_storage_cpu import build_cases, DEFAULT_DATA

    cases = build_cases(DEFAULT_DATA)
    if cases:
        c = next(x for x in cases if x.get("dia_id") == "D1:3")
        print("=== PROBE build_locomo_remember_text (conv-26 D1:3) ===")
        print(c["llm_response"][:700])
        print("=== PROBE user prefix (truncated) ===")
        print(storage_inference_messages(c["llm_response"][:400] + "...", output_format="json")[1]["content"][:500])


if __name__ == "__main__":
    main()
