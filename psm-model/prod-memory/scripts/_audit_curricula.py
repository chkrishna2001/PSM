import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"

FILES = [
    "prod-extraction-v3.jsonl",
    "prod-extraction-v7.jsonl",
    "hf-prod-v2.jsonl",
    "hf-prod-v5h.jsonl",
    "hf-prod-v5b.jsonl",
    "hf-prod-v5k-extract.jsonl",
    "hf-prod-v5k-gate-distill.jsonl",
    "hf-prod-v5k-gate.jsonl",
    "hf-prod-v5i.jsonl",
]


def audit(name: str) -> None:
    p = ROOT / name
    if not p.exists():
        return
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    sources = Counter()
    facts = temporal = f_lines = te_lines = 0
    for r in rows:
        sources[str(r.get("source", "?"))[:50]] += 1
        exp = r.get("expected") or {}
        if isinstance(exp, dict):
            if exp.get("facts"):
                facts += 1
            mem = exp.get("memory") or {}
            if isinstance(mem, dict) and (
                mem.get("temporal_expression") or mem.get("resolved_time")
            ):
                temporal += 1
        assistant = "".join(
            m.get("content", "")
            for m in (r.get("messages") or [])
            if m.get("role") == "assistant"
        )
        if "F:" in assistant:
            f_lines += 1
        if "TE:" in assistant or "RT:" in assistant:
            te_lines += 1
    n = len(rows)
    print(f"=== {name} ({n} rows) ===")
    print(f"  facts in expected: {facts} ({100*facts/n:.0f}%)")
    print(f"  temporal in memory: {temporal} ({100*temporal/n:.0f}%)")
    print(f"  F: in assistant msg: {f_lines}")
    print(f"  TE/RT in assistant msg: {te_lines}")
    print(f"  top sources: {dict(sources.most_common(6))}")
    print()


for f in FILES:
    audit(f)

# manifests with output_format
print("=== MANIFEST output_format ===")
for m in sorted(ROOT.glob("*.manifest.json")):
    d = json.loads(m.read_text(encoding="utf-8"))
    fmt = d.get("output_format", "-")
    rows = d.get("total_rows", d.get("storage_rows", "?"))
    rf = d.get("rows_with_facts", d.get("storage_stats", {}).get("rows_with_facts", "?"))
    print(f"  {m.name}: format={fmt} rows={rows} facts={rf}")
