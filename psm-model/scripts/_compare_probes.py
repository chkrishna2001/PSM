#!/usr/bin/env python3
"""Compare v5h vs v5m probe cases."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def load(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            cid = r["case"]["id"]
            out[cid] = r
    return out

v5h = load(REPO / "benchmark/locomo/results/probe-locomo-hf-storage-v5h-json-cpu.jsonl")
v5m = load(REPO / "benchmark/locomo/results/probe-locomo-hf-storage-v5m-json-cpu.jsonl")

for cid in sorted(v5h):
    h, m = v5h[cid], v5m[cid]
    exp = h["case"]["expect_store"]
    ha = h["analysis"]["action"]
    ma = m["analysis"]["action"]
    hp = h["analysis"]["checks"]["parse_ok"]
    mp = m["analysis"]["checks"]["parse_ok"]
    hf = h["analysis"]["checks"]["has_facts"]
    mf = m["analysis"]["checks"]["has_facts"]
    hsm = h["analysis"]["checks"]["store_decision_match"]
    msm = m["analysis"]["checks"]["store_decision_match"]
    hmem = (h["analysis"].get("memory_content") or "")[:80]
    mmem = (m["analysis"].get("memory_content") or "")[:80]
    delta = "same"
    if msm and not hsm:
        delta = "IMPROVED"
    elif hsm and not msm:
        delta = "REGRESSED"
    elif mp and not hp:
        delta = "parse+"
    elif hp and not mp:
        delta = "parse-"
    print(f"{cid:16} {h['case']['bucket']:12} exp={str(exp):5} h={ha:16} m={ma:16} hsm={hsm} msm={msm} delta={delta}")
    if delta != "same":
        print(f"  h_mem: {hmem}")
        print(f"  m_mem: {mmem}")
