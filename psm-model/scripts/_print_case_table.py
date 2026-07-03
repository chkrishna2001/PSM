#!/usr/bin/env python3
"""Print prod fixture case table from a grounding eval report."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "psm-model/prod-memory/results/hf-prod-v5q-qwen0.5b-prod-grounding.json"
r = json.load(open(path, encoding="utf-8"))
print(json.dumps(r["aggregate"], indent=2))
for c in r["cases"]:
    print(
        f"{c['id']:22} expect={str(c.get('expectAction')):6} action={str(c.get('action')):16} "
        f"eff={c['effective_stored']} repair={c.get('repair_status')} guard={c.get('guard_route')}"
    )
    if c.get("issues"):
        print("   issue:", c["issues"][0])
    raw = c.get("raw_output") or ""
    if '"indexables":[{' in raw.replace(" ", ""):
        print("   -> emits indexables[]")
