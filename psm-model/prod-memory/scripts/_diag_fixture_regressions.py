#!/usr/bin/env python3
"""Compare prod fixture eval cases across checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PACKAGE_ROOT / "checkpoints/_hf_verify/eval/hf-prod-v5n-qwen0.5b-prod-grounding.json"
DEFAULT_CANDIDATE = PACKAGE_ROOT / "results/hf-prod-v5p-qwen0.5b-prod-grounding.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_id(report: dict) -> dict[str, dict]:
    return {str(c["id"]): c for c in report.get("cases", []) if isinstance(c, dict)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--base-label", default="v5n")
    parser.add_argument("--candidate-label", default="v5p")
    args = parser.parse_args()

    base = _by_id(_load(args.base))
    cand = _by_id(_load(args.candidate))
    rows: list[dict] = []
    for case_id in sorted(set(base) | set(cand)):
        b = base.get(case_id, {})
        c = cand.get(case_id, {})
        rows.append(
            {
                "id": case_id,
                "suite": b.get("suite") or c.get("suite"),
                "expect": b.get("expectAction") or c.get("expectAction"),
                args.base_label: {
                    "effective_stored": b.get("effective_stored"),
                    "repair": b.get("repair_status"),
                    "action": b.get("action"),
                    "fail_safe": b.get("fail_safe"),
                    "raw_len": len(str(b.get("raw_output") or "")),
                },
                args.candidate_label: {
                    "effective_stored": c.get("effective_stored"),
                    "repair": c.get("repair_status"),
                    "action": c.get("action"),
                    "fail_safe": c.get("fail_safe"),
                    "raw_len": len(str(c.get("raw_output") or "")),
                },
                "delta": (
                    "regressed"
                    if b.get("effective_stored") and not c.get("effective_stored")
                    else "improved"
                    if c.get("effective_stored") and not b.get("effective_stored")
                    else "same"
                ),
            }
        )

    base_eff = sum(1 for r in rows if r[args.base_label].get("effective_stored"))
    cand_eff = sum(1 for r in rows if r[args.candidate_label].get("effective_stored"))
    regressed = [r["id"] for r in rows if r["delta"] == "regressed"]
    shared_fail = [
        r["id"]
        for r in rows
        if not r[args.base_label].get("effective_stored") and not r[args.candidate_label].get("effective_stored")
    ]
    out = {
        "base": args.base_label,
        "candidate": args.candidate_label,
        "base_effective_stored": base_eff,
        "candidate_effective_stored": cand_eff,
        "regressed_ids": regressed,
        "shared_failure_ids": shared_fail,
        "cases": rows,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
