#!/usr/bin/env python3
"""Classify eval across gate LoRA checkpoints + optional base model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))
sys.path.insert(0, str(PACKAGE_ROOT))

from prod_memory.eval_classify import classify_match  # noqa: E402
from prod_memory.eval_grounding import DEFAULT_FIXTURES  # noqa: E402
from prod_memory.eval_hf_grounding import open_hf_session  # noqa: E402


def _eval_adapter(label: str, adapter_dir: Path | None, *, model: str, device: str) -> dict:
    cases = json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8")).get("cases", [])
    session = open_hf_session(adapter_dir, model_key=model, device=device)
    per: list[dict] = []
    hits = 0
    for case in cases:
        raw = session.generate(str(case["llmResponse"]), output_format="binary", max_new_tokens=16)
        exp = str(case.get("expectAction") or "store")
        ok = classify_match(exp, raw, output_format="binary")
        hits += int(ok)
        per.append({"id": case["id"], "expect": exp, "raw": raw.strip()[:80], "ok": ok})
    return {"label": label, "match": f"{hits}/10", "hits": hits, "cases": per}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. checkpoints/hf-prod-v5k-gate-distill-qwen0.5b")
    parser.add_argument("--model", default="qwen0.5b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    targets: list[tuple[str, Path | None]] = []
    if args.include_base:
        targets.append(("base-qwen0.5b", None))
    run_dir = args.run_dir
    if (run_dir / "adapter" / "adapter_model.safetensors").is_file():
        targets.append(("adapter", run_dir / "adapter"))
    for ckpt in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        if (ckpt / "adapter_model.safetensors").is_file():
            targets.append((ckpt.name, ckpt))
    if not targets:
        print(f"no adapter weights under {run_dir}", file=sys.stderr)
        return 1

    report: list[dict] = []
    best = ("", -1)
    for label, adapter in targets:
        print(f"=== {label} ===", flush=True)
        row = _eval_adapter(label, adapter, model=args.model, device=args.device)
        report.append(row)
        print(json.dumps({"label": label, "match": row["match"]}, indent=2), flush=True)
        if row["hits"] > best[1]:
            best = (label, row["hits"])

    summary = {"run_dir": str(run_dir), "best": best[0], "best_hits": best[1], "results": report}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
