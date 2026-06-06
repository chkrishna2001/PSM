from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from psm_model.action_diagnostics import evaluate_action_prefixes, score_actions
from psm_model.generate import generate_storage_json, load_checkpoint_metadata
from psm_model.model import TinyDecoderModel
from psm_model.prompts import render_storage_prompt
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import ACTION_ORDER, resolve_device


_ACTION_RE = re.compile(r"^A:(ignore|store_episodic|promote_semantic|update_existing|flag_conflict|flag_and_store)\s*$", re.MULTILINE)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Action smoke requires PyTorch. Install torch to run psm_model.action_smoke.") from exc
    return torch


def parse_action_output(raw: str) -> str | None:
    match = _ACTION_RE.search(raw.strip())
    return match.group(1) if match else None


def load_probe_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "case" in row and "input" in row:
            rows.append(
                {
                    "id": row["case"],
                    "input": row["input"],
                    "expected_action": row.get("expected", {}).get("action") or row.get("expected_action"),
                }
            )
            continue
        expected = row.get("expected", {})
        rows.append(
            {
                "id": row.get("id") or row.get("case") or "row",
                "input": row["input"],
                "expected_action": expected.get("action") if isinstance(expected, dict) else row.get("expected_action"),
            }
        )
    return rows


def smoke_checkpoint(
    checkpoint: Path,
    probes: list[dict[str, Any]],
    *,
    output_format: str | None = None,
    device: str = "auto",
    max_new_tokens: int = 48,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    metadata = load_checkpoint_metadata(checkpoint)
    active_format = output_format or str(metadata.get("output_format", "action"))
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = TinyDecoderModel.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    model.eval()

    cases = probes[:sample_limit] if sample_limit is not None else probes
    reports: list[dict[str, Any]] = []
    correct = 0
    parsed = 0
    with torch.no_grad():
        for case in cases:
            payload = case["input"]
            expected = case.get("expected_action")
            raw = generate_storage_json(
                checkpoint,
                payload,
                max_new_tokens=max_new_tokens,
                output_format=active_format,
                device=str(device_obj),
            )
            scores = score_actions(model, tokenizer, payload, output_format=active_format, device=device_obj)
            ranked = sorted(scores.items(), key=lambda item: item[1])
            prefix_action = ranked[0][0]
            parsed_action = parse_action_output(raw)
            if parsed_action is not None:
                parsed += 1
            if expected and (parsed_action == expected or prefix_action == expected):
                correct += int(parsed_action == expected or prefix_action == expected)
            reports.append(
                {
                    "id": case["id"],
                    "expected_action": expected,
                    "prefix_action": prefix_action,
                    "parsed_action": parsed_action,
                    "prefix_ok": expected == prefix_action if expected else None,
                    "parsed_ok": expected == parsed_action if expected else None,
                    "raw": raw.strip(),
                    "top3_prefix_scores": [
                        {"action": action, "loss": round(loss, 4)} for action, loss in ranked[:3]
                    ],
                    "conversation_preview": str(payload.get("conversation", ""))[:160],
                }
            )

    total = len(cases)
    return {
        "checkpoint": str(checkpoint),
        "output_format": active_format,
        "device": str(device_obj),
        "cases": total,
        "parsed_rate": parsed / total if total else 0.0,
        "match_rate": correct / total if total else 0.0,
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualitative action smoke: raw generation + prefix scores vs expected.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("probes", type=Path, help="manual-probe.jsonl or any JSONL with input/expected.action")
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--prefix-eval", action="store_true", help="Also print full-prefix eval summary for the probe file.")
    args = parser.parse_args()

    probes = load_probe_rows(args.probes)
    report = smoke_checkpoint(
        args.checkpoint,
        probes,
        output_format=args.output_format,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        sample_limit=args.sample_limit,
    )
    if args.prefix_eval:
        prefix_report = evaluate_action_prefixes(
            args.checkpoint,
            args.probes,
            output_format=report["output_format"],
            device=args.device,
        )
        report["prefix_eval"] = {
            "macro_action_prefix_accuracy": prefix_report["macro_action_prefix_accuracy"],
            "per_action_accuracy": prefix_report["per_action_accuracy"],
            "predicted_action_counts": prefix_report["predicted_action_counts"],
            "collapse_fraction": prefix_report["collapse_fraction"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
