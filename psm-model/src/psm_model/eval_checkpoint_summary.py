from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from psm_model.gate_checkpoint import evaluate_checkpoint_gate
from psm_model.probe_checkpoint import _load_probe_file, probe_checkpoint


DEFAULT_MANUAL_PROBE = Path("psm-model/data/direct-behavior-v1/manual-probe.jsonl")


def summarize_checkpoint(checkpoint: Path, *, device: str = "cpu", manual_probe: Path = DEFAULT_MANUAL_PROBE) -> dict[str, Any]:
    gate = evaluate_checkpoint_gate(checkpoint, device=device, manual_probe=manual_probe)
    raw = probe_checkpoint(checkpoint, _load_probe_file(manual_probe), device=device)
    return {
        "checkpoint": str(checkpoint),
        "passed": gate["passed"],
        "failures": gate["failures"],
        "metrics": gate["metrics"],
        "prefix_failures": {
            "foundation": len(gate["details"]["foundation"]["failures"]),
            "concept": len(gate["details"]["concept"]["failures"]),
            "fast_mixed": len(gate["details"]["fast_mixed"]["failures"]),
        },
        "manual_action_head": {
            "accuracy": gate["metrics"].get("manual_action_head_accuracy"),
            "macro_accuracy": gate["metrics"].get("manual_macro_action_head_accuracy"),
            "predicted_action_counts": gate["details"]["manual_action_head"]["predicted_action_counts"],
            "failure_count": len(gate["details"]["manual_action_head"]["failures"]),
        },
        "manual_safe": {
            "expected_action_accuracy": gate["metrics"].get("manual_safe_expected_action_accuracy"),
            "valid_rate": gate["metrics"].get("manual_safe_valid_rate"),
            "model_action_accuracy": gate["metrics"].get("manual_model_action_accuracy"),
        },
        "manual_raw": {
            "expected_action_accuracy": raw["expected_action_accuracy"],
            "valid_rate": raw["valid_rate"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact post-training PSM checkpoint evaluation summary.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--manual-probe", type=Path, default=DEFAULT_MANUAL_PROBE)
    args = parser.parse_args()

    summary = summarize_checkpoint(args.checkpoint, device=args.device, manual_probe=args.manual_probe)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
