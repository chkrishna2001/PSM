from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from psm_model.build_gate4_curriculum import _copy_rows, _load_rows
from psm_model.build_gate5_train_v1 import _copy_task_rows
from psm_model.data import load_jsonl_rows
from psm_model.data.rows import infer_row_task
from psm_model.generate_recall_curriculum import build_recall_probe_rows

GATE6_PROFILES: dict[str, dict[str, int]] = {
    "conversation-bridge": {
        "expanded_copies": 25,
        "direct_copies": 50,
        "recall_copies": 500,
        "synthetic_copies": 8,
        "nano_copies": 3,
        "chatgpt_copies": 2,
    }
}


def resolve_gate6_profile(
    profile: str | None,
    *,
    expanded_copies: int | None = None,
    direct_copies: int | None = None,
    recall_copies: int | None = None,
    synthetic_copies: int | None = None,
    nano_copies: int | None = None,
    chatgpt_copies: int | None = None,
) -> dict[str, int]:
    base = GATE6_PROFILES.get(profile or "conversation-bridge", GATE6_PROFILES["conversation-bridge"])
    return {
        "expanded_copies": expanded_copies if expanded_copies is not None else base["expanded_copies"],
        "direct_copies": direct_copies if direct_copies is not None else base["direct_copies"],
        "recall_copies": recall_copies if recall_copies is not None else base["recall_copies"],
        "synthetic_copies": synthetic_copies if synthetic_copies is not None else base["synthetic_copies"],
        "nano_copies": nano_copies if nano_copies is not None else base["nano_copies"],
        "chatgpt_copies": chatgpt_copies if chatgpt_copies is not None else base["chatgpt_copies"],
    }


def _load_optional(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return _load_rows(path)


def _copy_conversation_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    copies: int,
    seen: set[str],
    output: list[dict[str, Any]],
) -> int:
    return _copy_task_rows(rows, prefix=prefix, copies=copies, seen=seen, output=output)


def build_gate6_train_v1(
    output: Path,
    *,
    expanded_probes: Path,
    recall_rows: list[dict[str, Any]] | None = None,
    direct_probes: Path | None = None,
    synthetic_rows: Path | None = None,
    nano_rows: Path | None = None,
    chatgpt_rows: Path | None = None,
    profile_copies: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    copies = profile_copies or resolve_gate6_profile("conversation-bridge")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    expanded_added = _copy_rows(
        _load_rows(expanded_probes),
        prefix="gate6-expanded",
        copies=copies["expanded_copies"],
        seen=seen,
        output=rows,
    )
    direct_added = 0
    if direct_probes and direct_probes.exists() and copies["direct_copies"] > 0:
        direct_added = _copy_rows(
            _load_rows(direct_probes),
            prefix="gate6-direct",
            copies=copies["direct_copies"],
            seen=seen,
            output=rows,
        )
    recall_source = recall_rows if recall_rows is not None else build_recall_probe_rows()
    recall_added = _copy_task_rows(
        recall_source,
        prefix="gate6-recall",
        copies=copies["recall_copies"],
        seen=seen,
        output=rows,
    )
    synthetic_added = _copy_conversation_rows(
        _load_optional(synthetic_rows),
        prefix="gate6-conv-synth",
        copies=copies["synthetic_copies"],
        seen=seen,
        output=rows,
    )
    nano_added = _copy_conversation_rows(
        _load_optional(nano_rows),
        prefix="gate6-conv-nano",
        copies=copies["nano_copies"],
        seen=seen,
        output=rows,
    )
    chatgpt_added = _copy_conversation_rows(
        _load_optional(chatgpt_rows),
        prefix="gate6-conv-chatgpt",
        copies=copies["chatgpt_copies"],
        seen=seen,
        output=rows,
    )

    rng = random.Random(seed)
    rng.shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    gate = load_jsonl_rows(output)
    task_counts = gate.task_counts
    storage_rows = task_counts.get("storage", 0)
    recall_total = sum(task_counts.get(key, 0) for key in ("recall_plan", "context_plan"))
    conversation_total = synthetic_added + nano_added + chatgpt_added
    return {
        "curriculum": "gate6-train-v1",
        "output": str(output),
        "rows": len(rows),
        "profile_copies": copies,
        "expanded_anchor_rows": expanded_added,
        "direct_anchor_rows": direct_added,
        "recall_anchor_rows": recall_added,
        "conversation_anchor_rows": conversation_total,
        "conversation_breakdown": {
            "synthetic": synthetic_added,
            "nano": nano_added,
            "chatgpt": chatgpt_added,
        },
        "task_counts": task_counts,
        "storage_fraction": storage_rows / len(rows) if rows else 0.0,
        "recall_fraction": recall_total / len(rows) if rows else 0.0,
        "conversation_storage_fraction": conversation_total / storage_rows if storage_rows else 0.0,
        "dataset_gate": gate.to_dict(),
        "action_counts": dict(
            sorted(
                Counter(
                    row["expected"]["action"]
                    for row in rows
                    if infer_row_task(row) == "storage" and isinstance(row.get("expected"), dict) and "action" in row["expected"]
                ).items()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Gate 6 curriculum with conversation bridge rows.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("psm-model/data/curriculum/psm-50m-gate6-train-v1.jsonl"),
    )
    parser.add_argument(
        "--expanded-probes",
        type=Path,
        default=Path("psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl"),
    )
    parser.add_argument("--direct-probes", type=Path, default=Path("psm-model/data/probes/direct_probes.jsonl"))
    parser.add_argument(
        "--conversation-synthetic",
        type=Path,
        default=Path("psm-model/data/curriculum/conversation-memory-synthetic-v1.jsonl"),
    )
    parser.add_argument(
        "--conversation-nano",
        type=Path,
        default=Path("psm-model/data/curriculum/conversation-memory-nano-v1.jsonl"),
    )
    parser.add_argument(
        "--conversation-chatgpt",
        type=Path,
        default=Path("psm-model/data/curriculum/conversation-memory-chatgpt-v1.jsonl"),
    )
    parser.add_argument("--profile", choices=sorted(GATE6_PROFILES), default="conversation-bridge")
    parser.add_argument("--expanded-copies", type=int, default=None)
    parser.add_argument("--direct-copies", type=int, default=None)
    parser.add_argument("--recall-copies", type=int, default=None)
    parser.add_argument("--synthetic-copies", type=int, default=None)
    parser.add_argument("--nano-copies", type=int, default=None)
    parser.add_argument("--chatgpt-copies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    copies = resolve_gate6_profile(
        args.profile,
        expanded_copies=args.expanded_copies,
        direct_copies=args.direct_copies,
        recall_copies=args.recall_copies,
        synthetic_copies=args.synthetic_copies,
        nano_copies=args.nano_copies,
        chatgpt_copies=args.chatgpt_copies,
    )
    direct_probes = args.direct_probes if copies["direct_copies"] > 0 else None
    summary = build_gate6_train_v1(
        args.out,
        expanded_probes=args.expanded_probes,
        direct_probes=direct_probes,
        synthetic_rows=args.conversation_synthetic,
        nano_rows=args.conversation_nano,
        chatgpt_rows=args.conversation_chatgpt,
        profile_copies=copies,
        seed=args.seed,
    )
    summary["profile"] = args.profile
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["dataset_gate"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
