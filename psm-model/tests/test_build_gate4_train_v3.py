from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psm_model.build_gate4_fact_format_drills import build_fact_format_drills
from psm_model.build_gate4_train_v3 import build_gate4_train_v3
from psm_model.convert_chatgpt_exports import convert_chatgpt_exports
from psm_model.generate_action_foundation_curriculum import expected_memory, row


class BuildGate4TrainV3Tests(unittest.TestCase):
    def test_fact_format_drills_validate(self) -> None:
        drills = build_fact_format_drills(variants=3)
        self.assertGreaterEqual(len(drills), 10)
        for item in drills:
            self.assertEqual(item["expected"]["action"], "promote_semantic")

    def test_v3_curriculum_builds_with_minimal_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct.jsonl"
            expanded = root / "expanded.jsonl"
            drills = root / "drills.jsonl"
            out = root / "v3.jsonl"

            sample = {
                "id": "probe-1",
                "input": {
                    "conversation": "User: I prefer JSON.",
                    "operation": "remember",
                    "source_id": "x",
                    "source_kind": "probe",
                    "source_timestamp": "2026-06-03T12:00:00Z",
                },
                "expected": expected_memory(
                    "promote_semantic",
                    "semantic",
                    "The user prefers JSON.",
                    ["preference"],
                    "User",
                    "prefers",
                    "json",
                    "I prefer JSON.",
                    "Preference.",
                ),
            }
            direct.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            expanded.write_text(json.dumps(sample | {"id": "probe-2"}) + "\n", encoding="utf-8")
            drills.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in build_fact_format_drills(variants=1)
                ),
                encoding="utf-8",
            )

            summary = build_gate4_train_v3(
                out,
                direct_probes=direct,
                expanded_probes=expanded,
                fact_drills=drills,
                direct_copies=1,
                expanded_copies=2,
                fact_drill_copies=2,
                chatgpt_copies=0,
                stratified_max=0,
                repair_copies=0,
            )
            self.assertEqual(summary["curriculum"], "gate4-train-v3")
            self.assertGreater(summary["rows"], 0)
            self.assertGreater(summary["expanded_budget_rows"], 0)

    def test_chatgpt_converter_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows, summary = convert_chatgpt_exports(Path(tmp))
            self.assertEqual(rows, [])
            self.assertEqual(summary["rows"], 0)


if __name__ == "__main__":
    unittest.main()
