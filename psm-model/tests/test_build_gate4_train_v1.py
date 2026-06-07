import json
import tempfile
import unittest
from pathlib import Path

from psm_model.build_gate4_train_v1 import build_gate4_train_v1
from psm_model.generate_action_foundation_curriculum import expected_memory


def _memory_row(row_id: str, action: str) -> dict:
    if action == "promote_semantic":
        expected = expected_memory(
            "promote_semantic",
            "semantic",
            "The user prefers concise answers.",
            ["preference", "answer_style"],
            "User",
            "prefers",
            "concise_answers",
            "I prefer concise answers.",
            "The user stated a durable preference.",
        )
    else:
        expected = expected_memory(
            "store_episodic",
            "episodic",
            "On 2026-06-01, the user met Dana.",
            ["event", "meeting"],
            "User",
            "meeting",
            "met_dana",
            "On 2026-06-01, I met Dana.",
            "The user described a completed dated event.",
            temporal_expression="2026-06-01",
            resolved_time="2026-06-01",
            decay_rate=0.05,
        )
    return {
        "id": row_id,
        "input": {
            "conversation": f"User: {row_id}",
            "operation": "remember",
            "source_id": row_id,
            "source_kind": "test",
            "source_timestamp": "2026-06-03T12:00:00Z",
        },
        "expected": expected,
    }


class BuildGate4TrainV1Tests(unittest.TestCase):
    def test_expanded_drills_and_stratified_without_base_dilution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct.jsonl"
            expanded = root / "expanded.jsonl"
            stratified = root / "stratified.jsonl"
            output = root / "out.jsonl"

            direct.write_text(
                json.dumps(
                    {
                        "id": "direct-1",
                        "input": {"conversation": "User: direct"},
                        "expected": {"action": "ignore", "memory": None, "facts": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expanded.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"expanded-{index}",
                            "input": {"conversation": f"User: expanded {index}"},
                            "expected": {
                                "action": "ignore" if index == 0 else "promote_semantic",
                                "memory": None,
                                "facts": [],
                            },
                        }
                    )
                    for index in range(3)
                )
                + "\n",
                encoding="utf-8",
            )
            stratified.write_text(
                "\n".join(json.dumps(_memory_row(f"real-{index}", "promote_semantic" if index % 2 == 0 else "store_episodic")) for index in range(4))
                + "\n",
                encoding="utf-8",
            )

            summary = build_gate4_train_v1(
                output,
                direct_probes=direct,
                expanded_probes=expanded,
                stratified_source=stratified,
                direct_copies=2,
                expanded_copies=3,
                drill_rows_per_action=2,
                drill_copies=2,
                stratified_max=4,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(summary["curriculum"], "gate4-train-v1")
            self.assertNotIn("base_rows", summary)
            self.assertEqual(summary["expanded_full_rows"], 9)
            self.assertEqual(summary["direct_anchor_rows"], 2)
            self.assertGreater(summary["parse_drill_rows"], 0)
            self.assertEqual(summary["stratified_real_rows"], 4)
            self.assertGreater(summary["mix_shares"]["expanded_full"], summary["mix_shares"]["stratified_real"])
            self.assertTrue(any(row["id"].startswith("expanded-full:") for row in rows))
            self.assertTrue(any(row["id"].startswith("parse-drill:") for row in rows))
            self.assertFalse(any(row.get("source", "").startswith("gate4_curriculum:base") for row in rows))


if __name__ == "__main__":
    unittest.main()
