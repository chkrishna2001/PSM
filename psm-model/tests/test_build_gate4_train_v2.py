import json
import tempfile
import unittest
from pathlib import Path

from psm_model.analyze_eval_report import classify_row
from psm_model.build_gate4_train_v2 import build_gate4_train_v2
from psm_model.generate_action_foundation_curriculum import expected_memory
from psm_model.mine_gate4_parse_failures import mine_parse_failures


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


class MineGate4ParseFailuresTests(unittest.TestCase):
    def test_mines_parse_fail_rows_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(_memory_row("good-1", "promote_semantic")),
                        json.dumps(_memory_row("bad-1", "store_episodic")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            report = {
                "reports": [
                    {"id": "good-1", "parse_valid": True, "schema_valid": True, "expected_action": "promote_semantic", "predicted_action": "promote_semantic", "expected_memory_type": "semantic", "predicted_memory_type": "semantic", "memory_content_exact": True, "expected_fact_count": 1, "predicted_fact_count": 1, "facts_exact": True},
                    {"id": "bad-1", "parse_valid": False, "schema_valid": False, "expected_action": "store_episodic", "predicted_action": None, "issues": []},
                ]
            }
            report_path = root / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            rows, summary = mine_parse_failures(report_path, source)
            self.assertEqual(summary["repair_rows"], 1)
            self.assertEqual(rows[0]["id"], "gate4-parse-repair:bad-1")
            self.assertEqual(rows[0]["expected"]["action"], "store_episodic")


class BuildGate4TrainV2Tests(unittest.TestCase):
    def test_parse_focus_share_meets_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = root / "direct.jsonl"
            expanded = root / "expanded.jsonl"
            repair = root / "repair.jsonl"
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
            repair.write_text(
                "\n".join(json.dumps(_memory_row(f"repair-{index}", "promote_semantic")) for index in range(2)) + "\n",
                encoding="utf-8",
            )

            summary = build_gate4_train_v2(
                output,
                direct_probes=direct,
                expanded_probes=expanded,
                parse_repair=repair,
                direct_copies=2,
                expanded_copies=3,
                drill_rows_per_action=2,
                drill_copies=4,
                repair_copies=3,
            )

            self.assertEqual(summary["curriculum"], "gate4-train-v2")
            self.assertGreaterEqual(summary["mix_shares"]["parse_focus"], 0.40)
            self.assertGreater(summary["parse_repair_rows"], 0)
            self.assertGreater(summary["parse_drill_rows"], 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(any(row["id"].startswith("parse-repair:") for row in rows))


if __name__ == "__main__":
    unittest.main()
