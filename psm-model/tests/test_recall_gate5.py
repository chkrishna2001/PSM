import json
import tempfile
import unittest
from pathlib import Path

from psm_model.build_gate5_train_v1 import build_gate5_train_v1
from psm_model.data import load_jsonl_rows, validate_training_row
from psm_model.generate_recall_curriculum import build_recall_probe_rows
from psm_model.gates import RECALL_PROBE_THRESHOLDS, gate_report
from psm_model.prompts import render_training_text, row_task
from psm_model.recall_schema import score_recall_plan, validate_recall_plan


VALID_STORAGE = {
    "action": "store_episodic",
    "memory": {
        "content": "The user prefers dark mode.",
        "type": "semantic",
        "confidence": 0.9,
        "tags": ["preference"],
    },
    "facts": [],
    "reasoning": "Stable preference.",
}

VALID_RECALL = {
    "intent": "recall",
    "target_tables": ["episodic"],
    "filters": {},
    "ranking_hints": ["Melanie", "painting"],
    "temporal_intent": "2022",
    "top_k": 5,
}


class RecallGate5Tests(unittest.TestCase):
    def test_validate_recall_plan_row(self):
        row, issues = validate_training_row(
            {
                "id": "recall-1",
                "task": "recall_plan",
                "input": {
                    "operation": "recall_plan",
                    "question": "What painting did Melanie share from 2022?",
                    "available_tables": ["episodic", "semantic", "archival"],
                    "requested_top_k": 5,
                },
                "expected": VALID_RECALL,
            }
        )
        self.assertEqual(issues, ())
        self.assertIsNotNone(row)
        self.assertEqual(row.task, "recall_plan")

    def test_render_recall_training_text_uses_json_output(self):
        input_payload = {
            "operation": "recall_plan",
            "question": "What painting did Melanie share from 2022?",
            "available_tables": ["episodic", "semantic", "archival"],
            "requested_top_k": 5,
        }
        text = render_training_text(input_payload, VALID_RECALL, output_format="tagged")
        self.assertEqual(row_task(input_payload), "recall_plan")
        self.assertIn('"target_tables":["episodic"]', text.replace(" ", ""))
        self.assertNotIn("A:store_episodic", text)

    def test_score_recall_plan_tables_and_hints(self):
        scores = score_recall_plan(
            VALID_RECALL,
            {
                "intent": "recall",
                "target_tables": ["episodic", "semantic"],
                "filters": {},
                "ranking_hints": ["Melanie", "art"],
                "top_k": 5,
            },
        )
        self.assertTrue(scores["parse_valid"])
        self.assertFalse(scores["target_tables_exact"])
        self.assertTrue(scores["target_tables_primary"])
        self.assertGreater(scores["ranking_hints_score"], 0.0)

    def test_recall_gate_thresholds(self):
        report = {
            "parse_valid_rate": 0.96,
            "schema_valid_rate": 0.96,
            "target_tables_exact_rate": 0.91,
            "target_tables_primary_rate": 0.96,
            "ranking_hints_score": 0.55,
            "top_k_exact_rate": 0.92,
        }
        gate = gate_report(report, RECALL_PROBE_THRESHOLDS)
        self.assertTrue(gate["passed"])

    def test_build_gate5_curriculum_mixes_storage_and_recall(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expanded = Path(temp_dir) / "expanded.jsonl"
            expanded.write_text(
                json.dumps(
                    {
                        "id": "storage-1",
                        "input": {"conversation": "User: I prefer dark mode."},
                        "expected": VALID_STORAGE,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = Path(temp_dir) / "gate5.jsonl"
            summary = build_gate5_train_v1(
                output,
                expanded_probes=expanded,
                direct_probes=None,
                expanded_copies=10,
                recall_copies=2,
                recall_rows=build_recall_probe_rows()[:4],
            )
            self.assertTrue(summary["dataset_gate"]["ok"])
            self.assertGreater(summary["storage_fraction"], 0.4)
            self.assertGreater(summary["recall_fraction"], 0.1)
            gate = load_jsonl_rows(output)
            self.assertIn("storage", gate.task_counts)
            self.assertIn("recall_plan", gate.task_counts)


if __name__ == "__main__":
    unittest.main()
