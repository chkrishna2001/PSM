import json
import tempfile
import unittest
from pathlib import Path

from psm_model.data import load_jsonl_rows, validate_training_row


VALID_EXPECTED = {
    "action": "store_episodic",
    "memory": {
        "content": "The user migrated the cache store to Redis.",
        "type": "episodic",
        "confidence": 0.9,
        "tags": ["redis"],
    },
    "facts": [
        {
            "subject": "user",
            "predicate": "migrated",
            "value": "cache store to Redis",
            "confidence": 0.9,
            "inference_kind": "explicit",
            "evidence_text": "I migrated the cache store to Redis.",
        }
    ],
    "reasoning": "The message describes a durable project event.",
}


class DatasetRowTests(unittest.TestCase):
    def test_valid_training_row_passes(self):
        row, issues = validate_training_row(
            {
                "id": "row-1",
                "input": {"conversation": "User: I migrated the cache store to Redis."},
                "expected": VALID_EXPECTED,
                "source": "unit",
                "split": "train",
            }
        )

        self.assertEqual(issues, ())
        self.assertIsNotNone(row)
        self.assertEqual(row.expected.action, "store_episodic")

    def test_training_row_requires_prompt_payload(self):
        row, issues = validate_training_row({"id": "row-1", "input": {}, "expected": VALID_EXPECTED})

        self.assertIsNone(row)
        self.assertIssue(issues, "$.input")

    def test_training_row_reports_expected_schema_errors(self):
        expected = json.loads(json.dumps(VALID_EXPECTED))
        expected["facts"][0]["inference_kind"] = "inferred"

        row, issues = validate_training_row(
            {
                "id": "row-1",
                "input": {"conversation": "User: maybe Redis was involved."},
                "expected": expected,
            }
        )

        self.assertIsNone(row)
        self.assertIssue(issues, "$.expected.facts[0].inference_kind")

    def test_jsonl_gate_reports_counts_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            rows = [
                {"id": "row-1", "input": {"conversation": "User: I migrated Redis."}, "expected": VALID_EXPECTED},
                {"id": "row-1", "input": {"conversation": "User: I migrated Redis again."}, "expected": VALID_EXPECTED},
                {"id": "bad", "input": {}, "expected": VALID_EXPECTED},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            report = load_jsonl_rows(path)

        self.assertFalse(report.ok)
        self.assertEqual(report.total, 3)
        self.assertEqual(report.valid, 2)
        self.assertEqual(report.action_counts, {"store_episodic": 2})
        self.assertEqual(report.memory_type_counts, {"episodic": 2})
        self.assertEqual(report.duplicate_ids, ("row-1",))
        self.assertEqual(report.failures[0]["id"], "bad")

    def assertIssue(self, issues, path):
        self.assertIn(path, [issue.path for issue in issues])


if __name__ == "__main__":
    unittest.main()

