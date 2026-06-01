import unittest
from pathlib import Path

from psm_model.convert_nano_dataset import convert_nano_row, split_rows


class ConvertNanoDatasetTests(unittest.TestCase):
    def test_converts_storage_row_to_canonical_training_row(self):
        raw = {
            "id": "row-1",
            "input": {
                "operation": "remember",
                "source_kind": "unit",
                "source_id": "source-1",
                "prior_context": [{"speaker": "User", "text": "Earlier context."}],
                "current_turn": {"speaker": "User", "text": "I prefer SQLite.", "timestamp": "2026-06-01T00:00:00Z"},
            },
            "output": {
                "action": "promote_semantic",
                "memory": {
                    "content": "The user prefers SQLite.",
                    "type": "semantic",
                    "confidence": 0.9,
                    "tags": ["preference"],
                },
                "facts": [
                    {
                        "subject": "user",
                        "predicate": "prefers",
                        "value": "SQLite",
                        "confidence": 0.9,
                        "inference_kind": "explicit",
                        "evidence_text": "I prefer SQLite.",
                    }
                ],
                "updates": [],
                "indexables": [],
                "reasoning": "The user stated a durable preference.",
            },
        }

        row, reason = convert_nano_row(raw, source_path=Path("source.jsonl"), line_number=1)

        self.assertEqual(reason, "")
        self.assertIsNotNone(row)
        self.assertTrue(row["id"].startswith("row-1:"))
        self.assertIn("Earlier context.", row["input"]["conversation"])
        self.assertEqual(row["expected"]["action"], "promote_semantic")
        self.assertNotIn("indexables", row["expected"])

    def test_skips_recall_context_for_storage_model(self):
        row, reason = convert_nano_row(
            {"id": "recall", "input": {"current_turn": {"text": "what do I know?"}}, "output": {"action": "recall_context"}},
            source_path=Path("source.jsonl"),
            line_number=1,
        )

        self.assertIsNone(row)
        self.assertEqual(reason, "unsupported_action:recall_context")

    def test_split_rows_is_deterministic(self):
        rows = [{"id": f"row-{index}"} for index in range(50)]

        first = split_rows(rows, validation_ratio=0.1, test_ratio=0.1)
        second = split_rows(rows, validation_ratio=0.1, test_ratio=0.1)

        self.assertEqual(first, second)
        self.assertEqual(sum(len(value) for value in first.values()), 50)


if __name__ == "__main__":
    unittest.main()
