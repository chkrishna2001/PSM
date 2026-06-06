from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psm_model.filter_label_risks import filter_label_risks


class FilterLabelRisksTests(unittest.TestCase):
    def test_drops_high_risk_rows_and_keeps_medium_by_default(self) -> None:
        rows = [
            {
                "id": "high",
                "input": {"conversation": "User: User prefers direct evidence before storing memory.", "source_kind": "local_psm_db"},
                "expected": {"action": "store_episodic", "memory": {"type": "episodic"}},
            },
            {
                "id": "medium",
                "input": {"conversation": "User: I might always run tests later if there is time."},
                "expected": {"action": "ignore"},
            },
            {
                "id": "clean",
                "input": {"conversation": "User: Today I met Dana at 3pm."},
                "expected": {"action": "store_episodic", "memory": {"type": "episodic"}},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.jsonl"
            output_path = Path(tmp) / "output.jsonl"
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = filter_label_risks(input_path, output_path)
            kept = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["input_rows"], 3)
        self.assertEqual(report["dropped_rows"], 1)
        self.assertEqual([row["id"] for row in kept], ["medium", "clean"])


if __name__ == "__main__":
    unittest.main()
