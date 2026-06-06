from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psm_model.label_audit import audit_label_risks, row_label_issues


class LabelAuditTests(unittest.TestCase):
    def test_flags_semantic_wording_labeled_episodic(self) -> None:
        row = {
            "id": "bad-episodic",
            "input": {"conversation": "User: User prefers direct evidence before storing memory.", "source_kind": "local_psm_db"},
            "expected": {"action": "store_episodic", "memory": {"type": "episodic"}},
        }

        reasons = {issue.reason for issue in row_label_issues(row, action="store_episodic", memory_type="episodic")}

        self.assertIn("semantic_wording_labeled_store_episodic", reasons)
        self.assertIn("local_psm_semantic_signal_labeled_store_episodic", reasons)

    def test_rule_like_future_text_is_allowed_as_semantic(self) -> None:
        row = {
            "id": "rule",
            "input": {"conversation": "User: For future coding tasks, always gate datasets before training."},
            "expected": {"action": "promote_semantic", "memory": {"type": "semantic"}},
        }

        reasons = {issue.reason for issue in row_label_issues(row, action="promote_semantic", memory_type="semantic")}

        self.assertNotIn("episodic_wording_labeled_promote_semantic", reasons)

    def test_audit_report_groups_examples_by_reason(self) -> None:
        rows = [
            {
                "id": "bad-episodic",
                "input": {"conversation": "User: User prefers concise answers.", "source_kind": "local_psm_db"},
                "expected": {"action": "store_episodic", "memory": {"type": "episodic"}},
            },
            {
                "id": "good-ignore",
                "input": {"conversation": "User: okay thanks haha and weather outside is cloudy."},
                "expected": {"action": "ignore"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            report = audit_label_risks(path)

        self.assertEqual(report["rows"], 2)
        self.assertIn("semantic_wording_labeled_store_episodic", report["examples_by_reason"])


if __name__ == "__main__":
    unittest.main()
