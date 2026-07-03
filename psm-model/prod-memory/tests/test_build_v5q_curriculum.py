from __future__ import annotations

import unittest

from prod_memory.build_v5q_indexables_rows import build_v5q_anchor_rows
from prod_memory.grounding import has_curriculum_bleed, stored_text_from_decision
from prod_memory.row_validation import validate_prod_row


class BuildV5qCurriculumTests(unittest.TestCase):
    def test_anchor_rows_validate_and_most_have_indexables(self) -> None:
        rows = build_v5q_anchor_rows(boost_copies=2)
        self.assertGreaterEqual(len(rows), 20)
        with_indexables = 0
        for row in rows:
            validate_prod_row(row)
            expected = row["expected"]
            if expected.get("indexables"):
                with_indexables += 1
            action = str(expected.get("action") or "")
            if action not in {"ignore", "ignore_noise"}:
                self.assertFalse(
                    has_curriculum_bleed(stored_text_from_decision(expected)),
                    msg=str(row.get("id")),
                )
        self.assertGreater(with_indexables, len(rows) // 2)


if __name__ == "__main__":
    unittest.main()
