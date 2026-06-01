import unittest

from psm_model.eval_generation import _fact_signature, _memory_content


class EvalMetricTests(unittest.TestCase):
    def test_memory_content_uses_actual_content(self):
        decision = {"memory": {"content": "The user prefers SQLite."}}

        self.assertEqual(_memory_content(decision), "The user prefers SQLite.")
        self.assertIsNone(_memory_content({"memory": None}))

    def test_fact_signature_includes_evidence_text(self):
        decision = {
            "facts": [
                {
                    "subject": "user",
                    "predicate": "prefers",
                    "value": "SQLite",
                    "evidence_text": "I prefer SQLite.",
                }
            ]
        }

        self.assertEqual(_fact_signature(decision), [("user", "prefers", "SQLite", "I prefer SQLite.")])


if __name__ == "__main__":
    unittest.main()
