import unittest

from psm_model.safe_generate import calibrate_action, constrained_decision
from psm_model.schema import validate_storage_decision


class SafeGenerateTests(unittest.TestCase):
    def test_constrained_ignore_is_valid(self):
        decision = constrained_decision("ignore", {"conversation": "User: okay thanks"})

        result = validate_storage_decision(decision)

        self.assertTrue(result.ok, result.issues)
        self.assertIsNone(decision["memory"])

    def test_constrained_non_ignore_is_valid_and_extractive(self):
        decision = constrained_decision(
            "promote_semantic",
            {"conversation": "User: I prefer concise technical answers.", "source_timestamp": "2026-06-03T12:00:00Z"},
        )

        result = validate_storage_decision(decision)

        self.assertTrue(result.ok, result.issues)
        self.assertEqual(decision["action"], "promote_semantic")
        self.assertIn("I prefer concise technical answers", decision["memory"]["content"])
        self.assertEqual(decision["facts"][0]["evidence_text"], "I prefer concise technical answers.")

    def test_calibrate_action_overrides_obvious_direct_cases(self):
        cases = [
            (
                "store_episodic",
                {"conversation": "User: I prefer concise technical answers."},
                "promote_semantic",
            ),
            (
                "update_existing",
                {"conversation": "User: okay thanks haha and the weather outside is cloudy."},
                "ignore",
            ),
            (
                "promote_semantic",
                {"conversation": "User: Today I met Dana at 3pm to review the roadmap."},
                "store_episodic",
            ),
            (
                "store_episodic",
                {"conversation": "User: Correction: use PowerShell one-line commands instead.", "context": "old memory"},
                "update_existing",
            ),
        ]

        for model_action, payload, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(calibrate_action(model_action, payload), expected)


if __name__ == "__main__":
    unittest.main()
