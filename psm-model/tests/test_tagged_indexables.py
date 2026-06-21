import unittest

from psm_model.lean_format import encode_tagged_decision, parse_tagged_decision


class TaggedIndexableTests(unittest.TestCase):
    def test_roundtrip_indexables(self):
        expected = {
            "action": "store_episodic",
            "memory": {
                "content": "Review PR with gh pr view and diff.",
                "type": "episodic",
                "strength": 0.86,
                "decay_rate": 0.02,
                "emotional_weight": 0.22,
                "confidence": 0.92,
                "tags": ["workflow", "review-pr"],
            },
            "facts": [],
            "indexables": [{
                "kind": "workflow",
                "key": "review-pr",
                "salience": 0.95,
                "reconstructive_hint": "Review PR with gh pr view and diff.",
                "evidence_text": "Review PR with gh pr view and diff.",
                "steps": ["get_pr_info", "review_files"],
            }],
            "reasoning": "Store workflow procedure.",
        }
        encoded = encode_tagged_decision(expected)
        self.assertIn("X:workflow|review-pr|", encoded)
        parsed, issues = parse_tagged_decision(encoded)
        self.assertFalse(issues)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["indexables"][0]["key"], "review-pr")
        self.assertEqual(parsed["indexables"][0]["steps"], ["get_pr_info", "review_files"])


if __name__ == "__main__":
    unittest.main()
