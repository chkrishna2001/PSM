import unittest

from psm_model.lean_format import encode_binary_decision, parse_binary_decision
from psm_model.prompts import render_expected_output, render_training_text


class BinaryFormatTests(unittest.TestCase):
    def test_encode_ignore(self):
        self.assertEqual(
            encode_binary_decision({"action": "ignore", "memory": None, "facts": [], "reasoning": "x"}),
            "ignore",
        )

    def test_encode_store(self):
        self.assertEqual(
            encode_binary_decision({"action": "store_episodic", "memory": {"content": "x"}, "facts": [], "reasoning": "x"}),
            "store",
        )

    def test_roundtrip(self):
        parsed, issues = parse_binary_decision("store")
        self.assertFalse(issues)
        self.assertEqual(parsed["action"], "store_episodic")

    def test_training_text(self):
        expected = {"action": "store_episodic", "memory": {"content": "x", "type": "episodic"}, "facts": [], "reasoning": "x"}
        text = render_training_text(
            {"operation": "remember_llm_response", "conversation": [{"role": "assistant", "content": "hi"}]},
            expected,
            output_format="binary",
        )
        self.assertTrue(text.endswith("store<|end|>"))


if __name__ == "__main__":
    unittest.main()
