import unittest

from psm_model.lean_format import encode_minimal_decision, parse_minimal_decision
from psm_model.prompts import render_expected_output, render_training_text


class MinimalFormatTests(unittest.TestCase):
    def test_encode_ignore(self):
        self.assertEqual(
            encode_minimal_decision({"action": "ignore", "memory": None, "facts": [], "reasoning": "x"}),
            "ignore",
        )

    def test_encode_store(self):
        out = encode_minimal_decision({
            "action": "store_episodic",
            "memory": {"content": "grounding and fixtures", "type": "episodic"},
            "facts": [],
            "reasoning": "x",
        })
        self.assertEqual(out, "store: grounding and fixtures")

    def test_roundtrip_store(self):
        expected = {
            "action": "store_episodic",
            "memory": {"content": "remember path rejects ungrounded storage", "type": "episodic"},
            "facts": [],
            "indexables": [],
            "reasoning": "x",
        }
        parsed, issues = parse_minimal_decision(render_expected_output(expected, output_format="minimal"))
        self.assertFalse(issues)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["memory"]["content"], expected["memory"]["content"])

    def test_training_text_uses_minimal_output(self):
        expected = {"action": "ignore", "memory": None, "facts": [], "reasoning": "none"}
        text = render_training_text(
            {"operation": "remember_llm_response", "conversation": [{"role": "assistant", "content": "ok"}]},
            expected,
            output_format="minimal",
        )
        self.assertTrue(text.endswith("ignore<|end|>"))


if __name__ == "__main__":
    unittest.main()
