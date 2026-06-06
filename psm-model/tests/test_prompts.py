import json
import unittest

from psm_model.prompts import render_expected_output, render_storage_prompt, render_training_text


class PromptTests(unittest.TestCase):
    def test_storage_prompt_has_boundaries_and_input_json(self):
        prompt = render_storage_prompt({"conversation": "User: I prefer SQLite."})

        self.assertTrue(prompt.startswith("<|system|>\n"))
        self.assertIn("<|user|>\n", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))
        self.assertIn('"conversation": "User: I prefer SQLite."', prompt)
        self.assertIn("tagged DSL", prompt)
        self.assertNotIn("produce the PSM storage JSON", prompt)

    def test_json_storage_prompt_requests_json(self):
        prompt = render_storage_prompt({"conversation": "User: I prefer SQLite."}, output_format="json")

        self.assertIn("strict JSON object", prompt)
        self.assertIn("produce the PSM storage JSON", prompt)

    def test_at_tag_storage_prompt_requests_at_tag_dsl(self):
        prompt = render_storage_prompt({"conversation": "User: I prefer SQLite."}, output_format="at_tag")

        self.assertIn("at-tag DSL", prompt)
        self.assertIn("@a", prompt)
        self.assertNotIn("produce the PSM storage JSON", prompt)

    def test_training_text_appends_compact_json_and_end_token(self):
        expected = {
            "action": "ignore",
            "memory": None,
            "facts": [],
            "reasoning": "No durable memory.",
        }

        text = render_training_text({"conversation": "User: okay thanks"}, expected, output_format="json")
        output_text = text.split("<|assistant|>\n", 1)[1].removesuffix("<|end|>")

        self.assertTrue(text.endswith("<|end|>"))
        self.assertEqual(json.loads(output_text), expected)
        self.assertNotIn("```", text)

    def test_training_text_supports_at_tag_output(self):
        expected = {
            "action": "ignore",
            "memory": None,
            "facts": [],
            "reasoning": "No durable memory.",
        }

        output = render_expected_output(expected, output_format="at_tag")
        text = render_training_text({"conversation": "User: okay thanks"}, expected, output_format="at_tag")

        self.assertIn("@a ignore", output)
        self.assertIn("@m none", output)
        self.assertTrue(text.endswith("@end<|end|>"))


if __name__ == "__main__":
    unittest.main()
