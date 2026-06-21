import unittest

from prod_memory.hf_prompts import row_messages
from prod_memory.row_validation import remember_target_from_input


class HfPromptTests(unittest.TestCase):
    def test_storage_uses_raw_llm_response_not_json_wrapper(self):
        llm = "## Plan\n\nChunk long assistant handoffs near 600-1200 tokens per chunk."
        row = {
            "input": {
                "operation": "remember_llm_response",
                "conversation": [{"role": "assistant", "content": llm}],
            },
            "expected": {
                "action": "store_episodic",
                "memory": {"content": "Chunk long assistant handoffs near 600-1200 tokens per chunk.", "type": "episodic"},
                "facts": [],
                "indexables": [],
                "reasoning": "x",
            },
        }
        messages = row_messages(row, output_format="tagged")
        user = messages[1]["content"]
        self.assertIn(llm, user)
        self.assertNotIn('"operation": "remember_llm_response"', user)
        self.assertEqual(remember_target_from_input(row["input"]), llm)

    def test_recall_uses_json_plan(self):
        row = {
            "input": {
                "operation": "recall_plan",
                "question": "What are my theme preferences?",
                "available_tables": ["episodic", "semantic", "archival"],
                "requested_top_k": 5,
            },
            "expected": {
                "intent": "recall",
                "target_tables": ["semantic"],
                "filters": {},
                "ranking_hints": ["theme"],
                "temporal_intent": None,
                "top_k": 5,
            },
        }
        messages = row_messages(row, output_format="tagged")
        self.assertIn("recall plan", messages[1]["content"].lower())
        self.assertIn('"intent":"recall"', messages[2]["content"].replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
