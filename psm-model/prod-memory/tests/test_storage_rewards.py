import unittest

from prod_memory.build_v5o_storage_dpo_rows import build_v5o_storage_dpo_rows
from prod_memory.hf_prompts import storage_inference_messages_from_input, storage_llm_response_from_input
from prod_memory.storage_rewards import chosen_rejected_gap, score_storage_decision


class StorageRewardTests(unittest.TestCase):
    def test_chosen_beats_wrong_action(self):
        remember = "User prefers dark mode and 14px font in the editor."
        expected = {
            "action": "store_episodic",
            "memory": {"content": "User prefers dark mode and 14px font.", "type": "episodic"},
            "facts": [],
            "indexables": [],
            "reasoning": "durable preference",
        }
        chosen = (
            '{"action":"store_episodic","facts":[],"indexables":[],"memory":{"content":"User prefers dark mode and 14px font.",'
            '"type":"episodic"},"reasoning":"durable preference"}'
        )
        rejected = (
            '{"action":"ignore","facts":[],"indexables":[],"memory":null,"reasoning":"wrong gate"}'
        )
        gap = chosen_rejected_gap(chosen, rejected, remember_target=remember, expected=expected)
        self.assertGreater(gap, 0.2)
        self.assertGreater(
            score_storage_decision(remember_target=remember, expected=expected, raw=chosen)["reward"],
            score_storage_decision(remember_target=remember, expected=expected, raw=rejected)["reward"],
        )

    def test_cli_input_matches_train_flatten(self):
        payload = {
            "operation": "remember_llm_response",
            "conversation": [{"role": "assistant", "content": "Ship PSM with JSON storage decisions."}],
        }
        self.assertEqual(storage_llm_response_from_input(payload), "Ship PSM with JSON storage decisions.")
        messages = storage_inference_messages_from_input(payload, output_format="json")
        self.assertIn("Ship PSM with JSON storage decisions.", messages[1]["content"])
        self.assertNotIn("User:", messages[1]["content"])


class V5oDpoRowTests(unittest.TestCase):
    def test_builds_pairs(self):
        rows = build_v5o_storage_dpo_rows(include_fixtures=True, include_noise=False)
        self.assertGreater(len(rows), 20)
        sample = rows[0]
        self.assertIn("prompt", sample)
        self.assertIn("chosen", sample)
        self.assertIn("rejected", sample)


if __name__ == "__main__":
    unittest.main()
