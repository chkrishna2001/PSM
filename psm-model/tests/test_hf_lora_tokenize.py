import json
import unittest
from pathlib import Path

from psm_model.hf_lora_train import _tokenize_sft_row
from psm_model.lean_format import parse_tagged_decision


class HfLoraTokenizeTests(unittest.TestCase):
    def test_long_prompt_keeps_full_assistant_labels(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        assistant = "A:store_episodic\nT:episodic\nC:short grounded content\nQ:0.86,0.02,0.22,0.92\nR:ok\nEND"
        row = {
            "id": "t",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "x" * 12000},
                {"role": "assistant", "content": assistant},
            ],
        }
        item = _tokenize_sft_row(row, tokenizer, max_length=512)
        label_ids = [label for label in item["labels"] if label != -100]
        decoded = tokenizer.decode(label_ids, skip_special_tokens=True)
        parsed, issues = parse_tagged_decision(decoded)
        self.assertFalse(issues, msg=str(issues))
        self.assertEqual(parsed["action"], "store_episodic")


if __name__ == "__main__":
    unittest.main()
