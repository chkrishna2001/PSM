import json
import tempfile
import unittest
from pathlib import Path

from psm_model.filter_by_token_budget import filter_jsonl_by_token_budget
from psm_model.tokenizer import ByteTokenizer


class FilterByTokenBudgetTests(unittest.TestCase):
    def test_filters_rows_over_budget(self):
        short = {
            "id": "short",
            "input": {"conversation": "User: okay thanks"},
            "expected": {"action": "ignore", "memory": None, "facts": [], "reasoning": "No durable memory."},
        }
        long = {
            "id": "long",
            "input": {"conversation": "User: " + ("x" * 1000)},
            "expected": {"action": "ignore", "memory": None, "facts": [], "reasoning": "No durable memory."},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.jsonl"
            output_path = temp / "output.jsonl"
            tokenizer_path = temp / "tokenizer.json"
            ByteTokenizer().save(tokenizer_path)
            input_path.write_text("\n".join(json.dumps(row) for row in [short, long]) + "\n", encoding="utf-8")

            report = filter_jsonl_by_token_budget(input_path, output_path, tokenizer_path=tokenizer_path, max_tokens=600)

            kept = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["kept"], 1)
        self.assertEqual(report["dropped"], 1)
        self.assertEqual(kept[0]["id"], "short")


if __name__ == "__main__":
    unittest.main()
