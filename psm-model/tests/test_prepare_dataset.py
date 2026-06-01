import json
import tempfile
import unittest
from pathlib import Path

from psm_model.prepare_dataset import prepare_jsonl


class PrepareDatasetTests(unittest.TestCase):
    def test_prepare_jsonl_writes_training_text(self):
        row = {
            "id": "probe-ignore",
            "input": {"conversation": "User: okay thanks"},
            "expected": {
                "action": "ignore",
                "memory": None,
                "facts": [],
                "reasoning": "The message has no durable memory value.",
            },
            "source": "unit",
            "split": "probe",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            output_path = Path(temp_dir) / "output.jsonl"
            input_path.write_text(json.dumps(row), encoding="utf-8")

            written = prepare_jsonl(input_path, output_path, output_format="at_tag")
            output = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(written, 1)
        self.assertEqual(output["id"], "probe-ignore")
        self.assertEqual(output["source"], "unit")
        self.assertEqual(output["split"], "probe")
        self.assertIn("<|assistant|>\n", output["text"])
        self.assertIn("@a ignore", output["text"])
        self.assertTrue(output["text"].endswith("<|end|>"))

    def test_prepare_jsonl_fails_invalid_rows(self):
        row = {
            "id": "bad",
            "input": {},
            "expected": {
                "action": "ignore",
                "memory": None,
                "facts": [],
                "reasoning": "No durable memory.",
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            output_path = Path(temp_dir) / "output.jsonl"
            input_path.write_text(json.dumps(row), encoding="utf-8")

            with self.assertRaises(ValueError):
                prepare_jsonl(input_path, output_path)


if __name__ == "__main__":
    unittest.main()
