import unittest

from psm_model.eval_generation import _parse_output, evaluate_model_rows
from psm_model.tokenizer import ByteTokenizer


class EvalGenerationTests(unittest.TestCase):
    def test_parse_tagged_output(self):
        raw = "\n".join(
            [
                "A:ignore",
                "M:-",
                "R:No durable memory.",
                "END",
            ]
        )

        parsed, issues = _parse_output(raw, "tagged")

        self.assertEqual(issues, ())
        self.assertEqual(parsed["action"], "ignore")

    def test_model_row_eval_requires_correct_action_and_content(self):
        class FakeModel:
            def generate(self, input_ids, *, max_new_tokens, eos_id, temperature):
                tokenizer = ByteTokenizer()
                raw = "\n".join(
                    [
                        "A:ignore",
                        "M:-",
                        "R:No durable memory.",
                        "END",
                        "<|end|>",
                    ]
                )
                return input_ids.new_tensor([tokenizer.encode("<|assistant|>\n" + raw, add_bos=True)])

        rows = [
            {
                "id": "expected-memory",
                "input": {"conversation": "User: I prefer SQLite."},
                "expected": {
                    "action": "promote_semantic",
                    "memory": {"type": "semantic", "content": "The user prefers SQLite."},
                    "facts": [
                        {
                            "subject": "user",
                            "predicate": "prefers",
                            "value": "SQLite",
                            "evidence_text": "I prefer SQLite.",
                        }
                    ],
                },
            }
        ]

        report = evaluate_model_rows(FakeModel(), ByteTokenizer(), rows, output_format="tagged", max_new_tokens=20)

        self.assertEqual(report["parse_valid_rate"], 1.0)
        self.assertEqual(report["schema_valid_rate"], 1.0)
        self.assertEqual(report["action_accuracy"], 0.0)
        self.assertEqual(report["memory_content_exact_rate"], 0.0)
        self.assertEqual(report["facts_exact_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
