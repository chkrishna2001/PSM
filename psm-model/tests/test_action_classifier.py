import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from psm_model.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    checkpoint_path_for_step,
    encode_action_batch,
    evaluate_examples,
    load_action_examples,
    render_action_input,
)
from psm_model.train import ACTION_TO_ID
from psm_model.tokenizer import ByteTokenizer


HAS_TORCH = importlib.util.find_spec("torch") is not None


class ActionClassifierTests(unittest.TestCase):
    def test_load_action_examples_supports_training_and_probe_shapes(self):
        rows = [
            {"id": "row-1", "input": {"conversation": "User: I prefer concise answers."}, "expected": {"action": "promote_semantic"}},
            {"case": "row-2", "text": "User: okay thanks", "expected_action": "ignore"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            examples = load_action_examples(path)

        self.assertEqual([example.action for example in examples], ["promote_semantic", "ignore"])
        self.assertEqual(examples[1].input_payload["conversation"], "User: okay thanks")

    def test_render_action_input_keeps_context_and_conversation(self):
        text = render_action_input({"conversation": "User: Correction: use short updates.", "context": "old memory", "operation": "remember"})

        self.assertIn("conversation: User: Correction", text)
        self.assertIn("context: old memory", text)
        self.assertIn("operation: remember", text)

    def test_encode_action_batch_pads_and_labels(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        examples = load_examples_inline([("a", "User: okay thanks", "ignore")])
        input_ids, labels = encode_action_batch(examples, ByteTokenizer(), context_length=32, device="cpu")

        self.assertEqual(tuple(input_ids.shape), (1, 32))
        self.assertEqual(labels.tolist(), [ACTION_TO_ID["ignore"]])

    def test_model_forward_returns_action_logits(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        examples = load_examples_inline([("a", "User: okay thanks", "ignore")])
        input_ids, labels = encode_action_batch(examples, ByteTokenizer(), context_length=32, device="cpu")
        model = ActionClassifier(ActionClassifierConfig(vocab_size=ByteTokenizer().vocab_size, context_length=32, n_embd=16, hidden_size=16))

        result = model(input_ids, labels=labels)

        self.assertEqual(tuple(result["logits"].shape), (1, 6))
        self.assertIsNotNone(result["loss"])

    def test_evaluate_examples_reports_macro_accuracy(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        examples = load_examples_inline([("a", "User: okay thanks", "ignore")])
        model = ActionClassifier(ActionClassifierConfig(vocab_size=ByteTokenizer().vocab_size, context_length=32, n_embd=16, hidden_size=16))

        report = evaluate_examples(model, ByteTokenizer(), examples, device="cpu")

        self.assertEqual(report["examples"], 1)
        self.assertIn("macro_action_accuracy", report)
        self.assertIn("predicted_action_counts", report)

    def test_checkpoint_path_for_step_normalizes_existing_step_suffix(self):
        self.assertEqual(
            checkpoint_path_for_step(Path("psm-model/checkpoints/action-step-000100.pt"), 300).name,
            "action-step-000300.pt",
        )


def load_examples_inline(rows):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "rows.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"id": row_id, "input": {"conversation": text}, "expected": {"action": action}})
                for row_id, text, action in rows
            )
            + "\n",
            encoding="utf-8",
        )
        return load_action_examples(path)


if __name__ == "__main__":
    unittest.main()
