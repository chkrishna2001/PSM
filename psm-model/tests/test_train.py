import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from psm_model.generate import load_checkpoint_metadata
from psm_model.tokenizer import ByteTokenizer
from psm_model.train import _encode_training_text, build_lm_batch, load_training_texts, overfit_texts
from psm_model.model import TinyDecoderConfig


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TrainTests(unittest.TestCase):
    def test_load_training_texts_from_probe_shape(self):
        row = {
            "id": "row-1",
            "input": {"conversation": "User: okay thanks"},
            "expected": {"action": "ignore", "memory": None, "facts": [], "reasoning": "No durable memory."},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")

            texts = load_training_texts(path, output_format="at_tag")

        self.assertEqual(len(texts), 1)
        self.assertIn("<|assistant|>\n", texts[0])
        self.assertIn("@a ignore", texts[0])
        self.assertTrue(texts[0].endswith("<|end|>"))

    def test_build_lm_batch_when_torch_available(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        input_ids, labels = build_lm_batch(["abc"], ByteTokenizer(), context_length=8)

        self.assertEqual(tuple(input_ids.shape), (1, 8))
        self.assertEqual(tuple(labels.shape), (1, 8))

    def test_build_lm_batch_rejects_truncation(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        with self.assertRaises(ValueError):
            build_lm_batch(["abcdef"], ByteTokenizer(), context_length=3)

    def test_encode_training_text_masks_prompt(self):
        ids, mask = _encode_training_text(ByteTokenizer(), "<|system|>\nx\n<|assistant|>\nA:ignore\nEND<|end|>")

        self.assertEqual(len(ids), len(mask))
        self.assertIn(True, mask)
        self.assertFalse(mask[0])
        first_answer = mask.index(True)
        self.assertGreater(first_answer, 0)

    def test_overfit_uses_supplied_tokenizer_vocab(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        class ToyTokenizer(ByteTokenizer):
            @property
            def vocab_size(self):
                return super().vocab_size + 10

        tokenizer = ToyTokenizer()
        config = TinyDecoderConfig(vocab_size=tokenizer.vocab_size, context_length=32, n_layer=1, n_head=1, n_embd=16)

        model, losses = overfit_texts(["<|assistant|>\nA:ignore\nEND"], config=config, tokenizer=tokenizer, steps=1)

        self.assertEqual(model.config.vocab_size, tokenizer.vocab_size)
        self.assertEqual(len(losses), 1)

    def test_missing_checkpoint_metadata_defaults_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = load_checkpoint_metadata(Path(temp_dir) / "missing.pt")

        self.assertEqual(metadata, {})


if __name__ == "__main__":
    unittest.main()
