import importlib.util
import tempfile
import unittest
from pathlib import Path

from psm_model.model import TinyDecoderConfig, TinyDecoderModel
from psm_model.tokenizer import ByteTokenizer


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TinyDecoderModelTests(unittest.TestCase):
    def test_parameter_estimate_is_positive(self):
        config = TinyDecoderConfig(vocab_size=ByteTokenizer().vocab_size, context_length=64, n_layer=1, n_head=2, n_embd=32)

        estimate = TinyDecoderModel.parameter_estimate(config)

        self.assertGreater(estimate, 0)

    def test_config_rejects_bad_head_shape_when_torch_available(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        with self.assertRaises(ValueError):
            TinyDecoderModel(TinyDecoderConfig(vocab_size=32, n_head=3, n_embd=32))

    def test_forward_and_checkpoint_when_torch_available(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")
        import torch

        config = TinyDecoderConfig(vocab_size=ByteTokenizer().vocab_size, context_length=16, n_layer=1, n_head=2, n_embd=32)
        model = TinyDecoderModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        result = model(input_ids, labels=input_ids)

        self.assertEqual(result["logits"].shape, (2, 8, config.vocab_size))
        self.assertIsNotNone(result["loss"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pt"
            model.save_checkpoint(path)
            loaded = TinyDecoderModel.load_checkpoint(path)
        self.assertEqual(loaded.config, model.config)


if __name__ == "__main__":
    unittest.main()

