import importlib.util
import unittest

from psm_model.model import TinyDecoderConfig, TinyDecoderModel
from psm_model.tokenizer import ByteTokenizer


HAS_TORCH = importlib.util.find_spec("torch") is not None


class KvCacheGenerateTests(unittest.TestCase):
    def test_cached_and_uncached_generate_match(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")
        import torch

        config = TinyDecoderConfig(vocab_size=ByteTokenizer().vocab_size, context_length=64, n_layer=2, n_head=2, n_embd=32)
        model = TinyDecoderModel(config)
        tokenizer = ByteTokenizer()
        prompt = tokenizer.encode("User: store this preference.", add_bos=True)
        input_ids = torch.tensor([prompt], dtype=torch.long)

        cached = model.generate(input_ids, max_new_tokens=12, eos_id=tokenizer.eos_id, temperature=0.0, use_kv_cache=True)
        uncached = model.generate(input_ids.clone(), max_new_tokens=12, eos_id=tokenizer.eos_id, temperature=0.0, use_kv_cache=False)

        self.assertEqual(cached.tolist(), uncached.tolist())

    def test_tagged_end_stops_early(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")
        import torch

        self.assertTrue(TinyDecoderModel._tagged_generation_complete("A:ignore\nM:-\nR:test\nEND"))
        self.assertFalse(TinyDecoderModel._tagged_generation_complete("A:ignore\nM:-\nR:test"))


if __name__ == "__main__":
    unittest.main()
