import unittest

from psm_model.configs import config_from_preset, describe_preset
from psm_model.tokenizer import ByteTokenizer


class ConfigPresetTests(unittest.TestCase):
    def test_50m_preset_is_near_target(self):
        report = describe_preset("50m", vocab_size=ByteTokenizer().vocab_size)

        self.assertGreater(report["parameter_estimate"], 45_000_000)
        self.assertLess(report["parameter_estimate"], 60_000_000)

    def test_context_length_override(self):
        config = config_from_preset("50m", vocab_size=300, context_length=4096)

        self.assertEqual(config.context_length, 4096)
        self.assertEqual(config.n_embd % config.n_head, 0)


if __name__ == "__main__":
    unittest.main()
