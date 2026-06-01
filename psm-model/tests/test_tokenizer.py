import tempfile
import unittest
from pathlib import Path

from psm_model.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    BpeTokenizer,
    ByteTokenizer,
    PatternTokenizer,
    DslTokenizer,
    load_tokenizer,
    train_bpe_tokenizer,
    train_pattern_tokenizer,
)


class ByteTokenizerTests(unittest.TestCase):
    def test_round_trip_ascii_and_unicode(self):
        tokenizer = ByteTokenizer()
        text = '{"memory":"SQLite preference","emoji":"ok"}'

        ids = tokenizer.encode(text)

        self.assertEqual(tokenizer.decode(ids), text)

    def test_special_tokens_are_optional(self):
        tokenizer = ByteTokenizer()
        ids = tokenizer.encode("abc", add_bos=True, add_eos=True)

        self.assertEqual(ids[0], tokenizer.bos_id)
        self.assertEqual(ids[-1], tokenizer.eos_id)
        self.assertEqual(tokenizer.decode(ids), "abc")
        self.assertEqual(tokenizer.decode(ids, skip_special=False), f"{BOS_TOKEN}abc{EOS_TOKEN}")

    def test_save_and_load(self):
        tokenizer = ByteTokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokenizer.json"
            tokenizer.save(path)

            loaded = ByteTokenizer.load(path)

        self.assertEqual(loaded, tokenizer)
        self.assertEqual(loaded.decode(loaded.encode("json")), "json")

    def test_rejects_out_of_range_ids(self):
        tokenizer = ByteTokenizer()

        with self.assertRaises(ValueError):
            tokenizer.decode([tokenizer.vocab_size])


class BpeTokenizerTests(unittest.TestCase):
    def test_round_trip_with_free_text(self):
        tokenizer = train_bpe_tokenizer(
            [
                "A:promote_semantic\nC:The user prefers SQLite for local prototypes.\nEND",
                "A:ignore\nM:-\nR:The message has no durable memory value.\nEND",
            ],
            vocab_size=320,
        )
        text = "C:The user prefers SQLite|DuckDB with backslash \\ and newline\nEND"

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_bpe_compresses_repeated_training_text(self):
        text = "A:promote_semantic\nT:semantic\nC:The user prefers SQLite.\nEND"
        tokenizer = train_bpe_tokenizer([text] * 20, vocab_size=360)
        byte = ByteTokenizer()

        self.assertLess(len(tokenizer.encode(text)), len(byte.encode(text)))

    def test_save_load_dispatches_bpe(self):
        tokenizer = train_bpe_tokenizer(["A:ignore\nM:-\nEND"] * 4, vocab_size=300)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokenizer.json"
            tokenizer.save(path)

            loaded = load_tokenizer(path)

        self.assertIsInstance(loaded, BpeTokenizer)
        self.assertEqual(loaded.decode(loaded.encode("A:ignore\nEND")), "A:ignore\nEND")

    def test_encode_pieces_prevents_cross_piece_merges(self):
        tokenizer = train_bpe_tokenizer(["abcX"] * 20, vocab_size=300)

        joined = tokenizer.encode("abcX")
        pieced = tokenizer.encode_pieces(["abc", "X"])

        self.assertEqual(tokenizer.decode(pieced), "abcX")
        self.assertGreaterEqual(len(pieced), len(joined))


class PatternTokenizerTests(unittest.TestCase):
    def test_round_trip_and_compression(self):
        text = "A:promote_semantic\nT:semantic\nC:The user prefers SQLite.\nEND"
        tokenizer = train_pattern_tokenizer([text] * 10, vocab_size=360)
        byte = ByteTokenizer()

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertLess(len(tokenizer.encode(text)), len(byte.encode(text)))

    def test_unknown_text_falls_back_to_bytes(self):
        tokenizer = train_pattern_tokenizer(["A:ignore\nEND"] * 4, vocab_size=280)
        text = "Unseen 🚀 text with pipes | and slashes \\"

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_save_load_dispatches_pattern(self):
        tokenizer = train_pattern_tokenizer(["A:ignore\nEND"] * 4, vocab_size=300)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokenizer.json"
            tokenizer.save(path)

            loaded = load_tokenizer(path)

        self.assertIsInstance(loaded, PatternTokenizer)
        self.assertEqual(loaded.decode(loaded.encode("A:ignore\nEND")), "A:ignore\nEND")


class DslTokenizerTests(unittest.TestCase):
    def test_round_trip_and_grammar_compression(self):
        tokenizer = DslTokenizer()
        byte = ByteTokenizer()
        text = "A:promote_semantic\nT:semantic\nF:user|prefers|SQLite|0.9|explicit|I prefer SQLite.\nEND"

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertLess(len(tokenizer.encode(text)), len(byte.encode(text)))

    def test_end_marker_is_not_atomic(self):
        tokenizer = DslTokenizer()

        self.assertNotIn("<|end|>", tokenizer.pieces)

    def test_unknown_text_falls_back_to_bytes(self):
        tokenizer = DslTokenizer()
        text = "Free text with 🚀 and unknown symbols"

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_save_load_dispatches_dsl(self):
        tokenizer = DslTokenizer()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokenizer.json"
            tokenizer.save(path)

            loaded = load_tokenizer(path)

        self.assertIsInstance(loaded, DslTokenizer)
        self.assertEqual(loaded.decode(loaded.encode("A:ignore\nEND")), "A:ignore\nEND")


if __name__ == "__main__":
    unittest.main()
