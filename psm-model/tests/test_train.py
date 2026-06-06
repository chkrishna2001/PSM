import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from psm_model.generate import load_checkpoint_metadata
from psm_model.tokenizer import ByteTokenizer
from psm_model.train import (
    _build_sampler,
    _action_span_token_ids,
    _encode_training_text,
    _learning_rate_for_step,
    action_span_loss_weights,
    build_lm_batch,
    checkpoint_path_for_step,
    configure_cuda_memory_fraction,
    first_label_positions,
    freeze_non_action_head_parameters,
    load_training_examples,
    load_training_texts,
    loss_summary,
    lm_loss_weights,
    merge_loss_weights,
    move_batch_to_device,
    move_optimizer_to_device,
    overfit_texts,
    parse_action_weight_overrides,
    resolve_device,
    resolve_resume_checkpoint,
    structural_loss_weights,
    train_texts,
)
from psm_model.model import TinyDecoderConfig, TinyDecoderModel


HAS_TORCH = importlib.util.find_spec("torch") is not None


class TrainTests(unittest.TestCase):
    def test_freeze_non_action_head_parameters_only_leaves_action_head_trainable(self):
        model = TinyDecoderModel(TinyDecoderConfig(vocab_size=32, context_length=16, n_layer=1, n_head=2, n_embd=16))

        freeze_non_action_head_parameters(model)

        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, {"action_head.weight", "action_head.bias"})

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

    def test_load_training_examples_keeps_action_label(self):
        row = {
            "id": "row-1",
            "input": {"conversation": "User: I prefer SQLite."},
            "expected": {
                "action": "promote_semantic",
                "memory": {
                    "content": "The user prefers SQLite.",
                    "type": "semantic",
                    "strength": 0.8,
                    "decay_rate": 0.02,
                    "emotional_weight": 0.1,
                    "confidence": 0.9,
                },
                "facts": [],
                "reasoning": "Durable preference.",
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")

            examples = load_training_examples(path)

        self.assertEqual(examples[0].action, "promote_semantic")
        self.assertIn("A:promote_semantic", examples[0].text)

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

    def test_resolve_device_auto_prefers_available_runtime(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        device = resolve_device("auto", torch)

        self.assertIn(device.type, {"cpu", "cuda"})

    def test_move_batch_to_device_keeps_shapes(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        input_ids, labels = build_lm_batch(["abc"], ByteTokenizer(), context_length=8)
        moved_input_ids, moved_labels = move_batch_to_device(input_ids, labels, "cpu")

        self.assertEqual(tuple(moved_input_ids.shape), (1, 8))
        self.assertEqual(tuple(moved_labels.shape), (1, 8))

    def test_first_label_positions_finds_first_target_token(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        labels = torch.tensor([[-100, -100, 4, 5], [-100, 6, 7, -100]])

        self.assertEqual(first_label_positions(labels).tolist(), [2, 1])

    def test_lm_loss_weights_upweights_first_target_token(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        labels = torch.tensor([[-100, 4, 5]])
        weights = lm_loss_weights(labels, first_token_weight=5.0)

        self.assertEqual(weights.tolist(), [[0.0, 5.0, 1.0]])

    def test_action_span_loss_weights_upweights_rendered_action(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        text = "<|system|>\nx\n<|assistant|>\nA:promote_semantic\nEND<|end|>"
        _, labels = build_lm_batch([text], ByteTokenizer(), context_length=64)
        weights = action_span_loss_weights(
            labels,
            ["promote_semantic"],
            ByteTokenizer(),
            output_format="tagged",
            action_span_weight=4.0,
        )
        action_tokens = _action_span_token_ids("promote_semantic", ByteTokenizer(), output_format="tagged")
        row = labels[0].tolist()
        start = next(index for index in range(len(row)) if row[index : index + len(action_tokens)] == action_tokens)

        self.assertEqual(weights[0, start : start + len(action_tokens)].tolist(), [4.0] * len(action_tokens))

    def test_action_span_loss_weights_supports_per_action_override(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        text = "<|system|>\nx\n<|assistant|>\nA:promote_semantic\nEND<|end|>"
        _, labels = build_lm_batch([text], ByteTokenizer(), context_length=64)
        weights = action_span_loss_weights(
            labels,
            ["promote_semantic"],
            ByteTokenizer(),
            output_format="tagged",
            action_span_weight=2.0,
            per_action_span_weights={"promote_semantic": 7.0},
        )
        action_tokens = _action_span_token_ids("promote_semantic", ByteTokenizer(), output_format="tagged")
        row = labels[0].tolist()
        start = next(index for index in range(len(row)) if row[index : index + len(action_tokens)] == action_tokens)

        self.assertEqual(weights[0, start : start + len(action_tokens)].tolist(), [7.0] * len(action_tokens))

    def test_parse_action_weight_overrides(self):
        self.assertEqual(parse_action_weight_overrides(["promote_semantic=25", "store_episodic=12"]), {"promote_semantic": 25.0, "store_episodic": 12.0})

    def test_structural_loss_weights_upweights_tagged_delimiters(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        tokenizer = ByteTokenizer()
        _, labels = build_lm_batch(["<|assistant|>\nA:ignore\nM:-\nR:No durable memory.\nEND"], tokenizer, context_length=64)
        weights = structural_loss_weights(labels, tokenizer, output_format="tagged", structural_weight=6.0)
        colon_id = tokenizer.encode(":")[0]
        colon_positions = [index for index, token_id in enumerate(labels[0].tolist()) if token_id == colon_id]

        self.assertTrue(colon_positions)
        self.assertTrue(all(float(weights[0, index]) == 6.0 for index in colon_positions))

    def test_structural_loss_weights_rejects_invalid_weight(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        _, labels = build_lm_batch(["<|assistant|>\nA:ignore\nEND"], ByteTokenizer(), context_length=32)

        with self.assertRaises(ValueError):
            structural_loss_weights(labels, ByteTokenizer(), output_format="tagged", structural_weight=0)

    def test_merge_loss_weights_uses_maximum_weight(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        merged = merge_loss_weights(torch.tensor([[0.0, 2.0]]), torch.tensor([[1.0, 0.5]]))

        self.assertEqual(merged.tolist(), [[1.0, 2.0]])

    def test_cuda_memory_fraction_rejects_invalid_value(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        with self.assertRaises(ValueError):
            configure_cuda_memory_fraction(torch, torch.device("cpu"), 0)

    def test_move_optimizer_to_device_handles_state_tensors(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        import torch

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        loss = parameter.sum()
        loss.backward()
        optimizer.step()

        move_optimizer_to_device(optimizer, torch.device("cpu"))

        self.assertTrue(all(value.device.type == "cpu" for state in optimizer.state.values() for value in state.values() if hasattr(value, "device")))

    def test_encode_training_text_masks_prompt(self):
        ids, mask = _encode_training_text(ByteTokenizer(), "<|system|>\nx\n<|assistant|>\nA:ignore\nEND<|end|>")

        self.assertEqual(len(ids), len(mask))
        self.assertIn(True, mask)
        self.assertFalse(mask[0])
        first_answer = mask.index(True)
        self.assertGreater(first_answer, 0)

    def test_rendered_training_text_contains_exact_assistant_marker(self):
        expected = {"action": "ignore", "memory": None, "facts": [], "reasoning": "No durable memory."}
        text = load_training_texts_from_row({"id": "row-1", "input": {"conversation": "User: okay"}, "expected": expected})

        self.assertIn("<|assistant|>\n", text)

    def test_learning_rate_warmup_and_decay(self):
        self.assertEqual(
            _learning_rate_for_step(0, total_steps=10, base_learning_rate=1.0, min_learning_rate=0.1, warmup_steps=2),
            0.5,
        )
        self.assertEqual(
            _learning_rate_for_step(1, total_steps=10, base_learning_rate=1.0, min_learning_rate=0.1, warmup_steps=2),
            1.0,
        )
        self.assertLess(
            _learning_rate_for_step(9, total_steps=10, base_learning_rate=1.0, min_learning_rate=0.1, warmup_steps=2),
            0.2,
        )

    def test_checkpoint_path_for_step_strips_existing_step_suffix(self):
        path = checkpoint_path_for_step(Path("psm-model/checkpoints/real-v1-50m-step-000500.pt"), 250)

        self.assertEqual(path, Path("psm-model/checkpoints/real-v1-50m-step-000250.pt"))

    def test_resolve_resume_checkpoint_auto_uses_latest_step_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "real-v2-50m-concept-repair-step-005300.pt"
            older = Path(temp_dir) / "real-v2-50m-concept-repair-step-005100.pt"
            latest = Path(temp_dir) / "real-v2-50m-concept-repair-step-005200.pt"
            older.write_text("old", encoding="utf-8")
            latest.write_text("new", encoding="utf-8")

            resolved = resolve_resume_checkpoint("auto", out)

        self.assertEqual(resolved, latest)

    def test_resolve_resume_checkpoint_auto_falls_back_to_out(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "run.pt"
            out.write_text("final", encoding="utf-8")

            resolved = resolve_resume_checkpoint("auto", out)

        self.assertEqual(resolved, out)

    def test_resolve_resume_checkpoint_auto_can_start_fresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "run.pt"

            resolved = resolve_resume_checkpoint("auto", out)

        self.assertIsNone(resolved)

    def test_resolve_resume_checkpoint_auto_uses_fallback_for_first_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "run.pt"
            fallback = Path(temp_dir) / "base.pt"

            resolved = resolve_resume_checkpoint("auto", out, fallback=fallback)

        self.assertEqual(resolved, fallback)

    def test_loss_summary_reports_recent_average(self):
        summary = loss_summary([3.0, 2.0, 1.0])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["initial_loss"], 3.0)
        self.assertEqual(summary["final_loss"], 1.0)
        self.assertEqual(summary["recent_loss_avg"], 2.0)

    def test_action_balanced_sampler_requires_matching_labels(self):
        with self.assertRaises(ValueError):
            _build_sampler(["a", "b"], action_labels=["ignore"], sampling="action_balanced")

    def test_action_balanced_sampler_samples_all_actions(self):
        random_state = __import__("random").getstate()
        try:
            __import__("random").seed(1)
            sampler = _build_sampler(
                ["ignore-1", "ignore-2", "store-1"],
                action_labels=["ignore", "ignore", "store_episodic"],
                sampling="action_balanced",
            )
            sampled_actions = set()
            labels = ["ignore", "ignore", "store_episodic"]
            for _ in range(20):
                sampled_actions.add(labels[sampler()])

            self.assertEqual(sampled_actions, {"ignore", "store_episodic"})
        finally:
            __import__("random").setstate(random_state)

    def test_train_texts_writes_and_resumes_checkpoint(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is not installed")

        config = TinyDecoderConfig(vocab_size=ByteTokenizer().vocab_size, context_length=64, n_layer=1, n_head=1, n_embd=16)
        text = "<|system|>\nx\n<|user|>\ny\n<|assistant|>\nA:ignore\nEND<|end|>"

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "run.pt"
            metrics = Path(temp_dir) / "metrics.jsonl"
            _, first_losses = train_texts(
                [text],
                config=config,
                tokenizer=ByteTokenizer(),
                action_labels=["ignore"],
                steps=2,
                batch_size=1,
                out=out,
                save_every=1,
                metrics_out=metrics,
                metadata={"dataset_path": "rows.jsonl", "output_format": "tagged"},
                action_loss_weight=0.25,
                first_token_loss_weight=2.0,
                action_span_loss_weight=3.0,
                action_span_weight_overrides={"promote_semantic": 4.0},
            )
            _, resumed_losses = train_texts(
                [text],
                config=config,
                tokenizer=ByteTokenizer(),
                action_labels=["ignore"],
                steps=3,
                batch_size=1,
                resume=out,
                out=out,
                metrics_out=metrics,
                metadata={"dataset_path": "rows.jsonl", "output_format": "tagged"},
                action_loss_weight=0.25,
                first_token_loss_weight=2.0,
                action_span_loss_weight=3.0,
                action_span_weight_overrides={"promote_semantic": 4.0},
            )
            metadata = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
            metric_events = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(first_losses), 2)
        self.assertEqual(len(resumed_losses), 3)
        self.assertEqual(metadata["completed_steps"], 3)
        self.assertEqual(metadata["dataset_path"], "rows.jsonl")
        self.assertTrue(any(event["event"] == "checkpoint" for event in metric_events))

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


def load_training_texts_from_row(row):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "rows.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return load_training_texts(path)[0]
