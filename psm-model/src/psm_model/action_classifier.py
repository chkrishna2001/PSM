from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from psm_model.schema import ACTIONS
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import ACTION_ORDER, ACTION_TO_ID, DEFAULT_REAL_TOKENIZER, LEGACY_REAL_TOKENIZER, resolve_device


@dataclass(frozen=True)
class ActionClassifierConfig:
    vocab_size: int
    context_length: int = 768
    n_embd: int = 128
    hidden_size: int = 256
    dropout: float = 0.1
    n_action: int = 6
    pad_id: int = 0


@dataclass(frozen=True)
class ActionExample:
    row_id: str
    input_payload: dict[str, Any]
    action: str


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Action classifier training requires PyTorch. Install torch to run psm_model.action_classifier.") from exc
    return torch


def _nn():
    torch = _torch()
    return torch.nn


_ModuleBase = object
try:
    import torch.nn as _torch_nn

    _ModuleBase = _torch_nn.Module
except ImportError:
    pass


class ActionClassifier(_ModuleBase):
    """Small standalone action selector: input payload/context -> storage action.

    This intentionally separates the failing action-selection problem from the
    50M decoder.  The model is cheap enough to train locally and to evaluate at
    frequent early-abort checkpoints before spending GPU time on full decoder
    experiments.
    """

    def __init__(self, config: ActionClassifierConfig):
        nn = _nn()
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_id)
        self.proj = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.n_action),
        )

    def forward(self, input_ids: Any, labels: Any | None = None) -> dict[str, Any]:
        torch = _torch()
        mask = input_ids != self.config.pad_id
        embedded = self.embedding(input_ids)
        masked = embedded * mask.unsqueeze(-1).to(embedded.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(embedded.dtype)
        mean_pool = masked.sum(dim=1) / denom
        max_pool = embedded.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values
        logits = self.proj(torch.cat([mean_pool, max_pool], dim=-1))
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

    @classmethod
    def load_checkpoint(cls, path: Path, *, map_location: str = "cpu") -> "ActionClassifier":
        torch = _torch()
        payload = torch.load(path, map_location=map_location)
        model = cls(ActionClassifierConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model


def render_action_input(input_payload: dict[str, Any]) -> str:
    """Render a compact classifier input with conversation first.

    The decoder prompt contains long static instructions.  For action selection,
    those instructions are label-independent noise, so the classifier sees the
    payload fields directly with stable labels.
    """

    parts: list[str] = []
    for key in ("conversation", "text", "message", "context", "operation", "source_kind", "source_id", "source_timestamp"):
        value = input_payload.get(key)
        if value is not None and str(value).strip():
            parts.append(f"{key}: {value}")
    if parts:
        return "\n".join(parts)
    return json.dumps(input_payload, ensure_ascii=False, sort_keys=True)


def load_action_examples(path: Path) -> list[ActionExample]:
    examples: list[ActionExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        input_payload = row.get("input")
        if not isinstance(input_payload, dict):
            if isinstance(row.get("text"), str):
                input_payload = {"conversation": row["text"], "operation": "remember"}
            else:
                raise ValueError(f"{path}:{line_number}: row has no input payload")
        action = _expected_action(row)
        if action not in ACTIONS:
            raise ValueError(f"{path}:{line_number}: unsupported action: {action}")
        row_id = str(row.get("id") or row.get("case") or row.get("source_id") or f"row-{line_number}")
        examples.append(ActionExample(row_id=row_id, input_payload=input_payload, action=action))
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def encode_action_batch(examples: list[ActionExample], tokenizer: Any, *, context_length: int, device: Any) -> tuple[Any, Any]:
    torch = _torch()
    rows: list[list[int]] = []
    labels: list[int] = []
    for example in examples:
        ids = tokenizer.encode(render_action_input(example.input_payload), add_bos=True)
        if len(ids) > context_length:
            ids = [ids[0]] + ids[-(context_length - 1) :]
        pad = context_length - len(ids)
        rows.append(ids + [tokenizer.pad_id] * pad)
        labels.append(ACTION_TO_ID[example.action])
    return torch.tensor(rows, dtype=torch.long, device=device), torch.tensor(labels, dtype=torch.long, device=device)


def train_action_classifier(
    train_file: Path,
    *,
    out: Path,
    validation_file: Path | None = None,
    probe_file: Path | None = None,
    tokenizer_path: Path | None = None,
    context_length: int = 768,
    n_embd: int = 128,
    hidden_size: int = 256,
    dropout: float = 0.1,
    steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    seed: int = 7,
    sampling: str = "action_balanced",
    eval_every: int | None = 100,
    save_every: int | None = None,
    metrics_out: Path | None = None,
    device: str = "cpu",
    collapse_threshold: float = 0.80,
    abort_after_step: int = 300,
) -> dict[str, Any]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer, tokenizer_source = load_cli_tokenizer(tokenizer_path)
    train_examples = load_action_examples(train_file)
    validation_examples = load_action_examples(validation_file) if validation_file else []
    probe_examples = load_action_examples(probe_file) if probe_file else []
    config = ActionClassifierConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        n_embd=n_embd,
        hidden_size=hidden_size,
        dropout=dropout,
        n_action=len(ACTION_ORDER),
        pad_id=tokenizer.pad_id,
    )
    model = ActionClassifier(config).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    sampler = _build_sampler(train_examples, sampling=sampling)
    metrics_handle = _open_metrics(metrics_out)
    losses: list[float] = []
    best_probe_macro = -1.0

    try:
        for step_index in range(steps):
            model.train()
            batch = [train_examples[sampler()] for _ in range(batch_size)]
            input_ids, labels = encode_action_batch(batch, tokenizer, context_length=context_length, device=device_obj)
            optimizer.zero_grad(set_to_none=True)
            result = model(input_ids, labels=labels)
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step = step_index + 1
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            event: dict[str, Any] = {"event": "step", "step": step, "loss": loss_value}

            should_eval = bool(eval_every and step % eval_every == 0) or step == steps
            if should_eval:
                if validation_examples:
                    event["validation"] = compact_report(evaluate_examples(model, tokenizer, validation_examples, device=device_obj))
                if probe_examples:
                    probe_report = evaluate_examples(model, tokenizer, probe_examples, device=device_obj)
                    event["probe"] = compact_report(probe_report)
                    best_probe_macro = max(best_probe_macro, float(probe_report["macro_action_accuracy"]))
                    if step >= abort_after_step and _collapsed(probe_report, collapse_threshold=collapse_threshold):
                        event["abort_reason"] = "prediction_collapse"
                        _write_metric(metrics_handle, event)
                        break
            _write_metric(metrics_handle, event)

            if save_every and save_every > 0 and step % save_every == 0:
                _save_checkpoint(
                    checkpoint_path_for_step(out, step),
                    model=model,
                    tokenizer=tokenizer,
                    metadata={"step": step, "tokenizer_source": tokenizer_source, "train_file": str(train_file)},
                )

        final_report = {
            "checkpoint": str(out),
            "train_file": str(train_file),
            "validation_file": str(validation_file) if validation_file else None,
            "probe_file": str(probe_file) if probe_file else None,
            "steps_completed": len(losses),
            "loss_final": losses[-1] if losses else None,
            "loss_recent_avg": sum(losses[-20:]) / len(losses[-20:]) if losses else None,
            "tokenizer_source": tokenizer_source,
            "config": asdict(config),
        }
        if validation_examples:
            final_report["validation"] = compact_report(evaluate_examples(model, tokenizer, validation_examples, device=device_obj))
        if probe_examples:
            final_report["probe"] = compact_report(evaluate_examples(model, tokenizer, probe_examples, device=device_obj))
        _save_checkpoint(out, model=model, tokenizer=tokenizer, metadata=final_report)
        return final_report
    finally:
        if metrics_handle is not None:
            metrics_handle.close()


def evaluate_checkpoint(checkpoint: Path, data: Path, *, device: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = ActionClassifier.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    examples = load_action_examples(data)
    return evaluate_examples(model, tokenizer, examples, device=device_obj)


def evaluate_examples(model: ActionClassifier, tokenizer: Any, examples: list[ActionExample], *, device: Any) -> dict[str, Any]:
    torch = _torch()
    model.eval()
    reports: list[dict[str, Any]] = []
    correct = 0
    expected_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    predicted_counts = {action: 0 for action in ACTION_ORDER}
    with torch.no_grad():
        for example in examples:
            probs = predict_action_probs(model, tokenizer, example.input_payload, device=device)
            ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            predicted = ranked[0][0]
            expected_counts[example.action] += 1
            predicted_counts[predicted] += 1
            correct += int(predicted == example.action)
            correct_counts[example.action] += int(predicted == example.action)
            reports.append(
                {
                    "id": example.row_id,
                    "expected_action": example.action,
                    "predicted_action": predicted,
                    "scores": {action: round(score, 6) for action, score in ranked},
                }
            )
    total = len(examples)
    per_action_accuracy = {
        action: correct_counts[action] / expected_counts[action]
        for action in sorted(expected_counts)
        if expected_counts[action]
    }
    return {
        "examples": total,
        "action_accuracy": correct / total if total else 0.0,
        "macro_action_accuracy": sum(per_action_accuracy.values()) / len(per_action_accuracy) if per_action_accuracy else 0.0,
        "expected_action_counts": dict(sorted(expected_counts.items())),
        "per_action_accuracy": per_action_accuracy,
        "predicted_action_counts": predicted_counts,
        "collapse_fraction": max(predicted_counts.values()) / total if total else 0.0,
        "reports": reports,
    }


def compact_report(report: dict[str, Any], *, max_failures: int = 12) -> dict[str, Any]:
    return {
        "examples": report["examples"],
        "action_accuracy": report["action_accuracy"],
        "macro_action_accuracy": report["macro_action_accuracy"],
        "expected_action_counts": report["expected_action_counts"],
        "per_action_accuracy": report["per_action_accuracy"],
        "predicted_action_counts": report["predicted_action_counts"],
        "collapse_fraction": report["collapse_fraction"],
        "failures": [
            {
                "id": row["id"],
                "expected_action": row["expected_action"],
                "predicted_action": row["predicted_action"],
            }
            for row in report["reports"]
            if row["expected_action"] != row["predicted_action"]
        ][:max_failures],
    }


def predict_checkpoint_action(checkpoint: Path, input_payload: dict[str, Any], *, device: str = "cpu") -> tuple[str, dict[str, float]]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = ActionClassifier.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    probs = predict_action_probs(model, tokenizer, input_payload, device=device_obj)
    return max(probs.items(), key=lambda item: item[1])[0], probs


def predict_action_probs(model: ActionClassifier, tokenizer: Any, input_payload: dict[str, Any], *, device: Any) -> dict[str, float]:
    torch = _torch()
    example = ActionExample(row_id="input", input_payload=input_payload, action=ACTION_ORDER[0])
    input_ids, _ = encode_action_batch([example], tokenizer, context_length=model.config.context_length, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)["logits"][0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
    return {action: float(probs[index].detach().cpu()) for index, action in enumerate(ACTION_ORDER)}


def load_cli_tokenizer(path: Path | None) -> tuple[Any, str]:
    if path is not None:
        return load_tokenizer(path), str(path)
    if DEFAULT_REAL_TOKENIZER.exists():
        return load_tokenizer(DEFAULT_REAL_TOKENIZER), str(DEFAULT_REAL_TOKENIZER)
    if LEGACY_REAL_TOKENIZER.exists():
        return load_tokenizer(LEGACY_REAL_TOKENIZER), str(LEGACY_REAL_TOKENIZER)
    return ByteTokenizer(), "byte:fallback"


def checkpoint_path_for_step(out: Path, step: int) -> Path:
    stem = re.sub(r"-step-\d+$", "", out.stem)
    return out.with_name(f"{stem}-step-{step:06d}{out.suffix}")


def _expected_action(row: dict[str, Any]) -> str:
    expected = row.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("action"), str):
        return expected["action"]
    if isinstance(row.get("expected_action"), str):
        return row["expected_action"]
    raise ValueError("row has no expected action")


def _build_sampler(examples: list[ActionExample], *, sampling: str) -> Any:
    if sampling == "random":
        return lambda: random.randrange(len(examples))
    if sampling != "action_balanced":
        raise ValueError(f"unsupported sampling mode: {sampling}")
    action_to_indices: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        action_to_indices.setdefault(example.action, []).append(index)
    actions = sorted(action_to_indices)

    def sample() -> int:
        action = actions[random.randrange(len(actions))]
        choices = action_to_indices[action]
        return choices[random.randrange(len(choices))]

    return sample


def _collapsed(report: dict[str, Any], *, collapse_threshold: float) -> bool:
    return float(report.get("collapse_fraction") or 0.0) > collapse_threshold


def _open_metrics(path: Path | None) -> Any | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _write_metric(handle: Any | None, event: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(event, sort_keys=True) + "\n")
    handle.flush()


def _save_checkpoint(path: Path, *, model: ActionClassifier, tokenizer: Any, metadata: dict[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict(), "metadata": metadata}, path)
    tokenizer.save(path.with_suffix(".tokenizer.json"))
    path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a standalone PSM action classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("data", type=Path)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--validation-file", type=Path)
    train_parser.add_argument("--probe-file", type=Path)
    train_parser.add_argument("--tokenizer", type=Path)
    train_parser.add_argument("--context-length", type=int, default=768)
    train_parser.add_argument("--n-embd", type=int, default=128)
    train_parser.add_argument("--hidden-size", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--sampling", choices=["random", "action_balanced"], default="action_balanced")
    train_parser.add_argument("--eval-every", type=int, default=100)
    train_parser.add_argument("--save-every", type=int)
    train_parser.add_argument("--metrics-out", type=Path)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--collapse-threshold", type=float, default=0.80)
    train_parser.add_argument("--abort-after-step", type=int, default=300)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("checkpoint", type=Path)
    eval_parser.add_argument("data", type=Path)
    eval_parser.add_argument("--device", default="cpu")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("checkpoint", type=Path)
    predict_parser.add_argument("input", help="JSON object payload")
    predict_parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    if args.command == "train":
        report = train_action_classifier(
            args.data,
            out=args.out,
            validation_file=args.validation_file,
            probe_file=args.probe_file,
            tokenizer_path=args.tokenizer,
            context_length=args.context_length,
            n_embd=args.n_embd,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            sampling=args.sampling,
            eval_every=args.eval_every,
            save_every=args.save_every,
            metrics_out=args.metrics_out,
            device=args.device,
            collapse_threshold=args.collapse_threshold,
            abort_after_step=args.abort_after_step,
        )
    elif args.command == "eval":
        report = evaluate_checkpoint(args.checkpoint, args.data, device=args.device)
    elif args.command == "predict":
        action, probs = predict_checkpoint_action(args.checkpoint, json.loads(args.input), device=args.device)
        report = {"action": action, "action_probs": dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))}
    else:
        raise ValueError(f"unsupported command: {args.command}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
