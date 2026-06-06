from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from psm_model.configs import config_from_preset
from psm_model.data import validate_training_row
from psm_model.model import TinyDecoderConfig, TinyDecoderModel
from psm_model.prompts import render_training_text
from psm_model.schema import ACTIONS
from psm_model.tokenizer import ByteTokenizer, load_tokenizer

DEFAULT_REAL_TOKENIZER = Path("psm-model/tokenizers/real-v2-pattern.json")
LEGACY_REAL_TOKENIZER = Path("psm-model/tokenizers/real-v1-pattern.json")
ACTION_ORDER = tuple(sorted(ACTIONS))
ACTION_TO_ID = {action: index for index, action in enumerate(ACTION_ORDER)}


@dataclass(frozen=True)
class TrainingText:
    text: str
    action: str


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Training requires PyTorch. Install torch to run psm_model.train.") from exc
    return torch


def load_training_texts(path: Path, *, output_format: str = "tagged") -> list[str]:
    return [example.text for example in load_training_examples(path, output_format=output_format)]


def load_training_examples(
    path: Path,
    *,
    output_format: str = "tagged",
    max_training_tokens: int | None = None,
    tokenizer: Any | None = None,
) -> list[TrainingText]:
    examples: list[TrainingText] = []
    skipped_long = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if output_format == "action":
                input_payload = raw.get("input")
                expected = raw.get("expected")
                action = expected.get("action") if isinstance(expected, dict) else None
                if not isinstance(input_payload, dict) or not isinstance(action, str) or action not in ACTION_TO_ID:
                    raise ValueError(f"{path}:{line_number}: invalid action-only row")
            else:
                row, issues = validate_training_row(raw)
                if issues or row is None:
                    formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in issues)
                    raise ValueError(f"{path}:{line_number}: invalid row: {formatted}")
            text = render_training_text(raw["input"], raw["expected"], output_format=output_format)
            if max_training_tokens is not None and tokenizer is not None:
                if len(tokenizer.encode(text, add_bos=True, add_eos=True)) > max_training_tokens:
                    skipped_long += 1
                    continue
            examples.append(TrainingText(text=text, action=str(raw["expected"]["action"])))
    if not examples:
        raise ValueError(f"no training examples remain in {path} after filtering")
    if skipped_long:
        print(json.dumps({"skipped_overlong_rows": skipped_long, "dataset": str(path)}, sort_keys=True))
    return examples


def build_lm_batch(texts: list[str], tokenizer: Any, *, context_length: int) -> tuple[Any, Any]:
    torch = _torch()
    if not texts:
        raise ValueError("at least one training text is required")

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for text in texts:
        ids, label_mask = _encode_training_text(tokenizer, text)
        if len(ids) < 2:
            raise ValueError("training text is too short")
        if len(ids) > context_length + 1:
            raise ValueError(f"training text length {len(ids)} exceeds context length plus label token {context_length + 1}")
        input_ids = ids[:-1]
        labels = [token_id if mask else -100 for token_id, mask in zip(ids[1:], label_mask[1:])]
        pad = context_length - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [tokenizer.pad_id] * pad
            labels = labels + [-100] * pad
        input_rows.append(input_ids)
        label_rows.append(labels)

    return torch.tensor(input_rows, dtype=torch.long), torch.tensor(label_rows, dtype=torch.long)


def move_batch_to_device(input_ids: Any, labels: Any, device: str | Any) -> tuple[Any, Any]:
    return input_ids.to(device), labels.to(device)


def first_label_positions(labels: Any) -> Any:
    torch = _torch()
    valid = labels != -100
    if not bool(valid.any(dim=1).all()):
        raise ValueError("each training row must contain at least one target label")
    return valid.to(torch.long).argmax(dim=1)


def lm_loss_weights(labels: Any, *, first_token_weight: float) -> Any | None:
    if first_token_weight == 1.0:
        return None
    if first_token_weight <= 0:
        raise ValueError("first output token weight must be greater than 0")
    weights = (labels != -100).to(_torch().float32)
    positions = first_label_positions(labels)
    batch_indices = _torch().arange(labels.size(0), device=labels.device)
    weights[batch_indices, positions] = first_token_weight
    return weights


def action_span_loss_weights(
    labels: Any,
    action_labels: list[str],
    tokenizer: Any,
    *,
    output_format: str,
    action_span_weight: float,
    per_action_span_weights: dict[str, float] | None = None,
) -> Any | None:
    per_action_span_weights = per_action_span_weights or {}
    if action_span_weight == 1.0 and not per_action_span_weights:
        return None
    if action_span_weight <= 0:
        raise ValueError("action span weight must be greater than 0")
    for action, weight in per_action_span_weights.items():
        if action not in ACTION_TO_ID:
            raise ValueError(f"unsupported action span weight action: {action}")
        if weight <= 0:
            raise ValueError("per-action span weights must be greater than 0")
    torch = _torch()
    weights = (labels != -100).to(torch.float32)
    for row_index, action in enumerate(action_labels):
        weight = per_action_span_weights.get(action, action_span_weight)
        action_tokens = _action_span_token_ids(action, tokenizer, output_format=output_format)
        start = _find_subsequence(labels[row_index].tolist(), action_tokens)
        if start is None:
            positions = [int(first_label_positions(labels[[row_index]])[0])]
        else:
            positions = list(range(start, start + len(action_tokens)))
        weights[row_index, torch.tensor(positions, dtype=torch.long, device=labels.device)] = weight
    return weights


def structural_loss_weights(labels: Any, tokenizer: Any, *, output_format: str, structural_weight: float) -> Any | None:
    if structural_weight == 1.0:
        return None
    if structural_weight <= 0:
        raise ValueError("structural token weight must be greater than 0")
    torch = _torch()
    structural_tokens = set(_structural_token_ids(tokenizer, output_format=output_format))
    if not structural_tokens:
        return None
    weights = (labels != -100).to(torch.float32)
    for token_id in structural_tokens:
        weights = torch.where(labels == token_id, torch.full_like(weights, structural_weight), weights)
    return weights


def parse_action_weight_overrides(values: list[str] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values or []:
        action, sep, raw_weight = value.partition("=")
        if not sep:
            raise ValueError(f"expected ACTION=WEIGHT for action span override, got: {value}")
        if action not in ACTION_TO_ID:
            raise ValueError(f"unsupported action in span override: {action}")
        result[action] = float(raw_weight)
    return result


def action_span_mask(labels: Any, action_labels: list[str], tokenizer: Any, *, output_format: str) -> Any:
    torch = _torch()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for row_index, action in enumerate(action_labels):
        action_tokens = _action_span_token_ids(action, tokenizer, output_format=output_format)
        start = _find_subsequence(labels[row_index].tolist(), action_tokens)
        if start is None:
            positions = [int(first_label_positions(labels[[row_index]])[0])]
        else:
            positions = list(range(start, start + len(action_tokens)))
        mask[row_index, torch.tensor(positions, dtype=torch.long, device=labels.device)] = True
    return mask


def region_loss(logits: Any, labels: Any, mask: Any) -> float | None:
    torch = _torch()
    active = mask & (labels != -100)
    if not bool(active.any()):
        return None
    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    return float(token_losses[active].mean().detach().cpu())


def merge_loss_weights(*weights: Any | None) -> Any | None:
    active = [weight for weight in weights if weight is not None]
    if not active:
        return None
    merged = active[0]
    for weight in active[1:]:
        merged = _torch().maximum(merged, weight.to(merged.device))
    return merged


def _action_span_token_ids(action: str, tokenizer: Any, *, output_format: str) -> list[int]:
    if output_format in {"tagged", "action"}:
        text = f"A:{action}"
    elif output_format == "at_tag":
        text = f"@a {action}"
    elif output_format == "json":
        text = f'"action":"{action}"'
    else:
        raise ValueError(f"unsupported output format: {output_format}")
    return tokenizer.encode(text)


def _structural_token_ids(tokenizer: Any, *, output_format: str) -> list[int]:
    if output_format == "tagged":
        pieces = ("A:", "M:-", "T:", "C:", "Q:", "G:", "TE:", "RT:", "F:", "R:", "END", "|", ",", "\n", ":")
    elif output_format == "action":
        pieces = ("A:", "END", "\n", ":")
    elif output_format == "at_tag":
        pieces = (
            "@a",
            "@m",
            "@t",
            "@c",
            "@s",
            "@d",
            "@e",
            "@p",
            "@g",
            "@te",
            "@rt",
            "@f",
            "@ef",
            "@r",
            "@end",
            "sub=",
            "pred=",
            "val=",
            "conf=",
            "kind=",
            "ev=",
            "\n",
            "=",
        )
    elif output_format == "json":
        pieces = ('"action"', '"memory"', '"facts"', '"reasoning"', "{", "}", "[", "]", ":", ",", '"')
    else:
        raise ValueError(f"unsupported output format: {output_format}")
    ids: set[int] = set()
    for piece in pieces:
        ids.update(tokenizer.encode(piece))
    special_ids = {getattr(tokenizer, "bos_id", None), getattr(tokenizer, "eos_id", None), getattr(tokenizer, "pad_id", None)}
    return sorted(token_id for token_id in ids if token_id not in special_ids)


def _find_subsequence(values: list[int], pattern: list[int]) -> int | None:
    if not pattern:
        return None
    last_start = len(values) - len(pattern)
    for start in range(max(0, last_start + 1)):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def _encode_training_text(tokenizer: Any, text: str) -> tuple[list[int], list[bool]]:
    marker = "<|assistant|>\n"
    if marker in text:
        prompt, output = text.split(marker, 1)
        prompt_ids = tokenizer.encode(prompt + marker, add_bos=True)
        output_ids = tokenizer.encode(output, add_eos=True)
        return prompt_ids + output_ids, [False] * len(prompt_ids) + [True] * len(output_ids)
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return ids, [True] * len(ids)


def overfit_texts(
    texts: list[str],
    *,
    config: TinyDecoderConfig,
    tokenizer: Any | None = None,
    steps: int = 100,
    learning_rate: float = 3e-4,
    seed: int = 7,
    device: str = "cpu",
) -> tuple[TinyDecoderModel, list[float]]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    torch.manual_seed(seed)
    tokenizer = tokenizer or ByteTokenizer()
    model = TinyDecoderModel(config).to(device_obj)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input_ids, labels = build_lm_batch(texts, tokenizer, context_length=config.context_length)
    input_ids, labels = move_batch_to_device(input_ids, labels, device_obj)
    losses: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result = model(input_ids, labels=labels)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return model, losses


def train_texts(
    texts: list[str],
    *,
    config: TinyDecoderConfig,
    tokenizer: Any | None = None,
    action_labels: list[str] | None = None,
    sampling: str = "random",
    steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    min_learning_rate: float = 0.0,
    warmup_steps: int = 0,
    max_grad_norm: float | None = 1.0,
    seed: int = 7,
    resume: Path | None = None,
    out: Path | None = None,
    save_every: int | None = None,
    metrics_out: Path | None = None,
    metadata: dict[str, Any] | None = None,
    device: str = "cpu",
    cuda_memory_fraction: float | None = None,
    action_loss_weight: float = 0.0,
    first_token_loss_weight: float = 1.0,
    action_span_loss_weight: float = 1.0,
    action_span_weight_overrides: dict[str, float] | None = None,
    structural_loss_weight: float = 1.0,
    output_format: str = "tagged",
    freeze_backbone: bool = False,
    reset_optimizer: bool = False,
    probe_path: Path | None = None,
    eval_every: int | None = None,
    abort_after_step: int = 300,
    collapse_threshold: float = 0.8,
) -> tuple[TinyDecoderModel, list[float]]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    if cuda_memory_fraction is not None:
        configure_cuda_memory_fraction(torch, device_obj, cuda_memory_fraction)
    if not texts:
        raise ValueError("at least one training text is required")
    sampler = _build_sampler(texts, action_labels=action_labels, sampling=sampling)
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = tokenizer or ByteTokenizer()
    completed_steps = 0
    prior_losses: list[float] = []
    resume_payload: dict[str, Any] | None = None
    resume_missing_action_head = False
    if resume is not None:
        resume_payload = torch.load(resume, map_location="cpu")
        model_config = TinyDecoderConfig(**resume_payload["config"])
        if asdict(model_config) != asdict(config):
            raise ValueError(f"resume checkpoint config does not match requested config: {resume}")
        model = TinyDecoderModel(model_config)
        missing, unexpected = model.load_state_dict(resume_payload["state_dict"], strict=False)
        allowed_missing = {"action_head.weight", "action_head.bias"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(f"incompatible checkpoint {resume}: missing={missing}, unexpected={unexpected}")
        resume_missing_action_head = bool(set(missing) & allowed_missing)
        training_state = resume_payload.get("training", {})
        completed_steps = int(training_state.get("completed_steps", 0))
        if completed_steps > steps:
            raise ValueError(f"resume checkpoint is already at step {completed_steps}, beyond requested target steps {steps}")
        prior_losses = [float(loss) for loss in training_state.get("losses", [])]
        if "random_state" in training_state:
            random.setstate(training_state["random_state"])
        if "torch_rng_state" in training_state:
            torch.set_rng_state(training_state["torch_rng_state"])
    else:
        model = TinyDecoderModel(config)
    model.to(device_obj)
    model.train()
    if freeze_backbone:
        freeze_non_action_head_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if resume_payload is not None and resume_payload.get("optimizer_state") is not None and not resume_missing_action_head and not reset_optimizer:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        move_optimizer_to_device(optimizer, device_obj)
    losses: list[float] = []
    metrics_file = _open_metrics_writer(metrics_out)

    try:
        for step in range(completed_steps, steps):
            batch_indices = [sampler() for _ in range(batch_size)]
            batch = [texts[index] for index in batch_indices]
            input_ids, labels = build_lm_batch(batch, tokenizer, context_length=config.context_length)
            input_ids, labels = move_batch_to_device(input_ids, labels, device_obj)
            action_targets = None
            action_positions = None
            if action_loss_weight > 0:
                if action_labels is None:
                    raise ValueError("action auxiliary loss requires action labels")
                action_targets = torch.tensor([ACTION_TO_ID[action_labels[index]] for index in batch_indices], dtype=torch.long, device=device_obj)
                action_positions = first_label_positions(labels)
            first_token_weights = lm_loss_weights(labels, first_token_weight=first_token_loss_weight)
            structure_weights = structural_loss_weights(labels, tokenizer, output_format=output_format, structural_weight=structural_loss_weight)
            span_weights = None
            span_mask = None
            if action_labels is not None:
                batch_action_labels = [action_labels[index] for index in batch_indices]
                span_mask = action_span_mask(labels, batch_action_labels, tokenizer, output_format=output_format)
                span_weights = action_span_loss_weights(
                    labels,
                    batch_action_labels,
                    tokenizer,
                    output_format=output_format,
                    action_span_weight=action_span_loss_weight,
                    per_action_span_weights=action_span_weight_overrides,
                )
            loss_weights = merge_loss_weights(first_token_weights, span_weights, structure_weights)
            current_learning_rate = _learning_rate_for_step(
                step,
                total_steps=steps,
                base_learning_rate=learning_rate,
                min_learning_rate=min_learning_rate,
                warmup_steps=warmup_steps,
            )
            _set_learning_rate(optimizer, current_learning_rate)
            optimizer.zero_grad(set_to_none=True)
            result = model(
                input_ids,
                labels=labels,
                loss_weights=loss_weights,
                action_labels=action_targets,
                action_positions=action_positions,
                action_loss_weight=action_loss_weight,
            )
            loss = result["loss"]
            body_mask = (labels != -100) & (~span_mask if span_mask is not None else (labels == -100))
            action_span_loss = region_loss(result["logits"], labels, span_mask) if span_mask is not None else None
            body_loss = region_loss(result["logits"], labels, body_mask)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            all_losses = prior_losses + losses
            completed = step + 1
            step_event: dict[str, Any] = {
                "event": "step",
                "step": completed,
                "loss": loss_value,
                "lm_loss": float(result["lm_loss"].detach().cpu()) if result.get("lm_loss") is not None else None,
                "action_loss": float(result["action_loss"].detach().cpu()) if result.get("action_loss") is not None else None,
                "action_span_loss": action_span_loss,
                "body_loss": body_loss,
                "actions": [action_labels[index] for index in batch_indices] if action_labels is not None else None,
                "learning_rate": current_learning_rate,
            }
            if out is not None and save_every is not None and save_every > 0 and completed % save_every == 0:
                checkpoint_path = checkpoint_path_for_step(out, completed)
                _save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    metadata=metadata or {},
                    completed_steps=completed,
                    losses=prior_losses + losses,
                    seed=seed,
                )
                _write_metric(metrics_file, {"event": "checkpoint", "step": completed, "checkpoint": str(checkpoint_path)})
            should_probe = bool(probe_path and eval_every and eval_every > 0 and (completed % eval_every == 0 or completed == steps))
            if should_probe and probe_path is not None:
                from psm_model.action_diagnostics import evaluate_action_prefixes

                probe_checkpoint = _resolve_probe_checkpoint(
                    out=out,
                    completed=completed,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    metadata=metadata or {},
                    losses=prior_losses + losses,
                    seed=seed,
                    save_every=save_every,
                )
                probe_report = evaluate_action_prefixes(
                    probe_checkpoint,
                    probe_path,
                    output_format=output_format,
                    device=str(device_obj),
                )
                step_event["probe"] = {
                    "checkpoint": str(probe_checkpoint),
                    "macro_action_prefix_accuracy": probe_report["macro_action_prefix_accuracy"],
                    "collapse_fraction": probe_report["collapse_fraction"],
                    "predicted_action_counts": probe_report["predicted_action_counts"],
                }
                if completed >= abort_after_step and probe_report["collapse_fraction"] > collapse_threshold:
                    step_event["abort_reason"] = "prediction_collapse"
                    _write_metric(metrics_file, step_event)
                    break
            _write_metric(metrics_file, step_event)
        if out is not None:
            _save_training_checkpoint(
                out,
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                metadata=metadata or {},
                completed_steps=steps,
                losses=prior_losses + losses,
                seed=seed,
            )
            _write_metric(metrics_file, {"event": "checkpoint", "step": steps, "checkpoint": str(out)})
    finally:
        if metrics_file is not None:
            metrics_file.close()

    return model.cpu(), prior_losses + losses


def resolve_device(device: str, torch: Any | None = None) -> Any:
    torch = torch or _torch()
    requested = device.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def configure_cuda_memory_fraction(torch: Any, device: Any, fraction: float) -> None:
    if fraction <= 0 or fraction > 1:
        raise ValueError("--cuda-memory-fraction must be greater than 0 and at most 1")
    if getattr(device, "type", str(device)) != "cuda":
        return
    device_index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)


def move_optimizer_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def freeze_non_action_head_parameters(model: TinyDecoderModel) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("action_head.")


def _build_sampler(texts: list[str], *, action_labels: list[str] | None, sampling: str) -> Any:
    if sampling == "random":
        return lambda: random.randrange(len(texts))
    if sampling != "action_balanced":
        raise ValueError(f"unsupported sampling mode: {sampling}")
    if action_labels is None:
        raise ValueError("action-balanced sampling requires action labels")
    if len(action_labels) != len(texts):
        raise ValueError("action label count must match text count")

    action_to_indices: dict[str, list[int]] = {}
    for index, action in enumerate(action_labels):
        action_to_indices.setdefault(action, []).append(index)
    actions = sorted(action_to_indices)
    if not actions:
        raise ValueError("action-balanced sampling requires at least one action")

    def sample() -> int:
        action = actions[random.randrange(len(actions))]
        indices = action_to_indices[action]
        return indices[random.randrange(len(indices))]

    return sample


def _open_metrics_writer(path: Path | None) -> Any | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def _write_metric(handle: Any | None, event: dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(event, sort_keys=True) + "\n")
    handle.flush()


def _resolve_probe_checkpoint(
    *,
    out: Path | None,
    completed: int,
    model: TinyDecoderModel,
    optimizer: Any,
    tokenizer: Any,
    metadata: dict[str, Any],
    losses: list[float],
    seed: int,
    save_every: int | None,
) -> Path:
    if out is not None and save_every and save_every > 0 and completed % save_every == 0:
        return checkpoint_path_for_step(out, completed)
    if out is None:
        raise ValueError("probe evaluation during training requires --out when step is not a save_every checkpoint")
    stem_base = re.sub(r"-step-\d+$", "", out.stem)
    probe_path = checkpoint_path_for_step(out, completed).with_name(
        f"{stem_base}-probe-{completed:06d}{out.suffix}"
    )
    _save_training_checkpoint(
        probe_path,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        metadata=metadata,
        completed_steps=completed,
        losses=losses,
        seed=seed,
    )
    return probe_path


def checkpoint_path_for_step(out: Path, step: int) -> Path:
    stem = re.sub(r"-step-\d+$", "", out.stem)
    return out.with_name(f"{stem}-step-{step:06d}{out.suffix}")


def resolve_resume_checkpoint(resume: str | None, out: Path, *, fallback: str | Path | None = None) -> Path | None:
    if resume is None:
        return None
    if resume != "auto":
        return Path(resume)
    stem = re.sub(r"-step-\d+$", "", out.stem)
    candidates = sorted(
        out.parent.glob(f"{stem}-step-*{out.suffix}"),
        key=_checkpoint_step_sort_key,
    )
    if candidates:
        return candidates[-1]
    if out.exists():
        return out
    if fallback is not None:
        return Path(fallback)
    return None


def _checkpoint_step_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"-step-(\d+)$", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def loss_summary(losses: list[float]) -> dict[str, Any]:
    if not losses:
        return {
            "count": 0,
            "initial_loss": None,
            "final_loss": None,
            "min_loss": None,
            "max_loss": None,
            "recent_loss_avg": None,
        }
    recent = losses[-min(20, len(losses)) :]
    return {
        "count": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "recent_loss_avg": sum(recent) / len(recent),
    }


def _save_training_checkpoint(
    path: Path,
    *,
    model: TinyDecoderModel,
    optimizer: Any,
    tokenizer: Any,
    metadata: dict[str, Any],
    completed_steps: int,
    losses: list[float],
    seed: int,
) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata.update(
        {
            "completed_steps": completed_steps,
            "loss_summary": loss_summary(losses),
        }
    )
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": checkpoint_metadata,
            "training": {
                "completed_steps": completed_steps,
                "losses": losses,
                "random_state": random.getstate(),
                "seed": seed,
                "torch_rng_state": torch.get_rng_state(),
            },
        },
        path,
    )
    tokenizer.save(path.with_suffix(".tokenizer.json"))
    path.with_suffix(".meta.json").write_text(json.dumps(checkpoint_metadata, indent=2, sort_keys=True), encoding="utf-8")


def _learning_rate_for_step(
    step: int,
    *,
    total_steps: int,
    base_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
) -> float:
    import math

    if total_steps <= 0:
        return base_learning_rate
    if warmup_steps > 0 and step < warmup_steps:
        return base_learning_rate * float(step + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (base_learning_rate - min_learning_rate) * cosine


def _set_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def load_cli_tokenizer(path: Path | None) -> tuple[Any, str]:
    if path is not None:
        return load_tokenizer(path), str(path)
    if DEFAULT_REAL_TOKENIZER.exists():
        return load_tokenizer(DEFAULT_REAL_TOKENIZER), str(DEFAULT_REAL_TOKENIZER)
    if LEGACY_REAL_TOKENIZER.exists():
        return load_tokenizer(LEGACY_REAL_TOKENIZER), str(LEGACY_REAL_TOKENIZER)
    return ByteTokenizer(), "byte:fallback"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tiny debug PSM decoder on canonical JSONL rows.")
    parser.add_argument("data", type=Path, help="Canonical JSONL rows")
    parser.add_argument("--out", type=Path, default=Path("psm-model/checkpoints/tiny-debug.pt"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preset", choices=["debug", "10m", "25m", "50m"], default="debug")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=0.0, help="Auxiliary action classification loss weight.")
    parser.add_argument(
        "--first-token-loss-weight",
        type=float,
        default=1.0,
        help="LM loss multiplier for the first generated output token.",
    )
    parser.add_argument(
        "--action-span-loss-weight",
        type=float,
        default=1.0,
        help="LM loss multiplier for tokens in the rendered action span, e.g. A:promote_semantic.",
    )
    parser.add_argument(
        "--action-span-weight",
        action="append",
        help="Per-action span multiplier override, e.g. promote_semantic=25. May be repeated.",
    )
    parser.add_argument(
        "--structural-loss-weight",
        type=float,
        default=1.0,
        help="LM loss multiplier for DSL/JSON structural tokens such as tags, separators, and END.",
    )
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"], default="tagged")
    parser.add_argument("--tokenizer", type=Path, help=f"Tokenizer JSON. Defaults to {DEFAULT_REAL_TOKENIZER} when present.")
    parser.add_argument("--sampling", choices=["random", "action_balanced"], default="random")
    parser.add_argument("--resume", help="Continue from a checkpoint path, or use 'auto' to pick the latest checkpoint for --out.")
    parser.add_argument("--resume-fallback", help="Checkpoint to use when --resume auto finds no run checkpoint yet.")
    parser.add_argument("--save-every", type=int, help="Write periodic checkpoints every N completed steps.")
    parser.add_argument("--metrics-out", type=Path, help="Append JSONL step/checkpoint metrics to this path.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the auxiliary action head; freeze all decoder/LM parameters.")
    parser.add_argument("--reset-optimizer", action="store_true", help="Do not load optimizer state from the resume checkpoint.")
    parser.add_argument("--device", default="cpu", help="Training device: cpu, cuda, or auto.")
    parser.add_argument(
        "--cuda-memory-fraction",
        type=float,
        help="Optional per-process CUDA memory cap from 0 to 1. Ignored on CPU.",
    )
    parser.add_argument("--probe", type=Path, help="JSONL probe file for periodic action-prefix evaluation during training.")
    parser.add_argument("--eval-every", type=int, help="Evaluate --probe every N completed steps.")
    parser.add_argument("--abort-after-step", type=int, default=300, help="Allow prediction-collapse abort only after this step.")
    parser.add_argument(
        "--collapse-threshold",
        type=float,
        default=0.8,
        help="Abort when probe collapse_fraction exceeds this value after --abort-after-step.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config and print parameter estimate without training")
    args = parser.parse_args()

    torch = _torch()
    device = resolve_device(args.device, torch)
    resume_checkpoint = resolve_resume_checkpoint(args.resume, args.out, fallback=args.resume_fallback)
    tokenizer, tokenizer_source = load_cli_tokenizer(args.tokenizer)
    config = config_from_preset(args.preset, vocab_size=tokenizer.vocab_size, context_length=args.context_length)
    examples = load_training_examples(
        args.data,
        output_format=args.output_format,
        max_training_tokens=config.context_length + 1,
        tokenizer=tokenizer,
    )
    texts = [example.text for example in examples]
    action_labels = [example.action for example in examples]
    action_span_weight_overrides = parse_action_weight_overrides(args.action_span_weight)
    report = {
        "config": asdict(config),
        "examples": len(texts),
        "checkpoint": str(args.out),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "save_every": args.save_every,
        "metrics_out": str(args.metrics_out) if args.metrics_out is not None else None,
        "output_format": args.output_format,
        "sampling": args.sampling,
        "action_loss_weight": args.action_loss_weight,
        "first_token_loss_weight": args.first_token_loss_weight,
        "action_span_loss_weight": args.action_span_loss_weight,
        "action_span_weight_overrides": action_span_weight_overrides,
        "structural_loss_weight": args.structural_loss_weight,
        "freeze_backbone": args.freeze_backbone,
        "reset_optimizer": args.reset_optimizer,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_source": tokenizer_source,
        "preset": args.preset,
        "resume": args.resume,
        "resume_fallback": args.resume_fallback,
        "resolved_resume": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "device": str(device),
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "cuda_available": bool(torch.cuda.is_available()),
        "probe": str(args.probe) if args.probe is not None else None,
        "eval_every": args.eval_every,
        "abort_after_step": args.abort_after_step,
        "collapse_threshold": args.collapse_threshold,
        "parameter_estimate": TinyDecoderModel.parameter_estimate(config),
    }
    if args.dry_run:
        print(json.dumps(report | {"dry_run": True}, indent=2, sort_keys=True))
        return 0

    training_metadata = {
        "dataset_path": str(args.data),
        "output_format": args.output_format,
        "preset": args.preset,
        "config": asdict(config),
        "parameter_estimate": TinyDecoderModel.parameter_estimate(config),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_source": tokenizer_source,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "action_loss_weight": args.action_loss_weight,
        "first_token_loss_weight": args.first_token_loss_weight,
        "action_span_loss_weight": args.action_span_loss_weight,
        "action_span_weight_overrides": action_span_weight_overrides,
        "structural_loss_weight": args.structural_loss_weight,
        "freeze_backbone": args.freeze_backbone,
        "reset_optimizer": args.reset_optimizer,
        "action_order": ACTION_ORDER,
        "batch_size": args.batch_size,
        "sampling": args.sampling,
        "target_steps": args.steps,
        "resume": args.resume,
        "resume_fallback": args.resume_fallback,
        "resolved_resume": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "device": str(device),
        "cuda_memory_fraction": args.cuda_memory_fraction,
    }
    model, losses = train_texts(
        texts,
        config=config,
        tokenizer=tokenizer,
        action_labels=action_labels,
        sampling=args.sampling,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        resume=resume_checkpoint,
        out=args.out,
        save_every=args.save_every,
        metrics_out=args.metrics_out,
        metadata=training_metadata,
        device=str(device),
        cuda_memory_fraction=args.cuda_memory_fraction,
        action_loss_weight=args.action_loss_weight,
        first_token_loss_weight=args.first_token_loss_weight,
        action_span_loss_weight=args.action_span_loss_weight,
        action_span_weight_overrides=action_span_weight_overrides,
        structural_loss_weight=args.structural_loss_weight,
        output_format=args.output_format,
        freeze_backbone=args.freeze_backbone,
        reset_optimizer=args.reset_optimizer,
        probe_path=args.probe,
        eval_every=args.eval_every,
        abort_after_step=args.abort_after_step,
        collapse_threshold=args.collapse_threshold,
    )
    report.update({
        "checkpoint": str(args.out),
        "steps": args.steps,
        "completed_steps": args.steps,
        "batch_size": args.batch_size,
        "sampling": args.sampling,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "action_loss_weight": args.action_loss_weight,
        "first_token_loss_weight": args.first_token_loss_weight,
        "action_span_loss_weight": args.action_span_loss_weight,
        "action_span_weight_overrides": action_span_weight_overrides,
        "structural_loss_weight": args.structural_loss_weight,
        "resume": args.resume,
        "resume_fallback": args.resume_fallback,
        "resolved_resume": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "save_every": args.save_every,
        "metrics_out": str(args.metrics_out) if args.metrics_out is not None else None,
        "device": str(device),
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "loss_summary": loss_summary(losses),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
    })
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
