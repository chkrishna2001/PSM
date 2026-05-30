from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from nano_psm.dataset import NanoPsmJsonlDataset, collate_batch, load_jsonl
from nano_psm.evaluate import evaluate_model
from nano_psm.model import NanoPsmConfig, build_model, require_torch
from nano_psm.schema import ACTIONS, action_to_id, normalize_action
from nano_psm.tokenizer import HashTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Nano PSM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--extra-validation", action="append", default=[], help="Additional validation set as name=path.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch, _ = require_torch()
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config_doc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_config = NanoPsmConfig(**config_doc["model"])
    train_config = config_doc["training"]
    max_steps = args.max_steps or int(train_config["max_steps"])
    eval_every = args.eval_every or int(train_config["eval_every"])
    save_every = args.save_every or int(train_config["save_every"])
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(torch, args.device)
    tokenizer = HashTokenizer(
        vocab_size=model_config.vocab_size,
        max_length=model_config.max_sequence_length,
    )
    tokenizer.save(checkpoint_dir / "tokenizer.json")
    (checkpoint_dir / "config.json").write_text(json.dumps(config_doc, indent=2), encoding="utf-8")

    train_examples = load_jsonl(args.train)
    validation_examples = load_jsonl(args.validation)
    extra_validation_examples = {
        name: load_jsonl(path)
        for name, path in parse_named_paths(args.extra_validation).items()
    }
    action_weights = compute_action_weights(torch, train_examples, device)
    train_dataset = NanoPsmJsonlDataset(train_examples, tokenizer)
    validation_dataset = NanoPsmJsonlDataset(validation_examples, tokenizer)
    extra_validation_datasets = {
        name: NanoPsmJsonlDataset(examples, tokenizer)
        for name, examples in extra_validation_examples.items()
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        collate_fn=collate_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        collate_fn=collate_batch,
    )
    extra_validation_loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(train_config["batch_size"]),
            shuffle=False,
            collate_fn=collate_batch,
        )
        for name, dataset in extra_validation_datasets.items()
    }

    wrapped = build_model(model_config)
    model = wrapped.module.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    state = {
        "global_step": 0,
        "best_score": -1.0,
        "model_name": config_doc["name"],
        "selection_mode": selection_mode(args.extra_validation),
    }
    last_checkpoint = checkpoint_dir / "checkpoint-last.pt"
    if args.resume == "auto" and last_checkpoint.exists():
        state = load_checkpoint(torch, last_checkpoint, model, optimizer, device)
        if state.get("selection_mode") != selection_mode(args.extra_validation):
            state["best_score"] = -1.0
            state["selection_mode"] = selection_mode(args.extra_validation)
    elif args.init_checkpoint:
        load_model_weights(torch, Path(args.init_checkpoint), model, device)

    print(json.dumps({
        "status": "training_start",
        "model": config_doc["name"],
        "parameter_budget": wrapped.parameter_budget_note(),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "extra_validation_examples": {name: len(examples) for name, examples in extra_validation_examples.items()},
        "device": str(device),
        "init_checkpoint": args.init_checkpoint,
        "resume_step": state["global_step"],
        "max_steps": max_steps,
    }, indent=2))

    if state["global_step"] == 0:
        metrics_by_name = evaluate_all(model, validation_loader, extra_validation_loaders, device)
        score = selection_score(metrics_by_name)
        append_validation_metrics(checkpoint_dir, 0, score, metrics_by_name)
        if score > state["best_score"]:
            state["best_score"] = score
            save_checkpoint(torch, checkpoint_dir / "checkpoint-best.pt", model, optimizer, state)
            write_trainer_state(checkpoint_dir, state)
        print(json.dumps({
            "status": "initial_validation",
            "score": score,
            "metrics": metrics_by_name,
        }, indent=2))

    grad_accum = int(train_config.get("gradient_accumulation_steps", 1))
    model.train()
    while state["global_step"] < max_steps:
        for batch in train_loader:
            batch = move_batch(batch, device)
            output = model(batch["input_ids"], batch["attention_mask"])
            losses = compute_losses(torch, output, batch, action_weights=action_weights)
            loss = losses["total"] / grad_accum
            loss.backward()

            if (state["global_step"] + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            state["global_step"] += 1
            if state["global_step"] % 25 == 0:
                append_jsonl(checkpoint_dir / "metrics.jsonl", {
                    "step": state["global_step"],
                    "split": "train",
                    **{key: float(value.detach().cpu()) for key, value in losses.items()},
                })

            if state["global_step"] % eval_every == 0:
                metrics_by_name = evaluate_all(model, validation_loader, extra_validation_loaders, device)
                score = selection_score(metrics_by_name)
                append_validation_metrics(checkpoint_dir, state["global_step"], score, metrics_by_name)
                if score > state["best_score"]:
                    state["best_score"] = score
                    save_checkpoint(torch, checkpoint_dir / "checkpoint-best.pt", model, optimizer, state)
                model.train()

            if state["global_step"] % save_every == 0:
                save_checkpoint(torch, last_checkpoint, model, optimizer, state)
                write_trainer_state(checkpoint_dir, state)

            if state["global_step"] >= max_steps:
                break

    save_checkpoint(torch, last_checkpoint, model, optimizer, state)
    write_trainer_state(checkpoint_dir, state)
    final_metrics_by_name = evaluate_all(model, validation_loader, extra_validation_loaders, device)
    print(json.dumps({
        "status": "training_complete",
        "global_step": state["global_step"],
        "best_score": state["best_score"],
        "final_validation": final_metrics_by_name["primary"],
        "final_validation_all": final_metrics_by_name,
        "checkpoint_dir": str(checkpoint_dir),
    }, indent=2))


def compute_action_weights(torch, examples, device):
    counts = Counter(action_to_id(normalize_action(example.output.get("action"))) for example in examples)
    total = sum(counts.values())
    if total <= 0:
        return None
    weights = [
        total / (len(ACTIONS) * counts[index]) if counts.get(index, 0) > 0 else 1.0
        for index in range(len(ACTIONS))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def parse_named_paths(values: list[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--extra-validation must use name=path, got: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--extra-validation must use name=path, got: {value}")
        parsed[name] = path
    return parsed


def selection_mode(extra_validation: list[str]) -> str:
    return "multi_validation_v1" if extra_validation else "primary_validation_v1"


def evaluate_all(model, primary_loader, extra_loaders, device) -> dict[str, dict[str, float]]:
    results = {"primary": evaluate_model(model, primary_loader, device)}
    for name, loader in extra_loaders.items():
        results[name] = evaluate_model(model, loader, device)
    return results


def append_validation_metrics(
    checkpoint_dir: Path,
    step: int,
    score: float,
    metrics_by_name: dict[str, dict[str, float]],
) -> None:
    append_jsonl(checkpoint_dir / "metrics.jsonl", {
        "step": step,
        "split": "validation",
        "score": score,
        **metrics_by_name["primary"],
    })
    for name, extra_metrics in metrics_by_name.items():
        if name == "primary":
            continue
        append_jsonl(checkpoint_dir / "metrics.jsonl", {
            "step": step,
            "split": f"validation:{name}",
            **extra_metrics,
        })


def compute_losses(torch, output, batch, action_weights=None):
    ce = torch.nn.functional.cross_entropy
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    mse = torch.nn.functional.mse_loss
    action_loss = ce(output["action_logits"], batch["action"], weight=action_weights)
    memory_type_loss = ce(output["memory_type_logits"], batch["memory_type"])
    score_mask = batch["has_score_target"] > 0
    if bool(score_mask.any().item()):
        score_loss = mse(output["scores"][score_mask], batch["scores"][score_mask])
    else:
        score_loss = output["scores"].sum() * 0.0
    indexable_loss = bce(output["indexable_logits"], batch["has_indexables"])
    fact_count_loss = ce(output["fact_count_logits"], batch["fact_count"])
    recall_count_loss = ce(output["recall_count_logits"], batch["recall_count"])
    total = (
        2.0 * action_loss
        + 1.0 * memory_type_loss
        + 1.5 * score_loss
        + 0.5 * indexable_loss
        + 0.3 * fact_count_loss
        + 0.3 * recall_count_loss
    )
    return {
        "total": total,
        "action_loss": action_loss,
        "memory_type_loss": memory_type_loss,
        "score_loss": score_loss,
        "indexable_loss": indexable_loss,
        "fact_count_loss": fact_count_loss,
        "recall_count_loss": recall_count_loss,
    }


def selection_score(metrics_by_name: dict[str, dict[str, float]]) -> float:
    scores = []
    penalties = 0.0
    for name, metrics in metrics_by_name.items():
        if metrics.get("rows", 0.0) < 50:
            continue
        score_quality = max(0.0, 1.0 - metrics.get("score_mae", 1.0))
        score = (
            0.45 * metrics.get("action_accuracy", 0.0)
            + 0.15 * metrics.get("memory_type_accuracy", 0.0)
            + 0.20 * score_quality
            + 0.10 * metrics.get("indexable_accuracy", 0.0)
            + 0.10 * metrics.get("recall_count_accuracy", 0.0)
        )
        scores.append(score)
        penalties += gate_penalty(name, metrics)
    if not scores:
        return -1.0
    return min(scores) - penalties


def gate_penalty(name: str, metrics: dict[str, float]) -> float:
    action_floor = 0.96
    recall_floor = 0.88
    decay_ceiling = 0.035
    penalty = 0.0
    if metrics.get("action_accuracy", 0.0) < action_floor:
        penalty += (action_floor - metrics.get("action_accuracy", 0.0)) * 5.0
    if metrics.get("recall_count_accuracy", 0.0) < recall_floor:
        penalty += (recall_floor - metrics.get("recall_count_accuracy", 0.0)) * 2.0
    if ("retention" in name or name == "primary") and metrics.get("decay_rate_mae", 0.0) > decay_ceiling:
        penalty += (metrics.get("decay_rate_mae", 0.0) - decay_ceiling) * 3.0
    return penalty


def resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def move_batch(batch, device):
    moved = {"ids": batch["ids"]}
    for key, value in batch.items():
        if key == "ids":
            continue
        moved[key] = value.to(device)
    return moved


def save_checkpoint(torch, path: Path, model, optimizer, state: dict[str, object]) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
    }, path)


def load_checkpoint(torch, path: Path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("state", {"global_step": 0, "best_score": -1.0})


def load_model_weights(torch, path: Path, model, device) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])


def write_trainer_state(checkpoint_dir: Path, state: dict[str, object]) -> None:
    (checkpoint_dir / "trainer-state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
