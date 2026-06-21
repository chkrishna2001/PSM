from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_MODELS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "smol360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _tokenize_sft_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    messages = row["messages"]
    prompt_messages = messages[:-1]
    assistant_text = messages[-1]["content"]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):
        raise ValueError(f"prompt prefix mismatch for row {row.get('id')}")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False)["input_ids"]
    # ponytail: never truncate assistant labels — trim prompt prefix if needed.
    if len(prompt_ids) + len(assistant_ids) > max_length:
        keep_prompt = max(256, max_length - len(assistant_ids))
        prompt_ids = prompt_ids[-keep_prompt:]
    full_ids = prompt_ids + assistant_ids
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + assistant_ids
    pad_id = tokenizer.pad_token_id or 0
    if len(full_ids) < max_length:
        pad = max_length - len(full_ids)
        full_ids = full_ids + [pad_id] * pad
        labels = labels + [-100] * pad
    elif len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]
    return {
        "input_ids": full_ids,
        "attention_mask": [1 if tid != pad_id else 0 for tid in full_ids],
        "labels": labels,
    }


def train_hf_lora(
    *,
    curriculum: Path,
    output_dir: Path,
    model_key: str = "qwen0.5b",
    model_id: str | None = None,
    max_length: int = 2048,
    steps: int = 1200,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 200,
    logging_steps: int = 20,
) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("hf train extras required: pip install torch transformers peft datasets accelerate") from exc

    resolved_model = model_id or DEFAULT_MODELS.get(model_key) or model_key
    rows = _load_rows(curriculum)
    if not rows:
        raise ValueError(f"empty curriculum: {curriculum}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized: list[dict[str, list[int]]] = []
    for index, row in enumerate(rows):
        tokenized.append(_tokenize_sft_row(row, tokenizer, max_length))
        if index and index % 200 == 0:
            print(f"tokenized {index}/{len(rows)} rows", flush=True)
    print(f"tokenized {len(rows)} rows", flush=True)
    lengths = sorted(len(item["input_ids"]) for item in tokenized)
    dataset = Dataset.from_list(tokenized)

    if os.environ.get("PSM_RUNPOD") == "1" and not torch.cuda.is_available():
        raise RuntimeError("PSM_RUNPOD=1 but CUDA unavailable — refusing CPU train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model on {device}", flush=True)

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch.float16}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(resolved_model, **load_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    print("starting trainer.train()", flush=True)
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metrics = {
        "model_id": resolved_model,
        "model_key": model_key,
        "curriculum": str(curriculum),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "steps": steps,
        "input_ids_p50": lengths[len(lengths) // 2],
        "input_ids_p90": lengths[int(len(lengths) * 0.9)],
        "input_ids_max": lengths[-1],
        "train_loss": train_result.training_loss,
    }
    (output_dir / "train.metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-0.5B / SmolLM2 on clean prod HF curriculum.")
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(DEFAULT_MODELS), default="qwen0.5b")
    parser.add_argument("--model-id", default=None, help="Override HF model id.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-steps", type=int, default=200)
    args = parser.parse_args(argv)
    metrics = train_hf_lora(
        curriculum=args.curriculum,
        output_dir=args.output_dir,
        model_key=args.model,
        model_id=args.model_id,
        max_length=args.max_length,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
