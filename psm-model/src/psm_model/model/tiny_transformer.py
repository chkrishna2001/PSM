from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


@dataclass(frozen=True)
class TinyDecoderConfig:
    vocab_size: int
    context_length: int = 512
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _torch():
    if torch is None:
        raise ImportError("psm_model.model requires PyTorch. Install torch to train or run the model.")
    return torch


def _nn():
    if nn is None:
        raise ImportError("psm_model.model requires PyTorch. Install torch to train or run the model.")
    return nn


_ModuleBase = nn.Module if nn is not None else object


class TinyDecoderModel(_ModuleBase):
    def __init__(self, config: TinyDecoderConfig):
        if nn is None:
            raise ImportError("psm_model.model requires PyTorch. Install torch to train or run the model.")
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([_DecoderBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight

        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length), persistent=False)
        self.apply(self._init_weights)

    def forward(self, input_ids: Any, labels: Any | None = None) -> dict[str, Any]:
        torch = _torch()
        _, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context length {self.config.context_length}")

        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, self.causal_mask[:, :, :seq_len, :seq_len])
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}

    @staticmethod
    def _init_weights(module: Any) -> None:
        nn = _nn()
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def save_checkpoint(self, path: Path) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path: Path, *, map_location: str = "cpu") -> "TinyDecoderModel":
        torch = _torch()
        payload = torch.load(path, map_location=map_location)
        model = cls(TinyDecoderConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        return model

    @classmethod
    def create(cls, config: TinyDecoderConfig) -> "TinyDecoderModel":
        return cls(config)

    @staticmethod
    def parameter_estimate(config: TinyDecoderConfig) -> int:
        embedding = config.vocab_size * config.n_embd
        positions = config.context_length * config.n_embd
        per_block = (
            4 * config.n_embd * config.n_embd
            + 2 * config.n_embd
            + 8 * config.n_embd * config.n_embd
            + 5 * config.n_embd
        )
        norm = 2 * config.n_embd
        return embedding + positions + config.n_layer * per_block + norm

    def generate(
        self,
        input_ids: Any,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Any:
        torch = _torch()
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    context = input_ids[:, -self.config.context_length :]
                    logits = self(context)["logits"][:, -1, :]
                    if temperature <= 0:
                        next_id = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        logits = logits / temperature
                        if top_k is not None:
                            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, -math.inf), logits)
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        next_id = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_id], dim=1)
                    if eos_id is not None and bool((next_id == eos_id).all()):
                        break
            return input_ids
        finally:
            self.train(was_training)


class _CausalSelfAttention(_ModuleBase):
    def __init__(self, config: TinyDecoderConfig):
        if nn is None:
            raise ImportError("psm_model.model requires PyTorch. Install torch to train or run the model.")
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Any, mask: Any) -> Any:
        torch = _torch()
        batch, seq_len, embd = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(embd, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, -math.inf)
        weights = torch.nn.functional.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, embd)
        return self.proj(y)


class _FeedForward(_ModuleBase):
    def __init__(self, config: TinyDecoderConfig):
        if nn is None:
            raise ImportError("psm_model.model requires PyTorch. Install torch to train or run the model.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


class _DecoderBlock(_ModuleBase):
    def __init__(self, config: TinyDecoderConfig):
        if nn is None:
            raise ImportError("psm_model.model requires PyTorch. Install torch to train or run the model.")
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embd)
        self.attn = _CausalSelfAttention(config)
        self.norm_2 = nn.LayerNorm(config.n_embd)
        self.ff = _FeedForward(config)

    def forward(self, x: Any, mask: Any) -> Any:
        x = x + self.attn(self.norm_1(x), mask)
        x = x + self.ff(self.norm_2(x))
        return x
