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
    n_action: int = 6


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
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([_DecoderBlock(config) for _ in range(config.n_layer)])
        self.norm = _RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.head.weight
        self.action_head = nn.Linear(config.n_embd, config.n_action)

        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: Any,
        labels: Any | None = None,
        *,
        loss_weights: Any | None = None,
        action_labels: Any | None = None,
        action_positions: Any | None = None,
        action_loss_weight: float = 0.0,
    ) -> dict[str, Any]:
        torch = _torch()
        _, seq_len = input_ids.shape
        if seq_len > self.config.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context length {self.config.context_length}")

        x = self.token_embedding(input_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        lm_loss = None
        if labels is not None:
            token_losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(labels.shape)
            valid = labels != -100
            if loss_weights is not None:
                weights = torch.where(valid, loss_weights.to(token_losses.device), torch.zeros_like(token_losses))
            else:
                weights = valid.to(token_losses.dtype)
            lm_loss = (token_losses * weights).sum() / weights.sum().clamp_min(1.0)
            loss = lm_loss

        action_logits = None
        action_loss = None
        if action_positions is not None:
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            action_states = x[batch_indices, action_positions]
            action_logits = self.action_head(action_states)
            if action_labels is not None:
                action_loss = torch.nn.functional.cross_entropy(action_logits, action_labels)
                if loss is None:
                    loss = action_loss * action_loss_weight
                elif action_loss_weight > 0:
                    loss = loss + action_loss * action_loss_weight
        return {"logits": logits, "action_logits": action_logits, "loss": loss, "lm_loss": lm_loss, "action_loss": action_loss}

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
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        allowed_missing = {"action_head.weight", "action_head.bias"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(f"incompatible checkpoint {path}: missing={missing}, unexpected={unexpected}")
        return model

    @classmethod
    def create(cls, config: TinyDecoderConfig) -> "TinyDecoderModel":
        return cls(config)

    @staticmethod
    def parameter_estimate(config: TinyDecoderConfig) -> int:
        embedding = config.vocab_size * config.n_embd
        per_block = (
            4 * config.n_embd * config.n_embd
            + config.n_embd
            + 8 * config.n_embd * config.n_embd
            + 5 * config.n_embd
        )
        norm = config.n_embd
        action_head = config.n_embd * config.n_action + config.n_action
        return embedding + config.n_layer * per_block + norm + action_head

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
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Any) -> Any:
        torch = _torch()
        batch, seq_len, embd = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(embd, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
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
        self.norm_1 = _RMSNorm(config.n_embd)
        self.attn = _CausalSelfAttention(config)
        self.norm_2 = _RMSNorm(config.n_embd)
        self.ff = _FeedForward(config)

    def forward(self, x: Any) -> Any:
        x = x + self.attn(self.norm_1(x))
        x = x + self.ff(self.norm_2(x))
        return x


class _RMSNorm(_ModuleBase):
    def __init__(self, dim: int, eps: float = 1e-6):
        if nn is None:
            raise ImportError("psm_model.model requires PyTorch. Install torch to train or run the model.")
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Any) -> Any:
        torch = _torch()
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight


def _apply_rope(q: Any, k: Any) -> tuple[Any, Any]:
    torch = _torch()
    _, _, seq_len, head_dim = q.shape
    half_dim = head_dim // 2
    positions = torch.arange(seq_len, device=q.device, dtype=q.dtype)
    freqs = torch.arange(half_dim, device=q.device, dtype=q.dtype)
    inv_freq = 1.0 / (10000 ** (freqs / half_dim))
    angles = positions[:, None] * inv_freq[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return _rotate_half(q, cos, sin), _rotate_half(k, cos, sin)


def _rotate_half(x: Any, cos: Any, sin: Any) -> Any:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)
