from __future__ import annotations

from dataclasses import dataclass

from .schema import ACTIONS, MEMORY_TYPES


@dataclass(frozen=True)
class NanoPsmConfig:
    vocab_size: int
    max_sequence_length: int
    embedding_dim: int
    encoder_layers: int
    attention_heads: int
    feed_forward_dim: int
    dropout: float


def require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for Nano PSM training. Use Colab or install torch locally.") from exc
    return torch, nn


def build_model(config: NanoPsmConfig):
    torch, nn = require_torch()
    return NanoPsmModel(config, nn)


class NanoPsmModel:
    def __init__(self, config: NanoPsmConfig, nn_module=None) -> None:
        torch_module = None
        if nn_module is None:
            torch_module, nn_module = require_torch()
        else:
            import torch as torch_module
        nn = nn_module
        torch = torch_module

        class _Model(nn.Module):
            def __init__(self, cfg: NanoPsmConfig) -> None:
                super().__init__()
                self.config = cfg
                self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=0)
                self.position_embedding = nn.Embedding(cfg.max_sequence_length, cfg.embedding_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.embedding_dim,
                    nhead=cfg.attention_heads,
                    dim_feedforward=cfg.feed_forward_dim,
                    dropout=cfg.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=False,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
                self.norm = nn.LayerNorm(cfg.embedding_dim)
                self.action_head = nn.Linear(cfg.embedding_dim, len(ACTIONS))
                self.memory_type_head = nn.Linear(cfg.embedding_dim, len(MEMORY_TYPES))
                self.score_head = nn.Sequential(
                    nn.Linear(cfg.embedding_dim, cfg.embedding_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.embedding_dim, 4),
                    nn.Sigmoid(),
                )
                self.indexable_head = nn.Linear(cfg.embedding_dim, 1)
                self.fact_count_head = nn.Linear(cfg.embedding_dim, 9)
                self.recall_count_head = nn.Linear(cfg.embedding_dim, 9)

            def forward(self, input_ids, attention_mask):
                batch_size, seq_len = input_ids.shape
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
                hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
                padding_mask = attention_mask <= 0
                encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
                mask = attention_mask.unsqueeze(-1).clamp(min=0.0, max=1.0)
                pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                pooled = self.norm(pooled)
                return {
                    "action_logits": self.action_head(pooled),
                    "memory_type_logits": self.memory_type_head(pooled),
                    "scores": self.score_head(pooled),
                    "indexable_logits": self.indexable_head(pooled).squeeze(-1),
                    "fact_count_logits": self.fact_count_head(pooled),
                    "recall_count_logits": self.recall_count_head(pooled),
                }

        self.module = _Model(config)
        self.config = config

    def parameter_budget_note(self) -> str:
        total = sum(param.numel() for param in self.module.parameters())
        return (
            f"{self.config.encoder_layers} layers, "
            f"{self.config.embedding_dim} dim, "
            f"{self.config.attention_heads} heads, "
            f"{total:,} parameters"
        )
