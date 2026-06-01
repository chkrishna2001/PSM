from __future__ import annotations

from dataclasses import asdict

from psm_model.model import TinyDecoderConfig, TinyDecoderModel


MODEL_PRESETS = {
    "debug": {
        "context_length": 2048,
        "n_layer": 2,
        "n_head": 4,
        "n_embd": 128,
    },
    "10m": {
        "context_length": 2048,
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 320,
    },
    "25m": {
        "context_length": 2048,
        "n_layer": 12,
        "n_head": 8,
        "n_embd": 416,
    },
    "50m": {
        "context_length": 2048,
        "n_layer": 16,
        "n_head": 8,
        "n_embd": 512,
    },
}


def config_from_preset(name: str, *, vocab_size: int, context_length: int | None = None) -> TinyDecoderConfig:
    try:
        values = dict(MODEL_PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"unknown model preset: {name}") from exc
    if context_length is not None:
        values["context_length"] = context_length
    return TinyDecoderConfig(vocab_size=vocab_size, **values)


def describe_preset(name: str, *, vocab_size: int, context_length: int | None = None) -> dict[str, object]:
    config = config_from_preset(name, vocab_size=vocab_size, context_length=context_length)
    return {
        "preset": name,
        "config": asdict(config),
        "parameter_estimate": TinyDecoderModel.parameter_estimate(config),
    }

