from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from psm_model.action_diagnostics import score_actions
from psm_model.action_classifier import predict_checkpoint_action
from psm_model.generate import load_checkpoint_metadata
from psm_model.model import TinyDecoderModel
from psm_model.schema import validate_storage_decision
from psm_model.tokenizer import ByteTokenizer, load_tokenizer
from psm_model.train import resolve_device


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Safe generation requires PyTorch. Install torch to run psm_model.safe_generate.") from exc
    return torch


def safe_storage_decision(
    checkpoint: Path,
    input_payload: dict[str, Any],
    *,
    output_format: str | None = None,
    device: str = "cpu",
    action_classifier: Path | None = None,
) -> dict[str, Any]:
    torch = _torch()
    device_obj = resolve_device(device, torch)
    metadata = load_checkpoint_metadata(checkpoint)
    active_output_format = output_format or str(metadata.get("output_format", "tagged"))
    tokenizer_path = checkpoint.with_suffix(".tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path) if tokenizer_path.exists() else ByteTokenizer()
    model = TinyDecoderModel.load_checkpoint(checkpoint, map_location=str(device_obj)).to(device_obj)
    model.eval()
    with torch.no_grad():
        scores = score_actions(model, tokenizer, input_payload, output_format=active_output_format, device=device_obj)
    decoder_action = min(scores.items(), key=lambda item: item[1])[0]
    classifier_scores = None
    if action_classifier is not None:
        model_action, classifier_scores = predict_checkpoint_action(action_classifier, input_payload, device=str(device_obj))
    else:
        model_action = decoder_action
    action = calibrate_action(model_action, input_payload)
    decision = constrained_decision(action, input_payload)
    validation = validate_storage_decision(decision)
    if not validation.ok:
        formatted = ", ".join(f"{issue.path}: {issue.message}" for issue in validation.issues)
        raise ValueError(f"constrained decision failed validation: {formatted}")
    return {
        "action_scores": {key: round(value, 6) for key, value in sorted(scores.items(), key=lambda item: item[1])},
        "calibrated_action": action,
        "classifier_action_scores": {key: round(value, 6) for key, value in sorted(classifier_scores.items(), key=lambda item: item[1], reverse=True)} if classifier_scores is not None else None,
        "decision": decision,
        "device": str(device_obj),
        "decoder_action": decoder_action,
        "model_action": model_action,
        "output_format": active_output_format,
        "valid": True,
    }


def calibrate_action(model_action: str, input_payload: dict[str, Any]) -> str:
    """Apply conservative product guardrails for obvious direct-memory cases."""
    text = _clean_text(_input_text(input_payload)).lower()
    context = str(input_payload.get("context") or "").lower()

    if _is_noise(text):
        return "ignore"
    if _has_conflict_marker(text) and context:
        if _has_storeable_replacement(text):
            return "flag_and_store"
        return "flag_conflict"
    if _has_update_marker(text) and context:
        return "update_existing"
    if _has_durable_rule_or_preference(text):
        return "promote_semantic"
    if _has_event_marker(text):
        return "store_episodic"
    return model_action


def constrained_decision(action: str, input_payload: dict[str, Any]) -> dict[str, Any]:
    text = _input_text(input_payload)
    evidence = _clean_text(text)
    if action == "ignore":
        return {"action": "ignore", "memory": None, "facts": [], "reasoning": "No durable memory value was selected."}

    memory_type = "episodic" if action == "store_episodic" else "semantic"
    content = _content_for(action, evidence)
    predicate = {
        "promote_semantic": "stated_preference",
        "store_episodic": "reported_event",
        "update_existing": "updated_memory",
        "flag_conflict": "conflicting_memory",
        "flag_and_store": "corrected_memory",
    }.get(action, "stated_memory")
    memory: dict[str, Any] = {
        "type": memory_type,
        "content": content,
        "strength": 0.75,
        "decay_rate": 0.05 if memory_type == "episodic" else 0.02,
        "emotional_weight": 0.2,
        "confidence": 0.7,
        "tags": [_slug(action), memory_type],
    }
    if memory_type == "episodic" and input_payload.get("source_timestamp"):
        timestamp = str(input_payload["source_timestamp"])
        memory["temporal_expression"] = timestamp[:10]
        memory["resolved_time"] = timestamp[:10]
    fact: dict[str, Any] = {
        "subject": _subject(evidence),
        "predicate": predicate,
        "value": _value_text(evidence),
        "confidence": 0.7,
        "inference_kind": "explicit",
        "evidence_text": evidence,
    }
    if memory.get("temporal_expression"):
        fact["temporal_expression"] = memory["temporal_expression"]
        fact["resolved_time"] = memory["resolved_time"]
    return {"action": action, "memory": memory, "facts": [fact], "reasoning": "Action selected by the 50M PSM model; fields were emitted by constrained extractive fallback."}


def _input_text(input_payload: dict[str, Any]) -> str:
    for key in ("conversation", "text", "message"):
        value = input_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(input_payload, ensure_ascii=False, sort_keys=True)


def _clean_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if text.lower().startswith("user:"):
        text = text[5:].strip()
    return text[:500] or "Current input"


def _content_for(action: str, evidence: str) -> str:
    prefix = {
        "promote_semantic": "Durable semantic memory",
        "store_episodic": "Episodic memory",
        "update_existing": "Updated memory",
        "flag_conflict": "Potential conflicting memory",
        "flag_and_store": "Corrected memory",
    }.get(action, "Memory")
    return f"{prefix}: {evidence}"[:500]


def _subject(evidence: str) -> str:
    match = re.match(r"([A-Z][A-Za-z0-9_-]{1,40}):", evidence)
    return match.group(1) if match else "User"


def _value_text(evidence: str) -> str:
    return evidence[:240] or "Current input"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "memory"


def _is_noise(text: str) -> bool:
    durable_markers = (
        "prefer",
        "remember",
        "always",
        "for future",
        "for this repo",
        "correction",
        "actually",
        "met ",
        "ran ",
        "uploaded",
        "fixed ",
        "created ",
        "validated",
    )
    noise_patterns = (
        "okay thanks",
        "ok thanks",
        "thanks haha",
        "haha",
        "sounds good",
        "please continue",
        "terminal is still running",
        "weather",
    )
    return any(pattern in text for pattern in noise_patterns) and not any(marker in text for marker in durable_markers)


def _has_durable_rule_or_preference(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "i prefer ",
            "i want future",
            "for future ",
            "always ",
            "for this repo ",
            "remember that ",
            "my preference is ",
        )
    )


def _has_event_marker(text: str) -> bool:
    return bool(
        re.search(r"\b(today|yesterday|on \d{4}-\d{2}-\d{2}|at \d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", text)
        or any(
            marker in text
            for marker in (
                " i met ",
                " i ran ",
                " i uploaded ",
                " i fixed ",
                " i created ",
                " i validated ",
                " completed ",
            )
        )
    )


def _has_update_marker(text: str) -> bool:
    return any(marker in text for marker in ("correction:", "instead", "replace", "update the", "now prefer"))


def _has_conflict_marker(text: str) -> bool:
    return any(marker in text for marker in ("actually", "do not", "don't", "contradict", "conflict"))


def _has_storeable_replacement(text: str) -> bool:
    return any(marker in text for marker in ("instead", "now ", "replace", "prefer", "always", "for future"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a schema-valid StorageDecision using model action scoring plus constrained fields.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", help="JSON object payload")
    parser.add_argument("--output-format", choices=["json", "tagged", "at_tag", "action"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-classifier", type=Path, help="Optional standalone action classifier checkpoint to select the action before calibration.")
    args = parser.parse_args()

    report = safe_storage_decision(args.checkpoint, json.loads(args.input), output_format=args.output_format, device=args.device, action_classifier=args.action_classifier)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
