from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from psm_model.hf_lora_train import DEFAULT_MODELS
from psm_model.remember_cli import PROD_STORAGE_MAX_NEW_TOKENS, apply_product_boundary

from prod_memory.eval_grounding import (
    DEFAULT_FIXTURES,
    DEFAULT_OUT,
    aggregate_by_suite,
    aggregate_metrics,
)
from prod_memory.grounding import (
    apply_storage_guards,
    grounding_overlap_score,
    has_curriculum_bleed,
    is_fail_safe_report,
    key_tokens_grounded,
    stored_text_from_decision,
    would_model_store,
)
from prod_memory.hf_prompts import apply_chat_prompt, storage_inference_messages
from prod_memory.storage_rewards import is_json_complete


from prod_memory.eval_classify import binary_predicts_store


def _binary_classify_match(expect_action: str, raw: str) -> bool:
    predicts_store = binary_predicts_store(raw)
    if expect_action == "ignore":
        return not predicts_store
    return predicts_store


# compact_storage_json() sort_keys=True + no separator spaces makes "action" the
# alphabetically-first (and therefore first-emitted) key in every StorageDecision.
# When the model commits to ignore, this literal prefix appears reliably within the
# first ~10-15 generated tokens, well before facts/indexables/memory/reasoning scaffold.
_IGNORE_JSON_PREFIX = '{"action":"ignore'

# Minimal valid closing shape once we've detected an ignore decision. schema.py's
# parse_and_validate_storage_decision() requires memory=null for action=ignore, and a
# non-empty reasoning string on every decision, so we can't just truncate mid-object.
_IGNORE_JSON_SUFFIX = '","memory":null,"facts":[],"indexables":[],"reasoning":"Nothing durable to store."}'


class _IgnorePrefixStoppingCriteria(StoppingCriteria):
    """Halts generation as soon as the decoded output unambiguously commits to action=ignore.

    Only meaningful for output_format="json" (the compact StorageDecision scaffold). Batch
    size is always 1 for this session, so decoding the whole new-token span each step is cheap
    relative to the tokens it saves (an ignore decision otherwise pays for the full
    facts/indexables/memory/reasoning scaffold even though none of that content is used).
    """

    def __init__(self, tokenizer: Any, prompt_len: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.triggered = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        new_tokens = input_ids[0, self.prompt_len :]
        # ponytail: cheap length guard avoids decoding on every single step before we're close.
        if new_tokens.shape[0] < 6:
            return False
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if _IGNORE_JSON_PREFIX in text:
            self.triggered = True
            return True
        return False


def _force_close_ignore_json(text: str) -> str:
    """Truncate at the unambiguous ignore prefix and append a minimal valid closing shape."""
    idx = text.find(_IGNORE_JSON_PREFIX)
    if idx == -1:
        return text
    prefix_end = idx + len(_IGNORE_JSON_PREFIX)
    return text[:prefix_end] + _IGNORE_JSON_SUFFIX


class HfGenerationSession:
    def __init__(self, model: Any, tokenizer: Any, *, adapter_name: str | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name

    def _input_device(self) -> torch.device:
        # ponytail: device_map="auto" has no model.device
        return next(self.model.parameters()).device

    def generate(self, llm_response: str, *, output_format: str, max_new_tokens: int) -> str:
        text, _ = self.generate_with_token_count(
            llm_response, output_format=output_format, max_new_tokens=max_new_tokens
        )
        return text

    def generate_with_token_count(
        self, llm_response: str, *, output_format: str, max_new_tokens: int
    ) -> tuple[str, int]:
        if self.adapter_name:
            self.model.set_adapter(self.adapter_name)
        messages = storage_inference_messages(llm_response, output_format=output_format)
        prompt = apply_chat_prompt(messages, self.tokenizer)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._input_device()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        stopping_criteria = None
        ignore_stop: _IgnorePrefixStoppingCriteria | None = None
        if output_format == "json":
            # ponytail: only the compact JSON scaffold has a deterministic
            # {"action":"ignore" prefix worth short-circuiting on; other output
            # formats render ignore as a bare word/tag that's already short.
            ignore_stop = _IgnorePrefixStoppingCriteria(self.tokenizer, prompt_len)
            stopping_criteria = StoppingCriteriaList([ignore_stop])

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )
        new_tokens = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        token_count = int(new_tokens.shape[0])

        if ignore_stop is not None and ignore_stop.triggered and _IGNORE_JSON_PREFIX in text:
            text = _force_close_ignore_json(text)
            # ponytail: report the token count we actually consumed generating, not the
            # length of the force-closed text (which includes synthesized suffix chars).
            token_count = int(new_tokens.shape[0])
        return text, token_count


def open_hf_base_session(
    *,
    model_key: str = "qwen0.5b",
    model_id: str | None = None,
    device: str = "cuda",
) -> HfGenerationSession:
    resolved = model_id or DEFAULT_MODELS.get(model_key) or model_key
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return HfGenerationSession(model, tokenizer)


def open_hf_two_pass_sessions(
    binary_dir: Path,
    extract_dir: Path,
    *,
    model_key: str = "qwen0.5b",
    model_id: str | None = None,
    device: str = "cuda",
) -> tuple[HfGenerationSession, HfGenerationSession]:
    """One base model + two LoRA adapters (half the VRAM of two open_hf_session calls)."""
    resolved = model_id or DEFAULT_MODELS.get(model_key) or model_key
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(binary_dir), adapter_name="binary")
    model.load_adapter(str(extract_dir), adapter_name="extract")
    model.eval()
    return (
        HfGenerationSession(model, tokenizer, adapter_name="binary"),
        HfGenerationSession(model, tokenizer, adapter_name="extract"),
    )


def open_hf_session(
    adapter_dir: Path | None,
    *,
    model_key: str = "qwen0.5b",
    model_id: str | None = None,
    device: str = "cuda",
) -> HfGenerationSession:
    if adapter_dir is None:
        return open_hf_base_session(model_key=model_key, model_id=model_id, device=device)
    resolved = model_id or DEFAULT_MODELS.get(model_key) or model_key
    # ponytail: tokenizer lives on base model, not LoRA adapter dir
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return HfGenerationSession(model, tokenizer)


def run_hf_case(
    session: HfGenerationSession,
    case: dict[str, Any],
    *,
    output_format: str,
    max_new_tokens: int,
    debug_raw: bool = False,
) -> dict[str, Any]:
    llm_response = str(case["llmResponse"])
    raw, generated_token_count = session.generate_with_token_count(
        llm_response, output_format=output_format, max_new_tokens=max_new_tokens
    )
    report = apply_product_boundary(raw, output_format=output_format)
    decision = report.get("parsed")
    if not isinstance(decision, dict):
        decision = {}
    stored_text = stored_text_from_decision(decision)
    model_store = would_model_store(decision)
    guarded = apply_storage_guards(llm_response, decision)
    effective_stored = model_store and not guarded["rejected"]
    overlap = grounding_overlap_score(llm_response, stored_text)
    key_tokens = case.get("keyTokens") if isinstance(case.get("keyTokens"), list) else []
    content_grounded = effective_stored and (
        key_tokens_grounded([str(token) for token in key_tokens], stored_text) or bool(overlap["grounded"])
    )
    parse_valid = bool(decision) and report.get("repair_status") != "unrecoverable"
    facts = decision.get("facts") if isinstance(decision.get("facts"), list) else []
    row: dict[str, Any] = {
        "id": case["id"],
        "suite": case["suite"],
        "expectAction": case.get("expectAction"),
        "action": decision.get("action"),
        "repair_status": report.get("repair_status"),
        "model_would_store": model_store,
        "effective_stored": effective_stored,
        "guard_rejected": guarded["rejected"],
        "guard_route": guarded["route"],
        "fail_safe": is_fail_safe_report(report),
        "curriculum_bleed": effective_stored and has_curriculum_bleed(stored_text),
        "content_grounded": content_grounded,
        "grounding_overlap": overlap["overlap"],
        "grounding_required": overlap["required"],
        "memory_content": stored_text[:240] if stored_text else None,
        "issues": report.get("issues"),
        "raw_output": raw if debug_raw else (raw[:500] if raw else None),
        "raw_output_len": len(raw) if raw else 0,
        "generated_token_count": generated_token_count,
        "hit_token_ceiling": generated_token_count >= max_new_tokens,
        "json_closes_cleanly": is_json_complete(raw, parse_valid=parse_valid),
        "facts_count_anomalous": len(facts) >= 3 and len(llm_response.split()) < 50,
    }
    if output_format == "binary" and case.get("expectAction"):
        row["classify_match"] = _binary_classify_match(str(case["expectAction"]), raw)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prod fixture grounding eval for HF LoRA adapter.")
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="")
    parser.add_argument("--model", choices=sorted(DEFAULT_MODELS), default="qwen0.5b")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--output-format", default="tagged", choices=["json", "tagged", "minimal", "binary", "minimal_extract"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=PROD_STORAGE_MAX_NEW_TOKENS)
    parser.add_argument("--debug-raw", action="store_true", help="Store untruncated raw_output in the report (default: 500-char slice).")
    args = parser.parse_args(argv)

    fixture = json.loads(args.fixtures.read_text(encoding="utf-8"))
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise SystemExit(f"Invalid fixtures file: {args.fixtures}")

    label = args.checkpoint_label or args.adapter_dir.parent.name
    args.out.parent.mkdir(parents=True, exist_ok=True)

    session = open_hf_session(
        args.adapter_dir,
        model_key=args.model,
        model_id=args.model_id,
        device=args.device,
    )
    results = [
        run_hf_case(
            session,
            case,
            output_format=args.output_format,
            max_new_tokens=args.max_new_tokens,
            debug_raw=args.debug_raw,
        )
        for case in cases
        if isinstance(case, dict)
    ]
    aggregate = aggregate_metrics(results)
    if any(row.get("classify_match") is not None for row in results):
        hits = sum(1 for row in results if row.get("classify_match"))
        aggregate["classify_match"] = hits
        aggregate["classify_match_rate"] = round(hits / max(1, len(results)), 4)
    report = {
        "checkpoint": label,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "model_key": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixtures": str(args.fixtures.resolve()),
        "eval_type": "prod_fixtures_hf_lora",
        "max_new_tokens": args.max_new_tokens,
        "suites": aggregate_by_suite(results),
        "aggregate": aggregate,
        "cases": results,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": label, "aggregate": aggregate, "suites": report["suites"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
