"""Production ONNX conversion pipeline for PSM's 3 LoRA adapters (storage, retrieval-plan,
consolidation), all trained on Qwen2.5-0.5B-Instruct with an identical LoRA config
(r=16, alpha=32, target_modules=[q,k,v,o,gate,up,down]_proj).

Produces ONE shared base ONNX graph with swappable LoRA input slots (via Olive's dynamo exporter +
the ExtractAdapters pass), plus one .onnx_adapter file per task. All three are loadable at runtime via
ONNX Runtime GenAI's Adapters/SetActiveAdapter API (C# or Python) with zero Python needed in production.

Root causes fixed here that silently no-op'd or crashed in earlier attempts (see
transient-dazzling-lake.md's plan for the full trail):
  - Olive's --use_model_builder merges LoRA into base weights at conversion time; there is nothing
    left for ExtractAdapters to find afterwards. Must use --use_dynamo_exporter with -a <adapter> to
    keep LoRA as separate graph branches.
  - Olive 0.13.0's dynamo-exporter Cache-compatibility patch (_patch_model_if_necessary) only covers
    transformers 4.45 <= version < 5.0. This script MUST run under the pinned conversion venv
    (transformers==4.48.3), not the main project environment (which runs transformers 5.x).
  - The genai_config.json's "past_present_share_buffer" must be false for this graph type: it's a
    Model-Builder-only static-KV-buffer optimization, incompatible with this graph's plain
    Concat-based cache growth (causes a shape-mismatch crash on the very first prefill otherwise).

One-time venv setup (only if the conversion venv doesn't exist yet):
    python -m venv psm-model/prod-memory/onnx-convert-venv
    psm-model/prod-memory/onnx-convert-venv/Scripts/pip install transformers==4.48.3 olive-ai==0.13.0 \
        peft onnx onnxruntime-genai accelerate safetensors sentencepiece torch

Usage (must run with the pinned venv's python, not the main project's):
    psm-model/prod-memory/onnx-convert-venv/Scripts/python psm-model/scripts/convert_adapters_onnx.py \
        --output-dir psm-model/prod-memory/onnx-runtime/v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REQUIRED_TRANSFORMERS = "4.48.3"

# Domain -> task -> PEFT adapter directory. "coding"/"storage" is used as the seed adapter for the
# initial trace+extract (its weight values don't matter for the resulting graph shape, only its
# config: r/alpha/target_modules, which is confirmed identical across every adapter in every domain
# below — that identical config is what lets all domains share one base ONNX graph).
#
# "conversational" is the extension point for the LoCoMo-style personal/social-memory adapters
# (see transient-dazzling-lake.md's Phase 2/3): once trained, add their checkpoint paths here under
# "conversational" using the same 3 task keys. A domain is only converted if ALL THREE of its task
# checkpoints exist on disk — a partial domain is skipped with a warning rather than half-converted,
# matching dotnet/src/PsmMemory.Core/Runtime/OnnxPsmRuntime.cs's "all three or none" loading rule.
DOMAIN_ADAPTERS: dict[str, dict[str, Path]] = {
    "coding": {
        "storage": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-storage-v16b-qwen0.5b/checkpoint-800",
        "retrieval_plan": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-recall-plan-v3-qwen0.5b/adapter",
        "consolidation": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-consolidation-v4-qwen0.5b/adapter",
    },
    "conversational": {
        # "storage": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-conversational-storage-v1-qwen0.5b/adapter",
        # "retrieval_plan": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-conversational-retrieval-plan-v1-qwen0.5b/adapter",
        # "consolidation": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-conversational-consolidation-v1-qwen0.5b/adapter",
    },
}

# Matches OnnxPsmRuntime.AdapterFilePrefix in the C# SDK — must stay in sync.
DOMAIN_FILE_PREFIX = {"coding": "", "conversational": "conversational_"}

TASKS = ("storage", "retrieval_plan", "consolidation")


def available_domains() -> dict[str, dict[str, Path]]:
    """Domains where all 3 task checkpoints actually exist on disk."""
    result = {}
    for domain, tasks in DOMAIN_ADAPTERS.items():
        if all(t in tasks and tasks[t].exists() for t in TASKS):
            result[domain] = tasks
        elif tasks:
            print(f"skipping domain '{domain}': incomplete adapter set ({sorted(tasks)})", file=sys.stderr)
    return result

# Known-correct Qwen2.5-0.5B-Instruct architecture values, confirmed via a working Olive
# --use_model_builder --use_ort_genai capture earlier in this investigation (same base model).
GENAI_CONFIG_TEMPLATE = {
    "model": {
        "bos_token_id": 151643,
        "context_length": 32768,
        "decoder": {
            "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
            "filename": "model.onnx",
            "head_size": 64,
            "hidden_size": 896,
            "inputs": {
                "input_ids": "input_ids",
                "attention_mask": "attention_mask",
                "past_key_names": "past_key_values.%d.key",
                "past_value_names": "past_key_values.%d.value",
            },
            "outputs": {
                "logits": "logits",
                "present_key_names": "present.%d.key",
                "present_value_names": "present.%d.value",
            },
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
        },
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "type": "qwen2",
        "vocab_size": 151936,
    },
    "search": {
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": True,
        "length_penalty": 1.0,
        "max_length": 4096,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "num_return_sequences": 1,
        # Model-Builder-only static-KV-buffer optimization; this graph (plain dynamo/torch.export
        # Concat-based cache growth) crashes on the first prefill if this is left true.
        "past_present_share_buffer": False,
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    },
}

# Reused as-is from an earlier working Olive --use_ort_genai capture of this exact base model —
# these are architecture-level tokenizer/config files, not adapter-specific, so no need to regenerate.
TOKENIZER_SCAFFOLDING_SOURCE = REPO_ROOT / "psm-model/prod-memory/onnx-spike/v16b-base-capture"
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "config.json",
]


def check_venv():
    import transformers

    if transformers.__version__ != REQUIRED_TRANSFORMERS:
        sys.exit(
            f"This script must run under the pinned conversion venv (transformers=={REQUIRED_TRANSFORMERS}), "
            f"got transformers=={transformers.__version__}. Olive 0.13.0's dynamo-exporter Cache-compatibility "
            f"patch only covers transformers 4.45-5.0; running under the main project's transformers 5.x will "
            f"silently reproduce the AttributeError/pytree-mismatch bugs this pipeline was built to avoid."
        )


def run_olive(args: list[str]):
    """Always runs from REPO_ROOT with fully-resolved absolute paths in args — Olive joins relative
    -o/-m paths onto its own cwd, so mixing a relative path with an explicit `cwd=` doubles the prefix.
    """
    env_fix = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    import os

    env = {**os.environ, **env_fix}
    print(f"$ {' '.join(args)}", file=sys.stderr)
    subprocess.run([sys.executable, "-m", "olive", *args], cwd=REPO_ROOT, env=env, check=True)


def capture_and_extract(work_dir: Path, seed_adapter: Path) -> Path:
    """Trace the base model with the seed adapter attached (LoRA kept unmerged), then extract the
    LoRA weights into swappable graph inputs. Returns the directory containing the extracted model.onnx.
    """
    traced_dir = (work_dir / "_traced").resolve()
    extracted_dir = (work_dir / "_extracted").resolve()
    seed_adapter = seed_adapter.resolve()

    if not (extracted_dir / "model.onnx").exists():
        if not (traced_dir / "model.onnx").exists():
            run_olive(
                [
                    "capture-onnx-graph",
                    "-m", BASE_MODEL,
                    "-a", str(seed_adapter),
                    "--use_dynamo_exporter",
                    "-o", str(traced_dir),
                    "--log_level", "1",
                ]
            )
        run_olive(
            [
                "run-pass",
                "--pass-name", "ExtractAdapters",
                "-m", str(traced_dir),
                "-o", str(extracted_dir),
                "--log_level", "1",
            ]
        )

    verify_lora_inputs(extracted_dir / "model.onnx")
    return extracted_dir


def verify_lora_inputs(model_path: Path):
    """Don't trust Olive's success messages alone (they've silently no-op'd before) — inspect the
    actual graph.
    """
    import onnx

    m = onnx.load(str(model_path), load_external_data=False)
    lora_inputs = [i.name for i in m.graph.input if "lora" in i.name.lower()]
    if not lora_inputs:
        sys.exit(
            f"ExtractAdapters produced a graph with zero LoRA inputs at {model_path} — "
            "extraction silently no-op'd. Do not proceed; see the plan's diagnostic trail."
        )
    print(f"verified {len(lora_inputs)} LoRA graph inputs in {model_path.name}", file=sys.stderr)


def convert_adapter_weights(adapter_dir: Path, out_file: Path):
    """Export a PEFT checkpoint's LoRA weights directly to .onnx_adapter format.

    Deliberately does NOT shell out to Olive's `convert-adapters` CLI: that CLI strips adapter
    checkpoints down to bare names like "model.layers.0...lora_A.weight", but the shared base graph
    (traced with `-a <adapter>`, which PEFT attaches as its in-memory adapter named "default") expects
    graph inputs named "...lora_A.default.weight" — the CLI's output silently fails to bind at runtime
    ("Invalid input name") for any adapter converted this way. This reimplements the CLI's own
    transform (see olive/cli/convert_adapters.py) with the ".default" segment restored.
    """
    if out_file.exists():
        return
    out_file = out_file.resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    import math

    import torch
    from olive.common.utils import WeightsFileFormat, save_weights
    from peft import LoraConfig, load_peft_weights

    lora_config = LoraConfig.from_pretrained(str(adapter_dir))
    if getattr(lora_config, "use_dora", False):
        raise ValueError("DoRA adapters are not supported for export.")
    scaling = (
        lora_config.lora_alpha / math.sqrt(lora_config.r)
        if getattr(lora_config, "use_rslora", False)
        else lora_config.lora_alpha / lora_config.r
    )

    adapter_weights = load_peft_weights(str(adapter_dir), device="cpu")
    transformed = {}
    for name, value in adapter_weights.items():
        new_name = name.replace("base_model.model.model", "model").replace(".weight", ".default.weight")
        float_weight = value.to(torch.float32).numpy().transpose().copy()
        if "lora_B" in new_name:
            float_weight *= scaling
        transformed[new_name] = float_weight

    save_weights(transformed, out_file, WeightsFileFormat.ONNX_ADAPTER)
    print(f"exported adapter weights (.default-named) to {out_file}", file=sys.stderr)


def assemble_output_dir(extracted_dir: Path, output_dir: Path, adapter_files: dict[str, Path]):
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(extracted_dir / "model.onnx", output_dir / "model.onnx")

    with open(output_dir / "genai_config.json", "w", encoding="utf-8") as f:
        json.dump(GENAI_CONFIG_TEMPLATE, f, indent=4)

    for fname in TOKENIZER_FILES:
        src = TOKENIZER_SCAFFOLDING_SOURCE / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
        else:
            print(f"warning: missing tokenizer scaffolding file {src}", file=sys.stderr)

    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(exist_ok=True)
    for name, path in adapter_files.items():
        shutil.copy2(path, adapters_dir / f"{name}.onnx_adapter")

    print(f"assembled production model directory at {output_dir}", file=sys.stderr)


# Deliberately the SAME prompt for all three: this validation's job is to confirm the swap mechanism
# itself works (SetActiveAdapter genuinely changes the model's behavior per adapter), not to re-prove
# each task's accuracy — that's what the dedicated 100-case gates (run separately, per task) are for.
SMOKE_PROMPT = (
    "You are a memory assistant for a coding agent.",
    "Fixed the null-pointer bug in auth.py by adding a guard before session.user access.",
)


def _generate(model, tokenizer, adapters, name, prompt) -> str:
    import onnxruntime_genai as og

    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=1024)
    generator = og.Generator(model, params)
    generator.set_active_adapter(adapters, name)
    generator.append_tokens(tokenizer.encode(prompt))
    output_tokens: list[int] = []
    while not generator.is_done() and len(output_tokens) < 200:
        generator.generate_next_token()
        output_tokens.append(generator.get_next_tokens()[0])
    # Decode the whole accumulated token list at once — decoding token-by-token can split a
    # multi-byte UTF-8 character across two decode() calls and mangle it into U+FFFD.
    return tokenizer.decode(output_tokens)


def validate(output_dir: Path, domains: dict[str, dict[str, Path]]):
    import onnxruntime_genai as og

    adapter_keys = [f"{DOMAIN_FILE_PREFIX[domain]}{task}" for domain, tasks in domains.items() for task in tasks]

    print("loading base model + all adapters for smoke validation...", file=sys.stderr)
    model = og.Model(str(output_dir))
    tokenizer = og.Tokenizer(model)
    adapters = og.Adapters(model)
    for name in adapter_keys:
        adapters.load(str(output_dir / "adapters" / f"{name}.onnx_adapter"), name)

    system, user = SMOKE_PROMPT
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    outputs = {}
    for name in adapter_keys:
        text = _generate(model, tokenizer, adapters, name, prompt)
        outputs[name] = text
        print(f"[{name}] {text[:200]!r}", file=sys.stderr)
        if not text.strip():
            sys.exit(f"validation failed for adapter '{name}': empty output")

    distinct = len(set(outputs.values()))
    if distinct < len(outputs):
        sys.exit(
            f"validation failed: only {distinct}/{len(outputs)} distinct outputs across adapters — "
            "SetActiveAdapter may not be swapping weights (same output for different adapters)."
        )

    print("all 3 adapters loaded, swapped, and produced distinct output for the same prompt.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=None, help="scratch dir for intermediate artifacts")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    check_venv()

    args.output_dir = args.output_dir.resolve()
    work_dir = (args.work_dir or (args.output_dir.parent / f"{args.output_dir.name}-work")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    domains = available_domains()
    if "coding" not in domains:
        sys.exit("'coding' domain adapters not found — this is the seed domain and must be present.")

    # The base graph's shape only depends on LoRA config (identical across every domain/task), so it
    # only needs to be traced once, seeded from coding/storage.
    extracted_dir = capture_and_extract(work_dir, seed_adapter=domains["coding"]["storage"])

    adapter_files: dict[str, Path] = {}
    for domain, tasks in domains.items():
        prefix = DOMAIN_FILE_PREFIX[domain]
        for task, checkpoint in tasks.items():
            adapter_key = f"{prefix}{task}"
            if domain == "coding" and task == "storage":
                # This exact adapter was the seed for capture_and_extract, so ExtractAdapters already
                # emitted a correctly name-matched .onnx_adapter for it — reuse instead of reconverting.
                auto_adapter = extracted_dir / "adapter_weights.onnx_adapter"
                if auto_adapter.exists():
                    adapter_files[adapter_key] = auto_adapter
                    continue
            out_file = work_dir / f"{adapter_key}.onnx_adapter"
            convert_adapter_weights(checkpoint, out_file)
            adapter_files[adapter_key] = out_file

    assemble_output_dir(extracted_dir, args.output_dir, adapter_files)

    if not args.skip_validate:
        validate(args.output_dir, domains)


if __name__ == "__main__":
    main()
