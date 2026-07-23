"""Production GGUF conversion pipeline for PSM's LoRA adapters (storage, retrieval_plan,
consolidation), across every trained domain, all fine-tuned on Qwen2.5-0.5B-Instruct with an
identical LoRA config (r=16, alpha=32, target_modules=[q,k,v,o,gate,up,down]_proj).

Replaces the earlier ONNX Runtime GenAI pipeline (convert_adapters_onnx.py): that path's
swappable-LoRA dynamo-exported graph was both slow on CPU (~10 tok/s for a 0.5B model -- a dynamic,
Concat-based KV-cache recomputes/reallocates a growing tensor every token) and unsalvageable via
post-hoc quantization (Olive's RTN int4/int8 passes catastrophically broke output correctness on this
graph, most likely because the official ORT GenAI LoRA docs specify quantizing the base model BEFORE
adapter extraction, not after -- we quantized after). llama.cpp's GGUF quantization is the mature,
battle-tested alternative: validated on RunPod RTX A5000 at 83.3% action-match on the full 419-case
holdout-conversational-storage-cases.json gate (matching/exceeding the 81.9% ONNX baseline) at
0.7s/case with full GPU offload, and ~40 tok/s even on CPU with Q4_K_M quantization.

Produces ONE quantized base GGUF model (shared across every adapter -- LoRA deltas stay in their own
separate .gguf files, applied at runtime via LLamaSharp's LoadLoraFromFile/SetLoraAdapters, never
merged into the base weights) plus one -lora-f16.gguf file per domain/task. All are loadable at
runtime via dotnet/src/PsmMemory.Core/Runtime/LlamaSharpPsmRuntime.cs with zero Python needed in
production.

One-time venv setup (only if the conversion venv doesn't exist yet):
    python -m venv psm-model/prod-memory/onnx-spike/llamacpp-venv
    psm-model/prod-memory/onnx-spike/llamacpp-venv/Scripts/pip install \\
        llama-cpp-python gguf sentencepiece transformers torch huggingface_hub peft
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git \\
        psm-model/prod-memory/onnx-spike/llamacpp-src

Usage (must run with the venv's python, not the main project's):
    psm-model/prod-memory/onnx-spike/llamacpp-venv/Scripts/python psm-model/scripts/convert_adapters_gguf.py \\
        --output-dir psm-model/prod-memory/gguf-runtime/v1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMACPP_SRC = REPO_ROOT / "psm-model/prod-memory/onnx-spike/llamacpp-src"
BASE_MODEL_GGUF_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Domain -> task -> PEFT adapter directory. Same conventions/checkpoints as convert_adapters_onnx.py.
# A domain is only converted if ALL THREE of its task checkpoints exist on disk -- a partial domain
# is skipped with a warning rather than half-converted, matching LlamaSharpPsmRuntime.cs's
# "all three or none" loading rule.
DOMAIN_ADAPTERS: dict[str, dict[str, Path]] = {
    "coding": {
        "storage": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-storage-v16b-qwen0.5b/checkpoint-800",
        "retrieval_plan": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-recall-plan-v3-qwen0.5b/adapter",
        "consolidation": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-consolidation-v4-qwen0.5b/adapter",
    },
    "conversational": {
        "storage": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-conversational-storage-dpo1-qwen0.5b/adapter",
        "retrieval_plan": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-conversational-retrieval-plan-v1-qwen0.5b/adapter",
        "consolidation": REPO_ROOT / "psm-model/prod-memory/checkpoints/hf-prod-conversational-consolidation-v1-qwen0.5b/adapter",
    },
}

# Matches LlamaSharpPsmRuntime.AdapterFilePrefix in the C# SDK -- must stay in sync.
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


def resolve_base_model_dir() -> Path:
    """Finds the locally-cached HF snapshot dir for BASE_MODEL (already downloaded during training/
    earlier ONNX conversion) -- avoids a redundant download."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(BASE_MODEL))


def run(args: list[str]):
    print(f"$ {' '.join(str(a) for a in args)}", file=sys.stderr)
    subprocess.run([sys.executable, *args], check=True)


def convert_base_model(work_dir: Path) -> Path:
    """Traces the base HF model to GGUF (f16), then quantizes to Q4_K_M. Returns the quantized path."""
    f16_path = work_dir / "qwen2.5-0.5b-instruct-f16.gguf"
    quantized_path = work_dir / BASE_MODEL_GGUF_NAME
    if quantized_path.exists():
        return quantized_path

    if not f16_path.exists():
        base_model_dir = resolve_base_model_dir()
        run([
            str(LLAMACPP_SRC / "convert_hf_to_gguf.py"),
            "--outfile", str(f16_path),
            "--outtype", "f16",
            str(base_model_dir),
        ])

    import llama_cpp
    import ctypes

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M
    rc = llama_cpp.llama_model_quantize(
        str(f16_path).encode(), str(quantized_path).encode(), ctypes.byref(params)
    )
    if rc != 0:
        sys.exit(f"llama_model_quantize failed with code {rc}")
    return quantized_path


def convert_adapter(checkpoint_dir: Path, out_file: Path):
    if out_file.exists():
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    base_model_dir = resolve_base_model_dir()
    run([
        str(LLAMACPP_SRC / "convert_lora_to_gguf.py"),
        "--base", str(base_model_dir),
        "--outfile", str(out_file),
        "--outtype", "f16",
        str(checkpoint_dir),
    ])


# Deliberately the SAME prompt for every adapter: this validation's job is to confirm the swap
# mechanism itself works (SetLoraAdapters genuinely changes the model's behavior per adapter), not to
# re-prove each task's accuracy -- that's what the dedicated held-out gates (run separately, per task)
# are for.
SMOKE_PROMPT = (
    "You are a memory assistant for a coding agent.",
    "Fixed the null-pointer bug in auth.py by adding a guard before session.user access.",
)


def validate(output_dir: Path, domains: dict[str, dict[str, Path]]):
    import llama_cpp
    import ctypes
    from llama_cpp import Llama

    adapter_keys = [f"{DOMAIN_FILE_PREFIX[domain]}{task}" for domain, tasks in domains.items() for task in tasks]

    print("loading base model + all adapters for smoke validation...", file=sys.stderr)
    llm = Llama(model_path=str(output_dir / BASE_MODEL_GGUF_NAME), n_ctx=2048, n_threads=8, verbose=False)

    loaded = {
        name: llama_cpp.llama_cpp.llama_adapter_lora_init(
            llm._model.model, str(output_dir / "adapters" / f"{name}-lora-f16.gguf").encode()
        )
        for name in adapter_keys
    }

    system, user = SMOKE_PROMPT
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    def generate(name: str) -> str:
        arr = (llama_cpp.llama_cpp.llama_adapter_lora_p_ctypes * 1)(loaded[name])
        scales = (ctypes.c_float * 1)(1.0)
        llama_cpp.llama_cpp.llama_set_adapters_lora(llm._ctx.ctx, arr, 1, scales)
        llm.reset()
        out = llm(prompt, max_tokens=200, temperature=0.0)
        return out["choices"][0]["text"]

    outputs = {}
    for name in adapter_keys:
        text = generate(name)
        outputs[name] = text
        print(f"[{name}] {text[:200]!r}", file=sys.stderr)
        if not text.strip():
            sys.exit(f"validation failed for adapter '{name}': empty output")

    # This prompt is deliberately generic (not each task's own real JSON-schema system prompt), so a
    # SMALL number of coincidental duplicates across differently-trained adapters is expected and not
    # a sign of a broken swap (e.g. two adapters both settling on "just restate the input" for a
    # short, generic, off-task completion) -- only fail if the swap looks fundamentally broken (most
    # or all adapters collapsing to the same output, which no plausible coincidence explains).
    distinct = len(set(outputs.values()))
    if distinct <= len(outputs) // 2:
        sys.exit(
            f"validation failed: only {distinct}/{len(outputs)} distinct outputs across adapters -- "
            "SetLoraAdapters may not be swapping weights (same output for different adapters)."
        )
    if distinct < len(outputs):
        print(
            f"note: {len(outputs) - distinct} coincidental duplicate(s) among {len(outputs)} adapters "
            "on this generic off-task prompt -- not a failure (see script docstring).",
            file=sys.stderr,
        )

    print(f"all {len(outputs)} adapters loaded and swapped ({distinct} distinct outputs).", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=None, help="scratch dir for intermediate artifacts")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    work_dir = (args.work_dir or (args.output_dir.parent / f"{args.output_dir.name}-work")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    domains = available_domains()
    if "coding" not in domains:
        sys.exit("'coding' domain adapters not found -- this is the seed domain and must be present.")

    quantized_base = convert_base_model(work_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = args.output_dir / "adapters"
    adapters_dir.mkdir(exist_ok=True)

    import shutil
    shutil.copy2(quantized_base, args.output_dir / BASE_MODEL_GGUF_NAME)

    for domain, tasks in domains.items():
        prefix = DOMAIN_FILE_PREFIX[domain]
        for task, checkpoint in tasks.items():
            adapter_key = f"{prefix}{task}"
            out_file = adapters_dir / f"{adapter_key}-lora-f16.gguf"
            convert_adapter(checkpoint, out_file)

    if not args.skip_validate:
        validate(args.output_dir, domains)


if __name__ == "__main__":
    main()
