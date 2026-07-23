FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Pinned to match psm-model/scripts/runpod_hf_lora_train.sh -- unpinned transformers on a fresh pod
# pulls a version requiring torch.distributed.tensor.DTensor, incompatible with this image's torch
# 2.4.0, breaking every training launch (see feedback_teacher_provider_order.md/session history).
RUN pip install --no-cache-dir \
    "transformers==4.46.3" "peft==0.13.2" "accelerate==1.0.1" \
    datasets huggingface_hub bitsandbytes hf_transfer

# Node.js 20.x -- needed by the LoCoMo TS benchmark harness (benchmark/locomo). Not present on the
# stock PyTorch image at all; installing it from scratch each pod session was the single biggest
# chunk of the ~20min stock-pod setup cost this image is meant to eliminate.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# `dotnet` shim: PsmMemory.Cli is deployed as a self-contained linux-x64 publish (no .NET runtime
# needed on the pod), but the LoCoMo TS driver hardcodes spawn("dotnet", [dllPath, ...args]) --
# redirect that to the matching native executable instead of installing the full SDK.
RUN printf '#!/bin/bash\ndll="$1"\nshift\nexec "${dll%%.dll}" "$@"\n' > /usr/local/bin/dotnet \
    && chmod +x /usr/local/bin/dotnet

# tmux/git: used throughout runpod_ctl.py's remote session management.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends tmux git \
    && rm -rf /var/lib/apt/lists/*

# Deliberately NOT baking in the GGUF base model / LoRA adapters here -- those are downloaded from
# HF at container start (LlamaSharpPsmRuntime.CreateAsync / LlamaSharpEmbeddingRuntime.CreateAsync
# already do this automatically), so this image stays reusable across training rounds without a
# rebuild every time a checkpoint changes, and has no proprietary content -- safe to publish public.

WORKDIR /workspace
