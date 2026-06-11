from __future__ import annotations

import os
import sys


def allow_local_gpu() -> bool:
    """GPU is enabled on RunPod or when explicitly opted in (not on dev laptops by default)."""
    return os.environ.get("PSM_ALLOW_LOCAL_GPU") == "1" or os.environ.get("PSM_RUNPOD") == "1"


def resolve_device_name(
    device: str,
    cuda_available: bool,
    *,
    context: str = "psm_model",
) -> str:
    """Resolve a device request: GPU on RunPod when available, CPU on local machines."""
    requested = device.strip().lower()
    if requested in {"", "cpu"}:
        return "cpu"
    if os.environ.get("PSM_FORCE_CPU") == "1":
        if requested not in {"", "cpu"}:
            print(
                f"{context}: PSM_FORCE_CPU=1; using cpu instead of {device!r}",
                file=sys.stderr,
            )
        return "cpu"
    gpu_ok = cuda_available and allow_local_gpu()
    if requested in {"auto", "cuda", "gpu"}:
        if gpu_ok:
            return "cuda"
        if requested in {"cuda", "gpu"}:
            print(
                f"{context}: local GPU disabled (requested {device!r}); use RunPod "
                f"(PSM_RUNPOD=1) or set PSM_ALLOW_LOCAL_GPU=1.",
                file=sys.stderr,
            )
        return "cpu"
    if requested.startswith("cuda"):
        if gpu_ok:
            return device.strip()
        print(
            f"{context}: local GPU disabled (requested {device!r}); using cpu.",
            file=sys.stderr,
        )
        return "cpu"
    return device.strip()


def enforce_local_device_policy(device: str, *, context: str = "psm_model") -> str:
    """Resolve device using torch CUDA availability when importable."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    return resolve_device_name(device, cuda_available, context=context)
