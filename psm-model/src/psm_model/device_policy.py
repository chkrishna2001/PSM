from __future__ import annotations

import os
import sys


def allow_local_gpu() -> bool:
    return os.environ.get("PSM_ALLOW_LOCAL_GPU") == "1" or os.environ.get("PSM_RUNPOD") == "1"


def enforce_local_device_policy(device: str, *, context: str = "psm_model") -> str:
    """Force CPU locally unless PSM_RUNPOD=1 or PSM_ALLOW_LOCAL_GPU=1 is set."""
    requested = device.lower()
    if requested in {"cpu", ""}:
        return "cpu"
    if allow_local_gpu():
        return device
    if requested not in {"cuda", "auto", "gpu"}:
        return device
    print(
        f"{context}: local GPU disabled (requested {device!r}); use RunPod for cuda training/eval "
        f"or set PSM_ALLOW_LOCAL_GPU=1 to override.",
        file=sys.stderr,
    )
    return "cpu"
