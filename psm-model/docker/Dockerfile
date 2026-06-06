# PSM 50M training image for RunPod (linux/amd64).
# Build from repo root:
#   docker build --platform linux/amd64 -f psm-model/docker/Dockerfile -t chkrishna2001/psm-50m-train:latest .
#   docker push chkrishna2001/psm-50m-train:latest

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir huggingface_hub numpy hf_transfer

WORKDIR /workspace/PSM

# Bake repo code (no checkpoints or large data).
COPY psm-model/src /workspace/PSM/psm-model/src
COPY psm-model/scripts /workspace/PSM/psm-model/scripts
COPY psm-model/pyproject.toml /workspace/PSM/psm-model/pyproject.toml
COPY docs/psm-model /workspace/PSM/docs/psm-model

ENV PYTHONPATH=/workspace/PSM/psm-model/src
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PSM_REPO_ROOT=/workspace/PSM
ENV PSM_HF_MODEL_REPO=chkrishna2001/psm-50m-mixed-v1-run
ENV PSM_HF_DATASET_REPO=chkrishna2001/psm-50m-action-mixed-v1

RUN chmod +x /workspace/PSM/psm-model/scripts/runpod_entrypoint.sh \
    && chmod +x /workspace/PSM/psm-model/scripts/runpod_bootstrap.sh

ENTRYPOINT ["/workspace/PSM/psm-model/scripts/runpod_entrypoint.sh"]
CMD ["sleep", "infinity"]
