from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_REPO = "subbu83/psm-50m-mixed-v1-run"
DEFAULT_DATASET_REPO = "chkrishna2001/psm-50m-action-mixed-v1"
DEFAULT_CURRICULUM_REPO = DEFAULT_DATASET_REPO

RESUME_STEP = 58_000
RUN_STEM = "real-v3-50m-full-v2"
SMOKE_TARGET_STEPS = RESUME_STEP + 2_000

CHECKPOINT_REL = f"psm-model/checkpoints/{RUN_STEM}-step-{RESUME_STEP:06d}.pt"
TOKENIZER_REL = f"psm-model/checkpoints/{RUN_STEM}-step-{RESUME_STEP:06d}.tokenizer.json"
META_REL = f"psm-model/checkpoints/{RUN_STEM}-step-{RESUME_STEP:06d}.meta.json"

CURRICULUM_REL = "prod-memory/prod-extraction-v2.jsonl"
MANIFEST_REL = "prod-memory/prod-extraction-v2.manifest.json"

EXPANDED_PROBE_REL = "data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl"
DIRECT_PROBE_REL = "data/probes/direct_probes.jsonl"

LOCAL_CURRICULUM = PACKAGE_ROOT / "data" / "prod-extraction-v2.jsonl"
LOCAL_MANIFEST = PACKAGE_ROOT / "data" / "prod-extraction-v2.manifest.json"
