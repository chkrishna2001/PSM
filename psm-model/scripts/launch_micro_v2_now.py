#!/usr/bin/env python3
"""Micro v2: resume 42k, target 42800, pre-built curriculum on warm pod."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as ctl  # noqa: E402

PROXY_USER = "6c9efizq1aoocf-64411022"
SSH_HOST = "ssh.runpod.io"
SSH_PORT = "22"


def main() -> int:
    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("HF_TOKEN not set", file=sys.stderr)
        return 1

    scripts = REPO / "psm-model" / "scripts"
    ctl._ssh_push_dir(
        ctl.SSH_CONFIG_HOST, scripts, "/workspace/PSM/psm-model/scripts",
        host=SSH_HOST, port=SSH_PORT, user=PROXY_USER,
    )

    hf_repo = os.environ.get("PSM_HF_MODEL_REPO", ctl.DEFAULT_HF_MODEL_REPO)
    fetch = REPO / "psm-model" / "scripts" / "pod_hf_fetch_42000.sh"
    print("Ensuring step-042000 on pod (HF fetch if missing)...", flush=True)
    ctl._ssh_run_script(
        ctl.SSH_CONFIG_HOST, fetch,
        host=SSH_HOST, port=SSH_PORT, user=PROXY_USER,
        timeout_sec=300,
        extra_env={"HF_TOKEN": hf, "PSM_HF_MODEL_REPO": hf_repo},
        skip_ssh_wait=True,
    )

    artifact_rels = [
        "psm-model/data/curriculum/psm-50m-gate4-train-micro-v2.jsonl",
        "psm-model/data/curriculum/gate4-parse-repair-step-043400.jsonl",
        "psm-model/checkpoints/gate-eval/gate4-full-expanded-step-043400.json",
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        for rel in artifact_rels:
            local = REPO / rel
            if local.is_file():
                dest = bundle / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(local.read_bytes())
        ctl._ssh_push_dir(
            ctl.SSH_CONFIG_HOST, bundle, "/workspace/PSM",
            host=SSH_HOST, port=SSH_PORT, user=PROXY_USER,
        )

    script = REPO / "psm-model" / "scripts" / "runpod_start_gate4_train_only.sh"
    extra_env = {
        "HF_TOKEN": hf,
        "PSM_HF_MODEL_REPO": os.environ.get("PSM_HF_MODEL_REPO", ctl.DEFAULT_HF_MODEL_REPO),
        "GATE4_CURRICULUM_BUILDER": "micro",
        "SKIP_CURRICULUM_BUILD": "1",
        "RESUME_CHECKPOINT": "psm-model/checkpoints/real-v3-50m-full-v2-step-042000.pt",
        "TOKENIZER": "psm-model/checkpoints/real-v3-50m-full-v2-step-042000.tokenizer.json",
        "TARGET_STEPS": "42800",
        "GATE4_CURRICULUM": "psm-model/data/curriculum/psm-50m-gate4-train-micro-v2.jsonl",
        "STRUCTURAL_LOSS_WEIGHT": "4",
        "PROMOTE_SPAN_WEIGHT": "4",
        "EVAL_EVERY": "200",
        "SAVE_EVERY": "200",
        "GATE4_PINNED_STEPS": "42000",
    }
    print("Starting micro v2 train...", flush=True)
    rc = ctl._ssh_run_script(
        ctl.SSH_CONFIG_HOST, script,
        host=SSH_HOST, port=SSH_PORT, user=PROXY_USER,
        timeout_sec=180, extra_env=extra_env, skip_ssh_wait=True,
    )
    if rc != 0:
        return rc if isinstance(rc, int) else rc[0]

    import time
    time.sleep(45)
    ok = ctl._verify_pod_job(
        ctl.SSH_CONFIG_HOST, host=SSH_HOST, port=SSH_PORT, user=PROXY_USER,
        tmux_session="psm-gate4", process_pattern="psm_model.train", label="micro-v2",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
