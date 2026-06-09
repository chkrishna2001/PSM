#!/usr/bin/env python3
"""Push scripts + micro artifacts, start gate4 micro train on warm pod, verify GPU."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as ctl  # noqa: E402

PROXY = "6c9efizq1aoocf-64411022@ssh.runpod.io"
HOST_ALIAS = ctl.SSH_CONFIG_HOST
POD_ID = "6c9efizq1aoocf"


def main() -> int:
    hf = os.environ.get("HF_TOKEN", "").strip()
    if not hf:
        print("HF_TOKEN not set — run: o chinnahftoken", file=sys.stderr)
        return 1

    ssh_host = "ssh.runpod.io"
    ssh_port = "22"
    ssh_user = PROXY.split("@", 1)[0]

    scripts = REPO / "psm-model" / "scripts"
    print(f"Pushing scripts -> pod:/workspace/PSM/psm-model/scripts", flush=True)
    rc = ctl._ssh_push_dir(HOST_ALIAS, scripts, "/workspace/PSM/psm-model/scripts", host=ssh_host, port=ssh_port, user=ssh_user)
    if rc != 0:
        print(f"scripts push failed (exit {rc})", file=sys.stderr)
        return rc

    import tempfile
    import tarfile

    artifact_rels = [
        "psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042000.json",
        "psm-model/data/curriculum/gate4-parse-repair-step-042000.jsonl",
        "psm-model/data/curriculum/psm-50m-gate4-train-micro.jsonl",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "micro-artifacts"
        bundle.mkdir()
        for rel in artifact_rels:
            local = REPO / rel
            if not local.is_file():
                print(f"skip missing {local}", file=sys.stderr)
                continue
            dest = bundle / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(local.read_bytes())
            print(f"Bundling {rel}", flush=True)
        print("Pushing micro artifacts via tar", flush=True)
        prc = ctl._ssh_push_dir(
            HOST_ALIAS,
            bundle,
            "/workspace/PSM",
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
        if prc != 0:
            print(f"artifact push failed (exit {prc})", file=sys.stderr)
            return prc

    script = REPO / "psm-model" / "scripts" / "runpod_start_gate4_train_only.sh"
    extra_env = {
        "HF_TOKEN": hf,
        "PSM_HF_MODEL_REPO": os.environ.get("PSM_HF_MODEL_REPO", ctl.DEFAULT_HF_MODEL_REPO),
        "GATE4_CURRICULUM_BUILDER": "micro",
        "RESUME_CHECKPOINT": "psm-model/checkpoints/real-v3-50m-full-v2-step-042000.pt",
        "TOKENIZER": "psm-model/checkpoints/real-v3-50m-full-v2-step-042000.tokenizer.json",
        "TARGET_STEPS": "43500",
        "STRUCTURAL_LOSS_WEIGHT": "8",
        "EVAL_EVERY": "200",
        "SAVE_EVERY": "200",
        "REPAIR_COPIES": "12",
        "DIRECT_COPIES": "20",
        "GATE4_EVAL_REPORT": "psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042000.json",
        "GATE4_PARSE_REPAIR": "psm-model/data/curriculum/gate4-parse-repair-step-042000.jsonl",
    }
    print("Starting micro train tmux on pod...", flush=True)
    rc = ctl._ssh_run_script(
        HOST_ALIAS,
        script,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=300,
        extra_env=extra_env,
        skip_ssh_wait=True,
    )
    print(f"start script exit {rc}", flush=True)
    return rc if isinstance(rc, int) else rc[0]


if __name__ == "__main__":
    raise SystemExit(main())
