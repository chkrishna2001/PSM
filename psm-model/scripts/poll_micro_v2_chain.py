#!/usr/bin/env python3
"""Poll micro v2 train until 42800, launch expanded eval, pull report."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as ctl  # noqa: E402

PROXY_USER = "6c9efizq1aoocf-64411022"
SSH_HOST = "ssh.runpod.io"
TARGET_STEP = 42800
CKPT_REL = f"psm-model/checkpoints/real-v3-50m-full-v2-step-{TARGET_STEP:06d}.pt"

def _poll_stdin(target_step: int) -> str:
    ckpt = f"/workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-step-{target_step:06d}.pt"
    return f"""\
test -f '{ckpt}' && echo TARGET_CKPT_OK || echo TARGET_CKPT_MISSING
test -f /tmp/psm-gate4.done && echo TRAIN_DONE || echo TRAIN_RUNNING
pgrep -af 'python3 -m psm_model.train' >/dev/null && echo TRAIN_PROC || echo NO_TRAIN_PROC
grep -oE '"step": [0-9]+' /tmp/psm-gate4-train.log 2>/dev/null | tail -1 || true
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
exit
"""


def _ssh_poll(target_step: int) -> dict[str, str]:
    proc = subprocess.run(
        [
            ctl.SSH_BIN,
            "-tt",
            "-i",
            ctl.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            f"{PROXY_USER}@{SSH_HOST}",
            "bash",
            "-s",
        ],
        input=_poll_stdin(target_step),
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="replace",
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        elif line in ("TARGET_CKPT_OK", "TARGET_CKPT_MISSING"):
            out["target_ckpt"] = line
        elif line.startswith('"step":'):
            out["log_step"] = line
        elif "%" in line and "MiB" in line:
            out["gpu"] = line
        elif line in {
            "TRAIN_DONE",
            "TRAIN_RUNNING",
            "TRAIN_PROC",
            "NO_TRAIN_PROC",
        }:
            out[line.lower()] = "1"
    return out


def _launch_eval() -> int:
    script = REPO / "psm-model" / "scripts" / "runpod_start_gate4_eval_only.sh"
    extra_env = {
        "PSM_EVAL_DEVICE": "cuda",
        "PSM_EVAL_FULL_CKPT": CKPT_REL,
    }
    ctl._ssh_push_dir(
        ctl.SSH_CONFIG_HOST,
        REPO / "psm-model" / "scripts",
        "/workspace/PSM/psm-model/scripts",
        host=SSH_HOST,
        port="22",
        user=PROXY_USER,
    )
    rc = ctl._ssh_run_script(
        ctl.SSH_CONFIG_HOST,
        script,
        host=SSH_HOST,
        port="22",
        user=PROXY_USER,
        timeout_sec=120,
        extra_env=extra_env,
        skip_ssh_wait=True,
    )
    return rc if isinstance(rc, int) else rc[0]


def _pull_eval_report() -> Path | None:
    remote = f"/workspace/PSM/psm-model/checkpoints/gate-eval/gate4-full-expanded-step-{TARGET_STEP:06d}.json"
    local = REPO / "psm-model" / "checkpoints" / "gate-eval" / f"gate4-full-expanded-step-{TARGET_STEP:06d}.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    stdin = f"cat '{remote}' 2>/dev/null || exit 1\nexit\n"
    proc = subprocess.run(
        [
            ctl.SSH_BIN,
            "-tt",
            "-i",
            ctl.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            f"{PROXY_USER}@{SSH_HOST}",
            "bash",
            "-s",
        ],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return None
    raw = proc.stdout
    idx = raw.find("{")
    if idx < 0:
        return None
    local.write_text(raw[idx:], encoding="utf-8")
    return local


def _parse_metrics(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    idx = raw.find("{")
    data = json.loads(raw[idx:])
    summary = data.get("summary") or data
    return {
        "parse_schema_pct": summary.get("parse_schema_pct") or summary.get("parse_pct"),
        "action_pct": summary.get("action_pct"),
        "parse_fails": summary.get("parse_fails") or summary.get("parse_fail_count"),
        "pass": summary.get("pass") or summary.get("gate_pass"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--max-train-polls", type=int, default=120)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    target_tag = f"step-{TARGET_STEP:06d}"
    print(f"Polling train until {target_tag}...", flush=True)
    for i in range(args.max_train_polls):
        st = _ssh_poll(TARGET_STEP)
        ckpt = st.get("target_ckpt", "TARGET_CKPT_MISSING")
        gpu = st.get("gpu", "?")
        log_step = st.get("log_step", "")
        done = st.get("train_done") == "1"
        no_proc = st.get("no_train_proc") == "1"
        print(f"[{i+1}] {ckpt} {log_step} gpu={gpu} done={done} no_proc={no_proc}", flush=True)
        if ckpt == "TARGET_CKPT_OK" and (done or no_proc):
            print("TRAIN_COMPLETE", flush=True)
            break
        time.sleep(args.poll_sec)
    else:
        print("TRAIN_TIMEOUT", flush=True)
        return 1

    if args.skip_eval:
        return 0

    print("Launching expanded eval...", flush=True)
    if _launch_eval() != 0:
        return 1

    eval_poll = """\
test -f /tmp/psm-gate4-eval.done && echo EVAL_DONE || echo EVAL_RUNNING
pgrep -af psm_model.eval_checkpoint | head -1 || echo NO_EVAL
exit
"""
    for i in range(90):
        proc = subprocess.run(
            [
                ctl.SSH_BIN,
                "-tt",
                "-i",
                ctl.SSH_KEY_PATH,
                "-o",
                "ConnectTimeout=20",
                f"{PROXY_USER}@{SSH_HOST}",
                "bash",
                "-s",
            ],
            input=eval_poll,
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        if "EVAL_DONE" in proc.stdout:
            print("EVAL_COMPLETE", flush=True)
            break
        time.sleep(120)
    else:
        print("EVAL_TIMEOUT", flush=True)
        return 1

    path = _pull_eval_report()
    if not path:
        print("Could not pull eval report", file=sys.stderr)
        return 1
    metrics = _parse_metrics(path)
    print(json.dumps({"report": str(path), **metrics}, indent=2), flush=True)
    parse_pct = float(metrics.get("parse_schema_pct") or 0)
    if parse_pct >= 95:
        print("GATE4_PARSE_PASS", flush=True)
        return 0
    print("GATE4_PARSE_FAIL — need micro v3", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
