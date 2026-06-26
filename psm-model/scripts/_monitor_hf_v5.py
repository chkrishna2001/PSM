#!/usr/bin/env python3
"""Poll v5 HF LoRA train until done; run eval pull if watcher missed."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"
POD = "pcnied8ov3f3rc"
PROXY = "pcnied8ov3f3rc-6441135a"
EVAL_OUT = REPO / "psm-model/prod-memory/results/hf-prod-v5-qwen0.5b-prod-grounding.json"
INTERVAL = 300  # 5 min


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)


def verify() -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "runpod_ctl.py"),
            "verify-pod",
            "--pod-id",
            POD,
            "--proxy-user",
            PROXY,
            "--train-log",
            "/tmp/psm-hf-lora-train.log",
            "--tmux-session",
            "psm-hf-lora",
            "--process-pattern",
            "hf_lora_train",
            "--timeout-sec",
            "90",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if line.startswith("{") and '"passed"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"exit_code": proc.returncode}


def probe_done() -> dict[str, str]:
    sys.path.insert(0, str(SCRIPTS))
    import argparse
    import runpod_ctl as rc

    ns = argparse.Namespace(
        pod_id=POD, proxy_user=PROXY, deploy=False, host_alias="runpod-psm-proxy",
        name="", image="", template="", gpu="", volume_gb=0, container_disk_gb=0,
        autostart=False, wait_ssh=0, ssh_ready_timeout_sec=300, auto_gpu=False,
    )
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=PROXY)
    probe = """
test -f /tmp/psm-hf-lora.done && echo PSM_DONE=1 || echo PSM_DONE=0
test -f /workspace/PSM/psm-model/prod-memory/checkpoints/hf-prod-v5-qwen0.5b/adapter/adapter_model.safetensors && echo PSM_ADPT=1 || echo PSM_ADPT=0
tail -5 /tmp/psm-hf-lora-train.log 2>/dev/null | while IFS= read -r line; do echo "PSM_LOG=$line"; done
"""
    proc = subprocess.run(
        [rc.SSH_BIN, "-tt", "-i", rc.SSH_KEY_PATH, "-o", "ConnectTimeout=20",
         *rc._ssh_endpoint("runpod-psm-proxy", host=host, port=port, user=user),
         "bash", "-s"],
        input=f"{probe}exit\n",
        capture_output=True, text=True, timeout=60, encoding="utf-8", errors="replace",
    )
    out: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if line.startswith("PSM_DONE="):
            out["done"] = line.split("=", 1)[1].strip()
        elif line.startswith("PSM_ADPT="):
            out["adapter"] = line.split("=", 1)[1].strip()
        elif line.startswith("PSM_LOG="):
            out.setdefault("log_tail", "")
            out["log_tail"] = (out["log_tail"] + line[7:] + " | ").strip(" |")
    return out


def finish_eval() -> int:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "_run_hf_lora_eval.py"),
         "--pod-id", POD, "--proxy-user", PROXY, "--profile", "v5"],
        cwd=REPO,
    ).returncode


def read_aggregate() -> dict | None:
    if not EVAL_OUT.is_file():
        return None
    data = json.loads(EVAL_OUT.read_text(encoding="utf-8"))
    return data.get("aggregate")


def main() -> int:
    log(f"monitoring pod {POD} every {INTERVAL}s")
    while True:
        if EVAL_OUT.is_file():
            agg = read_aggregate()
            log(f"DONE local eval exists: {EVAL_OUT}")
            if agg:
                log(f"aggregate: {json.dumps(agg)}")
            return 0

        status = verify()
        try:
            probe = probe_done()
        except subprocess.TimeoutExpired:
            probe = {"done": "?", "adapter": "?", "log_tail": "ssh probe timeout"}
        log(
            json.dumps({
                "state": status.get("job_state"),
                "gpu": status.get("gpu_util_pct"),
                "passed": status.get("passed"),
                "done": probe.get("done"),
                "adapter": probe.get("adapter"),
                "log_tail": (probe.get("log_tail") or status.get("train_log_tail", ""))[-200:],
            })
        )

        if probe.get("done") == "1" or (
            probe.get("adapter") == "1"
            and status.get("job_state") in {"train_finished", "idle_billing", "stopped", None}
            and int(status.get("gpu_util_pct") or 0) < 5
        ):
            log("train finished — running eval + pull")
            rc = finish_eval()
            agg = read_aggregate()
            log(f"eval exit={rc} aggregate={json.dumps(agg) if agg else 'missing'}")
            subprocess.run(
                [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "stop-pod", POD],
                cwd=REPO, check=False,
            )
            return 0 if rc == 0 and agg else 1

        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
