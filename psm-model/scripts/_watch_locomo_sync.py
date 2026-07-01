#!/usr/bin/env python3
"""Pull LoCoMo ingest artifacts from RunPod on an interval."""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LABEL = "hf-prod-v5k-two-pass"
LIMIT_TAG = "full"
REMOTE_RESULTS = "/workspace/PSM/benchmark/locomo/results"
LOCAL_RESULTS = REPO / "benchmark/locomo/results"
LOCAL_SYNC = LOCAL_RESULTS / "pod-sync"
PROXY_USER = "cyveakf0qhvqih-644111e0"

def _ssh_bash(user: str, script: str, *, timeout_sec: int = 90) -> tuple[int, str]:
    proc = subprocess.run(
        [
            rc.SSH_BIN,
            "-tt",
            "-i",
            rc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            "22",
            f"{user}@ssh.runpod.io",
            "bash",
            "-s",
        ],
        input=f"{script}\nexit\n",
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout or ""


def _ssh_info(pod_id: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "runpod_ctl.py"), "ssh-info", pod_id],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("pod_host_id"):
            return str(payload["pod_host_id"])
        for target in payload.get("targets") or []:
            if isinstance(target, dict) and target.get("user"):
                return str(target["user"])
    return ""


def _ingest_progress(proxy_user: str) -> dict[str, int]:
    log = f"{REMOTE_RESULTS}/locomo-{LABEL}-n{LIMIT_TAG}.log"
    _, text = _ssh_bash(proxy_user, f"grep -E 'ingested [0-9]+' '{log}' 2>/dev/null | tail -1")
    seen = stored = ignored = failed = 0
    for line in text.splitlines():
        m = re.search(r"ingested (\d+) \| stored=(\d+) ignored=(\d+) failed=(\d+)", line)
        if m:
            seen, stored, ignored, failed = map(int, m.groups())
    return {"seen": seen, "stored": stored, "ignored": ignored, "failed": failed}


def _pull_file_b64(proxy_user: str, remote: str, local: Path, *, timeout_sec: int = 180) -> bool:
    script = f"""
if [[ -f '{remote}' ]]; then
  echo ___PSM_SYNC_START___
  base64 -w0 '{remote}'
  echo
  echo ___PSM_SYNC_END___
else
  echo ___PSM_SYNC_MISSING___
fi
"""
    _, text = _ssh_bash(proxy_user, script, timeout_sec=timeout_sec)
    if "___PSM_SYNC_START___" not in text:
        return False
    begin = text.rfind("___PSM_SYNC_START___")
    end = text.rfind("___PSM_SYNC_END___")
    if end <= begin:
        return False
    payload = re.sub(r"[^A-Za-z0-9+/=]", "", text[begin + len("___PSM_SYNC_START___") : end])
    pad = (-len(payload)) % 4
    payload += "=" * pad
    try:
        data = base64.b64decode(payload, validate=False)
    except Exception:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".syncing")
    tmp.write_bytes(data)
    if local.is_file() and local.stat().st_size > len(data):
        tmp.unlink(missing_ok=True)
        return False
    if local.exists():
        local.unlink()
    tmp.replace(local)
    return True


def sync_once(pod_id: str, proxy_user: str) -> dict[str, object]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOCAL_SYNC.mkdir(parents=True, exist_ok=True)
    names = [
        f"locomo-{LABEL}-n{LIMIT_TAG}.log",
        f"locomo-{LABEL}-n{LIMIT_TAG}.db",
        f"locomo-{LABEL}-n{LIMIT_TAG}-results.json",
        "ingest-psm-model-summary.json",
    ]
    pulled: dict[str, int] = {}
    for name in names:
        local = LOCAL_SYNC / name
        timeout = 300 if name.endswith(".db") else 120
        if _pull_file_b64(proxy_user, f"{REMOTE_RESULTS}/{name}", local, timeout_sec=timeout):
            pulled[name] = local.stat().st_size
    progress = _ingest_progress(proxy_user)
    payload = {"ts": ts, "pod_id": pod_id, "pulled": pulled, "progress": progress}
    print(json.dumps(payload), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="cyveakf0qhvqih")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    proxy_user = args.proxy_user.strip() or _ssh_info(args.pod_id)
    if not proxy_user:
        print("proxy-user required", file=sys.stderr)
        return 1

    LOCAL_SYNC.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            sync_once(args.pod_id, proxy_user)
        except Exception as exc:
            print(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "error": str(exc)}), flush=True)
        if args.once:
            return 0
        time.sleep(max(args.interval_sec, 60))


if __name__ == "__main__":
    raise SystemExit(main())
