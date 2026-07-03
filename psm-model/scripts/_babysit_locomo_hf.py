#!/usr/bin/env python3
"""Monitor LoCoMo HF ingest on RunPod; finalize when done (pull, HF upload, delete pod)."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402
from _watch_locomo_sync import sync_once  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
LABEL = "hf-prod-v5k-two-pass"
LIMIT_TAG = "full"
TOTAL_TURNS = 5882
OFFSET = 2968  # full run after smoke
REMOTE_LOG = f"/workspace/PSM/benchmark/locomo/results/locomo-{LABEL}-n{LIMIT_TAG}.log"
REMOTE_RESULTS = f"/workspace/PSM/benchmark/locomo/results/locomo-{LABEL}-n{LIMIT_TAG}-results.json"
STATE_PATH = REPO / "benchmark/locomo/results/pod-sync/babysit-state.json"
LOCAL_SYNC = REPO / "benchmark/locomo/results/pod-sync"

HF_UPLOADS = {
    f"locomo-{LABEL}-n{LIMIT_TAG}-results.json": f"eval/locomo-{LABEL}-n{LIMIT_TAG}-results.json",
    f"locomo-{LABEL}-n{LIMIT_TAG}.log": f"locomo/locomo-{LABEL}-n{LIMIT_TAG}.log",
    f"locomo-{LABEL}-n{LIMIT_TAG}.db": f"locomo/locomo-{LABEL}-n{LIMIT_TAG}.db",
    "ingest-psm-model-summary.json": "locomo/ingest-psm-model-summary.json",
}


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    subprocess.run(["o", "krishnachhftoken"], check=False, capture_output=True)
    if os.name == "nt":
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-Clipboard -Raw).Trim()"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    return ""


def _ssh_bash(proxy_user: str, script: str, *, timeout_sec: int = 90) -> str:
    proc = subprocess.run(
        [
            rc.SSH_BIN,
            "-tt",
            "-i",
            rc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-p",
            "22",
            f"{proxy_user}@ssh.runpod.io",
            "bash",
            "-s",
        ],
        input=f"{script}\nexit\n",
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        errors="replace",
    )
    return proc.stdout or ""


def _pod_status(proxy_user: str) -> dict[str, object]:
    text = _ssh_bash(
        proxy_user,
        f"""
grep -E 'ingested [0-9]+' '{REMOTE_LOG}' 2>/dev/null | tail -1 || true
grep '=== LoCoMo done' '{REMOTE_LOG}' 2>/dev/null | tail -1 || true
test -f '{REMOTE_RESULTS}' && echo RESULTS_OK || echo RESULTS_MISSING
pgrep -af 'ingest-cli|hf_remember|runpod_locomo' 2>/dev/null | grep -v grep | head -2 || echo PROC_NONE
cat /tmp/psm-locomo.done 2>/dev/null || echo DONE_MISSING
""",
    )
    seen = stored = ignored = failed = 0
    for line in text.splitlines():
        m = re.search(r"ingested (\d+) \| stored=(\d+) ignored=(\d+) failed=(\d+)", line)
        if m:
            seen, stored, ignored, failed = map(int, m.groups())
    done_line = next(
        (ln.strip() for ln in text.splitlines() if re.match(r"^=== LoCoMo done ", ln)),
        "",
    )
    ingest_done = bool(done_line)
    eval_ok = "eval=0" in done_line if done_line else False
    results_ok = "RESULTS_OK" in text
    proc_running = ("ingest-cli" in text or "hf_remember" in text or "runpod_locomo" in text) and (
        "PROC_NONE" not in text or "ingest-cli" in text
    )
    remaining = max(TOTAL_TURNS - OFFSET - seen, 0)
    return {
        "seen": seen,
        "stored": stored,
        "ignored": ignored,
        "failed": failed,
        "remaining": remaining,
        "ingest_done": ingest_done,
        "eval_ok": eval_ok,
        "results_ok": results_ok,
        "proc_running": proc_running,
        "done_line": done_line,
    }


def _upload_hf(token: str) -> tuple[bool, list[str]]:
    uploaded: list[str] = []
    errors: list[str] = []
    for local_name, repo_path in HF_UPLOADS.items():
        local = LOCAL_SYNC / local_name
        if not local.is_file():
            if local_name.endswith("-results.json"):
                errors.append(f"missing local {local_name}")
            continue
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; from huggingface_hub import HfApi; "
                    "api=HfApi(token=os.environ['HF_TOKEN']); "
                    f"api.upload_file(path_in_repo={repo_path!r}, "
                    f"path_or_fileobj={str(local)!r}, repo_id={MODEL_REPO!r}, repo_type='model'); "
                    "print('ok')"
                ),
            ],
            cwd=REPO,
            env={**os.environ, "HF_TOKEN": token},
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            uploaded.append(repo_path)
        else:
            errors.append(f"{repo_path}: {proc.stderr.strip() or proc.stdout.strip()}")
    return len(errors) == 0 or (bool(uploaded) and not any("results.json" in e for e in errors)), uploaded + errors


def _verify_hf(token: str) -> bool:
    required = [
        HF_UPLOADS[f"locomo-{LABEL}-n{LIMIT_TAG}-results.json"],
        HF_UPLOADS[f"locomo-{LABEL}-n{LIMIT_TAG}.db"],
    ]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; from huggingface_hub import HfApi; "
                "api=HfApi(token=os.environ['HF_TOKEN']); "
                f"files=set(api.list_repo_files({MODEL_REPO!r}, repo_type='model')); "
                f"need={required!r}; "
                "missing=[p for p in need if p not in files]; "
                "print('missing', missing); raise SystemExit(1 if missing else 0)"
            ),
        ],
        cwd=REPO,
        env={**os.environ, "HF_TOKEN": token},
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _delete_pod(pod_id: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO / "psm-model/scripts/runpod_ctl.py"),
            "delete-pod",
            pod_id,
            "--force-delete-pod",
        ],
        cwd=REPO,
        check=False,
    )


def _save_state(payload: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _finalize(pod_id: str, proxy_user: str, token: str, status: dict[str, object]) -> int:
    print(json.dumps({"event": "finalize_start", "status": status}), flush=True)
    for _ in range(3):
        sync_once(pod_id, proxy_user)
        time.sleep(10)
    results = LOCAL_SYNC / f"locomo-{LABEL}-n{LIMIT_TAG}-results.json"
    db = LOCAL_SYNC / f"locomo-{LABEL}-n{LIMIT_TAG}.db"
    if not results.is_file():
        print(json.dumps({"event": "finalize_fail", "reason": "missing results.json"}), flush=True)
        return 1
    if not db.is_file():
        print(json.dumps({"event": "finalize_fail", "reason": "missing db"}), flush=True)
        return 1
    ok, detail = _upload_hf(token)
    if not ok:
        print(json.dumps({"event": "hf_upload_fail", "detail": detail}), flush=True)
        return 1
    if not _verify_hf(token):
        print(json.dumps({"event": "hf_verify_fail"}), flush=True)
        return 1
    _delete_pod(pod_id)
    payload = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "pod_id": pod_id,
        "status": status,
        "hf_uploads": HF_UPLOADS,
        "deleted_pod": True,
    }
    _save_state(payload)
    print(json.dumps({"event": "babysit_complete", **payload}), flush=True)
    return 0


def tick(pod_id: str, proxy_user: str, token: str) -> str:
    """One check cycle. Returns: running | done | failed | idle."""
    status = _pod_status(proxy_user)
    sync = sync_once(pod_id, proxy_user)
    progress = sync.get("progress") or {}
    failed = int(status.get("failed") or progress.get("failed") or 0)
    if failed > 0:
        print(json.dumps({"event": "ingest_failures", "failed": failed, "status": status}), flush=True)
        return "failed"

    expected = TOTAL_TURNS - OFFSET
    seen = int(status.get("seen") or progress.get("seen") or 0)

    if status.get("ingest_done") and status.get("eval_ok") and status.get("results_ok"):
        return "done"

    if status.get("ingest_done") and not status.get("eval_ok"):
        print(json.dumps({"event": "eval_failed", "done_line": status.get("done_line")}), flush=True)
        return "failed"

    if not status.get("proc_running") and seen > 0 and seen < expected and not status.get("ingest_done"):
        print(json.dumps({"event": "proc_stopped_early", "status": status}), flush=True)
        return "failed"

    remaining = max(expected - seen, 0)
    print(
        json.dumps(
            {
                "event": "babysit_tick",
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pod_id": pod_id,
                "seen": seen,
                "remaining": remaining,
                "stored": status.get("stored"),
                "failed": failed,
                "proc_running": status.get("proc_running"),
            }
        ),
        flush=True,
    )
    return "running"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="9e7w7yd0624dps")
    parser.add_argument("--proxy-user", default="9e7w7yd0624dps-644111de")
    parser.add_argument("--interval-sec", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    token = _hf_token()
    if not token:
        print("HF_TOKEN required", file=sys.stderr)
        return 1

    while True:
        try:
            state = tick(args.pod_id, args.proxy_user, token)
        except Exception as exc:
            print(json.dumps({"event": "tick_error", "error": str(exc)}), flush=True)
            state = "running"
        if state == "done":
            status = _pod_status(args.proxy_user)
            return _finalize(args.pod_id, args.proxy_user, token, status)
        if state == "failed":
            _save_state(
                {
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "pod_id": args.pod_id,
                    "note": "babysit stopped — pod left running for manual fix",
                }
            )
            return 1
        if args.once:
            return 0
        time.sleep(max(args.interval_sec, 60))


if __name__ == "__main__":
    raise SystemExit(main())
