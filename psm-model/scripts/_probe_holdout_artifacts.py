#!/usr/bin/env python3
"""Probe holdout gate artifacts on a RunPod pod."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

PROBE = r"""
echo PSM_PROBE_OK
echo ---GATES---
ls -la /workspace/PSM/benchmark/locomo/results/holdout-gate*.json 2>/dev/null || echo NO_GATE_JSON
echo ---DBS---
ls -la /workspace/PSM/benchmark/locomo/results/holdout-gate*.db 2>/dev/null || echo NO_DBS
echo ---LOGS---
ls -la /workspace/PSM/benchmark/locomo/results/holdout-gate*.log 2>/dev/null | head -20 || echo NO_LOGS
echo ---MATRIX---
tail -30 /tmp/psm-holdout-gate-matrix.log 2>/dev/null || echo NO_MATRIX_LOG
echo ---SNAPSHOT---
cat /workspace/PSM/benchmark/locomo/results/holdout-gate-progress.snapshot.json 2>/dev/null || echo NO_SNAPSHOT
echo ---INGEST---
ls -1 /workspace/PSM/benchmark/locomo/results/holdout-gate-*-ingest-summary.json 2>/dev/null || echo NO_INGEST
echo ---RETRIEVAL---
ls -1 /workspace/PSM/benchmark/locomo/results/holdout-gate-*-retrieval.json 2>/dev/null || echo NO_RETRIEVAL
echo ---ANSWER---
ls -1 /workspace/PSM/benchmark/locomo/results/holdout-gate-*-answer.json 2>/dev/null || echo NO_ANSWER
echo ---INGEST v5n-dpo---
cat /workspace/PSM/benchmark/locomo/results/holdout-gate-v5n-dpo-conv-30-conv-41-ingest-summary.json 2>/dev/null || true
echo ---INGEST v5n---
cat /workspace/PSM/benchmark/locomo/results/holdout-gate-v5n-conv-30-conv-41-ingest-summary.json 2>/dev/null || true
echo ---INGEST v5h---
cat /workspace/PSM/benchmark/locomo/results/holdout-gate-v5h-conv-30-conv-41-ingest-summary.json 2>/dev/null || true
echo ---LOGTAIL---
tail -5 /workspace/PSM/benchmark/locomo/results/holdout-gate-v5n-dpo-conv-30-conv-41.log 2>/dev/null || true
tail -5 /workspace/PSM/benchmark/locomo/results/holdout-gate-v5n-conv-30-conv-41.log 2>/dev/null || true
tail -5 /workspace/PSM/benchmark/locomo/results/holdout-gate-v5h-conv-30-conv-41.log 2>/dev/null || true
exit
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="69.30.85.202")
    parser.add_argument("--port", default="22067")
    parser.add_argument("--user", default="root")
    parser.add_argument("--proxy-user", default="w4cvqv33efjsks-644112a7")
    parser.add_argument("--mode", choices=("direct-tcp", "proxy", "both"), default="both")
    args = parser.parse_args()

    targets: list[dict[str, str]] = []
    if args.mode in ("direct-tcp", "both"):
        targets.append(
            {"mode": "direct-tcp", "host": args.host, "port": args.port, "user": args.user}
        )
    if args.mode in ("proxy", "both"):
        targets.append(
            {"mode": "proxy", "host": "ssh.runpod.io", "port": "22", "user": args.proxy_user}
        )

    for t in targets:
        print(f"\n=== {t['mode']} {t['user']}@{t['host']}:{t['port']} ===", flush=True)
        try:
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
                    str(t["port"]),
                    f"{t['user']}@{t['host']}",
                    "bash",
                    "-s",
                ],
                input=PROBE,
                capture_output=True,
                text=True,
                timeout=90,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            continue
        out = (proc.stdout or "") + (proc.stderr or "")
        print(out)
        print(f"exit={proc.returncode}")
        if "PSM_PROBE_OK" in out:
            return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
