#!/usr/bin/env python3
"""Print LoCoMo ingest progress from pod log."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

TOTAL_TURNS = 5882
LOG = "/workspace/PSM/benchmark/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.log"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pod-id", default="cyveakf0qhvqih")
    p.add_argument("--proxy-user", default="cyveakf0qhvqih-644111e0")
    args = p.parse_args()
    probe = f"""
grep -E 'ingested [0-9]+' '{LOG}' 2>/dev/null | tail -1
pgrep -af ingest-psm-model | grep -v grep | head -1 || echo INGEST_DONE
"""
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
            f"{args.proxy_user}@ssh.runpod.io",
            "bash",
            "-s",
        ],
        input=f"{probe}exit\n",
        capture_output=True,
        text=True,
        timeout=90,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stdout or ""
    seen = 0
    stored = ignored = failed = 0
    for line in text.splitlines():
        m = re.search(
            r"ingested (\d+) \| stored=(\d+) ignored=(\d+) failed=(\d+)",
            line,
        )
        if m:
            seen, stored, ignored, failed = map(int, m.groups())
    running = "ingest-psm-model" in text and "INGEST_DONE" not in text.split("INGEST_DONE")[0]
    left = max(TOTAL_TURNS - seen, 0)
    pct = round(100.0 * seen / TOTAL_TURNS, 1) if TOTAL_TURNS else 0.0
    print(
        {
            "seen": seen,
            "left": left,
            "total": TOTAL_TURNS,
            "pct": pct,
            "stored": stored,
            "ignored": ignored,
            "failed": failed,
            "ingest_running": running,
            "last_line": next(
                (ln.strip() for ln in text.splitlines() if "ingested " in ln),
                "",
            ),
        }
    )
    return 0 if seen else (1 if proc.returncode else 0)


if __name__ == "__main__":
    raise SystemExit(main())
