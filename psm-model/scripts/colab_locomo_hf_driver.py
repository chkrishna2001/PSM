#!/usr/bin/env python3
"""Run colab_locomo_hf.sh on the Colab VM via: colab exec -s SESSION --timeout N -f this.py"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT = os.environ.get("COLAB_LOCOMO_SCRIPT", "/content/colab_locomo_hf.sh")
ENV_FILE = os.environ.get("COLAB_ENV_FILE", "/content/colab_env.sh")
RUN_LOG = "/content/locomo/results/run.log"


def load_env_file(path: str) -> dict[str, str]:
    env = os.environ.copy()
    if not os.path.isfile(path):
        return env
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.removeprefix("export ").strip()
            val = val.strip().strip("'\"")
            if key:
                env[key] = val
    return env


def main() -> int:
    os.makedirs("/content/locomo/results", exist_ok=True)
    env = load_env_file(ENV_FILE)
    phase = env.get("COLAB_PHASE", "all")
    if not env.get("HF_TOKEN"):
        print("HF_TOKEN missing (set in /content/colab_env.sh)", file=sys.stderr)
        return 1
    if not os.path.isfile(SCRIPT):
        print(f"script not found: {SCRIPT}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(RUN_LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== driver phase={phase} {stamp} ===\n")
        logf.flush()
        proc = subprocess.run(
            ["bash", SCRIPT],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
        logf.write(f"=== driver exit rc={proc.returncode} phase={phase} ===\n")
    return int(proc.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
