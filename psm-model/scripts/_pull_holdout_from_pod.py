#!/usr/bin/env python3
"""Pull holdout gate artifacts from RunPod via proxy SSH (tar.gz + base64 lines)."""
from __future__ import annotations

import argparse
import base64
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "benchmark/locomo/results"

ARTIFACTS = [
    "holdout-gate-v5n-dpo-conv-30-conv-41-ingest-summary.json",
    "holdout-gate-v5n-conv-30-conv-41-ingest-summary.json",
    "holdout-gate-v5h-conv-30-conv-41-ingest-summary.json",
    "holdout-gate-v5n-dpo-conv-30-conv-41.db",
    "holdout-gate-v5n-conv-30-conv-41.db",
    "holdout-gate-v5h-conv-30-conv-41.db",
    "holdout-gate-v5n-dpo-conv-30-conv-41.log",
    "holdout-gate-v5n-conv-30-conv-41.log",
    "holdout-gate-v5h-conv-30-conv-41.log",
]

REMOTE_DIR = "/workspace/PSM/benchmark/locomo/results"
GLOB = "holdout-gate-v5*-conv-30-conv-41*"
TAR_REMOTE = "/tmp/psm-holdout-artifacts.tgz"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
B64_LINE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _strip_pty(text: str) -> str:
    return ANSI_RE.sub("", text.replace("\r", ""))


def _extract_b64_lines(text: str) -> bytes:
    lines = []
    for line in _strip_pty(text).splitlines():
        line = line.strip()
        if B64_LINE.match(line) and len(line) >= 16:
            lines.append(line)
    if not lines:
        raise ValueError("no base64 lines in ssh output")
    return base64.b64decode("".join(lines), validate=False)


def _ssh_run(proxy_user: str, script: str, *, timeout_sec: int = 900) -> str:
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
            f"{proxy_user}@ssh.runpod.io",
            "bash",
            "-s",
        ],
        input=script,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ssh failed rc={proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}")
    return proc.stdout or ""


def _pull_tarball(proxy_user: str) -> bytes:
    script = (
        f"cd {REMOTE_DIR} && tar czf {TAR_REMOTE} {GLOB}\n"
        f"ls -la {TAR_REMOTE}\n"
        f"base64 {TAR_REMOTE}\n"
        "echo ENDMARKER\n"
        "exit\n"
    )
    return _extract_b64_lines(_ssh_run(proxy_user, script))


def _extract(data: bytes, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            dest = out_dir / Path(member.name).name
            dest.write_bytes(tf.extractfile(member).read())
            written.append(dest)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull holdout gate artifacts from RunPod.")
    parser.add_argument("--pod-id", default="w4cvqv33efjsks")
    parser.add_argument("--proxy-user", default="w4cvqv33efjsks-644112a7")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stop-after", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    print("pulling tar.gz from pod...", flush=True)
    try:
        blob = _pull_tarball(args.proxy_user)
    except Exception as exc:
        print(f"pull failed: {exc}", file=sys.stderr)
        return 1
    print(f"received {len(blob)} bytes (compressed)", flush=True)
    files = _extract(blob, out_dir)
    for f in sorted(files):
        print(f"  {f.name} ({f.stat().st_size} bytes)")
    print(f"\npulled {len(files)} files -> {out_dir}")
    if args.stop_after:
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "runpod_ctl.py"), "stop-pod", args.pod_id],
            check=False,
        )
        print(f"stopped pod {args.pod_id}")
    return 0 if files else 1


if __name__ == "__main__":
    raise SystemExit(main())
