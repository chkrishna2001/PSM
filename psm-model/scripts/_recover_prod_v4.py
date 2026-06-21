#!/usr/bin/env python3
"""Recover prod-memory v4 artifacts from RunPod pod (ponytail: delete after use)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "psm-model" / "scripts"))
import runpod_ctl as rp  # noqa: E402

PROXY_USER = "st0luf214e32c5-64411541"
PROXY_HOST = "ssh.runpod.io"
PROXY_PORT = "22"
DEFAULT_TCP = ("69.30.85.241", "22136", "root")


def ssh_endpoint(*, tcp: bool) -> tuple[str | None, str | None, str]:
    if tcp:
        host, port, user = DEFAULT_TCP
        return host, port, user
    return PROXY_HOST, PROXY_PORT, PROXY_USER


def ssh_bash(
    command: str,
    *,
    tcp: bool = False,
    timeout: int = 120,
) -> tuple[int, str]:
    host, port, user = ssh_endpoint(tcp=tcp)
    proc = subprocess.run(
        [
            rp.SSH_BIN,
            "-tt",
            "-i",
            rp.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *rp._ssh_endpoint(rp.SSH_CONFIG_HOST, host=host, port=port, user=user),
            "bash",
            "-s",
        ],
        input=f"{command}\nexit\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, proc.stdout + proc.stderr


def probe(*, tcp: bool) -> int:
    cmd = r"""
echo PSM_PROBE_BEGIN
ls -1 /workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4-step-*.pt 2>/dev/null | sed 's|^|PSM_CKPT=|'
ls -1 /workspace/PSM/psm-model/prod-memory/results/prod-grounding-*.json 2>/dev/null | sed 's|^|PSM_RESULT=|'
test -f /workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4.metrics.jsonl && echo PSM_METRICS=OK || echo PSM_METRICS=MISSING
du -ch /workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-prod-memory-v4-step-*.pt 2>/dev/null | tail -1 | sed 's|^|PSM_SIZE=|'
echo PSM_PROBE_END
"""
    rc, out = ssh_bash(cmd, tcp=tcp, timeout=120)
    for line in out.splitlines():
        if line.startswith("PSM_"):
            print(line)
    if "PSM_PROBE_BEGIN" not in out:
        print("probe failed — no output markers", file=sys.stderr)
        print(out[-2000:], file=sys.stderr)
        return rc or 1
    return 0


def pull(remote_dir: str, local_dir: Path, *, tcp: bool) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    host, port, user = ssh_endpoint(tcp=tcp)
    print(f"pulling {remote_dir} -> {local_dir}")
    return rp._ssh_pull_dir(
        rp.SSH_CONFIG_HOST,
        remote_dir,
        local_dir,
        host=host,
        port=port,
        user=user,
    )


def upload_hf(local_ckpt_dir: Path) -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN missing — skip upload", file=sys.stderr)
        return 1
    repo = os.environ.get("PSM_HF_MODEL_REPO", rp.DEFAULT_HF_MODEL_REPO)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    uploaded = 0
    for path in sorted(local_ckpt_dir.glob("real-v3-50m-full-v2-prod-memory-v4*")):
        remote = f"psm-model/checkpoints/{path.name}"
        print(f"upload_hf {remote}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=repo,
            repo_type="model",
            commit_message=f"recover prod-memory v4 {path.name}",
        )
        uploaded += 1
    print(f"uploaded {uploaded} files to {repo}")
    return 0 if uploaded else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("probe", "pull", "upload", "all"), default="all")
    p.add_argument("--tcp", action="store_true", help="Use direct TCP port 22136")
    args = p.parse_args()

    if args.mode in ("probe", "all"):
        rc = probe(tcp=args.tcp)
        if rc != 0:
            return rc

    local_ckpt = REPO / "psm-model" / "checkpoints"
    local_results = REPO / "psm-model" / "prod-memory" / "results"

    if args.mode in ("pull", "all"):
        # ponytail: pull only v4 stem files via tar of matching globs
        pull_cmd = r"""
echo PSM_PULL_BEGIN
cd /workspace/PSM/psm-model/checkpoints && tar -czf - real-v3-50m-full-v2-prod-memory-v4* 2>/dev/null | base64 -w0
echo
echo PSM_PULL_CKPT_END
"""
        rc, out = ssh_bash(pull_cmd, tcp=args.tcp, timeout=1800)
        if "PSM_PULL_BEGIN" not in out:
            print("checkpoint pull failed", file=sys.stderr)
            return 1
        import base64
        import tarfile
        import tempfile

        begin = out.find("PSM_PULL_BEGIN")
        end = out.find("PSM_PULL_CKPT_END")
        payload = "".join(out[begin + len("PSM_PULL_BEGIN") : end].split())
        raw = base64.b64decode(payload, validate=False)
        local_ckpt.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            with tarfile.open(tmp_path, "r:gz") as archive:
                archive.extractall(local_ckpt)
            print(f"extracted checkpoints to {local_ckpt}")
        finally:
            tmp_path.unlink(missing_ok=True)

        rc2 = pull("/workspace/PSM/psm-model/prod-memory/results", local_results, tcp=args.tcp)
        if rc2 != 0:
            print(f"warning: results pull exit {rc2}", file=sys.stderr)

    if args.mode in ("upload", "all"):
        return upload_hf(local_ckpt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
