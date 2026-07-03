#!/usr/bin/env python3
"""Resume holdout gate: push local ingest DBs, run retrieval+answer eval sequentially on GPU."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rpc  # noqa: E402
from _pull_holdout_from_pod import ARTIFACTS as LOCAL_ARTIFACTS  # noqa: E402
from _run_hf_holdout_gate import (  # noqa: E402
    DEFAULT_SAMPLES,
    MATRIX_OUT,
    PROFILES,
    PUSH_FILES,
    SCRIPTS,
    _bootstrap,
    _cf_env,
    _deploy,
    _hf_token,
    _ssh,
    _start_poller,
    _stop_poller,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "benchmark/locomo/results"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
B64_LINE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _artifact_paths() -> list[str]:
    return [f"benchmark/locomo/results/{name}" for name in LOCAL_ARTIFACTS]


def _push_local_artifacts(pod_id: str, proxy_user: str) -> None:
    missing = [p for p in _artifact_paths() if not (REPO / p).is_file()]
    if missing:
        raise SystemExit(f"missing local artifacts: {missing}")
    alias, host, port, user = _ssh(pod_id, proxy_user)
    rels = list(PUSH_FILES) + _artifact_paths()
    print(f"tar-push {len(rels)} file(s) to pod...", flush=True)
    push_rc = rpc._push_repo_files_via_tar(alias, REPO, rels, "/workspace/PSM", host=host, port=port, user=user)
    if push_rc != 0:
        raise RuntimeError(f"tar-push failed rc={push_rc}")


def _pull_results_tar(proxy_user: str, profiles: list[str]) -> list[dict]:
    script = (
        "cd /workspace/PSM/benchmark/locomo/results\n"
        "tar czf /tmp/psm-holdout-gate-out.tgz holdout-gate-matrix.json "
        "holdout-gate-v5*-conv-30-conv-41.json "
        "holdout-gate-v5*-conv-30-conv-41-retrieval.json "
        "holdout-gate-v5*-conv-30-conv-41-answer.json 2>/dev/null || true\n"
        "base64 /tmp/psm-holdout-gate-out.tgz\n"
        "echo ENDMARKER\n"
        "exit\n"
    )
    proc = subprocess.run(
        [
            rpc.SSH_BIN,
            "-tt",
            "-i",
            rpc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
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
        timeout=300,
    )
    text = ANSI_RE.sub("", (proc.stdout or "").replace("\r", ""))
    lines = [ln.strip() for ln in text.splitlines() if B64_LINE.match(ln.strip()) and len(ln.strip()) >= 16]
    if lines:
        blob = base64.b64decode("".join(lines), validate=False)
        RESULTS.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    (RESULTS / Path(member.name).name).write_bytes(tf.extractfile(member).read())

    if MATRIX_OUT.is_file():
        matrix = json.loads(MATRIX_OUT.read_text(encoding="utf-8"))
        return matrix.get("profiles") or []

    rows: list[dict] = []
    for key in profiles:
        profile = PROFILES[key]
        local = REPO / profile["gate_out"]
        ok = local.is_file()
        row: dict = {"profile": key, "gate_out": str(local), "exit_code": 0 if ok else 1}
        if ok:
            data = json.loads(local.read_text(encoding="utf-8"))
            row["retrieval_hit_at_1"] = (data.get("retrieval") or {}).get("hit_at_1")
            row["answer_accuracy"] = (data.get("answer") or {}).get("answer_accuracy")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume holdout gate (ingest local, eval on GPU pod).")
    parser.add_argument("--profiles", default="v5n-dpo,v5n,v5h")
    parser.add_argument("--sample-ids", default=DEFAULT_SAMPLES)
    parser.add_argument("--answer-limit", type=int, default=30)
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--poll-interval", type=int, default=0)
    args = parser.parse_args()

    if not _hf_token():
        raise SystemExit("HF_TOKEN missing")

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    for key in profiles:
        if key not in PROFILES:
            raise SystemExit(f"unknown profile: {key}")

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if not pod_id:
        pod_id, proxy_user = _deploy("psm-holdout-gate-resume")
        print(json.dumps({"event": "pod_ready", "pod_id": pod_id, "proxy_user": proxy_user}), flush=True)

    os.environ["PSM_HOLDOUT_SKIP_TAR_PUSH"] = "1"
    _bootstrap(pod_id, proxy_user)
    _push_local_artifacts(pod_id, proxy_user)

    alias, host, port, user = _ssh(pod_id, proxy_user)
    extra = {
        "HF_TOKEN": _hf_token(),
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": "krishnach7262/psm-prod-memory-hf",
        "PSM_RUNPOD": "1",
        "GATE_PROFILES": args.profiles,
        "HOLDOUT_SAMPLE_IDS": args.sample_ids,
        "GATE_ANSWER_LIMIT": str(args.answer_limit),
        "GATE_SEQUENTIAL": "1",
        "GATE_SKIP_INGEST": "1",
        **_cf_env(),
    }
    poller = _start_poller(pod_id, proxy_user, args.profiles, args.sample_ids, args.poll_interval)
    run_log = RESULTS / "holdout-gate-resume-remote.log"
    try:
        code = int(
            rpc._ssh_run_script(
                alias,
                SCRIPTS / "runpod_holdout_gate_matrix.sh",
                host=host,
                port=port,
                user=user,
                timeout_sec=10800,
                extra_env=extra,
                log_out=run_log,
            )
        )
    finally:
        _stop_poller(poller)

    rows = _pull_results_tar(proxy_user, profiles)
    if MATRIX_OUT.is_file():
        matrix = json.loads(MATRIX_OUT.read_text(encoding="utf-8"))
        matrix["mode"] = "resume_sequential"
        MATRIX_OUT.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    else:
        matrix = {"mode": "resume_sequential", "profiles": rows}
        MATRIX_OUT.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")

    if not args.keep_pod:
        subprocess.run([sys.executable, str(SCRIPTS / "runpod_ctl.py"), "stop-pod", pod_id], check=False)

    print(json.dumps({"matrix_out": str(MATRIX_OUT), "remote_exit": code, "profiles": rows}, indent=2))
    ok = code == 0 and all(r.get("exit_code") == 0 for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
