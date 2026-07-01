#!/usr/bin/env python3
"""LoCoMo HF ingest+eval on RunPod — tmux launch, SSH gate, GPU preflight, auto-stop on fail."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_ctl as rc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "psm-model" / "scripts"
MODEL_REPO = "krishnach7262/psm-prod-memory-hf"
LOCOMO_LOG = "/tmp/psm-locomo.log"
TMUX = "psm-locomo"
LOCOMO_POD_NAME = "psm-locomo-hf"

BINARY_ADAPTER = "psm-model/prod-memory/checkpoints/hf-prod-v5k-gate-distill-qwen0.5b/adapter"
EXTRACT_ADAPTER = "psm-model/prod-memory/checkpoints/hf-prod-v5k-extract-qwen0.5b/adapter"
BINARY_PREFIX = "hf-prod-v5k-gate-distill-qwen0.5b"
EXTRACT_PREFIX = "hf-prod-v5k-extract-qwen0.5b"

BUNDLE_ROOT_FILES = ["package.json", "package-lock.json"]
BUNDLE_GLOBS = ["tsconfig*.json"]
# ponytail: only push what LoCoMo needs — full dist/ is ~120MB of test-cli junk
BUNDLE_TREE: list[tuple[str, str]] = [
    ("dist/benchmark/locomo", "dist/benchmark/locomo"),
    ("src/psm-core/dist", "src/psm-core/dist"),
    ("src/psm-core/package.json", "src/psm-core/package.json"),
    ("src/psm-cli/package.json", "src/psm-cli/package.json"),
    ("src/psm-pi-plugin/package.json", "src/psm-pi-plugin/package.json"),
    ("benchmark/locomo/package.json", "benchmark/locomo/package.json"),
    ("benchmark/locomo/data", "benchmark/locomo/data"),
    ("psm-model/src", "psm-model/src"),
    ("psm-model/prod-memory", "psm-model/prod-memory"),
    ("psm-model/scripts", "psm-model/scripts"),
]

FAIL_PATTERNS = [
    r"PREFLIGHT FAIL",
    r"ModuleNotFoundError",
    r"RuntimeError: operator torchvision",
    r"OSError: Could not load",
    r"LoCoMo ingest failed",
]


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


def _ssh_info(pod_id: str) -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "ssh-info", pod_id],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    text = proc.stdout + proc.stderr
    proxy_user = ""
    try:
        payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
        proxy_user = str(payload.get("pod_host_id") or "")
        if not proxy_user:
            proxy_user = str((payload.get("recommended") or {}).get("user") or "")
        if not proxy_user:
            for target in payload.get("targets") or []:
                if isinstance(target, dict) and target.get("user"):
                    proxy_user = str(target["user"])
                    break
    except (ValueError, json.JSONDecodeError):
        pass
    return pod_id, proxy_user


def _list_locomo_pods() -> list[dict]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "list-pods"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return [p for p in json.loads(proc.stdout) if p.get("name") == LOCOMO_POD_NAME]


def _find_or_deploy_pod(*, force_deploy: bool = False) -> tuple[str, str]:
    if not force_deploy:
        pods = sorted(_list_locomo_pods(), key=lambda p: str(p.get("createdAt") or ""), reverse=True)
        for pod in pods:
            if pod.get("desiredStatus") == "RUNNING":
                pod_id = str(pod["id"])
                print(json.dumps({"event": "reuse_pod", "pod_id": pod_id}), flush=True)
                return _ssh_info(pod_id)
        if pods:
            pod_id = str(pods[0]["id"])
            print(json.dumps({"event": "reuse_stopped_pod", "pod_id": pod_id}), flush=True)
            return _ssh_info(pod_id)

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "runpod_ctl.py"),
            "deploy",
            "--auto-gpu",
            "--name",
            LOCOMO_POD_NAME,
            "--wait-ssh",
            "300",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    pods = sorted(_list_locomo_pods(), key=lambda p: str(p.get("createdAt") or ""), reverse=True)
    if not pods:
        raise SystemExit("deploy finished but no psm-locomo-hf pod found")
    pod_id = str(pods[0]["id"])
    print(json.dumps({"event": "deployed_pod", "pod_id": pod_id}), flush=True)
    return _ssh_info(pod_id)


def _stop_pod(pod_id: str) -> None:
    print(json.dumps({"event": "stop_pod", "pod_id": pod_id}), flush=True)
    rc._rest("POST", f"/pods/{pod_id}/stop")


def _require_ssh(pod_id: str, proxy_user: str, *, timeout_sec: int = 120) -> bool:
    ok = rc._wait_ssh_shell(
        "runpod-psm-proxy",
        host="ssh.runpod.io",
        port="22",
        user=proxy_user,
        timeout_sec=timeout_sec,
    )
    if not ok:
        print(json.dumps({"event": "ssh_dead", "pod_id": pod_id}), flush=True)
        _stop_pod(pod_id)
    return ok


def _ensure_pod_running(pod_id: str, proxy_user: str) -> bool:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "runpod_ctl.py"), "list-pods"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    for pod in json.loads(proc.stdout):
        if pod.get("id") != pod_id:
            continue
        if pod.get("desiredStatus") == "EXITED":
            print(json.dumps({"event": "start_pod", "pod_id": pod_id}), flush=True)
            rc._rest("POST", f"/pods/{pod_id}/start")
            return _require_ssh(pod_id, proxy_user, timeout_sec=300)
        return _require_ssh(pod_id, proxy_user, timeout_sec=120)
    return False


def _ignore_bundle(_dir: str, names: list[str]) -> set[str]:
    skip = {"node_modules", "__pycache__", ".git", "checkpoints", "results", ".pytest_cache"}
    return {name for name in names if name in skip or name.endswith((".db", ".pt", ".pyc"))}


def _push_locomo_bundle(alias: str, *, host: str | None, port: str | None, user: str) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        for name in BUNDLE_ROOT_FILES:
            src = REPO / name
            if src.is_file():
                shutil.copy2(src, bundle / name)
        for pattern in BUNDLE_GLOBS:
            for src in REPO.glob(pattern):
                if src.is_file():
                    shutil.copy2(src, bundle / src.name)
        for src_rel, dst_rel in BUNDLE_TREE:
            src = REPO / src_rel
            dst = bundle / dst_rel
            if not src.exists():
                print(f"bundle skip missing: {src_rel}", file=sys.stderr)
                continue
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                shutil.copytree(src, dst, ignore=_ignore_bundle, dirs_exist_ok=True)
        size_mb = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file()) / (1024 * 1024)
        print(json.dumps({"event": "tar_push_bundle", "size_mb": round(size_mb, 1)}), flush=True)
        return rc._ssh_push_dir(alias, bundle, "/workspace/PSM", host=host, port=port, user=user)


def _push_checkpoint_db(
    alias: str,
    local_db: Path,
    remote_rel: str,
    *,
    host: str | None,
    port: str | None,
    user: str,
) -> int:
    import tempfile

    if not local_db.is_file():
        print(f"checkpoint missing: {local_db}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        bundle_root = Path(tmp) / "bundle"
        dest = bundle_root / remote_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_db, dest)
        return rc._ssh_push_dir(alias, bundle_root, "/workspace/PSM", host=host, port=port, user=user)


def _locomo_env(token: str, *, limit: int, offset: int, resume_db: str) -> dict[str, str]:
    return {
        "HF_TOKEN": token,
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": MODEL_REPO,
        "PSM_RUNPOD": "1",
        "PSM_SKIP_GIT_PULL": "1",
        "LOCOMO_WAIT_FOR_EVAL": "0",
        "LOCOMO_DEVICE": "cuda",
        "LOCOMO_LIMIT": str(limit),
        "LOCOMO_OFFSET": str(offset),
        "LOCOMO_RESUME_DB": resume_db,
        "LOCOMO_HF_BINARY_ADAPTER": BINARY_ADAPTER,
        "LOCOMO_HF_EXTRACT_ADAPTER": EXTRACT_ADAPTER,
        "LOCOMO_HF_BINARY_PREFIX": BINARY_PREFIX,
        "LOCOMO_HF_EXTRACT_PREFIX": EXTRACT_PREFIX,
        "LOCOMO_HF_MODEL_KEY": "qwen0.5b",
        "LOCOMO_HF_LABEL": "hf-prod-v5k-two-pass",
        "LOCOMO_SKIP_BUILD": "1",
    }


def _start_locomo_tmux(
    alias: str,
    env: dict[str, str],
    *,
    host: str | None,
    port: str | None,
    user: str,
) -> int:
    return int(
        rc._ssh_run_script(
            alias,
            SCRIPTS / "runpod_start_locomo_hf.sh",
            host=host,
            port=port,
            user=user,
            timeout_sec=180,
            extra_env=env,
            skip_ssh_wait=True,
        )
    )


def _verify_tmux(pod_id: str, proxy_user: str, *, require_gpu: bool) -> bool:
    args = [
        sys.executable,
        str(SCRIPTS / "runpod_ctl.py"),
        "verify-pod",
        "--pod-id",
        pod_id,
        "--proxy-user",
        proxy_user,
        "--tmux-session",
        TMUX,
        "--process-pattern",
        "runpod_locomo|ingest-cli|hf_remember",
        "--train-log",
        LOCOMO_LOG,
        "--timeout-sec",
        "45",
    ]
    if not require_gpu:
        args.append("--no-require-gpu")
    proc = subprocess.run(args, cwd=REPO)
    return proc.returncode == 0


def _log_tail(proxy_user: str) -> str:
    proc = subprocess.run(
        [
            rc.SSH_BIN,
            "-i",
            rc.SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-p",
            "22",
            f"{proxy_user}@ssh.runpod.io",
            f"tail -100 {LOCOMO_LOG} 2>/dev/null || true",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        errors="replace",
    )
    return proc.stdout or ""


def _wait_log(
    proxy_user: str,
    *,
    success: str,
    timeout_sec: int,
    label: str,
) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        text = _log_tail(proxy_user)
        if re.search(success, text):
            print(json.dumps({"event": f"{label}_ok"}), flush=True)
            return True
        for pat in FAIL_PATTERNS:
            if re.search(pat, text):
                print(json.dumps({"event": f"{label}_fail", "pattern": pat, "tail": text[-800:]}), flush=True)
                return False
        if re.search(r"failed=[1-9]", text) and label in {"smoke", "ingest"}:
            print(json.dumps({"event": f"{label}_fail", "reason": "nonzero failed count", "tail": text[-800:]}), flush=True)
            return False
        time.sleep(20)
    print(json.dumps({"event": f"{label}_timeout", "tail": _log_tail(proxy_user)[-800:]}), flush=True)
    return False


def _pull_results(alias: str, limit_tag: str, *, host: str | None, port: str | None, user: str) -> int:
    remote_results = "/workspace/PSM/benchmark/locomo/results"
    local_results = REPO / "benchmark/locomo/results"
    local_sync = local_results / "pod-sync"
    local_results.mkdir(parents=True, exist_ok=True)
    local_sync.mkdir(parents=True, exist_ok=True)
    tmp = local_results.parent / ".locomo_hf_pull_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    pull_code = rc._ssh_pull_dir(alias, remote_results, tmp, host=host, port=port, user=user)
    if pull_code != 0:
        return pull_code
    for name in [
        f"locomo-hf-prod-v5k-two-pass-n{limit_tag}.db",
        f"locomo-hf-prod-v5k-two-pass-n{limit_tag}-results.json",
        f"locomo-hf-prod-v5k-two-pass-n{limit_tag}.log",
        "ingest-psm-model-summary.json",
    ]:
        src = tmp / name
        if src.is_file():
            shutil.copy2(src, local_results / name)
            shutil.copy2(src, local_sync / name)
            print(f"pulled {name} ({src.stat().st_size} bytes)")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


def _run_phase(
    pod_id: str,
    proxy_user: str,
    alias: str,
    *,
    host: str | None,
    port: str | None,
    user: str,
    token: str,
    limit: int,
    offset: int,
    checkpoint_db: Path,
    resume_rel: str,
    phase: str,
) -> bool:
    limit_tag = "full" if limit == 0 else str(limit)
    resume_abs = f"/workspace/PSM/{resume_rel}"
    print(
        json.dumps(
            {"event": "phase_start", "phase": phase, "limit": limit, "offset": offset, "pod_id": pod_id}
        ),
        flush=True,
    )
    if _push_locomo_bundle(alias, host=host, port=port, user=user) != 0:
        return False
    if _push_checkpoint_db(alias, checkpoint_db, resume_rel, host=host, port=port, user=user) != 0:
        return False
    env = _locomo_env(token, limit=limit, offset=offset, resume_db=resume_abs)
    if _start_locomo_tmux(alias, env, host=host, port=port, user=user) != 0:
        return False
    if not _verify_tmux(pod_id, proxy_user, require_gpu=False):
        print(json.dumps({"event": "tmux_verify_fail", "phase": phase}), flush=True)
        return False
    if not _wait_log(proxy_user, success=r"HF preflight PASS", timeout_sec=1800, label=f"{phase}_preflight"):
        return False
    if not _verify_tmux(pod_id, proxy_user, require_gpu=True):
        print(json.dumps({"event": "gpu_verify_fail", "phase": phase}), flush=True)
        return False
    if limit > 0:
        return _wait_log(
            proxy_user,
            success=rf"ingested {limit} \| stored=.*failed=0",
            timeout_sec=3600,
            label="smoke",
        )
    if not _wait_log(proxy_user, success=r"=== LoCoMo done", timeout_sec=6 * 3600, label="benchmark"):
        return False
    return _pull_results(alias, limit_tag, host=host, port=port, user=user) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--proxy-user", default="")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--force-deploy", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="5-turn smoke only")
    parser.add_argument("--smoke-then-full", action="store_true", help="5-turn smoke then full ingest+eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=2963)
    parser.add_argument(
        "--checkpoint-db",
        default=str(REPO / "benchmark/locomo/results/pod-sync/locomo-hf-prod-v5k-two-pass-nfull.db"),
    )
    args = parser.parse_args()

    token = _hf_token()
    if not token:
        print("HF_TOKEN required", file=sys.stderr)
        return 1

    pod_id = args.pod_id.strip()
    proxy_user = args.proxy_user.strip()
    if args.deploy or not pod_id:
        pod_id, proxy_user = _find_or_deploy_pod(force_deploy=args.force_deploy)
    elif not proxy_user:
        _, proxy_user = _ssh_info(pod_id)
    if not pod_id or not proxy_user:
        print("need pod_id + proxy_user", file=sys.stderr)
        return 1

    if not _ensure_pod_running(pod_id, proxy_user):
        return 255

    ns = argparse.Namespace(pod_id=pod_id, proxy_user=proxy_user, deploy=False, host_alias="runpod-psm-proxy")
    _, host, port, user = rc._resolve_train_pod_ssh(ns, proxy_user=proxy_user)
    alias = "runpod-psm-proxy"
    checkpoint = Path(args.checkpoint_db)
    resume_rel = "benchmark/locomo/results/locomo-hf-prod-v5k-two-pass-nfull.db"

    try:
        if args.smoke or args.smoke_then_full:
            ok = _run_phase(
                pod_id,
                proxy_user,
                alias,
                host=host,
                port=port,
                user=user,
                token=token,
                limit=5,
                offset=args.offset,
                checkpoint_db=checkpoint,
                resume_rel=resume_rel,
                phase="smoke",
            )
            if not ok:
                _stop_pod(pod_id)
                return 1
            if args.smoke and not args.smoke_then_full:
                print(json.dumps({"event": "smoke_passed", "pod_id": pod_id}), flush=True)
                return 0

        limit = 0 if args.smoke_then_full or (not args.smoke and args.limit == 0) else args.limit
        ok = _run_phase(
            pod_id,
            proxy_user,
            alias,
            host=host,
            port=port,
            user=user,
            token=token,
            limit=limit,
            offset=args.offset,
            checkpoint_db=checkpoint,
            resume_rel=resume_rel,
            phase="full",
        )
        if not ok:
            _pull_results(alias, "full", host=host, port=port, user=user)
            _stop_pod(pod_id)
            return 1
        _stop_pod(pod_id)
        return 0
    except KeyboardInterrupt:
        _stop_pod(pod_id)
        raise
    except Exception as exc:
        print(json.dumps({"event": "launch_error", "error": str(exc)}), flush=True)
        _stop_pod(pod_id)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    raise SystemExit(main())
