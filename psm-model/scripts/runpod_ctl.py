#!/usr/bin/env python3
"""RunPod pod/template helpers. Set RUNPOD_API_KEY (e.g. from `o runpodkey`)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
SSH_CONFIG_HOST = "runpod-psm"
SSH_BIN = "ssh.exe" if os.name == "nt" else "ssh"
SCP_BIN = "scp.exe" if os.name == "nt" else "scp"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
SSH_CACHE_PATH = Path(os.environ.get("RUNPOD_SSH_CACHE", "psm-model/checkpoints/.runpod-ssh-cache.json"))

# RunPod secret named HF_TOKEN → injected as HF_TOKEN env at pod start.
HF_TOKEN_SECRET_REF = "{{ RUNPOD_SECRET_HF_TOKEN }}"

# Custom image chkrishna2001/psm-50m-train:latest is NOT on Docker Hub — use stock PyTorch only.
STOCK_PYTORCH_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

DEFAULT_TEMPLATE = {
    "name": "psm-50m-train",
    "imageName": STOCK_PYTORCH_IMAGE,
    "containerDiskInGb": 20,
    "volumeInGb": 40,
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
    "dockerStartCmd": ["sleep", "infinity"],
    "env": {
        "HF_TOKEN": HF_TOKEN_SECRET_REF,
        "PYTHONPATH": "psm-model/src",
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": "chkrishna2001/psm-50m-mixed-v1-run",
        "PSM_HF_DATASET_REPO": "chkrishna2001/psm-50m-action-mixed-v1",
    },
}


def _api_key_from_opener() -> str:
    opener = "o"
    if os.name == "nt":
        opener = "o.exe"
    subprocess.run([opener, "runpodkey"], check=True, capture_output=True, text=True)
    if os.name == "nt":
        clip = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
            check=True,
            capture_output=True,
            text=True,
        )
        return clip.stdout.strip()
    clip = subprocess.run(["pbpaste"], check=True, capture_output=True, text=True)
    return clip.stdout.strip()


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    try:
        key = _api_key_from_opener()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Could not load RunPod API key via `o runpodkey`") from exc
    if not key:
        raise SystemExit("RunPod API key from `o runpodkey` was empty")
    return key


def _graphql(query: str, variables: dict | None = None) -> dict:
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    req = urllib.request.Request(
        f"{GRAPHQL_URL}?api_key={_api_key()}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if body.get("errors"):
        raise RuntimeError(json.dumps(body["errors"], indent=2))
    return body["data"]


def _rest(method: str, path: str, data: dict | None = None) -> dict | list:
    url = f"{REST_URL}{path}"
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {"status": resp.status}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise SystemExit(f"RunPod REST {method} {path} failed ({exc.code}): {detail}") from exc


def cmd_list_pods(_: argparse.Namespace) -> int:
    pods = _rest("GET", "/pods")
    print(json.dumps(pods, indent=2))
    return 0


def cmd_stop_pod(args: argparse.Namespace) -> int:
    result = _rest("POST", f"/pods/{args.pod_id}/stop")
    print(json.dumps(result, indent=2))
    return 0


def cmd_delete_pod(args: argparse.Namespace) -> int:
    result = _rest("DELETE", f"/pods/{args.pod_id}")
    print(json.dumps(result, indent=2))
    return 0


def cmd_stop_all(_: argparse.Namespace) -> int:
    pods = _rest("GET", "/pods")
    for pod in pods:
        if pod.get("desiredStatus") in {"RUNNING", "EXITED"}:
            print(f"Stopping {pod['id']} ({pod.get('name')})...")
            _rest("POST", f"/pods/{pod['id']}/stop")
    return 0


def cmd_delete_all(_: argparse.Namespace) -> int:
    pods = _rest("GET", "/pods")
    for pod in pods:
        print(f"Deleting {pod['id']} ({pod.get('name')})...")
        _rest("DELETE", f"/pods/{pod['id']}")
    return 0


def cmd_create_template(args: argparse.Namespace) -> int:
    spec = dict(DEFAULT_TEMPLATE)
    if args.image:
        spec["imageName"] = args.image
    spec["name"] = args.name
    spec["readme"] = (
        "PSM 50M training image. Requires RunPod secret HF_TOKEN "
        f"(env HF_TOKEN={HF_TOKEN_SECRET_REF}). Bootstrap pulls checkpoints/data from HF on start."
    )
    try:
        result = _rest("POST", "/templates", spec)
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))
    return 0


AUTOSTART_CMD = (
    "pip install -q huggingface_hub hf_transfer numpy tmux git && "
    "hf download chkrishna2001/psm-50m-action-mixed-v1 runpod/runpod_autostart.sh "
    "--repo-type dataset --local-dir /tmp/psm-autostart && "
    "bash /tmp/psm-autostart/runpod/runpod_autostart.sh"
)


def _fetch_pod(pod_id: str) -> dict:
    return _rest("GET", f"/pods/{pod_id}")


def _ssh_cache_load() -> dict[str, str]:
    if not SSH_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(SSH_CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _ssh_cache_save(pod_id: str, pod_host_id: str) -> None:
    cache = _ssh_cache_load()
    cache[pod_id] = pod_host_id
    SSH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SSH_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _graphql_pod_details(pod_id: str) -> dict[str, object]:
    query = (
        "query PodSsh($podId: String!) {"
        " pod(input: { podId: $podId }) {"
        " id machine { podHostId }"
        " runtime { ports { ip isIpPublic privatePort publicPort type } }"
        " } }"
    )
    try:
        data = _graphql(query, {"podId": pod_id})
    except urllib.error.HTTPError:
        return {}
    pod = data.get("pod") or {}
    return pod if isinstance(pod, dict) else {}


def _resolve_pod_host_id(pod_id: str, pod: dict, *, proxy_user: str | None = None) -> str | None:
    if proxy_user:
        return proxy_user.strip()
    cached = _ssh_cache_load().get(pod_id)
    if cached:
        return cached
    gql = _graphql_pod_details(pod_id)
    machine = gql.get("machine") if isinstance(gql.get("machine"), dict) else {}
    pod_host_id = machine.get("podHostId")
    if isinstance(pod_host_id, str) and pod_host_id.strip():
        return pod_host_id.strip()
    rest_machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    pod_host_id = rest_machine.get("podHostId")
    if isinstance(pod_host_id, str) and pod_host_id.strip():
        return pod_host_id.strip()
    return None


def _direct_tcp_target(pod: dict) -> dict[str, str] | None:
    public_ip = str(pod.get("publicIp") or "").strip()
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22") or port_mappings.get(22)
    if not public_ip or not ssh_port:
        return None
    return {
        "mode": "direct-tcp",
        "host": public_ip,
        "port": str(ssh_port),
        "user": "root",
        "command": f"{SSH_BIN} -i {SSH_KEY_PATH} root@{public_ip} -p {ssh_port}",
        "pod_id": str(pod.get("id", "")),
    }


def _proxy_target(pod_host_id: str, *, pod_id: str) -> dict[str, str]:
    return {
        "mode": "proxy",
        "host": "ssh.runpod.io",
        "port": "22",
        "user": pod_host_id,
        "command": f"{SSH_BIN} -i {SSH_KEY_PATH} {pod_host_id}@ssh.runpod.io",
        "pod_id": pod_id,
    }


def _pod_ssh_targets(pod_id: str, *, proxy_user: str | None = None) -> list[dict[str, str]]:
    pod = _fetch_pod(pod_id)
    targets: list[dict[str, str]] = []
    pod_host_id = _resolve_pod_host_id(pod_id, pod, proxy_user=proxy_user)
    if pod_host_id:
        targets.append(_proxy_target(pod_host_id, pod_id=pod_id))
    direct = _direct_tcp_target(pod)
    if direct is not None:
        targets.append(direct)
    if not targets:
        raise SystemExit(
            json.dumps(
                {
                    "error": "no_ssh_targets",
                    "pod_id": pod_id,
                    "hint": "Pass --proxy-user from the pod Connect tab (e.g. znq...-64411407) or enable GraphQL API access for podHostId.",
                    "pod": pod,
                },
                indent=2,
            )
        )
    return targets


def _write_ssh_config(host_alias: str, target: dict[str, str], *, proxy_target: dict[str, str] | None = None) -> Path:
    config_path = Path(os.path.expanduser("~/.ssh/config"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    block = (
        f"Host {host_alias}\n"
        f"    HostName {target['host']}\n"
        f"    Port {target['port']}\n"
        f"    User {target['user']}\n"
        f"    IdentityFile {SSH_KEY_PATH}\n"
        f"    StrictHostKeyChecking accept-new\n"
    )
    if proxy_target is not None:
        block += (
            f"\nHost {host_alias}-proxy\n"
            f"    HostName {proxy_target['host']}\n"
            f"    Port {proxy_target['port']}\n"
            f"    User {proxy_target['user']}\n"
            f"    IdentityFile {SSH_KEY_PATH}\n"
            f"    StrictHostKeyChecking accept-new\n"
        )
    existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    lines = existing.splitlines()
    out: list[str] = []
    skipping = False
    skip_hosts = {host_alias, f"{host_alias}-proxy"}
    for line in lines:
        host_line = line.strip()
        if host_line in {f"Host {name}" for name in skip_hosts}:
            skipping = True
            continue
        if skipping and line.startswith("Host "):
            skipping = False
        if not skipping:
            out.append(line)
    if out and out[-1].strip():
        out.append("")
    out.extend(block.splitlines())
    config_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return config_path


def cmd_ssh_info(args: argparse.Namespace) -> int:
    targets = _pod_ssh_targets(args.pod_id, proxy_user=args.proxy_user or None)
    pod = _fetch_pod(args.pod_id)
    pod_host_id = _resolve_pod_host_id(args.pod_id, pod, proxy_user=args.proxy_user or None)
    if pod_host_id:
        _ssh_cache_save(args.pod_id, pod_host_id)
    report = {
        "pod_id": args.pod_id,
        "pod_name": pod.get("name"),
        "pod_host_id": pod_host_id,
        "public_ip": pod.get("publicIp"),
        "port_mappings": pod.get("portMappings"),
        "targets": targets,
        "recommended": targets[0],
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_ssh_config(args: argparse.Namespace) -> int:
    targets = _pod_ssh_targets(args.pod_id, proxy_user=args.proxy_user or None)
    pod = _fetch_pod(args.pod_id)
    direct = next((item for item in targets if item["mode"] == "direct-tcp"), targets[0])
    proxy = next((item for item in targets if item["mode"] == "proxy"), None)
    pod_host_id = _resolve_pod_host_id(args.pod_id, pod, proxy_user=args.proxy_user or None)
    if pod_host_id:
        _ssh_cache_save(args.pod_id, pod_host_id)
    config_path = _write_ssh_config(args.host_alias, direct, proxy_target=proxy)
    print(
        json.dumps(
            {
                "pod_id": pod.get("id"),
                "pod_name": pod.get("name"),
                "pod_host_id": pod_host_id,
                "ssh_config": str(config_path),
                "host_alias": args.host_alias,
                "proxy_host_alias": f"{args.host_alias}-proxy",
                "targets": targets,
            },
            indent=2,
        )
    )
    return 0


def cmd_wait_ssh(args: argparse.Namespace) -> int:
    try:
        _wait_pod_ssh_endpoint(
            args.pod_id,
            timeout_sec=args.timeout_sec,
            poll_sec=args.poll_sec,
            proxy_user=args.proxy_user or None,
        )
        return 0
    except SystemExit:
        return 1


def _ssh_endpoint(
    host_alias: str,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
) -> list[str]:
    if host and port:
        return ["-p", str(port), f"{user}@{host}"]
    return [host_alias]


def _ssh_probe(target: dict[str, str]) -> bool:
    # RunPod proxy SSH requires a PTY; remote one-liners fail — pipe a tiny script instead.
    probe = subprocess.run(
        [
            SSH_BIN,
            "-tt",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *_ssh_endpoint(
                SSH_CONFIG_HOST,
                host=target["host"],
                port=target["port"],
                user=target["user"],
            ),
            "bash",
            "-s",
        ],
        input="echo ssh-ready\nexit\n",
        capture_output=True,
        text=True,
        timeout=90,
    )
    ok = probe.returncode == 0 and "ssh-ready" in probe.stdout
    if ok and target.get("mode") == "proxy" and target.get("pod_id"):
        _ssh_cache_save(str(target["pod_id"]), str(target["user"]))
    return ok


def _wait_pod_ssh_endpoint(
    pod_id: str,
    *,
    timeout_sec: int = 420,
    poll_sec: int = 10,
    proxy_user: str | None = None,
) -> dict[str, str]:
    import time

    deadline = time.time() + timeout_sec
    last: dict | None = None
    while time.time() < deadline:
        last = _fetch_pod(pod_id)
        try:
            targets = _pod_ssh_targets(pod_id, proxy_user=proxy_user)
        except SystemExit:
            time.sleep(poll_sec)
            continue
        cmd_ssh_config(
            argparse.Namespace(
                pod_id=pod_id,
                host_alias=SSH_CONFIG_HOST,
                proxy_user=proxy_user or "",
            )
        )
        for target in targets:
            if _ssh_probe(target):
                return target
        time.sleep(poll_sec)
    raise SystemExit(json.dumps({"event": "ssh_endpoint_timeout", "pod": last}, indent=2))


def _wait_ssh_shell(
    host_alias: str,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
    timeout_sec: int = 300,
    poll_sec: int = 10,
) -> bool:
    import time

    deadline = time.time() + timeout_sec
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        probe = subprocess.run(
            [
                SSH_BIN,
                "-tt",
                "-i",
                SSH_KEY_PATH,
                "-o",
                "ConnectTimeout=20",
                "-o",
                "StrictHostKeyChecking=accept-new",
                *_ssh_endpoint(host_alias, host=host, port=port, user=user),
                "bash",
                "-s",
            ],
            input="echo ssh-ready\nexit\n",
            capture_output=True,
            text=True,
            timeout=90,
        )
        if probe.returncode == 0 and "ssh-ready" in probe.stdout:
            print(
                json.dumps(
                    {
                        "event": "ssh_ready",
                        "attempt": attempt,
                        "host": host or host_alias,
                        "port": port,
                        "user": user,
                    },
                    indent=2,
                )
            )
            return True
        print(
            json.dumps(
                {
                    "event": "ssh_wait",
                    "attempt": attempt,
                    "host": host or host_alias,
                    "port": port,
                    "user": user,
                    "stderr": (probe.stderr or "").strip()[-200:],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        time.sleep(poll_sec)
    return False


def _ssh_stream_print(line: str) -> None:
    try:
        print(line, end="" if line.endswith("\n") else "\n")
    except UnicodeEncodeError:
        safe = line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8",
            errors="replace",
        )
        print(safe, end="" if safe.endswith("\n") else "\n")


def _ssh_run_script(
    host_alias: str,
    script_path: Path,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
    timeout_sec: int = 7200,
    extra_env: dict[str, str] | None = None,
    ssh_ready_timeout_sec: int = 300,
    skip_ssh_wait: bool = False,
) -> int:
    if not skip_ssh_wait and not _wait_ssh_shell(
        host_alias,
        host=host,
        port=port,
        user=user,
        timeout_sec=ssh_ready_timeout_sec,
    ):
        print(f"SSH not ready on {host or host_alias} after {ssh_ready_timeout_sec}s", file=sys.stderr)
        return 255
    body = script_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if extra_env:
        exports = "\n".join(f"export {key}={value}" for key, value in extra_env.items())
        body = f"{exports}\n{body}"
    encoded = base64.b64encode(body.encode("utf-8")).decode("ascii")
    # RunPod proxy SSH ignores remote argv; pipe through bash -s. Chunk base64 — one-line printf stalls on PTY.
    chunk_lines = [encoded[index : index + 120] for index in range(0, len(encoded), 120)]
    heredoc = "\n".join(chunk_lines)
    stdin = (
        "cat > /tmp/psm-remote.b64 <<'PSM_B64_EOF'\n"
        f"{heredoc}\n"
        "PSM_B64_EOF\n"
        "base64 -d /tmp/psm-remote.b64 | bash\n"
        "exit\n"
    )
    proc = subprocess.Popen(
        [
            SSH_BIN,
            "-tt",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *_ssh_endpoint(host_alias, host=host, port=port, user=user),
            "bash",
            "-s",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdin is not None
    proc.stdin.write(stdin)
    proc.stdin.close()
    assert proc.stdout is not None
    import time

    deadline = time.time() + timeout_sec
    while True:
        if proc.poll() is not None:
            remainder = proc.stdout.read()
            if remainder:
                _ssh_stream_print(remainder)
            break
        if time.time() > deadline:
            proc.kill()
            print(f"SSH eval timed out after {timeout_sec}s", file=sys.stderr)
            return 124
        line = proc.stdout.readline()
        if line:
            _ssh_stream_print(line)
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.05)
    return proc.returncode if proc.returncode is not None else 124


def _ssh_pull_dir(
    host_alias: str,
    remote_path: str,
    local_path: Path,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
) -> int:
    """Tar+base64 over bash -s — works when RunPod proxy blocks scp."""
    import tarfile
    import tempfile

    local_path.mkdir(parents=True, exist_ok=True)
    pull_cmd = (
        f"if [[ -d '{remote_path}' ]]; then\n"
        "echo PSM_PULL_BEGIN\n"
        f"tar -C '{remote_path}' -czf - . | base64 -w0\n"
        "echo\n"
        "echo PSM_PULL_END\n"
        "else\n"
        "echo PSM_PULL_MISSING\n"
        "fi"
    )
    result = subprocess.run(
        [
            SSH_BIN,
            "-tt",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *_ssh_endpoint(host_alias, host=host, port=port, user=user),
            "bash",
            "-s",
        ],
        input=f"{pull_cmd}\nexit\n",
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        return result.returncode
    if "PSM_PULL_MISSING" in result.stdout:
        print(f"remote dir missing or empty: {remote_path}", file=sys.stderr)
        return 1
    begin = result.stdout.find("PSM_PULL_BEGIN")
    end = result.stdout.find("PSM_PULL_END")
    if begin < 0 or end < 0 or end <= begin:
        print("could not find pull markers in ssh output", file=sys.stderr)
        return 1
    payload = "".join(result.stdout[begin + len("PSM_PULL_BEGIN") : end].split())
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        print(f"could not decode pulled archive: {exc}", file=sys.stderr)
        return 1
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "r:gz") as archive:
            archive.extractall(local_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return 0


def _scp_from_pod(
    host_alias: str,
    remote_path: str,
    local_path: Path,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_target = f"{host_alias}:{remote_path}"
    if host and port:
        result = subprocess.run(
            [SCP_BIN, "-r", "-P", str(port), "-i", SSH_KEY_PATH, f"{user}@{host}:{remote_path}", str(local_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            return 0
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        print("scp failed; falling back to ssh tar pull", file=sys.stderr)
        return _ssh_pull_dir(host_alias, remote_path, local_path, host=host, port=port, user=user)
    result = subprocess.run(
        [SCP_BIN, "-r", "-i", SSH_KEY_PATH, remote_target, str(local_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        return 0
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    print("scp failed; falling back to ssh tar pull", file=sys.stderr)
    return _ssh_pull_dir(host_alias, remote_path, local_path, host=host, port=port, user=user)


def cmd_eval_gates(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve().parent / "runpod_eval_gates.sh"
    if not script_path.exists():
        raise SystemExit(f"missing eval script: {script_path}")

    pod_id = args.pod_id
    ssh_host: str | None = None
    ssh_port: str | None = None
    ssh_user = "root"
    proxy_user = args.proxy_user or None
    if args.pod_id and not args.deploy:
        target = _wait_pod_ssh_endpoint(
            args.pod_id,
            timeout_sec=max(args.wait_ssh, args.ssh_ready_timeout_sec),
            proxy_user=proxy_user,
        )
        ssh_host = target["host"]
        ssh_port = target["port"]
        ssh_user = target["user"]
        pod_id = args.pod_id
        print(json.dumps({"event": "using_ssh_endpoint", **target}, indent=2))

    if args.deploy:
        deploy_args = argparse.Namespace(
            name=args.name,
            image=args.image,
            template=args.template,
            gpu=args.gpu,
            volume_gb=args.volume_gb,
            container_disk_gb=args.container_disk_gb,
            autostart=False,
            wait_ssh=0,
        )
        data = _rest("POST", "/pods", _deploy_payload(deploy_args))
        print(json.dumps(data, indent=2))
        pod_id = str(data.get("id", ""))
        if not pod_id:
            raise SystemExit("deploy did not return pod id")
        target = _wait_pod_ssh_endpoint(
            pod_id,
            timeout_sec=max(args.wait_ssh, args.ssh_ready_timeout_sec),
            proxy_user=proxy_user,
        )
        ssh_host = target["host"]
        ssh_port = target["port"]
        ssh_user = target["user"]
        print(json.dumps({"event": "using_ssh_endpoint", **target}, indent=2))

    extra_env = {
        "PSM_EVAL_DEVICE": args.device,
        "PSM_EVAL_EXPANDED": "1" if args.expanded else "0",
    }
    rc = _ssh_run_script(
        args.host_alias,
        script_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=args.timeout_sec,
        extra_env=extra_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=bool(args.deploy or args.pod_id),
    )

    remote_report = "/workspace/PSM/psm-model/checkpoints/gate-eval"
    if args.pull_reports:
        local_report = Path(args.pull_reports)
        scp_rc = _scp_from_pod(
            args.host_alias,
            remote_report,
            local_report,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
        if scp_rc != 0:
            print(f"warning: could not scp reports from {remote_report}", file=sys.stderr)
        else:
            print(json.dumps({"pulled_reports": str(local_report.resolve())}, indent=2))

    if args.delete_after and pod_id:
        if rc == 255:
            print(f"Skipping pod delete because eval never started (pod {pod_id} left running).", file=sys.stderr)
        else:
            print(f"Deleting pod {pod_id}...")
            _rest("DELETE", f"/pods/{pod_id}")

    return rc


def _resolve_train_pod_ssh(
    args: argparse.Namespace,
    *,
    proxy_user: str | None,
) -> tuple[str, str | None, str | None, str]:
    pod_id = args.pod_id
    ssh_host: str | None = None
    ssh_port: str | None = None
    ssh_user = "root"
    if args.pod_id and not args.deploy:
        target = _wait_pod_ssh_endpoint(
            args.pod_id,
            timeout_sec=max(args.wait_ssh, args.ssh_ready_timeout_sec),
            proxy_user=proxy_user,
        )
        ssh_host = target["host"]
        ssh_port = target["port"]
        ssh_user = target["user"]
        pod_id = args.pod_id
        print(json.dumps({"event": "using_ssh_endpoint", **target}, indent=2))
        return pod_id, ssh_host, ssh_port, ssh_user

    if args.deploy:
        deploy_args = argparse.Namespace(
            name=args.name,
            image=args.image,
            template=args.template,
            gpu=args.gpu,
            volume_gb=args.volume_gb,
            container_disk_gb=args.container_disk_gb,
            autostart=False,
            wait_ssh=0,
        )
        data = _rest("POST", "/pods", _deploy_payload(deploy_args))
        print(json.dumps(data, indent=2))
        pod_id = str(data.get("id", ""))
        if not pod_id:
            raise SystemExit("deploy did not return pod id")
        target = _wait_pod_ssh_endpoint(
            pod_id,
            timeout_sec=max(args.wait_ssh, args.ssh_ready_timeout_sec),
            proxy_user=proxy_user,
        )
        ssh_host = target["host"]
        ssh_port = target["port"]
        ssh_user = target["user"]
        print(json.dumps({"event": "using_ssh_endpoint", **target}, indent=2))

    return pod_id, ssh_host, ssh_port, ssh_user


def cmd_upload_gate4(args: argparse.Namespace) -> int:
    upload_path = Path(__file__).resolve().parent / "runpod_upload_gate4.sh"
    if not upload_path.exists():
        raise SystemExit(f"missing upload script: {upload_path}")

    proxy_user = args.proxy_user or None
    if not args.pod_id and not args.deploy:
        raise SystemExit("upload-gate4 requires --pod-id or --deploy")

    pod_id, ssh_host, ssh_port, ssh_user = _resolve_train_pod_ssh(args, proxy_user=proxy_user)
    return _ssh_run_script(
        args.host_alias,
        upload_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=args.timeout_sec,
        extra_env={"KEEP_LOCAL": str(args.keep_local)},
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=bool(args.deploy or args.pod_id),
    )


def cmd_train_gate4(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve().parent / "runpod_train_gate4.sh"
    if not script_path.exists():
        raise SystemExit(f"missing train script: {script_path}")

    proxy_user = args.proxy_user or None
    pod_id, ssh_host, ssh_port, ssh_user = _resolve_train_pod_ssh(args, proxy_user=proxy_user)

    if args.upload_first:
        upload_path = Path(__file__).resolve().parent / "runpod_upload_gate4.sh"
        if not upload_path.exists():
            raise SystemExit(f"missing upload script: {upload_path}")
        upload_rc = _ssh_run_script(
            args.host_alias,
            upload_path,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            timeout_sec=min(args.timeout_sec, 7200),
            extra_env={"KEEP_LOCAL": str(args.upload_keep_local)},
            ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
            skip_ssh_wait=bool(args.deploy or args.pod_id),
        )
        if upload_rc != 0:
            print(f"HF upload failed (exit {upload_rc})", file=sys.stderr)
            return upload_rc

    extra_env = {
        "PSM_TRAIN_DEVICE": args.device,
        "TARGET_STEPS": str(args.target_steps),
        "RESUME_CHECKPOINT": args.resume_checkpoint,
        "TOKENIZER": args.tokenizer,
        "ABORT_AFTER_STEP": str(args.abort_after_step),
        "GATE4_CURRICULUM_BUILDER": args.curriculum_builder,
        "GATE4_CURRICULUM": "psm-model/data/curriculum/psm-50m-gate4-train-v1.jsonl"
        if args.curriculum_builder == "v1"
        else "psm-model/data/curriculum/psm-50m-full-storage-v4-gate4.jsonl",
        "DIRECT_COPIES": str(args.direct_copies),
        "EXPANDED_COPIES": str(args.expanded_copies),
        "DRILL_ROWS_PER_ACTION": str(args.drill_rows_per_action),
        "DRILL_COPIES": str(args.drill_copies),
        "STRATIFIED_MAX": str(args.stratified_max),
        "IGNORE_EXTRA_COPIES": str(args.ignore_extra_copies),
        "EVAL_EVERY": str(args.eval_every),
    }
    rc = _ssh_run_script(
        args.host_alias,
        script_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=args.timeout_sec,
        extra_env=extra_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=bool(args.deploy or args.pod_id),
    )

    if args.pull_metrics:
        local_metrics = Path(args.pull_metrics)
        scp_rc = _scp_from_pod(
            args.host_alias,
            "/workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-gate4.metrics.jsonl",
            local_metrics.parent,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
        if scp_rc != 0:
            print("warning: could not pull gate4 metrics", file=sys.stderr)

    if args.delete_after and pod_id:
        if rc == 255:
            print(f"Skipping pod delete because training never started (pod {pod_id} left running).", file=sys.stderr)
        else:
            print(f"Deleting pod {pod_id}...")
            _rest("DELETE", f"/pods/{pod_id}")

    return rc


def _deploy_payload(args: argparse.Namespace) -> dict[str, object]:
    start_cmd = ["bash", "-lc", AUTOSTART_CMD] if args.autostart else ["sleep", "infinity"]
    env = {
        "HF_TOKEN": HF_TOKEN_SECRET_REF,
        "PYTHONPATH": "psm-model/src",
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": "chkrishna2001/psm-50m-mixed-v1-run",
        "PSM_HF_DATASET_REPO": "chkrishna2001/psm-50m-action-mixed-v1",
        "PSM_SYNC_GIT": "1",
    }
    payload: dict[str, object] = {
        "name": args.name,
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "supportPublicIp": True,
        "env": env,
    }
    if args.template:
        payload["templateId"] = args.template
    else:
        payload.update(
            {
                "imageName": args.image,
                "volumeInGb": args.volume_gb,
                "containerDiskInGb": args.container_disk_gb,
                "volumeMountPath": "/workspace",
                "ports": ["22/tcp"],
                "dockerStartCmd": start_cmd,
            }
        )
    return payload


def cmd_deploy(args: argparse.Namespace) -> int:
    start_cmd = ["bash", "-lc", AUTOSTART_CMD] if args.autostart else ["sleep", "infinity"]
    env = {
        "HF_TOKEN": HF_TOKEN_SECRET_REF,
        "PYTHONPATH": "psm-model/src",
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": "chkrishna2001/psm-50m-mixed-v1-run",
        "PSM_HF_DATASET_REPO": "chkrishna2001/psm-50m-action-mixed-v1",
        "PSM_SYNC_GIT": "1",
    }
    data = _rest("POST", "/pods", _deploy_payload(args))
    print(json.dumps(data, indent=2))
    if args.wait_ssh:
        pod_id = str(data.get("id", ""))
        if pod_id:
            cmd_wait_ssh(
                argparse.Namespace(
                    pod_id=pod_id,
                    host_alias=SSH_CONFIG_HOST,
                    timeout_sec=args.wait_ssh,
                    poll_sec=5,
                )
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-pods", help="List your pods").set_defaults(func=cmd_list_pods)

    stop = sub.add_parser("stop-pod", help="Stop one pod by id")
    stop.add_argument("pod_id")
    stop.set_defaults(func=cmd_stop_pod)

    sub.add_parser("stop-all", help="Stop all running pods").set_defaults(func=cmd_stop_all)

    delete = sub.add_parser("delete-pod", help="Terminate one pod by id")
    delete.add_argument("pod_id")
    delete.set_defaults(func=cmd_delete_pod)

    sub.add_parser("delete-all", help="Terminate all pods").set_defaults(func=cmd_delete_all)

    tmpl = sub.add_parser("create-template", help="Register REST template for psm-50m-train image")
    tmpl.add_argument("--name", default=DEFAULT_TEMPLATE["name"])
    tmpl.add_argument("--image", default=DEFAULT_TEMPLATE["imageName"])
    tmpl.set_defaults(func=cmd_create_template)

    deploy = sub.add_parser("deploy", help="Deploy pod from image (after docker push)")
    deploy.add_argument("--name", default="psm-train")
    deploy.add_argument(
        "--image",
        default=STOCK_PYTORCH_IMAGE,
        help="Default: stock RunPod PyTorch. Do not use chkrishna2001/psm-50m-train until pushed to Docker Hub.",
    )
    deploy.add_argument(
        "--template",
        default="",
        help="RunPod template id (e.g. mo1fjgnycd). Uses template SSH/image/volume settings.",
    )
    deploy.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    deploy.add_argument("--volume-gb", type=int, default=40)
    deploy.add_argument("--container-disk-gb", type=int, default=20)
    deploy.add_argument(
        "--autostart",
        action="store_true",
        help="Run HF bootstrap + mixed-v2 training on pod start (prefer SSH control instead).",
    )
    deploy.add_argument(
        "--wait-ssh",
        type=int,
        default=0,
        metavar="SEC",
        help="After deploy, poll until direct TCP SSH is ready and update ~/.ssh/config.",
    )
    deploy.set_defaults(func=cmd_deploy)

    ssh_info = sub.add_parser("ssh-info", help="Print fresh SSH targets from RunPod API for a pod")
    ssh_info.add_argument("pod_id")
    ssh_info.add_argument(
        "--proxy-user",
        default="",
        help="Proxy SSH user from Connect tab (e.g. znq...-64411407). Cached after first success.",
    )
    ssh_info.set_defaults(func=cmd_ssh_info)

    ssh_cfg = sub.add_parser("ssh-config", help="Write ~/.ssh/config runpod-psm from fresh pod API data")
    ssh_cfg.add_argument("pod_id")
    ssh_cfg.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    ssh_cfg.add_argument("--proxy-user", default="", help="Proxy SSH user from Connect tab if GraphQL podHostId is unavailable.")
    ssh_cfg.set_defaults(func=cmd_ssh_config)

    wait_ssh = sub.add_parser("wait-ssh", help="Poll pod until SSH accepts connections")
    wait_ssh.add_argument("pod_id")
    wait_ssh.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    wait_ssh.add_argument("--proxy-user", default="", help="Proxy SSH user from Connect tab if needed.")
    wait_ssh.add_argument("--timeout-sec", type=int, default=180)
    wait_ssh.add_argument("--poll-sec", type=int, default=5)
    wait_ssh.set_defaults(func=cmd_wait_ssh)

    eval_gates = sub.add_parser(
        "eval-gates",
        help="Run Gate 2/3 eval on a RunPod GPU (deploy fresh pod or use existing SSH host)",
    )
    eval_gates.add_argument("--deploy", action="store_true", help="Deploy a new eval-only pod before running.")
    eval_gates.add_argument("--pod-id", default="", help="Pod id (for delete-after); filled automatically when --deploy is set.")
    eval_gates.add_argument("--name", default="psm-eval", help="Pod name when --deploy is set.")
    eval_gates.add_argument("--image", default=STOCK_PYTORCH_IMAGE)
    eval_gates.add_argument("--template", default="", help="RunPod template id (optional).")
    eval_gates.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    eval_gates.add_argument("--volume-gb", type=int, default=40)
    eval_gates.add_argument("--container-disk-gb", type=int, default=20)
    eval_gates.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    eval_gates.add_argument(
        "--proxy-user",
        default="",
        help="Proxy SSH user from Connect tab (pod_id-suffix@ssh.runpod.io). Required if GraphQL podHostId is unavailable.",
    )
    eval_gates.add_argument("--device", default="cuda", help="Eval device passed to psm_model (cuda recommended on pod).")
    eval_gates.add_argument("--expanded", action="store_true", help="Also run full-model eval on expanded probe (920 rows).")
    eval_gates.add_argument(
        "--wait-ssh",
        type=int,
        default=180,
        metavar="SEC",
        help="When --deploy: poll until direct TCP SSH is ready.",
    )
    eval_gates.add_argument("--timeout-sec", type=int, default=7200, help="SSH eval session timeout.")
    eval_gates.add_argument(
        "--ssh-ready-timeout-sec",
        type=int,
        default=300,
        help="After deploy: retry SSH until shell accepts connections.",
    )
    eval_gates.add_argument(
        "--pull-reports",
        default="psm-model/checkpoints/gate-eval",
        help="Local directory to scp gate-eval JSON reports into (empty to skip).",
    )
    eval_gates.add_argument("--delete-after", action="store_true", help="Delete the pod after eval (use with --deploy).")
    eval_gates.set_defaults(func=cmd_eval_gates)

    train_gate4 = sub.add_parser(
        "train-gate4",
        help="Deploy (optional) and run Gate 4 full-model training on RunPod GPU",
    )
    train_gate4.add_argument("--deploy", action="store_true", help="Deploy a new training pod before running.")
    train_gate4.add_argument("--pod-id", default="", help="Existing pod id (skip deploy).")
    train_gate4.add_argument("--name", default="psm-train-gate4", help="Pod name when --deploy is set.")
    train_gate4.add_argument("--image", default=STOCK_PYTORCH_IMAGE)
    train_gate4.add_argument("--template", default="", help="RunPod template id (optional).")
    train_gate4.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    train_gate4.add_argument("--volume-gb", type=int, default=40)
    train_gate4.add_argument("--container-disk-gb", type=int, default=20)
    train_gate4.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    train_gate4.add_argument(
        "--proxy-user",
        default="",
        help="Proxy SSH user from Connect tab (pod_id-suffix@ssh.runpod.io).",
    )
    train_gate4.add_argument("--device", default="cuda")
    train_gate4.add_argument(
        "--target-steps",
        type=int,
        default=32000,
        help="Absolute training step target (v1 default: 36000 = +13200 from step 22800).",
    )
    train_gate4.add_argument(
        "--resume-checkpoint",
        default="psm-model/checkpoints/real-v3-50m-full-v2-step-022800.pt",
        help="Gate 3 pass checkpoint; clean base for gate4-train-v1 curriculum.",
    )
    train_gate4.add_argument(
        "--tokenizer",
        default="psm-model/checkpoints/real-v3-50m-full-v2-step-022800.tokenizer.json",
    )
    train_gate4.add_argument("--upload-first", action="store_true", help="Sync checkpoints to HF before training.")
    train_gate4.add_argument(
        "--upload-keep-local",
        type=int,
        default=1,
        help="How many latest step checkpoints to upload (default 1 = latest only).",
    )
    train_gate4.add_argument(
        "--curriculum-builder",
        choices=("v1", "legacy"),
        default="v1",
        help="v1 = expanded-full dominant (production path); legacy = full-storage base dilution.",
    )
    train_gate4.add_argument("--direct-copies", type=int, default=500)
    train_gate4.add_argument("--expanded-copies", type=int, default=40, help="v1: copies per expanded-probe row.")
    train_gate4.add_argument("--drill-rows-per-action", type=int, default=120)
    train_gate4.add_argument("--drill-copies", type=int, default=25)
    train_gate4.add_argument("--stratified-max", type=int, default=2500)
    train_gate4.add_argument("--ignore-extra-copies", type=int, default=6, help="legacy builder only.")
    train_gate4.add_argument("--eval-every", type=int, default=200)
    train_gate4.add_argument("--abort-after-step", type=int, default=30000)
    train_gate4.add_argument("--wait-ssh", type=int, default=300, metavar="SEC")
    train_gate4.add_argument("--timeout-sec", type=int, default=28800, help="SSH training session timeout (8h default).")
    train_gate4.add_argument("--ssh-ready-timeout-sec", type=int, default=420)
    train_gate4.add_argument(
        "--pull-metrics",
        default="",
        help="Local path to directory for metrics jsonl pull after training (empty to skip).",
    )
    train_gate4.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete pod after training completes (not recommended — upload checkpoints first).",
    )
    train_gate4.set_defaults(func=cmd_train_gate4)

    upload_gate4 = sub.add_parser("upload-gate4", help="Upload latest Gate 4 checkpoints to HF")
    upload_gate4.add_argument("--pod-id", default="", help="Existing pod id.")
    upload_gate4.add_argument("--deploy", action="store_true")
    upload_gate4.add_argument("--name", default="psm-train-gate4")
    upload_gate4.add_argument("--image", default=STOCK_PYTORCH_IMAGE)
    upload_gate4.add_argument("--template", default="")
    upload_gate4.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    upload_gate4.add_argument("--volume-gb", type=int, default=40)
    upload_gate4.add_argument("--container-disk-gb", type=int, default=20)
    upload_gate4.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    upload_gate4.add_argument("--proxy-user", default="")
    upload_gate4.add_argument("--wait-ssh", type=int, default=120)
    upload_gate4.add_argument("--ssh-ready-timeout-sec", type=int, default=120)
    upload_gate4.add_argument("--timeout-sec", type=int, default=7200)
    upload_gate4.add_argument("--keep-local", type=int, default=1, help="Upload N latest step checkpoints.")
    upload_gate4.set_defaults(func=cmd_upload_gate4)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
