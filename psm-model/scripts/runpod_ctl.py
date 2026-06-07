#!/usr/bin/env python3
"""RunPod pod/template helpers. Set RUNPOD_API_KEY (e.g. from `o runpodkey`)."""

from __future__ import annotations

import argparse
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
# RunPod proxy SSH user suffix: {pod_id}-{suffix}@ssh.runpod.io (set RUNPOD_SSH_PROXY_SUFFIX if it changes).
SSH_PROXY_SUFFIX = os.environ.get("RUNPOD_SSH_PROXY_SUFFIX", "64411407")

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


def _pod_ssh_target(pod: dict) -> dict[str, str]:
    pod_id = str(pod.get("id", ""))
    public_ip = str(pod.get("publicIp") or "").strip()
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22") or port_mappings.get(22)
    if public_ip and ssh_port:
        return {
            "mode": "direct-tcp",
            "host": public_ip,
            "port": str(ssh_port),
            "user": "root",
            "command": f"ssh -i {SSH_KEY_PATH} root@{public_ip} -p {ssh_port}",
        }
    return {
        "mode": "proxy",
        "host": "ssh.runpod.io",
        "port": "22",
        "user": f"{pod_id}-{SSH_PROXY_SUFFIX}",
        "command": f"ssh -i {SSH_KEY_PATH} {pod_id}-{SSH_PROXY_SUFFIX}@ssh.runpod.io",
    }


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
    for line in lines:
        if line.strip() == f"Host {host_alias}":
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


def cmd_ssh_config(args: argparse.Namespace) -> int:
    pods = _rest("GET", "/pods")
    pod = next((item for item in pods if item.get("id") == args.pod_id), None)
    if pod is None:
        raise SystemExit(f"pod not found: {args.pod_id}")
    target = _pod_ssh_target(pod)
    proxy = _proxy_ssh_target(str(pod.get("id", "")))
    config_path = _write_ssh_config(args.host_alias, target, proxy_target=proxy)
    print(
        json.dumps(
            {
                "pod_id": pod.get("id"),
                "pod_name": pod.get("name"),
                "ssh_mode": target["mode"],
                "ssh_command": target["command"],
                "ssh_config": str(config_path),
                "host_alias": args.host_alias,
                "public_ip": pod.get("publicIp"),
                "port_mappings": pod.get("portMappings"),
            },
            indent=2,
        )
    )
    return 0


def cmd_wait_ssh(args: argparse.Namespace) -> int:
    import time

    deadline = time.time() + args.timeout_sec
    last: dict | None = None
    while time.time() < deadline:
        pod = _rest("GET", f"/pods/{args.pod_id}")
        last = pod
        target = _pod_ssh_target(pod)
        if target["mode"] == "direct-tcp":
            cmd_ssh_config(argparse.Namespace(pod_id=args.pod_id, host_alias=args.host_alias))
            return 0
        time.sleep(args.poll_sec)
    print(json.dumps({"event": "timeout", "pod": last}, indent=2), file=sys.stderr)
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


def _proxy_ssh_target(pod_id: str) -> dict[str, str]:
    return {
        "mode": "proxy",
        "host": "ssh.runpod.io",
        "port": "22",
        "user": f"{pod_id}-{SSH_PROXY_SUFFIX}",
        "command": f"ssh -i {SSH_KEY_PATH} {pod_id}-{SSH_PROXY_SUFFIX}@ssh.runpod.io",
    }


def _ssh_probe(target: dict[str, str]) -> bool:
    probe = subprocess.run(
        [
            SSH_BIN,
            "-tt",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *_ssh_endpoint(
                SSH_CONFIG_HOST,
                host=target["host"],
                port=target["port"],
                user=target["user"],
            ),
            "echo",
            "ssh-ready",
        ],
        capture_output=True,
        text=True,
        timeout=25,
    )
    return probe.returncode == 0 and "ssh-ready" in probe.stdout


def _wait_pod_ssh_endpoint(pod_id: str, *, timeout_sec: int = 420, poll_sec: int = 10) -> dict[str, str]:
    import time

    deadline = time.time() + timeout_sec
    last: dict | None = None
    while time.time() < deadline:
        pod = _rest("GET", f"/pods/{pod_id}")
        last = pod
        cmd_ssh_config(argparse.Namespace(pod_id=pod_id, host_alias=SSH_CONFIG_HOST))
        candidates = [_proxy_ssh_target(pod_id), _pod_ssh_target(pod)]
        for target in candidates:
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
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "StrictHostKeyChecking=accept-new",
                *_ssh_endpoint(host_alias, host=host, port=port, user=user),
                "echo",
                "ssh-ready",
            ],
            capture_output=True,
            text=True,
            timeout=20,
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
    env_prefix = ""
    if extra_env:
        parts = [f"{key}={value}" for key, value in extra_env.items()]
        env_prefix = " ".join(parts) + " "
    command = f"{env_prefix}bash -s"
    with script_path.open("r", encoding="utf-8") as stdin_file:
        result = subprocess.run(
            [SSH_BIN, "-tt", "-i", SSH_KEY_PATH, *_ssh_endpoint(host_alias, host=host, port=port, user=user), command],
            stdin=stdin_file,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result.returncode


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
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        return result.returncode
    result = subprocess.run(
        [SCP_BIN, "-r", "-i", SSH_KEY_PATH, remote_target, str(local_path)],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result.returncode


def cmd_eval_gates(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve().parent / "runpod_eval_gates.sh"
    if not script_path.exists():
        raise SystemExit(f"missing eval script: {script_path}")

    pod_id = args.pod_id
    ssh_host: str | None = None
    ssh_port: str | None = None
    ssh_user = "root"
    if args.pod_id and not args.deploy:
        target = _wait_pod_ssh_endpoint(args.pod_id, timeout_sec=max(args.wait_ssh, args.ssh_ready_timeout_sec))
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
        target = _wait_pod_ssh_endpoint(pod_id, timeout_sec=max(args.wait_ssh, args.ssh_ready_timeout_sec))
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
        skip_ssh_wait=bool(args.deploy),
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

    ssh_cfg = sub.add_parser("ssh-config", help="Write ~/.ssh/config runpod-psm from pod TCP/proxy endpoints")
    ssh_cfg.add_argument("pod_id")
    ssh_cfg.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    ssh_cfg.set_defaults(func=cmd_ssh_config)

    wait_ssh = sub.add_parser("wait-ssh", help="Poll pod until direct TCP SSH is mapped")
    wait_ssh.add_argument("pod_id")
    wait_ssh.add_argument("--host-alias", default=SSH_CONFIG_HOST)
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
