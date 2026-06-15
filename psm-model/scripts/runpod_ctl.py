#!/usr/bin/env python3
"""RunPod pod/template helpers. Set RUNPOD_API_KEY (e.g. from `o runpodkey`)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"
# api.runpod.io sits behind Cloudflare; default Python-urllib UA gets 403 error 1010.
GRAPHQL_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SSH_CONFIG_HOST = "runpod-psm"
SSH_BIN = "ssh.exe" if os.name == "nt" else "ssh"
SCP_BIN = "scp.exe" if os.name == "nt" else "scp"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")
SSH_CACHE_PATH = Path(os.environ.get("RUNPOD_SSH_CACHE", "psm-model/checkpoints/.runpod-ssh-cache.json"))

# RunPod secret HF_TOKEN_C (subbu83) → injected as HF_TOKEN env at pod start.
HF_TOKEN_SECRET_REF = "{{ RUNPOD_SECRET_HF_TOKEN_C }}"

# Model checkpoints (private storage). Dataset/bootstrap stays on chkrishna2001 until migrated.
DEFAULT_HF_MODEL_REPO = "subbu83/psm-50m-mixed-v1-run"
DEFAULT_HF_DATASET_REPO = "chkrishna2001/psm-50m-action-mixed-v1"

# Custom image chkrishna2001/psm-50m-train:latest is NOT on Docker Hub — use stock PyTorch only.
STOCK_PYTORCH_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# 50M trains at batch-size 1 (~3–6 GiB VRAM). 3090 + modest volume is enough; 4090 is overkill.
DEFAULT_GPU = "NVIDIA RTX A5000"
DEFAULT_VOLUME_GB = 20
DEFAULT_CONTAINER_DISK_GB = 10
DEFAULT_MIN_VRAM_GB = 12

# Cheapest-first preference order for --auto-gpu (GraphQL stockStatus must not be None).
PSM_GPU_PREFERENCES = (
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA GeForce RTX 4090",
)

GPU_TYPES_QUERY = """
query {
  gpuTypes {
    id
    displayName
    memoryInGb
    lowestPrice(input: { gpuCount: 1, secureCloud: true }) {
      stockStatus
      uninterruptablePrice
      availableGpuCounts
    }
  }
}
"""

DEFAULT_TEMPLATE = {
    "name": "psm-50m-train",
    "imageName": STOCK_PYTORCH_IMAGE,
    "containerDiskInGb": DEFAULT_CONTAINER_DISK_GB,
    "volumeInGb": DEFAULT_VOLUME_GB,  # 20GB OK with HF sync + keep-local=2
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
    "dockerStartCmd": ["sleep", "infinity"],
    "env": {
        "HF_TOKEN": HF_TOKEN_SECRET_REF,
        "PYTHONPATH": "psm-model/src",
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": DEFAULT_HF_MODEL_REPO,
        "PSM_HF_DATASET_REPO": DEFAULT_HF_DATASET_REPO,
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
        headers={
            "Content-Type": "application/json",
            "User-Agent": GRAPHQL_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and "1010" in detail:
            raise SystemExit(
                "RunPod GraphQL blocked by Cloudflare (error 1010). This is not an API-key permission "
                "issue — urllib's default User-Agent is rejected. Update runpod_ctl.py or retry with "
                "a current version that sets a browser User-Agent on GraphQL requests."
            ) from exc
        if exc.code == 403:
            raise SystemExit(
                "RunPod GraphQL returned 403. If your key has GraphQL Read/Write enabled, the block "
                f"may still be Cloudflare or account scope. Response: {detail[:300]}"
            ) from exc
        raise
    if body.get("errors"):
        raise RuntimeError(json.dumps(body["errors"], indent=2))
    return body["data"]


def _fetch_gpu_types() -> list[dict[str, object]]:
    data = _graphql(GPU_TYPES_QUERY)
    rows = data.get("gpuTypes")
    return list(rows) if isinstance(rows, list) else []


def _gpu_availability_row(row: dict[str, object]) -> dict[str, object]:
    price = row.get("lowestPrice") if isinstance(row.get("lowestPrice"), dict) else {}
    return {
        "id": row.get("id"),
        "displayName": row.get("displayName"),
        "memoryInGb": row.get("memoryInGb"),
        "stockStatus": price.get("stockStatus"),
        "pricePerHr": price.get("uninterruptablePrice"),
        "availableGpuCounts": price.get("availableGpuCounts"),
    }


def pick_gpu_from_preferences(
    preferences: list[str] | tuple[str, ...],
    *,
    min_vram_gb: int = DEFAULT_MIN_VRAM_GB,
    gpu_types: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    available = _fetch_gpu_types() if gpu_types is None else gpu_types
    by_id = {str(row.get("id")): row for row in available if isinstance(row.get("id"), str)}
    rejected: list[dict[str, object]] = []
    for gpu_id in preferences:
        row = by_id.get(gpu_id)
        if row is None:
            rejected.append({"id": gpu_id, "reason": "not_listed"})
            continue
        memory_gb = int(row.get("memoryInGb") or 0)
        if memory_gb < min_vram_gb:
            rejected.append({"id": gpu_id, "reason": "insufficient_vram", "memoryInGb": memory_gb})
            continue
        summary = _gpu_availability_row(row)
        if summary.get("stockStatus") == "None":
            rejected.append({"id": gpu_id, "reason": "no_stock", "stockStatus": summary.get("stockStatus")})
            continue
        return summary
    raise SystemExit(
        "No GPU from preference list is available right now. "
        f"Checked: {list(preferences)}. Rejected: {json.dumps(rejected, sort_keys=True)}"
    )


def _parse_gpu_preferences(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return PSM_GPU_PREFERENCES
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _apply_auto_gpu(args: argparse.Namespace) -> None:
    if not getattr(args, "auto_gpu", False):
        return
    preferences = _parse_gpu_preferences(getattr(args, "gpu_preferences", None))
    min_vram_gb = int(getattr(args, "min_vram_gb", DEFAULT_MIN_VRAM_GB))
    picked = pick_gpu_from_preferences(preferences, min_vram_gb=min_vram_gb)
    print(json.dumps({"event": "auto_gpu_picked", **picked}, indent=2), file=sys.stderr)
    args.gpu = str(picked["id"])


def _add_auto_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--auto-gpu",
        action="store_true",
        help="Query RunPod GraphQL stockStatus and pick the first available GPU from the preference list.",
    )
    parser.add_argument(
        "--gpu-preferences",
        default="",
        help=f"Comma-separated GPU ids for --auto-gpu (default: {', '.join(PSM_GPU_PREFERENCES)}).",
    )
    parser.add_argument(
        "--min-vram-gb",
        type=int,
        default=DEFAULT_MIN_VRAM_GB,
        help="Minimum GPU VRAM when using --auto-gpu (50M @ batch 1 needs ~12 GiB).",
    )


def _rest(method: str, path: str, data: dict | None = None) -> dict | list:
    ok, body = _rest_try(method, path, data)
    if not ok:
        raise SystemExit(body)
    return body


def _rest_try(method: str, path: str, data: dict | None = None) -> tuple[bool, dict | list | str]:
    url = f"{REST_URL}{path}"
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    body_bytes = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return True, json.loads(raw) if raw else {"status": resp.status}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        return False, f"RunPod REST {method} {path} failed ({exc.code}): {detail}"


def cmd_list_pods(_: argparse.Namespace) -> int:
    pods = _rest("GET", "/pods")
    print(json.dumps(pods, indent=2))
    return 0


def cmd_list_gpus(args: argparse.Namespace) -> int:
    gpu_types = _fetch_gpu_types()
    if args.all:
        rows = [_gpu_availability_row(row) for row in gpu_types if isinstance(row, dict)]
        rows.sort(
            key=lambda row: (
                row.get("stockStatus") in (None, "None"),
                row.get("pricePerHr") if isinstance(row.get("pricePerHr"), (int, float)) else 9999,
                str(row.get("id")),
            )
        )
    else:
        preferences = _parse_gpu_preferences(args.gpu_preferences or None)
        by_id = {str(row.get("id")): row for row in gpu_types if isinstance(row.get("id"), str)}
        rows = []
        for gpu_id in preferences:
            row = by_id.get(gpu_id)
            if row is None:
                rows.append({"id": gpu_id, "missing": True})
            else:
                rows.append(_gpu_availability_row(row))
    print(json.dumps({"gpus": rows, "defaults": {"gpu": DEFAULT_GPU, "volume_gb": DEFAULT_VOLUME_GB}}, indent=2, sort_keys=True))
    return 0


def cmd_pick_gpu(args: argparse.Namespace) -> int:
    preferences = _parse_gpu_preferences(args.gpu_preferences or None)
    picked = pick_gpu_from_preferences(preferences, min_vram_gb=args.min_vram_gb)
    print(json.dumps({"picked": picked, "volume_gb": args.volume_gb, "container_disk_gb": args.container_disk_gb}, indent=2, sort_keys=True))
    return 0


def cmd_stop_pod(args: argparse.Namespace) -> int:
    result = _rest("POST", f"/pods/{args.pod_id}/stop")
    print(json.dumps(result, indent=2))
    return 0


def _delete_pod_unless_kept(args: argparse.Namespace, pod_id: str, rc: int, *, job: str) -> None:
    if not pod_id or getattr(args, "keep_pod", False):
        return
    if rc == 255:
        print(f"Skipping pod delete because {job} never started (pod {pod_id} left running).", file=sys.stderr)
        return
    if getattr(args, "force_delete_pod", False):
        print(f"Force-deleting pod {pod_id} after {job} (--force-delete-pod).", file=sys.stderr)
        _rest("DELETE", f"/pods/{pod_id}")
        return
    missing = _gate4_missing_hf_checkpoints(getattr(args, "required_hf_steps", None))
    if missing:
        print(
            f"BLOCKING pod delete after {job}: checkpoint .pt not on HF: {', '.join(missing)}",
            file=sys.stderr,
        )
        print(f"Pod {pod_id} left running. Run upload-gate4 --pod-id {pod_id} then delete manually.", file=sys.stderr)
        return
    print(f"Deleting pod {pod_id} after {job}...")
    _rest("DELETE", f"/pods/{pod_id}")


def _hf_model_file_exists(repo_id: str, path_in_repo: str) -> bool:
    try:
        from huggingface_hub import file_exists

        # Signature is file_exists(repo_id, filename); passing the path first
        # raised TypeError and made every check a false negative.
        return bool(file_exists(repo_id, path_in_repo, repo_type="model"))
    except Exception:
        return False


def _gate4_best_steps_for_verify() -> list[int]:
    registry_path = Path(__file__).resolve().parents[1] / "checkpoints" / "gate4-checkpoint-registry.json"
    if not registry_path.exists():
        return []
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        best = registry.get("best") or {}
        if best.get("step") is not None:
            return [int(best["step"])]
    except json.JSONDecodeError:
        pass
    return []


def _gate4_missing_hf_checkpoints(required_steps: list[int] | None) -> list[str]:
    """Return remote paths for required step checkpoints missing from HF (pt + tokenizer + meta)."""
    repo_id = os.environ.get("PSM_HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO)
    run_stem = "real-v3-50m-full-v2"
    steps: set[int] = set(required_steps or [])
    registry_path = Path(__file__).resolve().parents[1] / "checkpoints" / "gate4-checkpoint-registry.json"
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            best = registry.get("best") or {}
            if best.get("step") is not None:
                steps.add(int(best["step"]))
        except json.JSONDecodeError:
            pass
    if not steps:
        return []
    try:
        from psm_model.gate4_checkpoint_registry import verify_hf_steps

        return verify_hf_steps(repo_id=repo_id, run_stem=run_stem, steps=steps)
    except Exception:
        missing: list[str] = []
        for step in sorted(steps):
            for suffix in (".pt", ".tokenizer.json", ".meta.json"):
                remote = f"psm-model/checkpoints/{run_stem}-step-{step:06d}{suffix}"
                if not _hf_model_file_exists(repo_id, remote):
                    missing.append(remote)
        return missing


def cmd_delete_pod(args: argparse.Namespace) -> int:
    missing = _gate4_missing_hf_checkpoints(getattr(args, "required_hf_steps", None))
    if missing and not getattr(args, "force_delete_pod", False):
        print(
            f"BLOCKING pod delete: checkpoint files not on HF: {', '.join(missing)}",
            file=sys.stderr,
        )
        print("Run upload-gate4 or use --force-delete-pod to override.", file=sys.stderr)
        return 1
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
    missing = _gate4_missing_hf_checkpoints(None)
    if missing:
        print(
            f"BLOCKING delete-all: checkpoint files not on HF: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
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
        "PSM 50M training image. Requires RunPod secret HF_TOKEN_C "
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
    ok = (
        probe.returncode == 0
        and "ssh-ready" in probe.stdout
        and "container not found" not in (probe.stdout + probe.stderr).lower()
    )
    if ok and target.get("mode") == "proxy" and target.get("pod_id"):
        _ssh_cache_save(str(target["pod_id"]), str(target["user"]))
    return ok


def _gpu_deploy_candidates(args: argparse.Namespace) -> tuple[str, ...]:
    if getattr(args, "gpu_preferences", ""):
        return _parse_gpu_preferences(args.gpu_preferences)
    if getattr(args, "auto_gpu", False):
        return PSM_GPU_PREFERENCES
    gpu = str(getattr(args, "gpu", "") or DEFAULT_GPU)
    return (gpu,)


def _create_pod_with_fallback(args: argparse.Namespace) -> dict:
    candidates = _gpu_deploy_candidates(args)
    rejected: list[dict[str, str]] = []
    for cloud_type in ("SECURE", "COMMUNITY"):
        for gpu in candidates:
            deploy_args = argparse.Namespace(
                name=args.name,
                image=getattr(args, "image", STOCK_PYTORCH_IMAGE),
                template=getattr(args, "template", ""),
                gpu=gpu,
                volume_gb=getattr(args, "volume_gb", DEFAULT_VOLUME_GB),
                container_disk_gb=getattr(args, "container_disk_gb", DEFAULT_CONTAINER_DISK_GB),
                autostart=getattr(args, "autostart", False),
                wait_ssh=0,
            )
            ok, result = _rest_try("POST", "/pods", _deploy_payload(deploy_args, cloud_type=cloud_type))
            if ok and isinstance(result, dict) and result.get("id"):
                print(
                    json.dumps(
                        {
                            "event": "pod_created",
                            "cloudType": cloud_type,
                            "gpu": gpu,
                            "id": result.get("id"),
                        },
                        indent=2,
                    )
                )
                return result
            rejected.append(
                {
                    "cloudType": cloud_type,
                    "gpu": gpu,
                    "error": str(result)[:240] if isinstance(result, str) else "unknown",
                }
            )
    raise SystemExit(
        "No RunPod GPU could be provisioned (tried SECURE + COMMUNITY). "
        f"Attempts: {json.dumps(rejected, indent=2)}"
    )


def _proxy_ssh_triplet(proxy_user: str) -> tuple[str, str, str]:
    """RunPod proxy: user is pod_id-suffix, host is ssh.runpod.io."""
    return "ssh.runpod.io", "22", proxy_user.split("@", 1)[0]


def _resolve_pod_ssh(
    args: argparse.Namespace,
    *,
    proxy_user: str | None,
) -> tuple[str, str | None, str | None, str]:
    """Prefer --proxy-user (instant). Fall back to _wait_pod_ssh_endpoint (slow, often flaky)."""
    pod_id = str(getattr(args, "pod_id", "") or "")
    if pod_id and not getattr(args, "deploy", False) and proxy_user and "@" in proxy_user:
        ssh_host, ssh_port, ssh_user = _proxy_ssh_triplet(proxy_user)
        print(
            json.dumps(
                {
                    "event": "using_ssh_endpoint",
                    "mode": "proxy",
                    "host": ssh_host,
                    "port": ssh_port,
                    "user": ssh_user,
                },
                indent=2,
            )
        )
        return pod_id, ssh_host, ssh_port, ssh_user

    if pod_id and not getattr(args, "deploy", False):
        target = _wait_pod_ssh_endpoint(
            pod_id,
            timeout_sec=max(getattr(args, "wait_ssh", 180), getattr(args, "ssh_ready_timeout_sec", 300)),
            proxy_user=proxy_user,
        )
        print(json.dumps({"event": "using_ssh_endpoint", **target}, indent=2))
        return pod_id, target["host"], target["port"], target["user"]

    if getattr(args, "deploy", False):
        data = _create_pod_with_fallback(args)
        print(json.dumps(data, indent=2))
        pod_id = str(data.get("id", ""))
        if not pod_id:
            raise SystemExit("deploy did not return pod id")
        target = _wait_pod_ssh_endpoint(
            pod_id,
            timeout_sec=max(getattr(args, "wait_ssh", 180), getattr(args, "ssh_ready_timeout_sec", 300)),
            proxy_user=proxy_user,
        )
        print(json.dumps({"event": "using_ssh_endpoint", **target}, indent=2))
        return pod_id, target["host"], target["port"], target["user"]

    return pod_id, None, None, "root"


def _verify_pod_job(
    host_alias: str,
    *,
    host: str | None,
    port: str | None,
    user: str,
    tmux_session: str,
    process_pattern: str,
    label: str,
) -> bool:
    """Confirm tmux session + GPU process exist within ~15s. Prevents silent idle-pod launches."""
    import time

    checks = (
        f"tmux has-session -t {tmux_session} 2>/dev/null && echo TMUX_OK || echo TMUX_MISSING\n"
        f"pgrep -af '{process_pattern}' | grep -v tmux | head -1 || echo PROC_MISSING\n"
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo GPU_NA\n"
    )
    time.sleep(8)
    attempts = 12 if "eval" in label else 4
    pause_sec = 15 if "eval" in label else 5
    for attempt in range(attempts):
        proc = subprocess.run(
            [
                SSH_BIN,
                "-tt",
                "-i",
                SSH_KEY_PATH,
                "-o",
                "ConnectTimeout=20",
                *_ssh_endpoint(host_alias, host=host, port=port, user=user),
                "bash",
                "-s",
            ],
            input=f"{checks}exit\n",
            capture_output=True,
            text=True,
            timeout=45,
            encoding="utf-8",
            errors="replace",
        )
        out = proc.stdout
        if "TMUX_OK" in out and "PROC_MISSING" not in out:
            for line in out.splitlines():
                if any(k in line for k in ("TMUX_", "PROC_", "%", "MiB", process_pattern)):
                    print(f"verify {label}: {line.strip()}", file=sys.stderr)
            return True
        time.sleep(pause_sec)
    print(f"verify {label}: FAILED — no tmux/{process_pattern} on pod", file=sys.stderr)
    return False


def _probe_pod_training(
    host_alias: str,
    *,
    host: str | None,
    port: str | None,
    user: str,
    tmux_session: str,
    process_pattern: str,
    train_log: str,
    timeout_sec: int = 60,
) -> tuple[dict[str, str], list[str]]:
    """Quick non-blocking pod probe (single SSH, hard timeout). Returns status dict + log tail."""
    probe = f"""
tmux has-session -t {tmux_session} 2>/dev/null && echo PSM_TMUX=OK || echo PSM_TMUX=MISSING
pgrep -af '{process_pattern}' 2>/dev/null | grep -v tmux | head -1 | sed 's/^/PSM_PROC=/' || echo PSM_PROC=MISSING
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | awk -F',' '{{printf "PSM_GPU_UTIL=%s\\nPSM_GPU_MEM_MIB=%s\\n", $1, $2}}' || echo PSM_GPU_UTIL=NA
python3 -c "import torch; print('PSM_CUDA_OK=1' if torch.cuda.is_available() else 'PSM_CUDA_OK=0')" 2>/dev/null || echo PSM_CUDA_OK=0
test -f /tmp/psm-gate5.done && echo PSM_TRAIN_DONE=1 || echo PSM_TRAIN_DONE=0
test -f /tmp/psm-gate5-dual-eval.done && echo PSM_EVAL_DONE=1 || echo PSM_EVAL_DONE=0
test -f /tmp/psm-gate4.done && echo PSM_GATE4_TRAIN_DONE=1 || echo PSM_GATE4_TRAIN_DONE=0
ls -t /workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-step-*.pt 2>/dev/null | head -1 | sed 's|.*/||;s/.pt$//' | sed 's/^/PSM_LATEST_CKPT=/' || echo PSM_LATEST_CKPT=none
ls /workspace/PSM/psm-model/checkpoints/gate-eval/gate5-dual-step-*.json 2>/dev/null | tail -1 | xargs -r basename | sed 's/^/PSM_DUAL_EVAL=/' || echo PSM_DUAL_EVAL=none
tmux has-session -t psm-gate5-sync 2>/dev/null && echo PSM_SYNC_TMUX=OK || echo PSM_SYNC_TMUX=MISSING
if [[ -f '{train_log}' ]]; then tail -3 '{train_log}' | while IFS= read -r line; do echo "PSM_LOG=$line"; done; else echo PSM_LOG=missing; fi
"""
    proc = subprocess.run(
        [
            SSH_BIN,
            "-tt",
            "-i",
            SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=15",
            *_ssh_endpoint(host_alias, host=host, port=port, user=user),
            "bash",
            "-s",
        ],
        input=f"{probe}exit\n",
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="replace",
    )
    status: dict[str, str] = {"ssh_rc": str(proc.returncode)}
    log_lines: list[str] = []
    for line in (proc.stdout or "").splitlines():
        if line.startswith("PSM_"):
            key, _, value = line.partition("=")
            if key == "PSM_LOG":
                log_lines.append(value.strip())
            else:
                status[key] = value.strip()
    if proc.stderr:
        status["ssh_stderr"] = proc.stderr.strip()[-500:]
    return status, log_lines


def cmd_verify_pod(args: argparse.Namespace) -> int:
    proxy_user = args.proxy_user or None
    pod_id, ssh_host, ssh_port, ssh_user = _resolve_pod_ssh(args, proxy_user=proxy_user)
    status, log_lines = _probe_pod_training(
        args.host_alias,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        tmux_session=args.tmux_session,
        process_pattern=args.process_pattern,
        train_log=args.train_log,
        timeout_sec=args.timeout_sec,
    )
    tmux_ok = status.get("PSM_TMUX") == "OK"
    proc_ok = status.get("PSM_PROC", "MISSING") != "MISSING"
    cuda_ok = status.get("PSM_CUDA_OK") == "1"
    gpu_util = 0
    try:
        gpu_util = int(float(status.get("PSM_GPU_UTIL", "0").replace("%", "").strip() or 0))
    except ValueError:
        gpu_util = 0
    gpu_ok = not args.require_gpu or gpu_util >= args.min_gpu_pct
    train_done = status.get("PSM_TRAIN_DONE") == "1"
    eval_done = status.get("PSM_EVAL_DONE") == "1"
    sync_tmux = status.get("PSM_SYNC_TMUX") == "OK"
    gpu_active = cuda_ok and gpu_util >= args.min_gpu_pct
    if proc_ok and cuda_ok and (not args.require_gpu or gpu_ok):
        job_state = "training"
    elif gpu_active and tmux_ok and not proc_ok:
        job_state = "gpu_active"
    elif eval_done and not proc_ok:
        job_state = "eval_finished"
    elif train_done and not proc_ok:
        job_state = "train_finished"
    elif train_done and eval_done:
        job_state = "eval_finished"
    elif not proc_ok and gpu_util < args.min_gpu_pct:
        job_state = "idle_billing" if sync_tmux or tmux_ok else "stopped"
    else:
        job_state = "unknown"
    passed = job_state in ("training", "gpu_active", "eval_finished") and cuda_ok and (
        job_state == "eval_finished" or (tmux_ok and (gpu_ok or not args.require_gpu))
    )
    report = {
        "pod_id": pod_id,
        "passed": passed,
        "job_state": job_state,
        "tmux": status.get("PSM_TMUX"),
        "sync_tmux": sync_tmux,
        "process": status.get("PSM_PROC", "MISSING")[:120],
        "cuda_ok": cuda_ok,
        "gpu_util_pct": gpu_util,
        "gpu_mem_mib": status.get("PSM_GPU_MEM_MIB"),
        "train_done": train_done,
        "eval_done": eval_done,
        "latest_checkpoint": status.get("PSM_LATEST_CKPT"),
        "dual_eval_report": status.get("PSM_DUAL_EVAL"),
        "train_log_tail": log_lines[-3:],
        "idle_billing": job_state in ("train_finished", "idle_billing", "eval_finished"),
    }
    print(json.dumps(report, indent=2))
    if job_state == "train_finished":
        print(
            "TRAIN_FINISHED_IDLE — GPU job done but pod still billing. "
            "Run dual eval or stop/delete pod.",
            file=sys.stderr,
        )
        return 2
    if job_state == "eval_finished":
        print("EVAL_FINISHED — safe to sync HF and delete pod after verify.", file=sys.stderr)
        return 0
    if job_state == "idle_billing":
        print("IDLE_BILLING — no train process, pod still running.", file=sys.stderr)
        if args.stop_on_fail:
            print(f"Stopping pod {pod_id} (idle GPU billing).", file=sys.stderr)
            _rest("POST", f"/pods/{pod_id}/stop")
        return 2
    if not passed:
        print(
            "verify-pod FAILED: "
            f"tmux={tmux_ok} proc={proc_ok} cuda={cuda_ok} gpu>={args.min_gpu_pct}% is {gpu_ok} (saw {gpu_util}%)",
            file=sys.stderr,
        )
        if args.stop_on_fail:
            print(f"Stopping pod {pod_id} (idle GPU billing).", file=sys.stderr)
            _rest("POST", f"/pods/{pod_id}/stop")
        return 1
    return 0


def _push_repo_files_via_tar(
    host_alias: str,
    repo_root: Path,
    rel_paths: list[str],
    remote_root: str,
    *,
    host: str | None,
    port: str | None,
    user: str,
) -> int:
    """Tar-push specific repo files (SCP/subsystem fails on RunPod proxy)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        bundle.mkdir()
        for rel in rel_paths:
            local = repo_root / rel
            if not local.is_file():
                print(f"skip missing artifact: {local}", file=sys.stderr)
                continue
            dest = bundle / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(local.read_bytes())
        if not any(bundle.rglob("*")):
            return 0
        print(f"tar-push {len(rel_paths)} file(s) -> pod:{remote_root}", file=sys.stderr)
        return _ssh_push_dir(host_alias, bundle, remote_root, host=host, port=port, user=user)


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


def _ssh_run_bash(
    host_alias: str,
    command: str,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
    timeout_sec: int = 60,
    skip_ssh_wait: bool = False,
) -> int:
    """Run a one-liner on the pod (proxy-safe)."""
    if not skip_ssh_wait and not _wait_ssh_shell(
        host_alias, host=host, port=port, user=user, timeout_sec=timeout_sec
    ):
        return 255
    proc = subprocess.run(
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
            "-lc",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if proc.stdout:
        _ssh_stream_print(proc.stdout)
    if proc.stderr:
        _ssh_stream_print(proc.stderr)
    return proc.returncode


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
    capture_output: bool = False,
) -> int | tuple[int, str]:
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
    captured: list[str] = []
    while True:
        if proc.poll() is not None:
            remainder = proc.stdout.read()
            if remainder:
                if capture_output:
                    captured.append(remainder)
                _ssh_stream_print(remainder)
            break
        if time.time() > deadline:
            proc.kill()
            print(f"SSH eval timed out after {timeout_sec}s", file=sys.stderr)
            rc = 124
            return (rc, "".join(captured)) if capture_output else rc
        line = proc.stdout.readline()
        if line:
            if capture_output:
                captured.append(line)
            _ssh_stream_print(line)
        elif proc.poll() is not None:
            break
        else:
            time.sleep(0.05)
    rc = proc.returncode if proc.returncode is not None else 124
    return (rc, "".join(captured)) if capture_output else rc


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


def _ssh_push_dir(
    host_alias: str,
    local_path: Path,
    remote_path: str,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
) -> int:
    """Push a local directory to the pod via tar+base64 over bash -s."""
    if not local_path.is_dir():
        print(f"local dir missing: {local_path}", file=sys.stderr)
        return 1
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "w:gz") as archive:
            archive.add(local_path, arcname=".")
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
        print(f"pushing {local_path} ({size_mb:.1f} MB tar) -> {remote_path}", flush=True)
        encoded = base64.b64encode(tmp_path.read_bytes()).decode("ascii")
    finally:
        tmp_path.unlink(missing_ok=True)

    chunk_lines = [encoded[index : index + 120] for index in range(0, len(encoded), 120)]
    heredoc = "\n".join(chunk_lines)
    push_cmd = (
        f"mkdir -p '{remote_path}'\n"
        "cat > /tmp/psm-push.b64 <<'PSM_PUSH_EOF'\n"
        f"{heredoc}\n"
        "PSM_PUSH_EOF\n"
        f"base64 -d /tmp/psm-push.b64 | tar -C '{remote_path}' -xzf -\n"
        "echo PSM_PUSH_OK\n"
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
        input=f"{push_cmd}\nexit\n",
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
        return result.returncode
    combined = f"{result.stdout}\n{result.stderr}"
    if "container not found" in combined.lower():
        print("remote push failed: pod container not running (stop+start pod, do not deploy another)", file=sys.stderr)
        return 2
    if "PSM_PUSH_OK" not in combined:
        print("remote push did not confirm PSM_PUSH_OK", file=sys.stderr)
        if result.stdout.strip():
            print(result.stdout[-400:], file=sys.stderr)
        return 1
    return 0


def _scp_to_pod(
    host_alias: str,
    local_path: Path,
    remote_path: str,
    *,
    host: str | None = None,
    port: str | None = None,
    user: str = "root",
) -> int:
    if not local_path.is_file():
        print(f"local file missing: {local_path}", file=sys.stderr)
        return 1
    remote_parent = str(Path(remote_path).parent).replace("\\", "/")
    mkdir_rc = subprocess.run(
        [
            SSH_BIN,
            "-i",
            SSH_KEY_PATH,
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *_ssh_endpoint(host_alias, host=host, port=port, user=user),
            f"mkdir -p '{remote_parent}'",
        ],
        capture_output=True,
        text=True,
    )
    if mkdir_rc.returncode != 0:
        if mkdir_rc.stderr:
            print(mkdir_rc.stderr, file=sys.stderr, end="" if mkdir_rc.stderr.endswith("\n") else "\n")
        return mkdir_rc.returncode
    if host and port:
        result = subprocess.run(
            [SCP_BIN, "-P", str(port), "-i", SSH_KEY_PATH, str(local_path), f"{user}@{host}:{remote_path}"],
            capture_output=True,
            text=True,
        )
    else:
        result = subprocess.run(
            [SCP_BIN, "-i", SSH_KEY_PATH, str(local_path), f"{host_alias}:{remote_path}"],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0 and result.stderr:
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
    scripts_dir = Path(__file__).resolve().parent
    warm_pod = getattr(args, "warm_pod", True) and bool(args.pod_id) and not args.deploy
    if warm_pod:
        script_path = scripts_dir / "runpod_start_gate4_eval_only.sh"
    else:
        script_path = scripts_dir / "runpod_eval_gates.sh"
    if not script_path.exists():
        raise SystemExit(f"missing eval script: {script_path}")

    proxy_user = args.proxy_user or None
    pod_id, ssh_host, ssh_port, ssh_user = _resolve_pod_ssh(args, proxy_user=proxy_user)

    if args.deploy:
        _apply_auto_gpu(args)
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

    repo_root = Path(__file__).resolve().parents[2]
    scripts_src = repo_root / "psm-model" / "scripts"
    if scripts_src.is_dir():
        print(f"Syncing {scripts_src} -> pod:/workspace/PSM/psm-model/scripts", file=sys.stderr)
        _ssh_push_dir(
            args.host_alias,
            scripts_src,
            "/workspace/PSM/psm-model/scripts",
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )

    extra_env = {
        "PSM_EVAL_DEVICE": args.device,
        "PSM_EVAL_EXPANDED": "1" if args.expanded else "0",
    }
    if args.full_checkpoint:
        extra_env["PSM_EVAL_FULL_CKPT"] = args.full_checkpoint
    start_timeout = 180 if warm_pod else args.timeout_sec
    rc = _ssh_run_script(
        args.host_alias,
        script_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=start_timeout,
        extra_env=extra_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
    )
    if rc == 0 and warm_pod and args.expanded:
        if not _verify_pod_job(
            args.host_alias,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            tmux_session="psm-gate4-eval",
            process_pattern="psm_model.eval_checkpoint",
            label="gate4-eval",
        ):
            return 1

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

    _delete_pod_unless_kept(args, pod_id, rc, job="eval")

    return rc


def _resolve_train_pod_ssh(
    args: argparse.Namespace,
    *,
    proxy_user: str | None,
) -> tuple[str, str | None, str | None, str]:
    return _resolve_pod_ssh(args, proxy_user=proxy_user)


def _parse_resume_checkpoint(upload_output: str) -> tuple[str, str]:
    resume_checkpoint = ""
    resume_step = ""
    for line in upload_output.splitlines():
        if line.startswith("RESUME_CHECKPOINT="):
            resume_checkpoint = line.split("=", 1)[1].strip()
        elif line.startswith("RESUME_STEP="):
            resume_step = line.split("=", 1)[1].strip()
    return resume_checkpoint, resume_step


def cmd_recover_gate4(args: argparse.Namespace) -> int:
    upload_path = Path(__file__).resolve().parent / "runpod_upload_gate4.sh"
    train_path = Path(__file__).resolve().parent / "runpod_train_gate4.sh"
    if not upload_path.exists() or not train_path.exists():
        raise SystemExit("missing upload or train script for recover-gate4")
    proxy_user = args.proxy_user or None
    if not args.pod_id:
        raise SystemExit("recover-gate4 requires --pod-id")
    _, ssh_host, ssh_port, ssh_user = _resolve_train_pod_ssh(
        argparse.Namespace(
            pod_id=args.pod_id,
            deploy=False,
            name="",
            image=STOCK_PYTORCH_IMAGE,
            template="",
            gpu=DEFAULT_GPU,
            volume_gb=DEFAULT_VOLUME_GB,
            container_disk_gb=DEFAULT_CONTAINER_DISK_GB,
            autostart=False,
            wait_ssh=args.wait_ssh,
            ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
            auto_gpu=False,
            gpu_preferences="",
            min_vram_gb=DEFAULT_MIN_VRAM_GB,
        ),
        proxy_user=proxy_user,
    )
    shared_env = {
        "KEEP_LOCAL": str(args.keep_local),
        "TARGET_STEPS": str(args.target_steps),
        "SAVE_EVERY": str(args.save_every),
        "SYNC_INTERVAL_SEC": "600",
    }
    prune_path = Path(__file__).resolve().parent / "runpod_prune_gate4.sh"
    prune_result = _ssh_run_script(
        args.host_alias,
        prune_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=600,
        extra_env={"KEEP_LOCAL": str(args.keep_local)},
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
        capture_output=True,
    )
    if isinstance(prune_result, tuple):
        prune_rc, prune_output = prune_result
    else:
        prune_rc, prune_output = prune_result, ""
    resume_checkpoint, resume_step = _parse_resume_checkpoint(prune_output)

    upload_result = _ssh_run_script(
        args.host_alias,
        upload_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=min(args.timeout_sec, 1800),
        extra_env={**shared_env, "UPLOAD_ALL": "0"},
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
        capture_output=True,
    )
    if isinstance(upload_result, tuple):
        upload_rc, upload_output = upload_result
    else:
        upload_rc, upload_output = upload_result, ""
    upload_resume, upload_step = _parse_resume_checkpoint(upload_output)
    if upload_resume:
        resume_checkpoint, resume_step = upload_resume, upload_step
    if prune_rc != 0 and not resume_checkpoint:
        return prune_rc if isinstance(prune_rc, int) else prune_rc[0]
    if upload_rc != 0 and not resume_checkpoint:
        return upload_rc if isinstance(upload_rc, int) else upload_rc[0]
    if not resume_checkpoint:
        raise SystemExit("upload-gate4 did not report RESUME_CHECKPOINT; cannot resume training")
    if upload_rc != 0:
        print(
            json.dumps(
                {
                    "event": "upload_partial_failure",
                    "upload_rc": upload_rc,
                    "resume_checkpoint": resume_checkpoint,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    tokenizer = f"{resume_checkpoint.removesuffix('.pt')}.tokenizer.json" if resume_checkpoint.endswith(".pt") else ""
    print(
        json.dumps(
            {"event": "recover_resume", "checkpoint": resume_checkpoint, "step": resume_step},
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )

    train_env = {
        **shared_env,
        "RESUME_CHECKPOINT": resume_checkpoint,
        "TOKENIZER": tokenizer,
    }
    train_rc = _ssh_run_script(
        args.host_alias,
        train_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=args.timeout_sec,
        extra_env=train_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
    )
    return train_rc if isinstance(train_rc, int) else train_rc[0]


def cmd_upload_gate4(args: argparse.Namespace) -> int:
    upload_path = Path(__file__).resolve().parent / "runpod_upload_gate4.sh"
    if not upload_path.exists():
        raise SystemExit(f"missing upload script: {upload_path}")

    proxy_user = args.proxy_user or None
    if not args.pod_id and not args.deploy:
        raise SystemExit("upload-gate4 requires --pod-id or --deploy")

    pod_id, ssh_host, ssh_port, ssh_user = _resolve_train_pod_ssh(args, proxy_user=proxy_user)
    upload_env: dict[str, str] = {
        "KEEP_LOCAL": str(args.keep_local),
        "PSM_HF_MODEL_REPO": os.environ.get("PSM_HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO),
        "UPLOAD_ALL": os.environ.get("UPLOAD_ALL", "0"),
        "GATE4_PINNED_STEPS": os.environ.get("GATE4_PINNED_STEPS", "42000"),
    }
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        upload_env["HF_TOKEN"] = hf_token
    return _ssh_run_script(
        args.host_alias,
        upload_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=args.timeout_sec,
        extra_env=upload_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=bool(args.deploy or args.pod_id),
    )


def _apply_micro_train_defaults(args: argparse.Namespace) -> None:
    if args.curriculum_builder != "micro":
        return
    if args.structural_loss_weight == 1.0:
        args.structural_loss_weight = 8.0
    if args.eval_every == 0:
        args.eval_every = 200
    if args.repair_copies == 1:
        args.repair_copies = 12
    if args.direct_copies == 300:
        args.direct_copies = 20
    if not args.eval_report:
        args.eval_report = "psm-model/checkpoints/gate-eval/gate4-full-expanded-step-042000.json"


def cmd_train_gate4(args: argparse.Namespace) -> int:
    scripts_dir = Path(__file__).resolve().parent
    warm_pod = getattr(args, "warm_pod", True) and bool(args.pod_id) and not args.deploy
    if warm_pod:
        script_path = scripts_dir / "runpod_start_gate4_train_only.sh"
    else:
        script_path = scripts_dir / "runpod_train_gate4.sh"
    if not script_path.exists():
        raise SystemExit(f"missing train script: {script_path}")

    _apply_micro_train_defaults(args)

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

    repo_root = Path(__file__).resolve().parents[2]
    if args.sync_src and not warm_pod:
        src_dir = repo_root / "psm-model" / "src"
        remote_src = "/workspace/PSM/psm-model/src"
        print(f"Syncing {src_dir} -> pod:{remote_src}", file=sys.stderr)
        push_rc = _ssh_push_dir(
            args.host_alias,
            src_dir,
            remote_src,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
        if push_rc != 0:
            print(f"warning: src sync failed (exit {push_rc}); pod may use stale git checkout", file=sys.stderr)
    elif args.sync_src and warm_pod:
        print("warm-pod: skipping full src sync (use --no-warm-pod for cold bootstrap)", file=sys.stderr)
    scripts_dir = repo_root / "psm-model" / "scripts"
    if scripts_dir.is_dir():
        remote_scripts = "/workspace/PSM/psm-model/scripts"
        print(f"Syncing {scripts_dir} -> pod:{remote_scripts}", file=sys.stderr)
        scripts_rc = _ssh_push_dir(
            args.host_alias,
            scripts_dir,
            remote_scripts,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
        if scripts_rc != 0:
            print(f"warning: scripts sync failed (exit {scripts_rc})", file=sys.stderr)

    if args.curriculum_builder == "micro":
        micro_files = [rel for rel in (args.eval_report, args.parse_repair, args.curriculum) if rel]
        if micro_files:
            _push_repo_files_via_tar(
                args.host_alias,
                repo_root,
                micro_files,
                "/workspace/PSM",
                host=ssh_host,
                port=ssh_port,
                user=ssh_user,
            )

    extra_env = {
        "PSM_TRAIN_DEVICE": args.device,
        "TARGET_STEPS": str(args.target_steps),
        "RESUME_CHECKPOINT": args.resume_checkpoint,
        "TOKENIZER": args.tokenizer,
        "ABORT_AFTER_STEP": str(args.abort_after_step),
        "GATE4_CURRICULUM_BUILDER": args.curriculum_builder,
        "GATE4_CURRICULUM": args.curriculum
        or {
            "v1": "psm-model/data/curriculum/psm-50m-gate4-train-v1.jsonl",
            "v2": "psm-model/data/curriculum/psm-50m-gate4-train-v2.jsonl",
            "v3": "psm-model/data/curriculum/psm-50m-gate4-train-v3.jsonl",
            "v4": "psm-model/data/curriculum/psm-50m-gate4-train-v4.jsonl",
            "micro": "psm-model/data/curriculum/psm-50m-gate4-train-micro.jsonl",
            "legacy": "psm-model/data/curriculum/psm-50m-full-storage-v4-gate4.jsonl",
        }[args.curriculum_builder],
        "GATE4_PARSE_REPAIR": args.parse_repair,
        "GATE4_EVAL_REPORT": args.eval_report,
        "GATE4_REPAIR_SOURCE": args.repair_source,
        "DIRECT_COPIES": str(args.direct_copies),
        "EXPANDED_COPIES": str(args.expanded_copies),
        "DRILL_ROWS_PER_ACTION": str(args.drill_rows_per_action),
        "DRILL_COPIES": str(args.drill_copies),
        "STRATIFIED_MAX": str(args.stratified_max),
        "IGNORE_EXTRA_COPIES": str(args.ignore_extra_copies),
        "REPAIR_COPIES": str(args.repair_copies),
        "STRUCTURAL_LOSS_WEIGHT": str(args.structural_loss_weight),
        "BATCH_SIZE": str(args.batch_size),
        "LEARNING_RATE": str(args.learning_rate),
        "MIN_LEARNING_RATE": str(args.min_learning_rate),
        "PROMOTE_SPAN_WEIGHT": "4" if args.curriculum_builder == "micro" else "8",
        "EVAL_EVERY": str(args.eval_every),
        "SAVE_EVERY": str(args.save_every),
        "KEEP_LOCAL": str(args.keep_local),
        "SYNC_INTERVAL_SEC": str(args.sync_interval_sec),
        "GATE4_EVAL_AFTER": "1" if args.eval_after else "0",
        "PSM_HF_MODEL_REPO": os.environ.get("PSM_HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO),
    }
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        extra_env["HF_TOKEN"] = hf_token
    dataset_hf_token = os.environ.get("DATASET_HF_TOKEN", "").strip()
    if dataset_hf_token:
        extra_env["DATASET_HF_TOKEN"] = dataset_hf_token
    resume_step = ""
    for part in Path(args.resume_checkpoint).stem.split("-"):
        if part.isdigit():
            resume_step = part
    if resume_step:
        extra_env["GATE4_PINNED_STEPS"] = resume_step
    if args.curriculum:
        extra_env["SKIP_CURRICULUM_BUILD"] = "1"
    start_timeout = 300 if warm_pod else args.timeout_sec
    rc = _ssh_run_script(
        args.host_alias,
        script_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=start_timeout,
        extra_env=extra_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
    )
    if rc == 0 and warm_pod:
        if not _verify_pod_job(
            args.host_alias,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            tmux_session="psm-gate4",
            process_pattern="psm_model.train",
            label="gate4-train",
        ):
            return 1

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

    local_ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    _scp_from_pod(
        args.host_alias,
        "/workspace/PSM/psm-model/checkpoints/gate4-checkpoint-registry.json",
        local_ckpt_dir,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
    )

    args.required_hf_steps = _gate4_best_steps_for_verify() or [int(args.target_steps)]
    _delete_pod_unless_kept(args, pod_id, rc, job="training")

    return rc


def cmd_train_gate5(args: argparse.Namespace) -> int:
    scripts_dir = Path(__file__).resolve().parent
    warm_pod = getattr(args, "warm_pod", True) and bool(args.pod_id) and not args.deploy
    script_path = scripts_dir / ("runpod_start_gate5_train_only.sh" if warm_pod else "runpod_train_gate5.sh")
    if not script_path.exists():
        raise SystemExit(f"missing train script: {script_path}")

    proxy_user = args.proxy_user or None
    pod_id, ssh_host, ssh_port, ssh_user = _resolve_train_pod_ssh(args, proxy_user=proxy_user)

    if args.upload_first:
        upload_path = scripts_dir / "runpod_upload_gate4.sh"
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
            return upload_rc

    repo_root = Path(__file__).resolve().parents[2]
    if args.sync_src and not warm_pod:
        src_dir = repo_root / "psm-model" / "src"
        remote_src = "/workspace/PSM/psm-model/src"
        print(f"Syncing {src_dir} -> pod:{remote_src}", file=sys.stderr)
        push_rc = _ssh_push_dir(
            args.host_alias, src_dir, remote_src, host=ssh_host, port=ssh_port, user=ssh_user
        )
        if push_rc != 0:
            print(f"warning: src sync failed (exit {push_rc})", file=sys.stderr)
    elif args.sync_src and warm_pod:
        print("warm-pod: skipping full src sync (use --no-warm-pod for cold bootstrap)", file=sys.stderr)

    scripts_dir_repo = repo_root / "psm-model" / "scripts"
    if scripts_dir_repo.is_dir():
        remote_scripts = "/workspace/PSM/psm-model/scripts"
        print(f"Syncing {scripts_dir_repo} -> pod:{remote_scripts}", file=sys.stderr)
        scripts_rc = _ssh_push_dir(
            args.host_alias, scripts_dir_repo, remote_scripts, host=ssh_host, port=ssh_port, user=ssh_user
        )
        if scripts_rc != 0:
            print(f"warning: scripts sync failed (exit {scripts_rc})", file=sys.stderr)

    gate5_artifacts: list[str] = []
    if args.curriculum:
        gate5_artifacts.append(args.curriculum)
    if args.recall_probe:
        gate5_artifacts.append(args.recall_probe)
    expanded_probe = repo_root / "probes" / "expanded-probe-v1-filtered.jsonl"
    if expanded_probe.is_file():
        gate5_artifacts.append("probes/expanded-probe-v1-filtered.jsonl")
    default_curriculum_rel = (
        "psm-model/data/curriculum/psm-50m-gate5-train-v2-recall-heavy.jsonl"
        if getattr(args, "profile", "recall-heavy") == "recall-heavy"
        else "psm-model/data/curriculum/psm-50m-gate5-train-v1.jsonl"
    )
    prebuilt_curriculum = repo_root / default_curriculum_rel
    if prebuilt_curriculum.is_file() and default_curriculum_rel not in gate5_artifacts:
        gate5_artifacts.append(default_curriculum_rel)
    if gate5_artifacts:
        _push_repo_files_via_tar(
            args.host_alias,
            repo_root,
            gate5_artifacts,
            "/workspace/PSM",
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )

    profile_presets = {
        "bridge": (25, 100, 20),
        "recall-heavy": (20, 0, 500),
    }
    expanded_copies, direct_copies, recall_copies = profile_presets.get(
        args.profile, profile_presets["recall-heavy"]
    )
    if args.expanded_copies is not None:
        expanded_copies = args.expanded_copies
    if args.direct_copies is not None:
        direct_copies = args.direct_copies
    if args.recall_copies is not None:
        recall_copies = args.recall_copies
    default_curriculum = (
        "psm-model/data/curriculum/psm-50m-gate5-train-v2-recall-heavy.jsonl"
        if args.profile == "recall-heavy"
        else "psm-model/data/curriculum/psm-50m-gate5-train-v1.jsonl"
    )
    extra_env = {
        "PSM_TRAIN_DEVICE": args.device,
        "TARGET_STEPS": str(args.target_steps),
        "RESUME_CHECKPOINT": args.resume_checkpoint,
        "TOKENIZER": args.tokenizer,
        "ABORT_AFTER_STEP": str(args.abort_after_step),
        "GATE5_CURRICULUM": args.curriculum or default_curriculum,
        "GATE5_RECALL_PROBE": args.recall_probe,
        "GATE5_PROFILE": args.profile,
        "EXPANDED_COPIES": str(expanded_copies),
        "DIRECT_COPIES": str(direct_copies),
        "RECALL_COPIES": str(recall_copies),
        "STRUCTURAL_LOSS_WEIGHT": str(args.structural_loss_weight),
        "BATCH_SIZE": str(args.batch_size),
        "LEARNING_RATE": str(args.learning_rate),
        "MIN_LEARNING_RATE": str(args.min_learning_rate),
        "WARMUP_STEPS": str(args.warmup_steps),
        "EVAL_EVERY": str(args.eval_every),
        "SAVE_EVERY": str(args.save_every),
        "KEEP_LOCAL": str(args.keep_local),
        "SYNC_INTERVAL_SEC": str(args.sync_interval_sec),
        "GATE5_EVAL_AFTER": "1" if args.eval_after else "0",
        "PSM_HF_MODEL_REPO": os.environ.get("PSM_HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO),
    }
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        extra_env["HF_TOKEN"] = hf_token
    dataset_hf_token = os.environ.get("DATASET_HF_TOKEN", "").strip()
    if dataset_hf_token:
        extra_env["DATASET_HF_TOKEN"] = dataset_hf_token
    resume_step = ""
    for part in Path(args.resume_checkpoint).stem.split("-"):
        if part.isdigit():
            resume_step = part
    if resume_step:
        extra_env["GATE4_PINNED_STEPS"] = resume_step
    curriculum_on_disk = (
        Path(args.curriculum)
        if args.curriculum
        else repo_root / default_curriculum
    )
    if curriculum_on_disk.is_file():
        extra_env["SKIP_CURRICULUM_BUILD"] = "1"

    start_timeout = 300 if warm_pod else args.timeout_sec
    rc = _ssh_run_script(
        args.host_alias,
        script_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=start_timeout,
        extra_env=extra_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
    )
    if rc == 0 and warm_pod:
        if not _verify_pod_job(
            args.host_alias,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            tmux_session="psm-gate5",
            process_pattern="psm_model.train",
            label="gate5-train",
        ):
            return 1

    if args.pull_metrics:
        scp_rc = _scp_from_pod(
            args.host_alias,
            "/workspace/PSM/psm-model/checkpoints/real-v3-50m-full-v2-gate5.metrics.jsonl",
            Path(args.pull_metrics).parent,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
        if scp_rc != 0:
            print("warning: could not pull gate5 metrics", file=sys.stderr)

    local_ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    _scp_from_pod(
        args.host_alias,
        "/workspace/PSM/psm-model/checkpoints/gate4-checkpoint-registry.json",
        local_ckpt_dir,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
    )
    if args.pull_reports:
        target_step = f"{int(args.target_steps):06d}"
        _scp_from_pod(
            args.host_alias,
            f"/workspace/PSM/psm-model/checkpoints/gate-eval/gate5-dual-step-{target_step}.json",
            Path(args.pull_reports),
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )

    args.required_hf_steps = _gate4_best_steps_for_verify() or [int(args.target_steps)]
    _delete_pod_unless_kept(args, pod_id, rc, job="training")
    return rc


def cmd_eval_gate5_dual(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve().parent / "runpod_eval_gate5_dual.sh"
    if not script_path.exists():
        raise SystemExit(f"missing eval script: {script_path}")

    proxy_user = args.proxy_user or None
    pod_id, ssh_host, ssh_port, ssh_user = _resolve_train_pod_ssh(args, proxy_user=proxy_user)
    repo_root = Path(__file__).resolve().parents[2]
    if args.sync_src:
        src_dir = repo_root / "psm-model" / "src"
        _ssh_push_dir(
            args.host_alias, src_dir, "/workspace/PSM/psm-model/src", host=ssh_host, port=ssh_port, user=ssh_user
        )
    scripts_dir = repo_root / "psm-model" / "scripts"
    if not getattr(args, "no_sync_scripts", False):
        push_rc = _ssh_push_dir(
            args.host_alias, scripts_dir, "/workspace/PSM/psm-model/scripts", host=ssh_host, port=ssh_port, user=ssh_user
        )
        if push_rc != 0:
            print(
                f"script sync failed (exit {push_rc}); continuing — eval script bootstraps from git/HF",
                file=sys.stderr,
            )

    extra_env = {
        "EVAL_STEP": f"{int(args.eval_step):06d}",
        "PSM_EVAL_DEVICE": args.device,
        "GATE5_STORAGE_PROBE": args.storage_probe,
        "GATE5_RECALL_PROBE": args.recall_probe,
        "PSM_HF_MODEL_REPO": os.environ.get("PSM_HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO),
    }
    if os.environ.get("HF_TOKEN", "").strip():
        extra_env["HF_TOKEN"] = os.environ["HF_TOKEN"].strip()
    if os.environ.get("DATASET_HF_TOKEN", "").strip():
        extra_env["DATASET_HF_TOKEN"] = os.environ["DATASET_HF_TOKEN"].strip()

    bootstrap_env = {**extra_env, "BOOTSTRAP_ONLY": "1"}
    rc = _ssh_run_script(
        args.host_alias,
        script_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=min(args.timeout_sec, 1800),
        extra_env=bootstrap_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=bool(args.deploy or args.pod_id),
    )
    if rc != 0:
        if getattr(args, "stop_on_fail", False):
            _rest("POST", f"/pods/{pod_id}/stop")
        _delete_pod_unless_kept(args, pod_id, rc, job="eval")
        return rc if isinstance(rc, int) else rc[0]

    start_env = {**extra_env, "WAIT_EVAL_DONE": "0"}
    rc = _ssh_run_script(
        args.host_alias,
        script_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=min(args.timeout_sec, 900),
        extra_env=start_env,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
    )
    if rc == 0 and getattr(args, "verify_after_start", True):
        if not _verify_pod_job(
            args.host_alias,
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            tmux_session="psm-gate5-eval",
            process_pattern="eval_dual_gate",
            label="gate5-eval",
        ):
            # Eval loads checkpoint before GPU spikes; don't abort if tmux already started.
            probe = _probe_pod_training(
                args.host_alias,
                host=ssh_host,
                port=ssh_port,
                user=ssh_user,
                tmux_session="psm-gate5-eval",
                process_pattern="eval_dual_gate",
                train_log="/tmp/psm-gate5-dual-eval.log",
                timeout_sec=30,
            )
            status, _ = probe
            if status.get("PSM_EVAL_DONE") != "1" and status.get("PSM_TMUX") != "OK":
                print("gate5-eval verify failed: no GPU job on pod", file=sys.stderr)
                if getattr(args, "stop_on_fail", False):
                    _rest("POST", f"/pods/{pod_id}/stop")
                _delete_pod_unless_kept(args, pod_id, 1, job="eval")
                return 1
            print("gate5-eval verify: tmux up or eval already done — continuing wait", file=sys.stderr)

    wait_path = Path(__file__).resolve().parent / "runpod_wait_gate5_dual.sh"
    rc = _ssh_run_script(
        args.host_alias,
        wait_path,
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        timeout_sec=args.timeout_sec,
        ssh_ready_timeout_sec=args.ssh_ready_timeout_sec,
        skip_ssh_wait=True,
    )
    if rc == 0 and args.pull_reports:
        step = f"{int(args.eval_step):06d}"
        _scp_from_pod(
            args.host_alias,
            f"/workspace/PSM/psm-model/checkpoints/gate-eval/gate5-dual-step-{step}.json",
            Path(args.pull_reports),
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
        )
    _delete_pod_unless_kept(args, pod_id, rc, job="eval")
    return rc


def _deploy_payload(args: argparse.Namespace, *, cloud_type: str = "SECURE") -> dict[str, object]:
    start_cmd = ["bash", "-lc", AUTOSTART_CMD] if args.autostart else ["sleep", "infinity"]
    env = {
        "HF_TOKEN": HF_TOKEN_SECRET_REF,
        "PYTHONPATH": "psm-model/src",
        "PSM_REPO_ROOT": "/workspace/PSM",
        "PSM_HF_MODEL_REPO": DEFAULT_HF_MODEL_REPO,
        "PSM_HF_DATASET_REPO": DEFAULT_HF_DATASET_REPO,
        "PSM_SYNC_GIT": "1",
    }
    payload: dict[str, object] = {
        "name": args.name,
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "cloudType": cloud_type,
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
    data = _create_pod_with_fallback(args)
    print(json.dumps(data, indent=2))
    if args.wait_ssh:
        pod_id = str(data.get("id", ""))
        if pod_id:
            cmd_wait_ssh(
                argparse.Namespace(
                    pod_id=pod_id,
                    host_alias=SSH_CONFIG_HOST,
                    proxy_user="",
                    timeout_sec=args.wait_ssh,
                    poll_sec=5,
                )
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-pods", help="List your pods").set_defaults(func=cmd_list_pods)

    list_gpus = sub.add_parser("list-gpus", help="Query RunPod GraphQL GPU stockStatus and pricing")
    list_gpus.add_argument("--all", action="store_true", help="List every GPU type (not just PSM preferences).")
    list_gpus.add_argument(
        "--gpu-preferences",
        default="",
        help=f"Comma-separated ids to show (default: {', '.join(PSM_GPU_PREFERENCES)}).",
    )
    list_gpus.set_defaults(func=cmd_list_gpus)

    pick_gpu = sub.add_parser("pick-gpu", help="Pick first available GPU from the PSM preference list")
    pick_gpu.add_argument("--gpu-preferences", default="", help="Override comma-separated preference order.")
    pick_gpu.add_argument("--min-vram-gb", type=int, default=DEFAULT_MIN_VRAM_GB)
    pick_gpu.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB, help="Echo default volume with pick result.")
    pick_gpu.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    pick_gpu.set_defaults(func=cmd_pick_gpu)

    stop = sub.add_parser("stop-pod", help="Stop one pod by id")
    stop.add_argument("pod_id")
    stop.set_defaults(func=cmd_stop_pod)

    sub.add_parser("stop-all", help="Stop all running pods").set_defaults(func=cmd_stop_all)

    delete = sub.add_parser("delete-pod", help="Terminate one pod by id")
    delete.add_argument("pod_id")
    delete.add_argument(
        "--force-delete-pod",
        action="store_true",
        help="Delete even if Gate 4 best checkpoint is not fully on HF (dangerous).",
    )
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
    deploy.add_argument("--gpu", default=DEFAULT_GPU)
    deploy.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    deploy.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    _add_auto_gpu_arguments(deploy)
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

    verify_pod = sub.add_parser(
        "verify-pod",
        help="Probe tmux + train process + CUDA + GPU util (hard timeout, no 8h block)",
    )
    verify_pod.add_argument("--pod-id", required=True)
    verify_pod.add_argument("--proxy-user", default="", help="pod_id-suffix@ssh.runpod.io (required for reliable SSH)")
    verify_pod.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    verify_pod.add_argument("--tmux-session", default="psm-gate5")
    verify_pod.add_argument("--process-pattern", default="psm_model.train")
    verify_pod.add_argument("--train-log", default="/tmp/psm-gate5-train.log")
    verify_pod.add_argument("--timeout-sec", type=int, default=60, help="Max SSH probe time (default 60s)")
    verify_pod.add_argument("--min-gpu-pct", type=int, default=5, help="Fail if GPU util below this")
    verify_pod.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require GPU util >= min-gpu-pct (use --no-require-gpu during apt/bootstrap)",
    )
    verify_pod.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop pod immediately if verify fails (prevents idle billing)",
    )
    verify_pod.set_defaults(func=cmd_verify_pod)

    eval_gates = sub.add_parser(
        "eval-gates",
        help="Run Gate 2/3 eval on a RunPod GPU (deploy fresh pod or use existing SSH host)",
    )
    eval_gates.add_argument("--deploy", action="store_true", help="Deploy a new eval-only pod before running.")
    eval_gates.add_argument("--pod-id", default="", help="Pod id (for delete-after); filled automatically when --deploy is set.")
    eval_gates.add_argument("--name", default="psm-eval", help="Pod name when --deploy is set.")
    eval_gates.add_argument("--image", default=STOCK_PYTORCH_IMAGE)
    eval_gates.add_argument("--template", default="", help="RunPod template id (optional).")
    eval_gates.add_argument("--gpu", default=DEFAULT_GPU)
    eval_gates.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    eval_gates.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    _add_auto_gpu_arguments(eval_gates)
    eval_gates.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    eval_gates.add_argument(
        "--proxy-user",
        default="",
        help="Proxy SSH user from Connect tab (pod_id-suffix@ssh.runpod.io). Required for existing pods.",
    )
    eval_gates.add_argument(
        "--warm-pod",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Existing pod: skip bootstrap, tmux eval-only, verify GPU (default: on when --pod-id).",
    )
    eval_gates.add_argument("--device", default="cuda", help="Eval device passed to psm_model (cuda recommended on pod).")
    eval_gates.add_argument("--expanded", action="store_true", help="Also run full-model eval on expanded probe (920 rows).")
    eval_gates.add_argument(
        "--full-checkpoint",
        default="",
        help="Full-model checkpoint path under repo (default: promoted real-v3-50m-full-v2.pt).",
    )
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
    eval_gates.add_argument("--keep-pod", action="store_true", help="Keep pod running after eval (default: delete when --pod-id is set).")
    eval_gates.add_argument("--delete-after", action="store_true", help=argparse.SUPPRESS)
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
    train_gate4.add_argument("--gpu", default=DEFAULT_GPU)
    train_gate4.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    train_gate4.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    _add_auto_gpu_arguments(train_gate4)
    train_gate4.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    train_gate4.add_argument(
        "--proxy-user",
        default="",
        help="Proxy SSH user from Connect tab (pod_id-suffix@ssh.runpod.io). Required for existing pods.",
    )
    train_gate4.add_argument(
        "--warm-pod",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Existing pod: skip bootstrap/src sync, tmux train-only, verify GPU (default: on when --pod-id).",
    )
    train_gate4.add_argument("--device", default="cuda")
    train_gate4.add_argument(
        "--target-steps",
        type=int,
        default=42000,
        help="Absolute training step target (recovery default: re-train 36000→42000).",
    )
    train_gate4.add_argument(
        "--resume-checkpoint",
        default="psm-model/checkpoints/real-v3-50m-full-v2-step-036000.pt",
        help="Gate 4 resume checkpoint (recovery: 36000 is best on HF from v2 lineage).",
    )
    train_gate4.add_argument(
        "--tokenizer",
        default="psm-model/checkpoints/real-v3-50m-full-v2-step-036000.tokenizer.json",
    )
    train_gate4.add_argument(
        "--sync-src",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tar-sync local psm-model/src to pod before training (default: on).",
    )
    train_gate4.add_argument("--upload-first", action="store_true", help="Sync checkpoints to HF before training.")
    train_gate4.add_argument(
        "--upload-keep-local",
        type=int,
        default=2,
        help="Step checkpoints to retain on pod after HF sync.",
    )
    train_gate4.add_argument("--save-every", type=int, default=400)
    train_gate4.add_argument("--keep-local", type=int, default=2, help="Pod disk: retain N step checkpoints locally.")
    train_gate4.add_argument("--sync-interval-sec", type=int, default=120, help="HF sync interval during training (upload all steps).")
    train_gate4.add_argument(
        "--curriculum",
        default="",
        help="Override curriculum JSONL path (warm-pod: use pre-built file on pod).",
    )
    train_gate4.add_argument(
        "--curriculum-builder",
        choices=("v1", "v2", "v3", "v4", "micro", "legacy"),
        default="v4",
        help="v4 = production (expanded ×100 + complete-tag drills, resume 42k); v3/v2/micro = prior experiments.",
    )
    train_gate4.add_argument("--direct-copies", type=int, default=300)
    train_gate4.add_argument("--expanded-copies", type=int, default=100, help="v4: copies per expanded-budget row.")
    train_gate4.add_argument("--drill-rows-per-action", type=int, default=120)
    train_gate4.add_argument("--drill-copies", type=int, default=50, help="v2 default: 50 (v1 default: 25).")
    train_gate4.add_argument("--stratified-max", type=int, default=1500, help="v2 default: 1500 (v1 default: 2500).")
    train_gate4.add_argument("--repair-copies", type=int, default=1, help="micro: 12; v4: light repair from 42k failures.")
    train_gate4.add_argument(
        "--structural-loss-weight",
        type=float,
        default=1.0,
        help="Tagged DSL structural loss multiplier (micro default: 8).",
    )
    train_gate4.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size. A5000 24GB fits 16-32 for the 50m preset at ctx 2048.",
    )
    train_gate4.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Base learning rate (cosine-decayed over absolute steps; see --min-learning-rate).",
    )
    train_gate4.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Cosine floor. Set equal to --learning-rate for a constant LR (recommended when resuming near target).",
    )
    train_gate4.add_argument(
        "--parse-repair",
        default="psm-model/data/curriculum/gate4-parse-repair-step-42000.jsonl",
        help="Pre-mined parse-repair JSONL (downloaded from HF if missing on pod).",
    )
    train_gate4.add_argument(
        "--eval-report",
        default="",
        help="Optional full Gate 4 eval JSON to mine parse failures on pod if repair pack missing.",
    )
    train_gate4.add_argument(
        "--repair-source",
        default="psm-model/data/direct-behavior-v1/expanded-probe-v1-budget.jsonl",
        help="Gold probe JSONL for on-pod parse-repair mining.",
    )
    train_gate4.add_argument("--ignore-extra-copies", type=int, default=6, help="legacy builder only.")
    train_gate4.add_argument("--eval-every", type=int, default=0, help="Mid-train probe eval interval (0 = off).")
    train_gate4.add_argument("--abort-after-step", type=int, default=60000)
    train_gate4.add_argument(
        "--eval-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run expanded Gate 4 eval on target checkpoint after training (default: on).",
    )
    train_gate4.add_argument("--wait-ssh", type=int, default=300, metavar="SEC")
    train_gate4.add_argument("--timeout-sec", type=int, default=28800, help="SSH training session timeout (8h default).")
    train_gate4.add_argument("--ssh-ready-timeout-sec", type=int, default=420)
    train_gate4.add_argument(
        "--pull-metrics",
        default="",
        help="Local path to directory for metrics jsonl pull after training (empty to skip).",
    )
    train_gate4.add_argument("--keep-pod", action="store_true", help="Keep pod running after training (default: delete when --pod-id is set).")
    train_gate4.add_argument(
        "--force-delete-pod",
        action="store_true",
        help="Delete pod even if production checkpoint .pt is missing from HF (dangerous).",
    )
    train_gate4.add_argument(
        "--delete-after",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    train_gate4.set_defaults(func=cmd_train_gate4)

    train_gate5 = sub.add_parser(
        "train-gate5",
        help="Run Gate 5 mixed storage+recall training (resume 48000 default); dual gate eval after train",
    )
    train_gate5.add_argument("--deploy", action="store_true")
    train_gate5.add_argument("--pod-id", default="")
    train_gate5.add_argument("--name", default="psm-train-gate5")
    train_gate5.add_argument("--image", default=STOCK_PYTORCH_IMAGE)
    train_gate5.add_argument("--template", default="")
    train_gate5.add_argument("--gpu", default=DEFAULT_GPU)
    train_gate5.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    train_gate5.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    _add_auto_gpu_arguments(train_gate5)
    train_gate5.set_defaults(auto_gpu=True)
    train_gate5.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    train_gate5.add_argument("--proxy-user", default="")
    train_gate5.add_argument("--warm-pod", action=argparse.BooleanOptionalAction, default=True)
    train_gate5.add_argument("--device", default="cuda")
    train_gate5.add_argument("--target-steps", type=int, default=58000, help="Absolute step target (phase 2 default: 051000 to 58000).")
    train_gate5.add_argument(
        "--resume-checkpoint",
        default="psm-model/checkpoints/real-v3-50m-full-v2-step-057000.pt",
    )
    train_gate5.add_argument(
        "--tokenizer",
        default="psm-model/checkpoints/real-v3-50m-full-v2-step-057000.tokenizer.json",
    )
    train_gate5.add_argument("--sync-src", action=argparse.BooleanOptionalAction, default=True)
    train_gate5.add_argument("--upload-first", action="store_true")
    train_gate5.add_argument("--upload-keep-local", type=int, default=2)
    train_gate5.add_argument("--save-every", type=int, default=200)
    train_gate5.add_argument("--keep-local", type=int, default=2)
    train_gate5.add_argument("--sync-interval-sec", type=int, default=120)
    train_gate5.add_argument(
        "--curriculum",
        default="",
        help="Pre-built gate5 JSONL on pod (sets SKIP_CURRICULUM_BUILD).",
    )
    train_gate5.add_argument(
        "--recall-probe",
        default="psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl",
    )
    train_gate5.add_argument(
        "--profile",
        choices=("bridge", "recall-heavy"),
        default="recall-heavy",
        help="Gate5 curriculum mix (recall-heavy for phase 2 from step 051000).",
    )
    train_gate5.add_argument("--expanded-copies", type=int, default=None)
    train_gate5.add_argument("--direct-copies", type=int, default=None)
    train_gate5.add_argument("--recall-copies", type=int, default=None)
    train_gate5.add_argument("--structural-loss-weight", type=float, default=1.0)
    train_gate5.add_argument("--batch-size", type=int, default=16)
    train_gate5.add_argument("--learning-rate", type=float, default=5e-5)
    train_gate5.add_argument("--min-learning-rate", type=float, default=1e-5)
    train_gate5.add_argument("--warmup-steps", type=int, default=50)
    train_gate5.add_argument("--eval-every", type=int, default=400)
    train_gate5.add_argument("--abort-after-step", type=int, default=60000)
    train_gate5.add_argument("--eval-after", action=argparse.BooleanOptionalAction, default=True)
    train_gate5.add_argument("--wait-ssh", type=int, default=300)
    train_gate5.add_argument("--timeout-sec", type=int, default=28800)
    train_gate5.add_argument("--ssh-ready-timeout-sec", type=int, default=420)
    train_gate5.add_argument("--pull-metrics", default="")
    train_gate5.add_argument(
        "--pull-reports",
        default="psm-model/checkpoints/gate-eval",
        help="Local dir for gate5-dual-step-*.json after training (warm pod: may be empty until train completes).",
    )
    train_gate5.add_argument("--keep-pod", action="store_true")
    train_gate5.add_argument("--force-delete-pod", action="store_true")
    train_gate5.add_argument("--delete-after", action="store_true", help=argparse.SUPPRESS)
    train_gate5.set_defaults(func=cmd_train_gate5)

    eval_gate5 = sub.add_parser("eval-gate5-dual", help="Dual eval: Gate 4 storage + Gate 5 recall on one checkpoint step")
    eval_gate5.add_argument("--deploy", action="store_true")
    eval_gate5.add_argument("--pod-id", default="")
    eval_gate5.add_argument("--name", default="psm-eval-gate5")
    eval_gate5.add_argument("--image", default=STOCK_PYTORCH_IMAGE)
    eval_gate5.add_argument("--gpu", default=DEFAULT_GPU)
    eval_gate5.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    eval_gate5.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    _add_auto_gpu_arguments(eval_gate5)
    eval_gate5.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    eval_gate5.add_argument("--proxy-user", default="")
    eval_gate5.add_argument("--eval-step", type=int, required=True, help="Checkpoint step, e.g. 51000")
    eval_gate5.add_argument("--device", default="cuda")
    eval_gate5.add_argument(
        "--storage-probe",
        default="psm-model/data/direct-behavior-v1/expanded-probe-v1-filtered.jsonl",
    )
    eval_gate5.add_argument(
        "--recall-probe",
        default="psm-model/data/curriculum/psm-50m-recall-plan-v1.jsonl",
    )
    eval_gate5.add_argument("--sync-src", action="store_true", help="Push local psm-model/src before eval.")
    eval_gate5.add_argument(
        "--no-sync-scripts",
        action="store_true",
        help="Skip tar-push of scripts (eval bootstraps from git/HF on pod).",
    )
    eval_gate5.add_argument("--timeout-sec", type=int, default=7200)
    eval_gate5.add_argument("--ssh-ready-timeout-sec", type=int, default=420)
    eval_gate5.add_argument("--pull-reports", default="psm-model/checkpoints/gate-eval")
    eval_gate5.add_argument("--keep-pod", action="store_true")
    eval_gate5.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop pod if CUDA/GPU verify fails after eval tmux start",
    )
    eval_gate5.add_argument("--delete-after", action="store_true", help=argparse.SUPPRESS)
    eval_gate5.set_defaults(func=cmd_eval_gate5_dual)

    upload_gate4 = sub.add_parser("upload-gate4", help="Upload latest Gate 4 checkpoints to HF")
    upload_gate4.add_argument("--pod-id", default="", help="Existing pod id.")
    upload_gate4.add_argument("--deploy", action="store_true")
    upload_gate4.add_argument("--name", default="psm-train-gate4")
    upload_gate4.add_argument("--image", default=STOCK_PYTORCH_IMAGE)
    upload_gate4.add_argument("--template", default="")
    upload_gate4.add_argument("--gpu", default=DEFAULT_GPU)
    upload_gate4.add_argument("--volume-gb", type=int, default=DEFAULT_VOLUME_GB)
    upload_gate4.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    _add_auto_gpu_arguments(upload_gate4)
    upload_gate4.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    upload_gate4.add_argument("--proxy-user", default="")
    upload_gate4.add_argument("--wait-ssh", type=int, default=120)
    upload_gate4.add_argument("--ssh-ready-timeout-sec", type=int, default=120)
    upload_gate4.add_argument("--timeout-sec", type=int, default=7200)
    upload_gate4.add_argument("--keep-local", type=int, default=2, help="Retain N newest step checkpoints on pod after HF sync.")
    upload_gate4.set_defaults(func=cmd_upload_gate4)

    recover_gate4 = sub.add_parser(
        "recover-gate4",
        help="Prune corrupt checkpoints, sync all to HF, resume Gate 4 training with periodic sync",
    )
    recover_gate4.add_argument("--pod-id", default="", help="Existing pod id.")
    recover_gate4.add_argument("--proxy-user", default="")
    recover_gate4.add_argument("--host-alias", default=SSH_CONFIG_HOST)
    recover_gate4.add_argument("--target-steps", type=int, default=36000)
    recover_gate4.add_argument("--save-every", type=int, default=400)
    recover_gate4.add_argument("--keep-local", type=int, default=2)
    recover_gate4.add_argument("--timeout-sec", type=int, default=28800)
    recover_gate4.add_argument("--ssh-ready-timeout-sec", type=int, default=420)
    recover_gate4.add_argument("--wait-ssh", type=int, default=120)
    recover_gate4.set_defaults(func=cmd_recover_gate4)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
