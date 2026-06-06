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

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"

# RunPod secret named HF_TOKEN → injected as HF_TOKEN env at pod start.
HF_TOKEN_SECRET_REF = "{{ RUNPOD_SECRET_HF_TOKEN }}"

DEFAULT_TEMPLATE = {
    "name": "psm-50m-train",
    "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
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


def cmd_deploy(args: argparse.Namespace) -> int:
    payload = {
        "name": args.name,
        "imageName": args.image,
        "gpuTypeIds": [args.gpu],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "volumeInGb": args.volume_gb,
        "containerDiskInGb": args.container_disk_gb,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "dockerStartCmd": ["sleep", "infinity"],
        "env": {
            "HF_TOKEN": HF_TOKEN_SECRET_REF,
            "PYTHONPATH": "psm-model/src",
            "PSM_REPO_ROOT": "/workspace/PSM",
            "PSM_HF_MODEL_REPO": "chkrishna2001/psm-50m-mixed-v1-run",
            "PSM_HF_DATASET_REPO": "chkrishna2001/psm-50m-action-mixed-v1",
            "PSM_SYNC_GIT": "1",
        },
    }
    data = _rest("POST", "/pods", payload)
    print(json.dumps(data, indent=2))
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
        default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        help="Default: stock RunPod PyTorch (custom psm-50m-train image may not be published yet).",
    )
    deploy.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    deploy.add_argument("--volume-gb", type=int, default=40)
    deploy.add_argument("--container-disk-gb", type=int, default=20)
    deploy.set_defaults(func=cmd_deploy)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
