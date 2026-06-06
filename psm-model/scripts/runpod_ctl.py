#!/usr/bin/env python3
"""RunPod pod/template helpers. Set RUNPOD_API_KEY (e.g. from `o runpodkey`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"

# RunPod secret named HF_TOKEN → injected as HF_TOKEN env at pod start.
HF_TOKEN_SECRET_REF = "{{ RUNPOD_SECRET_HF_TOKEN }}"

DEFAULT_TEMPLATE = {
    "name": "psm-50m-train",
    "imageName": "chkrishna2001/psm-50m-train:latest",
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


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set RUNPOD_API_KEY (run: o runpodkey, then $env:RUNPOD_API_KEY = Get-Clipboard)")
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


def _rest(method: str, path: str, data: dict | None = None) -> dict:
    url = f"{REST_URL}{path}"
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def cmd_list_pods(_: argparse.Namespace) -> int:
    data = _graphql(
        """
        query {
          myself {
            pods {
              id
              name
              desiredStatus
              imageName
              machine { podHostId }
            }
          }
        }
        """
    )
    pods = data["myself"]["pods"]
    print(json.dumps(pods, indent=2))
    return 0


def cmd_stop_pod(args: argparse.Namespace) -> int:
    data = _graphql(
        'mutation Stop($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }',
        {"input": {"podId": args.pod_id}},
    )
    print(json.dumps(data, indent=2))
    return 0


def cmd_stop_all(_: argparse.Namespace) -> int:
    data = _graphql(
        """
        query {
          myself {
            pods {
              id
              name
              desiredStatus
            }
          }
        }
        """
    )
    for pod in data["myself"]["pods"]:
        if pod["desiredStatus"] in {"RUNNING", "EXITED"}:
            print(f"Stopping {pod['id']} ({pod['name']})...")
            _graphql(
                'mutation Stop($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }',
                {"input": {"podId": pod["id"]}},
            )
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
    data = _graphql(
        """
        mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            name
            imageName
            machineId
            desiredStatus
          }
        }
        """,
        {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "volumeInGb": args.volume_gb,
                "containerDiskInGb": args.container_disk_gb,
                "gpuTypeId": args.gpu,
                "name": args.name,
                "imageName": args.image,
                "ports": "22/tcp",
                "volumeMountPath": "/workspace",
                "dockerArgs": "sleep infinity",
                "env": [
                    {"key": "HF_TOKEN", "value": HF_TOKEN_SECRET_REF},
                    {"key": "PYTHONPATH", "value": "psm-model/src"},
                    {"key": "PSM_REPO_ROOT", "value": "/workspace/PSM"},
                ],
            }
        },
    )
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

    tmpl = sub.add_parser("create-template", help="Register REST template for psm-50m-train image")
    tmpl.add_argument("--name", default=DEFAULT_TEMPLATE["name"])
    tmpl.add_argument("--image", default=DEFAULT_TEMPLATE["imageName"])
    tmpl.set_defaults(func=cmd_create_template)

    deploy = sub.add_parser("deploy", help="Deploy pod from image (after docker push)")
    deploy.add_argument("--name", default="psm-train")
    deploy.add_argument("--image", default=DEFAULT_TEMPLATE["imageName"])
    deploy.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    deploy.add_argument("--volume-gb", type=int, default=40)
    deploy.add_argument("--container-disk-gb", type=int, default=20)
    deploy.set_defaults(func=cmd_deploy)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
